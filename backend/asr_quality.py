"""ASR quality profiles — preset env defaults applied before model preload."""

from __future__ import annotations

import logging
import os

from backend.enterprise_config import ENTERPRISE_ACCURACY_BASE

logger = logging.getLogger(__name__)

PROFILE_BALANCED = "balanced"
PROFILE_HIGH = "high"

_PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    PROFILE_BALANCED: {
        "ASR_CHUNK_LENGTH_S": "30",
        "ASR_8GB_CHUNK_LENGTH_S": "30",
        "ASR_8GB_MAX_CHUNK_LENGTH_S": "30",
        "ASR_8GB_RETRY_CHUNK_LENGTH_S": "10",
        "ASR_CUDA_BATCH_SIZE": "1",
        "ASR_8GB_BATCH_SIZE": "1",
        "ASR_8GB_MAX_BATCH_SIZE": "1",
        "ASR_MIN_CHUNKED_DURATION_S": "60",
        "ASR_LONG_FORM_WINDOW_S": "60",
        "ASR_LONG_FORM_OVERLAP_S": "30",
        "PATHUMMA_WORD_TIMESTAMPS_ON_8GB": "false",
        "TYPHOON_WORD_TIMESTAMPS_ON_8GB": "false",
        "DIARIZATION_TRANSCRIPT_MERGE_GAP_S": "1.25",
    },
    PROFILE_HIGH: {
        **ENTERPRISE_ACCURACY_BASE,
        "ASR_CHUNK_LENGTH_S": "30",
        "ASR_8GB_CHUNK_LENGTH_S": "30",
        "ASR_8GB_MAX_CHUNK_LENGTH_S": "30",
        "ASR_8GB_RETRY_CHUNK_LENGTH_S": "15",
        "ASR_MIN_CHUNKED_DURATION_S": "120",
        "ASR_LONG_FORM_WINDOW_S": "300",
        "ASR_LONG_FORM_OVERLAP_S": "60",
        "PATHUMMA_WORD_TIMESTAMPS_ON_8GB": "true",
        "TYPHOON_WORD_TIMESTAMPS_ON_8GB": "false",
        "AUDIO_ENHANCE_DEFAULT": "false",
        "AUDIO_ENHANCE_ADAPTIVE": "true",
        "AUDIO_ENHANCE_TARGET_PEAK_DB": "-0.5",
        "AUDIO_ENHANCE_MAX_GAIN_DB": "24",
        "DIARIZATION_NOISE_REDUCTION": "0.0",
        # Meeting-scale VBx (309-validated); fixture overlays can tighten further.
        "DIARIZATION_OVERCLUSTER_EXTRA": "7",
        "DIARIZATION_VBX_FA": "0.25",
        "DIARIZATION_VBX_FB": "0.8",
    },
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def active_quality_profile() -> str:
    profile = os.getenv("ASR_QUALITY_PROFILE", PROFILE_HIGH).strip().lower()
    if profile not in _PROFILE_DEFAULTS:
        return PROFILE_HIGH
    return profile


def is_high_quality_profile() -> bool:
    return active_quality_profile() == PROFILE_HIGH


def is_accuracy_mode() -> bool:
    return _env_bool("DIARIZATION_ACCURACY_MODE", is_high_quality_profile())


def _8gb_chunk_cap() -> str:
    # Whisper's encoder tops out at 30s; larger chunks are truncated and lose audio.
    return "30"


def _multi_pass_diarization_active_env() -> bool:
    return _env_bool("DIARIZATION_MULTI_SAMPLE", False) and (
        _env_int("DIARIZATION_MULTI_SAMPLE_PASSES", 0) > 0
        or _env_bool("DIARIZATION_MULTI_SAMPLE_FULL_GRID", False)
    )


def _build_low_vram_forced_env(
    chunk_cap: str,
    co_resident: bool,
    unload_for_diar: str,
    *,
    multi_pass_diar: bool,
) -> dict[str, str]:
    requested = os.getenv("DIARIZATION_DEVICE", "auto").strip().lower()
    require_cuda = _env_bool("DIARIZATION_REQUIRE_CUDA", False)
    cuda_staging = (
        requested in {"cuda", "gpu"}
        and (_env_bool("DIARIZATION_ALLOW_8GB_CUDA", False) or require_cuda)
        and not co_resident
    )
    if co_resident:
        diar_device = "auto"
    elif cuda_staging or require_cuda:
        diar_device = "cuda" if requested in {"cuda", "gpu", "auto"} else requested
    else:
        diar_device = "cpu"
    preload_device = diar_device if cuda_staging or require_cuda else "cpu"
    batch_values = _sequential_staging_batch_env(co_resident, multi_pass_diar=multi_pass_diar)
    return {
        "PATHUMMA_WORD_TIMESTAMPS_ON_8GB": "false",
        "ASR_8GB_CHUNK_LENGTH_S": chunk_cap,
        "ASR_8GB_MAX_CHUNK_LENGTH_S": chunk_cap,
        "ASR_8GB_RETRY_CHUNK_LENGTH_S": "15",
        "DIARIZATION_PRELOAD_DEVICE": preload_device,
        "DIARIZATION_DEVICE": diar_device,
        "DIARIZATION_ALLOW_8GB_CUDA": "true" if (co_resident or cuda_staging) else "false",
        "DIARIZATION_CUDA_MIN_FREE_MB": "1536" if co_resident else "1024",
        "DIARIZATION_CUDA_RUN_MIN_FREE_MB": "1024",
        "ASR_UNLOAD_FOR_DIARIZATION": "true" if cuda_staging else unload_for_diar,
        "TYPHOON_WORD_TIMESTAMPS_ON_8GB": "false",
        **batch_values,
    }


# Production CUDA memory cap — locked at 0.92 (companion batch=1 guards prevent OOM).
_MAX_CUDA_MEMORY_FRACTION = 0.92
_DEFAULT_CUDA_MEMORY_FRACTION = 0.92


def _cuda_fraction_cap() -> float:
    """Upper bound for ASR_CUDA_MEMORY_FRACTION (0.92 in production)."""
    cap_raw = os.getenv("ASR_CUDA_MEMORY_FRACTION_MAX", "").strip()
    try:
        cap = float(cap_raw) if cap_raw else _MAX_CUDA_MEMORY_FRACTION
    except ValueError:
        cap = _MAX_CUDA_MEMORY_FRACTION
    return min(_MAX_CUDA_MEMORY_FRACTION, max(0.5, cap))


def _clamp_cuda_memory_fraction(forced: dict[str, str]) -> None:
    """Clamp ASR_CUDA_MEMORY_FRACTION to 0.92 max; never lower below env if ≤0.92."""
    cap = _cuda_fraction_cap()
    raw = os.getenv("ASR_CUDA_MEMORY_FRACTION", "").strip()
    try:
        current = float(raw) if raw else _DEFAULT_CUDA_MEMORY_FRACTION
    except ValueError:
        current = _DEFAULT_CUDA_MEMORY_FRACTION
    clamped = min(cap, max(0.5, current))
    if f"{clamped:.2f}" != raw:
        forced["ASR_CUDA_MEMORY_FRACTION"] = f"{clamped:.2f}"


# Backward-compatible alias used by existing call sites.
_ensure_min_cuda_memory_fraction = _clamp_cuda_memory_fraction


def _sequential_staging_batch_env(
    co_resident: bool,
    *,
    multi_pass_diar: bool,
) -> dict[str, str]:
    """Batch caps for sequential diar→ASR staging (ASR owns the GPU alone)."""
    if co_resident or multi_pass_diar:
        return {
            "ASR_8GB_MAX_BATCH_SIZE": "1",
            "ASR_8GB_BATCH_SIZE": "1",
            "ASR_CUDA_BATCH_SIZE": "1",
        }
    max_batch = max(1, min(4, _env_int("ASR_8GB_MAX_BATCH_SIZE", 4)))
    cuda_batch = max(1, min(max_batch, _env_int("ASR_CUDA_BATCH_SIZE", max_batch)))
    return {
        "ASR_8GB_MAX_BATCH_SIZE": str(max_batch),
        "ASR_8GB_BATCH_SIZE": str(cuda_batch),
        "ASR_CUDA_BATCH_SIZE": str(cuda_batch),
    }


def _cap_cuda_batch_in_forced(
    forced: dict[str, str],
    multi_pass_diar: bool,
    co_resident: bool,
) -> None:
    if co_resident or multi_pass_diar:
        forced.update(_sequential_staging_batch_env(co_resident, multi_pass_diar=multi_pass_diar))
        return
    # 8 GB sequential staging: one utterance at a time — batch>1 OOMs with beams+Typhoon.
    forced.update({
        "ASR_8GB_MAX_BATCH_SIZE": "1",
        "ASR_8GB_BATCH_SIZE": "1",
        "ASR_CUDA_BATCH_SIZE": "1",
    })


def _apply_env_overrides(
    values: dict[str, str],
    applied: list[str],
    *,
    only_if_unset: bool = False,
) -> None:
    for key, value in values.items():
        if only_if_unset and os.getenv(key, "").strip():
            continue
        if os.getenv(key, "") != value:
            os.environ[key] = value
            applied.append(f"{key}={value}")


def apply_low_vram_overrides() -> list[str]:
    """Force safe env defaults on 8 GB CUDA GPUs (overrides risky compose/.env values)."""
    try:
        from backend.services.asr_local import strict_memory_mode_active

        if not strict_memory_mode_active():
            return []
    except ImportError:
        return []

    chunk_cap = _8gb_chunk_cap()
    co_resident = _env_bool("DIARIZATION_GPU_CO_RESIDENT", False)
    multi_pass_diar = _multi_pass_diarization_active_env()
    unload_for_diar = "true"
    forced = _build_low_vram_forced_env(
        chunk_cap, co_resident, unload_for_diar, multi_pass_diar=multi_pass_diar,
    )
    _cap_cuda_batch_in_forced(forced, multi_pass_diar, co_resident)
    _clamp_cuda_memory_fraction(forced)

    applied: list[str] = []
    _apply_env_overrides(forced, applied)
    _apply_env_overrides(
        {
            "DIARIZATION_MULTI_SAMPLE_PASSES_8GB": "0",
            "DIARIZATION_MULTI_SAMPLE_EARLY_STOP": "true",
        },
        applied,
        only_if_unset=True,
    )
    if applied:
        logger.info("8 GB VRAM safety overrides: %s", ", ".join(applied))
    return applied


# Fixture-adaptive audio enhancement (profiles A/B/C from enterprise plan).
_ENHANCE_PROFILE_SHORT: dict[str, str] = {
    "AUDIO_ENHANCE_WHEN_DIARIZATION": "true",
    "AUDIO_ENHANCE_DEFAULT": "true",
    "AUDIO_ENHANCE_ASR_ONLY": "false",
    "AUDIO_ENHANCE_NOISE_REDUCTION": "0.45",
    "AUDIO_ENHANCE_LOUDNORM_I": "-14.0",
    "AUDIO_ENHANCE_TARGET_PEAK_DB": "-1.0",
    "AUDIO_ENHANCE_MAX_GAIN_DB": "18",
    "AUDIO_ENHANCE_ATEMPO": "1.0",
    "AUDIO_ENHANCE_NOISE_PROFILE": "leading",
    "AUDIO_ENHANCE_NOISE_PROFILE_SECONDS": "0.75",
    "AUDIO_ENHANCE_GATE_THRESHOLD_DB": "-42",
    "DIARIZATION_NOISE_REDUCTION": "0.0",
}

_ENHANCE_PROFILE_LONG: dict[str, str] = {
    "AUDIO_ENHANCE_WHEN_DIARIZATION": "false",
    "AUDIO_ENHANCE_DEFAULT": "false",
    "AUDIO_ENHANCE_ASR_ONLY": "true",
    "AUDIO_ENHANCE_NOISE_REDUCTION": "0.25",
    "AUDIO_ENHANCE_LOUDNORM_I": "-16.0",
    "AUDIO_ENHANCE_ATEMPO": "1.0",
    "DIARIZATION_NOISE_REDUCTION": "0.0",
}

_ENHANCE_PROFILE_TWO_SPK: dict[str, str] = {
    "AUDIO_ENHANCE_WHEN_DIARIZATION": "true",
    "AUDIO_ENHANCE_DEFAULT": "true",
    "AUDIO_ENHANCE_ASR_ONLY": "false",
    "AUDIO_ENHANCE_NOISE_REDUCTION": "0.35",
    "AUDIO_ENHANCE_LOUDNORM_I": "-14.0",
    "AUDIO_ENHANCE_ATEMPO": "1.0",
    "DIARIZATION_NOISE_REDUCTION": "0.0",
}


def select_enhance_profile(duration_s: float, max_speakers: int) -> dict[str, str]:
    """Return enhancement env dict for this job (profiles A/B/C)."""
    if max_speakers <= 2:
        return dict(_ENHANCE_PROFILE_TWO_SPK)
    if duration_s >= 30 * 60 and max_speakers >= 8:
        return dict(_ENHANCE_PROFILE_LONG)
    if max_speakers >= 3:
        return dict(_ENHANCE_PROFILE_SHORT)
    return {}


def apply_enhance_profile(duration_s: float, max_speakers: int) -> list[str]:
    """Apply fixture-adaptive enhancement env before normalize/enhance stages."""
    if not _env_bool("AUDIO_ENHANCE_ADAPTIVE", False):
        return []
    presets = select_enhance_profile(duration_s, max_speakers)
    if not presets:
        return []
    applied: list[str] = []
    _apply_env_overrides(presets, applied)
    if applied:
        logger.info(
            "Enhance profile (%.0fs, max_spk=%d): %s",
            duration_s,
            max_speakers,
            ", ".join(applied),
        )
    return applied


def apply_two_speaker_overrides(max_speakers: int) -> list[str]:
    """Phone-call jobs: skip multi-sample diar sweep, wider turn merge."""
    if max_speakers != 2:
        return []
    overrides = {
        "DIARIZATION_MULTI_SAMPLE": "false",
        "DIARIZATION_MULTI_SAMPLE_PASSES": "0",
        "ASR_TURN_GUIDED_MERGE_GAP_S": "0.35",
    }
    applied: list[str] = []
    _apply_env_overrides(overrides, applied)
    return applied


def apply_short_audio_asr_overrides(audio_duration_s: float) -> list[str]:
    """Tighter turn cap and full beams for short multi-speaker clips."""
    if audio_duration_s >= 600:
        return []
    overrides: dict[str, str] = {
        "ASR_TURN_GUIDED_MAX_TURN_S": "20",
        "ASR_TURN_PAD_S": "0.25",
    }
    if audio_duration_s < 10 * 60:
        overrides["ASR_NUM_BEAMS"] = os.getenv("ASR_NUM_BEAMS_MAX", "6")
    applied: list[str] = []
    _apply_env_overrides(overrides, applied, only_if_unset=False)
    return applied


def enhance_asr_only_enabled() -> bool:
    return _env_bool("AUDIO_ENHANCE_ASR_ONLY", False)


# Enterprise GPU compose defaults — mirrors docker-compose.gpu.yml + enterprise_config.
ENTERPRISE_DOCKER_ENV: dict[str, str] = {
    **ENTERPRISE_ACCURACY_BASE,
    "DIARIZATION_DEVICE": "cuda",
    "DIARIZATION_REQUIRE_CUDA": "1",
    "DIARIZATION_ALLOW_8GB_CUDA": "true",
    "DIARIZATION_PRELOAD_DEVICE": "cuda",
    "DIARIZATION_EXACT_NUM_SPEAKERS": "false",
    "CUDA_AUTO_RESTART": "1",
}


def apply_quality_profile() -> str:
    """Apply profile defaults only for env vars not already set (explicit .env wins)."""
    profile = active_quality_profile()
    if profile not in _PROFILE_DEFAULTS:
        logger.warning("Unknown ASR_QUALITY_PROFILE=%r; using high.", profile)
        profile = PROFILE_HIGH

    applied: list[str] = []
    for key, value in _PROFILE_DEFAULTS[profile].items():
        if not os.getenv(key, "").strip():
            os.environ[key] = value
            applied.append(key)

    logger.info(
        "ASR quality profile=%s%s",
        profile,
        f" applied: {', '.join(applied)}" if applied else " (no overrides; env already set)",
    )
    apply_low_vram_overrides()
    return profile
