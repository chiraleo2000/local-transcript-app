"""ASR quality profiles — preset env defaults applied before model preload."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

PROFILE_BALANCED = "balanced"
PROFILE_HIGH = "high"

_PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    PROFILE_BALANCED: {
        "ASR_CHUNK_LENGTH_S": "30",
        "ASR_8GB_CHUNK_LENGTH_S": "30",
        "ASR_8GB_MAX_CHUNK_LENGTH_S": "40",
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
        "ASR_CHUNK_LENGTH_S": "300",
        "ASR_8GB_CHUNK_LENGTH_S": "90",
        "ASR_8GB_MAX_CHUNK_LENGTH_S": "90",
        "ASR_8GB_RETRY_CHUNK_LENGTH_S": "15",
        "ASR_CUDA_BATCH_SIZE": "4",
        "ASR_8GB_BATCH_SIZE": "4",
        "ASR_8GB_MAX_BATCH_SIZE": "4",
        "ASR_MIN_CHUNKED_DURATION_S": "120",
        "ASR_LONG_FORM_WINDOW_S": "300",
        "ASR_LONG_FORM_OVERLAP_S": "45",
        "PATHUMMA_WORD_TIMESTAMPS_ON_8GB": "true",
        "TYPHOON_WORD_TIMESTAMPS_ON_8GB": "false",
        "ASR_WORD_TIMESTAMPS_WITH_DIARIZATION": "true",
        "DIARIZATION_TRANSCRIPT_MERGE_GAP_S": "1.0",
        "DIARIZATION_ACCURACY_MODE": "true",
        "DIARIZATION_MULTI_SAMPLE": "true",
        "DIARIZATION_MULTI_SAMPLE_PASSES": "3",
        "DIARIZATION_MULTI_SAMPLE_PASSES_8GB": "3",
        "DIARIZATION_MULTI_SAMPLE_MAX_TOTAL": "9",
        "DIARIZATION_MULTI_SAMPLE_EARLY_STOP": "true",
        "DIARIZATION_MULTI_SAMPLE_TUNE_WINDOW": "true",
        "ASR_TURN_GUIDED": "true",
        "ASR_TURN_GUIDED_MAX_TURN_S": "35",
        "ASR_TURN_GUIDED_MERGE_GAP_S": "0.8",
        "DIARIZATION_MULTI_SAMPLE_TUNE_MIN_AUDIO_S": "300",
        "DIARIZATION_MULTI_SAMPLE_TUNE_MAX_S": "150",
        "DIARIZATION_PREPROCESS_SR": "44100",
        "DIARIZATION_SEGMENT_S": "360",
        "DIARIZATION_SEGMENT_OVERLAP_S": "90",
        "DIARIZATION_REFINE_AFTER_SEGMENTED": "true",
        "DIARIZATION_MEGA_TURN_MAX_REFINES": "10",
        "DIARIZATION_CHUNK_ALIGN_MIN_OVERLAP_S": "1.5",
        "DIARIZATION_MEGA_TURN_RETRY_S": "90",
        "DIARIZATION_ASSIGN_TURN_MERGE_GAP_S": "2.5",
        "DIARIZATION_SEGMENT_LONG_AUDIO": "true",
        "DIARIZATION_MAX_ASR_WINDOW_S": "300",
        "AUDIO_ENHANCE_DEFAULT": "true",
        "AUDIO_ENHANCE_WHEN_DIARIZATION": "true",
        "AUDIO_ENHANCE_NOISE_REDUCTION": "0.96",
        "AUDIO_ENHANCE_LOUDNORM_I": "-12",
        "AUDIO_ENHANCE_TARGET_PEAK_DB": "-0.5",
        "AUDIO_ENHANCE_MAX_GAIN_DB": "24",
        "AUDIO_ENHANCE_ATEMPO": "0.92",
        "DIARIZATION_NOISE_REDUCTION": "0.0",
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
    if _env_bool("ASR_TURN_GUIDED", False):
        return "60"
    if _env_bool("ASR_8GB_ALLOW_LARGE_CHUNKS", False):
        return "90"
    return "60"


def _multi_pass_diarization_active_env() -> bool:
    return _env_bool("DIARIZATION_MULTI_SAMPLE", False) and (
        _env_int("DIARIZATION_MULTI_SAMPLE_PASSES", 0) > 0
        or _env_bool("DIARIZATION_MULTI_SAMPLE_FULL_GRID", False)
    )


def _build_low_vram_forced_env(
    chunk_cap: str,
    co_resident: bool,
    unload_for_diar: str,
) -> dict[str, str]:
    requested = os.getenv("DIARIZATION_DEVICE", "auto").strip().lower()
    cuda_staging = (
        requested in {"cuda", "gpu"}
        and _env_bool("DIARIZATION_ALLOW_8GB_CUDA", False)
        and not co_resident
    )
    if co_resident:
        diar_device = "auto"
    elif cuda_staging:
        diar_device = requested
    else:
        diar_device = "cpu"
    preload_device = diar_device if cuda_staging else "cpu"
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
        "ASR_8GB_MAX_BATCH_SIZE": "1",
        "ASR_8GB_BATCH_SIZE": "1",
    }


def _cap_cuda_batch_in_forced(
    forced: dict[str, str],
    multi_pass_diar: bool,
    co_resident: bool,
) -> None:
    raw = os.getenv("ASR_CUDA_BATCH_SIZE", "").strip()
    if raw:
        try:
            if int(raw) > 1:
                forced["ASR_CUDA_BATCH_SIZE"] = "1"
        except ValueError:
            pass
        return
    if multi_pass_diar or co_resident:
        forced["ASR_CUDA_BATCH_SIZE"] = "1"


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
    forced = _build_low_vram_forced_env(chunk_cap, co_resident, unload_for_diar)
    _cap_cuda_batch_in_forced(forced, multi_pass_diar, co_resident)

    applied: list[str] = []
    _apply_env_overrides(forced, applied)
    _apply_env_overrides(
        {
            "DIARIZATION_MULTI_SAMPLE_PASSES_8GB": "4",
            "DIARIZATION_MULTI_SAMPLE_EARLY_STOP": "true",
        },
        applied,
        only_if_unset=True,
    )
    if applied:
        logger.info("8 GB VRAM safety overrides: %s", ", ".join(applied))
    return applied


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

    if profile == PROFILE_HIGH and _env_bool("ASR_8GB_ALLOW_LARGE_CHUNKS", False):
        large = os.getenv("ASR_CHUNK_LENGTH_S", "300")
        os.environ["ASR_8GB_MAX_CHUNK_LENGTH_S"] = large
        applied.append(f"ASR_8GB_MAX_CHUNK_LENGTH_S={large} (large-chunks)")

    logger.info(
        "ASR quality profile=%s%s",
        profile,
        f" applied: {', '.join(applied)}" if applied else " (no overrides; env already set)",
    )
    apply_low_vram_overrides()
    return profile
