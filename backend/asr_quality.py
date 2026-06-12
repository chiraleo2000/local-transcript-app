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
        "DIARIZATION_MULTI_SAMPLE_PASSES": "6",
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
        "AUDIO_ENHANCE_NOISE_REDUCTION": "0.92",
        "AUDIO_ENHANCE_LOUDNORM_I": "-14",
        "AUDIO_ENHANCE_TARGET_PEAK_DB": "-1.5",
        "AUDIO_ENHANCE_MAX_GAIN_DB": "18",
        "AUDIO_ENHANCE_ATEMPO": "0.92",
        "DIARIZATION_NOISE_REDUCTION": "0.0",
    },
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def active_quality_profile() -> str:
    profile = os.getenv("ASR_QUALITY_PROFILE", PROFILE_HIGH).strip().lower()
    if profile not in _PROFILE_DEFAULTS:
        return PROFILE_HIGH
    return profile


def is_high_quality_profile() -> bool:
    return active_quality_profile() == PROFILE_HIGH


def is_accuracy_mode() -> bool:
    return _env_bool("DIARIZATION_ACCURACY_MODE", is_high_quality_profile())


def apply_low_vram_overrides() -> list[str]:
    """Force safe env defaults on 8 GB CUDA GPUs (overrides risky compose/.env values)."""
    try:
        from backend.services.asr_local import strict_memory_mode_active

        if not strict_memory_mode_active():
            return []
    except ImportError:
        return []

    chunk_cap = "90" if _env_bool("ASR_8GB_ALLOW_LARGE_CHUNKS", False) else "60"
    diar_device = "auto" if _env_bool("DIARIZATION_GPU_CO_RESIDENT", False) else "cpu"
    forced = {
        "PATHUMMA_WORD_TIMESTAMPS_ON_8GB": "false",
        "ASR_8GB_CHUNK_LENGTH_S": chunk_cap,
        "ASR_8GB_MAX_CHUNK_LENGTH_S": chunk_cap,
        "ASR_8GB_RETRY_CHUNK_LENGTH_S": "15",
        "DIARIZATION_PRELOAD_DEVICE": "cpu",
        "DIARIZATION_DEVICE": diar_device,
        "DIARIZATION_ALLOW_8GB_CUDA": "false",
        "ASR_UNLOAD_FOR_DIARIZATION": "false",
    }
    applied: list[str] = []
    for key, value in forced.items():
        if os.getenv(key, "") != value:
            os.environ[key] = value
            applied.append(f"{key}={value}")
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
