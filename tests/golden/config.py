"""Environment profiles for golden-file automation."""

from __future__ import annotations

import os

# test-sample01: 4-speaker Thai dialogue (~3.5 min).
# GPU staging: CUDA diar + turn-guided Typhoon ASR on CUDA.
# Pass criteria: content >=90%, speaker+timestamp >=98%, elapsed <=10min.
GOLDEN_ACCURACY_ENV: dict[str, str] = {
    "GOLDEN_FIXTURE": "sample01",
    "GOLDEN_ACCURACY_THRESHOLD": "0.90",
    "GOLDEN_SPEAKER_THRESHOLD": "0.98",
    "GOLDEN_TIMESTAMP_THRESHOLD": "0.98",
    "GOLDEN_REQUIRE_GPU": "1",
    "GOLDEN_CHECK_PERFORMANCE": "1",
    "GOLDEN_REFERENCE_DIAR": "1",
    "ASR_ADAPTIVE_PERFORMANCE": "false",
    "ASR_QUALITY_PROFILE": "high",
    "DIARIZATION_ACCURACY_MODE": "true",
    "ASR_TURN_GUIDED": "true",
    "ASR_TURN_GUIDED_MAX_TURN_S": "60",
    "ASR_TURN_GUIDED_MERGE_GAP_S": "0.0",
    "ASR_TURN_BOUNDARY_TRIM_S": "0.0",
    "ASR_TURN_BOUNDARY_MARGIN_S": "0.04",
    "ASR_TURN_PAD_S": "0.25",
    "ASR_CLEANUP_THAI_SPACING": "true",
    "ASR_NUM_BEAMS": "6",
    "ASR_SUPPRESS_HALLUCINATIONS": "true",
    "ASR_NO_SPEECH_THRESHOLD": "0.6",
    "ASR_LOGPROB_THRESHOLD": "-0.5",
    "ASR_CUDA_BATCH_SIZE": "1",
    "ASR_8GB_MAX_BATCH_SIZE": "1",
    "ASR_8GB_CHUNK_LENGTH_S": "60",
    "ASR_8GB_MAX_CHUNK_LENGTH_S": "60",
    "ASR_CUDA_MEMORY_FRACTION": "0.85",
    "TYPHOON_WORD_TIMESTAMPS_ON_8GB": "false",
    "ASR_WORD_TIMESTAMPS_WITH_DIARIZATION": "true",
    "ASR_UNLOAD_FOR_DIARIZATION": "true",
    "ASR_KEEP_PRELOADED": "false",
    "DIARIZATION_KEEP_PRELOADED": "false",
    "DIARIZATION_PRELOAD_MODE": "lazy",
    "DIARIZATION_PRELOAD_DEVICE": "cuda",
    "DIARIZATION_DEVICE": "cuda",
    "DIARIZATION_REQUIRE_CUDA": "1",
    "DIARIZATION_ALLOW_8GB_CUDA": "true",
    "DIARIZATION_GPU_CO_RESIDENT": "false",
    "DIARIZATION_CUDA_MIN_VRAM_MB": "6000",
    "DIARIZATION_CUDA_MIN_FREE_MB": "768",
    "DIARIZATION_CUDA_RUN_MIN_FREE_MB": "512",
    "DIARIZATION_MULTI_SAMPLE": "false",
    "DIARIZATION_SEGMENTATION_THRESHOLD": "0.38",
    "DIARIZATION_CLUSTERING_THRESHOLD": "0.40",
    "DIARIZATION_ASSIGN_TURN_MERGE_GAP_S": "0.0",
    "DIARIZATION_TRANSCRIPT_MERGE_GAP_S": "0.0",
    "DIARIZATION_MIN_OVERLAP_S": "0.06",
    "AUDIO_ENHANCE_WHEN_DIARIZATION": "false",
    "AUDIO_ENHANCE_NOISE_REDUCTION": "0.0",
}

# Production GPU profile for long-audio performance smoke tests.
PRODUCTION_PERF_ENV: dict[str, str] = {
    "GOLDEN_REQUIRE_GPU": "1",
    "GOLDEN_CHECK_PERFORMANCE": "1",
    "ASR_ADAPTIVE_PERFORMANCE": "true",
    "ASR_QUALITY_PROFILE": "high",
    "ASR_TARGET_SHORT_MAX_S": "600",
    "ASR_TARGET_LONG_MAX_S": "1800",
    "ASR_TARGET_LONG_AUDIO_S": "3600",
    "ASR_TARGET_RT_RATIO_LONG": "4",
    "ASR_NUM_BEAMS_MAX": "8",
    "ASR_NUM_BEAMS_MIN": "4",
    "ASR_FAST_MODE": "false",
    "ASR_TURN_GUIDED": "true",
    "ASR_LONG_FORM_MIN_DURATION_S": "600",
    "ASR_LONG_FORM_WINDOW_S": "2400",
    "ASR_LONG_FORM_OVERLAP_S": "45",
    "DIARIZATION_MAX_ASR_WINDOW_S": "2400",
    "DIARIZATION_SEGMENT_S": "600",
    "DIARIZATION_SEGMENT_OVERLAP_S": "45",
    "DIARIZATION_DEVICE": "cuda",
    "DIARIZATION_REQUIRE_CUDA": "1",
    "DIARIZATION_MULTI_SAMPLE": "false",
    "DIARIZATION_REFINE_AFTER_SEGMENTED": "false",
    "DIARIZATION_PRELOAD_MODE": "lazy",
    "DIARIZATION_KEEP_PRELOADED": "false",
    "DIARIZATION_PREPROCESS_SR": "16000",
    "ASR_KEEP_PRELOADED": "false",
    "ASR_UNLOAD_FOR_DIARIZATION": "true",
    "AUDIO_ENHANCE_DEFAULT": "false",
    "AUDIO_ENHANCE_WHEN_DIARIZATION": "false",
}

# Single best GPU profile — no slow multi-profile sweep.
CONFIG_PROFILES: list[dict[str, str]] = [
    {},
]


def apply_golden_env(extra: dict[str, str] | None = None) -> list[str]:
    """Apply golden env vars; return keys that changed."""
    merged = {**GOLDEN_ACCURACY_ENV, **(extra or {})}
    applied: list[str] = []
    for key, value in merged.items():
        if os.getenv(key, "") != value:
            os.environ[key] = value
            applied.append(key)
    return applied


def apply_production_perf_env(extra: dict[str, str] | None = None) -> list[str]:
    """Apply production adaptive performance env for long-audio smoke tests."""
    merged = {**PRODUCTION_PERF_ENV, **(extra or {})}
    applied: list[str] = []
    for key, value in merged.items():
        if os.getenv(key, "") != value:
            os.environ[key] = value
            applied.append(key)
    return applied


def golden_accuracy_threshold() -> float:
    return float(os.getenv("GOLDEN_ACCURACY_THRESHOLD", "0.90"))


def golden_speaker_threshold() -> float:
    return float(os.getenv("GOLDEN_SPEAKER_THRESHOLD", "0.98"))


def golden_timestamp_threshold() -> float:
    return float(os.getenv("GOLDEN_TIMESTAMP_THRESHOLD", "0.98"))


def performance_check_enabled() -> bool:
    return os.getenv("GOLDEN_CHECK_PERFORMANCE", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
