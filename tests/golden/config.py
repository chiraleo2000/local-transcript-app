"""Environment profiles for golden-file automation."""

from __future__ import annotations

import os

from backend.asr_quality import ENTERPRISE_DOCKER_ENV

# test-sample01: 4-speaker Thai dialogue (~3.5 min).
# GPU staging: CUDA diar + turn-guided Typhoon ASR on CUDA (single-pass VBx).
# Accuracy is measured against REAL pyannote diarization output — never
# reference-injected turns (GOLDEN_REFERENCE_DIAR stays off so the score
# reflects what the shipped app actually produces).
GOLDEN_ACCURACY_ENV: dict[str, str] = {
    "GOLDEN_FIXTURE": "sample01",
    "GOLDEN_ACCURACY_THRESHOLD": "0.99",
    "GOLDEN_SPEAKER_THRESHOLD": "0.98",
    "GOLDEN_TIMESTAMP_THRESHOLD": "0.98",
    "GOLDEN_REQUIRE_GPU": "1",
    "GOLDEN_CHECK_PERFORMANCE": "1",
    "GOLDEN_REFERENCE_DIAR": "0",
    "ASR_ADAPTIVE_PERFORMANCE": "false",
    "ASR_QUALITY_PROFILE": "high",
    "DIARIZATION_ACCURACY_MODE": "true",
    "ASR_TURN_GUIDED": "true",
    "ASR_TURN_USE_DIAR_TIMESTAMPS": "true",
    "ASR_TURN_GUIDED_MAX_TURN_S": "20",
    "ASR_TURN_GUIDED_MERGE_GAP_S": "0.35",
    "ASR_TURN_OUTPUT_MERGE_GAP_S": "1.5",
    "ASR_TURN_BOUNDARY_TRIM_S": "0.0",
    "ASR_TURN_BOUNDARY_MARGIN_S": "0.04",
    "ASR_TURN_PAD_S": "0.18",
    "ASR_CLEANUP_THAI_SPACING": "true",
    "ASR_NUM_BEAMS": "6",
    "ASR_SUPPRESS_HALLUCINATIONS": "true",
    "ASR_REJECT_HALLUCINATED_TURNS": "true",
    "ASR_NO_SPEECH_THRESHOLD": "0.6",
    "ASR_LOGPROB_THRESHOLD": "-0.5",
    "ASR_CUDA_BATCH_SIZE": "1",
    "ASR_8GB_MAX_BATCH_SIZE": "1",
    "ASR_8GB_CHUNK_LENGTH_S": "30",
    "ASR_8GB_MAX_CHUNK_LENGTH_S": "30",
    "TYPHOON_WORD_TIMESTAMPS_ON_8GB": "false",
    "ASR_WORD_TIMESTAMPS_WITH_DIARIZATION": "false",
    "ASR_UNLOAD_FOR_DIARIZATION": "true",
    "ASR_KEEP_PRELOADED": "true",
    "DIARIZATION_KEEP_PRELOADED": "false",
    "DIARIZATION_PRELOAD_MODE": "eager",
    "DIARIZATION_PRELOAD_DEVICE": "cuda",
    "DIARIZATION_DEVICE": "cuda",
    "DIARIZATION_REQUIRE_CUDA": "1",
    "DIARIZATION_ALLOW_8GB_CUDA": "true",
    "DIARIZATION_GPU_CO_RESIDENT": "false",
    "DIARIZATION_CUDA_MIN_VRAM_MB": "6000",
    "DIARIZATION_CUDA_MIN_FREE_MB": "768",
    "DIARIZATION_CUDA_RUN_MIN_FREE_MB": "512",
    "DIARIZATION_MULTI_SAMPLE": "false",
    "DIARIZATION_MULTI_SAMPLE_PASSES": "0",
    "DIARIZATION_MIN_DURATION_OFF": "0.02",
    "DIARIZATION_SEGMENTATION_THRESHOLD": "0.31",
    "DIARIZATION_CLUSTERING_THRESHOLD": "0.38",
    "DIARIZATION_LOCK_PARAMS": "true",
    "DIARIZATION_VBX_FA": "0.30",
    "DIARIZATION_EXACT_NUM_SPEAKERS": "true",
    "AUDIO_ENHANCE_ADAPTIVE": "false",
    "DIARIZATION_REFINE_AFTER_SEGMENTED": "false",
    "DIARIZATION_MEGA_TURN_MAX_REFINES": "0",
    "DIARIZATION_DOMINANCE_RETRY_RATIO": "0.80",
    "DIARIZATION_ASSIGN_TURN_MERGE_GAP_S": "0.0",
    "DIARIZATION_TRANSCRIPT_MERGE_GAP_S": "0.0",
    "DIARIZATION_MIN_OVERLAP_S": "0.06",
    "DIARIZATION_ASSIGN_IMBALANCE_RATIO": "0.65",
    "AUDIO_ENHANCE_WHEN_DIARIZATION": "false",
    "AUDIO_ENHANCE_ASR_ONLY": "true",
    "AUDIO_ENHANCE_NOISE_REDUCTION": "0.35",
    "AUDIO_ENHANCE_TARGET_PEAK_DB": "-1.0",
    "AUDIO_ENHANCE_MAX_GAIN_DB": "18",
    "AUDIO_ENHANCE_LOUDNORM_I": "-14",
    "AUDIO_ENHANCE_ATEMPO": "1.0",
}

# Production GPU profile for long-audio performance smoke tests.
PRODUCTION_PERF_ENV: dict[str, str] = {
    "GOLDEN_REQUIRE_GPU": "1",
    "GOLDEN_CHECK_PERFORMANCE": "1",
    "ASR_ADAPTIVE_PERFORMANCE": "false",
    "ASR_QUALITY_PROFILE": "high",
    "ASR_TARGET_SHORT_MAX_S": "600",
    "ASR_TARGET_LONG_MAX_S": "0",
    "ASR_TARGET_MEDIUM_AUDIO_S": "1200",
    "ASR_TARGET_LONG_AUDIO_S": "3600",
    "ASR_TARGET_RT_RATIO_LONG": "2",
    "ASR_BUDGET_SEC_PER_TURN": "4.5",
    "ASR_DIAR_WINDOWED_FAST": "false",
    "ASR_DIAR_WINDOWED_MIN_DURATION_S": "999999",
    "ASR_DIAR_WINDOWED_TURN_THRESHOLD": "999999",
    "ASR_DIAR_WINDOWED_WINDOW_S": "300",
    "ASR_NUM_BEAMS_MAX": "6",
    "ASR_NUM_BEAMS_MIN": "4",
    "ASR_FAST_MODE": "false",
    "ASR_TURN_GUIDED": "true",
    "ASR_TURN_USE_DIAR_TIMESTAMPS": "true",
    "ASR_TURN_GUIDED_MAX_DURATION_S": "86400",
    "ASR_LONG_FORM_MIN_DURATION_S": "600",
    "ASR_LONG_FORM_WINDOW_S": "300",
    "ASR_LONG_FORM_OVERLAP_S": "60",
    "DIARIZATION_MAX_ASR_WINDOW_S": "300",
    "DIARIZATION_SEGMENT_S": "1200",
    "DIARIZATION_SEGMENT_OVERLAP_S": "60",
    "DIARIZATION_SEGMENT_LONG_AUDIO_MIN_S": "7200",
    "DIARIZATION_DEVICE": "cuda",
    "DIARIZATION_REQUIRE_CUDA": "1",
    "DIARIZATION_ACCURACY_MODE": "true",
    "DIARIZATION_MULTI_SAMPLE": "false",
    "DIARIZATION_MULTI_SAMPLE_PASSES": "0",
    "DIARIZATION_INTRO_RECOVERY": "false",
    "DIARIZATION_REFINE_AFTER_SEGMENTED": "false",
    "DIARIZATION_MEGA_TURN_RETRY_S": "35",
    "DIARIZATION_MEGA_TURN_MAX_REFINES": "12",
    "DIARIZATION_DOMINANCE_RETRY_RATIO": "0.75",
    "DIARIZATION_PRELOAD_MODE": "eager",
    "DIARIZATION_KEEP_PRELOADED": "false",
    "DIARIZATION_PREPROCESS_SR": "16000",
    "ASR_KEEP_PRELOADED": "true",
    "ASR_PRELOAD_MODE": "eager",
    "ASR_UNLOAD_FOR_DIARIZATION": "true",
    "ASR_REJECT_HALLUCINATED_TURNS": "true",
    "ASR_WORD_TIMESTAMPS_WITH_DIARIZATION": "false",
    "AUDIO_ENHANCE_DEFAULT": "false",
    "AUDIO_ENHANCE_WHEN_DIARIZATION": "false",
}

_MEETING_GATES: dict[str, str] = {
    "GOLDEN_MEETING_SPEAKER_TOLERANCE": "0",
    "GOLDEN_MEETING_TIME_ACC": "0.85",
    "GOLDEN_MEETING_TURN_ACC": "0.90",
    "GOLDEN_MEETING_BOUNDARY_1S": "0.70",
    "GOLDEN_MEETING_BOUNDARY_MEDIAN_S": "1.0",
    "GOLDEN_MEETING_TURN_TEXT_ACC": "0.85",
    "GOLDEN_MEETING_BOUNDARY_2S": "0.70",
}

# Unified enterprise acceptance profile (sample01 + meeting309).
# ENTERPRISE_DOCKER_ENV from backend.asr_quality is the single source of truth
# for production CUDA defaults; meeting309 adds long-audio overrides at runtime.
ENTERPRISE_ENV: dict[str, str] = {
    **PRODUCTION_PERF_ENV,
    **ENTERPRISE_DOCKER_ENV,
    **_MEETING_GATES,
    "ASR_ADAPTIVE_PERFORMANCE": "false",
    "GOLDEN_ACCURACY_THRESHOLD": "0.99",
    "GOLDEN_SPEAKER_THRESHOLD": "0.98",
    "GOLDEN_TIMESTAMP_THRESHOLD": "0.55",
    "GOLDEN_STRICT_THRESHOLD": "0.83",
    "GOLDEN_SAMPLE01_MISMATCHED_MAX": "2",
}

from backend.enterprise_config import ENTERPRISE_LONG_AUDIO_ENV

# Single best GPU profile — no slow multi-profile sweep.
CONFIG_PROFILES: list[dict[str, str]] = [
    {},
]


def apply_golden_env(extra: dict[str, str] | None = None) -> list[str]:
    """Apply golden env vars; return keys that changed."""
    merged = {**GOLDEN_ACCURACY_ENV, **ENTERPRISE_DOCKER_ENV, **(extra or {})}
    applied: list[str] = []
    for key, value in merged.items():
        if os.getenv(key, "") != value:
            os.environ[key] = value
            applied.append(key)
    return applied


def apply_production_perf_env(extra: dict[str, str] | None = None) -> list[str]:
    """Apply production adaptive performance env for long-audio smoke tests."""
    merged = {**PRODUCTION_PERF_ENV, **ENTERPRISE_DOCKER_ENV, **(extra or {})}
    applied: list[str] = []
    for key, value in merged.items():
        if os.getenv(key, "") != value:
            os.environ[key] = value
            applied.append(key)
    return applied


def apply_enterprise_env(extra: dict[str, str] | None = None) -> list[str]:
    """Apply enterprise acceptance env for full validation runs.

    Only fills keys not already set so ``docker compose run -e`` and fixture
    overlays win over the canonical enterprise defaults.
    """
    merged = {**ENTERPRISE_ENV, **(extra or {})}
    applied: list[str] = []
    for key, value in merged.items():
        if not os.getenv(key, "").strip():
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
