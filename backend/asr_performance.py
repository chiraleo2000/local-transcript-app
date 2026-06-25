"""Adaptive ASR tuning — high accuracy within tiered realtime budgets."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def performance_target_seconds(audio_duration_s: float) -> float:
    """Target wall time by audio length.

    - Short clips (<20 min): up to ASR_TARGET_SHORT_MAX_S (default 10 min).
    - Medium long (20 min–1 h): up to ASR_TARGET_LONG_MAX_S (default 30 min).
    - 1 h+: min(long cap, audio / ASR_TARGET_RT_RATIO_LONG) — default 1/4 realtime.
    """
    if audio_duration_s <= 0:
        return 0.0
    short_cap = max(60.0, _env_float("ASR_TARGET_SHORT_MAX_S", 600.0))
    long_cap = max(short_cap, _env_float("ASR_TARGET_LONG_MAX_S", 1800.0))
    long_audio_s = max(60.0, _env_float("ASR_TARGET_LONG_AUDIO_S", 3600.0))
    medium_audio_s = max(60.0, _env_float("ASR_TARGET_MEDIUM_AUDIO_S", 20 * 60))
    ratio_long = max(1.0, _env_float("ASR_TARGET_RT_RATIO_LONG", 4.0))

    if audio_duration_s >= long_audio_s:
        return min(long_cap, audio_duration_s / ratio_long)
    if audio_duration_s >= medium_audio_s:
        return long_cap
    return short_cap


def adaptive_num_beams(audio_duration_s: float, *, diarization: bool) -> int:
    """Pick beam width from duration (higher on short clips)."""
    beam_max = max(1, _env_int("ASR_NUM_BEAMS_MAX", 8))
    beam_min = max(1, min(beam_max, _env_int("ASR_NUM_BEAMS_MIN", 4)))
    long_audio_s = _env_float("ASR_TARGET_LONG_AUDIO_S", 3600.0)

    if audio_duration_s < 5 * 60:
        beams = beam_max
    elif audio_duration_s < 12 * 60:
        beams = max(beam_min, min(beam_max, 6))
    elif audio_duration_s < 20 * 60:
        beams = max(beam_min, min(beam_max, 5))
    elif audio_duration_s >= long_audio_s:
        beams = beam_min
    else:
        beams = max(beam_min, min(beam_max, 4))

    if diarization and 12 * 60 <= audio_duration_s < long_audio_s:
        target = performance_target_seconds(audio_duration_s)
        diar_overhead = 25.0 + audio_duration_s * 0.03
        asr_budget = max(60.0, target - diar_overhead)
        merge_gap = adaptive_turn_merge_gap_s(audio_duration_s)
        sec_per_turn = 18.0 if merge_gap >= 0.7 else 14.0
        est_turns = max(1, int(audio_duration_s / sec_per_turn))
        per_turn = asr_budget / est_turns
        if per_turn < 8:
            beams = beam_min
        elif per_turn < 12:
            beams = min(beams, 4)
        elif per_turn < 16:
            beams = min(beams, 5)

    return max(beam_min, min(beam_max, beams))


def adaptive_turn_merge_gap_s(audio_duration_s: float) -> float:
    """Wider merge on long audio → fewer ASR passes."""
    long_audio_s = _env_float("ASR_TARGET_LONG_AUDIO_S", 3600.0)
    if audio_duration_s < 5 * 60:
        return 0.5
    if audio_duration_s < 12 * 60:
        return 0.6
    if audio_duration_s < 20 * 60:
        return 0.7
    if audio_duration_s < long_audio_s:
        return 0.85
    return 1.0


def adaptive_chunk_length_s(audio_duration_s: float) -> int:
    long_audio_s = _env_float("ASR_TARGET_LONG_AUDIO_S", 3600.0)
    if audio_duration_s < 8 * 60:
        return max(30, _env_int("ASR_8GB_CHUNK_LENGTH_S", 45))
    if audio_duration_s < 20 * 60:
        return max(45, _env_int("ASR_8GB_CHUNK_LENGTH_S", 60))
    if audio_duration_s < long_audio_s:
        return max(60, _env_int("ASR_8GB_CHUNK_LENGTH_S", 90))
    return max(90, _env_int("ASR_8GB_CHUNK_LENGTH_S", 120))


def apply_performance_policy(audio_duration_s: float, *, diarization: bool) -> dict[str, str]:
    """Apply per-job ASR settings for accuracy within the realtime budget."""
    if not _env_bool("ASR_ADAPTIVE_PERFORMANCE", True):
        return {}

    beams = adaptive_num_beams(audio_duration_s, diarization=diarization)
    merge_gap = adaptive_turn_merge_gap_s(audio_duration_s)
    chunk_s = adaptive_chunk_length_s(audio_duration_s)
    applied = {
        "ASR_NUM_BEAMS": str(beams),
        "ASR_TURN_GUIDED_MERGE_GAP_S": f"{merge_gap:.2f}",
        "ASR_8GB_CHUNK_LENGTH_S": str(chunk_s),
        "ASR_8GB_MAX_CHUNK_LENGTH_S": str(chunk_s),
    }
    long_audio_s = _env_float("ASR_TARGET_LONG_AUDIO_S", 3600.0)
    if audio_duration_s >= long_audio_s:
        beams = 1
        applied["ASR_FAST_MODE"] = "true"
        applied["ASR_TURN_GUIDED"] = "false"
        applied["ASR_LONG_FORM_MIN_DURATION_S"] = "600"
        applied["ASR_LONG_FORM_WINDOW_S"] = "2400"
        applied["ASR_LONG_FORM_OVERLAP_S"] = "45"
        applied["DIARIZATION_MAX_ASR_WINDOW_S"] = "2400"
        applied["DIARIZATION_SEGMENT_S"] = "600"
        applied["DIARIZATION_SEGMENT_OVERLAP_S"] = "45"
        applied["DIARIZATION_REFINE_AFTER_SEGMENTED"] = "false"
        applied["DIARIZATION_KEEP_PRELOADED"] = "false"
        applied["ASR_KEEP_PRELOADED"] = "false"
        applied["ASR_NUM_BEAMS"] = str(beams)

    for key, value in applied.items():
        os.environ[key] = value

    target = performance_target_seconds(audio_duration_s)
    logger.info(
        "Performance policy: audio=%.1fs target=%.1fs beams=%d merge_gap=%.2fs chunk=%ds",
        audio_duration_s,
        target,
        beams,
        merge_gap,
        chunk_s,
    )
    return applied
