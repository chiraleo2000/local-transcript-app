"""Adaptive ASR tuning — high accuracy within tiered realtime budgets."""

from __future__ import annotations

import logging
import math
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
    """Target wall time: tiered realtime budget.

  - Audio < ``ASR_TARGET_MEDIUM_AUDIO_S`` (default 20 min): flat cap
    ``ASR_TARGET_SHORT_MAX_S`` (default 600 s / 10 min) when > 0.
  - Audio >= medium threshold: ``audio_duration / ASR_TARGET_RT_RATIO_LONG``
    (default half realtime). Optional ``ASR_TARGET_LONG_MAX_S`` caps long jobs.
    """
    if audio_duration_s <= 0:
        return 0.0
    medium_threshold = max(60.0, _env_float("ASR_TARGET_MEDIUM_AUDIO_S", 20 * 60))
    short_max = _env_float("ASR_TARGET_SHORT_MAX_S", 600.0)
    ratio = max(1.0, _env_float("ASR_TARGET_RT_RATIO_LONG", 2.0))

    if audio_duration_s < medium_threshold and short_max > 0:
        return short_max

    target = audio_duration_s / ratio
    long_cap = _env_float("ASR_TARGET_LONG_MAX_S", 0.0)
    if long_cap > 0:
        return min(target, long_cap)
    return target


def _estimated_diar_overhead_s(audio_duration_s: float) -> float:
    base = _env_float("ASR_DIAR_OVERHEAD_BASE_S", 75.0)
    per_min = _env_float("ASR_DIAR_OVERHEAD_PER_MIN_S", 1.0)
    return base + (audio_duration_s / 60.0) * per_min


def adaptive_diar_window_s(audio_duration_s: float) -> int:
    """ASR window size for windowed diarization fast path."""
    configured = _env_int("ASR_DIAR_WINDOWED_WINDOW_S", 0)
    if configured > 0 and audio_duration_s < 45 * 60:
        return configured
    if audio_duration_s < 45 * 60:
        return max(300, _env_int("ASR_DIAR_WINDOWED_WINDOW_S", 480))
    if audio_duration_s < 75 * 60:
        return max(480, _env_int("ASR_DIAR_WINDOWED_WINDOW_S", 600))
    if audio_duration_s < 120 * 60:
        return max(600, _env_int("ASR_DIAR_WINDOWED_WINDOW_S", 720))
    return max(720, _env_int("ASR_DIAR_WINDOWED_WINDOW_S", 900))


def should_use_windowed_diar_asr(
    audio_duration_s: float,
    diarization_segments: list[dict],
    max_speakers: int = 0,
) -> bool:
    """Use few large ASR windows + turn-centric assign_speakers on long dialogue."""
    from backend.asr_quality import is_accuracy_mode

    if is_accuracy_mode():
        return False
    if not _env_bool("ASR_DIAR_WINDOWED_FAST", True):
        return False
    if not diarization_segments:
        return False

    min_duration_s = _env_float("ASR_DIAR_WINDOWED_MIN_DURATION_S", 30 * 60)
    if audio_duration_s < min_duration_s:
        return False

    medium_cap_s = max(60.0, _env_float("ASR_TARGET_MEDIUM_AUDIO_S", 20 * 60))
    if audio_duration_s <= medium_cap_s:
        return False

    from engines.diarization import prepare_asr_turns

    est_turns = len(prepare_asr_turns(diarization_segments, max_speakers))
    turn_threshold = _env_int("ASR_DIAR_WINDOWED_TURN_THRESHOLD", 50)
    if est_turns >= turn_threshold:
        logger.info(
            "Windowed diar ASR: %d estimated turns >= threshold %d (audio=%.0fs).",
            est_turns,
            turn_threshold,
            audio_duration_s,
        )
        return True

    long_only_s = _env_float("ASR_DIAR_WINDOWED_LONG_ONLY_S", 45 * 60)
    return audio_duration_s >= long_only_s


def adaptive_turn_merge_gap_s(audio_duration_s: float) -> float:
    """Wider merge on long audio → fewer turn-guided ASR passes."""
    from backend.asr_quality import is_accuracy_mode

    configured = os.getenv("ASR_TURN_GUIDED_MERGE_GAP_S", "").strip()
    if configured and audio_duration_s < 5 * 60 and is_accuracy_mode():
        try:
            value = float(configured)
            if value > 0:
                return value
        except ValueError:
            pass

    if audio_duration_s < 5 * 60:
        return 0.35 if is_accuracy_mode() else 0.5
    if audio_duration_s < 12 * 60:
        return 0.55
    if audio_duration_s < 20 * 60:
        return 0.75
    if audio_duration_s < 45 * 60:
        return 1.0
    if audio_duration_s < 75 * 60:
        return 1.5
    return 2.0


def adaptive_turn_max_s(audio_duration_s: float) -> float:
    from engines.whisper_utils import whisper_max_asr_turn_body_s

    configured = _env_int("ASR_TURN_GUIDED_MAX_TURN_S", 90)
    if audio_duration_s < 600:
        value = float(min(configured, 20))
    elif audio_duration_s < 20 * 60:
        value = float(min(configured, 60))
    elif audio_duration_s < 45 * 60:
        value = float(min(max(configured, 75), 90))
    else:
        value = float(min(max(configured, 90), 120))
    return min(value, whisper_max_asr_turn_body_s())


def adaptive_turn_settings_for_diarization(
    audio_duration_s: float,
) -> tuple[float, float]:
    """Tune merge gap + max turn size for turn-guided jobs."""
    merge_gap = adaptive_turn_merge_gap_s(audio_duration_s)
    max_turn = adaptive_turn_max_s(audio_duration_s)

    target = performance_target_seconds(audio_duration_s)
    diar_overhead = _estimated_diar_overhead_s(audio_duration_s)
    asr_budget = max(120.0, target - diar_overhead)
    sec_per_turn = max(4.5, _env_float("ASR_BUDGET_SEC_PER_TURN", 6.0))
    max_turns = max(24, int(asr_budget / sec_per_turn))

    raw_turn_est = max(1, int(audio_duration_s / 12.0))
    if raw_turn_est <= max_turns:
        return merge_gap, max_turn

    compression = raw_turn_est / max_turns
    merge_gap = min(3.5, merge_gap + 0.4 * math.log(compression))
    from engines.whisper_utils import whisper_max_asr_turn_body_s

    max_turn = min(
        whisper_max_asr_turn_body_s(),
        max(max_turn, 60.0 * compression ** 0.35),
    )
    logger.info(
        "Turn budget tuning: audio=%.0fs target=%.0fs asr_budget=%.0fs "
        "est_turns=%d cap=%d → merge_gap=%.2fs max_turn=%.0fs",
        audio_duration_s,
        target,
        asr_budget,
        raw_turn_est,
        max_turns,
        merge_gap,
        max_turn,
    )
    return merge_gap, max_turn


def adaptive_num_beams(audio_duration_s: float, *, diarization: bool) -> int:
    beam_max = max(1, _env_int("ASR_NUM_BEAMS_MAX", 8))
    beam_min = max(1, min(beam_max, _env_int("ASR_NUM_BEAMS_MIN", 4)))
    long_audio_s = _env_float("ASR_TARGET_LONG_AUDIO_S", 3600.0)

    if audio_duration_s < 10 * 60:
        beams = beam_max
    elif audio_duration_s < 12 * 60:
        beams = max(beam_min, min(beam_max, 6))
    elif audio_duration_s < 20 * 60:
        beams = max(beam_min, min(beam_max, 5))
    elif audio_duration_s >= long_audio_s:
        beams = beam_min
    else:
        beams = max(beam_min, min(beam_max, 4))

    if diarization and 12 * 60 <= audio_duration_s < _env_float(
        "ASR_DIAR_WINDOWED_MIN_DURATION_S", 30 * 60,
    ):
        target = performance_target_seconds(audio_duration_s)
        diar_overhead = _estimated_diar_overhead_s(audio_duration_s)
        asr_budget = max(60.0, target - diar_overhead)
        merge_gap, max_turn = adaptive_turn_settings_for_diarization(audio_duration_s)
        avg_turn_s = max(12.0, max_turn * 0.55)
        est_turns = max(1, int(audio_duration_s / avg_turn_s))
        per_turn = asr_budget / est_turns
        if per_turn < 7:
            beams = beam_min
        elif per_turn < 11:
            beams = min(beams, 4)
        elif per_turn < 15:
            beams = min(beams, 5)
        del merge_gap, max_turn, target, diar_overhead, asr_budget

    return max(beam_min, min(beam_max, beams))


def adaptive_chunk_length_s(audio_duration_s: float) -> int:
    """Chunk length for the HF Whisper pipeline.

    Whisper's encoder only processes 30s per chunk; the feature extractor
    truncates anything longer and silently drops audio. Growing the chunk with
    duration (the old 45-120s behaviour) therefore lost most of every chunk, so
    we always clamp to the encoder ceiling.
    """
    del audio_duration_s
    from engines.whisper_utils import WHISPER_MAX_CHUNK_S

    configured = _env_int("ASR_8GB_CHUNK_LENGTH_S", WHISPER_MAX_CHUNK_S)
    return max(10, min(WHISPER_MAX_CHUNK_S, configured))


def apply_performance_policy(audio_duration_s: float, *, diarization: bool) -> dict[str, str]:
    """Apply per-job ASR settings for accuracy within the realtime budget."""
    if not _env_bool("ASR_ADAPTIVE_PERFORMANCE", True):
        return {}

    beams = adaptive_num_beams(audio_duration_s, diarization=diarization)
    chunk_s = adaptive_chunk_length_s(audio_duration_s)
    beam_min = max(1, _env_int("ASR_NUM_BEAMS_MIN", 4))
    turn_guided_max_s = _env_float("ASR_TURN_GUIDED_MAX_DURATION_S", 20 * 60)

    windowed = diarization and audio_duration_s >= _env_float(
        "ASR_DIAR_WINDOWED_MIN_DURATION_S", 30 * 60,
    )
    if windowed:
        merge_gap = adaptive_turn_merge_gap_s(audio_duration_s)
        max_turn_s = adaptive_turn_max_s(audio_duration_s)
        diar_window = adaptive_diar_window_s(audio_duration_s)
    elif diarization and audio_duration_s >= 12 * 60:
        merge_gap, max_turn_s = adaptive_turn_settings_for_diarization(audio_duration_s)
        diar_window = 0
    else:
        merge_gap = adaptive_turn_merge_gap_s(audio_duration_s)
        max_turn_s = adaptive_turn_max_s(audio_duration_s)
        diar_window = 0

    applied = {
        "ASR_NUM_BEAMS": str(beams),
        "ASR_TURN_GUIDED_MERGE_GAP_S": f"{merge_gap:.2f}",
        "ASR_TURN_GUIDED_MAX_TURN_S": str(int(max_turn_s)),
        "ASR_8GB_CHUNK_LENGTH_S": str(chunk_s),
        "ASR_8GB_MAX_CHUNK_LENGTH_S": str(chunk_s),
    }

    if windowed:
        applied["ASR_DIAR_WINDOWED_WINDOW_S"] = str(diar_window)
        applied["ASR_WORD_TIMESTAMPS_WITH_DIARIZATION"] = "false"

    if audio_duration_s >= turn_guided_max_s and not diarization:
        long_beams = max(beam_min, min(5, beams))
        applied.update({
            "ASR_TURN_GUIDED": "false",
            "ASR_FAST_MODE": "false",
            "ASR_NUM_BEAMS": str(long_beams),
            "ASR_LONG_FORM_MIN_DURATION_S": str(
                max(60, int(_env_float("ASR_LONG_FORM_MIN_DURATION_S", 600))),
            ),
            "ASR_LONG_FORM_WINDOW_S": os.getenv("ASR_LONG_FORM_WINDOW_S", "2400"),
            "ASR_LONG_FORM_OVERLAP_S": os.getenv("ASR_LONG_FORM_OVERLAP_S", "45"),
        })
    elif diarization:
        applied["ASR_TURN_GUIDED"] = "true"

    for key, value in applied.items():
        os.environ[key] = value

    target = performance_target_seconds(audio_duration_s)
    logger.info(
        "Performance policy: audio=%.1fs target=%.1fs beams=%s merge_gap=%.2fs "
        "max_turn=%.0fs chunk=%ds windowed=%s diar_window=%ds turn_guided=%s",
        audio_duration_s,
        target,
        applied.get("ASR_NUM_BEAMS"),
        merge_gap,
        max_turn_s,
        chunk_s,
        windowed,
        diar_window,
        os.getenv("ASR_TURN_GUIDED", "true"),
    )
    return applied
