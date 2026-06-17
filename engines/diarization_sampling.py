"""Multi-pass pyannote diarization — sample configs and pick the best scoring result."""

from __future__ import annotations

import copy
import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)


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


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def _accuracy_mode() -> bool:
    try:
        from backend.asr_quality import is_accuracy_mode

        return is_accuracy_mode()
    except ImportError:
        return _env_bool("DIARIZATION_ACCURACY_MODE", False)


def multi_sample_enabled() -> bool:
    return _env_bool("DIARIZATION_MULTI_SAMPLE", False)


def multi_sample_full_grid() -> bool:
    """Run every grid config (slow; for offline tuning only)."""
    return _env_bool("DIARIZATION_MULTI_SAMPLE_FULL_GRID", False)


def multi_sample_pass_limit() -> int:
    """Max grid configs to try; 0 = adaptive base only (no sweep)."""
    return _env_int("DIARIZATION_MULTI_SAMPLE_PASSES", 0)


def multi_sample_early_stop_enabled() -> bool:
    return _env_bool("DIARIZATION_MULTI_SAMPLE_EARLY_STOP", True)


def multi_sample_preprocess_srs() -> list[int]:
    """Optional list of FFmpeg preprocess sample rates to compare (e.g. 16000,44100)."""
    raw = os.getenv("DIARIZATION_MULTI_SAMPLE_SR", "").strip()
    if not raw:
        return []
    srs: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value >= 8000:
            srs.append(value)
    return srs


def long_audio_tune_enabled() -> bool:
    """Tune hyperparameters on a short window, then run one full-file pass."""
    return _env_bool("DIARIZATION_MULTI_SAMPLE_TUNE_WINDOW", True)


def multi_sample_sweep_enabled() -> bool:
    """True when hyperparameter sweep should run (not just adaptive base)."""
    if not multi_sample_enabled():
        return False
    if multi_sample_full_grid():
        return True
    return multi_sample_pass_limit() > 0


def _strict_8gb_class() -> bool:
    try:
        from backend.services.asr_local import strict_memory_mode_active

        return strict_memory_mode_active()
    except ImportError:
        return False


def effective_pass_limit(audio_duration_s: float, max_speakers: int) -> int:
    """Duration- and VRAM-aware cap on how many configs to evaluate."""
    if not multi_sample_sweep_enabled():
        return 0
    limit = multi_sample_pass_limit()
    if multi_sample_full_grid():
        return limit
    if _strict_8gb_class():
        limit = min(limit, _env_int("DIARIZATION_MULTI_SAMPLE_PASSES_8GB", 4))
    if max_speakers <= 1:
        limit = min(limit, 3)
    elif audio_duration_s < 60.0:
        limit = min(limit, max(3, limit))
    elif audio_duration_s >= 600.0 and long_audio_tune_enabled():
        limit = min(limit, max(4, limit - 2))
    return max(1, limit)


def multi_sample_max_total_tries() -> int:
    """Global cap on diarization passes across all multi-SR rates combined."""
    return max(1, _env_int("DIARIZATION_MULTI_SAMPLE_MAX_TOTAL", 9))


def multi_sample_sr_count() -> int:
    srs = multi_sample_preprocess_srs()
    return max(1, len(srs))


def multi_sample_max_tries() -> int:
    """Per-SR cap on hyperparameter passes (including adaptive base)."""
    explicit = os.getenv("DIARIZATION_MULTI_SAMPLE_MAX_TRIES", "").strip()
    if explicit:
        return max(1, _env_int("DIARIZATION_MULTI_SAMPLE_MAX_TRIES", 3))
    return max(1, multi_sample_max_total_tries() // multi_sample_sr_count())


def tune_window_bounds(audio_duration_s: float) -> tuple[float, float] | None:
    """Return (start_s, end_s) for hyperparameter tuning, or None for full-file sweep."""
    if multi_sample_preprocess_srs():
        return None
    if not long_audio_tune_enabled():
        return None
    min_audio_s = _env_float("DIARIZATION_MULTI_SAMPLE_TUNE_MIN_AUDIO_S", 300.0)
    if audio_duration_s < min_audio_s:
        return None
    max_tune_s = _env_float("DIARIZATION_MULTI_SAMPLE_TUNE_MAX_S", 150.0)
    tune_dur = min(max_tune_s, max(60.0, audio_duration_s * 0.12))
    tune_start = min(
        max(30.0, audio_duration_s * 0.06),
        max(0.0, audio_duration_s - tune_dur - 45.0),
    )
    return tune_start, tune_start + tune_dur


def _base_seg_threshold(audio_duration_s: float) -> float:
    if 0 < audio_duration_s < 30.0:
        return 0.40
    if audio_duration_s >= 600.0:
        return 0.44
    return 0.42


def _min_cluster_size(max_speakers: int) -> int:
    if max_speakers >= 4:
        return 2
    if max_speakers >= 3:
        return 2
    return 3


def _curated_accuracy_configs(max_speakers: int) -> list[tuple[str, dict]]:
    """Hand-picked community-1 combos that cover the highest-impact tuning region."""
    min_cluster = _min_cluster_size(max_speakers)
    if max_speakers == 2:
        combos = [
            (0.38, 0.44, 0.03),
            (0.36, 0.42, 0.03),
            (0.40, 0.46, 0.04),
            (0.38, 0.48, 0.05),
            (0.40, 0.44, 0.04),
        ]
    elif max_speakers >= 3:
        combos = [
            (0.36, 0.42, 0.03),
            (0.38, 0.44, 0.03),
            (0.38, 0.46, 0.04),
            (0.40, 0.48, 0.05),
            (0.36, 0.40, 0.03),
            (0.38, 0.50, 0.04),
        ]
    else:
        combos = [
            (0.40, 0.48, 0.04),
            (0.38, 0.46, 0.03),
            (0.42, 0.50, 0.05),
        ]
    configs: list[tuple[str, dict]] = []
    for seg_t, clust_t, min_off in combos:
        label = f"curated_seg={seg_t:.2f}_clust={clust_t:.2f}"
        configs.append((
            label,
            {
                "segmentation": {"threshold": seg_t, "min_duration_off": min_off},
                "clustering": {"threshold": clust_t, "min_cluster_size": min_cluster},
            },
        ))
    return configs


def _accuracy_sweep_params(max_speakers: int) -> tuple[list[float], list[float], list[float], int]:
    cluster_thresholds = [0.40, 0.42, 0.45, 0.48, 0.50]
    if max_speakers >= 3:
        cluster_thresholds = [0.38, 0.40, 0.42, 0.45, 0.48]
    seg_thresholds = [0.36, 0.38, 0.40, 0.42]
    min_offs = [0.03, 0.04, 0.06]
    return seg_thresholds, cluster_thresholds, min_offs, _min_cluster_size(max_speakers)


def _standard_sweep_params(
    max_speakers: int,
    base_seg: float,
    audio_duration_s: float,
) -> tuple[list[float], list[float], list[float], int]:
    cluster_thresholds = [0.48, 0.52, 0.56, 0.60]
    if max_speakers >= 3:
        cluster_thresholds = [0.45, 0.50, 0.55, 0.58]
    seg_thresholds = [max(0.35, base_seg - 0.02), base_seg, min(0.50, base_seg + 0.02)]
    min_offs = [0.06, 0.08, 0.10]
    min_cluster = _min_cluster_size(max_speakers)
    if 0 < audio_duration_s < 90.0:
        min_cluster = max(2, min_cluster - 1)
    return seg_thresholds, cluster_thresholds, min_offs, min_cluster


def _build_param_grid(
    seg_thresholds: list[float],
    cluster_thresholds: list[float],
    min_offs: list[float],
    min_cluster: int,
) -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []
    for seg_t in seg_thresholds:
        for clust_t in cluster_thresholds:
            for min_off in min_offs:
                label = f"seg={seg_t:.2f}_clust={clust_t:.2f}_off={min_off:.2f}"
                params = {
                    "segmentation": {"threshold": seg_t, "min_duration_off": min_off},
                    "clustering": {"threshold": clust_t, "min_cluster_size": min_cluster},
                }
                configs.append((label, params))
    return configs


def _params_key(params: dict) -> tuple:
    seg = params.get("segmentation") or {}
    clust = params.get("clustering") or {}
    return (
        round(float(seg.get("threshold", -1)), 3),
        round(float(seg.get("min_duration_off", -1)), 3),
        round(float(clust.get("threshold", -1)), 3),
        int(clust.get("min_cluster_size", -1)),
    )


def _prioritize_configs(
    curated: list[tuple[str, dict]],
    grid: list[tuple[str, dict]],
    limit: int,
) -> list[tuple[str, dict]]:
    if limit <= 0:
        return []
    seen: set[tuple] = set()
    picked: list[tuple[str, dict]] = []
    for label, params in curated + grid:
        key = _params_key(params)
        if key in seen:
            continue
        seen.add(key)
        picked.append((label, params))
        if len(picked) >= limit:
            break
    if len(picked) < limit and len(grid) > len(picked):
        step = max(1, len(grid) // max(1, limit - len(picked)))
        for label, params in grid[::step]:
            key = _params_key(params)
            if key in seen:
                continue
            seen.add(key)
            picked.append((label, params))
            if len(picked) >= limit:
                break
    logger.info(
        "Diarization multi-sample: evaluating %d config(s) (limit=%d).",
        len(picked),
        limit,
    )
    return picked


def sample_param_sets(max_speakers: int, audio_duration_s: float) -> list[tuple[str, dict]]:
    """Return labelled instantiate() configs to try (curated first, then grid expansion)."""
    base_seg = _base_seg_threshold(audio_duration_s)
    curated = _curated_accuracy_configs(max_speakers) if _accuracy_mode() else []
    if _accuracy_mode():
        seg_thresholds, cluster_thresholds, min_offs, min_cluster = _accuracy_sweep_params(max_speakers)
    else:
        seg_thresholds, cluster_thresholds, min_offs, min_cluster = _standard_sweep_params(
            max_speakers, base_seg, audio_duration_s,
        )
    grid = _build_param_grid(seg_thresholds, cluster_thresholds, min_offs, min_cluster)
    if multi_sample_full_grid():
        logger.info("Diarization multi-sample: using full grid (%d configs).", len(grid))
        return grid

    limit = effective_pass_limit(audio_duration_s, max_speakers)
    if limit <= 0:
        if multi_sample_enabled():
            logger.info(
                "Diarization multi-sample: grid sweep skipped "
                "(DIARIZATION_MULTI_SAMPLE_PASSES=%d); adaptive base only.",
                multi_sample_pass_limit(),
            )
        return []
    return _prioritize_configs(curated, grid, limit)


def _speaker_durations(segments: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for seg in segments:
        spk = seg["speaker"]
        totals[spk] = totals.get(spk, 0.0) + max(0.0, seg["end"] - seg["start"])
    return totals


def _speaker_fit_score(n_speakers: int, max_speakers: int) -> float:
    if max_speakers > 0:
        if n_speakers > max_speakers:
            return max(0.0, 0.35 - (n_speakers - max_speakers) * 0.2)
        if n_speakers == max_speakers:
            return 1.0
        if n_speakers >= 2:
            return 0.70 + 0.30 * (n_speakers / max_speakers)
        return 0.35
    if n_speakers <= 1:
        return 0.5
    return min(1.0, n_speakers / 4.0)


def _coverage_gap_penalty(segments: list[dict], audio_duration_s: float) -> float:
    if not segments or audio_duration_s <= 0:
        return 0.0
    ordered = sorted(segments, key=lambda s: s["start"])
    gap_s = 0.0
    if ordered[0]["start"] > 1.0:
        gap_s += ordered[0]["start"]
    for left, right in zip(ordered, ordered[1:]):
        gap = max(0.0, right["start"] - left["end"])
        if gap > 0.4:
            gap_s += gap
    tail_gap = max(0.0, audio_duration_s - ordered[-1]["end"])
    if tail_gap > 2.0:
        gap_s += tail_gap
    return min(0.20, (gap_s / audio_duration_s) * 0.6)


def _short_turn_penalty(segments: list[dict]) -> float:
    penalty = 0.0
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur < 0.20:
            penalty += 0.025
        elif dur < 0.45:
            penalty += 0.010
    return min(0.18, penalty)


def _dominant_speaker_penalty(segments: list[dict], max_speakers: int) -> float:
    if max_speakers < 2:
        return 0.0
    totals = _speaker_durations(segments)
    if len(totals) < 2:
        return 0.0
    total = sum(totals.values())
    if total <= 0:
        return 0.0
    max_share = max(totals.values()) / total
    if max_share > 0.90:
        return min(0.25, (max_share - 0.90) * 2.5)
    return 0.0


def _switch_rate_penalty(segments: list[dict], audio_duration_s: float) -> float:
    if len(segments) < 2 or audio_duration_s <= 0:
        return 0.0
    switches = sum(
        1 for a, b in zip(segments, segments[1:]) if a["speaker"] != b["speaker"]
    )
    rate = switches / max(audio_duration_s / 60.0, 0.1)
    if rate > 35:
        return min(0.15, (rate - 35) * 0.004)
    return 0.0


def _accuracy_penalties(
    segments: list[dict],
    audio_duration_s: float,
    max_speakers: int,
) -> tuple[float, float]:
    frag_penalty = max(0, len(segments) - int(audio_duration_s * 1.8)) * 0.0015
    if not _accuracy_mode() or max_speakers < 2:
        return 0.0, frag_penalty
    mega_turn_penalty = 0.0
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur > 55.0:
            mega_turn_penalty += (dur - 55.0) * 0.012
    if audio_duration_s >= 180.0:
        expected_turns = max(2, int(audio_duration_s / 40.0))
        if len(segments) < expected_turns // 2:
            mega_turn_penalty += 0.12
    return mega_turn_penalty, frag_penalty


def score_segments(
    segments: list[dict],
    audio_duration_s: float,
    max_speakers: int,
) -> float:
    """Higher is better — rewards coverage, plausible speaker count, stable turns."""
    if not segments or audio_duration_s <= 0:
        return -1.0

    covered = sum(max(0.0, seg["end"] - seg["start"]) for seg in segments)
    coverage = min(1.0, covered / audio_duration_s)
    n_speakers = len({seg["speaker"] for seg in segments})

    if max_speakers >= 2 and n_speakers == 1:
        return -1.0

    speaker_fit = _speaker_fit_score(n_speakers, max_speakers)
    mega_turn_penalty, frag_penalty = _accuracy_penalties(segments, audio_duration_s, max_speakers)
    gap_penalty = _coverage_gap_penalty(segments, audio_duration_s)
    short_penalty = _short_turn_penalty(segments)
    dominant_penalty = _dominant_speaker_penalty(segments, max_speakers)
    switch_penalty = _switch_rate_penalty(segments, audio_duration_s)

    if _accuracy_mode():
        return (
            (coverage * 0.36)
            + (speaker_fit * 0.42)
            - mega_turn_penalty
            - frag_penalty
            - gap_penalty
            - short_penalty
            - dominant_penalty
            - switch_penalty
        )

    avg_turn = covered / max(len(segments), 1)
    turn_quality = min(1.0, avg_turn / 2.5)
    return (
        (coverage * 0.42)
        + (speaker_fit * 0.33)
        + (turn_quality * 0.18)
        - frag_penalty
        - gap_penalty
        - short_penalty
    )


def _should_early_stop(
    tries: int,
    best_score: float,
    last_score: float,
    stall_count: int,
) -> tuple[bool, int]:
    if not multi_sample_early_stop_enabled():
        return False, stall_count
    min_tries = _env_int("DIARIZATION_MULTI_SAMPLE_EARLY_MIN_TRIES", 3)
    target = _env_float("DIARIZATION_MULTI_SAMPLE_EARLY_STOP_SCORE", 0.88)
    if tries >= min_tries and best_score >= target:
        logger.info(
            "Diarization multi-sample early stop: score %.3f >= %.3f after %d tries.",
            best_score,
            target,
            tries,
        )
        return True, stall_count
    stall_delta = _env_float("DIARIZATION_MULTI_SAMPLE_EARLY_STALL_DELTA", 0.012)
    if tries >= min_tries and last_score < best_score - stall_delta:
        stall_count += 1
        if stall_count >= _env_int("DIARIZATION_MULTI_SAMPLE_EARLY_STALL_COUNT", 2):
            logger.info(
                "Diarization multi-sample early stop: no improvement after %d tries "
                "(best=%.3f).",
                tries,
                best_score,
            )
            return True, stall_count
        return False, stall_count
    return False, 0


def run_multi_sample_diarization(
    instantiate_fn: Callable[[dict, str], None],
    run_fn: Callable[[], list[dict]],
    audio_duration_s: float,
    max_speakers: int,
    base_params: dict | None,
) -> tuple[list[dict], str, float]:
    """Try several hyperparameter sets; return the highest-scoring segment list."""
    best_segments, best_label, best_score, _best_params = _run_multi_sample_core(
        instantiate_fn, run_fn, audio_duration_s, max_speakers, base_params,
    )
    return best_segments, best_label, best_score


def select_best_diarization_params(
    instantiate_fn: Callable[[dict, str], None],
    run_fn: Callable[[], list[dict]],
    audio_duration_s: float,
    max_speakers: int,
    base_params: dict | None,
) -> tuple[dict | None, str, float]:
    """Pick the best instantiate() config without returning segment output."""
    _segments, best_label, best_score, best_params = _run_multi_sample_core(
        instantiate_fn, run_fn, audio_duration_s, max_speakers, base_params,
    )
    return best_params, best_label, best_score


def _run_multi_sample_core(
    instantiate_fn: Callable[[dict, str], None],
    run_fn: Callable[[], list[dict]],
    audio_duration_s: float,
    max_speakers: int,
    base_params: dict | None,
) -> tuple[list[dict], str, float, dict | None]:
    """Try several hyperparameter sets; return the highest-scoring segment list."""
    candidates = sample_param_sets(max_speakers, audio_duration_s)
    if base_params:
        candidates.insert(0, ("adaptive_base", copy.deepcopy(base_params)))
    max_tries = multi_sample_max_tries()
    if candidates:
        candidates = candidates[:max_tries]
        logger.info(
            "Diarization multi-sample: capped to %d pass(es) per SR (global budget=%d).",
            len(candidates),
            multi_sample_max_total_tries(),
        )

    best_segments: list[dict] = []
    best_label = "none"
    best_score = -1.0
    best_params: dict | None = None
    stall_count = 0
    tried = 0

    for index, (label, params) in enumerate(candidates):
        tried = index + 1
        instantiate_fn(params, f"multi-sample:{label}")
        segments = run_fn()
        score = score_segments(segments, audio_duration_s, max_speakers)
        n_spk = len({s["speaker"] for s in segments})
        logger.info(
            "Diarization sample %s: segments=%d speakers=%d score=%.3f",
            label,
            len(segments),
            n_spk,
            score,
        )
        if score > best_score:
            best_score = score
            best_segments = segments
            best_label = label
            best_params = copy.deepcopy(params)
        stop, stall_count = _should_early_stop(
            index + 1,
            best_score,
            score,
            stall_count,
        )
        if stop:
            break

    logger.info(
        "Diarization multi-sample winner: %s (score=%.3f, segments=%d, tried=%d)",
        best_label,
        best_score,
        len(best_segments),
        tried,
    )
    return best_segments, best_label, best_score, best_params
