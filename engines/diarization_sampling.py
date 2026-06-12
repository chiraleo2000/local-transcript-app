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


def multi_sample_sweep_enabled() -> bool:
    """True when hyperparameter sweep should run (not just adaptive base)."""
    if not multi_sample_enabled():
        return False
    if multi_sample_full_grid():
        return True
    return multi_sample_pass_limit() > 0


def _base_seg_threshold(audio_duration_s: float) -> float:
    if 0 < audio_duration_s < 30.0:
        return 0.40
    if audio_duration_s >= 600.0:
        return 0.44
    return 0.42


def _accuracy_sweep_params(max_speakers: int) -> tuple[list[float], list[float], list[float], int]:
    cluster_thresholds = [0.42, 0.45, 0.48, 0.50]
    if max_speakers >= 3:
        cluster_thresholds = [0.40, 0.42, 0.45, 0.48]
    seg_thresholds = [0.36, 0.38, 0.40, 0.42]
    min_offs = [0.03, 0.04, 0.06, 0.08]
    min_cluster = 2 if max_speakers >= 3 else 3
    return seg_thresholds, cluster_thresholds, min_offs, min_cluster


def _standard_sweep_params(
    max_speakers: int,
    base_seg: float,
    audio_duration_s: float,
) -> tuple[list[float], list[float], list[float], int]:
    cluster_thresholds = [0.48, 0.52, 0.56, 0.60]
    if max_speakers >= 3:
        cluster_thresholds = [0.45, 0.50, 0.55, 0.58]
    seg_thresholds = [max(0.35, base_seg - 0.02), base_seg, min(0.50, base_seg + 0.02)]
    min_offs = [0.08]
    min_cluster = 3 if max_speakers >= 3 else 4
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


def _limit_param_grid(configs: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    if not multi_sample_sweep_enabled():
        if configs:
            logger.info(
                "Diarization multi-sample: grid sweep skipped "
                "(DIARIZATION_MULTI_SAMPLE_PASSES=%d); adaptive base only.",
                multi_sample_pass_limit(),
            )
        return []
    if multi_sample_full_grid():
        logger.info("Diarization multi-sample: using full grid (%d configs).", len(configs))
        return configs
    limit = multi_sample_pass_limit()
    if len(configs) <= limit:
        logger.info("Diarization multi-sample: using full grid (%d configs).", len(configs))
        return configs
    step = max(1, len(configs) // limit)
    picked = [configs[i] for i in range(0, len(configs), step)][:limit]
    logger.info("Diarization multi-sample: using %d of %d configs.", len(picked), len(configs))
    return picked


def sample_param_sets(max_speakers: int, audio_duration_s: float) -> list[tuple[str, dict]]:
    """Return labelled instantiate() configs to try (segmentation + clustering sweeps)."""
    base_seg = _base_seg_threshold(audio_duration_s)
    if _accuracy_mode():
        seg_thresholds, cluster_thresholds, min_offs, min_cluster = _accuracy_sweep_params(max_speakers)
    else:
        seg_thresholds, cluster_thresholds, min_offs, min_cluster = _standard_sweep_params(
            max_speakers, base_seg, audio_duration_s,
        )
    configs = _build_param_grid(seg_thresholds, cluster_thresholds, min_offs, min_cluster)
    return _limit_param_grid(configs)


def _speaker_fit_score(n_speakers: int, max_speakers: int) -> float:
    if max_speakers > 0:
        if n_speakers > max_speakers:
            return max(0.0, 0.35 - (n_speakers - max_speakers) * 0.2)
        if n_speakers == max_speakers:
            return 1.0
        return 0.65 + 0.35 * (n_speakers / max_speakers)
    if n_speakers <= 1:
        return 0.5
    return min(1.0, n_speakers / 4.0)


def _accuracy_penalties(
    segments: list[dict],
    audio_duration_s: float,
    max_speakers: int,
) -> tuple[float, float]:
    frag_penalty = max(0, len(segments) - int(audio_duration_s * 1.5)) * 0.002
    if not _accuracy_mode() or max_speakers < 2:
        return 0.0, frag_penalty
    mega_turn_penalty = 0.0
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur > 60.0:
            mega_turn_penalty += (dur - 60.0) * 0.015
    if audio_duration_s >= 180.0:
        expected_turns = max(2, int(audio_duration_s / 45.0))
        if len(segments) < expected_turns // 2:
            mega_turn_penalty += 0.15
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

    if _accuracy_mode():
        return (coverage * 0.40) + (speaker_fit * 0.45) - mega_turn_penalty - frag_penalty

    avg_turn = covered / max(len(segments), 1)
    turn_quality = min(1.0, avg_turn / 2.5)
    return (coverage * 0.45) + (speaker_fit * 0.35) + (turn_quality * 0.20) - frag_penalty


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

    best_segments: list[dict] = []
    best_label = "none"
    best_score = -1.0
    best_params: dict | None = None

    for label, params in candidates:
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

    logger.info(
        "Diarization multi-sample winner: %s (score=%.3f, segments=%d)",
        best_label,
        best_score,
        len(best_segments),
    )
    return best_segments, best_label, best_score, best_params
