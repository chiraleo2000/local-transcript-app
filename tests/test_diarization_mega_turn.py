"""Tests for long-turn diarization refinement helpers."""

from engines.diarization import (
    _max_segment_duration,
    _mega_turn_retry_params,
    _replace_interval_segments,
)


def test_max_segment_duration():
    segments = [
        {"start": 0.0, "end": 30.0, "speaker": "A"},
        {"start": 30.0, "end": 200.0, "speaker": "B"},
    ]
    assert _max_segment_duration(segments) == 170.0


def test_mega_turn_retry_params_loosen_clustering():
    base = {
        "segmentation": {"threshold": 0.40},
        "clustering": {"threshold": 0.48, "min_cluster_size": 3},
    }
    retry = _mega_turn_retry_params(base)
    assert retry["clustering"]["threshold"] < base["clustering"]["threshold"]
    assert retry["segmentation"]["threshold"] < base["segmentation"]["threshold"]
    assert retry["clustering"]["min_cluster_size"] == 2


def test_replace_interval_segments_swaps_span():
    segments = [
        {"start": 0.0, "end": 100.0, "speaker": "A"},
        {"start": 100.0, "end": 400.0, "speaker": "A"},
        {"start": 400.0, "end": 500.0, "speaker": "B"},
    ]
    replacements = [
        {"start": 120.0, "end": 200.0, "speaker": "X"},
        {"start": 200.0, "end": 350.0, "speaker": "Y"},
    ]
    merged = _replace_interval_segments(
        segments,
        {"start": 100.0, "end": 400.0, "speaker": "A"},
        replacements,
    )
    assert len(merged) == 4
    assert merged[1]["speaker"] == "X"
    assert merged[2]["speaker"] == "Y"
