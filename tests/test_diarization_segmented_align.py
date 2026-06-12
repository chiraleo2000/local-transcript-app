"""Tests for cross-chunk speaker alignment in segmented diarization."""

from engines.diarization import _align_segmented_speakers


def _segment_at(aligned: list[dict], start: float) -> dict:
    matches = [seg for seg in aligned if abs(seg["start"] - start) < 1e-9]
    assert len(matches) == 1, f"expected one segment at {start}, got {len(matches)}"
    return matches[0]


def test_align_segmented_speakers_links_overlap_labels():
    segments = [
        {"start": 250.0, "end": 270.0, "speaker": "SPEAKER_00_S0"},
        {"start": 275.0, "end": 295.0, "speaker": "SPEAKER_01_S0"},
        {"start": 240.0, "end": 258.0, "speaker": "SPEAKER_00_S1"},
        {"start": 285.0, "end": 300.0, "speaker": "SPEAKER_01_S1"},
    ]
    aligned = _align_segmented_speakers(segments, min_overlap_s=0.5)
    speakers = {seg["speaker"] for seg in aligned}
    assert len(speakers) == 2
    assert _segment_at(aligned, 240.0)["speaker"] == _segment_at(aligned, 250.0)["speaker"]
    assert _segment_at(aligned, 285.0)["speaker"] == _segment_at(aligned, 275.0)["speaker"]


def test_align_segmented_speakers_keeps_distinct_non_overlapping():
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00_S0"},
        {"start": 300.0, "end": 305.0, "speaker": "SPEAKER_00_S1"},
    ]
    aligned = _align_segmented_speakers(segments, min_overlap_s=0.5)
    assert len({seg["speaker"] for seg in aligned}) == 2
