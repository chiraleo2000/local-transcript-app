"""Tests for long-audio segmented diarization scaling."""

from engines.diarization import (
    _count_diarization_segments,
    _effective_diarization_segment_s,
    _refine_after_segmented_enabled,
)


def test_effective_segment_s_scales_for_two_hour_audio(monkeypatch):
    monkeypatch.setenv("DIARIZATION_SEGMENT_S", "300")
    monkeypatch.setenv("DIARIZATION_ADAPTIVE_SEGMENT_S", "true")
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "false")
    assert _effective_diarization_segment_s(8359.0) == 600


def test_count_segments_fewer_with_larger_chunks():
    short_chunks = _count_diarization_segments(8359.0, segment_s=300, overlap_s=60)
    long_chunks = _count_diarization_segments(8359.0, segment_s=600, overlap_s=60)
    assert short_chunks == 35
    assert long_chunks == 16
    assert long_chunks < short_chunks


def test_tail_does_not_crawl_one_second_steps():
    """Regression: tiny tail must not advance 1s per pass (was ~80 extra runs)."""
    assert _count_diarization_segments(8359.0, segment_s=300, overlap_s=60) < 40


def test_refine_after_segmented_enabled_in_accuracy_mode(monkeypatch):
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
    monkeypatch.delenv("DIARIZATION_REFINE_AFTER_SEGMENTED", raising=False)
    from engines.diarization import _refine_after_segmented_enabled

    assert _refine_after_segmented_enabled()


def test_effective_segment_s_accuracy_caps_chunk_size(monkeypatch):
    monkeypatch.setenv("DIARIZATION_SEGMENT_S", "600")
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
    monkeypatch.setenv("DIARIZATION_ADAPTIVE_SEGMENT_S", "true")
    from engines.diarization import _effective_diarization_segment_s

    assert _effective_diarization_segment_s(8359.0) == 360


def test_refine_after_segmented_disabled_by_default(monkeypatch):
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "false")
    monkeypatch.delenv("DIARIZATION_REFINE_AFTER_SEGMENTED", raising=False)
    from engines.diarization import _refine_after_segmented_enabled

    assert not _refine_after_segmented_enabled()
