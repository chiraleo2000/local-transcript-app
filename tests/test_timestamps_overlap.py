"""Unit tests for long-form window overlap merge behaviour."""

from engines.timestamps import merge_window_results


def _chunk(text: str, start: float, end: float) -> dict:
    return {"text": text, "timestamp": (start, end)}


def test_merge_window_results_concatenates_non_overlapping_windows():
    window_a = {
        "text": "hello",
        "chunks": [_chunk("hello", 0.0, 2.0)],
    }
    window_b = {
        "text": "world",
        "chunks": [_chunk("world", 360.0, 362.0)],
    }
    merged = merge_window_results([window_a, window_b])
    assert merged["text"] == "hello\nworld"
    assert len(merged["chunks"]) == 2


def test_merge_window_results_dedupes_overlap_duplicate_text():
    overlap = _chunk("สวัสดีครับ", 330.0, 360.0)
    repeat = _chunk("สวัสดีครับ", 350.0, 360.0)
    window_a = {"text": "สวัสดีครับ", "chunks": [overlap]}
    window_b = {"text": "สวัสดีครับ", "chunks": [repeat]}
    merged = merge_window_results([window_a, window_b])
    assert merged["text"] == "สวัสดีครับ"
    assert len(merged["chunks"]) == 1


def test_merge_window_results_uses_fallback_text_when_chunks_empty():
    merged = merge_window_results([{"text": "only text", "chunks": []}])
    assert merged["text"] == "only text"
    assert merged["chunks"] == []
