"""Regression: fragmented diar turns must not repeat cumulative ASR text."""

from __future__ import annotations

from engines.diarization import assign_speakers, _novel_turn_text
from engines.timestamps import _dedupe_overlapped_chunks, _strip_prefix_overlap


def test_novel_turn_text_strips_cumulative_prefix():
    prev = "ดีไให้คนไทยใช้อย่างนี้มันไม่ต้องซื้อ"
    new = prev + " แต่ว่าพูดแค่นี้ก็ผิดแล้ว"
    assert _novel_turn_text(prev, new) == "แต่ว่าพูดแค่นี้ก็ผิดแล้ว"
    assert _novel_turn_text(prev, prev) == ""


def test_assign_speakers_no_cumulative_duplicates_across_fragmented_turns():
    """Many short same-speaker turns on one long chunk should not repeat full text."""
    long_text = (
        "ดีไให้คนไทยใช้อย่างนี้มันไม่ต้องซื้อไม่ใช้อะไรก็ได้ "
        "แต่ว่าพูดแค่นี้ก็ผิดแล้วไงถ้ามันดีไฟล์ได้พี่เรื่องนี้จบ "
        "ต้องชื่นชมแต่มันก็ไม่เป็นที่อุตสาหรณ์ "
        "มันมีงบดีดีไม่ได้ทำเพื่อภารกิจดีนำเพื่อภารกิจประเทศ"
    )
    result = {
        "chunks": [{"text": long_text, "timestamp": (2000.0, 2090.0)}],
    }
    segments = []
    t = 2002.0
    while t < 2088.0:
        segments.append({
            "start": t,
            "end": t + 8.0,
            "speaker": "SPEAKER_02",
        })
        t += 10.0

    text = assign_speakers(result, segments, max_speakers=2)
    lines = [ln for ln in text.splitlines() if "[SPEAKER_" in ln]
    bodies = [ln.split("]:", 1)[-1].strip() for ln in lines]
    assert len(lines) <= 4, f"too many fragmented lines: {len(lines)}"
    joined = " ".join(bodies)
    assert joined.count("ดีไให้คนไทยใช้อย่างนี้") <= 1
    assert "มันมีงบดีดี" in joined


def test_dedupe_overlapped_chunks_trims_cumulative_prefix():
    chunks = [
        {"text": "hello world from window one", "timestamp": (0.0, 30.0)},
        {"text": "hello world from window one and window two", "timestamp": (25.0, 55.0)},
    ]
    kept = _dedupe_overlapped_chunks(chunks)
    texts = [c["text"] for c in kept]
    assert "hello world from window one" in texts[0]
    assert any("window two" in t for t in texts)
    assert not any(t.count("hello world from window one") > 1 for t in texts)


def test_strip_prefix_overlap_identical():
    assert _strip_prefix_overlap("same text", "same text") == ""
