"""Tests for transcript repetition cleanup."""

from engines.text_cleanup import clean_transcript_lines, clean_transcript_text


def test_collapse_spaced_phrase():
    text = "ที่มีความสำคัญ ที่มีความสำคัญ ที่มีความสำคัญ hello"
    assert clean_transcript_text(text) == "ที่มีความสำคัญ hello"


def test_collapse_compact_thai_tail():
    text = "สวัสดี" + "ตอนนี้" * 8
    cleaned = clean_transcript_text(text)
    assert cleaned.count("ตอนนี้") == 1
    assert cleaned.startswith("สวัสดี")


def test_collapse_char_garble():
    text = "helloสสสสสสสสสสworld"
    cleaned = clean_transcript_text(text)
    assert "สสสส" not in cleaned


def test_clean_transcript_lines_preserves_speaker():
    text = "[SPEAKER_01]: word " + "repeat " * 5
    cleaned = clean_transcript_lines(text)
    assert cleaned.startswith("[SPEAKER_01]: word repeat")
