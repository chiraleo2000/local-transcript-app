"""Thai transcript cleanup and Whisper language lock."""

from __future__ import annotations

import importlib

text_cleanup = importlib.import_module("engines.text_cleanup")
whisper_utils = importlib.import_module("engines.whisper_utils")


def test_thai_question_and_loanword(monkeypatch):
    monkeypatch.setenv("ASR_THAI_LINGUISTIC_CLEANUP", "false")
    cleaned = text_cleanup.clean_transcript_text("โอเครไหม")
    assert "โอเค" in cleaned
    assert cleaned.endswith("?")


def test_repeat_collapse_without_spaces(monkeypatch):
    monkeypatch.setenv("ASR_THAI_LINGUISTIC_CLEANUP", "false")
    cleaned = text_cleanup.clean_transcript_text("สวัสดีสวัสดีสวัสดี")
    assert cleaned.count("สวัสดี") == 1


def test_whisper_forces_thai_code_and_beams(monkeypatch):
    monkeypatch.setenv("ASR_GPU_PROFILE", "off")
    monkeypatch.delenv("ASR_THAI_TEMPERATURE", raising=False)
    monkeypatch.setenv("ASR_THAI_NUM_BEAMS", "5")
    monkeypatch.setenv("ASR_THAI_ADAPTIVE_PERFORMANCE", "false")
    monkeypatch.setenv("ASR_NUM_BEAMS", "2")
    kwargs = whisper_utils.whisper_generate_kwargs("Thai")
    assert kwargs["language"] == "th"
    assert kwargs["num_beams"] == 5
    assert kwargs["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def test_short_same_speaker_turns_merge(monkeypatch):
    monkeypatch.setenv("ASR_MERGE_SHORT_TURNS", "true")
    from engines.diarization import _coalesce_short_same_speaker

    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.05, "end": 1.2, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01"},
    ]
    merged = _coalesce_short_same_speaker(turns, 0.4)
    assert len(merged) == 2
    assert merged[0]["end"] == 1.2
