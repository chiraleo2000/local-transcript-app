"""Gap-fill and Thai unit slicing for turn-centric speaker assignment."""

from __future__ import annotations

from engines.diarization import assign_speakers


def test_assign_speakers_fills_gap_between_diarization_turns():
    """Speech between diar turns must still appear in the transcript."""
    result = {
        "chunks": [
            {"text": "alpha beta gamma delta", "timestamp": (0.0, 20.0)},
        ],
    }
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01"},
        {"start": 15.0, "end": 20.0, "speaker": "SPEAKER_02"},
    ]
    text = assign_speakers(result, segments, max_speakers=2, audio_duration_s=20.0)
    assert "alpha" in text
    assert "gamma" in text
    assert "delta" in text


def test_assign_speakers_uses_full_audio_duration_not_only_diar_end():
    """ASR tail after the last diar segment must not be dropped."""
    result = {
        "chunks": [
            {"text": "head middle tail", "timestamp": (0.0, 30.0)},
        ],
    }
    segments = [
        {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_01"},
        {"start": 12.0, "end": 18.0, "speaker": "SPEAKER_02"},
    ]
    text = assign_speakers(result, segments, max_speakers=2, audio_duration_s=30.0)
    assert "tail" in text


def test_assign_speakers_slices_unsegmented_thai_text():
    thai = "ได้ไหมครับอีกหนึ่งชั่วโมงฮึฝากด้วยครับ"
    result = {
        "chunks": [{"text": thai, "timestamp": (0.0, 10.0)}],
    }
    segments = [
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_01"},
        {"start": 6.0, "end": 10.0, "speaker": "SPEAKER_02"},
    ]
    text = assign_speakers(result, segments, max_speakers=2, audio_duration_s=10.0)
    assert "ได้" in text
    assert "ฝาก" in text
