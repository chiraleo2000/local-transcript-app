"""Turn-centric speaker assignment — prevents 30-minute single-speaker blocks."""

from __future__ import annotations

import re

from engines.diarization import assign_speakers

SPEAKER_LINE_RE = re.compile(
    r"^\[\d{2}:\d{2}:\d{2} → \d{2}:\d{2}:\d{2}\] \[SPEAKER_\d+\]:"
)


def _speaker_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if "[SPEAKER_" in ln]


def test_assign_speakers_splits_long_asr_chunk_across_turns():
    """One 30-minute ASR chunk must not become one speaker line."""
    words = [f"w{i}" for i in range(400)]
    result = {
        "chunks": [{"text": " ".join(words), "timestamp": (0.0, 1800.0)}],
    }
    segments = []
    speakers = ["SPEAKER_01", "SPEAKER_02", "SPEAKER_03"]
    for minute in range(30):
        segments.append({
            "start": float(minute * 60),
            "end": float((minute + 1) * 60),
            "speaker": speakers[minute % 3],
        })
    text = assign_speakers(result, segments, max_speakers=3)
    lines = _speaker_lines(text)
    assert len(lines) >= 10, f"expected many turns, got {len(lines)} lines"
    labels = {ln.split("[", 3)[2].split("]")[0] for ln in lines}
    assert len(labels) >= 2


def test_assign_speakers_merges_short_same_speaker_turns():
    result = {
        "chunks": [
            {"text": "part one", "timestamp": (1.0, 4.0)},
            {"text": "part two", "timestamp": (5.0, 8.0)},
        ],
    }
    segments = [{"start": 0.0, "end": 12.0, "speaker": "SPEAKER_01"}]
    text = assign_speakers(result, segments, max_speakers=1)
    lines = _speaker_lines(text)
    assert len(lines) == 1
    assert "part one" in lines[0]
    assert "two" in lines[0]


def test_assign_speakers_all_ts_none_still_valid():
    result = {
        "text": "hello world",
        "chunks": [
            {"text": "hello", "timestamp": (None, None)},
            {"text": "world", "timestamp": (None, None)},
        ],
    }
    segments = [
        {"start": 12.0, "end": 34.0, "speaker": "SPEAKER_01"},
        {"start": 35.0, "end": 62.0, "speaker": "SPEAKER_02"},
    ]
    text = assign_speakers(result, segments)
    for line in _speaker_lines(text):
        assert SPEAKER_LINE_RE.match(line.strip()), line
