"""Tests for diarization timestamp output when ASR timestamps are missing."""

from __future__ import annotations

import re

from engines.diarization import assign_speakers

SPEAKER_TS_RE = re.compile(
    r"^\[\d{2}:\d{2}:\d{2} → \d{2}:\d{2}:\d{2}\] \[SPEAKER_\d+\]:"
)


def test_assign_speakers_all_ts_none_uses_diarization_bounds():
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
    for line in text.splitlines():
        if "[SPEAKER_" in line:
            assert SPEAKER_TS_RE.match(line.strip()), line
