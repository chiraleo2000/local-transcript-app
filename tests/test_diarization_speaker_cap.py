"""Speaker cap and transcript merge helpers."""

from engines.diarization import (
    _enforce_max_speakers,
    _merge_transcript_lines,
    assign_speakers,
)


def test_enforce_max_speakers_caps_labels():
    segments = [
        {"start": 0.0, "end": 10.0, "speaker": "A"},
        {"start": 10.0, "end": 20.0, "speaker": "B"},
        {"start": 20.0, "end": 25.0, "speaker": "C"},
        {"start": 25.0, "end": 60.0, "speaker": "D"},
    ]
    capped = _enforce_max_speakers(segments, 2)
    speakers = {s["speaker"] for s in capped}
    assert len(speakers) <= 2
    assert sum(s["end"] - s["start"] for s in capped) > 50.0


def test_merge_transcript_lines_same_speaker():
    lines = [
        "[00:00:01 → 00:00:05] [SPEAKER_01]: Hello",
        "[00:00:06 → 00:00:10] [SPEAKER_01]: world",
        "[00:00:20 → 00:00:25] [SPEAKER_02]: Other",
    ]
    merged = _merge_transcript_lines(lines, max_gap_s=2.0)
    assert len(merged) == 2
    assert "Hello world" in merged[0]
    assert "SPEAKER_02" in merged[1]


def test_assign_speakers_merges_consecutive_same_speaker():
    result = {
        "chunks": [
            {"text": "part one", "timestamp": (1.0, 4.0)},
            {"text": "part two", "timestamp": (5.0, 8.0)},
        ],
    }
    segments = [{"start": 0.0, "end": 12.0, "speaker": "SPEAKER_01"}]
    text = assign_speakers(result, segments, max_speakers=1)
    speaker_lines = [ln for ln in text.splitlines() if "SPEAKER_" in ln]
    assert len(speaker_lines) == 1
    assert "part one" in speaker_lines[0]
    assert "part" in speaker_lines[0].split(":", 1)[-1]
