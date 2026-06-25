"""Parse golden reference transcript into diarization segments."""

from __future__ import annotations

import re
from pathlib import Path

_LINE_RE = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}:\d{2})\s*→\s*(?P<end>\d{2}:\d{2}:\d{2})\]\s*"
    r"(?:\[(?P<speaker>SPEAKER_\d+)\]:)?",
    re.IGNORECASE,
)


def _ts_to_seconds(ts: str) -> float:
    hours, minutes, seconds = (int(part) for part in ts.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def load_reference_segments(path: Path) -> list[dict]:
    segments: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _LINE_RE.match(line.strip())
        if not match or not match.group("speaker"):
            continue
        segments.append({
            "start": _ts_to_seconds(match.group("start")),
            "end": _ts_to_seconds(match.group("end")),
            "speaker": match.group("speaker").upper(),
        })
    return segments
