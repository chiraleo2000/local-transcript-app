#!/usr/bin/env python3
"""Quick diarization smoke test on first 3 minutes of SM kmutt file."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
SM = REPO / "tests" / "SM-พี่วีระ x รองยุ่น ศรชล x ดร เบิร์ด kmutt.m4a"
HEAD = Path(tempfile.gettempdir()) / "sm_head.wav"


def main() -> int:
    if not SM.is_file():
        print(f"Missing audio: {SM}", file=sys.stderr)
        return 1
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-t", "180", "-i", str(SM), "-ac", "1", "-ar", "16000", str(HEAD),
        ],
        check=True,
    )
    from _bootstrap import bootstrap

    bootstrap()
    from engines.diarization import diarize

    # No min_speakers_hint: exercise the real user path where "Max Speakers" is a
    # pure upper bound and collapse is recovered by the built-in retries.
    segs = diarize(
        str(HEAD),
        max_speakers=4,
        audio_duration_s=180.0,
    )
    first_min = sorted({s["speaker"] for s in segs if s["start"] < 60})
    total = sorted({s["speaker"] for s in segs})
    print(f"first_minute_speakers={len(first_min)} {first_min}")
    print(f"total_speakers={len(total)} {total}")
    ok = len(first_min) >= 2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
