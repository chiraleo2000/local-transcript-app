#!/usr/bin/env python3
"""Run production pipeline on SM file head and verify multi-speaker output."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SM = REPO / "tests" / "SM-พี่วีระ x รองยุ่น ศรชล x ดร เบิร์ด kmutt.m4a"


def _probe_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float((proc.stdout or "").strip())
    except ValueError:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=600.0, help="duration to test")
    parser.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    parser.add_argument("--max-speakers", type=int, default=4)
    args = parser.parse_args()

    if not SM.is_file():
        print(f"Missing audio: {SM}", file=sys.stderr)
        return 1

    clip = Path(tempfile.gettempdir()) / "sm_pipeline_clip.wav"
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(args.start), "-t", str(args.seconds), "-i", str(SM),
        "-ac", "1", "-ar", "16000", str(clip),
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    os.environ.setdefault("DIARIZATION_DEVICE", "cuda")
    os.environ.setdefault("DIARIZATION_ACCURACY_MODE", "true")
    os.environ.setdefault("DIARIZATION_REFINE_AFTER_SEGMENTED", "true")
    os.environ.setdefault("ASR_TURN_GUIDED", "true")
    os.environ.setdefault("ASR_UNLOAD_FOR_DIARIZATION", "true")

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    from _bootstrap import bootstrap

    bootstrap()
    from backend.pipeline import run_transcription_job
    from tests.golden.accuracy import count_speaker_lines, unique_speakers_in_window

    t0 = time.perf_counter()
    result = run_transcription_job(
        media_path=str(clip),
        selected_engines=["Typhoon Whisper"],
        language="Thai",
        diarization=True,
        max_speakers=args.max_speakers,
        enhance=False,
        # No forced min_speakers: validate the real default path (upper bound only).
        diarize_kwargs=None,
    )
    elapsed = time.perf_counter() - t0
    text = result["results"]["Typhoon Whisper"].get("text", "")
    if not text or text.startswith("ERROR:"):
        print(text[:500], file=sys.stderr)
        return 1

    spk = count_speaker_lines(text)
    first_min = sorted(unique_speakers_in_window(text, 60.0))
    print(f"window start={args.start:.0f}s dur={args.seconds:.0f}s elapsed={elapsed:.1f}s lines={len(text.splitlines())}")
    print(f"speakers={len(spk)} counts={spk}")
    print(f"first_minute={first_min}")
    ok = len(spk) >= 2 and len(first_min) >= 2
    print("PASS" if ok else "FAIL")
    out = REPO / "tests" / "output" / f"SM_pipeline_{int(args.start)}_{int(args.seconds)}s_actual.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"saved: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
