#!/usr/bin/env python3
"""Diarization-only probe: report detected speaker count + per-speaker time.

Faster than a full ASR+diarization run — use it to check how many speakers the
diarizer finds on a file (and how the count responds to config sweeps) before
committing to a long transcription.

Examples:
    python scripts/diarize_count.py --audio tests/309.m4a --max-speakers 11
    python scripts/diarize_count.py --audio tests/309.m4a --set DIARIZATION_CLUSTERING_THRESHOLD=0.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--max-speakers", type=int, default=0)
    parser.add_argument("--min-speakers", type=int, default=0)
    parser.add_argument("--num-speakers", type=int, default=0)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="env override (repeatable)")
    return parser.parse_args()


def _to_wav(src: Path) -> Path:
    clip = Path(tempfile.gettempdir()) / f"diar_probe_{src.stem}_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src), "-ac", "1", "-ar", "16000", str(clip)],
        check=True,
    )
    return clip


def main() -> int:
    args = _parse_args()
    import os

    for item in args.overrides:
        key, _, value = item.partition("=")
        if _:
            os.environ[key.strip()] = value.strip()

    from _bootstrap import bootstrap

    bootstrap()

    src = Path(args.audio)
    if not src.is_absolute():
        src = (REPO / src).resolve()
    if not src.is_file():
        print(f"Missing audio: {src}", file=sys.stderr)
        return 1

    from backend.services.media_pipeline import audio_duration_seconds
    from engines.diarization import diarize

    wav = _to_wav(src)
    duration = audio_duration_seconds(str(wav))
    print(f"audio={src.name} duration={duration:.1f}s "
          f"max_speakers={args.max_speakers} min_speakers={args.min_speakers} "
          f"num_speakers={args.num_speakers}")

    t0 = time.perf_counter()
    segments = diarize(
        str(wav),
        num_speakers=args.num_speakers,
        max_speakers=args.max_speakers,
        audio_duration_s=duration,
        min_speakers_hint=args.min_speakers,
    )
    elapsed = time.perf_counter() - t0

    per_speaker: dict[str, float] = defaultdict(float)
    for seg in segments:
        per_speaker[seg["speaker"]] += max(0.0, seg["end"] - seg["start"])

    ordered = sorted(per_speaker.items(), key=lambda kv: kv[1], reverse=True)
    print(f"elapsed={elapsed:.1f}s segments={len(segments)} speakers={len(per_speaker)}")
    for spk, secs in ordered:
        print(f"  {spk}: {secs:.1f}s ({secs / max(duration, 1) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    from _gpu_queue import run_locked

    raise SystemExit(run_locked(main))
