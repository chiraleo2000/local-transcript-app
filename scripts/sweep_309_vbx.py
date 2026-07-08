#!/usr/bin/env python3
"""Grid-sweep community-1 VBx clustering params on a 309.m4a window.

Loads the pyannote pipeline once, then evaluates each (threshold, Fa, Fb)
combo against tests/309.txt with the meeting evaluator. Use to pick
production defaults for multi-speaker Thai meetings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
AUDIO = REPO / "tests" / "309.m4a"
EXPECTED = REPO / "tests" / "309.txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--max-speakers", type=int, default=11)
    parser.add_argument("--grid", default="",
                        help="semicolon list of t,fa,fb (e.g. '0.55,0.2,0.8;0.6,0.3,1.5')")
    args = parser.parse_args()

    from _bootstrap import bootstrap

    bootstrap()

    import os

    # Offline tuning only — production and validation use single-pass VBx.
    os.environ.setdefault("DIARIZATION_OFFLINE_TUNING", "1")
    os.environ.setdefault("DIARIZATION_MULTI_SAMPLE", "false")

    from tests.golden.meeting_eval import (
        evaluate_meeting_diarization,
        load_reference_turns,
    )

    ref_all = load_reference_turns(EXPECTED, total_duration_s=5380.6)
    span_start, span_end = args.start, args.start + args.seconds
    ref = []
    for turn in ref_all:
        s, e = max(turn["start"], span_start), min(turn["end"], span_end)
        if e - s > 0.2:
            ref.append({"start": s, "end": e, "speaker": turn["speaker"]})

    clip = Path(tempfile.gettempdir()) / f"sweep309_{int(span_start)}_{int(args.seconds)}.wav"
    if not clip.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-ss", str(span_start), "-t", str(args.seconds), "-i", str(AUDIO),
             "-ac", "1", "-ar", "16000", str(clip)],
            check=True,
        )

    import torch

    from engines.diarization import (
        _prepare_audio_for_pyannote,
        _segments_from_diarization,
        load_offline_pyannote_pipeline,
    )

    pipe = load_offline_pyannote_pipeline()
    pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    audio_input = _prepare_audio_for_pyannote(str(clip))

    if args.grid:
        combos = [tuple(float(x) for x in part.split(",")) for part in args.grid.split(";")]
    else:
        combos = [
            (0.60, 0.07, 0.8),   # model defaults
            (0.55, 0.20, 0.8),
            (0.55, 0.30, 0.8),
            (0.55, 0.40, 0.8),
            (0.50, 0.30, 0.8),
            (0.55, 0.30, 3.0),
            (0.55, 0.30, 8.0),
            (0.65, 0.30, 0.8),
        ]

    rows = []
    for threshold, fa, fb in combos:
        pipe.instantiate({
            "clustering": {"threshold": threshold, "Fa": fa, "Fb": fb},
        })
        t0 = time.perf_counter()
        diarization = pipe(dict(audio_input), max_speakers=args.max_speakers)
        elapsed = time.perf_counter() - t0
        segments = _segments_from_diarization(diarization)
        for seg in segments:
            seg["start"] += span_start
            seg["end"] += span_start
        rep = evaluate_meeting_diarization(ref, segments)
        row = {
            "t": threshold, "Fa": fa, "Fb": fb,
            "detected": rep["detected_speakers"],
            "expected": rep["expected_speakers"],
            "time_acc": rep["speaker_time_accuracy"],
            "turn_acc": rep["turn_accuracy"],
            "b2s": rep["boundary_within_2s"],
            "elapsed": round(elapsed, 1),
        }
        rows.append(row)
        print(json.dumps(row))

    rows.sort(key=lambda r: (r["time_acc"] + r["turn_acc"]), reverse=True)
    print("=== best ===")
    for row in rows[:3]:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    from _gpu_queue import run_locked

    raise SystemExit(run_locked(main))
