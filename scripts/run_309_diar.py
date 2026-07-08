#!/usr/bin/env python3
"""Run diarization on tests/309.m4a and score speakers/timestamps vs tests/309.txt.

The reference is a named-speaker meeting transcript (11 speakers). Scoring maps
SPEAKER_XX labels to reference names by optimal overlap assignment, then reports
speaker count, time-weighted attribution accuracy, per-turn accuracy, and
speaker-change boundary alignment.

Examples:
    python scripts/run_309_diar.py --seconds 600 --tag head10min
    python scripts/run_309_diar.py --set DIARIZATION_CLUSTERING_THRESHOLD=0.55
    python scripts/run_309_diar.py --eval tests/output/309_actual.txt
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
OUT_DIR = REPO / "tests" / "output"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 = full file")
    parser.add_argument("--max-speakers", type=int, default=11)
    parser.add_argument("--num-speakers", type=int, default=0,
                        help="exact speaker hint (DIARIZATION_EXACT_NUM_SPEAKERS)")
    parser.add_argument("--min-speakers", type=int, default=0)
    parser.add_argument("--clust-threshold", type=float, default=None)
    parser.add_argument("--min-cluster", type=int, default=None)
    parser.add_argument("--min-off", type=float, default=None)
    parser.add_argument("--fa", type=float, default=None)
    parser.add_argument("--fb", type=float, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE")
    parser.add_argument("--eval", dest="eval_path", default="",
                        help="score an existing transcript/segments file instead of running")
    return parser.parse_args()


def _apply_overrides(overrides: list[str]) -> None:
    import os

    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Bad --set (need KEY=VALUE): {item}")
        os.environ[key.strip()] = value.strip()


def _load_hypothesis(path: Path) -> list[dict]:
    from tests.golden.meeting_eval import parse_hypothesis_transcript

    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return parse_hypothesis_transcript(path.read_text(encoding="utf-8"))


def _clip_reference(turns: list[dict], start: float, end: float) -> list[dict]:
    clipped: list[dict] = []
    for turn in turns:
        s = max(turn["start"], start)
        e = min(turn["end"], end)
        if e - s > 0.2:
            clipped.append({"start": s, "end": e, "speaker": turn["speaker"]})
    return clipped


def _report(ref_turns: list[dict], hyp_segments: list[dict], tag: str) -> dict:
    from tests.golden.meeting_eval import evaluate_meeting_diarization

    report = evaluate_meeting_diarization(ref_turns, hyp_segments)
    per_speaker = report.pop("per_speaker")
    mapping = report.pop("mapping")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("--- mapping (hyp -> ref) ---")
    for hyp, ref in sorted(mapping.items()):
        print(f"  {hyp} -> {ref}")
    print("--- per reference speaker ---")
    for name, info in sorted(per_speaker.items(), key=lambda kv: -kv[1]["ref_time_s"]):
        print(
            f"  {info['ref_time_s']:8.1f}s turns={info['turns']:4d} "
            f"turn_acc={info['turn_acc']:.3f} hyp={info['mapped_hyp']} {name}"
        )
    print(
        f"SUMMARY tag={tag} detected={report['detected_speakers']} "
        f"expected={report['expected_speakers']} "
        f"time_acc={report['speaker_time_accuracy']:.4f} "
        f"turn_acc={report['turn_accuracy']:.4f} "
        f"boundary<=2s={report['boundary_within_2s']:.4f}"
    )
    return report


def main() -> int:
    args = _parse_args()
    from _bootstrap import bootstrap

    _apply_overrides(args.overrides)
    bootstrap()
    _apply_overrides(args.overrides)  # win over .env/profile defaults

    import os

    from backend.services.media_pipeline import audio_duration_seconds
    from tests.golden.meeting_eval import load_reference_turns

    total_dur = audio_duration_seconds(str(AUDIO))
    ref_turns = load_reference_turns(EXPECTED, total_duration_s=total_dur)

    if args.eval_path:
        hyp = _load_hypothesis(Path(args.eval_path))
        if not hyp:
            print(f"No hypothesis segments in {args.eval_path}", file=sys.stderr)
            return 1
        end = max(seg["end"] for seg in hyp)
        start = min(seg["start"] for seg in hyp)
        _report(_clip_reference(ref_turns, max(0.0, start - 5.0), end + 5.0), hyp,
                args.tag or Path(args.eval_path).stem)
        return 0

    span_start = max(0.0, args.start)
    span_dur = args.seconds if args.seconds > 0 else max(0.0, total_dur - span_start)
    span_end = min(total_dur, span_start + span_dur)
    tag = args.tag or f"{int(span_start)}_{int(span_dur)}s"

    if args.num_speakers > 0:
        os.environ["DIARIZATION_EXACT_NUM_SPEAKERS"] = "true"

    audio_path = AUDIO
    tmp_clip: Path | None = None
    if span_start > 0.01 or span_end < total_dur - 0.01:
        tmp_clip = Path(tempfile.gettempdir()) / f"clip309_{int(span_start)}_{int(span_dur)}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", str(span_start), "-t", str(span_dur), "-i", str(AUDIO),
                "-ac", "1", "-ar", "16000", str(tmp_clip),
            ],
            check=True,
        )
        audio_path = tmp_clip

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )

    from engines.diarization import diarize

    t0 = time.perf_counter()
    segments = diarize(
        str(audio_path),
        num_speakers=args.num_speakers,
        max_speakers=args.max_speakers,
        audio_duration_s=span_dur,
        min_speakers_hint=args.min_speakers,
        clust_threshold=args.clust_threshold,
        clust_min_size=args.min_cluster,
        seg_min_duration_off=args.min_off,
        vbx_fa=args.fa,
        vbx_fb=args.fb,
    )
    elapsed = time.perf_counter() - t0

    # Shift back to absolute file time for scoring/saving.
    for seg in segments:
        seg["start"] += span_start
        seg["end"] += span_start

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"309_diar_{tag}.json"
    out_json.write_text(
        json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"=== 309 diarization span={span_start:.0f}-{span_end:.0f}s "
          f"elapsed={elapsed:.1f}s segments={len(segments)} ===")
    _report(_clip_reference(ref_turns, span_start, span_end), segments, tag)
    print(f"saved: {out_json}")
    return 0


if __name__ == "__main__":
    from _gpu_queue import run_locked

    raise SystemExit(run_locked(main))
