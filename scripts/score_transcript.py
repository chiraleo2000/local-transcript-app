#!/usr/bin/env python3
"""Score a transcript against a golden reference without GPU (fast improve loop).

Examples:
  python scripts/score_transcript.py tests/test-sample01.txt tests/output/sample01_actual.txt
  python scripts/score_transcript.py --meeting tests/309.txt path/to/actual.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score transcript vs golden (CPU-only)")
    p.add_argument("expected", type=Path, help="Golden / reference transcript")
    p.add_argument("actual", type=Path, help="Hypothesis transcript to score")
    p.add_argument(
        "--meeting",
        action="store_true",
        help="Use meeting309 named-speaker scorer instead of [SPEAKER_XX] scorer",
    )
    p.add_argument("--json", action="store_true", help="Print JSON only")
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Total audio duration (s) for meeting reference parsing",
    )
    return p.parse_args()


def _score_speaker_xx(expected: str, actual: str) -> dict:
    from tests.golden.accuracy import accuracy_report

    return accuracy_report(expected, actual)


def _score_meeting(expected: str, actual: str, duration_s: float) -> dict:
    from tests.golden.meeting_eval import (
        evaluate_meeting_diarization,
        parse_hypothesis_lines_with_text,
        parse_hypothesis_transcript,
        parse_named_reference,
        parse_reference_turns_with_text,
    )

    ref_turns = parse_named_reference(expected, total_duration_s=duration_s)
    ref_text_turns = parse_reference_turns_with_text(
        expected, total_duration_s=duration_s
    )
    hyp_segments = parse_hypothesis_transcript(actual)
    hyp_text_segments = parse_hypothesis_lines_with_text(actual)
    return evaluate_meeting_diarization(
        ref_turns,
        hyp_segments,
        ref_text_turns=ref_text_turns,
        hyp_text_segments=hyp_text_segments,
    )


def main() -> int:
    args = _parse_args()
    if not args.expected.is_file():
        print(f"Missing expected: {args.expected}", file=sys.stderr)
        return 2
    if not args.actual.is_file():
        print(f"Missing actual: {args.actual}", file=sys.stderr)
        return 2
    expected = args.expected.read_text(encoding="utf-8")
    actual = args.actual.read_text(encoding="utf-8")

    if args.meeting:
        report = _score_meeting(expected, actual, args.duration)
    else:
        report = _score_speaker_xx(expected, actual)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"expected: {args.expected}")
    print(f"actual:   {args.actual}")
    keys = (
        "content_accuracy",
        "speaker_sequence",
        "timestamp_accuracy",
        "strict_accuracy",
        "accuracy",
        "mismatched_lines",
        "expected_speakers",
        "detected_speakers",
        "speaker_time_accuracy",
        "turn_accuracy",
        "boundary_within_1s",
        "boundary_median_s",
        "turn_text_accuracy",
    )
    for key in keys:
        if key in report:
            print(f"  {key}: {report[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
