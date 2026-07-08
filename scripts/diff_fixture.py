#!/usr/bin/env python3
"""Line-by-line diff between a golden reference and an actual transcript."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.golden.accuracy import _is_non_dialogue_line, _line_body, normalize_transcript_text
from tests.golden.fixtures import FIXTURES


def _dialogue_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip() or _is_non_dialogue_line(raw):
            continue
        body = _line_body(raw).strip()
        if body:
            lines.append(normalize_transcript_text(body))
    return lines


def diff_fixture(expected_path: Path, actual_path: Path) -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass
    expected = expected_path.read_text(encoding="utf-8")
    actual = actual_path.read_text(encoding="utf-8")
    exp_lines = _dialogue_lines(expected)
    act_lines = _dialogue_lines(actual)
    print(f"Expected lines: {len(exp_lines)}  Actual lines: {len(act_lines)}")
    for line in difflib.unified_diff(
        exp_lines,
        act_lines,
        fromfile=str(expected_path),
        tofile=str(actual_path),
        lineterm="",
    ):
        print(line)
    mismatches = sum(
        1
        for idx, exp in enumerate(exp_lines)
        if idx >= len(act_lines) or exp != act_lines[idx]
    )
    mismatches += max(0, len(act_lines) - len(exp_lines))
    print(f"\nMismatched lines (positional): {mismatches}")
    return 0 if mismatches == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff golden reference vs actual output")
    parser.add_argument("--fixture", choices=tuple(FIXTURES.keys()), default="")
    parser.add_argument("--actual", type=Path, default=None)
    parser.add_argument("expected", nargs="?", type=Path, default=None)
    parser.add_argument("actual_pos", nargs="?", type=Path, default=None)
    args = parser.parse_args()

    if args.fixture:
        fx = FIXTURES[args.fixture]
        if fx.expected is None:
            print(f"Fixture {args.fixture!r} has no reference transcript", file=sys.stderr)
            return 2
        expected = fx.expected
        actual = args.actual or fx.output
    else:
        expected = args.expected
        actual = args.actual or args.actual_pos
    if expected is None or actual is None:
        parser.error("provide --fixture or both expected and actual paths")
    if not expected.is_file():
        print(f"Missing expected file: {expected}", file=sys.stderr)
        return 2
    if not actual.is_file():
        print(f"Missing actual file: {actual}", file=sys.stderr)
        return 2
    return diff_fixture(expected, actual)


if __name__ == "__main__":
    raise SystemExit(main())
