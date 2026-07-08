#!/usr/bin/env python3
"""Score the real app pipeline against a golden reference transcript.

Unlike tests/golden/runner.py, this runs the pipeline the way the *app* does by
default: diarization with ``max_speakers`` as a pure upper bound and NO forced
``min_speakers`` (so it reflects what users actually get). It prints the full
accuracy report and writes the produced transcript for diffing.

Config can be overridden per-run with repeated --set KEY=VALUE flags, so ASR /
diarization parameters can be swept without editing files.

Examples:
    python scripts/score_fixture.py --fixture sample01
    python scripts/score_fixture.py --fixture recording6250 --max-speakers 2
    python scripts/score_fixture.py --fixture sample01 --set ASR_NUM_BEAMS=8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", help="golden fixture name (see tests/golden/fixtures.py)")
    parser.add_argument("--audio", help="audio path (overrides fixture)")
    parser.add_argument("--expected", help="expected transcript path (overrides fixture)")
    parser.add_argument("--engine", default="Typhoon Whisper")
    parser.add_argument("--language", default="Thai")
    parser.add_argument("--max-speakers", type=int, default=0)
    parser.add_argument("--force-min", type=int, default=0,
                        help="force min_speakers (0 = app default: upper bound only)")
    parser.add_argument("--reference-diar", action="store_true",
                        help="use golden turn boundaries instead of pyannote")
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--tag", default="", help="output filename tag")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="env override (repeatable)")
    return parser.parse_args()


def _resolve_target(args: argparse.Namespace) -> tuple[Path, Path | None, int, str]:
    """Return (audio, expected, max_speakers, tag) from --fixture or --audio."""
    from tests.golden.fixtures import active_fixture

    max_speakers = args.max_speakers
    tag = args.tag
    if args.fixture:
        fx = active_fixture(args.fixture)
        audio_path = fx.audio
        expected_path = fx.expected
        max_speakers = max_speakers if max_speakers > 0 else fx.max_speakers
        tag = tag or fx.name
    else:
        audio_path = Path(args.audio).resolve()
        expected_path = Path(args.expected).resolve() if args.expected else None
        tag = tag or audio_path.stem
    if args.expected:
        expected_path = Path(args.expected).resolve()
    return audio_path, expected_path, (max_speakers if max_speakers > 0 else 4), tag


def _apply_overrides(overrides: list[str]) -> None:
    import os

    for item in overrides:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"Bad --set (need KEY=VALUE): {item}")
        os.environ[key.strip()] = value.strip()


def main() -> int:
    args = _parse_args()
    if not args.fixture and not args.audio:
        print("Provide --fixture or --audio", file=sys.stderr)
        return 2

    from _bootstrap import bootstrap

    bootstrap()

    import os

    audio_path, expected_path, max_speakers, tag = _resolve_target(args)
    if not audio_path.is_file():
        print(f"Missing audio: {audio_path}", file=sys.stderr)
        return 1

    _apply_overrides(args.overrides)

    from backend.asr_performance import performance_target_seconds
    from backend.pipeline import run_transcription_job
    from backend.services.media_pipeline import audio_duration_seconds

    diarize_kwargs: dict = {}
    if args.reference_diar and expected_path and expected_path.is_file():
        from tests.golden.reference_diar import load_reference_segments

        diarize_kwargs["reference_segments"] = load_reference_segments(expected_path)
        os.environ["ASR_TURN_GUIDED"] = "true"
    elif args.force_min > 0:
        diarize_kwargs["min_speakers"] = args.force_min

    duration = audio_duration_seconds(str(audio_path))
    target = performance_target_seconds(duration)

    t0 = time.perf_counter()
    result = run_transcription_job(
        media_path=str(audio_path),
        selected_engines=[args.engine],
        language=args.language,
        diarization=True,
        max_speakers=max_speakers,
        enhance=args.enhance,
        diarize_kwargs=diarize_kwargs or None,
    )
    elapsed = time.perf_counter() - t0

    engine_key = next(iter(result.get("results", {})), None)
    payload = result.get("results", {}).get(engine_key, {}) if engine_key else {}
    actual_text = payload.get("text", "") or ""

    out_dir = REPO / "tests" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    actual_file = out_dir / f"{tag}_actual.txt"
    actual_file.write_text(actual_text, encoding="utf-8")

    print(f"=== fixture={tag} audio={audio_path.name} duration={duration:.1f}s ===")
    within = elapsed <= target
    print(f"elapsed={elapsed:.1f}s target={target:.1f}s within_budget={within}")
    _report(expected_path, actual_text, tag, within)
    print(f"saved: {actual_file}")
    return 0


def _report(expected_path: Path | None, actual_text: str, tag: str, within: bool) -> None:
    from tests.golden.accuracy import accuracy_report, count_speaker_lines

    if not (expected_path and expected_path.is_file()):
        print(f"(no expected transcript) speakers={count_speaker_lines(actual_text)}")
        return
    expected_text = expected_path.read_text(encoding="utf-8")
    report = accuracy_report(expected_text, actual_text)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    content = report.get("content_accuracy") or 0.0
    speaker = report.get("speaker_sequence") or 0.0
    exp_spk = len(report.get("expected_speakers") or {})
    act_spk = len(report.get("actual_speakers") or {})
    print(f"SUMMARY tag={tag} content={content:.4f} speaker_seq={speaker:.4f} "
          f"speakers actual={act_spk} expected={exp_spk} within_budget={within}")


if __name__ == "__main__":
    raise SystemExit(main())
