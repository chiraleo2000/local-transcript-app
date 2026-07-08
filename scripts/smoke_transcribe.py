#!/usr/bin/env python3
"""End-to-end pipeline smoke test on an arbitrary audio file.

Runs the real transcription job (ASR + optional diarization) exactly like the
app does, and reports the transcript, detected speakers, elapsed time and the
performance target. Useful for validating changes on GPU when the golden
fixtures are unavailable.

Example:
    python scripts/smoke_transcribe.py --audio tests/e2e/fixtures/large.wav \
        --language Thai --diarization --max-speakers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="path to an audio/video file")
    parser.add_argument("--language", default="Thai")
    parser.add_argument("--engine", default="Auto")
    parser.add_argument("--diarization", action="store_true")
    parser.add_argument("--max-speakers", type=int, default=4)
    parser.add_argument("--enhance", action="store_true")
    args = parser.parse_args()

    # Resolve before importing app (app.py chdir()s to the install root).
    audio = Path(args.audio).resolve()
    if not audio.is_file():
        print(f"Missing audio: {audio}", file=sys.stderr)
        return 1

    from _bootstrap import bootstrap

    bootstrap()

    from backend.asr_performance import performance_target_seconds
    from backend.pipeline import run_transcription_job
    from backend.services.media_pipeline import audio_duration_seconds

    duration = audio_duration_seconds(str(audio))
    target = performance_target_seconds(duration)

    t0 = time.perf_counter()
    result = run_transcription_job(
        media_path=str(audio),
        selected_engines=[args.engine],
        language=args.language,
        diarization=args.diarization,
        max_speakers=args.max_speakers,
        enhance=args.enhance,
    )
    elapsed = time.perf_counter() - t0

    engine_key = next(iter(result.get("results", {})), None)
    payload = result.get("results", {}).get(engine_key, {}) if engine_key else {}
    text = payload.get("text", "") or ""

    print(f"audio={audio.name} duration={duration:.1f}s")
    print(f"elapsed={elapsed:.1f}s target={target:.1f}s within_budget={elapsed <= target}")
    print(f"engine={engine_key} chars={len(text)} lines={len(text.splitlines())}")

    if args.diarization:
        speakers = sorted({
            line.split("]")[-1].split(":")[0].strip()
            for line in text.splitlines()
            if "SPEAKER" in line
        }) if text else []
        print(f"speakers={len(speakers)} {speakers}")

    print("----- transcript (first 1200 chars) -----")
    print(text[:1200] if text else "<empty>")

    ok = bool(text) and not text.startswith("ERROR:")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    from _gpu_queue import run_locked

    raise SystemExit(run_locked(main))
