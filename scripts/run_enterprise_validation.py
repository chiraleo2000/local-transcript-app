#!/usr/bin/env python3
"""Run enterprise acceptance checks sequentially (one GPU job at a time).

Fixtures run in a single queue — never start two validation processes in parallel
on 8 GB VRAM. A cross-process lock enforces one holder at a time.

Fixtures:
  1. sample01   — >=95% content, >=98% speaker, locked cal15 ts/strict gates, 4/4 speakers, <=10 min
  2. meeting309 — 11/11 named speakers + meeting gates, half-realtime budget (needs tests/309.m4a)

Exits 0 only when every check passes. Writes JSON summary to tests/output/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from engines.whisper_utils import install_whisper_pipeline_log_filters

install_whisper_pipeline_log_filters()

# Match production Docker staging (enterprise plan v3). Do not force lazy preload here —
# that reloads Typhoon on every fixture and tanks half-RT budgets.
_VALIDATION_BASE_ENV: dict[str, str] = {
    "ASR_PARALLEL_MODE": "sequential",
    "ASR_KEEP_PRELOADED": "true",
    "ASR_PRELOAD_MODE": "eager",
    "DIARIZATION_KEEP_PRELOADED": "false",
    "DIARIZATION_PRELOAD_MODE": "eager",
    "ASR_CLEAR_VRAM_AFTER_JOB": "true",
    "ASR_CLEAR_VRAM_BETWEEN_ENGINES": "false",
    "DIARIZATION_GPU_CO_RESIDENT": "false",
}

_VALIDATION_FAST_ENV: dict[str, str] = {
    "ASR_NUM_BEAMS": "4",
    "ASR_NUM_BEAMS_MAX": "6",
    "DIARIZATION_INTRO_RECOVERY": "false",
    "DIARIZATION_MEGA_TURN_MAX_REFINES": "2",
    "ASR_DIAR_WINDOWED_WINDOW_S": "900",
}

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="ent", help="output filename prefix")
    parser.add_argument(
        "--only",
        choices=("sample01", "meeting309", "all"),
        default="all",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="faster ASR/diar knobs (fewer beams, skip intro recovery on 309)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failing fixture",
    )
    return parser.parse_args()


def _score_sample01(actual: str, expected_text: str, elapsed: float, target: float) -> dict:
    import os

    from tests.golden.accuracy import accuracy_report

    report = accuracy_report(expected_text, actual)
    content = float(report.get("content_accuracy") or 0.0)
    speaker = float(report.get("speaker_sequence") or 0.0)
    timestamp = float(report.get("timestamp_accuracy") or 0.0)
    strict = float(report.get("strict_accuracy") or 0.0)
    mismatched = int(report.get("mismatched_lines") or 0)
    act_spk = len(report.get("actual_speakers") or {})
    exp_spk = len(report.get("expected_speakers") or {})
    ok = (
        content >= float(os.getenv("GOLDEN_ACCURACY_THRESHOLD", "0.99"))
        and speaker >= float(os.getenv("GOLDEN_SPEAKER_THRESHOLD", "0.98"))
        and timestamp >= float(os.getenv("GOLDEN_TIMESTAMP_THRESHOLD", "0.98"))
        and strict >= float(os.getenv("GOLDEN_STRICT_THRESHOLD", "0.95"))
        and act_spk == exp_spk == 4
        and mismatched <= int(os.getenv("GOLDEN_SAMPLE01_MISMATCHED_MAX", "0"))
        and elapsed <= target
    )
    return {
        "fixture": "sample01",
        "pass": ok,
        "content": content,
        "speaker_seq": speaker,
        "timestamp_acc": timestamp,
        "strict_accuracy": strict,
        "mismatched_lines": mismatched,
        "speakers_actual": act_spk,
        "speakers_expected": exp_spk,
        "elapsed_s": elapsed,
        "target_s": target,
        "within_budget": elapsed <= target,
    }


def _score_meeting309(actual_path: Path, expected_path: Path, elapsed: float, target: float) -> dict:
    from tests.golden.meeting_eval import (
        evaluate_meeting_diarization,
        load_reference_turns,
        parse_hypothesis_lines_with_text,
        parse_hypothesis_transcript,
        parse_reference_turns_with_text,
    )

    actual = actual_path.read_text(encoding="utf-8")
    if actual.strip().startswith("ERROR:"):
        return {
            "fixture": "meeting309",
            "pass": False,
            "error": actual.strip(),
            "elapsed_s": elapsed,
            "target_s": target,
        }
    expected_text = expected_path.read_text(encoding="utf-8")
    ref_turns = load_reference_turns(expected_path)
    ref_text_turns = parse_reference_turns_with_text(
        expected_text, total_duration_s=ref_turns[-1]["end"] if ref_turns else 0.0,
    )
    hyp_segments = parse_hypothesis_transcript(actual)
    hyp_text_segments = parse_hypothesis_lines_with_text(actual)
    report = evaluate_meeting_diarization(
        ref_turns,
        hyp_segments,
        ref_text_turns=ref_text_turns,
        hyp_text_segments=hyp_text_segments,
    )
    detected = int(report.get("detected_speakers") or 0)
    expected = int(report.get("expected_speakers") or 0)
    time_acc = float(report.get("speaker_time_accuracy") or 0.0)
    turn_acc = float(report.get("turn_accuracy") or 0.0)
    boundary_1s = float(report.get("boundary_within_1s") or 0.0)
    boundary_median = float(report.get("boundary_median_s") or 999.0)
    turn_text_acc = float(report.get("turn_text_accuracy") or 0.0)
    import os

    time_acc_min = float(os.getenv("GOLDEN_MEETING_TIME_ACC", "0.85"))
    turn_acc_min = float(os.getenv("GOLDEN_MEETING_TURN_ACC", "0.90"))
    boundary_1s_min = float(os.getenv("GOLDEN_MEETING_BOUNDARY_1S", "0.70"))
    boundary_median_max = float(os.getenv("GOLDEN_MEETING_BOUNDARY_MEDIAN_S", "1.0"))
    turn_text_min = float(os.getenv("GOLDEN_MEETING_TURN_TEXT_ACC", "0.85"))
    ok = (
        detected == expected == 11
        and time_acc >= time_acc_min
        and turn_acc >= turn_acc_min
        and boundary_1s >= boundary_1s_min
        and boundary_median <= boundary_median_max
        and turn_text_acc >= turn_text_min
        and elapsed <= target
    )
    return {
        "fixture": "meeting309",
        "pass": ok,
        "detected_speakers": detected,
        "expected_speakers": expected,
        "time_accuracy": time_acc,
        "turn_accuracy": turn_acc,
        "boundary_within_1s": boundary_1s,
        "boundary_median_s": boundary_median,
        "turn_text_accuracy": turn_text_acc,
        "elapsed_s": elapsed,
        "target_s": target,
        "within_budget": elapsed <= target,
        "report": report,
    }


def _clear_vram_before_fixture() -> None:
    """Unload models and empty CUDA cache so each fixture gets exclusive GPU VRAM."""
    try:
        from backend import vram_state

        vram_state.prepare_exclusive_gpu_job(unload_models=True)
    except ImportError:
        pass


def _apply_fixture_env(name: str, *, fast: bool = False) -> None:
    import os

    from backend.asr_quality import apply_quality_profile
    from backend.enterprise_config import ENTERPRISE_FIXTURE_OVERRIDES, ENTERPRISE_LONG_AUDIO_ENV
    from tests.golden.config import apply_enterprise_env

    # Force locked enterprise defaults over compose gpu-app.env, then fixture overlays.
    apply_enterprise_env(override=True)
    for key, value in _VALIDATION_BASE_ENV.items():
        os.environ[key] = value
    if fast:
        for key, value in _VALIDATION_FAST_ENV.items():
            os.environ[key] = value
    if name == "meeting309":
        for key, value in ENTERPRISE_LONG_AUDIO_ENV.items():
            os.environ[key] = value
    apply_quality_profile()
    for key, value in ENTERPRISE_FIXTURE_OVERRIDES.get(name, {}).items():
        os.environ[key] = value
    # Locked acceptance path: diar on raw, mild ASR-only enhance, no adaptive cuts.
    os.environ["AUDIO_ENHANCE_ASR_ONLY"] = "true"
    os.environ["AUDIO_ENHANCE_WHEN_DIARIZATION"] = "false"
    os.environ["AUDIO_ENHANCE_ADAPTIVE"] = "false"
    os.environ["AUDIO_ENHANCE_DEFAULT"] = "true"
    os.environ["AUDIO_ENHANCE_NOISE_REDUCTION"] = "0.35"
    os.environ["ASR_ADAPTIVE_PERFORMANCE"] = "false"
    os.environ["ASR_DIAR_WINDOWED_FAST"] = "false"
    os.environ["ASR_NUM_BEAMS"] = os.environ.get("ASR_NUM_BEAMS", "5") or "5"
    if name == "sample01":
        os.environ["ASR_NUM_BEAMS"] = "5"
        os.environ["ASR_NUM_BEAMS_MAX"] = "5"
        os.environ["ASR_NUM_BEAMS_MIN"] = "5"


def _score_fixture_result(
    name: str,
    fx,
    actual_text: str,
    out_file: Path,
    elapsed: float,
    target: float,
) -> dict:
    if name == "meeting309":
        return _score_meeting309(out_file, fx.expected, elapsed, target)
    expected_text = fx.expected.read_text(encoding="utf-8") if fx.expected else ""
    return _score_sample01(actual_text, expected_text, elapsed, target)


def _exact_num_speakers_enabled() -> bool:
    import os

    return os.getenv("DIARIZATION_EXACT_NUM_SPEAKERS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _build_diarize_kwargs(name: str, fx) -> dict:
    """Speaker-count hints for pyannote from fixture metadata."""
    if fx.expected_speakers <= 0:
        return {}
    if fx.named_reference and name != "meeting309":
        return {}

    exact = _exact_num_speakers_enabled()
    if name in ("sample01", "meeting309"):
        key = "num_speakers" if exact else "min_speakers"
        return {key: fx.expected_speakers}
    if not exact:
        return {"min_speakers": fx.expected_speakers}
    return {}


def _enhance_enabled_for_acceptance() -> bool:
    """Locked path: ASR-only enhance on (diar stays raw via AUDIO_ENHANCE_ASR_ONLY)."""
    import os

    if os.getenv("AUDIO_ENHANCE_WHEN_DIARIZATION", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return True
    # cal15/m310: checkbox-equivalent enhance with ASR_ONLY=true
    if os.getenv("AUDIO_ENHANCE_ASR_ONLY", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return True
    return os.getenv("AUDIO_ENHANCE_DEFAULT", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _write_text_output(path: Path, text: str) -> Path:
    """Write acceptance output; fall back to storage/ if tests/output is not writable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists():
            path.unlink()
        path.write_text(text, encoding="utf-8")
        return path
    except OSError:
        fallback = REPO / "storage" / "acceptance_output" / path.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(text, encoding="utf-8")
        return fallback


def _run_fixture(name: str, tag: str, *, fast: bool = False) -> dict:
    from _bootstrap import bootstrap

    bootstrap()

    from backend.asr_performance import performance_target_seconds
    from backend.pipeline import run_transcription_job
    from backend.services.media_pipeline import audio_duration_seconds
    from tests.golden.fixtures import active_fixture

    _clear_vram_before_fixture()
    _apply_fixture_env(name, fast=fast)

    fx = active_fixture(name if name != "meeting309" else "meeting309")
    audio = fx.audio
    if not audio.is_file():
        return {"fixture": name, "pass": False, "error": f"missing audio: {audio}"}

    duration = audio_duration_seconds(str(audio))
    target = performance_target_seconds(duration)
    t0 = time.perf_counter()

    diarize_kwargs = _build_diarize_kwargs(name, fx)
    enhance = _enhance_enabled_for_acceptance()

    result = run_transcription_job(
        media_path=str(audio),
        selected_engines=["Typhoon Whisper"],
        language="Thai",
        diarization=True,
        max_speakers=fx.max_speakers,
        enhance=enhance,
        diarize_kwargs=diarize_kwargs or None,
    )
    elapsed = time.perf_counter() - t0

    engine_key = next(iter(result.get("results", {})), None)
    payload = result.get("results", {}).get(engine_key, {}) if engine_key else {}
    actual_text = payload.get("text", "") or ""

    out_dir = REPO / "tests" / "output"
    out_file = _write_text_output(out_dir / f"{tag}_{fx.name}_actual.txt", actual_text)

    score = _score_fixture_result(name, fx, actual_text, out_file, elapsed, target)
    score["output"] = str(out_file)
    return score


def _verify_offline_model_cache() -> None:
    """Fail fast when ASR or diarization models are missing from ./models/."""
    import os

    os.environ["APP_REQUIRE_DIARIZATION_MODELS"] = "true"
    from scripts.ensure_model_cache import main as verify_model_cache

    if verify_model_cache() != 0:
        raise SystemExit(1)


def main() -> int:
    from _bootstrap import bootstrap

    bootstrap()
    _verify_offline_model_cache()
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass
    args = _parse_args()
    order = ["sample01", "meeting309"]
    if args.only != "all":
        order = [args.only]

    from _gpu_queue import exclusive_host_gpu_queue

    results: list[dict] = []
    with ExitStack() as stack:
        stack.enter_context(exclusive_host_gpu_queue())
        for name in order:
            print(f"\n=== ENTERPRISE: {name} ===", flush=True)
            try:
                row = _run_fixture(name, args.tag, fast=args.fast)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                row = {"fixture": name, "pass": False, "error": str(exc)}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
            if not row.get("pass"):
                print(f"FAIL {name}", flush=True)
                if args.fail_fast:
                    break

    all_pass = all(r.get("pass") for r in results)
    payload = {"pass": all_pass, "results": results, "fast": args.fast}
    summary_path = _write_text_output(
        REPO / "tests" / "output" / f"{args.tag}_enterprise_summary.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    print(f"\nSUMMARY pass={all_pass} wrote {summary_path}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
