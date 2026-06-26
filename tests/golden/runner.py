"""Execute golden pipeline for configured fixture and return accuracy report."""

from __future__ import annotations

import os
import time
from pathlib import Path

from tests.golden.accuracy import accuracy_report
from tests.golden.config import (
    apply_golden_env,
    apply_production_perf_env,
    golden_accuracy_threshold,
    golden_speaker_threshold,
    golden_timestamp_threshold,
    performance_check_enabled,
)
from tests.golden.debug_log import debug_log
from tests.golden.fixtures import GoldenFixture, active_fixture


def _gpu_required() -> bool:
    return os.getenv("GOLDEN_REQUIRE_GPU", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _apply_env(profile_extra: dict[str, str] | None, *, production_mode: bool) -> None:
    if production_mode:
        apply_production_perf_env(profile_extra)
    else:
        apply_golden_env(profile_extra)


def _pipeline_gpu_status() -> dict[str, str | bool]:
    """Report whether diarization and ASR ran on CUDA."""
    status: dict[str, str | bool] = {"diarization_cuda": False, "asr_cuda": False}
    try:
        from engines import diarization

        tracked = diarization.last_inference_device() or getattr(
            diarization, "_tracked_device", None
        ) or ""
        status["diarization_device"] = tracked
        status["diarization_cuda"] = "cuda" in str(tracked).lower()
    except ImportError:
        status["diarization_device"] = "unknown"

    try:
        from engines.typhoon_asr import _get_pipeline

        pipe = _get_pipeline()
        model = getattr(pipe, "model", None)
        if model is not None and hasattr(model, "device"):
            asr_device = str(model.device)
            status["asr_device"] = asr_device
            status["asr_cuda"] = "cuda" in asr_device.lower()
    except (ImportError, AttributeError, RuntimeError, ValueError):
        try:
            from engines.hardware import detect_hardware

            hw = detect_hardware()
            asr_device = str(hw.get("selected_device", "unknown"))
            status["asr_device"] = asr_device
            status["asr_cuda"] = "cuda" in asr_device.lower()
        except ImportError:
            status["asr_device"] = "unknown"
    return status


def _assert_gpu_pipeline(gpu_status: dict[str, str | bool], *, reference_diar: bool = False) -> None:
    if not _gpu_required():
        return
    if not reference_diar and not gpu_status.get("diarization_cuda"):
        raise RuntimeError(
            f"GOLDEN_REQUIRE_GPU: diarization did not run on CUDA "
            f"(device={gpu_status.get('diarization_device')!r})"
        )
    if not gpu_status.get("asr_cuda"):
        raise RuntimeError(
            f"GOLDEN_REQUIRE_GPU: ASR did not run on CUDA "
            f"(device={gpu_status.get('asr_device')!r})"
        )


def _stage_audio_for_inference(audio_path: Path) -> str:
    """Copy audio to Linux-local storage (avoids slow Windows bind-mount per turn)."""
    import shutil

    if not Path("/app").is_dir():
        return str(audio_path)
    staging = Path("/tmp/golden_audio")
    staging.mkdir(parents=True, exist_ok=True)
    dest = staging / audio_path.name.replace(" ", "_")
    try:
        src_mtime = audio_path.stat().st_mtime
        if not dest.exists() or dest.stat().st_mtime < src_mtime:
            shutil.copy2(audio_path, dest)
    except OSError:
        return str(audio_path)
    return str(dest)


def run_golden_fixture(
    fixture: GoldenFixture | None = None,
    *,
    threshold: float | None = None,
    run_id: str = "golden",
    profile_extra: dict[str, str] | None = None,
    production_mode: bool = False,
) -> dict:
    """Run full pipeline and score against expected transcript (if configured)."""
    fixture = fixture or active_fixture(os.getenv("GOLDEN_FIXTURE"))
    _apply_env(profile_extra, production_mode=production_mode)

    use_reference_diar = os.getenv("GOLDEN_REFERENCE_DIAR", "").strip().lower() in {
        "1", "true", "yes", "on",
    }

    if not fixture.audio.is_file():
        raise FileNotFoundError(f"golden audio missing: {fixture.audio}")
    if fixture.requires_accuracy() and fixture.expected is not None:
        if not fixture.expected.is_file():
            raise FileNotFoundError(f"golden transcript missing: {fixture.expected}")

    if threshold is None:
        threshold = fixture.accuracy_threshold

    expected_text = ""
    if fixture.expected is not None and fixture.expected.is_file():
        expected_text = fixture.expected.read_text(encoding="utf-8")

    target_s = fixture.performance_target_s()
    t0 = time.perf_counter()

    debug_log(
        hypothesis_id="H0",
        location="tests/golden/runner.py:start",
        message="golden run starting",
        data={
            "fixture": fixture.name,
            "audio": str(fixture.audio),
            "threshold": threshold,
            "target_s": target_s,
            "max_speakers": fixture.max_speakers,
            "production_mode": production_mode,
            "asr_turn_guided": os.getenv("ASR_TURN_GUIDED"),
            "diarization_device": os.getenv("DIARIZATION_DEVICE"),
            "require_gpu": _gpu_required(),
        },
        run_id=run_id,
    )

    from backend.asr_quality import apply_quality_profile
    from backend.pipeline import run_transcription_job

    profile = apply_quality_profile()
    _apply_env(profile_extra, production_mode=production_mode)

    if _gpu_required():
        try:
            from backend import vram_state

            vram_state.teardown(aggressive=True)
        except ImportError:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                probe = torch.zeros(1, device="cuda")
                del probe
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass

    debug_log(
        hypothesis_id="H3",
        location="tests/golden/runner.py:profile",
        message="quality profile applied",
        data={"profile": profile, "turn_guided": os.getenv("ASR_TURN_GUIDED")},
        run_id=run_id,
    )

    diarize_kwargs: dict = {}
    if use_reference_diar and fixture.expected is not None and fixture.expected.is_file():
        from tests.golden.reference_diar import load_reference_segments

        diarize_kwargs["reference_segments"] = load_reference_segments(fixture.expected)
        os.environ["ASR_TURN_GUIDED"] = "true"
    elif fixture.max_speakers > 0:
        diarize_kwargs["num_speakers"] = fixture.max_speakers

    result = run_transcription_job(
        media_path=_stage_audio_for_inference(fixture.audio),
        selected_engines=["Typhoon Whisper"],
        language="Thai",
        diarization=True,
        max_speakers=fixture.max_speakers,
        enhance=os.getenv("AUDIO_ENHANCE_WHEN_DIARIZATION", "").strip().lower()
        in {"1", "true", "yes", "on"},
        diarize_kwargs=diarize_kwargs or None,
    )

    elapsed_s = time.perf_counter() - t0
    gpu_status = _pipeline_gpu_status()
    _assert_gpu_pipeline(gpu_status, reference_diar=use_reference_diar)

    engine_result = result["results"]["Typhoon Whisper"]
    actual_text = engine_result.get("text", "")
    if not actual_text:
        raise RuntimeError("pipeline returned empty transcript")
    if actual_text.startswith("ERROR:"):
        raise RuntimeError(actual_text[:500])

    speaker_tags = sum(
        1 for line in actual_text.splitlines() if "SPEAKER_" in line.upper()
    )
    if _gpu_required() and fixture.max_speakers > 1 and speaker_tags < 2:
        raise RuntimeError(
            "GOLDEN_REQUIRE_GPU: diarization appears to have failed "
            f"(only {speaker_tags} speaker-tagged lines in output)"
        )

    fixture.output.parent.mkdir(parents=True, exist_ok=True)
    fixture.output.write_text(actual_text, encoding="utf-8")

    manifest_target = float(result.get("target_elapsed_s") or 0.0)
    if target_s <= 0 and manifest_target > 0:
        target_s = manifest_target
    performance_met = (
        not performance_check_enabled()
        or target_s <= 0
        or elapsed_s <= target_s
    )

    report = accuracy_report(expected_text, actual_text) if expected_text else {
        "accuracy": None,
        "line_best_match": None,
        "time_aligned": None,
        "corpus_similarity": None,
        "expected_chars": 0,
        "actual_chars": len(actual_text),
        "coverage_ratio": 1.0,
        "expected_lines": 0,
        "actual_lines": len([ln for ln in actual_text.splitlines() if ln.strip()]),
        "expected_speakers": {},
        "actual_speakers": {},
    }

    score = report.get("accuracy")
    content = report.get("content_accuracy")
    speaker = report.get("speaker_sequence")
    timestamp = report.get("timestamp_accuracy")
    if fixture.requires_accuracy() and threshold is not None and score is not None:
        content_ok = (content or 0.0) >= golden_accuracy_threshold()
        speaker_ok = (speaker or 0.0) >= golden_speaker_threshold()
        timestamp_ok = (timestamp or 0.0) >= golden_timestamp_threshold()
        passed = content_ok and speaker_ok and timestamp_ok and performance_met
    else:
        line_count = int(report.get("actual_lines") or 0)
        passed = performance_met and line_count >= 20 and len(actual_text) >= 500

    debug_log(
        hypothesis_id="H1-H5",
        location="tests/golden/runner.py:score",
        message="golden accuracy report",
        data={
            **{k: v for k, v in report.items() if v is not None},
            "fixture": fixture.name,
            "threshold": threshold,
            "target_s": target_s,
            "performance_met": performance_met,
            "passed": passed,
            "elapsed_s": round(elapsed_s, 1),
            **gpu_status,
            "output_path": str(fixture.output),
            "actual_preview": actual_text[:400],
        },
        run_id=run_id,
    )

    return {
        "passed": passed,
        "threshold": threshold,
        "target_s": target_s,
        "performance_met": performance_met,
        "fixture": fixture.name,
        "report": report,
        "elapsed_s": elapsed_s,
        "gpu_status": gpu_status,
        "output_path": str(fixture.output),
        "actual_text": actual_text,
    }


def run_golden_sample01(**kwargs) -> dict:
    return run_golden_fixture(active_fixture("sample01"), **kwargs)
