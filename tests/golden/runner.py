"""Execute golden pipeline for configured fixture and return accuracy report."""

from __future__ import annotations

import os
import time

from tests.golden.accuracy import accuracy_report
from tests.golden.config import apply_golden_env
from tests.golden.debug_log import debug_log
from tests.golden.fixtures import GoldenFixture, active_fixture


def _gpu_required() -> bool:
    return os.getenv("GOLDEN_REQUIRE_GPU", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


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


def _assert_gpu_pipeline(gpu_status: dict[str, str | bool]) -> None:
    if not _gpu_required():
        return
    if not gpu_status.get("diarization_cuda"):
        raise RuntimeError(
            f"GOLDEN_REQUIRE_GPU: diarization did not run on CUDA "
            f"(device={gpu_status.get('diarization_device')!r})"
        )
    if not gpu_status.get("asr_cuda"):
        raise RuntimeError(
            f"GOLDEN_REQUIRE_GPU: ASR did not run on CUDA "
            f"(device={gpu_status.get('asr_device')!r})"
        )


def run_golden_fixture(
    fixture: GoldenFixture | None = None,
    *,
    threshold: float | None = None,
    run_id: str = "golden",
    profile_extra: dict[str, str] | None = None,
) -> dict:
    """Run full pipeline and score against expected transcript."""
    fixture = fixture or active_fixture(os.getenv("GOLDEN_FIXTURE"))
    apply_golden_env(profile_extra)
    threshold = threshold if threshold is not None else float(
        os.getenv("GOLDEN_ACCURACY_THRESHOLD", "0.95")
    )
    use_reference_diar = os.getenv("GOLDEN_REFERENCE_DIAR", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if use_reference_diar and _gpu_required():
        raise RuntimeError(
            "GOLDEN_REFERENCE_DIAR skips GPU diarization; disable GOLDEN_REQUIRE_GPU "
            "or set GOLDEN_REFERENCE_DIAR=0"
        )

    if not fixture.audio.is_file():
        raise FileNotFoundError(f"golden audio missing: {fixture.audio}")
    if not fixture.expected.is_file():
        raise FileNotFoundError(f"golden transcript missing: {fixture.expected}")

    expected_text = fixture.expected.read_text(encoding="utf-8")
    t0 = time.perf_counter()

    debug_log(
        hypothesis_id="H0",
        location="tests/golden/runner.py:start",
        message="golden run starting",
        data={
            "fixture": fixture.name,
            "audio": str(fixture.audio),
            "threshold": threshold,
            "max_speakers": fixture.max_speakers,
            "asr_turn_guided": os.getenv("ASR_TURN_GUIDED"),
            "diarization_device": os.getenv("DIARIZATION_DEVICE"),
            "require_gpu": _gpu_required(),
        },
        run_id=run_id,
    )

    from backend.asr_quality import apply_quality_profile
    from backend.pipeline import run_transcription_job

    profile = apply_quality_profile()
    # Quality profile 8GB safety must not override explicit golden CUDA staging.
    apply_golden_env(profile_extra)

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

    result = run_transcription_job(
        media_path=str(fixture.audio),
        selected_engines=["Typhoon Whisper"],
        language="Thai",
        diarization=True,
        max_speakers=fixture.max_speakers,
        enhance=os.getenv("AUDIO_ENHANCE_WHEN_DIARIZATION", "").strip().lower()
        in {"1", "true", "yes", "on"},
    )

    elapsed_s = time.perf_counter() - t0
    gpu_status = _pipeline_gpu_status()
    _assert_gpu_pipeline(gpu_status)

    engine_result = result["results"]["Typhoon Whisper"]
    actual_text = engine_result.get("text", "")
    if not actual_text:
        raise RuntimeError("pipeline returned empty transcript")
    if actual_text.startswith("ERROR:"):
        raise RuntimeError(actual_text[:500])
    if _gpu_required() and fixture.max_speakers > 1:
        speaker_tags = sum(
            1 for line in actual_text.splitlines() if "SPEAKER_" in line.upper()
        )
        if speaker_tags < 2:
            raise RuntimeError(
                "GOLDEN_REQUIRE_GPU: diarization appears to have failed "
                f"(only {speaker_tags} speaker-tagged lines in output)"
            )

    fixture.output.parent.mkdir(parents=True, exist_ok=True)
    fixture.output.write_text(actual_text, encoding="utf-8")

    report = accuracy_report(expected_text, actual_text)
    score = report["accuracy"]
    passed = score >= threshold

    debug_log(
        hypothesis_id="H1-H5",
        location="tests/golden/runner.py:score",
        message="golden accuracy report",
        data={
            **report,
            "fixture": fixture.name,
            "threshold": threshold,
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
        "fixture": fixture.name,
        "report": report,
        "elapsed_s": elapsed_s,
        "gpu_status": gpu_status,
        "output_path": str(fixture.output),
        "actual_text": actual_text,
    }


def run_golden_sample01(**kwargs) -> dict:
    return run_golden_fixture(active_fixture("sample01"), **kwargs)
