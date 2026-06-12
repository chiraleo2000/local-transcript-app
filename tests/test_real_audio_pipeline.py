"""Integration tests against user-provided real audio fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHORT_AUDIO = REPO_ROOT / "tests" / "Recording 1274.wav"
LONG_AUDIO = REPO_ROOT / "tests" / "sudar-0001.m4a"


@pytest.fixture(scope="module")
def short_audio_path() -> str:
    if not SHORT_AUDIO.is_file():
        pytest.skip(f"short fixture missing: {SHORT_AUDIO}")
    return str(SHORT_AUDIO)


@pytest.fixture(scope="module")
def long_audio_path() -> str:
    if not LONG_AUDIO.is_file():
        pytest.skip(f"long fixture missing: {LONG_AUDIO}")
    return str(LONG_AUDIO)


@pytest.mark.gpu
def test_short_real_audio_pipeline(short_audio_path: str):
    if not os.getenv("RUN_GPU_INTEGRATION"):
        pytest.skip("set RUN_GPU_INTEGRATION=1 to run GPU integration test")

    from backend.pipeline import run_transcription_job

    result = run_transcription_job(
        media_path=short_audio_path,
        selected_engines=["Typhoon Whisper"],
        language="Thai",
        diarization=True,
        max_speakers=4,
        enhance=False,
    )
    engine_result = result["results"]["Typhoon Whisper"]
    text = engine_result.get("text", "")
    assert text
    assert not text.startswith("ERROR:")


@pytest.mark.slow
@pytest.mark.gpu
def test_long_real_audio_first_window(long_audio_path: str):
    if not os.getenv("RUN_GPU_INTEGRATION"):
        pytest.skip("set RUN_GPU_INTEGRATION=1 to run GPU integration test")

    from backend.services.media_pipeline import audio_duration_seconds, normalize_media
    from engines.audio_io import count_audio_windows, iter_audio_windows_from_path
    from engines.typhoon_asr import _long_form_window_s, _long_form_overlap_s

    normalized = normalize_media(long_audio_path)
    duration_s = audio_duration_seconds(normalized)
    assert duration_s > 60

    window_s = _long_form_window_s(duration_s)
    overlap_s = min(_long_form_overlap_s(), max(0, window_s // 4))
    total = count_audio_windows(duration_s, window_s, overlap_s)
    assert total >= 1

    first = next(iter(iter_audio_windows_from_path(normalized, window_s, overlap_s)))
    offset_s, window = first
    assert offset_s == pytest.approx(0.0)
    assert len(window["raw"]) > 0
