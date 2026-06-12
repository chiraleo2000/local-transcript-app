"""Tests for ASR quality profile loader."""

import os

from backend.asr_quality import apply_quality_profile, is_accuracy_mode, is_high_quality_profile


def test_balanced_profile_applies_chunk_defaults(monkeypatch):
    for key in os.environ:
        if key.startswith(("ASR_", "PATHUMMA_", "TYPHOON_", "DIARIZATION_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("backend.services.asr_local._cuda_vram_mb", lambda: 24 * 1024)
    monkeypatch.setenv("ASR_QUALITY_PROFILE", "balanced")
    apply_quality_profile()
    assert os.environ["ASR_CHUNK_LENGTH_S"] == "30"
    assert os.environ["ASR_8GB_MAX_CHUNK_LENGTH_S"] == "40"
    assert not is_high_quality_profile()


def test_high_profile_applies_300s_defaults(monkeypatch):
    for key in os.environ:
        if key.startswith(("ASR_", "PATHUMMA_", "TYPHOON_", "DIARIZATION_", "AUDIO_ENHANCE")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("backend.services.asr_local._cuda_vram_mb", lambda: 24 * 1024)
    monkeypatch.setenv("ASR_QUALITY_PROFILE", "high")
    apply_quality_profile()
    assert os.environ["ASR_CHUNK_LENGTH_S"] == "300"
    assert os.environ["ASR_8GB_MAX_CHUNK_LENGTH_S"] == "90"
    assert os.environ["ASR_8GB_MAX_BATCH_SIZE"] == "4"
    assert os.environ["ASR_MIN_CHUNKED_DURATION_S"] == "120"
    assert os.environ["ASR_LONG_FORM_WINDOW_S"] == "300"
    assert os.environ["PATHUMMA_WORD_TIMESTAMPS_ON_8GB"] == "true"
    assert os.environ["DIARIZATION_MULTI_SAMPLE"] == "true"
    assert os.environ["DIARIZATION_MULTI_SAMPLE_PASSES"] == "6"
    assert os.environ["DIARIZATION_SEGMENT_S"] == "360"
    assert os.environ["DIARIZATION_SEGMENT_OVERLAP_S"] == "90"
    assert os.environ["DIARIZATION_REFINE_AFTER_SEGMENTED"] == "true"
    assert os.environ["DIARIZATION_PREPROCESS_SR"] == "44100"
    assert is_high_quality_profile()
    assert is_accuracy_mode()


def test_low_vram_co_resident_keeps_auto_diarization(monkeypatch):
    monkeypatch.setenv("ASR_QUALITY_PROFILE", "high")
    monkeypatch.setenv("DIARIZATION_GPU_CO_RESIDENT", "true")
    monkeypatch.setattr("backend.services.asr_local._cuda_vram_mb", lambda: 8192)
    monkeypatch.setenv("ASR_HARD_MEMORY_SAFE", "true")
    apply_quality_profile()
    assert os.environ["DIARIZATION_DEVICE"] == "auto"
    assert os.environ["ASR_UNLOAD_FOR_DIARIZATION"] == "false"


def test_low_vram_overrides_on_8gb_class(monkeypatch):
    monkeypatch.setenv("ASR_QUALITY_PROFILE", "high")
    monkeypatch.setenv("PATHUMMA_WORD_TIMESTAMPS_ON_8GB", "true")
    monkeypatch.setenv("ASR_8GB_MAX_CHUNK_LENGTH_S", "120")
    monkeypatch.setattr("backend.services.asr_local._cuda_vram_mb", lambda: 8192)
    monkeypatch.setenv("ASR_HARD_MEMORY_SAFE", "true")
    apply_quality_profile()
    assert os.environ["PATHUMMA_WORD_TIMESTAMPS_ON_8GB"] == "false"
    assert os.environ["ASR_8GB_MAX_CHUNK_LENGTH_S"] == "60"
    assert os.environ["DIARIZATION_PRELOAD_DEVICE"] == "cpu"
    assert os.environ["DIARIZATION_DEVICE"] == "cpu"
    assert os.environ["DIARIZATION_ALLOW_8GB_CUDA"] == "false"
    assert os.environ["ASR_UNLOAD_FOR_DIARIZATION"] == "false"
    assert os.environ["DIARIZATION_MULTI_SAMPLE_PASSES"] == "6"
    assert os.environ["DIARIZATION_REFINE_AFTER_SEGMENTED"] == "true"


def test_high_profile_large_chunks_override(monkeypatch):
    for key in os.environ:
        if key.startswith(("ASR_", "PATHUMMA_", "TYPHOON_", "DIARIZATION_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("backend.services.asr_local._cuda_vram_mb", lambda: 24 * 1024)
    monkeypatch.setenv("ASR_QUALITY_PROFILE", "high")
    monkeypatch.setenv("ASR_8GB_ALLOW_LARGE_CHUNKS", "true")
    apply_quality_profile()
    assert os.environ["ASR_8GB_MAX_CHUNK_LENGTH_S"] == "300"


def test_explicit_env_wins_over_profile(monkeypatch):
    monkeypatch.setenv("ASR_QUALITY_PROFILE", "high")
    monkeypatch.setenv("ASR_CHUNK_LENGTH_S", "45")
    apply_quality_profile()
    assert os.environ["ASR_CHUNK_LENGTH_S"] == "45"


def test_long_form_window_honors_quality_profile(monkeypatch):
    from engines.pathumma_asr import _long_form_window_s

    monkeypatch.setenv("ASR_QUALITY_PROFILE", "high")
    monkeypatch.setenv("ASR_LONG_FORM_WINDOW_S", "300")
    apply_quality_profile()
    assert _long_form_window_s(5400.0) == 300
