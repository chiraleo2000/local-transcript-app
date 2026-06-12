"""Tests for GPU co-resident model policy."""

import os

from backend.services.asr_local import (
    model_is_loaded,
    models_resident_on_gpu,
    requires_sequential_gpu_models,
    should_clear_models_after_job,
    should_unload_asr_for_diarization,
    should_unload_on_cancel,
)


def test_models_resident_on_gpu(monkeypatch):
    monkeypatch.setenv("ASR_KEEP_PRELOADED", "true")
    monkeypatch.setenv("ASR_CLEAR_VRAM_AFTER_JOB", "false")
    monkeypatch.setattr("backend.services.asr_local._cuda_vram_mb", lambda: 24 * 1024)
    assert models_resident_on_gpu()
    assert not should_clear_models_after_job()
    assert not should_unload_asr_for_diarization()


def test_sequential_gpu_policy_on_8gb_class(monkeypatch):
    monkeypatch.setenv("ASR_HARD_MEMORY_SAFE", "true")
    monkeypatch.setenv("ASR_KEEP_PRELOADED", "true")
    monkeypatch.setenv("ASR_CLEAR_VRAM_AFTER_JOB", "false")
    monkeypatch.setenv("ASR_UNLOAD_FOR_DIARIZATION", "false")
    monkeypatch.setenv("DIARIZATION_DEVICE", "cpu")
    monkeypatch.setattr("backend.services.asr_local._cuda_vram_mb", lambda: 8192)
    monkeypatch.setattr(
        "backend.services.asr_local.diarization_inference_uses_cuda",
        lambda: False,
    )
    assert requires_sequential_gpu_models()
    assert models_resident_on_gpu()
    assert not should_unload_asr_for_diarization()


def test_models_not_resident_when_clear_after_job(monkeypatch):
    monkeypatch.setenv("ASR_KEEP_PRELOADED", "true")
    monkeypatch.setenv("ASR_CLEAR_VRAM_AFTER_JOB", "true")
    monkeypatch.setattr("backend.services.asr_local._cuda_vram_mb", lambda: 24 * 1024)
    assert not models_resident_on_gpu()


def test_co_resident_never_unloads_asr_for_diarization(monkeypatch):
    monkeypatch.setenv("DIARIZATION_GPU_CO_RESIDENT", "true")
    monkeypatch.setattr(
        "backend.services.asr_local.diarization_inference_uses_cuda",
        lambda: True,
    )
    assert not should_unload_asr_for_diarization()


def test_model_is_loaded_false_when_empty():
    assert not model_is_loaded("Typhoon Whisper")


def test_cancel_keeps_models_by_default(monkeypatch):
    monkeypatch.delenv("ASR_UNLOAD_ON_CANCEL", raising=False)
    monkeypatch.setenv("ASR_KEEP_PRELOADED", "true")
    monkeypatch.setattr("backend.services.asr_local._cuda_vram_mb", lambda: 8192)
    assert not should_unload_on_cancel()


def test_cancel_unload_only_when_explicit(monkeypatch):
    monkeypatch.setenv("ASR_UNLOAD_ON_CANCEL", "true")
    assert should_unload_on_cancel()
