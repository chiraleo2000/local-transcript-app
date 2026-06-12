"""Tests for shared Whisper runtime helpers."""

from engines.whisper_utils import effective_asr_batch_size, whisper_generate_kwargs


def test_whisper_generate_kwargs_omits_condition_on_previous_text(monkeypatch):
    monkeypatch.setenv("ASR_CONDITION_ON_PREVIOUS_TEXT", "true")
    monkeypatch.setenv("ASR_SUPPRESS_HALLUCINATIONS", "true")
    kwargs = whisper_generate_kwargs("thai")
    assert "condition_on_previous_text" not in kwargs
    assert "language" in kwargs
    assert "compression_ratio_threshold" in kwargs

def test_effective_batch_scales_down_for_long_audio(monkeypatch):
    monkeypatch.setenv("ASR_HARD_MEMORY_SAFE", "false")
    assert effective_asr_batch_size(8, 120.0) == 8
    assert effective_asr_batch_size(8, 600.0) == 2
    assert effective_asr_batch_size(8, 3600.0) == 1


def test_effective_batch_strict_mode_forces_one(monkeypatch):
    monkeypatch.setenv("ASR_HARD_MEMORY_SAFE", "true")
    monkeypatch.setenv("ASR_8GB_MAX_BATCH_SIZE", "1")
    assert effective_asr_batch_size(8, 120.0) == 1
    assert effective_asr_batch_size(4, 60.0) == 1


def test_effective_batch_strict_mode_allows_capped_batch(monkeypatch):
    monkeypatch.setenv("ASR_HARD_MEMORY_SAFE", "true")
    monkeypatch.setenv("ASR_8GB_MAX_BATCH_SIZE", "4")
    monkeypatch.setenv("ASR_BATCH_DURATION_CAP", "true")
    assert effective_asr_batch_size(4, 30.0) == 4
    assert effective_asr_batch_size(4, 60.0) == 4
    assert effective_asr_batch_size(4, 120.0) == 2


def test_effective_batch_windowed_uses_full_cap_not_total_duration(monkeypatch):
    monkeypatch.setenv("ASR_HARD_MEMORY_SAFE", "true")
    monkeypatch.setenv("ASR_8GB_MAX_BATCH_SIZE", "4")
    assert effective_asr_batch_size(4, 7200.0) <= 2
    assert effective_asr_batch_size(4, 300.0, windowed=True) == 4
