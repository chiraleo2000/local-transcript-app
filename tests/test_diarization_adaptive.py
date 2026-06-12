"""Tests for adaptive pyannote diarization parameter helpers."""

import pytest

from engines.diarization import (
    _accuracy_mode_params,
    _adaptive_pipeline_params,
    _build_diarize_kwargs,
    _long_audio_adaptive_params,
    _max_speaker_cap_params,
    _retry_pipeline_params,
)


def test_adaptive_params_short_clip_three_speakers():
    params = _adaptive_pipeline_params(45.0, max_speakers=3)
    assert params is not None
    assert params["clustering"]["min_cluster_size"] == 3
    assert params["clustering"]["threshold"] == pytest.approx(0.55)
    assert params["segmentation"]["threshold"] == pytest.approx(0.42)


def test_adaptive_params_skipped_for_long_audio():
    assert _adaptive_pipeline_params(120.0, max_speakers=3) is None


def test_adaptive_params_very_short_adds_min_duration_off():
    params = _adaptive_pipeline_params(20.0, max_speakers=2)
    assert params is not None
    assert params["segmentation"]["min_duration_off"] == pytest.approx(0.05)


def test_build_diarize_kwargs_respects_max_only(monkeypatch):
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "false")
    monkeypatch.setenv("ASR_QUALITY_PROFILE", "balanced")
    kwargs = _build_diarize_kwargs(0, max_speakers=3, audio_duration_s=60.0)
    assert kwargs == {"max_speakers": 3}
    assert "min_speakers" not in kwargs


def test_build_diarize_kwargs_no_forced_min_speakers(monkeypatch):
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "false")
    monkeypatch.setenv("ASR_QUALITY_PROFILE", "balanced")
    kwargs = _build_diarize_kwargs(0, max_speakers=3, audio_duration_s=180.0)
    assert "min_speakers" not in kwargs


def test_build_diarize_kwargs_min_speakers_in_accuracy_mode(monkeypatch):
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
    kwargs = _build_diarize_kwargs(0, max_speakers=3, audio_duration_s=180.0)
    assert kwargs == {"max_speakers": 3, "min_speakers": 2}


def test_retry_params_loosen_clustering():
    base = _adaptive_pipeline_params(30.0, max_speakers=3)
    retry = _retry_pipeline_params(base)
    assert retry["clustering"]["min_cluster_size"] == 2
    assert retry["clustering"]["threshold"] < base["clustering"]["threshold"]


def test_accuracy_mode_params_use_lower_clustering(monkeypatch):
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
    params = _accuracy_mode_params(max_speakers=3, audio_duration_s=300.0)
    assert params["clustering"]["threshold"] <= 0.48
    assert params["segmentation"]["min_duration_off"] == pytest.approx(0.03)


def test_long_audio_adaptive_params_for_meetings(monkeypatch):
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
    params = _long_audio_adaptive_params(max_speakers=3, audio_duration_s=240.0)
    assert params is not None
    assert params["clustering"]["threshold"] <= 0.50


def test_cap_params_in_accuracy_mode_do_not_merge_speakers(monkeypatch):
    monkeypatch.setenv("DIARIZATION_ACCURACY_MODE", "true")
    params = _max_speaker_cap_params(max_speakers=3, audio_duration_s=300.0)
    assert params["clustering"]["threshold"] <= 0.48

