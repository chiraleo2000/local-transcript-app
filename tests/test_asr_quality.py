"""Unit tests for ASR quality-profile env handling (no GPU required)."""

from __future__ import annotations

import importlib

import pytest

asr_quality = importlib.import_module("backend.asr_quality")


@pytest.fixture(autouse=True)
def _restore_fraction_env(monkeypatch):
    # monkeypatch auto-reverts os.environ changes after each test.
    yield


class TestClampCudaMemoryFraction:
    """VRAM fraction is capped at 0.92 for production GPUs."""

    def test_leaves_value_below_cap_untouched(self, monkeypatch):
        monkeypatch.setenv("ASR_CUDA_MEMORY_FRACTION", "0.80")
        forced: dict[str, str] = {}
        asr_quality._clamp_cuda_memory_fraction(forced)
        assert "ASR_CUDA_MEMORY_FRACTION" not in forced

    def test_defaults_when_unset(self, monkeypatch):
        monkeypatch.delenv("ASR_CUDA_MEMORY_FRACTION", raising=False)
        forced: dict[str, str] = {}
        asr_quality._clamp_cuda_memory_fraction(forced)
        assert forced["ASR_CUDA_MEMORY_FRACTION"] == "0.92"

    def test_defaults_on_invalid_value(self, monkeypatch):
        monkeypatch.setenv("ASR_CUDA_MEMORY_FRACTION", "not-a-number")
        forced: dict[str, str] = {}
        asr_quality._clamp_cuda_memory_fraction(forced)
        assert forced["ASR_CUDA_MEMORY_FRACTION"] == "0.92"

    def test_clamps_values_above_cap(self, monkeypatch):
        monkeypatch.setenv("ASR_CUDA_MEMORY_FRACTION", "0.95")
        forced: dict[str, str] = {}
        asr_quality._clamp_cuda_memory_fraction(forced)
        assert forced["ASR_CUDA_MEMORY_FRACTION"] == "0.92"

    def test_leaves_value_at_cap_untouched(self, monkeypatch):
        monkeypatch.setenv("ASR_CUDA_MEMORY_FRACTION", "0.92")
        forced: dict[str, str] = {}
        asr_quality._clamp_cuda_memory_fraction(forced)
        assert "ASR_CUDA_MEMORY_FRACTION" not in forced


class TestEnterpriseDockerEnvDrift:
    """Critical compose keys must match ENTERPRISE_DOCKER_ENV."""

    _CRITICAL_KEYS = {
        "ASR_CUDA_MEMORY_FRACTION": "0.92",
        "ASR_CUDA_BATCH_SIZE": "1",
        "ASR_TARGET_SHORT_MAX_S": "600",
        "UI_MAX_CONCURRENT_JOBS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "APP_AUTO_DOWNLOAD_MISSING_MODELS": "false",
        "APP_REQUIRE_DIARIZATION_MODELS": "true",
    }

    def test_enterprise_docker_env_matches_compose_policy(self, monkeypatch):
        from backend.asr_quality import ENTERPRISE_DOCKER_ENV

        for key, expected in self._CRITICAL_KEYS.items():
            assert ENTERPRISE_DOCKER_ENV.get(key) == expected, (
                f"{key}: enterprise={ENTERPRISE_DOCKER_ENV.get(key)!r} expected {expected!r}"
            )


class TestSequentialStagingBatchEnv:
    def test_forces_batch_one_on_8gb_low_vram_cap(self, monkeypatch):
        monkeypatch.setenv("ASR_CUDA_BATCH_SIZE", "4")
        monkeypatch.setenv("ASR_8GB_MAX_BATCH_SIZE", "4")
        forced: dict[str, str] = {}
        asr_quality._cap_cuda_batch_in_forced(
            forced, multi_pass_diar=False, co_resident=False,
        )
        assert forced["ASR_CUDA_BATCH_SIZE"] == "1"
        assert forced["ASR_8GB_MAX_BATCH_SIZE"] == "1"

    def test_forces_batch_one_when_co_resident(self, monkeypatch):
        monkeypatch.setenv("ASR_CUDA_BATCH_SIZE", "4")
        values = asr_quality._sequential_staging_batch_env(
            co_resident=True, multi_pass_diar=False,
        )
        assert values["ASR_CUDA_BATCH_SIZE"] == "1"


class TestEnhanceProfileAdaptive:
    def test_skips_profile_when_adaptive_disabled(self, monkeypatch):
        import os

        monkeypatch.setenv("AUDIO_ENHANCE_ADAPTIVE", "false")
        monkeypatch.setenv("AUDIO_ENHANCE_WHEN_DIARIZATION", "false")
        monkeypatch.setenv("AUDIO_ENHANCE_ASR_ONLY", "true")
        applied = asr_quality.apply_enhance_profile(220.0, 4)
        assert applied == []
        assert os.getenv("AUDIO_ENHANCE_WHEN_DIARIZATION") == "false"
        assert os.getenv("AUDIO_ENHANCE_ASR_ONLY") == "true"
