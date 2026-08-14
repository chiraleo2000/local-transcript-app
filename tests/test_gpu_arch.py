"""Tesla P4 / Pascal GPU profile tests (no GPU required)."""

from __future__ import annotations

import os

from backend.gpu_arch import (
    PASCAL_P4_SPEED_ENV,
    apply_pascal_p4_overrides,
    is_pascal_speed_gpu,
    is_tesla_p4_name,
    pascal_speed_gpu_active,
    recommended_cuda_stack,
)
from engines.whisper_utils import asr_cuda_dtypes_to_try, resolve_asr_cuda_dtype


class _FakeDType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


class _FakeTorch:
    float16 = _FakeDType("float16")
    float32 = _FakeDType("float32")
    bfloat16 = _FakeDType("bfloat16")


def test_detects_tesla_p4_name():
    assert is_tesla_p4_name("Tesla P4")
    assert is_tesla_p4_name("NVIDIA Tesla P4")
    assert not is_tesla_p4_name("NVIDIA GeForce RTX 4060 Laptop GPU")
    assert not is_tesla_p4_name("NVIDIA GeForce RTX 4090")


def test_pascal_from_compute_capability():
    assert is_pascal_speed_gpu("Tesla P4", 6)
    assert is_pascal_speed_gpu("Unknown", 6)
    assert not is_pascal_speed_gpu("NVIDIA GeForce RTX 4060 Laptop GPU", 8)


def test_profile_off_skips_even_on_p4(monkeypatch):
    monkeypatch.setenv("ASR_GPU_PROFILE", "off")
    assert pascal_speed_gpu_active() is False


def test_profile_p4_forces_without_gpu(monkeypatch):
    monkeypatch.setenv("ASR_GPU_PROFILE", "p4")
    assert pascal_speed_gpu_active() is True
    assert recommended_cuda_stack() == "cuda124"


def test_apply_overrides_on_forced_p4(monkeypatch):
    for key in PASCAL_P4_SPEED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ASR_GPU_PROFILE", "p4")
    monkeypatch.setenv("ASR_NUM_BEAMS", "5")
    applied = apply_pascal_p4_overrides()
    assert any(item.startswith("ASR_NUM_BEAMS=2") for item in applied)
    assert os.getenv("ASR_CUDA_DTYPE") == "float32"
    assert os.getenv("ASR_ATTENTION_IMPLEMENTATION") == "eager"
    assert os.getenv("ASR_TEMPERATURE") == PASCAL_P4_SPEED_ENV["ASR_TEMPERATURE"]


def test_4060_keeps_float16(monkeypatch):
    monkeypatch.setenv("ASR_GPU_PROFILE", "off")
    monkeypatch.delenv("ASR_CUDA_DTYPE", raising=False)
    assert resolve_asr_cuda_dtype(_FakeTorch) is _FakeTorch.float16
    assert asr_cuda_dtypes_to_try(_FakeTorch) == (_FakeTorch.float16,)


def test_p4_prefers_float32_then_float16(monkeypatch):
    monkeypatch.setenv("ASR_GPU_PROFILE", "p4")
    monkeypatch.setenv("ASR_CUDA_DTYPE", "float32")
    assert resolve_asr_cuda_dtype(_FakeTorch) is _FakeTorch.float32
    assert asr_cuda_dtypes_to_try(_FakeTorch) == (
        _FakeTorch.float32,
        _FakeTorch.float16,
    )


def test_apply_quality_profile_does_not_slow_4060(monkeypatch):
    monkeypatch.setenv("ASR_GPU_PROFILE", "off")
    monkeypatch.setenv("ASR_NUM_BEAMS", "5")
    assert apply_pascal_p4_overrides() == []
    assert os.getenv("ASR_NUM_BEAMS") == "5"
