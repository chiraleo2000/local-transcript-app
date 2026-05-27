"""Tests for ASR pipeline backend selection helpers."""

from engines.runtime_backend import uses_openvino_pipeline, uses_pytorch_cuda_pipeline


def test_cuda_backend_uses_pytorch():
    hw = {"backend": "cuda", "selected_device": "cuda"}
    assert uses_pytorch_cuda_pipeline(hw)
    assert not uses_openvino_pipeline(hw)


def test_rocm_backend_uses_pytorch():
    hw = {"backend": "rocm", "selected_device": "cuda"}
    assert uses_pytorch_cuda_pipeline(hw)
    assert not uses_openvino_pipeline(hw)


def test_openvino_backend():
    hw = {"backend": "openvino", "selected_device": "GPU"}
    assert uses_openvino_pipeline(hw)
    assert not uses_pytorch_cuda_pipeline(hw)
