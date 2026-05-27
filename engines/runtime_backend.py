"""Map hardware_policy results to ASR pipeline loaders."""

from __future__ import annotations


def uses_pytorch_cuda_pipeline(hw: dict) -> bool:
    """True when ASR should use the PyTorch CUDA/HIP (NVIDIA or AMD ROCm) path."""
    backend = (hw.get("backend") or "").lower()
    device = (hw.get("selected_device") or "").lower()
    return backend in {"cuda", "rocm"} or device == "cuda"


def uses_openvino_pipeline(hw: dict) -> bool:
    """True when ASR should use the OpenVINO IR export/load path."""
    return (hw.get("backend") or "").lower() == "openvino"
