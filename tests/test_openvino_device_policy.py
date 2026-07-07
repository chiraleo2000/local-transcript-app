"""OpenVINO device selection policy tests."""

from __future__ import annotations

import os
from unittest.mock import patch

from backend.services import hardware_policy as hp


def _ov_info(devices: list[str]) -> dict:
    return {
        "available_devices": devices,
        "gpu": any(d.startswith("GPU") for d in devices),
        "npu": "NPU" in devices,
    }


def test_gpu_first_when_gpu_visible():
    with patch.dict(os.environ, {"OV_DEVICE": "GPU"}, clear=False):
        device, reason = hp._select_openvino_device(
            {"cuda": False},
            _ov_info(["CPU", "GPU.0"]),
            False,
        )
    assert device == "GPU.0"
    assert "OV_DEVICE=GPU" in reason


def test_auto_alias_prefers_gpu():
    with patch.dict(os.environ, {"OV_DEVICE": "AUTO"}, clear=False):
        device, reason = hp._select_openvino_device(
            {"cuda": False},
            _ov_info(["CPU", "GPU"]),
            False,
        )
    assert device == "GPU"
    assert "GPU-first" in reason


def test_gpu_requested_but_only_cpu_visible():
    with patch.dict(os.environ, {"OV_DEVICE": "GPU"}, clear=False):
        device, reason = hp._select_openvino_device(
            {"cuda": False},
            _ov_info(["CPU"]),
            False,
        )
    assert device == "CPU"
    assert "no OpenVINO GPU visible" in reason
