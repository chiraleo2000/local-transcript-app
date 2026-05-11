"""Hardware detection and backend selection policy for local inference."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import os
import shutil
import subprocess

from backend.storage import update_config


logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        logger.warning("Invalid %s=%r; using %d.", name, value, default)
        return default


MIN_NVIDIA_VRAM_MB = _env_int("MIN_NVIDIA_VRAM_MB", 6000)
_hw_cache: list[dict] = []


def _check_torch() -> dict:
    result = {
        "torch_version": None,
        "cuda": False,
        "cuda_device_count": 0,
        "cuda_device_name": "",
        "cuda_vram_mb": 0,
    }
    try:
        import torch

        result["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            result["cuda"] = True
            result["cuda_device_count"] = torch.cuda.device_count()
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            result["cuda_vram_mb"] = int(total // (1024 * 1024))
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        logger.debug("Torch/CUDA probe failed: %s", exc)
    return result


def _check_openvino() -> dict:
    result = {
        "openvino_version": None,
        "available_devices": [],
        "cpu": False,
        "gpu": False,
        "npu": False,
    }
    try:
        from openvino import Core, get_version

        devices = Core().available_devices
        result.update({
            "openvino_version": get_version(),
            "available_devices": devices,
            "cpu": "CPU" in devices,
            "gpu": any(device.startswith("GPU") for device in devices),
            "npu": "NPU" in devices,
        })
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        logger.debug("OpenVINO probe failed: %s", exc)
    return result


def _detect_amd_gpu() -> bool:
    if os.name != "nt":
        return False
    try:
        result = subprocess.run(
            [
                "wmic",
                "path",
                "win32_VideoController",
                "get",
                "name",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return "amd" in result.stdout.lower() or "radeon" in result.stdout.lower()
    except (OSError, subprocess.SubprocessError):
        return False


def _select_openvino_device(
    torch_info: dict,
    ov_info: dict,
    has_amd_gpu: bool,
) -> tuple[str, str]:
    devices = ov_info["available_devices"]
    env_device = os.getenv("OV_DEVICE", "").upper()
    if env_device == "AUTO":
        return "AUTO", "OpenVINO AUTO selected by OV_DEVICE."
    if env_device and env_device in devices:
        return env_device, f"OpenVINO {env_device} selected by OV_DEVICE."
    if ov_info["npu"]:
        return "NPU", "OpenVINO NPU detected and selected."
    if ov_info["gpu"]:
        return (
            next(device for device in devices if device.startswith("GPU")),
            "OpenVINO GPU detected and selected.",
        )
    if torch_info["cuda"]:
        return (
            "CPU",
            f"NVIDIA GPU has less than {MIN_NVIDIA_VRAM_MB} MB VRAM; "
            "using OpenVINO CPU fallback.",
        )
    if has_amd_gpu:
        return "CPU", "AMD GPU detected; using OpenVINO CPU fallback in v1."
    return "CPU", "OpenVINO CPU selected."


def _select_backend(
    torch_info: dict,
    ov_info: dict,
    has_amd_gpu: bool,
) -> tuple[str, str, str, bool]:
    nvidia_vram_ok = bool(
        torch_info["cuda"] and torch_info["cuda_vram_mb"] >= MIN_NVIDIA_VRAM_MB
    )
    if torch_info["cuda"] and nvidia_vram_ok:
        return (
            "cuda",
            "cuda",
            f"NVIDIA CUDA selected; VRAM meets the {MIN_NVIDIA_VRAM_MB} MB minimum.",
            nvidia_vram_ok,
        )
    if ov_info["openvino_version"]:
        device, reason = _select_openvino_device(torch_info, ov_info, has_amd_gpu)
        return "openvino", device, reason, nvidia_vram_ok
    if torch_info["cuda"]:
        return (
            "cpu",
            "cpu",
            f"NVIDIA GPU has less than {MIN_NVIDIA_VRAM_MB} MB VRAM and "
            "OpenVINO is unavailable; using CPU.",
            nvidia_vram_ok,
        )
    if has_amd_gpu:
        return (
            "cpu",
            "cpu",
            "AMD GPU detected; no AMD acceleration path enabled in v1, using CPU.",
            nvidia_vram_ok,
        )
    return "cpu", "cpu", "No CUDA/OpenVINO acceleration detected; using CPU.", nvidia_vram_ok


def detect_hardware(refresh: bool = False) -> dict:
    """Detect hardware and apply the app backend policy."""
    if _hw_cache and not refresh:
        return _hw_cache[0]

    torch_info = _check_torch()
    ov_info = _check_openvino()
    ffmpeg_path = shutil.which("ffmpeg")
    has_amd_gpu = _detect_amd_gpu()

    selected_backend, selected_device, backend_reason, nvidia_vram_ok = _select_backend(
        torch_info,
        ov_info,
        has_amd_gpu,
    )

    info = {
        **torch_info,
        **ov_info,
        "ffmpeg": ffmpeg_path,
        "amd_gpu": has_amd_gpu,
        "nvidia_vram_ok": nvidia_vram_ok,
        "min_nvidia_vram_mb": MIN_NVIDIA_VRAM_MB,
        "backend": selected_backend,
        "selected_device": selected_device,
        "backend_reason": backend_reason,
    }

    _hw_cache.clear()
    _hw_cache.append(info)
    update_config(hardware=info)
    logger.info("Selected backend: %s/%s (%s)", selected_backend, selected_device, backend_reason)
    return info


def hardware_summary() -> str:
    """Return a Markdown summary of the selected hardware backend."""
    hw = detect_hardware()
    lines = [
        "### Hardware Status",
        f"- **Selected backend:** {hw['backend'].upper()} / {hw['selected_device']}",
        f"- **Reason:** {hw['backend_reason']}",
        f"- **PyTorch:** {hw['torch_version'] or 'not installed'}",
    ]
    if hw["cuda"]:
        lines.append(
            f"- **NVIDIA GPU:** {hw['cuda_device_name']} "
            f"({hw['cuda_vram_mb']} MB VRAM; minimum {hw['min_nvidia_vram_mb']} MB)"
        )
        try:
            class_limit = int(os.getenv("ASR_8GB_CLASS_MAX_MB", "9000"))
        except ValueError:
            class_limit = 9000
        if os.getenv("ASR_HARD_MEMORY_SAFE", "true").strip().lower() in {"1", "true", "yes", "on"}:
            if hw["cuda_vram_mb"] <= class_limit:
                lines.append(
                    "- **VRAM policy:** strict low-VRAM mode; one GPU ASR model at a time, "
                    "CPU diarization by default"
                )
    if hw["amd_gpu"]:
        lines.append("- **AMD GPU:** detected; CPU/OpenVINO fallback used in v1")
    lines.extend([
        f"- **OpenVINO:** {hw['openvino_version'] or 'not installed'}",
        f"- **OpenVINO devices:** {', '.join(hw['available_devices']) or 'none'}",
        f"- **FFmpeg:** {'available' if hw['ffmpeg'] else 'NOT FOUND'}",
    ])
    return "\n".join(lines)
