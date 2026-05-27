"""Hardware detection and backend selection policy for local inference."""

# pylint: disable=duplicate-code,import-outside-toplevel

from __future__ import annotations

import logging
import os
import platform
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
MIN_SYSTEM_RAM_MB = _env_int("MIN_SYSTEM_RAM_MB", 16000)
_hw_cache: list[dict] = []


def _check_system_ram() -> dict:
    """Detect total system RAM for CPU-mode eligibility warning."""
    result = {"system_ram_mb": 0, "system_ram_ok": False}
    try:
        import psutil

        total = psutil.virtual_memory().total
        result["system_ram_mb"] = int(total // (1024 * 1024))
        result["system_ram_ok"] = result["system_ram_mb"] >= MIN_SYSTEM_RAM_MB
    except (ImportError, OSError, AttributeError) as exc:
        logger.debug("psutil RAM probe failed: %s", exc)
    return result


def _check_torch() -> dict:
    """Probe PyTorch for CUDA (NVIDIA) and HIP (AMD ROCm) acceleration."""
    result = {
        "torch_version": None,
        "cuda": False,
        "cuda_device_count": 0,
        "cuda_device_name": "",
        "cuda_vram_mb": 0,
        "rocm": False,
        "rocm_version": None,
    }
    try:
        import torch

        result["torch_version"] = torch.__version__
        # ROCm PyTorch exposes the same torch.cuda.* API but reports torch.version.hip.
        hip_version = getattr(torch.version, "hip", None)
        if hip_version:
            result["rocm"] = True
            result["rocm_version"] = hip_version
        if torch.cuda.is_available():
            result["cuda"] = not result["rocm"]  # NVIDIA CUDA only when HIP absent
            result["cuda_device_count"] = torch.cuda.device_count()
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            result["cuda_vram_mb"] = int(total // (1024 * 1024))
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        logger.debug("Torch/CUDA probe failed: %s", exc)
    return result


def _check_directml() -> dict:
    """Probe torch-directml for Windows AMD/Intel GPU acceleration."""
    result = {"directml": False, "directml_device_count": 0}
    if os.name != "nt":
        return result
    try:
        import torch_directml  # type: ignore[import-not-found]

        count = torch_directml.device_count()
        if count > 0:
            result["directml"] = True
            result["directml_device_count"] = int(count)
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        logger.debug("torch-directml probe failed: %s", exc)
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


def _check_platform() -> dict:
    """Detect CPU architecture for ARM/x86 policy hints."""
    machine = platform.machine().lower()
    return {
        "cpu_arch": machine,
        "is_arm": machine in {"aarch64", "arm64", "armv7l", "armv8", "armv8l"},
        "is_x86_64": machine in {"x86_64", "amd64"},
    }


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
        selected = "AUTO", "OpenVINO AUTO selected by OV_DEVICE."
    elif env_device and env_device in devices:
        selected = env_device, f"OpenVINO {env_device} selected by OV_DEVICE."
    elif ov_info["npu"]:
        selected = "NPU", "OpenVINO NPU detected and selected."
    elif ov_info["gpu"]:
        selected = (
            next(device for device in devices if device.startswith("GPU")),
            "OpenVINO GPU detected and selected.",
        )
    elif torch_info["cuda"]:
        selected = (
            "CPU",
            f"NVIDIA GPU has less than {MIN_NVIDIA_VRAM_MB} MB VRAM; "
            "using OpenVINO CPU fallback.",
        )
    elif has_amd_gpu:
        selected = "CPU", "AMD GPU detected; using OpenVINO CPU fallback in v1."
    else:
        selected = "CPU", "OpenVINO CPU selected."
    return selected


def _select_backend(
    torch_info: dict,
    ov_info: dict,
    dml_info: dict,
    has_amd_gpu: bool,
) -> tuple[str, str, str, bool]:
    """Select acceleration backend with parity across CUDA / OpenVINO / DirectML / ROCm / CPU.

    Preference order when multiple backends are available (overridable via
    APP_FORCE_BACKEND in {cuda, openvino, directml, rocm, cpu}):
      1. NVIDIA CUDA (if VRAM >= MIN_NVIDIA_VRAM_MB)
      2. AMD ROCm (Linux PyTorch HIP build)
      3. OpenVINO NPU / GPU (Intel Core Ultra, Arc, integrated GPU)
      4. DirectML (Windows AMD/Intel GPU)
      5. CPU fallback
    """
    nvidia_vram_ok = bool(
        torch_info["cuda"] and torch_info["cuda_vram_mb"] >= MIN_NVIDIA_VRAM_MB
    )
    force = os.getenv("APP_FORCE_BACKEND", "").strip().lower()

    if force == "cuda" and torch_info["cuda"]:
        return "cuda", "cuda", "NVIDIA CUDA forced by APP_FORCE_BACKEND.", nvidia_vram_ok
    if force == "rocm" and torch_info["rocm"]:
        return "rocm", "cuda", "AMD ROCm forced by APP_FORCE_BACKEND.", nvidia_vram_ok
    if force == "openvino" and ov_info["openvino_version"]:
        device, reason = _select_openvino_device(torch_info, ov_info, has_amd_gpu)
        return "openvino", device, f"{reason} (forced)", nvidia_vram_ok
    if force == "directml" and dml_info["directml"]:
        return "directml", "directml", "DirectML forced by APP_FORCE_BACKEND.", nvidia_vram_ok
    if force == "cpu":
        return "cpu", "cpu", "CPU forced by APP_FORCE_BACKEND.", nvidia_vram_ok

    if torch_info["cuda"] and nvidia_vram_ok:
        return (
            "cuda",
            "cuda",
            f"NVIDIA CUDA selected; VRAM meets the {MIN_NVIDIA_VRAM_MB} MB minimum.",
            nvidia_vram_ok,
        )
    if torch_info["rocm"]:
        return (
            "rocm",
            "cuda",  # ROCm uses torch.cuda.* API; engines treat it like CUDA
            f"AMD ROCm selected (HIP {torch_info['rocm_version']}); using PyTorch HIP backend.",
            nvidia_vram_ok,
        )
    plat = _check_platform()
    if plat["is_arm"] and ov_info["openvino_version"]:
        device, reason = _select_openvino_device(torch_info, ov_info, has_amd_gpu)
        return (
            "openvino",
            device,
            f"{reason} ARM64 host prefers OpenVINO.",
            nvidia_vram_ok,
        )
    if ov_info["openvino_version"] and (ov_info["npu"] or ov_info["gpu"]):
        device, reason = _select_openvino_device(torch_info, ov_info, has_amd_gpu)
        return "openvino", device, reason, nvidia_vram_ok
    if dml_info["directml"]:
        return (
            "directml",
            "directml",
            "DirectML selected; using Windows AMD/Intel GPU via torch-directml.",
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
            "no other accelerator available; using CPU.",
            nvidia_vram_ok,
        )
    if has_amd_gpu:
        return (
            "cpu",
            "cpu",
            "AMD GPU detected but no DirectML/ROCm/OpenVINO backend available; using CPU.",
            nvidia_vram_ok,
        )
    return "cpu", "cpu", "No GPU acceleration detected; using CPU.", nvidia_vram_ok


def detect_hardware(refresh: bool = False) -> dict:
    """Detect hardware and apply the app backend policy."""
    if _hw_cache and not refresh:
        return _hw_cache[0]

    torch_info = _check_torch()
    ov_info = _check_openvino()
    dml_info = _check_directml()
    ram_info = _check_system_ram()
    plat_info = _check_platform()
    ffmpeg_path = shutil.which("ffmpeg")
    has_amd_gpu = _detect_amd_gpu()

    selected_backend, selected_device, backend_reason, nvidia_vram_ok = _select_backend(
        torch_info,
        ov_info,
        dml_info,
        has_amd_gpu,
    )

    # Warn-only policy: append RAM warning to reason when CPU-bound and under-spec.
    if selected_backend == "cpu" and not ram_info["system_ram_ok"] and ram_info["system_ram_mb"]:
        backend_reason = (
            f"{backend_reason} WARNING: system RAM {ram_info['system_ram_mb']} MB "
            f"is below recommended {MIN_SYSTEM_RAM_MB} MB; CPU mode may be slow or OOM."
        )

    info = {
        **torch_info,
        **ov_info,
        **dml_info,
        **ram_info,
        **plat_info,
        "ffmpeg": ffmpeg_path,
        "amd_gpu": has_amd_gpu,
        "nvidia_vram_ok": nvidia_vram_ok,
        "min_nvidia_vram_mb": MIN_NVIDIA_VRAM_MB,
        "min_system_ram_mb": MIN_SYSTEM_RAM_MB,
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
                    "- **VRAM policy:** strict low-VRAM mode; "
                    "preloaded ASR models retained for reuse, "
                    "CPU diarization by default"
                )
    if hw.get("rocm"):
        lines.append(
            f"- **AMD ROCm:** detected (HIP {hw.get('rocm_version', 'unknown')})"
        )
    if hw.get("directml"):
        lines.append(
            f"- **DirectML:** detected ({hw.get('directml_device_count', 0)} device(s))"
        )
    if hw["amd_gpu"]:
        lines.append("- **AMD GPU:** detected via system probe")
    lines.extend([
        f"- **OpenVINO:** {hw['openvino_version'] or 'not installed'}",
        f"- **OpenVINO devices:** {', '.join(hw['available_devices']) or 'none'}",
        f"- **FFmpeg:** {'available' if hw['ffmpeg'] else 'NOT FOUND'}",
    ])
    arch = hw.get("cpu_arch") or "unknown"
    lines.append(f"- **CPU arch:** {arch}" + (" (ARM)" if hw.get("is_arm") else ""))
    ram_mb = hw.get("system_ram_mb") or 0
    if ram_mb:
        ram_note = "OK" if hw.get("system_ram_ok") else f"below recommended {hw.get('min_system_ram_mb', MIN_SYSTEM_RAM_MB)} MB"
        lines.append(f"- **System RAM:** {ram_mb} MB ({ram_note})")
    return "\n".join(lines)
