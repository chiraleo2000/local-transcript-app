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


# Minimum VRAM required to enable the CUDA backend in release/offline mode.
# 8GB class GPUs are the minimum supported NVIDIA target.
MIN_NVIDIA_VRAM_MB = _env_int("MIN_NVIDIA_VRAM_MB", 8192)
# Minimum host RAM for OpenVINO/CPU/DirectML deployments (Intel/AMD/ARM).
MIN_SYSTEM_RAM_MB = _env_int("MIN_SYSTEM_RAM_MB", 8192)
# Minimum logical CPU threads recommended (4 cores / 4 threads).
MIN_CPU_THREADS = _env_int("MIN_CPU_THREADS", 4)
_hw_cache: list[dict] = []


def _check_system_ram() -> dict:
    """Detect total system RAM for CPU-mode eligibility warning."""
    result = {"system_ram_mb": 0, "system_ram_ok": False}
    try:
        import psutil  # type: ignore[import-not-found,import-untyped]

        total = psutil.virtual_memory().total
        result["system_ram_mb"] = int(total // (1024 * 1024))
        result["system_ram_ok"] = result["system_ram_mb"] >= MIN_SYSTEM_RAM_MB
    except (ImportError, OSError, AttributeError) as exc:
        logger.debug("psutil RAM probe failed: %s", exc)
    return result


def _check_cpu_threads() -> dict:
    """Detect logical CPU thread count for minimum-host warnings."""
    detected = max(1, os.cpu_count() or 1)
    return {
        "cpu_threads": detected,
        "cpu_threads_ok": detected >= MIN_CPU_THREADS,
        "min_cpu_threads": MIN_CPU_THREADS,
    }


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
        import torch  # type: ignore[import-not-found]

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
        from openvino import Core, get_version  # type: ignore[import-not-found]

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


def _first_openvino_gpu(devices: list[str]) -> str | None:
    for device in devices:
        if device.startswith("GPU"):
            return device
    return None


def _openvino_gpu_first(
    torch_info: dict,
    ov_info: dict,
    has_amd_gpu: bool,
    *,
    reason_prefix: str = "",
) -> tuple[str, str]:
    """Prefer Intel GPU, then NPU, then CPU."""
    devices = ov_info["available_devices"]
    prefix = f"{reason_prefix} " if reason_prefix else ""
    if ov_info["gpu"]:
        gpu = _first_openvino_gpu(devices) or "GPU"
        return gpu, f"{prefix}OpenVINO GPU selected (GPU-first policy)."
    if ov_info["npu"]:
        return "NPU", f"{prefix}OpenVINO NPU selected (no GPU available)."
    if torch_info["cuda"]:
        return (
            "CPU",
            f"{prefix}NVIDIA GPU has less than {MIN_NVIDIA_VRAM_MB} MB VRAM; "
            "using OpenVINO CPU fallback.",
        )
    if has_amd_gpu:
        return (
            "CPU",
            f"{prefix}AMD GPU without OpenVINO GPU device; using OpenVINO CPU "
            "(prefer DirectML on Windows or ROCm on Linux when available).",
        )
    return "CPU", f"{prefix}OpenVINO CPU selected (no GPU/NPU available)."


def _select_openvino_device(
    torch_info: dict,
    ov_info: dict,
    has_amd_gpu: bool,
) -> tuple[str, str]:
    devices = ov_info["available_devices"]
    env_device = os.getenv("OV_DEVICE", "GPU").strip().upper()
    if not env_device or env_device == "AUTO":
        return _openvino_gpu_first(torch_info, ov_info, has_amd_gpu)
    if env_device == "GPU":
        if ov_info["gpu"]:
            gpu = _first_openvino_gpu(devices) or "GPU"
            return gpu, f"OpenVINO {gpu} selected by OV_DEVICE=GPU."
        return (
            "CPU",
            "OV_DEVICE=GPU but no OpenVINO GPU visible; using CPU.",
        )
    if env_device == "NPU":
        if ov_info["npu"]:
            return "NPU", "OpenVINO NPU selected by OV_DEVICE."
        return _openvino_gpu_first(
            torch_info,
            ov_info,
            has_amd_gpu,
            reason_prefix="OV_DEVICE=NPU but NPU unavailable;",
        )
    if env_device == "CPU":
        return "CPU", "OpenVINO CPU forced by OV_DEVICE."
    if env_device in devices:
        return env_device, f"OpenVINO {env_device} selected by OV_DEVICE."
    return _openvino_gpu_first(
        torch_info,
        ov_info,
        has_amd_gpu,
        reason_prefix=f"OV_DEVICE={env_device} not in {devices};",
    )


def _try_forced_backend(
    force: str,
    torch_info: dict,
    ov_info: dict,
    dml_info: dict,
    has_amd_gpu: bool,
) -> tuple[str, str, str] | None:
    """Return a forced backend selection when APP_FORCE_BACKEND is set and available."""
    if force == "cuda" and torch_info["cuda"]:
        return "cuda", "cuda", "NVIDIA CUDA forced by APP_FORCE_BACKEND."
    if force == "rocm" and torch_info["rocm"]:
        return "rocm", "cuda", "AMD ROCm forced by APP_FORCE_BACKEND."
    if force == "openvino" and ov_info["openvino_version"]:
        device, reason = _select_openvino_device(torch_info, ov_info, has_amd_gpu)
        return "openvino", device, f"{reason} (forced)"
    if force == "directml" and dml_info["directml"]:
        return "directml", "directml", "DirectML forced by APP_FORCE_BACKEND."
    if force == "cpu":
        return "cpu", "cpu", "CPU forced by APP_FORCE_BACKEND."
    return None


def _auto_select_backend(
    torch_info: dict,
    ov_info: dict,
    dml_info: dict,
    has_amd_gpu: bool,
    nvidia_vram_ok: bool,
) -> tuple[str, str, str]:
    """Pick backend by preference when no APP_FORCE_BACKEND override applies."""
    if torch_info["cuda"] and nvidia_vram_ok:
        return (
            "cuda",
            "cuda",
            f"NVIDIA CUDA selected; VRAM meets the {MIN_NVIDIA_VRAM_MB} MB minimum.",
        )
    if torch_info["rocm"]:
        return (
            "rocm",
            "cuda",  # ROCm uses torch.cuda.* API; engines treat it like CUDA
            f"AMD ROCm selected (HIP {torch_info['rocm_version']}); using PyTorch HIP backend.",
        )
    plat = _check_platform()
    if plat["is_arm"] and ov_info["openvino_version"]:
        device, reason = _select_openvino_device(torch_info, ov_info, has_amd_gpu)
        return "openvino", device, f"{reason} ARM64 host prefers OpenVINO."
    if ov_info["openvino_version"] and (ov_info["npu"] or ov_info["gpu"]):
        device, reason = _select_openvino_device(torch_info, ov_info, has_amd_gpu)
        return "openvino", device, reason
    if dml_info["directml"]:
        return (
            "directml",
            "directml",
            "DirectML selected; using Windows AMD/Intel GPU via torch-directml.",
        )
    if ov_info["openvino_version"]:
        device, reason = _select_openvino_device(torch_info, ov_info, has_amd_gpu)
        return "openvino", device, reason
    if torch_info["cuda"]:
        return (
            "cpu",
            "cpu",
            f"NVIDIA GPU has less than {MIN_NVIDIA_VRAM_MB} MB VRAM and "
            "no other accelerator available; using CPU.",
        )
    if has_amd_gpu:
        return (
            "cpu",
            "cpu",
            "AMD GPU detected but no DirectML/ROCm/OpenVINO backend available; using CPU.",
        )
    return "cpu", "cpu", "No GPU acceleration detected; using CPU."


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
    forced = _try_forced_backend(force, torch_info, ov_info, dml_info, has_amd_gpu)
    if forced is not None:
        backend, device, reason = forced
        return backend, device, reason, nvidia_vram_ok
    backend, device, reason = _auto_select_backend(
        torch_info, ov_info, dml_info, has_amd_gpu, nvidia_vram_ok
    )
    return backend, device, reason, nvidia_vram_ok


def detect_hardware(refresh: bool = False) -> dict:
    """Detect hardware and apply the app backend policy."""
    if _hw_cache and not refresh:
        return _hw_cache[0]

    torch_info = _check_torch()
    ov_info = _check_openvino()
    dml_info = _check_directml()
    ram_info = _check_system_ram()
    cpu_info = _check_cpu_threads()
    plat_info = _check_platform()
    ffmpeg_path = shutil.which("ffmpeg")
    has_amd_gpu = _detect_amd_gpu()

    selected_backend, selected_device, backend_reason, nvidia_vram_ok = _select_backend(
        torch_info,
        ov_info,
        dml_info,
        has_amd_gpu,
    )

    # Warn-only policy when host is under the documented 4-thread / 8 GB floor.
    if not ram_info["system_ram_ok"] and ram_info["system_ram_mb"]:
        backend_reason = (
            f"{backend_reason} WARNING: system RAM {ram_info['system_ram_mb']} MB "
            f"is below minimum {MIN_SYSTEM_RAM_MB} MB; may be slow or OOM."
        )
    if not cpu_info["cpu_threads_ok"]:
        backend_reason = (
            f"{backend_reason} WARNING: {cpu_info['cpu_threads']} CPU threads "
            f"(minimum {MIN_CPU_THREADS} recommended)."
        )

    info = {
        **torch_info,
        **ov_info,
        **dml_info,
        **ram_info,
        **cpu_info,
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


def _vram_policy_lines(hw: dict) -> list[str]:
    """Append NVIDIA VRAM status and low-VRAM policy notes when CUDA is present."""
    if not hw["cuda"]:
        return []
    lines = [
        f"- **NVIDIA GPU:** {hw['cuda_device_name']} "
        f"({hw['cuda_vram_mb']} MB VRAM; minimum {hw['min_nvidia_vram_mb']} MB)"
    ]
    try:
        class_limit = int(os.getenv("ASR_8GB_CLASS_MAX_MB", "9000"))
    except ValueError:
        class_limit = 9000
    hard_safe = os.getenv("ASR_HARD_MEMORY_SAFE", "true").strip().lower()
    if hard_safe in {"1", "true", "yes", "on"} and hw["cuda_vram_mb"] <= class_limit:
        lines.append(
            "- **VRAM policy:** strict low-VRAM mode; "
            "preloaded ASR models retained for reuse, "
            "CPU diarization by default"
        )
    return lines


def _accelerator_summary_lines(hw: dict) -> list[str]:
    """Build Markdown lines for optional accelerators (ROCm, DirectML, AMD probe)."""
    lines: list[str] = []
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
    return lines


def _host_resource_lines(hw: dict) -> list[str]:
    """Build Markdown lines for CPU arch/threads and system RAM."""
    lines: list[str] = []
    arch = hw.get("cpu_arch") or "unknown"
    lines.append(f"- **CPU arch:** {arch}" + (" (ARM)" if hw.get("is_arm") else ""))
    threads = hw.get("cpu_threads") or 0
    if threads:
        thr_note = (
            "OK"
            if hw.get("cpu_threads_ok")
            else f"below minimum {hw.get('min_cpu_threads', MIN_CPU_THREADS)}"
        )
        lines.append(f"- **CPU threads:** {threads} ({thr_note})")
    ram_mb = hw.get("system_ram_mb") or 0
    if ram_mb:
        ram_note = (
            "OK"
            if hw.get("system_ram_ok")
            else f"below minimum {hw.get('min_system_ram_mb', MIN_SYSTEM_RAM_MB)} MB"
        )
        lines.append(f"- **System RAM:** {ram_mb} MB ({ram_note})")
    return lines


def hardware_summary() -> str:
    """Return a Markdown summary of the selected hardware backend."""
    hw = detect_hardware()
    lines = [
        "### Hardware Status",
        f"- **Selected backend:** {hw['backend'].upper()} / {hw['selected_device']}",
        f"- **Reason:** {hw['backend_reason']}",
        f"- **PyTorch:** {hw['torch_version'] or 'not installed'}",
        *_vram_policy_lines(hw),
        *_accelerator_summary_lines(hw),
        f"- **OpenVINO:** {hw['openvino_version'] or 'not installed'}",
        f"- **OpenVINO devices:** {', '.join(hw['available_devices']) or 'none'}",
        f"- **FFmpeg:** {'available' if hw['ffmpeg'] else 'NOT FOUND'}",
        *_host_resource_lines(hw),
    ]
    return "\n".join(lines)
