"""NVIDIA GPU generation helpers (Tesla P4 / Pascal vs Ampere+ such as RTX 4060).

Tesla P4 is compute capability 6.1 with almost no FP16 throughput (1/64 of FP32)
and no Tensor cores. The RTX 4060 path (CUDA 13 + float16 + 5-beam decode) is
several times slower on P4. These helpers switch P4 onto CUDA 12.4-compatible
knobs: FP32 compute, 2-beam decode, shorter temperature fallback, fewer turns.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# Explicit profile: auto | p4 | off | 4060 | ampere
_PROFILE_ENV = "ASR_GPU_PROFILE"

# Pre-Volta (CC < 7) has no Tensor cores; Tesla P4 FP16 is ~1/64 of FP32.
_PASCAL_MAJOR_MAX = 6

_P4_NAME_RE = re.compile(
    r"(tesla\s*p4|quadro\s*p4(?:000|0000)?\b|\bp4\b)",
    re.IGNORECASE,
)


def gpu_profile_override() -> str:
    """Return ASR_GPU_PROFILE (auto/p4/off/…)."""
    return os.getenv(_PROFILE_ENV, "auto").strip().lower() or "auto"


def probe_cuda_device() -> dict:
    """Best-effort CUDA name / VRAM / compute capability (zeros when unavailable)."""
    info = {
        "cuda": False,
        "cuda_device_name": "",
        "cuda_vram_mb": 0,
        "cuda_capability_major": 0,
        "cuda_capability_minor": 0,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return info
        props = torch.cuda.get_device_properties(0)
        info["cuda"] = True
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_vram_mb"] = int(props.total_memory // (1024 * 1024))
        info["cuda_capability_major"] = int(getattr(props, "major", 0) or 0)
        info["cuda_capability_minor"] = int(getattr(props, "minor", 0) or 0)
    except (ImportError, RuntimeError, OSError, AttributeError):
        return info
    return info


def is_tesla_p4_name(name: str) -> bool:
    text = (name or "").strip()
    if not text:
        return False
    if re.search(r"rtx|geforce", text, re.IGNORECASE):
        return False
    return bool(_P4_NAME_RE.search(text))


def is_pre_volta_capability(major: int) -> bool:
    return 0 < int(major) <= _PASCAL_MAJOR_MAX


def is_pascal_speed_gpu(
    name: str = "",
    major: int = 0,
) -> bool:
    """True for Tesla P4 and other Pascal cards (CC 6.x)."""
    if is_tesla_p4_name(name):
        return True
    return is_pre_volta_capability(major)


def pascal_speed_gpu_active() -> bool:
    """Whether the Tesla P4 / Pascal speed profile should apply."""
    profile = gpu_profile_override()
    if profile in {"off", "0", "false", "4060", "ampere", "ada", "high"}:
        return False
    if profile in {"p4", "pascal", "tesla-p4", "tesla_p4"}:
        return True
    gpu = probe_cuda_device()
    return is_pascal_speed_gpu(
        gpu.get("cuda_device_name", ""),
        int(gpu.get("cuda_capability_major") or 0),
    )


def prefers_float32_compute() -> bool:
    """Pascal/P4 should compute in FP32; Ampere+ stays on FP16."""
    raw = os.getenv("ASR_CUDA_DTYPE", "auto").strip().lower()
    if raw in {"float32", "fp32", "32"}:
        return True
    if raw in {"float16", "fp16", "16", "half", "bfloat16", "bf16"}:
        return False
    return pascal_speed_gpu_active()


# Runtime overrides applied on Tesla P4 / Pascal. Forced over gpu-app.env so the
# 4060 accuracy profile does not make P4 jobs take many times longer.
PASCAL_P4_SPEED_ENV: dict[str, str] = {
    "ASR_GPU_PROFILE": "p4",
    "ASR_CUDA_DTYPE": "float32",
    "ASR_ATTENTION_IMPLEMENTATION": "eager",
    "ASR_NUM_BEAMS": "2",
    "ASR_NUM_BEAMS_MAX": "2",
    "ASR_NUM_BEAMS_MIN": "1",
    "ASR_TEMPERATURE": "0.0,0.2,0.4",
    "ASR_TURN_GUIDED": "true",
    "ASR_TURN_GUIDED_MAX_TURN_S": "28",
    "ASR_TURN_GUIDED_MERGE_GAP_S": "0.55",
    "ASR_ADAPTIVE_PERFORMANCE": "true",
    "ASR_BUDGET_SEC_PER_TURN": "8",
    "ASR_CUDA_MEMORY_FRACTION": "0.90",
    "ASR_UNLOAD_FOR_DIARIZATION": "true",
    "DIARIZATION_PRELOAD_DEVICE": "cpu",
    "DIARIZATION_GPU_CO_RESIDENT": "false",
    "DIARIZATION_MEGA_TURN_MAX_REFINES": "1",
    "DIARIZATION_MULTI_SAMPLE": "false",
    "DIARIZATION_MULTI_SAMPLE_PASSES": "0",
    "AUDIO_ENHANCE_USE_GPU": "false",
}


def apply_pascal_p4_overrides() -> list[str]:
    """Force P4 speed knobs when a Pascal GPU is detected (or ASR_GPU_PROFILE=p4)."""
    if not pascal_speed_gpu_active():
        return []
    applied: list[str] = []
    for key, value in PASCAL_P4_SPEED_ENV.items():
        if os.getenv(key, "") != value:
            os.environ[key] = value
            applied.append(f"{key}={value}")
    gpu = probe_cuda_device()
    logger.info(
        "Tesla P4 / Pascal speed profile active (gpu=%s cc=%d.%d). %s",
        gpu.get("cuda_device_name") or "forced",
        int(gpu.get("cuda_capability_major") or 0),
        int(gpu.get("cuda_capability_minor") or 0),
        ", ".join(applied) if applied else "already set",
    )
    return applied


def recommended_cuda_stack() -> str:
    """Compose stack that still ships sm_61 kernels (Pascal / Tesla P4)."""
    if pascal_speed_gpu_active():
        return "cuda124"
    return "latest"
