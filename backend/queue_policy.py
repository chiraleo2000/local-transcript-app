"""Hardware-aware queue / concurrency defaults (explicit env always wins)."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Env keys we may auto-fill when unset/empty.
_AUTO_KEYS = (
    "UI_MAX_CONCURRENT_JOBS",
    "API_MAX_QUEUED_JOBS",
    "UI_MAX_BATCH_FILES",
)


def _env_unset(key: str) -> bool:
    return not (os.getenv(key) or "").strip()


def _parallel_min_vram_mb() -> int:
    try:
        return max(1, int(os.getenv("ASR_PARALLEL_MIN_VRAM_MB", str(12 * 1024))))
    except ValueError:
        return 12 * 1024


def detect_queue_tier(gpu: dict[str, Any] | None = None) -> str:
    """Return tier letter A|B|C based on GPU profile and VRAM.

    A — Pascal / Tesla P4 / low VRAM (~8 GB): sequential GPU
    B — Modern Ampere+ but under parallel VRAM threshold: sequential GPU
    C — VRAM >= ASR_PARALLEL_MIN_VRAM_MB: allow 2 GPU slots when env unset
    """
    from backend.gpu_arch import (
        is_pascal_speed_gpu,
        pascal_speed_gpu_active,
        probe_cuda_device,
    )

    info = gpu if gpu is not None else probe_cuda_device()
    name = str(info.get("cuda_device_name") or "")
    major = int(info.get("cuda_capability_major") or 0)
    vram = int(info.get("cuda_vram_mb") or 0)

    if pascal_speed_gpu_active() or is_pascal_speed_gpu(name, major):
        return "A"
    if vram >= _parallel_min_vram_mb():
        return "C"
    return "B"


def recommended_caps(tier: str | None = None) -> dict[str, int]:
    """Recommended defaults for the given (or detected) tier."""
    letter = (tier or detect_queue_tier()).upper()
    if letter == "C":
        return {
            "UI_MAX_CONCURRENT_JOBS": 2,
            "API_MAX_QUEUED_JOBS": 8,
            "UI_MAX_BATCH_FILES": 5,
        }
    # A and B: safe sequential GPU, small batch
    return {
        "UI_MAX_CONCURRENT_JOBS": 1,
        "API_MAX_QUEUED_JOBS": 4,
        "UI_MAX_BATCH_FILES": 3,
    }


def max_batch_files() -> int:
    try:
        return max(1, int(os.getenv("UI_MAX_BATCH_FILES", "3")))
    except ValueError:
        return 3


def apply_queue_policy(*, force: bool = False) -> dict[str, str]:
    """Fill unset queue env vars from hardware tier; never override explicit values.

    Call once at process startup before accepting jobs. Returns applied key=value map.
    """
    tier = detect_queue_tier()
    caps = recommended_caps(tier)
    applied: dict[str, str] = {}
    for key in _AUTO_KEYS:
        if force or _env_unset(key):
            value = str(caps[key])
            os.environ[key] = value
            applied[key] = value

    try:
        from backend.pipeline import configure_job_semaphore

        configure_job_semaphore()
    except Exception:  # pylint: disable=broad-exception-caught
        logger.debug("configure_job_semaphore skipped", exc_info=True)

    from backend.gpu_arch import probe_cuda_device

    gpu = probe_cuda_device()
    logger.info(
        "Queue policy tier=%s gpu=%s vram_mb=%s applied=%s",
        tier,
        gpu.get("cuda_device_name") or "none",
        gpu.get("cuda_vram_mb") or 0,
        applied or "(explicit env kept)",
    )
    return applied


__all__ = [
    "apply_queue_policy",
    "detect_queue_tier",
    "max_batch_files",
    "recommended_caps",
]
