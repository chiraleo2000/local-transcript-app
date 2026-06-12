"""Centralized VRAM snapshot, logging, and teardown helpers."""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def snapshot() -> dict[str, int]:
    """Return CUDA memory stats in MB, or zeros when CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0}
        allocated = int(torch.cuda.memory_allocated() // (1024 * 1024))
        reserved = int(torch.cuda.memory_reserved() // (1024 * 1024))
        free_bytes, _total = torch.cuda.mem_get_info()
        free_mb = int(free_bytes // (1024 * 1024))
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "free_mb": free_mb,
        }
    except (ImportError, RuntimeError, OSError, AttributeError):
        return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0}


def log_phase(phase: str, *, before: bool = True) -> None:
    """Log VRAM usage for a pipeline phase."""
    snap = snapshot()
    suffix = "start" if before else "end"
    logger.info(
        "VRAM [%s:%s]: alloc=%dMB reserved=%dMB free=%dMB",
        phase,
        suffix,
        snap["allocated_mb"],
        snap["reserved_mb"],
        snap["free_mb"],
    )


def teardown(*, aggressive: bool = False) -> None:
    """Release Python and CUDA cache; aggressive also runs ipc_collect."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if aggressive:
                try:
                    torch.cuda.ipc_collect()
                except (RuntimeError, AttributeError):
                    pass
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        logger.debug("VRAM teardown skipped: %s", exc)


def recover_cuda() -> None:
    """Best-effort CUDA cleanup after driver or allocator errors."""
    gc.collect()
    try:
        import torch

        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except (RuntimeError, AttributeError):
            pass
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        logger.debug("CUDA recover skipped: %s", exc)


def cuda_device_healthy() -> bool:
    """Return False when a tiny CUDA alloc fails (corrupted context)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        probe = torch.zeros(1, device="cuda")
        del probe
        torch.cuda.synchronize()
        return True
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        logger.warning("CUDA health probe failed: %s", exc)
        recover_cuda()
        return False


def ensure_headroom(min_free_mb: int) -> bool:
    """Warn and run light teardown when free VRAM is below threshold."""
    snap = snapshot()
    free_mb = snap["free_mb"]
    if free_mb and free_mb < min_free_mb:
        logger.warning(
            "VRAM headroom low: free=%dMB < min=%dMB; running light teardown.",
            free_mb,
            min_free_mb,
        )
        teardown(aggressive=False)
        return False
    return True
