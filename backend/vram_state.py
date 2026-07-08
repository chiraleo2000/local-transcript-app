"""Centralized VRAM snapshot, logging, and teardown helpers."""

from __future__ import annotations

import gc
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Exit code used when we deliberately kill the process to recover a dead CUDA
# context. A container supervisor (Docker `restart: unless-stopped`) brings the
# process back with a fresh CUDA driver context.
CUDA_RESTART_EXIT_CODE = 42


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


def _cuda_memory_fraction() -> float:
    """Return clamped per-process CUDA memory fraction (default/max 0.92)."""
    raw = os.getenv("ASR_CUDA_MEMORY_FRACTION", "0.92").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.92
    cap_raw = os.getenv("ASR_CUDA_MEMORY_FRACTION_MAX", "0.92").strip()
    try:
        cap = float(cap_raw)
    except ValueError:
        cap = 0.92
    cap = min(1.0, max(0.5, cap))
    return min(cap, max(0.5, value))


def apply_cuda_memory_fraction() -> float:
    """Apply ASR_CUDA_MEMORY_FRACTION cap before loading GPU models."""
    fraction = _cuda_memory_fraction()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            logger.info("CUDA memory fraction set to %.2f", fraction)
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        logger.debug("CUDA memory fraction skipped: %s", exc)
    return fraction


def prepare_exclusive_gpu_job(*, unload_models: bool = True) -> None:
    """Clear VRAM and enforce sequential GPU staging (diar then ASR, never both resident)."""
    if unload_models:
        try:
            from backend.model_registry import unload_all_models

            unload_all_models()
        except ImportError:
            pass
    teardown(aggressive=True)
    recover_cuda()
    apply_cuda_memory_fraction()
    log_phase("gpu_job_prep", before=False)


def cuda_device_healthy() -> bool:
    """Return False when a tiny CUDA alloc fails (corrupted context)."""
    import warnings

    try:
        import torch

        if not torch.cuda.is_available():
            return False
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*expandable_segments not supported.*",
            )
            probe = torch.zeros(1, device="cuda")
            del probe
            torch.cuda.synchronize()
        return True
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        logger.warning("CUDA health probe failed: %s", exc)
        recover_cuda()
        return False


def cuda_auto_restart_enabled() -> bool:
    """Return True when the process may self-restart to recover a dead context.

    Only safe under a supervisor that restarts the process (Docker
    `restart: unless-stopped`). Off by default so a bare local run is never
    hard-killed with no way back.
    """
    return os.getenv("CUDA_AUTO_RESTART", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def request_cuda_restart(reason: str) -> bool:
    """Restart the process to rebuild the CUDA context, if auto-restart is on.

    A `cudaErrorUnknown` (context corrupted by a host GPU reset / TDR /
    contention on shared laptop GPUs under WSL2) cannot be rebuilt inside the
    running process: PyTorch caches the primary CUDA context and there is no API
    to tear it down and re-init. A fresh process, however, gets a clean context.
    So when auto-restart is enabled we exit non-zero and let the supervisor
    bring us back healthy.

    Returns False when auto-restart is disabled (caller should surface the error
    normally); never returns when it exits the process.
    """
    if not cuda_auto_restart_enabled():
        logger.error(
            "CUDA context is dead and cannot be rebuilt in-process (%s). "
            "The GPU was likely reset by the host (e.g. another GPU app, a "
            "display-driver TDR, or contention on a shared laptop GPU). "
            "Restart the service to recover, and avoid running GPU-heavy apps "
            "during transcription. Set CUDA_AUTO_RESTART=1 to auto-restart.",
            reason,
        )
        return False
    logger.error(
        "Restarting process to recover a dead CUDA context: %s "
        "(exit %d; supervisor will restart with a fresh GPU context).",
        reason,
        CUDA_RESTART_EXIT_CODE,
    )
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (ValueError, OSError):
        pass
    logging.shutdown()
    # os._exit avoids atexit/CUDA teardown hooks that would hang on a dead context.
    os._exit(CUDA_RESTART_EXIT_CODE)


def ensure_cuda_healthy_or_restart(context: str) -> bool:
    """Guard a GPU phase: probe CUDA, recover once, else restart the process.

    Returns True when CUDA is healthy (or unavailable / CPU mode). May not
    return: when the context is dead, unrecoverable, and auto-restart is on it
    exits the process. Returns False only when the context is dead and
    auto-restart is disabled (so callers can fall through to their own error).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return True
    except (ImportError, RuntimeError, OSError, AttributeError):
        return True

    if cuda_device_healthy():
        return True

    logger.warning(
        "CUDA context unhealthy before %s; attempting in-process recovery.",
        context,
    )
    recover_cuda()
    if cuda_device_healthy():
        logger.info("CUDA context recovered in-process before %s.", context)
        return True

    return request_cuda_restart(f"unrecoverable cudaErrorUnknown before {context}")


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


def _gpu_lock_path() -> "Path":
    from pathlib import Path

    raw = os.getenv("GPU_JOB_LOCK_PATH", "").strip()
    if raw:
        return Path(raw)
    if sys.platform == "win32":
        base = Path(os.getenv("LOCALAPPDATA", str(Path.home())))
    else:
        base = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "local-transcript-app" / "gpu.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED
            if not handle:
                return False
            kernel32.CloseHandle(handle)
            return True
        except (AttributeError, OSError, ValueError):
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _stale_gpu_lock(lock_path: "Path", *, max_age_s: float) -> bool:
    import time

    try:
        if not lock_path.is_file():
            return False
        age_s = time.time() - lock_path.stat().st_mtime
        if age_s > max_age_s:
            return True
        raw = lock_path.read_text(encoding="utf-8").strip().splitlines()
        if not raw:
            return age_s > 300.0
        return not _pid_alive(int(raw[0]))
    except (OSError, ValueError):
        return True


def acquire_gpu_job_lock(*, timeout_s: float | None = None) -> None:
    """Block until this process holds the cross-process GPU job lock (8 GB VRAM).

    Only one validation / transcription GPU job may run at a time. A second
    process waits instead of competing for VRAM with the first.
    """
    import time

    lock_path = _gpu_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    wait_s = timeout_s if timeout_s is not None else float(
        os.getenv("GPU_JOB_LOCK_TIMEOUT_S", "14400"),
    )
    poll_s = max(1.0, float(os.getenv("GPU_JOB_LOCK_POLL_S", "5")))
    stale_s = max(300.0, float(os.getenv("GPU_JOB_LOCK_STALE_S", "21600")))
    deadline = time.time() + wait_s
    pid = os.getpid()

    while time.time() < deadline:
        try:
            with open(lock_path, "x", encoding="utf-8") as handle:
                handle.write(f"{pid}\n{time.time():.0f}\n")
                handle.flush()
            logger.info("Acquired GPU job lock: %s (pid=%d)", lock_path, pid)
            return
        except FileExistsError:
            if _stale_gpu_lock(lock_path, max_age_s=stale_s):
                try:
                    lock_path.unlink()
                    logger.warning("Removed stale GPU job lock: %s", lock_path)
                    continue
                except OSError:
                    pass
            remaining = max(0.0, deadline - time.time())
            logger.info(
                "Waiting for GPU job lock (%s); %.0fs remaining …",
                lock_path,
                remaining,
            )
            print(
                f"GPU busy — waiting for exclusive 8 GB VRAM lock ({lock_path.name}) …",
                flush=True,
            )
            time.sleep(min(poll_s, remaining))
    raise TimeoutError(
        f"Timed out after {wait_s:.0f}s waiting for GPU job lock: {lock_path}"
    )


def release_gpu_job_lock() -> None:
    """Release the cross-process GPU job lock held by this process."""
    lock_path = _gpu_lock_path()
    try:
        if lock_path.is_file():
            raw = lock_path.read_text(encoding="utf-8").strip().splitlines()
            holder = int(raw[0]) if raw else -1
            if holder in (-1, os.getpid()):
                lock_path.unlink()
                logger.info("Released GPU job lock: %s", lock_path)
    except (OSError, ValueError) as exc:
        logger.debug("GPU job lock release skipped: %s", exc)


def gpu_job_lock(*, timeout_s: float | None = None):
    """Context manager: exclusive GPU job lock for 8 GB sequential staging."""
    import contextlib

    @contextlib.contextmanager
    def _manager():
        acquire_gpu_job_lock(timeout_s=timeout_s)
        try:
            yield
        finally:
            release_gpu_job_lock()

    return _manager()
