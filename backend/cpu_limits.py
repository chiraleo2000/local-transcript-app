"""CPU thread limits for enhancement and diarization preprocessing."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def cpu_thread_count() -> int:
    try:
        return max(1, int(os.getenv("APP_CPU_THREADS", "16")))
    except ValueError:
        return 16


def apply_cpu_thread_limits() -> int:
    """Cap BLAS/OpenMP and PyTorch CPU threads (default 16)."""
    threads = cpu_thread_count()
    for key in _THREAD_ENV_KEYS:
        os.environ[key] = str(threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch

        torch.set_num_threads(threads)
    except (ImportError, RuntimeError, OSError):
        pass
    logger.info("CPU thread limits applied: APP_CPU_THREADS=%d", threads)
    return threads
