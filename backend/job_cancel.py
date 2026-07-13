"""Cancel in-flight tab jobs and release GPU memory before restart."""

from __future__ import annotations

import logging
import os
import threading

from backend.progress import JobProgress
from backend.services.asr_local import should_unload_on_cancel
from backend.ui_session import clear_active_job

logger = logging.getLogger(__name__)


def cancel_join_timeout_s() -> float:
    try:
        return max(1.0, float(os.getenv("UI_CANCEL_JOIN_TIMEOUT_S", "120")))
    except ValueError:
        return 120.0


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def should_free_gpu_for_queue_on_cancel() -> bool:
    """Clear accelerator cache on cancel so the next queued job can start cleanly."""
    return _env_bool("UI_CANCEL_FREES_GPU_FOR_QUEUE", True)


def cancel_tab_job(
    runtime: dict,
    *,
    unload_models: bool | None = None,
    tracker: JobProgress | None = None,
    message: str = "Cancelled.",
) -> None:
    """Signal cancel, wait for the worker, and free VRAM for queued jobs when configured."""
    runtime["cancel_event"].set()
    worker = runtime.get("worker")
    if worker is not None and worker.is_alive():
        worker.join(timeout=cancel_join_timeout_s())
        if worker.is_alive():
            logger.warning(
                "Cancel join timed out after %.0fs; worker may still hold VRAM.",
                cancel_join_timeout_s(),
            )
    do_unload = (
        unload_models if unload_models is not None else should_unload_on_cancel()
    )
    if do_unload:
        from backend.pipeline import unload_all_pipeline_models

        unload_all_pipeline_models()
    elif should_free_gpu_for_queue_on_cancel():
        from backend.services.asr_local import clear_accelerator_cache

        clear_accelerator_cache()
        logger.info("Cancel freed GPU cache for queued jobs (models kept warm).")
    if tracker is not None:
        tracker.fail(message)
    clear_active_job(runtime)
