"""Process-local job queue and job_id cancel registry for headless API jobs."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.progress import JobProgress

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_JOB_EVENTS: dict[str, threading.Event] = {}
_JOB_PROGRESS: dict[str, JobProgress] = {}
_JOB_THREADS: dict[str, threading.Thread] = {}
_QUEUED_COUNT = 0
_ACTIVE_API_JOBS = 0


def max_queued_jobs() -> int:
    try:
        return max(1, int(os.getenv("API_MAX_QUEUED_JOBS", "4")))
    except ValueError:
        return 4


def register_job_handles(
    job_id: str,
    cancel_event: threading.Event,
    progress: JobProgress,
) -> None:
    with _LOCK:
        _JOB_EVENTS[job_id] = cancel_event
        _JOB_PROGRESS[job_id] = progress


def unregister_job_handles(job_id: str) -> None:
    with _LOCK:
        _JOB_EVENTS.pop(job_id, None)
        _JOB_PROGRESS.pop(job_id, None)
        _JOB_THREADS.pop(job_id, None)


def get_job_progress(job_id: str) -> JobProgress | None:
    with _LOCK:
        return _JOB_PROGRESS.get(job_id)


def get_cancel_event(job_id: str) -> threading.Event | None:
    with _LOCK:
        return _JOB_EVENTS.get(job_id)


def cancel_job_by_id(job_id: str) -> bool:
    """Signal cancel for an API/headless job. Returns False if unknown/finished."""
    with _LOCK:
        event = _JOB_EVENTS.get(job_id)
        worker = _JOB_THREADS.get(job_id)
    if event is None:
        return False
    event.set()
    if worker is not None and worker.is_alive():
        try:
            timeout = float(os.getenv("UI_CANCEL_JOIN_TIMEOUT_S", "120"))
        except ValueError:
            timeout = 120.0
        worker.join(timeout=max(1.0, timeout))
    from backend.job_cancel import should_free_gpu_for_queue_on_cancel

    if should_free_gpu_for_queue_on_cancel():
        try:
            from backend.services.asr_local import clear_accelerator_cache

            clear_accelerator_cache()
        except Exception:  # pylint: disable=broad-exception-caught
            logger.debug("GPU cache clear after API cancel failed", exc_info=True)
    return True


def queued_and_active_count() -> int:
    with _LOCK:
        return _QUEUED_COUNT + _ACTIVE_API_JOBS


def try_reserve_queue_slot() -> bool:
    """Reserve a queue slot; False when at API_MAX_QUEUED_JOBS capacity."""
    global _QUEUED_COUNT
    with _LOCK:
        if _QUEUED_COUNT + _ACTIVE_API_JOBS >= max_queued_jobs():
            return False
        _QUEUED_COUNT += 1
        return True


def release_queue_slot(*, started: bool = False) -> None:
    """Release a reserved slot (call when job leaves queue or finishes)."""
    global _QUEUED_COUNT, _ACTIVE_API_JOBS
    with _LOCK:
        if started:
            _ACTIVE_API_JOBS = max(0, _ACTIVE_API_JOBS - 1)
        else:
            _QUEUED_COUNT = max(0, _QUEUED_COUNT - 1)


def mark_queue_slot_running() -> None:
    """Move one reserved slot from queued → active."""
    global _QUEUED_COUNT, _ACTIVE_API_JOBS
    with _LOCK:
        _QUEUED_COUNT = max(0, _QUEUED_COUNT - 1)
        _ACTIVE_API_JOBS += 1


@dataclass
class SubmittedJob:
    job_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    progress: JobProgress = field(default_factory=JobProgress)


def submit_background_job(
    job_id: str,
    worker_fn: Callable[[SubmittedJob], None],
) -> SubmittedJob:
    """Start a daemon worker for an already-reserved queue slot."""
    handle = SubmittedJob(job_id=job_id)
    register_job_handles(job_id, handle.cancel_event, handle.progress)

    def _run() -> None:
        mark_queue_slot_running()
        try:
            handle.progress.start(job_id)
            worker_fn(handle)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Background job %s crashed", job_id)
        finally:
            release_queue_slot(started=True)
            unregister_job_handles(job_id)

    thread = threading.Thread(target=_run, name=f"api-job-{job_id}", daemon=True)
    with _LOCK:
        _JOB_THREADS[job_id] = thread
    thread.start()
    return handle


def snapshot_queue() -> dict[str, Any]:
    with _LOCK:
        return {
            "queued": _QUEUED_COUNT,
            "active": _ACTIVE_API_JOBS,
            "max": max_queued_jobs(),
            "job_ids": list(_JOB_EVENTS.keys()),
            "ts": time.time(),
        }
