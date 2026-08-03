"""Helpers for durable job status (UI recover / history load)."""

from __future__ import annotations

from typing import Any

_IN_FLIGHT_STATUSES = frozenset({"queued", "running"})
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def job_status_norm(job: dict[str, Any] | None) -> str:
    if not job:
        return ""
    return str(job.get("status") or "").strip().lower()


def job_is_in_flight(job: dict[str, Any] | None) -> bool:
    """True while a job is queued/running, or incomplete without terminal results."""
    if not job:
        return False
    status = job_status_norm(job)
    if status in _IN_FLIGHT_STATUSES:
        return True
    if status in _TERMINAL_STATUSES:
        return False
    # Legacy/throttled manifests may lack status while progress is updating.
    if job.get("progress") and not (job.get("results") or {}):
        return True
    return False
