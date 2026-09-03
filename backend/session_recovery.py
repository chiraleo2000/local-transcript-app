"""Recover in-flight and recently finished jobs after UI reconnect."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.job_status import job_is_in_flight, job_status_norm
from backend.storage import list_jobs

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def job_has_terminal_results(job: dict[str, Any] | None) -> bool:
    if not job:
        return False
    return job_status_norm(job) == "completed" or bool(job.get("results"))


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _append_job_id(
    row: dict[str, Any],
    *,
    candidates: list[str],
    completed: list[str],
) -> None:
    jid = str(row.get("job_id") or "")
    if not jid:
        return
    if job_is_in_flight(row) or job_status_norm(row) == "running":
        _append_unique(candidates, jid)
        return
    if job_has_terminal_results(row):
        _append_unique(completed, jid)


def _job_recently_updated(row: dict[str, Any], *, within_s: int) -> bool:
    raw = row.get("updated_at") or row.get("created_at") or ""
    if not isinstance(raw, str) or not raw:
        return False
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - stamp).total_seconds() <= max(60, within_s)


def _seed_from_runtime(
    runtime: dict | None,
    *,
    candidates: list[str],
    completed: list[str],
) -> None:
    if not runtime:
        return
    if runtime.get("active_job_id"):
        candidates.append(str(runtime["active_job_id"]))
    if runtime.get("last_completed_job_id"):
        completed.append(str(runtime["last_completed_job_id"]))


def _collect_tab_scoped(
    rows: list[dict[str, Any]],
    tab_id: str,
    *,
    candidates: list[str],
    completed: list[str],
) -> bool:
    """Append tab-scoped jobs; return True if any completed job was found."""
    tab_has_completed = False
    for row in rows:
        if row.get("tab_id") != tab_id:
            continue
        before = len(completed)
        _append_job_id(row, candidates=candidates, completed=completed)
        if len(completed) > before and job_has_terminal_results(row):
            tab_has_completed = True
    return tab_has_completed


def _append_user_inflight(
    rows: list[dict[str, Any]],
    *,
    candidates: list[str],
) -> None:
    for row in rows:
        jid = str(row.get("job_id") or "")
        if not jid or jid in candidates:
            continue
        if job_is_in_flight(row):
            candidates.append(jid)


def _append_recent_user_completed(
    rows: list[dict[str, Any]],
    *,
    completed: list[str],
    within_s: int,
) -> None:
    for row in rows:
        jid = str(row.get("job_id") or "")
        if not jid or jid in completed:
            continue
        if not job_has_terminal_results(row):
            continue
        if _job_recently_updated(row, within_s=within_s):
            completed.append(jid)
            return


def collect_recovery_job_candidates(
    tab_id: str,
    runtime: dict | None,
    *,
    username: str | None = None,
    user_id: int | None = None,
    limit: int = 50,
    recent_completed_within_s: int = 14400,
) -> tuple[list[str], list[str]]:
    """Return (in-flight candidates, completed ids) for session recovery.

    Tab-scoped jobs are preferred. When the browser tab id changes (refresh,
    re-login, or sessionStorage loss), fall back to the logged-in account's
    in-flight work and very recent completed jobs so disconnects do not lose output.
    """
    candidates: list[str] = []
    completed: list[str] = []
    _seed_from_runtime(runtime, candidates=candidates, completed=completed)

    rows = list_jobs(limit, username=username, user_id=user_id)
    tab_has_completed = _collect_tab_scoped(
        rows, tab_id, candidates=candidates, completed=completed
    )

    if username or user_id:
        _append_user_inflight(rows, candidates=candidates)
        if not tab_has_completed:
            _append_recent_user_completed(
                rows, completed=completed, within_s=recent_completed_within_s
            )

    return candidates, completed
