"""Per-browser-tab UI session state (isolated by tab_instance_id, not Gradio session)."""

from __future__ import annotations

import threading
import uuid

from backend.progress import JobProgress

DEFAULT_TAB_ID = "__default__"

_REGISTRY_LOCK = threading.Lock()
_SESSION_REGISTRY: dict[str, dict] = {}


def _new_runtime() -> dict:
    return {
        "cancel_event": threading.Event(),
        "progress": JobProgress(),
        "last_upload_path": None,
        "active_job_id": None,
        "worker": None,
        "selected_asr_engine": None,
    }


def _normalize_tab_id(tab_id: str | None) -> str:
    if isinstance(tab_id, str) and tab_id.strip():
        return tab_id.strip()
    return DEFAULT_TAB_ID


def resolve_runtime(tab_id: str | None) -> tuple[dict, str]:
    """Return runtime bucket for this tab; create registry entry if needed."""
    tid = _normalize_tab_id(tab_id)
    with _REGISTRY_LOCK:
        if tid not in _SESSION_REGISTRY:
            _SESSION_REGISTRY[tid] = _new_runtime()
        return _SESSION_REGISTRY[tid], tid


def init_tab_instance_id(current: str | None) -> str:
    """Seed a stable tab id on page load (Gradio demo.load)."""
    if isinstance(current, str) and current.strip():
        return current.strip()
    new_id = uuid.uuid4().hex
    with _REGISTRY_LOCK:
        _SESSION_REGISTRY[new_id] = _new_runtime()
    return new_id


def fresh_cancel_event(runtime: dict, *, cancel_previous: bool = True) -> threading.Event:
    """Return a new cancel event; optionally signal the previous in-flight job."""
    if cancel_previous:
        runtime["cancel_event"].set()
    runtime["cancel_event"] = threading.Event()
    return runtime["cancel_event"]


def set_active_job(runtime: dict, job_id: str, worker: threading.Thread | None) -> None:
    runtime["active_job_id"] = job_id
    runtime["worker"] = worker


def clear_active_job(runtime: dict) -> None:
    runtime["active_job_id"] = None
    runtime["worker"] = None


def is_job_running(runtime: dict) -> bool:
    worker = runtime.get("worker")
    if worker is not None and worker.is_alive():
        return True
    return bool(runtime["progress"].snapshot().get("active"))
