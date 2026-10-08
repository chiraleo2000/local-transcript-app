"""Local transcript outbox so a finished job is not lost if the sidecar blips."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def result_dir() -> Path:
    """Directory of unsynced and synced result payloads."""
    override = os.getenv("RESULT_OUTBOX_DIR", "").strip()
    if override:
        path = Path(override)
        path.mkdir(parents=True, exist_ok=True)
        return path
    from backend.storage import STORAGE_DIR, ensure_app_dirs

    ensure_app_dirs()
    path = STORAGE_DIR / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_engine(engine: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in (engine or "transcript"))
    return cleaned.strip("_")[:48] or "transcript"


def result_path(job_id: str, engine: str) -> Path:
    return result_dir() / f"{job_id}__{_safe_engine(engine)}.json"


def remember_result(
    job_id: str,
    engine: str,
    text: str,
    *,
    language: str = "",
    duration_s: float = 0.0,
    transcript_path: str = "",
    segments: list[dict[str, Any]] | None = None,
    error: str = "",
    synced: bool = False,
) -> Path:
    """Write the transcript payload to disk before the sidecar call."""
    path = result_path(job_id, engine)
    payload = {
        "job_id": job_id,
        "engine": engine,
        "text": text,
        "language": language,
        "duration_s": duration_s,
        "transcript_path": transcript_path,
        "segments": segments or [],
        "error": error,
        "synced": synced,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def mark_synced(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    payload["synced"] = True
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_outbox_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Skipping unreadable result outbox file %s", path)
        return None
    if not isinstance(payload, dict) or payload.get("synced"):
        return None
    if not str(payload.get("job_id") or ""):
        return None
    return payload


def _push_outbox_payload(path: Path, payload: dict[str, Any]) -> str:
    """Store one payload. Return 'stored', 'skip', or 'stop'."""
    from backend.storage_client import StorageUnavailable, save_result

    job_id = str(payload.get("job_id") or "")
    segments = payload.get("segments")
    try:
        save_result(
            job_id=job_id,
            engine=str(payload.get("engine") or "transcript"),
            text=str(payload.get("text") or ""),
            language=str(payload.get("language") or ""),
            duration_s=float(payload.get("duration_s") or 0),
            transcript_path=str(payload.get("transcript_path") or ""),
            segments=segments if isinstance(segments, list) else None,
            error=str(payload.get("error") or ""),
        )
    except StorageUnavailable:
        logger.exception("Result outbox sync stopped at %s", job_id)
        return "stop"
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Result outbox skipped %s after an unexpected sync error", job_id)
        return "skip"
    mark_synced(path)
    return "stored"


def _sync_outbox_file(path: Path) -> str:
    payload = _read_outbox_payload(path)
    if payload is None:
        return "skip"
    return _push_outbox_payload(path, payload)


def _sync_outbox_batch(paths: list[Path], limit: int) -> int:
    stored = 0
    for path in paths:
        if stored >= limit:
            return stored
        outcome = _sync_outbox_file(path)
        if outcome == "stop":
            return stored
        if outcome == "stored":
            stored += 1
    return stored


def flush_pending_results(limit: int = 50) -> int:
    """Push unsynced local results to the sidecar. Returns how many were stored."""
    from backend.storage_client import storage_configured

    if not storage_configured():
        return 0
    stored = _sync_outbox_batch(sorted(result_dir().glob("*.json")), limit)
    if stored:
        logger.info("Synced %d local transcript result(s) to storage.", stored)
    return stored
