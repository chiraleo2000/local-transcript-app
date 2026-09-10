"""SQLite job history index (metadata + paths; transcript text stays on disk)."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from backend.paths import app_root

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()
SCHEMA_VERSION = 1

_JOB_COLUMNS = (
    "job_id",
    "user_id",
    "username",
    "status",
    "display_name",
    "source_filename",
    "created_at",
    "updated_at",
    "transcript_path",
    "input_path",
    "error",
    "progress_json",
    "options_json",
    "client_ip",
    "audio_duration_s",
    "total_elapsed_s",
    "tab_id",
    "selected_engines_json",
)


def jobs_db_path() -> Path:
    raw = os.getenv("APP_JOBS_DB", "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = app_root() / path
        return path
    return app_root() / "storage" / "jobs.db"


def _connect() -> sqlite3.Connection:
    path = jobs_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_jobs_db() -> None:
    """Create schema and indexes if missing."""
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL DEFAULT 0,
                    username TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'unknown',
                    display_name TEXT NOT NULL DEFAULT '',
                    source_filename TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT '',
                    transcript_path TEXT NOT NULL DEFAULT '',
                    input_path TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    progress_json TEXT NOT NULL DEFAULT '{}',
                    options_json TEXT NOT NULL DEFAULT '{}',
                    client_ip TEXT NOT NULL DEFAULT '',
                    audio_duration_s REAL NOT NULL DEFAULT 0,
                    total_elapsed_s REAL NOT NULL DEFAULT 0,
                    tab_id TEXT NOT NULL DEFAULT '',
                    selected_engines_json TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_user_created "
                "ON jobs(user_id, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_username_created "
                "ON jobs(username, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
            )
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            conn.commit()
        finally:
            conn.close()


def schema_version() -> int:
    init_jobs_db()
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            return int(row["value"]) if row else 0
        finally:
            conn.close()


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value if value is not None else {}, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _json_loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _transcript_path_from_manifest(data: dict[str, Any]) -> str:
    explicit = data.get("transcript_path") or ""
    if explicit:
        return str(explicit)
    results = data.get("results") or {}
    if isinstance(results, dict):
        for payload in results.values():
            if isinstance(payload, dict) and payload.get("download_path"):
                return str(payload["download_path"])
    return ""


def _input_path_from_manifest(data: dict[str, Any]) -> str:
    for key in ("input_path", "source_path"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _options_from_manifest(data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "language",
        "diarization",
        "enhance",
        "max_speakers",
        "selected_engines",
        "output_name",
    )
    return {k: data[k] for k in keys if k in data}


def _engines_json(merged: dict[str, Any], current: dict[str, Any]) -> str:
    engines = merged.get("selected_engines")
    if engines is None and current.get("selected_engines_json"):
        return current["selected_engines_json"]
    if isinstance(engines, str):
        engines = [engines] if engines else []
    return _json_dumps(engines if engines is not None else [])


def _progress_json(patch: dict[str, Any], merged: dict[str, Any], current: dict[str, Any]) -> str:
    progress = patch.get("progress", merged.get("progress"))
    if progress is None and current.get("progress_json"):
        return current["progress_json"]
    return _json_dumps(progress if progress is not None else {})


def _options_json(merged: dict[str, Any], current: dict[str, Any]) -> str:
    options = _options_from_manifest(merged)
    if current.get("options_json") and not options:
        return current["options_json"]
    prev_opts = _json_loads(current.get("options_json"), {})
    if isinstance(prev_opts, dict):
        prev_opts.update(options)
        options = prev_opts
    return _json_dumps(options)


def _str_field(*values: Any, default: str = "") -> str:
    for value in values:
        if value is not None and value != "":
            return str(value)
    return default


def _build_job_row(
    job_id: str,
    patch: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    merged = {**current, **{k: v for k, v in patch.items() if v is not None}}
    merged["job_id"] = job_id
    transcript_path = (
        _transcript_path_from_manifest(merged) or current.get("transcript_path") or ""
    )
    input_path = _input_path_from_manifest(merged) or current.get("input_path") or ""
    return {
        "job_id": job_id,
        "user_id": int(merged.get("user_id") or 0),
        "username": _str_field(merged.get("username")),
        "status": _str_field(merged.get("status"), current.get("status"), default="unknown"),
        "display_name": _str_field(
            merged.get("display_name"), current.get("display_name")
        ),
        "source_filename": _str_field(
            merged.get("source_filename"), current.get("source_filename")
        ),
        "created_at": _str_field(merged.get("created_at"), current.get("created_at")),
        "updated_at": _str_field(merged.get("updated_at"), current.get("updated_at")),
        "transcript_path": str(transcript_path),
        "input_path": str(input_path),
        "error": _str_field(merged.get("error")),
        "progress_json": _progress_json(patch, merged, current),
        "options_json": _options_json(merged, current),
        "client_ip": _str_field(merged.get("client_ip"), current.get("client_ip")),
        "audio_duration_s": float(merged.get("audio_duration_s") or 0.0),
        "total_elapsed_s": float(merged.get("total_elapsed_s") or 0.0),
        "tab_id": _str_field(merged.get("tab_id"), current.get("tab_id")),
        "selected_engines_json": _engines_json(merged, current),
    }


def _execute_upsert(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    placeholders = ", ".join(f":{c}" for c in _JOB_COLUMNS)
    columns = ", ".join(_JOB_COLUMNS)
    updates = ", ".join(f"{c}=excluded.{c}" for c in _JOB_COLUMNS if c != "job_id")
    conn.execute(
        f"""
        INSERT INTO jobs ({columns}) VALUES ({placeholders})
        ON CONFLICT(job_id) DO UPDATE SET {updates}
        """,
        row,
    )
    conn.commit()


def upsert_job(job_id: str, patch: dict[str, Any]) -> None:
    """Insert or merge-update a job row from a manifest-style patch."""
    if not job_id:
        return
    init_jobs_db()
    with _DB_LOCK:
        conn = _connect()
        try:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            current: dict[str, Any] = dict(existing) if existing else {}
            _execute_upsert(conn, _build_job_row(job_id, patch, current))
        finally:
            conn.close()


def get_job_row(job_id: str) -> dict[str, Any] | None:
    init_jobs_db()
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return _row_to_summary(dict(row)) if row else None
        finally:
            conn.close()


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        return [value]
    return []


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _row_to_summary(row: dict[str, Any]) -> dict[str, Any]:
    engines = _as_list(_json_loads(row.get("selected_engines_json"), []))
    return {
        "job_id": row.get("job_id") or "",
        "created_at": row.get("created_at") or "",
        "updated_at": row.get("updated_at") or "",
        "status": row.get("status") or "unknown",
        "display_name": row.get("display_name") or "",
        "source_filename": row.get("source_filename") or "",
        "selected_engines": engines,
        "audio_duration_s": float(row.get("audio_duration_s") or 0.0),
        "total_elapsed_s": float(row.get("total_elapsed_s") or 0.0),
        "tab_id": row.get("tab_id") or "",
        "client_ip": row.get("client_ip") or "",
        "user_id": int(row.get("user_id") or 0),
        "username": row.get("username") or "",
        "progress": _as_dict(_json_loads(row.get("progress_json"), {})),
        "results": {},
        "transcript_path": row.get("transcript_path") or "",
        "input_path": row.get("input_path") or "",
        "error": row.get("error") or "",
        "options": _as_dict(_json_loads(row.get("options_json"), {})),
    }


def list_job_rows(
    limit: int = 50,
    *,
    client_ip: str | None = None,
    username: str | None = None,
    user_id: int | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Return job summary rows sorted by created_at descending."""
    init_jobs_db()
    clauses: list[str] = []
    params: list[Any] = []
    if user_id is not None and int(user_id) > 0:
        clauses.append("user_id = ?")
        params.append(int(user_id))
    elif username:
        clauses.append("LOWER(username) = LOWER(?)")
        params.append(username.strip())
    elif client_ip:
        clauses.append("client_ip = ?")
        params.append(client_ip)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        f"SELECT * FROM jobs {where} "
        "ORDER BY created_at DESC LIMIT ?"
    )
    params.append(max(1, int(limit)))
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_summary(dict(r)) for r in rows]
        finally:
            conn.close()


def list_resumable_jobs() -> list[dict[str, Any]]:
    """Jobs that were queued/running when the process stopped."""
    init_jobs_db()
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') "
                "ORDER BY created_at ASC"
            ).fetchall()
            return [_row_to_summary(dict(r)) for r in rows]
        finally:
            conn.close()


def import_job_manifest(data: dict[str, Any], *, job_id: str | None = None) -> str:
    """Upsert one JSON manifest into SQLite; return job_id."""
    jid = job_id or str(data.get("job_id") or "")
    if not jid:
        raise ValueError("job_id required")
    upsert_job(jid, data)
    return jid


def migrate_json_jobs(job_dir: Path | None = None) -> int:
    """Idempotent import of storage/jobs/*.json into SQLite. Returns import count."""
    from backend.storage import JOB_DIR

    directory = job_dir or JOB_DIR
    if not directory.is_dir():
        return 0
    init_jobs_db()
    imported = 0
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping corrupt job manifest %s", path)
            continue
        if not isinstance(data, dict):
            continue
        jid = str(data.get("job_id") or path.stem)
        try:
            import_job_manifest(data, job_id=jid)
            imported += 1
        except ValueError:
            continue
    logger.info("Migrated %d job manifest(s) into %s", imported, jobs_db_path())
    return imported


__all__ = [
    "SCHEMA_VERSION",
    "get_job_row",
    "import_job_manifest",
    "init_jobs_db",
    "jobs_db_path",
    "list_job_rows",
    "list_resumable_jobs",
    "migrate_json_jobs",
    "schema_version",
    "upsert_job",
]
