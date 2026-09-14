"""SQLite schema v2 used by the storage sidecar (users, jobs, results, segments)."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
_LOCK = threading.Lock()

_SEGMENT_LINE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2})\s*→\s*(\d{2}):(\d{2}):(\d{2})\]\s*"
    r"\[(\S+)\]:\s*(.*)$"
)
_SPEAKER_LINE = re.compile(r"^\[(SPEAKER_\d+|[^\[\]]+)\]:\s*(.*)$")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path) -> None:
    with _LOCK:
        conn = connect(db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                );
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
                );
                CREATE TABLE IF NOT EXISTS transcript_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    engine TEXT NOT NULL DEFAULT '',
                    language TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL DEFAULT '',
                    duration_s REAL NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT '',
                    UNIQUE(job_id, engine)
                );
                CREATE TABLE IF NOT EXISTS transcript_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id INTEGER NOT NULL,
                    seq INTEGER NOT NULL,
                    speaker TEXT NOT NULL DEFAULT '',
                    start_s REAL NOT NULL DEFAULT 0,
                    end_s REAL NOT NULL DEFAULT 0,
                    text TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(result_id) REFERENCES transcript_results(id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_user_created
                    ON jobs(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_jobs_username_created
                    ON jobs(username, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_results_job ON transcript_results(job_id);
                CREATE INDEX IF NOT EXISTS idx_segments_result
                    ON transcript_segments(result_id, seq);
                """
            )
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            else:
                conn.execute(
                    "UPDATE meta SET value = ? WHERE key = 'schema_version'",
                    (str(SCHEMA_VERSION),),
                )
            conn.commit()
        finally:
            conn.close()


def schema_version(db_path: Path) -> int:
    init_db(db_path)
    with _LOCK:
        conn = connect(db_path)
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            return int(row["value"]) if row else 0
        finally:
            conn.close()


def _hms_to_s(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_transcript_segments(text: str) -> list[dict[str, Any]]:
    """Pull speaker turns out of a formatted transcript."""
    segments: list[dict[str, Any]] = []
    if not text:
        return segments
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        timed = _SEGMENT_LINE.match(stripped)
        if timed:
            segments.append({
                "speaker": timed.group(7),
                "start_s": _hms_to_s(*timed.group(1, 2, 3)),
                "end_s": _hms_to_s(*timed.group(4, 5, 6)),
                "text": timed.group(8).strip(),
            })
            continue
        speaker = _SPEAKER_LINE.match(stripped)
        if speaker:
            segments.append({
                "speaker": speaker.group(1),
                "start_s": 0.0,
                "end_s": 0.0,
                "text": speaker.group(2).strip(),
            })
    return segments


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        return [value]
    return []


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _json_loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _results_for_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT * FROM transcript_results WHERE job_id = ? ORDER BY id",
        (job_id,),
    ).fetchall()
    results: dict[str, Any] = {}
    for row in rows:
        engine = row["engine"] or "transcript"
        segs = conn.execute(
            "SELECT speaker, start_s, end_s, text FROM transcript_segments "
            "WHERE result_id = ? ORDER BY seq",
            (row["id"],),
        ).fetchall()
        results[engine] = {
            "text": row["text"] or "",
            "elapsed": float(row["duration_s"] or 0.0),
            "download_path": "",
            "language": row["language"] or "",
            "error": row["error"] or "",
            "note": "",
            "segments": [dict(seg) for seg in segs],
        }
    return results


def row_to_summary(row: dict[str, Any], results: dict[str, Any] | None = None) -> dict[str, Any]:
    engines = _as_list(_json_loads(row.get("selected_engines_json"), []))
    payload = {
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
        "results": results or {},
        "transcript_path": row.get("transcript_path") or "",
        "input_path": row.get("input_path") or "",
        "error": row.get("error") or "",
        "options": _as_dict(_json_loads(row.get("options_json"), {})),
    }
    return payload


def upsert_job_row(db_path: Path, row: dict[str, Any]) -> None:
    from backend.jobs_db import _JOB_COLUMNS

    init_db(db_path)
    columns = ", ".join(_JOB_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in _JOB_COLUMNS)
    updates = ", ".join(f"{c}=excluded.{c}" for c in _JOB_COLUMNS if c != "job_id")
    with _LOCK:
        conn = connect(db_path)
        try:
            conn.execute(
                f"""
                INSERT INTO jobs ({columns}) VALUES ({placeholders})
                ON CONFLICT(job_id) DO UPDATE SET {updates}
                """,
                row,
            )
            conn.commit()
        finally:
            conn.close()


def get_job(db_path: Path, job_id: str) -> dict[str, Any] | None:
    init_db(db_path)
    with _LOCK:
        conn = connect(db_path)
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            return row_to_summary(dict(row), _results_for_job(conn, job_id))
        finally:
            conn.close()


def list_jobs(
    db_path: Path,
    limit: int = 50,
    *,
    client_ip: str | None = None,
    username: str | None = None,
    user_id: int | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    init_db(db_path)
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
    sql = f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ?"
    params.append(max(1, int(limit)))
    with _LOCK:
        conn = connect(db_path)
        try:
            rows = conn.execute(sql, params).fetchall()
            out = []
            for row in rows:
                data = dict(row)
                out.append(row_to_summary(data, _results_for_job(conn, data["job_id"])))
            return out
        finally:
            conn.close()


def list_resumable(db_path: Path) -> list[dict[str, Any]]:
    init_db(db_path)
    with _LOCK:
        conn = connect(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') "
                "ORDER BY created_at ASC"
            ).fetchall()
            return [
                row_to_summary(dict(row), _results_for_job(conn, dict(row)["job_id"]))
                for row in rows
            ]
        finally:
            conn.close()


def save_result(
    db_path: Path,
    *,
    job_id: str,
    engine: str,
    text: str,
    language: str = "",
    duration_s: float = 0.0,
    error: str = "",
    segments: list[dict[str, Any]] | None = None,
    transcript_path: str = "",
) -> None:
    init_db(db_path)
    segs = segments if segments is not None else parse_transcript_segments(text)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _LOCK:
        conn = connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO transcript_results
                    (job_id, engine, language, text, duration_s, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, engine) DO UPDATE SET
                    language=excluded.language,
                    text=excluded.text,
                    duration_s=excluded.duration_s,
                    error=excluded.error,
                    created_at=excluded.created_at
                """,
                (job_id, engine or "transcript", language, text or "", float(duration_s or 0), error or "", now),
            )
            result_id = conn.execute(
                "SELECT id FROM transcript_results WHERE job_id = ? AND engine = ?",
                (job_id, engine or "transcript"),
            ).fetchone()["id"]
            conn.execute(
                "DELETE FROM transcript_segments WHERE result_id = ?",
                (result_id,),
            )
            for seq, seg in enumerate(segs):
                conn.execute(
                    """
                    INSERT INTO transcript_segments
                        (result_id, seq, speaker, start_s, end_s, text)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result_id,
                        seq,
                        str(seg.get("speaker") or ""),
                        float(seg.get("start_s") or seg.get("start") or 0),
                        float(seg.get("end_s") or seg.get("end") or 0),
                        str(seg.get("text") or ""),
                    ),
                )
            if transcript_path:
                conn.execute(
                    "UPDATE jobs SET transcript_path = ? WHERE job_id = ?",
                    (transcript_path, job_id),
                )
            conn.commit()
        finally:
            conn.close()


def upsert_user(
    db_path: Path,
    *,
    username: str,
    password_hash: str,
    user_id: int | None = None,
    created_at: float | None = None,
    is_active: int = 1,
) -> dict[str, Any]:
    init_db(db_path)
    stamp = time.time() if created_at is None else float(created_at)
    with _LOCK:
        conn = connect(db_path)
        try:
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ? COLLATE NOCASE",
                (username,),
            ).fetchone()
            if existing is not None:
                row = conn.execute(
                    "SELECT id, username, is_active FROM users WHERE id = ?",
                    (existing["id"],),
                ).fetchone()
                return dict(row)
            if user_id:
                conn.execute(
                    "INSERT INTO users (id, username, password_hash, created_at, is_active) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (int(user_id), username, password_hash, stamp, int(is_active)),
                )
                new_id = int(user_id)
            else:
                cur = conn.execute(
                    "INSERT INTO users (username, password_hash, created_at, is_active) "
                    "VALUES (?, ?, ?, ?)",
                    (username, password_hash, stamp, int(is_active)),
                )
                new_id = int(cur.lastrowid)
            conn.commit()
            return {"id": new_id, "username": username, "is_active": int(is_active)}
        finally:
            conn.close()


def get_user(db_path: Path, *, user_id: int | None = None, username: str | None = None) -> dict[str, Any] | None:
    init_db(db_path)
    with _LOCK:
        conn = connect(db_path)
        try:
            if user_id:
                row = conn.execute(
                    "SELECT id, username, password_hash, is_active FROM users WHERE id = ?",
                    (int(user_id),),
                ).fetchone()
            elif username:
                row = conn.execute(
                    "SELECT id, username, password_hash, is_active FROM users "
                    "WHERE username = ? COLLATE NOCASE",
                    (username,),
                ).fetchone()
            else:
                return None
            return dict(row) if row else None
        finally:
            conn.close()


def import_legacy(db_path: Path, import_dir: Path) -> dict[str, int]:
    """Idempotent import of host jobs.db, users.db, JSON manifests, and .txt files."""
    init_db(db_path)
    counts = {"users": 0, "jobs": 0, "transcripts": 0}
    users_db = import_dir / "users.db"
    jobs_db = import_dir / "jobs.db"
    if users_db.is_file():
        counts["users"] += _import_users(db_path, users_db)
    if jobs_db.is_file():
        counts["jobs"] += _import_jobs_db(db_path, jobs_db)
    job_dir = import_dir / "jobs"
    if job_dir.is_dir():
        counts["jobs"] += _import_json_jobs(db_path, job_dir)
    transcripts = import_dir / "transcripts"
    if transcripts.is_dir():
        counts["transcripts"] += _import_transcript_files(db_path, transcripts)
    logger.info("Legacy import from %s: %s", import_dir, counts)
    return counts


def _import_users(db_path: Path, users_db: Path) -> int:
    src = sqlite3.connect(str(users_db))
    src.row_factory = sqlite3.Row
    imported = 0
    try:
        rows = src.execute(
            "SELECT id, username, password_hash, created_at, is_active FROM users"
        ).fetchall()
    except sqlite3.Error:
        src.close()
        return 0
    src.close()
    for row in rows:
        before = get_user(db_path, username=row["username"])
        upsert_user(
            db_path,
            username=row["username"],
            password_hash=row["password_hash"],
            user_id=None if before else int(row["id"]),
            created_at=float(row["created_at"] or 0),
            is_active=int(row["is_active"] or 1),
        )
        if before is None:
            imported += 1
    return imported


def _import_jobs_db(db_path: Path, legacy: Path) -> int:
    from backend.jobs_db import _JOB_COLUMNS, _build_job_row

    src = sqlite3.connect(str(legacy))
    src.row_factory = sqlite3.Row
    try:
        rows = src.execute("SELECT * FROM jobs").fetchall()
    except sqlite3.Error:
        src.close()
        return 0
    src.close()
    imported = 0
    for row in rows:
        data = dict(row)
        jid = str(data.get("job_id") or "")
        if not jid or get_job(db_path, jid):
            continue
        patch = dict(data)
        engines = _json_loads(data.get("selected_engines_json"), [])
        patch["selected_engines"] = engines
        patch["progress"] = _json_loads(data.get("progress_json"), {})
        built = _build_job_row(jid, patch, {})
        # Keep columns the sidecar expects.
        upsert_job_row(db_path, {k: built.get(k) for k in _JOB_COLUMNS})
        imported += 1
    return imported


def _import_json_jobs(db_path: Path, job_dir: Path) -> int:
    from backend.jobs_db import _JOB_COLUMNS, _build_job_row

    imported = 0
    for path in sorted(job_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        jid = str(data.get("job_id") or path.stem)
        if get_job(db_path, jid):
            continue
        built = _build_job_row(jid, data, {})
        upsert_job_row(db_path, {k: built.get(k) for k in _JOB_COLUMNS})
        imported += 1
    return imported


def _import_transcript_files(db_path: Path, transcripts: Path) -> int:
    imported = 0
    for path in sorted(transcripts.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not text.strip():
            continue
        stem = path.stem
        job_id = stem
        engine = "imported"
        existing = get_job(db_path, stem)
        if existing is None:
            # Prefer a job whose transcript_path ends with this filename.
            init_db(db_path)
            with _LOCK:
                conn = connect(db_path)
                try:
                    row = conn.execute(
                        "SELECT job_id FROM jobs WHERE transcript_path LIKE ? LIMIT 1",
                        (f"%{path.name}",),
                    ).fetchone()
                finally:
                    conn.close()
            if row:
                job_id = row["job_id"]
                engine = "imported"
            else:
                from backend.jobs_db import _JOB_COLUMNS, _build_job_row

                built = _build_job_row(
                    stem,
                    {
                        "job_id": stem,
                        "status": "completed",
                        "display_name": path.name,
                        "source_filename": path.name,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "transcript_path": str(path),
                        "selected_engines": ["imported"],
                    },
                    {},
                )
                upsert_job_row(db_path, {k: built.get(k) for k in _JOB_COLUMNS})
        current = get_job(db_path, job_id) or {}
        if current.get("results"):
            continue
        save_result(
            db_path,
            job_id=job_id,
            engine=engine,
            text=text,
            transcript_path=str(path),
        )
        imported += 1
    return imported
