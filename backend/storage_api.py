"""HTTP API for the storage sidecar. SQLite stays in this process only.

Run: python -m backend.storage_api
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from backend.storage_store import (
    SCHEMA_VERSION,
    get_job,
    get_user,
    import_legacy,
    init_db,
    list_jobs,
    list_resumable,
    save_result,
    schema_version,
    upsert_job_row,
    upsert_user,
)

logger = logging.getLogger(__name__)


def db_path() -> Path:
    raw = os.getenv("APP_STORAGE_DB", "/data/app.db").strip() or "/data/app.db"
    return Path(raw)


def _token() -> str:
    return os.getenv("APP_STORAGE_TOKEN", "").strip()


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    data = json.loads(raw.decode("utf-8") or "{}")
    if not isinstance(data, dict):
        raise ValueError("JSON object required")
    return data


def _authorized(handler: BaseHTTPRequestHandler) -> bool:
    expected = _token()
    if not expected:
        return True
    return handler.headers.get("X-Storage-Token", "") == expected


class StorageHandler(BaseHTTPRequestHandler):
    server_version = "LtaStorage/2"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _deny(self) -> bool:
        if _authorized(self):
            return False
        self._send(401, {"error": "unauthorized"})
        return True

    def do_GET(self) -> None:  # noqa: N802
        if self._deny():
            return
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            if path in {"/health", "/"}:
                self._send(200, {"ok": True, "schema_version": SCHEMA_VERSION})
                return
            if path == "/v1/meta":
                self._send(200, {"schema_version": schema_version(db_path())})
                return
            if path == "/v1/jobs/resumable":
                self._send(200, {"jobs": list_resumable(db_path())})
                return
            if path == "/v1/jobs":
                self._send(200, {"jobs": _list_from_query(query)})
                return
            if path.startswith("/v1/jobs/"):
                job = get_job(db_path(), path.split("/", 3)[3])
                if job is None:
                    self._send(404, {"error": "not found"})
                    return
                self._send(200, job)
                return
            if path.startswith("/v1/users/"):
                user = get_user(db_path(), user_id=int(path.split("/", 3)[3]))
                if user is None:
                    self._send(404, {"error": "not found"})
                    return
                self._send(200, _public_user(user))
                return
            if path == "/v1/users":
                name = (query.get("username") or [""])[0]
                user = get_user(db_path(), username=name) if name else None
                if user is None:
                    self._send(404, {"error": "not found"})
                    return
                self._send(200, _public_user(user))
                return
            self._send(404, {"error": "not found"})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("storage GET failed")
            self._send(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        if self._deny():
            return
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            body = _read_json(self)
        except (json.JSONDecodeError, ValueError) as exc:
            self._send(400, {"error": str(exc)})
            return
        try:
            if path == "/v1/jobs":
                self._upsert_job(body)
                return
            if path == "/v1/results":
                self._save_result(body)
                return
            if path == "/v1/users":
                self._register(body)
                return
            if path == "/v1/users/authenticate":
                self._authenticate(body)
                return
            if path == "/v1/migrate":
                counts = import_legacy(db_path(), Path(body.get("import_dir") or _import_dir()))
                self._send(200, counts)
                return
            self._send(404, {"error": "not found"})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("storage POST failed")
            self._send(500, {"error": str(exc)})

    def _upsert_job(self, body: dict[str, Any]) -> None:
        from backend.jobs_db import _JOB_COLUMNS, _build_job_row

        job_id = str(body.get("job_id") or "")
        if not job_id:
            self._send(400, {"error": "job_id required"})
            return
        current = get_job(db_path(), job_id) or {}
        # _build_job_row expects DB column names on current.
        current_row = {
            "selected_engines_json": json.dumps(current.get("selected_engines") or []),
            "progress_json": json.dumps(current.get("progress") or {}),
            "options_json": json.dumps(current.get("options") or {}),
            **{k: current.get(k) for k in (
                "transcript_path", "input_path", "status", "display_name",
                "source_filename", "created_at", "updated_at", "client_ip",
                "tab_id", "user_id", "username", "error",
                "audio_duration_s", "total_elapsed_s",
            )},
        }
        built = _build_job_row(job_id, body, current_row)
        upsert_job_row(db_path(), {k: built.get(k) for k in _JOB_COLUMNS})
        self._send(200, get_job(db_path(), job_id) or {"job_id": job_id})

    def _save_result(self, body: dict[str, Any]) -> None:
        job_id = str(body.get("job_id") or "")
        if not job_id:
            self._send(400, {"error": "job_id required"})
            return
        save_result(
            db_path(),
            job_id=job_id,
            engine=str(body.get("engine") or "transcript"),
            text=str(body.get("text") or ""),
            language=str(body.get("language") or ""),
            duration_s=float(body.get("duration_s") or 0),
            error=str(body.get("error") or ""),
            segments=body.get("segments") if isinstance(body.get("segments"), list) else None,
            transcript_path=str(body.get("transcript_path") or ""),
        )
        self._send(200, {"ok": True, "job_id": job_id})

    def _register(self, body: dict[str, Any]) -> None:
        from backend.auth_users import _hash_password, validate_username

        name = validate_username(str(body.get("username") or ""))
        password = str(body.get("password") or "")
        if len(password) < 6 and not body.get("password_hash"):
            self._send(400, {"error": "Password must be at least 6 characters."})
            return
        if get_user(db_path(), username=name):
            self._send(409, {"error": "Username already taken."})
            return
        password_hash = str(body.get("password_hash") or "") or _hash_password(password)
        user = upsert_user(
            db_path(),
            username=name,
            password_hash=password_hash,
            user_id=int(body["id"]) if body.get("id") else None,
            created_at=body.get("created_at"),
            is_active=int(body.get("is_active") or 1),
        )
        self._send(200, _public_user(user))

    def _authenticate(self, body: dict[str, Any]) -> None:
        from backend.auth_users import _verify_password

        name = str(body.get("username") or "").strip()
        password = str(body.get("password") or "")
        user = get_user(db_path(), username=name) if name and password else None
        if user is None or int(user.get("is_active") or 0) != 1:
            self._send(401, {"error": "invalid"})
            return
        if not _verify_password(password, str(user.get("password_hash") or "")):
            self._send(401, {"error": "invalid"})
            return
        self._send(200, _public_user(user))


def _public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(user.get("id") or 0),
        "username": user.get("username") or "",
        "is_active": bool(int(user.get("is_active") or 0)),
    }


def _list_from_query(query: dict[str, list[str]]) -> list[dict[str, Any]]:
    def _first(key: str) -> str:
        return (query.get(key) or [""])[0]

    limit = int(_first("limit") or "50")
    user_raw = _first("user_id")
    return list_jobs(
        db_path(),
        limit,
        client_ip=_first("client_ip") or None,
        username=_first("username") or None,
        user_id=int(user_raw) if user_raw else None,
        status=_first("status") or None,
    )


def _import_dir() -> Path:
    raw = os.getenv("APP_IMPORT_DIR", "").strip()
    return Path(raw) if raw else Path("/import")


def _seed_user() -> None:
    from backend.auth_users import _hash_password, validate_username

    seed_user = (os.getenv("APP_SEED_USER") or "").strip()
    seed_password = (os.getenv("APP_SEED_PASSWORD") or "").strip()
    if not seed_user or not seed_password:
        return
    try:
        name = validate_username(seed_user)
    except ValueError:
        logger.warning("Invalid APP_SEED_USER; skipping storage seed.")
        return
    if get_user(db_path(), username=name):
        return
    upsert_user(db_path(), username=name, password_hash=_hash_password(seed_password))
    logger.info("Seeded storage user username=%s", name)


def serve(host: str | None = None, port: int | None = None) -> ThreadingHTTPServer:
    init_db(db_path())
    import_root = _import_dir()
    if import_root.is_dir():
        import_legacy(db_path(), import_root)
    _seed_user()
    bind_host = host or os.getenv("APP_STORAGE_HOST", "0.0.0.0")
    bind_port = port if port is not None else int(os.getenv("APP_STORAGE_PORT", "8081"))
    server = ThreadingHTTPServer((bind_host, bind_port), StorageHandler)
    logger.info("Storage API listening on %s:%s db=%s", bind_host, server.server_address[1], db_path())
    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    server = serve()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
