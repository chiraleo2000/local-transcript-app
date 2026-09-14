"""HTTP client for the storage sidecar. Raises if the service is unreachable."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class StorageUnavailable(RuntimeError):
    """Sidecar is configured but the request could not be completed."""


def storage_url() -> str:
    return os.getenv("APP_STORAGE_URL", "").strip().rstrip("/")


def storage_configured() -> bool:
    return bool(storage_url())


def _request(method: str, path: str, payload: dict[str, Any] | None = None, query: dict | None = None) -> Any:
    base = storage_url()
    if not base:
        raise StorageUnavailable("APP_STORAGE_URL is not set")
    url = f"{base}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode({k: v for k, v in query.items() if v not in (None, '')})}"
    data = None
    headers = {"Accept": "application/json"}
    token = os.getenv("APP_STORAGE_TOKEN", "").strip()
    if token:
        headers["X-Storage-Token"] = token
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code in {401, 404, 409}:
            try:
                parsed = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed = {"error": body or exc.reason}
            err = StorageUnavailable(str(parsed.get("error") or exc.reason))
            err.status = exc.code  # type: ignore[attr-defined]
            err.payload = parsed  # type: ignore[attr-defined]
            raise err from exc
        raise StorageUnavailable(f"storage {method} {path} failed: HTTP {exc.code} {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise StorageUnavailable(f"storage sidecar unreachable at {base}: {exc}") from exc


def upsert_job(job_id: str, patch: dict[str, Any]) -> None:
    body = dict(patch)
    body["job_id"] = job_id
    _request("POST", "/v1/jobs", body)


def get_job(job_id: str) -> dict[str, Any] | None:
    try:
        return _request("GET", f"/v1/jobs/{urllib.parse.quote(job_id, safe='')}")
    except StorageUnavailable as exc:
        if getattr(exc, "status", None) == 404:
            return None
        raise


def list_jobs(
    limit: int = 50,
    *,
    client_ip: str | None = None,
    username: str | None = None,
    user_id: int | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    payload = _request(
        "GET",
        "/v1/jobs",
        query={
            "limit": limit,
            "client_ip": client_ip,
            "username": username,
            "user_id": user_id,
            "status": status,
        },
    )
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    return jobs if isinstance(jobs, list) else []


def list_resumable() -> list[dict[str, Any]]:
    payload = _request("GET", "/v1/jobs/resumable")
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    return jobs if isinstance(jobs, list) else []


def save_result(**payload: Any) -> None:
    _request("POST", "/v1/results", payload)


def schema_version() -> int:
    payload = _request("GET", "/v1/meta")
    return int(payload.get("schema_version") or 0)


def register_user(username: str, password: str) -> dict[str, Any]:
    return _request("POST", "/v1/users", {"username": username, "password": password})


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    try:
        return _request("POST", "/v1/users/authenticate", {"username": username, "password": password})
    except StorageUnavailable as exc:
        if getattr(exc, "status", None) == 401:
            return None
        raise


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    try:
        return _request("GET", f"/v1/users/{int(user_id)}")
    except StorageUnavailable as exc:
        if getattr(exc, "status", None) == 404:
            return None
        raise


def get_user_by_username(username: str) -> dict[str, Any] | None:
    try:
        return _request("GET", "/v1/users", query={"username": username})
    except StorageUnavailable as exc:
        if getattr(exc, "status", None) == 404:
            return None
        raise
