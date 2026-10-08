"""HTTP client for the storage sidecar. Raises if the service is unreachable."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class StorageUnavailable(RuntimeError):
    """Sidecar is configured but the request could not be completed."""


_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


def storage_url() -> str:
    return os.getenv("APP_STORAGE_URL", "").strip().rstrip("/")


def storage_configured() -> bool:
    return bool(storage_url())


def _request(method: str, path: str, payload: dict[str, Any] | None = None, query: dict | None = None) -> Any:
    """Call the sidecar, retrying brief network and 5xx failures."""
    delay_s = 0.2
    last: StorageUnavailable | None = None
    for attempt in range(1, 4):
        try:
            return _request_once(method, path, payload, query)
        except StorageUnavailable as exc:
            last = exc
            status = getattr(exc, "status", None)
            if status not in _RETRY_STATUS and status is not None:
                raise
            if attempt == 3:
                raise
            time.sleep(delay_s)
            delay_s *= 2
    if last is not None:
        raise last
    raise StorageUnavailable("storage request failed")


def _with_query(url: str, query: dict | None) -> str:
    if not query:
        return url
    filtered = {key: value for key, value in query.items() if value not in (None, "")}
    return f"{url}?{urllib.parse.urlencode(filtered)}"


def _json_body(payload: dict[str, Any] | None) -> bytes | None:
    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _request_headers(token: str, *, has_body: bool) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["X-Storage-Token"] = token
    if has_body:
        headers["Content-Type"] = "application/json"
    return headers


def _parse_storage_error_body(body: str, reason: object) -> Any:
    try:
        return json.loads(body) if body else {}
    except json.JSONDecodeError:
        return {"error": body or reason}


def _error_from_http(exc: urllib.error.HTTPError, method: str, path: str) -> StorageUnavailable:
    body = exc.read().decode("utf-8", errors="replace")
    if exc.code in {401, 404, 409}:
        parsed = _parse_storage_error_body(body, exc.reason)
        message = parsed.get("error") if isinstance(parsed, dict) else None
        err = StorageUnavailable(str(message or exc.reason))
        err.payload = parsed  # type: ignore[attr-defined]
    else:
        err = StorageUnavailable(f"storage {method} {path} failed: HTTP {exc.code} {body}")
    err.status = exc.code  # type: ignore[attr-defined]
    return err


def _request_once(method: str, path: str, payload: dict[str, Any] | None = None, query: dict | None = None) -> Any:
    base = storage_url()
    if not base:
        raise StorageUnavailable("APP_STORAGE_URL is not set")
    data = _json_body(payload)
    token = os.getenv("APP_STORAGE_TOKEN", "").strip()
    req = urllib.request.Request(
        _with_query(f"{base}{path}", query),
        data=data,
        headers=_request_headers(token, has_body=data is not None),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise _error_from_http(exc, method, path) from exc
    except OSError as exc:
        raise StorageUnavailable(f"storage sidecar unreachable at {base}: {exc}") from exc
    return json.loads(raw) if raw else {}


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
