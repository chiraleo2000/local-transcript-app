"""REST API: auth + headless transcription jobs (no open browser required)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from backend.auth_users import (
    authenticate_user,
    init_user_db,
    issue_session_token,
    register_user,
    user_public_dict,
    verify_session_token,
)
from backend.job_queue import (
    cancel_job_by_id,
    get_job_progress,
    release_queue_slot,
    snapshot_queue,
    submit_background_job,
    try_reserve_queue_slot,
)
from backend.pipeline import JobMeta, run_transcription_job
from backend.register_page import build_register_routes
from backend.storage import (
    INPUT_DIR,
    ensure_app_dirs,
    list_jobs,
    load_job,
    new_job_id,
    safe_name,
    write_job_record,
)

logger = logging.getLogger(__name__)

SESSION_COOKIE = "lta_session"
_ERR_JOB_NOT_FOUND = "Job not found."
_ERR_FORBIDDEN = "Forbidden."


def _json(data: Any, status: int = 200, **headers: str) -> JSONResponse:
    response = JSONResponse(data, status_code=status)
    for key, value in headers.items():
        response.headers[key] = value
    return response


def _error(message: str, status: int = 400) -> JSONResponse:
    return _json({"ok": False, "error": message}, status=status)


async def _parse_json(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:  # pylint: disable=broad-exception-caught
        return {}
    return payload if isinstance(payload, dict) else {}


def _token_from_request(request: Request) -> str | None:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    cookie = request.cookies.get(SESSION_COOKIE)
    if cookie:
        return cookie.strip()
    return None


def _user_from_request(request: Request):
    return verify_session_token(_token_from_request(request))


def _require_user(request: Request):
    user = _user_from_request(request)
    if user is None:
        return None, _error("Authentication required.", status=401)
    return user, None


def _cookie_secure(request: Request) -> bool:
    """Secure cookies on HTTPS, or when APP_COOKIE_SECURE=1 (LAN HTTP stays off)."""
    flag = os.getenv("APP_COOKIE_SECURE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    xf = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    return xf == "https" or request.url.scheme == "https"


def _set_session_cookie(response: Response, token: str, request: Request) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(request),
        max_age=7 * 24 * 3600,
    )


async def auth_register(request: Request) -> Response:
    payload = await _parse_json(request)
    try:
        user = register_user(
            str(payload.get("username") or ""),
            str(payload.get("password") or ""),
        )
    except ValueError as exc:
        return _error(str(exc), status=400)
    token = issue_session_token(user)
    response = _json({"ok": True, "user": user_public_dict(user), "token": token})
    _set_session_cookie(response, token, request)
    return response


async def auth_login(request: Request) -> Response:
    payload = await _parse_json(request)
    user = authenticate_user(
        str(payload.get("username") or ""),
        str(payload.get("password") or ""),
    )
    if user is None:
        return _error("Invalid username or password.", status=401)
    token = issue_session_token(user)
    response = _json({"ok": True, "user": user_public_dict(user), "token": token})
    _set_session_cookie(response, token, request)
    return response


def auth_logout(_request: Request) -> Response:
    response = _json({"ok": True})
    response.delete_cookie(SESSION_COOKIE)
    return response


def auth_me(request: Request) -> Response:
    user, err = _require_user(request)
    if err is not None:
        return err
    return _json({"ok": True, "user": user_public_dict(user)})


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


async def jobs_create(request: Request) -> Response:
    user, err = _require_user(request)
    if err is not None:
        return err
    if not try_reserve_queue_slot():
        return _error(
            f"Job queue full (max {os.getenv('API_MAX_QUEUED_JOBS', '4')} jobs).",
            status=429,
        )

    ensure_app_dirs()
    form = await request.form()
    upload = form.get("file") or form.get("audio") or form.get("media")
    if upload is None or not hasattr(upload, "filename"):
        release_queue_slot(started=False)
        return _error("Missing multipart file field (file|audio|media).")

    filename = safe_name(getattr(upload, "filename", None) or "upload.bin")
    language = str(form.get("language") or "Thai")
    diarization = _parse_bool(form.get("diarization"), True)
    enhance = _parse_bool(form.get("enhance"), True)
    max_speakers = int(form.get("max_speakers") or 0)
    engines_raw = str(form.get("engines") or form.get("selected_engines") or "Auto")
    selected_engines = [e.strip() for e in engines_raw.split(",") if e.strip()] or ["Auto"]

    job_id = new_job_id()
    dest = INPUT_DIR / f"{job_id}_{filename}"
    client_ip = request.client.host if request.client else ""
    user_id = user.id
    username = user.username
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
    except OSError as exc:
        release_queue_slot(started=False)
        return _error(f"Failed to store upload: {exc}", status=500)

    write_job_record(
        job_id,
        {
            "job_id": job_id,
            "status": "queued",
            "display_name": Path(filename).stem,
            "source_filename": filename,
            "source_path": str(dest),
            "user_id": user_id,
            "username": username,
            "language": language,
            "diarization": diarization,
            "enhance": enhance,
            "max_speakers": max_speakers,
            "selected_engines": selected_engines,
            "client_ip": client_ip,
        },
    )

    def _worker(handle) -> None:
        try:
            run_transcription_job(
                media_path=str(dest),
                selected_engines=selected_engines,
                language=language,
                diarization=diarization,
                max_speakers=max_speakers,
                enhance=enhance,
                cancel_event=handle.cancel_event,
                progress=handle.progress,
                meta=JobMeta(
                    tab_id=f"api:{job_id}",
                    display_name=Path(filename).stem,
                    source_filename=filename,
                    output_name=Path(filename).stem,
                    client_ip=client_ip,
                    user_id=user_id,
                    username=username,
                ),
                job_id=job_id,
            )
        except RuntimeError as exc:
            status = "cancelled" if "cancel" in str(exc).lower() else "failed"
            write_job_record(
                job_id,
                {
                    "status": status,
                    "error": str(exc),
                    "user_id": user_id,
                    "username": username,
                },
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("API job %s failed", job_id)
            write_job_record(
                job_id,
                {
                    "status": "failed",
                    "error": str(exc),
                    "user_id": user_id,
                    "username": username,
                },
            )

    try:
        submit_background_job(job_id, _worker)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        release_queue_slot(started=False)
        write_job_record(job_id, {"status": "failed", "error": str(exc)})
        return _error(f"Failed to start job: {exc}", status=500)
    return _json(
        {
            "ok": True,
            "job_id": job_id,
            "status": "queued",
            "queue": snapshot_queue(),
        },
        status=202,
    )


def jobs_list(request: Request) -> Response:
    user, err = _require_user(request)
    if err is not None:
        return err
    try:
        limit = int(request.query_params.get("limit") or 50)
    except ValueError:
        limit = 50
    rows = list_jobs(limit, username=user.username, user_id=user.id)
    return _json({"ok": True, "jobs": rows, "queue": snapshot_queue()})


def jobs_get(request: Request) -> Response:
    user, err = _require_user(request)
    if err is not None:
        return err
    job_id = request.path_params.get("job_id") or ""
    job = load_job(job_id)
    if job is None:
        return _error(_ERR_JOB_NOT_FOUND, status=404)
    if not _job_owned_by(job, user):
        return _error(_ERR_FORBIDDEN, status=403)
    live = get_job_progress(job_id)
    payload = dict(job)
    if live is not None:
        payload["live_progress"] = live.snapshot()
    return _json({"ok": True, "job": payload})


def jobs_transcript(request: Request) -> Response:
    user, err = _require_user(request)
    if err is not None:
        return err
    job_id = request.path_params.get("job_id") or ""
    job = load_job(job_id)
    if job is None:
        return _error(_ERR_JOB_NOT_FOUND, status=404)
    if not _job_owned_by(job, user):
        return _error(_ERR_FORBIDDEN, status=403)
    results = job.get("results") or {}
    path = None
    for payload in results.values():
        if isinstance(payload, dict) and payload.get("download_path"):
            path = payload["download_path"]
            break
    if not path or not Path(path).is_file():
        return _error("Transcript not ready.", status=404)
    return FileResponse(
        path,
        filename=Path(path).name,
        media_type="text/plain; charset=utf-8",
    )


def jobs_cancel(request: Request) -> Response:
    user, err = _require_user(request)
    if err is not None:
        return err
    job_id = request.path_params.get("job_id") or ""
    job = load_job(job_id)
    if job is None:
        return _error(_ERR_JOB_NOT_FOUND, status=404)
    if not _job_owned_by(job, user):
        return _error(_ERR_FORBIDDEN, status=403)
    cancelled = cancel_job_by_id(job_id)
    if cancelled:
        write_job_record(job_id, {"status": "cancelled", "error": "Cancelled by user."})
    return _json({"ok": True, "cancelled": cancelled, "job_id": job_id})


def jobs_queue(request: Request) -> Response:
    _, err = _require_user(request)
    if err is not None:
        return err
    return _json({"ok": True, "queue": snapshot_queue()})


def _job_owned_by(job: dict[str, Any], user) -> bool:
    if int(job.get("user_id") or 0) == int(user.id):
        return True
    username = (job.get("username") or "").strip().lower()
    return bool(username) and username == user.username.lower()


def build_api_routes() -> list[Route]:
    init_user_db()
    return [
        *build_register_routes(),
        Route("/api/auth/register", auth_register, methods=["POST"]),
        Route("/api/auth/login", auth_login, methods=["POST"]),
        Route("/api/auth/logout", auth_logout, methods=["POST"]),
        Route("/api/auth/me", auth_me, methods=["GET"]),
        Route("/api/jobs", jobs_create, methods=["POST"]),
        Route("/api/jobs", jobs_list, methods=["GET"]),
        Route("/api/jobs/queue", jobs_queue, methods=["GET"]),
        Route("/api/jobs/{job_id}", jobs_get, methods=["GET"]),
        Route("/api/jobs/{job_id}/transcript", jobs_transcript, methods=["GET"]),
        Route("/api/jobs/{job_id}/cancel", jobs_cancel, methods=["POST"]),
    ]


__all__ = [
    "SESSION_COOKIE",
    "build_api_routes",
]
