"""Gradio UI session idle timeout (default 60 minutes)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


def session_ttl_s() -> int:
    """Idle session lifetime in seconds (APP_SESSION_TTL_S, default 3600 = 60 min)."""
    try:
        return max(60, int(os.getenv("APP_SESSION_TTL_S", "3600")))
    except ValueError:
        return 3600


def _cookie_names(app: Any) -> tuple[str, str]:
    cid = getattr(app, "cookie_id", "") or ""
    return f"access-token-{cid}", f"access-token-unsecure-{cid}"


def _request_token(request: Request, app: Any) -> str | None:
    secure_name, unsecure_name = _cookie_names(app)
    return request.cookies.get(secure_name) or request.cookies.get(unsecure_name)


def _clear_auth_cookies(response: Response, app: Any) -> None:
    secure_name, unsecure_name = _cookie_names(app)
    response.delete_cookie(secure_name, path="/")
    response.delete_cookie(unsecure_name, path="/")
    # Also clear API session cookie used by REST auth.
    response.delete_cookie("lta_session", path="/")


def _username_for_token(app: Any, token: str | None) -> str | None:
    if not token:
        return None
    tokens: dict = getattr(app, "tokens", None) or {}
    name = tokens.get(token)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _user_has_inflight_job(username: str | None) -> bool:
    """True when this account still has a queued/running transcription on disk."""
    if not username:
        return False
    try:
        from backend.job_status import job_is_in_flight
        from backend.storage import list_jobs
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    try:
        for row in list_jobs(30, username=username):
            if job_is_in_flight(row):
                return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return False


def _user_has_recent_job_activity(username: str | None, *, within_s: int = 600) -> bool:
    """True when a job for this user was updated recently (post-complete download grace)."""
    if not username:
        return False
    try:
        from datetime import datetime, timezone

        from backend.storage import list_jobs
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    now = datetime.now(timezone.utc)
    try:
        for row in list_jobs(10, username=username):
            raw = row.get("updated_at") or row.get("created_at") or ""
            if not isinstance(raw, str) or not raw:
                continue
            try:
                stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            if (now - stamp).total_seconds() <= within_s:
                return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return False


def _should_extend_idle_session(username: str | None) -> bool:
    return _user_has_inflight_job(username) or _user_has_recent_job_activity(username)


def _touch_session_meta(meta: dict, now: float, username: Any = None) -> None:
    meta["last"] = now
    if username is not None:
        meta["username"] = username
    elif "username" not in meta:
        meta["username"] = None


def _refresh_active_token(
    tokens: dict,
    token: str,
    meta: dict,
    now: float,
) -> None:
    _touch_session_meta(meta, now)
    if meta.get("username") is None:
        meta["username"] = tokens.get(token)


def _expire_or_extend_token(
    *,
    app: Any,
    tokens: dict,
    sessions: dict,
    token: str,
    meta: dict,
    ttl: int,
    now: float,
) -> bool:
    """Return True when the token should be expired (cookies cleared)."""
    if (now - float(meta.get("last") or 0)) <= ttl:
        _refresh_active_token(tokens, token, meta, now)
        return False
    username = _username_for_token(app, token) or meta.get("username")
    # Do not kick users mid-transcription or right after completion —
    # SSE progress does not refresh idle timers, and Download .txt
    # would otherwise 401 immediately after a long job finishes.
    if _should_extend_idle_session(username if isinstance(username, str) else None):
        _touch_session_meta(meta, now, username)
        logger.info("Extended Gradio session for @%s — active/recent job.", username)
        return False
    tokens.pop(token, None)
    sessions.pop(token, None)
    logger.info("Gradio session expired after %ss idle — login required.", ttl)
    return True


def _register_new_login_tokens(
    tokens: dict,
    sessions: dict,
    before: set,
) -> None:
    for new_token in set(tokens.keys()) - before:
        sessions[new_token] = {
            "last": time.time(),
            "created": time.time(),
            "username": tokens.get(new_token),
        }


class SessionIdleMiddleware(BaseHTTPMiddleware):
    """Expire Gradio auth tokens after APP_SESSION_TTL_S of inactivity.

    Long transcriptions often hold a single SSE connection without new HTTP
    requests. Keep the session alive while the user still has an in-flight job,
    and treat keepalive pings as normal activity.
    """

    def __init__(self, app, gradio_app: Any):
        super().__init__(app)
        self.gradio_app = gradio_app

    async def dispatch(self, request: Request, call_next) -> Response:
        app = self.gradio_app
        tokens: dict = getattr(app, "tokens", None) or {}
        sessions: dict = getattr(app, "_lta_sessions", None)
        if sessions is None:
            sessions = {}
            app._lta_sessions = sessions

        ttl = session_ttl_s()
        now = time.time()
        token = _request_token(request, app)
        expired = False

        if token and token in tokens:
            meta = sessions.get(token)
            if meta is None:
                sessions[token] = {
                    "last": now,
                    "created": now,
                    "username": tokens.get(token),
                }
            else:
                expired = _expire_or_extend_token(
                    app=app,
                    tokens=tokens,
                    sessions=sessions,
                    token=token,
                    meta=meta,
                    ttl=ttl,
                    now=now,
                )
        elif token and token not in tokens:
            # Stale cookie after logout/expiry — force clean login screen.
            expired = True

        before = set(tokens.keys())
        response = await call_next(request)
        _register_new_login_tokens(tokens, sessions, before)
        if expired:
            _clear_auth_cookies(response, app)
        return response


def force_logout(request: Request) -> Response:
    """Clear Gradio + API session cookies and show the login page."""
    from starlette.responses import RedirectResponse

    app = request.app
    tokens: dict = getattr(app, "tokens", None) or {}
    sessions: dict = getattr(app, "_lta_sessions", None) or {}
    token = _request_token(request, app)
    user = tokens.get(token) if token else None
    if token:
        tokens.pop(token, None)
        sessions.pop(token, None)
    # Also drop other sessions for the same user (matches Gradio default).
    if user:
        for other in [k for k, name in tokens.items() if name == user]:
            tokens.pop(other, None)
            sessions.pop(other, None)
    response = RedirectResponse(url="/", status_code=302)
    _clear_auth_cookies(response, app)
    logger.info("User logged out%s.", f" ({user})" if user else "")
    return response


def install_gradio_session_timeout(app: Any) -> None:
    """Attach idle-timeout middleware once per Gradio FastAPI app."""
    if getattr(app, "_lta_session_middleware", False):
        return
    if getattr(app, "auth", None) is None and getattr(app, "auth_dependency", None) is None:
        return
    app._lta_sessions = {}
    app.add_middleware(SessionIdleMiddleware, gradio_app=app)
    app._lta_session_middleware = True
    logger.info("Gradio idle session timeout enabled (%ss).", session_ttl_s())
