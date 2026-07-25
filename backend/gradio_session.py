"""Gradio UI session idle timeout (default 15 minutes)."""

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
    """Idle session lifetime in seconds (APP_SESSION_TTL_S, default 900 = 15 min)."""
    try:
        return max(60, int(os.getenv("APP_SESSION_TTL_S", "900")))
    except ValueError:
        return 900


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


class SessionIdleMiddleware(BaseHTTPMiddleware):
    """Expire Gradio auth tokens after APP_SESSION_TTL_S of inactivity."""

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
                sessions[token] = {"last": now, "created": now}
            elif (now - float(meta.get("last") or 0)) > ttl:
                tokens.pop(token, None)
                sessions.pop(token, None)
                expired = True
                logger.info("Gradio session expired after %ss idle — login required.", ttl)
            else:
                meta["last"] = now
        elif token and token not in tokens:
            # Stale cookie after logout/expiry — force clean login screen.
            expired = True

        # Track tokens created by Gradio /login during this request.
        before = set(tokens.keys())
        response = await call_next(request)
        for new_token in set(tokens.keys()) - before:
            sessions[new_token] = {"last": time.time(), "created": time.time()}

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
