"""Unit tests for Gradio idle session timeout helpers."""

from __future__ import annotations

import os
import time
import unittest
from unittest.mock import MagicMock, patch

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from backend.gradio_session import SessionIdleMiddleware, session_ttl_s


class TestSessionTtl(unittest.TestCase):
    def test_default_15_minutes(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APP_SESSION_TTL_S", None)
            self.assertEqual(session_ttl_s(), 900)

    def test_env_override(self) -> None:
        with patch.dict(os.environ, {"APP_SESSION_TTL_S": "120"}, clear=False):
            self.assertEqual(session_ttl_s(), 120)


class TestSessionIdleMiddleware(unittest.TestCase):
    def test_expires_idle_token(self) -> None:
        gradio_app = MagicMock()
        gradio_app.cookie_id = "abc"
        gradio_app.tokens = {"tok1": "alice"}
        gradio_app._lta_sessions = {
            "tok1": {"last": time.time() - 1000, "created": time.time() - 2000}
        }

        async def homepage(_request):
            return PlainTextResponse("ok")

        app = Starlette(routes=[Route("/", homepage)])
        app.add_middleware(SessionIdleMiddleware, gradio_app=gradio_app)

        with patch.dict(os.environ, {"APP_SESSION_TTL_S": "60"}, clear=False):
            client = TestClient(app)
            resp = client.get(
                "/",
                cookies={
                    "access-token-unsecure-abc": "tok1",
                },
            )
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("tok1", gradio_app.tokens)


if __name__ == "__main__":
    unittest.main()
