"""Unit tests for Gradio idle session timeout helpers."""

from __future__ import annotations

import os
import time
import unittest
from unittest.mock import MagicMock, patch

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from backend.gradio_session import SessionIdleMiddleware, session_ttl_s
from backend.ui_session import (
    clear_active_job,
    resolve_runtime,
    set_active_job,
    set_last_completed_job,
)


class TestSessionTtl(unittest.TestCase):
    def test_default_60_minutes(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APP_SESSION_TTL_S", None)
            self.assertEqual(session_ttl_s(), 3600)

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
            with patch(
                "backend.gradio_session._should_extend_idle_session",
                return_value=False,
            ):
                client = TestClient(app)
                resp = client.get(
                    "/",
                    cookies={
                        "access-token-unsecure-abc": "tok1",
                    },
                )
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("tok1", gradio_app.tokens)

    def test_keeps_session_while_job_inflight(self) -> None:
        gradio_app = MagicMock()
        gradio_app.cookie_id = "abc"
        gradio_app.tokens = {"tok1": "alice"}
        gradio_app._lta_sessions = {
            "tok1": {
                "last": time.time() - 1000,
                "created": time.time() - 2000,
                "username": "alice",
            }
        }

        async def homepage(_request):
            return PlainTextResponse("ok")

        app = Starlette(routes=[Route("/", homepage)])
        app.add_middleware(SessionIdleMiddleware, gradio_app=gradio_app)

        with patch.dict(os.environ, {"APP_SESSION_TTL_S": "60"}, clear=False):
            with patch(
                "backend.gradio_session._should_extend_idle_session",
                return_value=True,
            ):
                client = TestClient(app)
                resp = client.get(
                    "/",
                    cookies={
                        "access-token-unsecure-abc": "tok1",
                    },
                )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("tok1", gradio_app.tokens)
        self.assertGreater(
            float(gradio_app._lta_sessions["tok1"]["last"]),
            time.time() - 5,
        )

    def test_keeps_session_for_recent_completed_job(self) -> None:
        gradio_app = MagicMock()
        gradio_app.cookie_id = "abc"
        gradio_app.tokens = {"tok1": "alice"}
        gradio_app._lta_sessions = {
            "tok1": {
                "last": time.time() - 1000,
                "created": time.time() - 2000,
                "username": "alice",
            }
        }

        async def homepage(_request):
            return PlainTextResponse("ok")

        app = Starlette(routes=[Route("/", homepage)])
        app.add_middleware(SessionIdleMiddleware, gradio_app=gradio_app)

        with patch.dict(os.environ, {"APP_SESSION_TTL_S": "60"}, clear=False):
            with patch(
                "backend.gradio_session._user_has_inflight_job",
                return_value=False,
            ), patch(
                "backend.gradio_session._user_has_recent_job_activity",
                return_value=True,
            ):
                client = TestClient(app)
                resp = client.get(
                    "/",
                    cookies={"access-token-unsecure-abc": "tok1"},
                )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("tok1", gradio_app.tokens)


class TestUiSessionCompletedJob(unittest.TestCase):
    def test_clear_active_remembers_completed(self) -> None:
        runtime, _ = resolve_runtime("tab-test-completed")
        set_active_job(runtime, "job-123", None)
        clear_active_job(runtime, completed_job_id="job-123")
        self.assertIsNone(runtime["active_job_id"])
        self.assertEqual(runtime["last_completed_job_id"], "job-123")

    def test_set_last_completed_job(self) -> None:
        runtime, _ = resolve_runtime("tab-test-set-completed")
        set_last_completed_job(runtime, " job-abc ")
        self.assertEqual(runtime["last_completed_job_id"], "job-abc")
        set_last_completed_job(runtime, "")
        self.assertIsNone(runtime["last_completed_job_id"])


if __name__ == "__main__":
    unittest.main()
