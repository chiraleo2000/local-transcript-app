"""Unit tests for headless job queue, history isolation, and API auth helpers."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


class TestJobQueueCap(unittest.TestCase):
    def test_max_four_slots_then_reject(self) -> None:
        import backend.job_queue as jq

        with patch.dict(os.environ, {"API_MAX_QUEUED_JOBS": "4"}, clear=False):
            with patch.object(jq, "_QUEUED_COUNT", 0), patch.object(
                jq, "_ACTIVE_API_JOBS", 0
            ), patch.object(jq, "_JOB_EVENTS", {}), patch.object(
                jq, "_JOB_PROGRESS", {}
            ), patch.object(jq, "_JOB_THREADS", {}):
                reserved = [jq.try_reserve_queue_slot() for _ in range(4)]
                self.assertTrue(all(reserved))
                self.assertFalse(jq.try_reserve_queue_slot())
                snap = jq.snapshot_queue()
                self.assertEqual(snap["queued"], 4)
                self.assertEqual(snap["max"], 4)
                for _ in range(4):
                    jq.release_queue_slot(started=False)


class TestHistoryIsolation(unittest.TestCase):
    def test_list_jobs_filters_by_username(self) -> None:
        from backend import storage

        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "jobs"
            job_dir.mkdir()
            for job_id, user in (("a", "alice"), ("b", "bob"), ("c", "alice")):
                (job_dir / f"{job_id}.json").write_text(
                    json.dumps(
                        {
                            "job_id": job_id,
                            "created_at": f"2026-01-0{ord(job_id) - 96}T00:00:00",
                            "username": user,
                            "user_id": 1 if user == "alice" else 2,
                            "status": "completed",
                        }
                    ),
                    encoding="utf-8",
                )
            with patch.object(storage, "JOB_DIR", job_dir), patch.object(
                storage, "ensure_app_dirs", lambda: None
            ):
                rows = storage.list_jobs(10, username="alice")
                self.assertEqual({row["job_id"] for row in rows}, {"a", "c"})
                rows_b = storage.list_jobs(10, user_id=2)
                self.assertEqual({row["job_id"] for row in rows_b}, {"b"})


class TestGpuSemaphorePolicy(unittest.TestCase):
    def test_ui_max_concurrent_defaults_to_one_in_enterprise(self) -> None:
        from backend.asr_quality import ENTERPRISE_DOCKER_ENV

        self.assertEqual(ENTERPRISE_DOCKER_ENV.get("UI_MAX_CONCURRENT_JOBS"), "1")
        self.assertEqual(ENTERPRISE_DOCKER_ENV.get("API_MAX_QUEUED_JOBS"), "4")
        self.assertEqual(ENTERPRISE_DOCKER_ENV.get("UI_GRADIO_TRANSCRIBE_CONCURRENCY"), "4")


class TestRegisterPage(unittest.TestCase):
    def test_register_html_is_public(self) -> None:
        from starlette.applications import Starlette
        from starlette.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "users.db"
            with patch.dict(
                os.environ,
                {
                    "APP_USERS_DB": str(db_path),
                    "APP_SEED_PASSWORD": "seed-pass",
                    "APP_AUTH_SECRET": "s",
                },
                clear=False,
            ):
                from backend.job_api import build_api_routes

                client = TestClient(Starlette(routes=build_api_routes()))
                page = client.get("/register")
                self.assertEqual(page.status_code, 200)
                self.assertIn("Create account", page.text)
                created = client.post(
                    "/api/auth/register",
                    json={"username": "newbie", "password": "pass1234"},
                )
                self.assertEqual(created.status_code, 200)
                self.assertEqual(created.json()["user"]["username"], "newbie")


class TestApiAuthRoutes(unittest.TestCase):
    def test_login_and_me_roundtrip(self) -> None:
        from starlette.testclient import TestClient
        from starlette.applications import Starlette

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "users.db"
            env = {
                "APP_USERS_DB": str(db_path),
                "APP_SEED_USER": "chira",
                "APP_SEED_PASSWORD": "api-test-pass",
                "APP_AUTH_SECRET": "api-test-secret",
            }
            with patch.dict(os.environ, env, clear=False):
                from backend.job_api import build_api_routes

                app = Starlette(routes=build_api_routes())
                client = TestClient(app)
                bad = client.post(
                    "/api/auth/login",
                    json={"username": "chira", "password": "wrong"},
                )
                self.assertEqual(bad.status_code, 401)
                ok = client.post(
                    "/api/auth/login",
                    json={"username": "chira", "password": "api-test-pass"},
                )
                self.assertEqual(ok.status_code, 200)
                token = ok.json()["token"]
                me = client.get(
                    "/api/auth/me",
                    headers={"Authorization": f"Bearer {token}"},
                )
                self.assertEqual(me.status_code, 200)
                self.assertEqual(me.json()["user"]["username"], "chira")

    def test_jobs_list_requires_auth(self) -> None:
        from starlette.testclient import TestClient
        from starlette.applications import Starlette

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "users.db"
            with patch.dict(
                os.environ,
                {
                    "APP_USERS_DB": str(db_path),
                    "APP_SEED_PASSWORD": "x",
                    "APP_AUTH_SECRET": "s",
                },
                clear=False,
            ):
                from backend.job_api import build_api_routes

                client = TestClient(Starlette(routes=build_api_routes()))
                resp = client.get("/api/jobs")
                self.assertEqual(resp.status_code, 401)


class TestCancelByJobId(unittest.TestCase):
    def test_cancel_signals_event(self) -> None:
        import backend.job_queue as jq
        from backend.job_cancel import cancel_job_by_id
        from backend.progress import JobProgress

        event = threading.Event()
        progress = JobProgress()
        with patch.object(jq, "_JOB_EVENTS", {"job1": event}), patch.object(
            jq, "_JOB_PROGRESS", {"job1": progress}
        ), patch.object(jq, "_JOB_THREADS", {}):
            self.assertTrue(cancel_job_by_id("job1"))
            self.assertTrue(event.is_set())
            self.assertFalse(cancel_job_by_id("missing"))


if __name__ == "__main__":
    unittest.main()
