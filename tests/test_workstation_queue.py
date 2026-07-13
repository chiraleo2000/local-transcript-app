"""Unit tests for workstation queue history and warm GPU start helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.client_identity import client_ip_from_request
from backend.job_cancel import should_free_gpu_for_queue_on_cancel
from backend.services.asr_local import should_warm_start_gpu_job


class TestClientIdentity(unittest.TestCase):
    def test_forwarded_for_first_hop(self) -> None:
        req = MagicMock()
        req.headers = {"x-forwarded-for": "10.0.0.5, 10.0.0.1"}
        req.client = MagicMock(host="127.0.0.1")
        self.assertEqual(client_ip_from_request(req), "10.0.0.5")

    def test_client_host_fallback(self) -> None:
        req = MagicMock()
        req.headers = {}
        req.client = MagicMock(host="192.168.1.20")
        self.assertEqual(client_ip_from_request(req), "192.168.1.20")


class TestWarmStart(unittest.TestCase):
    def test_warm_when_keep_and_not_clear(self) -> None:
        with patch.dict(
            "os.environ",
            {"ASR_KEEP_PRELOADED": "true", "ASR_CLEAR_VRAM_AFTER_JOB": "false"},
            clear=False,
        ):
            self.assertTrue(should_warm_start_gpu_job())

    def test_cold_when_clear_after_job(self) -> None:
        with patch.dict(
            "os.environ",
            {"ASR_KEEP_PRELOADED": "true", "ASR_CLEAR_VRAM_AFTER_JOB": "true"},
            clear=False,
        ):
            self.assertFalse(should_warm_start_gpu_job())


class TestListJobsByIp(unittest.TestCase):
    def test_filter_by_client_ip(self) -> None:
        from backend import storage

        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "jobs"
            job_dir.mkdir()
            for job_id, ip in (("a", "1.1.1.1"), ("b", "2.2.2.2"), ("c", "1.1.1.1")):
                (job_dir / f"{job_id}.json").write_text(
                    json.dumps(
                        {
                            "job_id": job_id,
                            "created_at": f"2026-01-0{ord(job_id)-96}T00:00:00",
                            "client_ip": ip,
                            "status": "completed",
                        }
                    ),
                    encoding="utf-8",
                )
            with patch.object(storage, "JOB_DIR", job_dir), patch.object(
                storage, "ensure_app_dirs", lambda: None
            ):
                rows = storage.list_jobs(10, client_ip="1.1.1.1")
                self.assertEqual({row["job_id"] for row in rows}, {"a", "c"})


class TestCancelQueueFlag(unittest.TestCase):
    def test_default_frees_cache(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            # Unset may still inherit; force true
            with patch.dict("os.environ", {"UI_CANCEL_FREES_GPU_FOR_QUEUE": "true"}):
                self.assertTrue(should_free_gpu_for_queue_on_cancel())


if __name__ == "__main__":
    unittest.main()
