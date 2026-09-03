"""Unit tests for UI session recovery after disconnect / re-login."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from backend.session_recovery import collect_recovery_job_candidates


def _recent_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TestSessionRecovery(unittest.TestCase):
    def test_recovers_inflight_job_for_user_when_tab_id_changes(self) -> None:
        from backend import storage

        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "jobs"
            job_dir.mkdir()
            (job_dir / "job_a.json").write_text(
                json.dumps(
                    {
                        "job_id": "job_a",
                        "tab_id": "old-tab",
                        "username": "alice",
                        "user_id": 1,
                        "status": "running",
                        "created_at": _recent_iso(),
                        "updated_at": _recent_iso(),
                        "progress": {"phase": "asr", "message": "Working…"},
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(storage, "JOB_DIR", job_dir), patch.object(
                storage, "ensure_app_dirs", return_value=None
            ):
                candidates, completed = collect_recovery_job_candidates(
                    "new-tab",
                    None,
                    username="alice",
                    user_id=1,
                )
                self.assertEqual(candidates, ["job_a"])
                self.assertEqual(completed, [])

    def test_recovers_recent_completed_job_for_user_on_new_tab(self) -> None:
        from backend import storage

        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "jobs"
            job_dir.mkdir()
            (job_dir / "job_done.json").write_text(
                json.dumps(
                    {
                        "job_id": "job_done",
                        "tab_id": "old-tab",
                        "username": "alice",
                        "user_id": 1,
                        "status": "completed",
                        "created_at": _recent_iso(),
                        "updated_at": _recent_iso(),
                        "results": {"Typhoon": {"text": "hello", "download_path": "/x.txt"}},
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(storage, "JOB_DIR", job_dir), patch.object(
                storage, "ensure_app_dirs", return_value=None
            ):
                candidates, completed = collect_recovery_job_candidates(
                    "new-tab",
                    None,
                    username="alice",
                    user_id=1,
                    recent_completed_within_s=86400,
                )
                self.assertEqual(candidates, [])
                self.assertEqual(completed, ["job_done"])

    def test_prefers_tab_scoped_jobs_first(self) -> None:
        from backend import storage

        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "jobs"
            job_dir.mkdir()
            for job_id, tab_id, status in (
                ("newer_other_tab", "other", "running"),
                ("tab_job", "my-tab", "running"),
            ):
                (job_dir / f"{job_id}.json").write_text(
                    json.dumps(
                        {
                            "job_id": job_id,
                            "tab_id": tab_id,
                            "username": "alice",
                            "status": status,
                            "created_at": _recent_iso(),
                            "updated_at": _recent_iso(),
                        }
                    ),
                    encoding="utf-8",
                )
            with patch.object(storage, "JOB_DIR", job_dir), patch.object(
                storage, "ensure_app_dirs", return_value=None
            ):
                candidates, _ = collect_recovery_job_candidates(
                    "my-tab",
                    None,
                    username="alice",
                )
                self.assertEqual(candidates[0], "tab_job")
                self.assertIn("newer_other_tab", candidates)


if __name__ == "__main__":
    unittest.main()
