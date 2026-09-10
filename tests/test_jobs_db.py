"""Tests for SQLite job history, queue policy, and shared enqueue."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestJobsDb(unittest.TestCase):
    def test_upsert_list_and_schema(self) -> None:
        from backend import jobs_db

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "jobs.db"
            with patch.dict(os.environ, {"APP_JOBS_DB": str(db_path)}, clear=False):
                jobs_db.init_jobs_db()
                self.assertEqual(jobs_db.schema_version(), 1)
                jobs_db.upsert_job(
                    "job_a",
                    {
                        "job_id": "job_a",
                        "user_id": 1,
                        "username": "alice",
                        "status": "completed",
                        "display_name": "meeting",
                        "source_filename": "meeting.wav",
                        "created_at": "2026-01-02T00:00:00",
                        "updated_at": "2026-01-02T00:01:00",
                        "selected_engines": ["Typhoon Whisper"],
                        "results": {
                            "Typhoon Whisper": {
                                "text": "hello",
                                "download_path": str(Path(tmp) / "out.txt"),
                            }
                        },
                        "source_path": str(Path(tmp) / "in.wav"),
                    },
                )
                jobs_db.upsert_job(
                    "job_b",
                    {
                        "job_id": "job_b",
                        "user_id": 2,
                        "username": "bob",
                        "status": "queued",
                        "created_at": "2026-01-01T00:00:00",
                    },
                )
                alice = jobs_db.list_job_rows(10, user_id=1)
                self.assertEqual([r["job_id"] for r in alice], ["job_a"])
                self.assertTrue(alice[0]["transcript_path"].endswith("out.txt"))
                self.assertEqual(alice[0]["input_path"], str(Path(tmp) / "in.wav"))
                bob = jobs_db.list_job_rows(10, username="bob")
                self.assertEqual([r["job_id"] for r in bob], ["job_b"])

    def test_migrate_json_jobs(self) -> None:
        from backend import jobs_db

        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "jobs"
            job_dir.mkdir()
            db_path = Path(tmp) / "jobs.db"
            (job_dir / "legacy.json").write_text(
                json.dumps(
                    {
                        "job_id": "legacy",
                        "username": "carol",
                        "user_id": 3,
                        "status": "completed",
                        "created_at": "2026-02-01T00:00:00",
                        "display_name": "legacy",
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"APP_JOBS_DB": str(db_path)}, clear=False):
                count = jobs_db.migrate_json_jobs(job_dir)
                self.assertEqual(count, 1)
                count2 = jobs_db.migrate_json_jobs(job_dir)
                self.assertEqual(count2, 1)
                rows = jobs_db.list_job_rows(10, username="carol")
                self.assertEqual(rows[0]["job_id"], "legacy")


class TestQueuePolicy(unittest.TestCase):
    def test_tier_a_pascal(self) -> None:
        from backend.queue_policy import detect_queue_tier, recommended_caps

        gpu = {
            "cuda": True,
            "cuda_device_name": "Tesla P4",
            "cuda_vram_mb": 7680,
            "cuda_capability_major": 6,
            "cuda_capability_minor": 1,
        }
        with patch("backend.gpu_arch.pascal_speed_gpu_active", return_value=True):
            self.assertEqual(detect_queue_tier(gpu), "A")
        caps = recommended_caps("A")
        self.assertEqual(caps["UI_MAX_CONCURRENT_JOBS"], 1)
        self.assertEqual(caps["API_MAX_QUEUED_JOBS"], 4)
        self.assertEqual(caps["UI_MAX_BATCH_FILES"], 3)

    def test_tier_c_high_vram(self) -> None:
        from backend.queue_policy import detect_queue_tier, recommended_caps

        gpu = {
            "cuda": True,
            "cuda_device_name": "NVIDIA RTX 4090",
            "cuda_vram_mb": 24576,
            "cuda_capability_major": 8,
            "cuda_capability_minor": 9,
        }
        with patch("backend.gpu_arch.pascal_speed_gpu_active", return_value=False), patch(
            "backend.gpu_arch.is_pascal_speed_gpu", return_value=False
        ), patch.dict(os.environ, {"ASR_PARALLEL_MIN_VRAM_MB": "12288"}, clear=False):
            self.assertEqual(detect_queue_tier(gpu), "C")
        caps = recommended_caps("C")
        self.assertEqual(caps["UI_MAX_CONCURRENT_JOBS"], 2)
        self.assertEqual(caps["API_MAX_QUEUED_JOBS"], 8)

    def test_apply_respects_explicit_env(self) -> None:
        from backend import queue_policy

        with patch.dict(
            os.environ,
            {
                "UI_MAX_CONCURRENT_JOBS": "1",
                "API_MAX_QUEUED_JOBS": "4",
                "UI_MAX_BATCH_FILES": "3",
            },
            clear=False,
        ), patch.object(queue_policy, "detect_queue_tier", return_value="C"), patch(
            "backend.pipeline.configure_job_semaphore", return_value=1
        ), patch("backend.gpu_arch.probe_cuda_device", return_value={}):
            applied = queue_policy.apply_queue_policy()
            self.assertEqual(applied, {})
            self.assertEqual(os.environ["UI_MAX_CONCURRENT_JOBS"], "1")

    def test_apply_fills_unset_for_tier_c(self) -> None:
        from backend import queue_policy

        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in {
                "UI_MAX_CONCURRENT_JOBS",
                "API_MAX_QUEUED_JOBS",
                "UI_MAX_BATCH_FILES",
            }
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
            queue_policy, "detect_queue_tier", return_value="C"
        ), patch("backend.pipeline.configure_job_semaphore", return_value=2), patch(
            "backend.gpu_arch.probe_cuda_device", return_value={}
        ):
            applied = queue_policy.apply_queue_policy()
            self.assertEqual(applied["UI_MAX_CONCURRENT_JOBS"], "2")
            self.assertEqual(applied["API_MAX_QUEUED_JOBS"], "8")
            self.assertEqual(applied["UI_MAX_BATCH_FILES"], "5")


class TestEnqueueBatch(unittest.TestCase):
    def test_batch_respects_cap_and_queue(self) -> None:
        from backend.job_enqueue import EnqueueOptions, enqueue_media_batch

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(4):
                p = Path(tmp) / f"a{i}.wav"
                p.write_bytes(b"RIFF")
                paths.append(str(p))
            db_path = Path(tmp) / "jobs.db"
            job_dir = Path(tmp) / "jobs"
            job_dir.mkdir()
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()

            def fake_submit(job_id, worker_fn):
                handle = MagicMock()
                handle.cancel_event = MagicMock()
                handle.progress = MagicMock()
                return handle

            with patch.dict(
                os.environ,
                {
                    "APP_JOBS_DB": str(db_path),
                    "API_MAX_QUEUED_JOBS": "4",
                    "UI_MAX_BATCH_FILES": "3",
                },
                clear=False,
            ), patch("backend.job_enqueue.try_reserve_queue_slot", return_value=True), patch(
                "backend.job_enqueue.submit_background_job", side_effect=fake_submit
            ), patch(
                "backend.job_enqueue.snapshot_queue",
                return_value={"queued": 3, "active": 0, "max": 4},
            ), patch(
                "backend.storage.JOB_DIR", job_dir
            ), patch(
                "backend.storage.INPUT_DIR", input_dir
            ), patch(
                "backend.storage.ensure_app_dirs", return_value=None
            ):
                batch = enqueue_media_batch(
                    paths,
                    EnqueueOptions(username="alice", user_id=1),
                    max_files=3,
                )
                self.assertEqual(len(batch.accepted), 3)
                self.assertEqual(len(batch.rejected), 1)
                self.assertIn("Batch limit", batch.rejected[0].error)


if __name__ == "__main__":
    unittest.main()
