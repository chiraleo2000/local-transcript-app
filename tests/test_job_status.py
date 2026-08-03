"""Unit tests for durable job status helpers."""

from __future__ import annotations

import unittest

from backend.job_status import job_is_in_flight, job_status_norm


class TestJobStatus(unittest.TestCase):
    def test_running_and_queued(self) -> None:
        self.assertTrue(job_is_in_flight({"status": "running"}))
        self.assertTrue(job_is_in_flight({"status": "queued"}))
        self.assertEqual(job_status_norm({"status": "Running"}), "running")

    def test_completed_not_in_flight(self) -> None:
        self.assertFalse(
            job_is_in_flight({"status": "completed", "results": {"Typhoon": {"text": "x"}}})
        )
        self.assertFalse(job_is_in_flight({"status": "failed", "error": "x"}))
        self.assertFalse(job_is_in_flight({"status": "cancelled"}))

    def test_legacy_progress_without_status(self) -> None:
        self.assertTrue(
            job_is_in_flight({"progress": {"phase": "asr", "message": "…"}})
        )
        self.assertFalse(
            job_is_in_flight(
                {
                    "progress": {"phase": "finalize"},
                    "results": {"Typhoon": {"text": "hi"}},
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
