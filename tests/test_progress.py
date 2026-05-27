"""Unit tests for job progress tracking."""

import time

from backend.progress import JobProgress


def test_progress_snapshot_idle_after_reset():
    tracker = JobProgress()
    tracker.reset()
    snap = tracker.snapshot()
    assert snap["active"] is False
    assert snap["phase"] == "idle"
    assert snap["percent"] == 0.0


def test_progress_advances_through_phases():
    tracker = JobProgress()
    tracker.start("job-1")
    tracker.set_phase("normalize", "Normalizing\u2026", 10)
    tracker.set_phase("asr", "Transcribing\u2026", 50)
    snap = tracker.snapshot()
    assert snap["active"] is True
    assert snap["percent"] >= 50
    assert snap["elapsed_s"] >= 0


def test_progress_window_updates_percent():
    tracker = JobProgress()
    tracker.start()
    tracker.set_asr_window(2, 4)
    snap = tracker.snapshot()
    assert 55 <= snap["percent"] <= 75


def test_progress_finish_reaches_one_hundred():
    tracker = JobProgress()
    tracker.start()
    tracker.finish()
    snap = tracker.snapshot()
    assert snap["percent"] == 100.0
    assert snap["phase"] == "done"
    assert snap["active"] is False
    assert snap["remaining_s"] == 0.0
