"""Cancel tab jobs and VRAM unload on restart."""

from __future__ import annotations

import threading
import time

from backend.job_cancel import cancel_tab_job
from backend.progress import JobProgress
from backend.services.asr_local import should_unload_on_cancel


def test_should_unload_on_cancel_off_by_default(monkeypatch):
    monkeypatch.delenv("ASR_UNLOAD_ON_CANCEL", raising=False)
    monkeypatch.setenv("ASR_HARD_MEMORY_SAFE", "true")
    monkeypatch.setattr("backend.services.asr_local._cuda_vram_mb", lambda: 8192)
    assert not should_unload_on_cancel()


def test_should_unload_on_cancel_explicit_false(monkeypatch):
    monkeypatch.setenv("ASR_UNLOAD_ON_CANCEL", "false")
    monkeypatch.setattr("backend.services.asr_local._cuda_vram_mb", lambda: 8192)
    assert not should_unload_on_cancel()


def test_cancel_tab_job_keeps_models_by_default(monkeypatch):
    monkeypatch.delenv("ASR_UNLOAD_ON_CANCEL", raising=False)
    unloaded: list[str] = []

    def _unload():
        unloaded.append("ok")

    monkeypatch.setattr("backend.pipeline.unload_all_pipeline_models", _unload)
    runtime = {
        "cancel_event": threading.Event(),
        "worker": None,
        "progress": JobProgress(),
        "active_job_id": None,
    }
    cancel_tab_job(runtime)
    assert unloaded == []


def test_cancel_tab_job_unloads_models(monkeypatch):
    monkeypatch.setenv("ASR_UNLOAD_ON_CANCEL", "true")
    unloaded: list[str] = []

    def _unload():
        unloaded.append("ok")

    monkeypatch.setattr("backend.pipeline.unload_all_pipeline_models", _unload)
    runtime = {
        "cancel_event": threading.Event(),
        "worker": None,
        "progress": JobProgress(),
        "active_job_id": None,
    }
    cancel_tab_job(runtime)
    assert unloaded == ["ok"]


def test_cancel_tab_job_joins_worker_before_unload(monkeypatch):
    monkeypatch.setenv("ASR_UNLOAD_ON_CANCEL", "true")
    monkeypatch.setenv("UI_CANCEL_JOIN_TIMEOUT_S", "5")
    order: list[str] = []
    finished = threading.Event()

    def _worker():
        time.sleep(0.05)
        order.append("worker_done")
        finished.set()

    worker = threading.Thread(target=_worker)
    worker.start()
    runtime = {
        "cancel_event": threading.Event(),
        "worker": worker,
        "progress": JobProgress(),
        "active_job_id": "job-1",
    }

    def _unload():
        order.append("unload")
        assert finished.is_set()

    monkeypatch.setattr("backend.pipeline.unload_all_pipeline_models", _unload)
    cancel_tab_job(runtime)
    worker.join()
    assert order == ["worker_done", "unload"]
