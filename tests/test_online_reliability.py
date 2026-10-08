"""Storage retries, local result outbox, and Turbo engine selection."""

from __future__ import annotations

import io
import urllib.error

from backend.services.asr_local import (
    ENGINE_PATHUMMA,
    ENGINE_TURBO,
    ENGINE_TYPHOON,
    best_asr_engine_for_language,
)


def test_storage_retries_transient_failure(monkeypatch):
    from backend.storage_client import upsert_job

    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    def urlopen(req, timeout=30):
        del req, timeout
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("blip")
        return _Resp()

    monkeypatch.setenv("APP_STORAGE_URL", "http://storage:8081")
    monkeypatch.setattr("backend.storage_client.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("backend.storage_client.time.sleep", lambda _seconds: None)
    upsert_job("job", {"status": "queued"})
    assert calls["n"] == 3


def test_storage_does_not_retry_not_found(monkeypatch):
    from backend.storage_client import StorageUnavailable, upsert_job

    calls = {"n": 0}

    def urlopen(req, timeout=30):
        del timeout
        calls["n"] += 1
        raise urllib.error.HTTPError(
            req.full_url, 404, "missing", hdrs=None, fp=io.BytesIO(b'{"error":"missing"}')
        )

    monkeypatch.setenv("APP_STORAGE_URL", "http://storage:8081")
    monkeypatch.setattr("backend.storage_client.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("backend.storage_client.time.sleep", lambda _seconds: None)
    try:
        upsert_job("missing", {"status": "queued"})
    except StorageUnavailable:
        pass
    else:
        raise AssertionError("expected StorageUnavailable")
    assert calls["n"] == 1


def test_storage_retries_timeout(monkeypatch):
    from backend.storage_client import upsert_job

    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    def urlopen(req, timeout=30):
        del req, timeout
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("slow")
        return _Resp()

    monkeypatch.setenv("APP_STORAGE_URL", "http://storage:8081")
    monkeypatch.setattr("backend.storage_client.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("backend.storage_client.time.sleep", lambda _seconds: None)
    upsert_job("job", {"status": "queued"})
    assert calls["n"] == 2


def test_outbox_skips_bad_payload_and_stops_when_down(monkeypatch, tmp_path):
    from backend.result_outbox import flush_pending_results, remember_result
    from backend.storage_client import StorageUnavailable

    saved: list[str] = []

    def save_result(**payload):
        job_id = payload["job_id"]
        if job_id == "a-bad":
            raise RuntimeError("boom")
        if job_id == "c-down":
            raise StorageUnavailable("down")
        saved.append(job_id)

    monkeypatch.setenv("RESULT_OUTBOX_DIR", str(tmp_path))
    monkeypatch.setenv("APP_STORAGE_URL", "http://storage:8081")
    monkeypatch.setattr("backend.storage_client.save_result", save_result)
    monkeypatch.setattr("backend.storage_client.storage_configured", lambda: True)
    remember_result("a-bad", "Typhoon Whisper", "nope")
    remember_result("b-good", "Typhoon Whisper", "hello")
    remember_result("c-down", "Typhoon Whisper", "later")
    remember_result("d-later", "Typhoon Whisper", "wait")

    assert flush_pending_results() == 1
    assert saved == ["b-good"]


def test_load_job_skips_sidecar_while_running(monkeypatch, tmp_path):
    import json

    from backend import storage

    job_dir = tmp_path / "jobs"
    job_dir.mkdir()
    (job_dir / "running-job.json").write_text(
        json.dumps({"job_id": "running-job", "status": "running", "progress": {"percent": 10}}),
        encoding="utf-8",
    )
    calls = {"n": 0}

    def get_job_row(_job_id):
        calls["n"] += 1
        return {"results": {"engine": {"text": "late"}}, "transcript_path": "t.txt"}

    monkeypatch.setattr(storage, "JOB_DIR", job_dir)
    monkeypatch.setattr(storage, "ensure_app_dirs", lambda: None)
    monkeypatch.setattr("backend.storage_client.storage_configured", lambda: True)
    monkeypatch.setattr("backend.jobs_db.get_job_row", get_job_row)
    job = storage.load_job("running-job")
    assert job["status"] == "running"
    assert calls["n"] == 0


def test_load_job_fills_finished_results_from_sidecar(monkeypatch, tmp_path):
    import json

    from backend import storage

    job_dir = tmp_path / "jobs"
    job_dir.mkdir()
    (job_dir / "done-job.json").write_text(
        json.dumps({"job_id": "done-job", "status": "completed"}),
        encoding="utf-8",
    )

    def get_job_row(_job_id):
        return {
            "results": {"engine": {"text": "hello"}},
            "transcript_path": "storage/transcripts/done-job.txt",
        }

    monkeypatch.setattr(storage, "JOB_DIR", job_dir)
    monkeypatch.setattr(storage, "ensure_app_dirs", lambda: None)
    monkeypatch.setattr("backend.storage_client.storage_configured", lambda: True)
    monkeypatch.setattr("backend.jobs_db.get_job_row", get_job_row)
    job = storage.load_job("done-job")
    assert job["results"]["engine"]["text"] == "hello"
    assert job["transcript_path"].endswith("done-job.txt")


def test_outbox_flushes_once(monkeypatch, tmp_path):
    from backend.result_outbox import flush_pending_results, remember_result

    saved: dict = {}

    def save_result(**payload):
        saved.update(payload)

    monkeypatch.setenv("RESULT_OUTBOX_DIR", str(tmp_path))
    monkeypatch.setenv("APP_STORAGE_URL", "http://storage:8081")
    monkeypatch.setattr("backend.storage_client.save_result", save_result)
    monkeypatch.setattr("backend.storage_client.storage_configured", lambda: True)
    remember_result("job-1", "Typhoon Whisper", "hello")
    assert flush_pending_results() == 1
    assert saved["text"] == "hello"
    assert flush_pending_results() == 0


def test_quality_policy_stays_on_large_v3(monkeypatch):
    monkeypatch.setenv("ASR_AUTO_POLICY", "quality")
    assert best_asr_engine_for_language("Thai") == ENGINE_TYPHOON


def test_fast_policy_uses_turbo_when_cached(monkeypatch):
    monkeypatch.setenv("ASR_AUTO_POLICY", "fast")
    monkeypatch.setattr("backend.services.asr_local.turbo_snapshot_cached", lambda: True)
    assert best_asr_engine_for_language("Thai") == ENGINE_TURBO


def test_fast_policy_falls_back_to_pathumma(monkeypatch):
    monkeypatch.setenv("ASR_AUTO_POLICY", "fast")
    monkeypatch.setattr("backend.services.asr_local.turbo_snapshot_cached", lambda: False)
    assert best_asr_engine_for_language("Thai") == ENGINE_PATHUMMA


def test_unload_leaves_other_checkpoint_resident():
    import engines.typhoon_asr as typhoon

    typhoon._pipeline_cache.clear()
    typhoon._pipeline_cache.append("resident")
    typhoon._loaded_model_id = "typhoon-ai/typhoon-whisper-turbo"
    typhoon.unload_model("typhoon-ai/typhoon-whisper-large-v3")
    assert typhoon._pipeline_cache == ["resident"]
    assert typhoon.loaded_model_id() == "typhoon-ai/typhoon-whisper-turbo"
    typhoon._pipeline_cache.clear()
    typhoon._loaded_model_id = None
