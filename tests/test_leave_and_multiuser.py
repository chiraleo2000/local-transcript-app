"""Jobs keep running after the browser leaves, and users do not share results."""

from __future__ import annotations

import json

import backend.job_queue as job_queue
from backend.job_enqueue import EnqueueOptions, enqueue_media_job


def _reset_queue() -> None:
    with job_queue._LOCK:  # noqa: SLF001
        job_queue._QUEUED_COUNT = 0  # noqa: SLF001
        job_queue._ACTIVE_API_JOBS = 0  # noqa: SLF001


def _isolate_storage(monkeypatch, tmp_path):
    monkeypatch.delenv("APP_STORAGE_URL", raising=False)
    job_dir = tmp_path / "jobs"
    transcript_dir = tmp_path / "transcripts"
    job_dir.mkdir()
    transcript_dir.mkdir()
    monkeypatch.setattr("backend.storage.JOB_DIR", job_dir)
    monkeypatch.setattr("backend.storage.TRANSCRIPT_DIR", transcript_dir)
    monkeypatch.setattr("backend.storage.ensure_app_dirs", lambda: None)
    monkeypatch.setattr("backend.jobs_db.jobs_db_path", lambda: tmp_path / "missing.db")
    monkeypatch.setenv("API_MAX_QUEUED_JOBS", "4")
    monkeypatch.setenv("UI_MAX_CONCURRENT_JOBS", "1")
    return job_dir, transcript_dir


def _wait(job_id: str) -> None:
    thread = job_queue._JOB_THREADS.get(job_id)  # noqa: SLF001
    assert thread is not None
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_transcript_is_stored_when_the_user_leaves(monkeypatch, tmp_path):
    job_dir, _transcripts = _isolate_storage(monkeypatch, tmp_path)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"not-real-audio")
    monkeypatch.setattr(
        "backend.job_enqueue.copy_input_file",
        lambda *_args, **_kwargs: str(audio),
    )

    def finish_without_browser(**kwargs):
        from backend.storage import save_transcript, write_job_record

        assert not kwargs["cancel_event"].is_set()
        job_id = kwargs["job_id"]
        path = save_transcript(job_id, "Typhoon Whisper", "still here")
        write_job_record(
            job_id,
            {
                "status": "completed",
                "username": kwargs["meta"].username,
                "user_id": kwargs["meta"].user_id,
                "results": {
                    "Typhoon Whisper": {"text": "still here", "download_path": path},
                },
            },
        )
        return {"results": {"Typhoon Whisper": {"text": "still here"}}}

    monkeypatch.setattr("backend.job_enqueue.run_transcription_job", finish_without_browser)
    _reset_queue()
    queued = enqueue_media_job(
        str(audio),
        EnqueueOptions(username="alice", user_id=1, tab_id="tab-alice"),
    )
    assert queued.status == "queued"
    # Leaving the page does not set the worker cancel event.
    _wait(queued.job_id)
    saved = json.loads((job_dir / f"{queued.job_id}.json").read_text(encoding="utf-8"))
    assert saved["status"] == "completed"
    assert saved["results"]["Typhoon Whisper"]["text"] == "still here"
    assert list((tmp_path / "transcripts").glob("*.txt"))


def test_two_users_keep_separate_stored_jobs(monkeypatch, tmp_path):
    job_dir, _transcripts = _isolate_storage(monkeypatch, tmp_path)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"not-real-audio")
    monkeypatch.setattr(
        "backend.job_enqueue.copy_input_file",
        lambda *_args, **_kwargs: str(audio),
    )

    def finish_for_owner(**kwargs):
        from backend.storage import write_job_record

        meta = kwargs["meta"]
        write_job_record(
            kwargs["job_id"],
            {
                "status": "completed",
                "username": meta.username,
                "user_id": meta.user_id,
                "results": {"Typhoon Whisper": {"text": meta.username}},
            },
        )
        return {"results": {}}

    monkeypatch.setattr("backend.job_enqueue.run_transcription_job", finish_for_owner)
    _reset_queue()
    alice = enqueue_media_job(
        str(audio),
        EnqueueOptions(username="alice", user_id=1, tab_id="tab-a"),
    )
    bob = enqueue_media_job(
        str(audio),
        EnqueueOptions(username="bob", user_id=2, tab_id="tab-b"),
    )
    assert alice.status == "queued"
    assert bob.status == "queued"
    assert alice.job_id != bob.job_id
    _wait(alice.job_id)
    _wait(bob.job_id)

    from backend.storage import list_jobs

    alice_rows = list_jobs(10, username="alice", user_id=1)
    bob_rows = list_jobs(10, username="bob", user_id=2)
    assert {row["job_id"] for row in alice_rows} == {alice.job_id}
    assert {row["job_id"] for row in bob_rows} == {bob.job_id}
