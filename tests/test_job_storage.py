"""Job storage CRUD helpers."""

import json
from pathlib import Path

import pytest

from backend.storage import (
    JOB_DIR,
    TRANSCRIPT_DIR,
    list_jobs,
    load_job,
    new_job_id,
    save_transcript,
    write_job_record,
)


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.storage.JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr("backend.storage.TRANSCRIPT_DIR", tmp_path / "transcripts")
    monkeypatch.setattr("backend.storage.INPUT_DIR", tmp_path / "input")
    for path in (tmp_path / "jobs", tmp_path / "transcripts", tmp_path / "input"):
        path.mkdir(parents=True)


def test_write_job_record_merges_patches():
    job_id = new_job_id()
    write_job_record(job_id, {"status": "running", "display_name": "meeting"})
    write_job_record(job_id, {"progress": {"phase": "asr", "percent": 50}})
    job = load_job(job_id)
    assert job is not None
    assert job["status"] == "running"
    assert job["display_name"] == "meeting"
    assert job["progress"]["phase"] == "asr"
    assert job.get("created_at")
    assert job.get("updated_at")


def test_list_jobs_orders_newest_first():
    first = new_job_id()
    second = new_job_id()
    write_job_record(first, {"status": "completed", "created_at": "2026-01-01T00:00:00+00:00"})
    write_job_record(second, {"status": "completed", "created_at": "2026-01-02T00:00:00+00:00"})
    rows = list_jobs(limit=10)
    assert rows[0]["job_id"] == second
    assert rows[1]["job_id"] == first


def test_list_jobs_handles_null_created_at(tmp_path):
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "job_a.json").write_text(
        json.dumps({"job_id": "job_a", "created_at": None, "status": "running"}),
        encoding="utf-8",
    )
    (jobs_dir / "job_b.json").write_text(
        json.dumps({"job_id": "job_b", "created_at": None, "status": "completed"}),
        encoding="utf-8",
    )
    rows = list_jobs(10)
    assert len(rows) == 2
    assert all(isinstance(row["created_at"], str) for row in rows)


def test_save_transcript_custom_output_name():
    job_id = new_job_id()
    path = save_transcript(job_id, "Typhoon Whisper", "hello world", output_name="my_meeting")
    assert path is not None
    assert Path(path).name == "my_meeting.txt"
    assert Path(path).read_text(encoding="utf-8") == "hello world"
