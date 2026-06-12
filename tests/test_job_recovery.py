"""Session recovery from running job manifests."""

from backend.storage import list_jobs, write_job_record


def test_running_manifest_visible_in_list_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.storage.JOB_DIR", tmp_path / "jobs")
    (tmp_path / "jobs").mkdir(parents=True)
    job_id = "20260529_120000_abcd1234"
    write_job_record(job_id, {
        "status": "running",
        "tab_id": "tab-recover",
        "display_name": "demo.wav",
        "progress": {"phase": "asr", "message": "Transcribing", "percent": 45, "elapsed_s": 12},
    })
    rows = [row for row in list_jobs(10) if row["job_id"] == job_id]
    assert len(rows) == 1
    assert rows[0]["status"] == "running"
    assert rows[0]["tab_id"] == "tab-recover"
    assert rows[0]["progress"]["phase"] == "asr"
