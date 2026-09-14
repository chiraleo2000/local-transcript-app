"""Storage sidecar schema, migration, and HTTP client."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

import pytest


@pytest.fixture()
def storage_server(tmp_path, monkeypatch):
    from backend.storage_api import serve

    legacy = tmp_path / "import"
    (legacy / "transcripts").mkdir(parents=True)
    (legacy / "jobs").mkdir()
    users = legacy / "users.db"
    conn = sqlite3.connect(users)
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT, "
        "created_at REAL, is_active INTEGER)"
    )
    conn.execute(
        "INSERT INTO users VALUES (7, 'alice', 'hash', 1.0, 1)"
    )
    conn.commit()
    conn.close()
    jobs = legacy / "jobs.db"
    jconn = sqlite3.connect(jobs)
    jconn.execute(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            user_id INTEGER,
            username TEXT,
            status TEXT,
            display_name TEXT,
            source_filename TEXT,
            created_at TEXT,
            updated_at TEXT,
            transcript_path TEXT,
            input_path TEXT,
            error TEXT,
            progress_json TEXT,
            options_json TEXT,
            client_ip TEXT,
            audio_duration_s REAL,
            total_elapsed_s REAL,
            tab_id TEXT,
            selected_engines_json TEXT
        )
        """
    )
    jconn.execute(
        "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "job1", 7, "alice", "completed", "meet", "meet.wav",
            "2026-01-01T00:00:00", "2026-01-01T00:01:00", "", "", "",
            "{}", "{}", "", 12.0, 3.0, "", "[]",
        ),
    )
    jconn.commit()
    jconn.close()
    (legacy / "transcripts" / "old.txt").write_text(
        "[00:00:01 → 00:00:02] [SPEAKER_00]: สวัสดีครับ",
        encoding="utf-8",
    )
    (legacy / "jobs" / "extra.json").write_text(
        json.dumps({"job_id": "extra", "status": "completed", "username": "alice", "user_id": 7}),
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_STORAGE_DB", str(tmp_path / "app.db"))
    monkeypatch.setenv("APP_IMPORT_DIR", str(legacy))
    monkeypatch.delenv("APP_STORAGE_URL", raising=False)
    monkeypatch.delenv("APP_SEED_USER", raising=False)
    monkeypatch.delenv("APP_SEED_PASSWORD", raising=False)
    server = serve("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    monkeypatch.setenv("APP_STORAGE_URL", f"http://127.0.0.1:{port}")
    yield server
    server.shutdown()


def test_import_and_roundtrip(storage_server):
    del storage_server
    from backend.storage_client import get_job, list_jobs, save_result

    rows = list_jobs(20, username="alice")
    ids = {row["job_id"] for row in rows}
    assert "job1" in ids
    assert "extra" in ids
    old = get_job("old")
    assert old is not None
    assert "สวัสดี" in old["results"]["imported"]["text"]
    assert old["results"]["imported"]["segments"][0]["speaker"] == "SPEAKER_00"

    save_result(
        job_id="job1",
        engine="Typhoon Whisper",
        text="[00:00:00 → 00:00:01] [SPEAKER_01]: ทดสอบ",
        language="th",
        duration_s=1.2,
        transcript_path="/tmp/job1.txt",
    )
    job = get_job("job1")
    assert job is not None
    assert job["results"]["Typhoon Whisper"]["text"].startswith("[00:00:00")
    assert job["transcript_path"] == "/tmp/job1.txt"
    assert job["results"]["Typhoon Whisper"]["segments"][0]["speaker"] == "SPEAKER_01"


def test_sidecar_down_raises(monkeypatch):
    from backend.storage_client import StorageUnavailable, upsert_job

    monkeypatch.setenv("APP_STORAGE_URL", "http://127.0.0.1:9")
    with pytest.raises(StorageUnavailable):
        upsert_job("missing", {"status": "queued"})
