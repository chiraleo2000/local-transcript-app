"""App-local storage and config helpers."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.paths import app_root, resolve_path


APP_ROOT = app_root()
CONFIG_DIR = APP_ROOT / "config"
STORAGE_DIR = APP_ROOT / "storage"
INPUT_DIR = STORAGE_DIR / "input"
AUDIO_DIR = STORAGE_DIR / "audio"
TRANSCRIPT_DIR = STORAGE_DIR / "transcripts"
JOB_DIR = STORAGE_DIR / "jobs"
LOG_DIR = STORAGE_DIR / "logs"
MODEL_DIR = resolve_path(os.getenv("APP_MODEL_ROOT", "models"))
HF_CACHE_DIR = MODEL_DIR / "hf_cache"
OV_CACHE_DIR = MODEL_DIR / "ov_cache"
APP_CONFIG_PATH = CONFIG_DIR / "app_config.json"


def ensure_app_dirs() -> None:
    """Create all app-owned folders and set cache env vars."""
    for path in [
        CONFIG_DIR,
        INPUT_DIR,
        AUDIO_DIR,
        TRANSCRIPT_DIR,
        JOB_DIR,
        LOG_DIR,
        MODEL_DIR,
        HF_CACHE_DIR,
        MODEL_DIR / "torch",
        OV_CACHE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("APP_MODEL_ROOT", str(MODEL_DIR))
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("TORCH_HOME", str(MODEL_DIR / "torch"))
    os.environ.setdefault("OV_CACHE_DIR", str(OV_CACHE_DIR))


def read_config() -> dict[str, Any]:
    """Read the persisted app config JSON, returning an empty dict on any failure."""
    ensure_app_dirs()
    if not APP_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_config(config: dict[str, Any]) -> None:
    """Write *config* to the persisted app config JSON file."""
    ensure_app_dirs()
    APP_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_config(**values: Any) -> dict[str, Any]:
    """Merge *values* into the persisted config and return the updated dict."""
    config = read_config()
    config.update(values)
    return config


def new_job_id() -> str:
    """Return a unique job ID string based on the current UTC timestamp plus a random suffix."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def safe_name(value: str) -> str:
    """Sanitise *value* for use as a filesystem-safe filename component."""
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "transcript"


def save_transcript(
    job_id: str,
    engine_name: str,
    text: str,
    output_name: str | None = None,
) -> str | None:
    """Write *text* to a transcript file and return its path, or None if skipped."""
    if not text or text.startswith(("(", "ERROR")):
        return None
    ensure_app_dirs()
    if output_name:
        path = TRANSCRIPT_DIR / f"{safe_name(output_name)}.txt"
    else:
        path = TRANSCRIPT_DIR / f"{job_id}_{safe_name(engine_name)}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def write_job_record(job_id: str, patch: dict[str, Any]) -> str:
    """Create or merge-update a per-job manifest JSON file."""
    ensure_app_dirs()
    path = JOB_DIR / f"{job_id}.json"
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    preserved_created = existing.get("created_at")
    existing.update(patch)
    if not existing.get("job_id"):
        existing["job_id"] = job_id
    if not existing.get("created_at"):
        existing["created_at"] = preserved_created or now_iso()
    existing["updated_at"] = now_iso()
    path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def load_job(job_id: str) -> dict[str, Any] | None:
    """Load a full job manifest dict, or None if missing/invalid."""
    ensure_app_dirs()
    path = JOB_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _job_timestamp(value: Any) -> str:
    """Normalize manifest timestamps; JSON null must not reach sort comparisons."""
    return value if isinstance(value, str) and value else ""


def _job_row_from_path(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    job_id = data.get("job_id") or path.stem
    engines = data.get("selected_engines") or []
    if isinstance(engines, str):
        engines = [engines]
    return {
        "job_id": job_id,
        "created_at": _job_timestamp(data.get("created_at")),
        "updated_at": _job_timestamp(data.get("updated_at")),
        "status": data.get("status", "unknown"),
        "display_name": data.get("display_name") or "",
        "source_filename": data.get("source_filename") or "",
        "selected_engines": engines,
        "audio_duration_s": float(data.get("audio_duration_s") or 0.0),
        "total_elapsed_s": float(data.get("total_elapsed_s") or 0.0),
        "tab_id": data.get("tab_id") or "",
        "progress": data.get("progress") or {},
        "results": data.get("results") or {},
    }


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    """Return job summary rows sorted by created_at descending."""
    ensure_app_dirs()
    rows: list[dict[str, Any]] = []
    for path in JOB_DIR.glob("*.json"):
        row = _job_row_from_path(path)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: row["created_at"], reverse=True)
    return rows[: max(1, limit)]


def copy_input_file(source_path: str, job_id: str, filename: str) -> str | None:
    """Best-effort copy of uploaded media into storage/input for job history."""
    import shutil

    ensure_app_dirs()
    src = Path(source_path)
    if not src.is_file():
        return None
    safe = safe_name(filename or src.name)
    dest = INPUT_DIR / f"{job_id}_{safe}"
    try:
        shutil.copy2(src, dest)
        return str(dest)
    except OSError:
        return None


def save_job_manifest(job_id: str, manifest: dict[str, Any]) -> str:
    """Serialise *manifest* to a per-job JSON file and return its path."""
    ensure_app_dirs()
    path = JOB_DIR / f"{job_id}.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
