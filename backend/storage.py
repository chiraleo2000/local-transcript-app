"""App-local storage and config helpers."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = APP_ROOT / "config"
STORAGE_DIR = APP_ROOT / "storage"
INPUT_DIR = STORAGE_DIR / "input"
AUDIO_DIR = STORAGE_DIR / "audio"
TRANSCRIPT_DIR = STORAGE_DIR / "transcripts"
JOB_DIR = STORAGE_DIR / "jobs"
LOG_DIR = STORAGE_DIR / "logs"
MODEL_DIR = APP_ROOT / "models"
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
    os.environ.setdefault("TORCH_HOME", str(MODEL_DIR / "torch"))
    os.environ.setdefault("OV_CACHE_DIR", str(OV_CACHE_DIR))


def read_config() -> dict[str, Any]:
    ensure_app_dirs()
    if not APP_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_config(config: dict[str, Any]) -> None:
    ensure_app_dirs()
    APP_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_config(**values: Any) -> dict[str, Any]:
    config = read_config()
    config.update(values)
    write_config(config)
    return config


def new_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def safe_name(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "transcript"


def save_transcript(job_id: str, engine_name: str, text: str) -> str | None:
    if not text or text.startswith(("(", "ERROR")):
        return None
    ensure_app_dirs()
    path = TRANSCRIPT_DIR / f"{job_id}_{safe_name(engine_name)}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def save_job_manifest(job_id: str, manifest: dict[str, Any]) -> str:
    ensure_app_dirs()
    path = JOB_DIR / f"{job_id}.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
