"""Hugging Face cache helpers for local model loading."""

from __future__ import annotations

import os
from pathlib import Path


def env_bool(name: str, default: bool) -> bool:
    """Return a boolean from a common environment flag value."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def hf_offline_enabled() -> bool:
    """Return whether Hugging Face hub offline mode is enabled."""
    return env_bool("HF_HUB_OFFLINE", False) or env_bool("TRANSFORMERS_OFFLINE", False)


def pretrained_local_files_only() -> bool:
    """Return the local_files_only value for from_pretrained calls."""
    return hf_offline_enabled()


def _hub_cache_dir() -> Path:
    cache_dir = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if cache_dir:
        return Path(cache_dir)
    hf_home = os.getenv("HF_HOME") or os.path.join(os.getcwd(), "models", "hf_cache")
    return Path(hf_home) / "hub"


def _model_cache_dir(model_id: str) -> Path:
    return _hub_cache_dir() / f"models--{model_id.replace('/', '--')}"


def has_cached_model_file(model_id: str, filename: str = "config.json") -> bool:
    """Return whether a Hugging Face snapshot contains a required model file."""
    snapshots_dir = _model_cache_dir(model_id) / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    return any(snapshot.joinpath(filename).exists() for snapshot in snapshots_dir.iterdir())


def allow_online_download_if_missing(model_id: str, logger) -> None:
    """Disable offline flags when the requested model is not cached yet.

    Model swaps should not brick the runtime just because a previous version had
    offline mode enabled after bootstrapping a different model.
    """
    if has_cached_model_file(model_id):
        return
    if not hf_offline_enabled() or not env_bool("APP_AUTO_DOWNLOAD_MISSING_MODELS", True):
        return

    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.HF_HUB_OFFLINE = False
    except (ImportError, AttributeError):
        pass
    logger.warning(
        "Model %s is not in the local Hugging Face cache; enabling online download.",
        model_id,
    )