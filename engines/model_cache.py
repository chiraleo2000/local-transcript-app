"""Hugging Face cache helpers for local, offline model loading."""

# pylint: disable=duplicate-code

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

try:
    import huggingface_hub.constants as hub_constants
except ImportError:
    hub_constants = None

logger = logging.getLogger(__name__)

_CONFIG_JSON = "config.json"
_CONFIG_YAML = "config.yaml"
_CONFIG_CANDIDATES = (_CONFIG_JSON, _CONFIG_YAML)

DEFAULT_ASR_MODELS = (
    "nectec/Pathumma-whisper-th-large-v3",
    "typhoon-ai/typhoon-whisper-large-v3",
)

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# Pyannote pipeline repos reference weights in separate hub repos.
PYANNOTE_PIPELINE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "pyannote/speaker-diarization-3.1": (
        "pyannote/segmentation-3.0",
        "pyannote/wespeaker-voxceleb-resnet34-LM",
    ),
    "pyannote/speaker-diarization-community-1": (
        "pyannote/segmentation-3.0",
        "pyannote/wespeaker-voxceleb-resnet34-LM",
    ),
}


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


def hub_pretrained_kwargs(hf_token: str | None = None) -> dict:
    """Common kwargs for Hugging Face from_pretrained / hub loads."""
    kwargs: dict = {"local_files_only": pretrained_local_files_only()}
    if hf_token:
        kwargs["token"] = hf_token
    cache = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if cache:
        kwargs["cache_dir"] = cache
    return kwargs


def resolve_pretrained_checkpoint(model_id: str) -> str:
    """Return a local snapshot path when offline; otherwise the hub model id."""
    _sync_hub_constants()
    if hf_offline_enabled():
        snapshot = cached_snapshot_path(model_id)
        if snapshot is None:
            raise RuntimeError(offline_cache_error_message(model_id))
        return str(snapshot)
    return model_id


def hub_cache_dir() -> Path:
    """Return the active Hugging Face hub cache directory."""
    cache_dir = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if cache_dir:
        return Path(cache_dir)
    hf_home = os.getenv("HF_HOME") or os.path.join(os.getcwd(), "models", "hf_cache")
    return Path(hf_home) / "hub"


def _hub_cache_dir() -> Path:
    return hub_cache_dir()


def _model_cache_dir(model_id: str) -> Path:
    return _hub_cache_dir() / f"models--{model_id.replace('/', '--')}"


def _sync_hub_constants() -> None:
    """Push offline/cache flags into huggingface_hub after env changes."""
    if hub_constants is None:
        return
    try:
        hub_constants.HF_HUB_OFFLINE = hf_offline_enabled()
    except AttributeError:
        pass
    cache = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if cache:
        try:
            hub_constants.HF_HUB_CACHE = cache
        except AttributeError:
            pass


def configure_project_cache_paths(
    project_root: str | Path | None = None,
    *,
    model_root: str | Path | None = None,
) -> dict[str, str]:
    """Pin Hugging Face, Torch, and OpenVINO caches under a project-local tree."""
    root = Path(project_root or os.getcwd()).resolve()
    resolved_model_root = Path(model_root or os.getenv("APP_MODEL_ROOT") or root / "models")
    if not resolved_model_root.is_absolute():
        resolved_model_root = (root / resolved_model_root).resolve()

    hf_home = resolved_model_root / "hf_cache"
    hub_cache = hf_home / "hub"
    torch_home = resolved_model_root / "torch"
    ov_cache = resolved_model_root / "ov_cache"
    paths = {
        "APP_MODEL_ROOT": str(resolved_model_root),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hub_cache),
        "HUGGINGFACE_HUB_CACHE": str(hub_cache),
        "TRANSFORMERS_CACHE": str(hub_cache),
        "TORCH_HOME": str(torch_home),
        "OV_CACHE_DIR": str(ov_cache),
    }
    for key, value in paths.items():
        os.environ.setdefault(key, value)
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return paths


def apply_runtime_cache_env_defaults() -> None:
    """Default to offline, project-local cache unless explicitly overridden."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("APP_AUTO_DOWNLOAD_MISSING_MODELS", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("ASR_PRELOAD_MODE", "eager")
    os.environ.setdefault("DIARIZATION_PRELOAD_MODE", "eager")
    if env_bool("APP_AUTO_DOWNLOAD_MISSING_MODELS", False):
        logger.warning(
            "APP_AUTO_DOWNLOAD_MISSING_MODELS=true is ignored at runtime; "
            "populate ./models/ once via scripts/bootstrap_models.py, then run offline."
        )
        os.environ["APP_AUTO_DOWNLOAD_MISSING_MODELS"] = "false"
    _sync_hub_constants()


def consolidate_misplaced_hub_caches(project_root: str | Path | None = None) -> list[str]:
    """Move stray models--* folders from the project root into ./models/hf_cache/hub."""
    root = Path(project_root or os.getcwd())
    dest_hub = _hub_cache_dir()
    dest_hub.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for repo_dir in root.glob("models--*"):
        if not repo_dir.is_dir():
            continue
        model_id = repo_dir.name.replace("models--", "").replace("--", "/")
        destination = dest_hub / repo_dir.name
        if destination.exists():
            if has_cached_model_file(model_id):
                logger.info("Removing duplicate root cache %s; hub copy is complete.", repo_dir.name)
                shutil.rmtree(repo_dir, ignore_errors=True)
                continue
            logger.info("Replacing incomplete hub cache for %s with %s", model_id, repo_dir)
            shutil.rmtree(destination, ignore_errors=True)
        try:
            logger.info("Moving misplaced cache %s -> %s", repo_dir, destination)
            shutil.move(str(repo_dir), str(destination))
            moved.append(repo_dir.name)
        except OSError as exc:
            logger.warning("Could not move %s (%s); remove it manually if duplicate.", repo_dir, exc)
    return moved


def cached_snapshot_path(model_id: str) -> Path | None:
    """Return the newest usable snapshot directory for *model_id*, if cached."""
    snapshots_dir = _model_cache_dir(model_id) / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    usable: list[tuple[float, Path]] = []
    for snapshot in snapshots_dir.iterdir():
        if not snapshot.is_dir():
            continue
        for name in _CONFIG_CANDIDATES:
            candidate = snapshot / name
            if _snapshot_file_usable(candidate):
                try:
                    usable.append((candidate.stat().st_mtime, snapshot))
                except OSError:
                    continue
                break
    if not usable:
        return None
    usable.sort(key=lambda item: item[0], reverse=True)
    return usable[0][1]


def missing_cached_models(model_ids: tuple[str, ...] | list[str]) -> list[str]:
    """Return model IDs that are not present in the local Hugging Face cache."""
    return [model_id for model_id in model_ids if not has_cached_model_file(model_id)]


def _snapshot_file_usable(path: Path) -> bool:
    """Return whether a cached snapshot entry is readable (not a broken symlink)."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _snapshot_shard_files(snapshot: Path) -> set[str]:
    """Return the weight filenames required for a snapshot (handles sharded checkpoints)."""
    index_path = snapshot / "model.safetensors.index.json"
    if _snapshot_file_usable(index_path):
        try:
            import json

            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map", {})
            if isinstance(weight_map, dict) and weight_map:
                return {str(name) for name in weight_map.values()}
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    shards = {
        path.name
        for path in snapshot.glob("model-*.safetensors")
        if _snapshot_file_usable(path)
    }
    if shards:
        return shards
    for name in ("model.safetensors", "pytorch_model.bin", "model.bin"):
        if _snapshot_file_usable(snapshot / name):
            return {name}
    return set()


def _snapshot_weights_complete(snapshot: Path) -> bool:
    """Return whether all required weight files for a snapshot are present."""
    required = _snapshot_shard_files(snapshot)
    if not required:
        return False
    return all(_snapshot_file_usable(snapshot / name) for name in required)


def _pipeline_bundle_complete(snapshot: Path) -> bool:
    """Pyannote pipeline repos may ship weights under subfolders (embedding/, etc.)."""
    if not _snapshot_file_usable(snapshot / _CONFIG_YAML):
        return False
    if _snapshot_weights_complete(snapshot):
        return True
    weight_names = ("pytorch_model.bin", "model.safetensors", "model.bin")
    for sub in snapshot.iterdir():
        if not sub.is_dir() or sub.name in {".github", "reproducible_research"}:
            continue
        for name in weight_names:
            if _snapshot_file_usable(sub / name):
                return True
    return False


def _hf_pipeline_complete(model_id: str, snapshot: Path) -> bool:
    """Return whether a Hugging Face pipeline snapshot is usable offline."""
    if _pipeline_bundle_complete(snapshot):
        return True
    if not _snapshot_file_usable(snapshot / _CONFIG_YAML):
        return False
    deps = diarization_pipeline_dependencies(model_id)
    if not deps:
        return False
    return all(has_cached_model_file(dep) for dep in deps)


def has_cached_model_file(model_id: str, filename: str | None = None) -> bool:
    """Return whether a Hugging Face snapshot contains a required model file."""
    snapshots_dir = _model_cache_dir(model_id) / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    candidates = (filename,) if filename else _CONFIG_CANDIDATES
    for snapshot in snapshots_dir.iterdir():
        if not snapshot.is_dir():
            continue
        has_config = any(_snapshot_file_usable(snapshot / name) for name in candidates)
        if not has_config:
            continue
        if filename:
            return _snapshot_file_usable(snapshot / filename)
        if _snapshot_weights_complete(snapshot):
            return True
        if _hf_pipeline_complete(model_id, snapshot):
            return True
    return False


def configured_diarization_model_id() -> str:
    """Return the configured pyannote diarization pipeline model id."""
    return os.getenv("DIARIZATION_MODEL_ID", DEFAULT_DIARIZATION_MODEL)


def diarization_pipeline_dependencies(model_id: str) -> tuple[str, ...]:
    """Return sub-model repos required by a pyannote pipeline id."""
    return PYANNOTE_PIPELINE_DEPENDENCIES.get(model_id, ())


def has_cached_pipeline(model_id: str) -> bool:
    """Return whether a pyannote pipeline and its sub-models are cached offline."""
    if not has_cached_model_file(model_id):
        return False
    return all(has_cached_model_file(dep) for dep in diarization_pipeline_dependencies(model_id))


def configured_asr_model_ids() -> tuple[str, ...]:
    """Return all ASR model IDs that may be selected at runtime."""
    model_ids = [
        os.getenv("PATHUMMA_MODEL_ID", DEFAULT_ASR_MODELS[0]),
        os.getenv("TYPHOON_MODEL_ID", DEFAULT_ASR_MODELS[1]),
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for model_id in model_ids:
        if model_id and model_id not in seen:
            seen.add(model_id)
            ordered.append(model_id)
    return tuple(ordered)


def require_cached_model(model_id: str, logger) -> None:
    """Fail fast when a model is missing from the local cache (never goes online)."""
    if has_cached_model_file(model_id):
        return
    message = offline_cache_error_message(model_id)
    logger.error("%s", message)
    raise RuntimeError(message)


def require_cached_pipeline(model_id: str, logger) -> None:
    """Fail fast when a pyannote pipeline or its sub-models are missing locally."""
    require_cached_model(model_id, logger)
    for dep in diarization_pipeline_dependencies(model_id):
        require_cached_model(dep, logger)


def offline_cache_error_message(model_id: str) -> str:
    """Human-readable guidance when a model is missing in offline mode."""
    return (
        f"Model '{model_id}' is not in the local cache at {_model_cache_dir(model_id)}. "
        "Place the model under ./models/hf_cache/hub/ (maintainer bootstrap) before running offline."
    )
