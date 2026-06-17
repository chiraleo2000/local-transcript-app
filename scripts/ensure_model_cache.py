"""Verify local Hugging Face caches and repair missing/broken snapshot files.

Windows often stores HF hub entries as broken reparse-point symlinks unless
Developer Mode is enabled. This script materializes any missing model files
into ./models/hf_cache/hub before the app starts in offline-friendly mode.
"""

from __future__ import annotations

import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env.production"), override=False)

from backend.paths import app_root, resolve_path
from engines.model_cache import has_cached_model_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

REQUIRED_MODELS = (
    "nectec/Pathumma-whisper-th-large-v3",
    "typhoon-ai/typhoon-whisper-large-v3",
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
)


def _configure_cache_paths() -> str:
    """Point Hugging Face caches at the project-local models directory."""
    model_root = os.getenv("APP_MODEL_ROOT") or str(app_root() / "models")
    if not os.path.isabs(model_root):
        model_root = str(resolve_path(model_root))
    hf_home = os.path.join(model_root, "hf_cache")
    hub_cache = os.path.join(hf_home, "hub")

    os.environ["APP_MODEL_ROOT"] = model_root
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    os.environ.setdefault("TORCH_HOME", os.path.join(model_root, "torch"))
    os.environ.setdefault("OV_CACHE_DIR", os.path.join(model_root, "ov_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    for cache_dir in (
        model_root,
        hf_home,
        hub_cache,
        os.environ["TORCH_HOME"],
        os.environ["OV_CACHE_DIR"],
    ):
        os.makedirs(cache_dir, exist_ok=True)
    return hub_cache


def _materialize_model(model_id: str, hub_cache: str) -> None:
    if has_cached_model_file(model_id):
        logger.info("%s cache OK.", model_id)
        return

    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or None
    logger.info("Materializing %s into %s ...", model_id, hub_cache)
    snapshot_download(
        repo_id=model_id,
        cache_dir=hub_cache,
        local_files_only=False,
        token=token,
    )
    if not has_cached_model_file(model_id):
        raise RuntimeError(f"Model {model_id} is still missing after cache repair.")


def main() -> int:
    hub_cache = _configure_cache_paths()
    logger.info("Using Hugging Face hub cache: %s", hub_cache)

    failures: dict[str, str] = {}
    for model_id in REQUIRED_MODELS:
        try:
            _materialize_model(model_id, hub_cache)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failures[model_id] = str(exc)
            logger.error("Failed to prepare %s: %s", model_id, exc)

    if failures:
        logger.error(
            "Model cache incomplete (%d/%d failed). "
            "Set HF_TOKEN in .env for gated models, then rerun.",
            len(failures),
            len(REQUIRED_MODELS),
        )
        return 1

    logger.info("All %d local model caches are ready.", len(REQUIRED_MODELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
