"""Verify local Hugging Face caches under ./models/ before offline runtime.

Runtime is verify-only: it never downloads or deletes model files. Maintainers
populate ./models/ once via scripts/bootstrap_models.py, then all engine swaps
read from the local cache only.
"""

from __future__ import annotations

import argparse
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
from engines.model_cache import (
    apply_runtime_cache_env_defaults,
    configured_asr_model_ids,
    consolidate_misplaced_hub_caches,
    env_bool,
    has_cached_model_file,
    missing_cached_models,
    offline_cache_error_message,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

OPTIONAL_DIARIZATION_MODELS = (
    "pyannote/speaker-diarization-community-1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    "pyannote/speaker-diarization-3.1",
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


def main() -> int:
    _configure_cache_paths()
    apply_runtime_cache_env_defaults()
    moved = consolidate_misplaced_hub_caches(PROJECT_ROOT)
    if moved:
        logger.info("Consolidated misplaced hub caches: %s", ", ".join(moved))

    hub_cache = os.environ["HF_HUB_CACHE"]
    logger.info("Using Hugging Face hub cache: %s", hub_cache)
    logger.info("Offline verify-only mode (no Hugging Face downloads).")

    required = list(configured_asr_model_ids())
    require_diarization = env_bool("APP_REQUIRE_DIARIZATION_MODELS", False)
    if require_diarization:
        required.extend(OPTIONAL_DIARIZATION_MODELS[:3])

    missing = missing_cached_models(required)
    if missing:
        for model_id in missing:
            logger.error("%s", offline_cache_error_message(model_id))
        logger.error(
            "Model cache incomplete (%d/%d required missing). "
            "Run scripts/bootstrap_models.py once on a maintainer machine, "
            "then use the app fully offline.",
            len(missing),
            len(required),
        )
        return 1

    for model_id in required:
        logger.info("%s cache OK.", model_id)

    diarization_missing = missing_cached_models(OPTIONAL_DIARIZATION_MODELS)
    if diarization_missing and not require_diarization:
        logger.warning(
            "Optional diarization models missing (%s); ASR works offline without speaker labels.",
            ", ".join(diarization_missing),
        )

    logger.info("Local model cache verification passed.")
    return 0


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # pylint: disable=broad-exception-caught
        try:
            mod = __import__(name.replace("-", "_").split(".")[0])
            return getattr(mod, "__version__", "unknown")
        except ImportError:
            return "not installed"


def print_versions() -> None:
    print("=== Local Transcript App — versions ===")
    for pkg in ("torch", "transformers", "gradio", "pyannote.audio", "librosa", "accelerate", "fastapi"):
        print(f"  {pkg}: {_pkg_version(pkg)}")

    print("\n=== Model IDs (from env) ===")
    print(f"  PATHUMMA_MODEL_ID: {os.getenv('PATHUMMA_MODEL_ID', 'nectec/Pathumma-whisper-th-large-v3')}")
    print(f"  TYPHOON_MODEL_ID: {os.getenv('TYPHOON_MODEL_ID', 'typhoon-ai/typhoon-whisper-large-v3')}")
    print(f"  DIARIZATION_MODEL_ID: {os.getenv('DIARIZATION_MODEL_ID', 'pyannote/speaker-diarization-community-1')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify local HF model caches (offline runtime).")
    parser.add_argument("--versions", action="store_true", help="Print package/model versions and exit.")
    args = parser.parse_args()
    if args.versions:
        print_versions()
        raise SystemExit(0)
    raise SystemExit(main())
