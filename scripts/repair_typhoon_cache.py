"""Maintainer-only: complete the Typhoon Whisper HF cache under ./models/."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

MODEL_ID = os.getenv("TYPHOON_MODEL_ID", "typhoon-ai/typhoon-whisper-large-v3")


def _enable_online_hub() -> None:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.HF_HUB_OFFLINE = False
    except (ImportError, AttributeError):
        pass


load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
_enable_online_hub()
os.environ.setdefault("APP_MODEL_ROOT", os.path.join(PROJECT_ROOT, "models"))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "models", "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(PROJECT_ROOT, "models", "hf_cache", "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def main() -> int:
    _enable_online_hub()
    from huggingface_hub import snapshot_download

    from engines.model_cache import has_cached_model_file

    if has_cached_model_file(MODEL_ID):
        print(f"{MODEL_ID} cache is already complete.")
        return 0

    token = os.getenv("HF_TOKEN") or None
    cache_dir = os.environ["HF_HUB_CACHE"]
    print(f"Downloading {MODEL_ID} into {cache_dir} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=cache_dir,
        local_files_only=False,
        token=token,
    )
    if not has_cached_model_file(MODEL_ID):
        print(f"ERROR: {MODEL_ID} is still incomplete after download.")
        return 1
    print(f"{MODEL_ID} cache is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
