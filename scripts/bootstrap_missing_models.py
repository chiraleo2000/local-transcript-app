"""Download missing Hugging Face snapshots when a token is available.

Inference stays offline. This runs once at container start so an online host
can fill ./models and then serve without further hub calls.
"""

from __future__ import annotations

import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.dotenv_load import load_dotenv_safe

load_dotenv_safe(os.path.join(PROJECT_ROOT, ".env"))

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from engines.model_cache import (  # noqa: E402
    _sync_hub_constants,
    configure_project_cache_paths,
    configured_asr_model_ids,
    configured_diarization_model_id,
    configured_turbo_model_id,
    diarization_pipeline_dependencies,
    env_bool,
    has_cached_model_file,
    hub_cache_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("bootstrap_missing_models")


def _download(model_id: str, token: str | None, cache_dir: str) -> str | None:
    if has_cached_model_file(model_id):
        logger.info("%s already cached.", model_id)
        return None
    from huggingface_hub import snapshot_download

    try:
        logger.info("Downloading %s ...", model_id)
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=False,
            token=token,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Download failed for %s", model_id)
        return str(exc)
    if not has_cached_model_file(model_id):
        message = "download finished but cache is still incomplete"
        logger.error("%s: %s", model_id, message)
        return message
    logger.info("%s cache OK.", model_id)
    return None


def main() -> int:
    configure_project_cache_paths(PROJECT_ROOT)
    _sync_hub_constants()
    token = os.getenv("HF_TOKEN") or None
    if not token:
        logger.warning("HF_TOKEN is unset; only already-cached models can be used.")
    cache_dir = str(hub_cache_dir())
    required = list(configured_asr_model_ids())
    if env_bool("APP_REQUIRE_DIARIZATION_MODELS", False):
        diar = configured_diarization_model_id()
        required.append(diar)
        required.extend(diarization_pipeline_dependencies(diar))
    optional = [configured_turbo_model_id()]

    required_failures: list[str] = []
    for model_id in required:
        if _download(model_id, token, cache_dir):
            required_failures.append(model_id)
    for model_id in optional:
        _download(model_id, token, cache_dir)

    if required_failures:
        logger.error("Required models still missing: %s", ", ".join(required_failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
