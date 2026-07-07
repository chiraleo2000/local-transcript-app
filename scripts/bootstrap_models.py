"""Bootstrap local models and app folders for the transcript app.

This script has two roles:

1. **Build-time / maintainer use** (default): pre-cache gated Hugging Face
   models into ./models/ using a valid HF_TOKEN, so the resulting cache can
   be bundled into the GUI installer payload. End-user runtimes consume the
   cache offline and never need a token.

2. **Developer venv setup**: pre-warm models on a development workstation.

For the maintainer flow we temporarily disable offline flags so the cache
can actually be populated, regardless of the production .env that may have
shipped on this machine.
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import logging
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

# Build-time mode: ensure we can actually download gated models into the
# bundled cache. The packaged .env.production keeps these offline at runtime.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["APP_AUTO_DOWNLOAD_MISSING_MODELS"] = "true"

from backend.services.asr_local import (  # noqa: E402
    ALL_ENGINES,
    clear_accelerator_cache,
    load_model,
    unload_model,
)
from backend.services.hardware_policy import detect_hardware  # noqa: E402
from backend.storage import ensure_app_dirs, update_config  # noqa: E402
from engines.model_cache import (  # noqa: E402
    configured_asr_model_ids,
    consolidate_misplaced_hub_caches,
    has_cached_model_file,
    hub_cache_dir,
)

MAINTAINER_MODELS = (
    *configured_asr_model_ids(),
    "pyannote/speaker-diarization-community-1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _download_missing_models() -> dict[str, str]:
    """Download any maintainer models that are not yet complete in ./models/."""
    from huggingface_hub import snapshot_download

    failures: dict[str, str] = {}
    token = os.getenv("HF_TOKEN") or None
    cache_dir = str(hub_cache_dir())
    for model_id in MAINTAINER_MODELS:
        if has_cached_model_file(model_id):
            logger.info("%s already cached.", model_id)
            continue
        try:
            logger.info("Downloading %s into %s ...", model_id, cache_dir)
            snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_files_only=False,
                token=token,
            )
            if not has_cached_model_file(model_id):
                failures[model_id] = "download finished but cache is still incomplete"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failures[model_id] = str(exc)
            logger.error("Download failed for %s: %s", model_id, exc)
    return failures


def main() -> int:
    """Prepare local folders and bootstrap all ASR engines."""
    if not os.getenv("HF_TOKEN"):
        logger.warning(
            "HF_TOKEN is not set. Gated models (Typhoon Whisper, pyannote) will "
            "fail to download. Set HF_TOKEN in the environment or .env before "
            "running this maintainer bootstrap step."
        )
    ensure_app_dirs()
    consolidate_misplaced_hub_caches(PROJECT_ROOT)
    download_failures = _download_missing_models()
    hardware = detect_hardware(refresh=True)
    logger.info("Backend policy: %s / %s", hardware["backend"], hardware["selected_device"])
    logger.info("Reason: %s", hardware["backend_reason"])

    failures: dict[str, str] = dict(download_failures)
    for engine in ALL_ENGINES:
        try:
            logger.info("Preparing %s...", engine)
            load_model(engine)
            logger.info("%s ready.", engine)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failures[engine] = str(exc)
            logger.error("%s bootstrap failed: %s", engine, exc, exc_info=True)
        finally:
            try:
                unload_model(engine)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("%s unload skipped during bootstrap: %s", engine, exc)
            clear_accelerator_cache()

    try:
        from engines.diarization import load_model as load_diarization_model
        from engines.diarization import unload_model as unload_diarization_model

        logger.info("Preparing pyannote diarization...")
        load_diarization_model()
        logger.info("Pyannote diarization ready.")
        unload_diarization_model()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        failures["diarization"] = str(exc)
        logger.error("Diarization bootstrap failed: %s", exc, exc_info=True)

    update_config(model_bootstrap={"ready": not failures, "failures": failures})
    if failures:
        logger.error("Model bootstrap finished with failures: %s", failures)
        return 1
    logger.info("All local models are ready. Bundle ./models/ into the installer payload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
