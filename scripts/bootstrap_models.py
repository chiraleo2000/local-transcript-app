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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Prepare local folders and bootstrap all ASR engines."""
    if not os.getenv("HF_TOKEN"):
        logger.warning(
            "HF_TOKEN is not set. Gated models (Typhoon Whisper, pyannote) will "
            "fail to download. Set HF_TOKEN in the environment or .env before "
            "running this maintainer bootstrap step."
        )
    ensure_app_dirs()
    hardware = detect_hardware(refresh=True)
    logger.info("Backend policy: %s / %s", hardware["backend"], hardware["selected_device"])
    logger.info("Reason: %s", hardware["backend_reason"])

    failures: dict[str, str] = {}
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

    update_config(model_bootstrap={"ready": not failures, "failures": failures})
    if failures:
        logger.error("Model bootstrap finished with failures: %s", failures)
        return 1
    logger.info("All local models are ready. Bundle ./models/ into the installer payload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
