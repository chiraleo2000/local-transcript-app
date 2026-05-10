"""Bootstrap local models and app folders for the transcript app.

This script is designed for installers and normal-user setup flows. It creates
the app-local storage/model folders, records hardware policy, and pre-downloads
or exports the ASR models for the selected backend.
"""

from __future__ import annotations

import logging
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

from backend.services.asr_local import ALL_ENGINES, load_model
from backend.services.hardware_policy import detect_hardware
from backend.storage import ensure_app_dirs, update_config


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
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

    update_config(model_bootstrap={"ready": not failures, "failures": failures})
    if failures:
        logger.error("Model bootstrap finished with failures: %s", failures)
        return 1
    logger.info("All local models are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
