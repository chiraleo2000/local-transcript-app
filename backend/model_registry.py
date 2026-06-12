"""Split ASR and diarization model unload paths."""

from __future__ import annotations

import logging

from backend.services.asr_local import ALL_ENGINES, clear_accelerator_cache, unload_model

logger = logging.getLogger(__name__)


def unload_asr_models() -> None:
    """Unload Typhoon and Pathumma only."""
    for engine in ALL_ENGINES:
        try:
            unload_model(engine)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("ASR unload failed for %s: %s", engine, exc)
    clear_accelerator_cache()


def unload_diarization_model() -> None:
    """Unload pyannote diarization only."""
    from backend.services.media_pipeline import clear_diarization_model

    try:
        clear_diarization_model()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("Diarization unload failed: %s", exc)


def unload_all_models() -> None:
    """Full pipeline reset (cancel with unload, debug)."""
    unload_asr_models()
    unload_diarization_model()
