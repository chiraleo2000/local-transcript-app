"""Model registry keeps diarization when only ASR is swapped."""

from unittest.mock import MagicMock, patch

from backend.services.asr_local import ALL_ENGINES

import backend.model_registry as registry


def test_unload_asr_models_only():
    with patch("backend.model_registry.unload_model") as unload:
        with patch("backend.model_registry.clear_accelerator_cache") as clear:
            registry.unload_asr_models()
    assert unload.call_count == len(ALL_ENGINES)
    clear.assert_called_once()


def test_unload_diarization_delegates():
    with patch("backend.model_registry.clear_diarization_model", create=True) as clear_diar:
        with patch(
            "backend.services.media_pipeline.clear_diarization_model",
            clear_diar,
        ):
            registry.unload_diarization_model()
    clear_diar.assert_called_once()


def test_unload_all_models():
    with patch.object(registry, "unload_asr_models") as asr:
        with patch.object(registry, "unload_diarization_model") as diar:
            registry.unload_all_models()
    asr.assert_called_once()
    diar.assert_called_once()


def test_engine_switch_does_not_touch_diarization(monkeypatch):
    from backend.services.asr_local import switch_asr_engine

    diar_unload = MagicMock()
    monkeypatch.setattr(
        "backend.services.media_pipeline.clear_diarization_model",
        diar_unload,
    )
    with patch("backend.services.asr_local.unload_model") as asr_unload:
        with patch("backend.services.asr_local.load_model"):
            with patch("backend.services.asr_local.model_is_loaded", return_value=False):
                switch_asr_engine("Pathumma Whisper")
    asr_unload.assert_called()
    diar_unload.assert_not_called()
