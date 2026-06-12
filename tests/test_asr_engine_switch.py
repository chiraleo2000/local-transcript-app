"""ASR engine swap — only on explicit UI selection."""

from unittest.mock import patch

from backend.services.asr_local import ENGINE_PATHUMMA, ENGINE_TYPHOON, switch_asr_engine


def test_switch_asr_engine_unloads_others():
    unloaded: list[str] = []
    loaded: list[str] = []

    def _unload(name):
        unloaded.append(name)

    def _load(name):
        loaded.append(name)

    with (
        patch("backend.services.asr_local.unload_model", side_effect=_unload),
        patch("backend.services.asr_local.load_model", side_effect=_load),
        patch("backend.services.asr_local.model_is_loaded", return_value=False),
    ):
        switch_asr_engine(ENGINE_PATHUMMA)

    assert ENGINE_TYPHOON in unloaded
    assert ENGINE_PATHUMMA in loaded


def test_switch_skips_reload_when_already_loaded():
    with (
        patch("backend.services.asr_local.unload_model") as unload_mock,
        patch("backend.services.asr_local.load_model") as load_mock,
        patch("backend.services.asr_local.model_is_loaded", return_value=True),
    ):
        switch_asr_engine(ENGINE_PATHUMMA)

    unload_mock.assert_called_once_with(ENGINE_TYPHOON)
    load_mock.assert_not_called()
