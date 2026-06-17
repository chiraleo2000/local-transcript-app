"""Language-aware Auto ASR engine routing."""

import os

from backend.services.asr_local import (
    ENGINE_AUTO,
    ENGINE_PATHUMMA,
    ENGINE_TYPHOON,
    best_asr_engine_for_language,
    default_asr_engines,
    engine_for_preload,
    is_auto_engine,
    resolve_asr_engine,
    resolve_asr_engines,
)


def test_is_auto_engine_aliases():
    assert is_auto_engine("Auto")
    assert is_auto_engine("auto")
    assert not is_auto_engine(ENGINE_TYPHOON)


def test_best_engine_thai_quality():
    assert best_asr_engine_for_language("Thai") == ENGINE_TYPHOON


def test_best_engine_english():
    assert best_asr_engine_for_language("English") == ENGINE_TYPHOON


def test_best_engine_thai_fast_policy(monkeypatch):
    monkeypatch.setenv("ASR_AUTO_POLICY", "fast")
    assert best_asr_engine_for_language("Thai") == ENGINE_PATHUMMA


def test_resolve_auto_english(monkeypatch):
    monkeypatch.setenv("ASR_AUTO_POLICY", "quality")
    assert resolve_asr_engine("English", ENGINE_AUTO) == ENGINE_TYPHOON


def test_resolve_explicit_engine():
    assert resolve_asr_engine("Thai", ENGINE_PATHUMMA) == ENGINE_PATHUMMA


def test_resolve_asr_engines_list():
    assert resolve_asr_engines("English", [ENGINE_AUTO]) == [ENGINE_TYPHOON]


def test_default_asr_engines_auto(monkeypatch):
    monkeypatch.delenv("ASR_DEFAULT_ENGINES", raising=False)
    assert default_asr_engines() == [ENGINE_AUTO]


def test_default_asr_engines_from_env(monkeypatch):
    monkeypatch.setenv("ASR_DEFAULT_ENGINES", "Auto")
    assert default_asr_engines() == [ENGINE_AUTO]


def test_engine_for_preload_auto_warms_typhoon(monkeypatch):
    monkeypatch.setenv("ASR_AUTO_POLICY", "quality")
    assert engine_for_preload(ENGINE_AUTO) == ENGINE_TYPHOON
