"""Offline model cache helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from engines.model_cache import (
    DEFAULT_DIARIZATION_MODEL,
    apply_runtime_cache_env_defaults,
    configure_project_cache_paths,
    configured_diarization_model_id,
    diarization_pipeline_dependencies,
    has_cached_model_file,
    has_cached_pipeline,
    hf_offline_enabled,
    hub_pretrained_kwargs,
    pretrained_local_files_only,
    resolve_pretrained_checkpoint,
)


def test_configure_project_cache_paths_uses_absolute_model_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = configure_project_cache_paths(tmp_path)
    assert Path(paths["APP_MODEL_ROOT"]).is_absolute()
    assert paths["HF_HUB_CACHE"] == str(Path(paths["HF_HOME"]) / "hub")
    assert paths["TRANSFORMERS_CACHE"] == paths["HF_HUB_CACHE"]
    assert Path(paths["HF_HUB_CACHE"]).is_dir()


def test_apply_runtime_cache_env_defaults_forces_offline(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv("APP_AUTO_DOWNLOAD_MISSING_MODELS", "true")
    apply_runtime_cache_env_defaults()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["APP_AUTO_DOWNLOAD_MISSING_MODELS"] == "false"


def test_hub_pretrained_kwargs_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    configure_project_cache_paths(tmp_path)
    kwargs = hub_pretrained_kwargs("token")
    assert kwargs["local_files_only"] is True
    assert kwargs["token"] == "token"
    assert "cache_dir" in kwargs


def test_resolve_pretrained_checkpoint_requires_local_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    configure_project_cache_paths(tmp_path)
    with pytest.raises(RuntimeError, match="not in the local cache"):
        resolve_pretrained_checkpoint("example/missing-model")


def test_community_diarization_dependencies():
    model_id = configured_diarization_model_id()
    assert model_id == DEFAULT_DIARIZATION_MODEL
    deps = diarization_pipeline_dependencies(model_id)
    assert "pyannote/segmentation-3.0" in deps
    assert "pyannote/wespeaker-voxceleb-resnet34-LM" in deps


@pytest.mark.skipif(
    not has_cached_model_file("typhoon-ai/typhoon-whisper-large-v3"),
    reason="local model pack not present",
)
def test_local_asr_models_are_cached():
    assert has_cached_model_file("typhoon-ai/typhoon-whisper-large-v3")
    assert has_cached_model_file("nectec/Pathumma-whisper-th-large-v3")
    checkpoint = resolve_pretrained_checkpoint("typhoon-ai/typhoon-whisper-large-v3")
    assert Path(checkpoint).is_dir()


@pytest.mark.skipif(
    not has_cached_pipeline(DEFAULT_DIARIZATION_MODEL),
    reason="local diarization pack not present",
)
def test_local_diarization_pipeline_is_cached():
    assert has_cached_pipeline(DEFAULT_DIARIZATION_MODEL)


def test_has_cached_pipeline_checks_submodels(tmp_path, monkeypatch):
    import engines.model_cache as model_cache_module

    model_id = "example/test-pipeline"
    hub = tmp_path / "hub"
    main_snap = hub / "models--example--test-pipeline" / "snapshots" / "main"
    main_snap.mkdir(parents=True)
    (main_snap / "config.yaml").write_text("pipeline: test\n", encoding="utf-8")

    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setitem(
        model_cache_module.PYANNOTE_PIPELINE_DEPENDENCIES,
        model_id,
        ("example/dep-a",),
    )

    dep_snap = hub / "models--example--dep-a" / "snapshots" / "dep"
    dep_snap.mkdir(parents=True)
    (dep_snap / "config.json").write_text("{}", encoding="utf-8")
    (dep_snap / "model.safetensors").write_bytes(b"x" * 16)

    assert has_cached_pipeline(model_id) is True

    shutil.rmtree(dep_snap.parent.parent)
    assert has_cached_pipeline(model_id) is False


def test_typhoon_load_raises_when_cache_missing(monkeypatch, tmp_path):
    from engines import typhoon_asr

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    configure_project_cache_paths(tmp_path)
    typhoon_asr._pipeline_cache.clear()
    monkeypatch.setattr(typhoon_asr, "MODEL_ID", "example/missing-typhoon")

    with pytest.raises(RuntimeError, match="not in the local cache"):
        typhoon_asr._get_pipeline()


def test_pretrained_local_files_only_tracks_offline_flags(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    assert pretrained_local_files_only() is False
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert hf_offline_enabled() is True
    assert pretrained_local_files_only() is True
