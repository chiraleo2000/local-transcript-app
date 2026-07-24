"""Shared startup for standalone scripts.

Reproduces the essential parts of app.py's bootstrap (env loading, HF cache
dirs, torchcodec stub, quality profile) WITHOUT importing gradio, so the
scripts run outside the Gradio app / Docker container.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def install_torchcodec_stub() -> None:
    """Avoid torchcodec native DLL failures (librosa is used for audio input)."""
    if "torchcodec" in sys.modules:
        return
    torchcodec = types.ModuleType("torchcodec")
    torchcodec.__spec__ = importlib.machinery.ModuleSpec("torchcodec", None)
    torchcodec.AudioDecoder = type("AudioDecoder", (), {})
    torchcodec.AudioSamples = type("AudioSamples", (), {})
    torchcodec.AudioStreamMetadata = type("AudioStreamMetadata", (), {})
    for name in ["decoders", "encoders", "samplers", "transforms"]:
        module = types.ModuleType(f"torchcodec.{name}")
        module.__spec__ = importlib.machinery.ModuleSpec(f"torchcodec.{name}", None)
        if name == "decoders":
            module.AudioDecoder = type("AudioDecoder", (), {})
            module.AudioSamples = type("AudioSamples", (), {})
            module.AudioStreamMetadata = type("AudioStreamMetadata", (), {})
        setattr(torchcodec, name, module)
        sys.modules[f"torchcodec.{name}"] = module
    sys.modules["torchcodec"] = torchcodec


def bootstrap() -> None:
    """Load .env, configure HF cache dirs, install the stub, apply the profile."""
    import warnings

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    # expandable_segments is Linux-only; on Windows it spams CUDAAllocator warnings.
    if sys.platform != "win32":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    warnings.filterwarnings(
        "ignore",
        message=".*doesn't match a supported version.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Whisper did not predict an ending timestamp.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*return_token_timestamps.*deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*sequentially on GPU.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*expandable_segments not supported.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*degrees of freedom is <= 0.*",
    )

    import logging

    from engines.whisper_utils import install_whisper_pipeline_log_filters

    install_whisper_pipeline_log_filters()

    from backend.dotenv_load import load_dotenv_safe

    # Docker compose `environment:` / `docker compose run -e` must win over .env files.
    _preserve_prefixes = ("ASR_", "DIARIZATION_", "CUDA_", "UI_", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    _preserved = {
        k: v for k, v in os.environ.items()
        if k.startswith(_preserve_prefixes) or k == "CUDA_AUTO_RESTART"
    }
    load_dotenv_safe(REPO / ".env")
    load_dotenv_safe(REPO / ".env.production", override=False)
    for key, value in _preserved.items():
        os.environ[key] = value

    model_root = os.getenv("APP_MODEL_ROOT") or str(REPO / "models")
    if not os.path.isabs(model_root):
        model_root = str((REPO / model_root).resolve())
    hf_home = os.path.join(model_root, "hf_cache")
    os.environ["APP_MODEL_ROOT"] = model_root
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["TORCH_HOME"] = os.path.join(model_root, "torch")
    os.environ["OV_CACHE_DIR"] = os.path.join(model_root, "ov_cache")

    from engines.model_cache import (
        apply_runtime_cache_env_defaults,
        consolidate_misplaced_hub_caches,
    )

    apply_runtime_cache_env_defaults()
    consolidate_misplaced_hub_caches(REPO)

    install_torchcodec_stub()

    from backend.asr_quality import apply_quality_profile

    apply_quality_profile()
