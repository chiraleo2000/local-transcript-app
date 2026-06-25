"""Runtime bootstrap for golden automation (no pytest dependency)."""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def bootstrap_golden_runtime() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    os.chdir(REPO_ROOT)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model_root = os.getenv("APP_MODEL_ROOT") or str(REPO_ROOT / "models")
    if not os.path.isabs(model_root):
        model_root = str(REPO_ROOT / model_root)
    hf_home = os.path.join(model_root, "hf_cache")
    os.environ.setdefault("APP_MODEL_ROOT", model_root)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("TORCH_HOME", os.path.join(model_root, "torch"))
    os.environ.setdefault("OV_CACHE_DIR", os.path.join(model_root, "ov_cache"))
    for cache_dir in [
        os.environ["APP_MODEL_ROOT"],
        os.environ["HF_HOME"],
        os.environ["HF_HUB_CACHE"],
        os.environ["TORCH_HOME"],
        os.environ["OV_CACHE_DIR"],
    ]:
        os.makedirs(cache_dir, exist_ok=True)

    _install_torchcodec_stub()


def _install_torchcodec_stub() -> None:
    """Satisfy transformers metadata probe without native torchcodec DLLs."""
    import site

    for site_dir in site.getsitepackages():
        dist_info = Path(site_dir) / "torchcodec-0.0.1.dist-info"
        if dist_info.is_dir():
            break
        try:
            dist_info.mkdir(parents=True, exist_ok=True)
            (dist_info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: torchcodec\nVersion: 0.0.1\n",
                encoding="utf-8",
            )
            (dist_info / "RECORD").write_text("", encoding="utf-8")
            (dist_info / "INSTALLER").write_text("pip", encoding="utf-8")
            break
        except OSError:
            continue

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
            setattr(module, "AudioDecoder", type("AudioDecoder", (), {}))
            setattr(module, "AudioSamples", type("AudioSamples", (), {}))
            setattr(module, "AudioStreamMetadata", type("AudioStreamMetadata", (), {}))
        setattr(torchcodec, name, module)
        sys.modules[f"torchcodec.{name}"] = module
    sys.modules["torchcodec"] = torchcodec
