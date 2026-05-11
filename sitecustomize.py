"""Process-wide warning/log filters for the local transcript runtime."""

import logging
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_model_root = os.getenv("APP_MODEL_ROOT") or os.path.join(os.getcwd(), "models")
_hf_home = os.path.join(_model_root, "hf_cache")
os.environ.setdefault("APP_MODEL_ROOT", _model_root)
os.environ.setdefault("HF_HOME", _hf_home)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_hf_home, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_hf_home, "hub"))
os.environ.setdefault("TORCH_HOME", os.path.join(_model_root, "torch"))
os.environ.setdefault("OV_CACHE_DIR", os.path.join(_model_root, "ov_cache"))

for _cache_dir in [
    os.environ["APP_MODEL_ROOT"],
    os.environ["HF_HOME"],
    os.environ["HF_HUB_CACHE"],
    os.environ["TORCH_HOME"],
    os.environ["OV_CACHE_DIR"],
]:
    os.makedirs(_cache_dir, exist_ok=True)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if _env_bool("APP_SUPPRESS_WARNING_LOGS", True):
    logging.disable(logging.WARNING)

warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec.*|.*libtorchcodec.*|.*FFmpeg.*version.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"[\s\S]*torchcodec[\s\S]*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat-32.*|.*TF32.*",
)

for logger_name in [
    "torch.utils.flop_counter",
    "torch._dynamo",
    "torch._inductor",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
