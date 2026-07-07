"""Build an offline Model Pack under ./models for releases/installers.

Maintainer-only: this script may download gated Hugging Face models (HF_TOKEN)
and export OpenVINO IR. End-user runtimes should stay fully offline and only
*verify* this pack via scripts/ensure_model_cache.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_DIARIZATION_MODELS = (
    "pyannote/speaker-diarization-community-1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
)


@dataclass(frozen=True)
class PackSpec:
    include_diarization: bool
    export_openvino: bool


def _sha256(path: Path, *, max_mb: int = 64) -> str | None:
    """Return sha256 for small-ish files; None for very large ones."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > max_mb * 1024 * 1024:
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _configure_model_root(model_root: Path) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    hf_home = model_root / "hf_cache"
    hub = hf_home / "hub"
    ov_cache = model_root / "ov_cache"
    torch_home = model_root / "torch"
    for p in (hf_home, hub, ov_cache, torch_home):
        p.mkdir(parents=True, exist_ok=True)

    os.environ["APP_MODEL_ROOT"] = str(model_root)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["OV_CACHE_DIR"] = str(ov_cache)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _set_online_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["APP_AUTO_DOWNLOAD_MISSING_MODELS"] = "true"


def _set_release_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["APP_AUTO_DOWNLOAD_MISSING_MODELS"] = "false"


def _download_models(model_ids: list[str]) -> None:
    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or None
    cache_dir = os.environ["HF_HUB_CACHE"]
    for model_id in model_ids:
        logger.info("Ensuring cached: %s", model_id)
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=False,
            token=token,
        )


def _export_openvino_ir() -> None:
    """Export OpenVINO IR for both ASR engines into OV_CACHE_DIR.

    Uses the same engine codepaths as runtime, but in an online maintainer context.
    """

    # Force OpenVINO path; export dir is under OV_CACHE_DIR.
    os.environ["APP_FORCE_BACKEND"] = "openvino"
    os.environ.setdefault("OV_DEVICE", "CPU")

    from backend.services.asr_local import ENGINE_PATHUMMA, ENGINE_TYPHOON, load_model, unload_model
    from backend.services.asr_local import clear_accelerator_cache

    for engine in (ENGINE_TYPHOON, ENGINE_PATHUMMA):
        logger.info("Exporting/loading OpenVINO IR for %s ...", engine)
        load_model(engine)
        unload_model(engine)
        clear_accelerator_cache()


def _snapshot_required_files(snapshot_dir: Path) -> list[dict]:
    required: list[dict] = []
    # include config + tokenizer-ish files (small)
    for name in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    ):
        p = snapshot_dir / name
        if p.is_file():
            required.append(
                {"path": name, "size": p.stat().st_size, "sha256": _sha256(p)}
            )
    # include weights/shards (no sha to avoid huge cost)
    for p in sorted(snapshot_dir.glob("model*.safetensors")):
        try:
            required.append({"path": p.name, "size": p.stat().st_size, "sha256": None})
        except OSError:
            continue
    return required


def _write_manifest(model_root: Path, model_ids: list[str], *, include_diarization: bool) -> Path:
    from engines.model_cache import cached_snapshot_path

    entries: list[dict] = []
    for model_id in model_ids:
        snap = cached_snapshot_path(model_id)
        if snap is None:
            raise RuntimeError(f"Missing cached snapshot for {model_id}")
        entries.append(
            {
                "model_id": model_id,
                "snapshot_dir": str(snap),
                "required_files": _snapshot_required_files(Path(snap)),
            }
        )

    ov_cache = model_root / "ov_cache"
    ov_exports = []
    for export_dir in sorted(ov_cache.glob("*")):
        if not export_dir.is_dir():
            continue
        encoder = export_dir / "openvino_encoder_model.xml"
        if encoder.is_file():
            ov_exports.append(
                {
                    "export_dir": str(export_dir),
                    "has_encoder_xml": True,
                    "has_decoder_xml": (export_dir / "openvino_decoder_model.xml").is_file(),
                }
            )

    manifest = {
        "schema": 1,
        "app": {"name": "local-transcript-app"},
        "pack": {
            "include_diarization": bool(include_diarization),
            "model_root": str(model_root),
        },
        "models": entries,
        "openvino_exports": ov_exports,
    }

    out = model_root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _make_archive(model_root: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # make_archive wants basename without extension
    base = out_path
    fmt = "zip"
    if out_path.suffix.lower() in {".tar", ".gz", ".tgz"}:
        fmt = "gztar"
        base = out_path.with_suffix("") if out_path.suffix.lower() != ".tar" else out_path
    if out_path.suffix.lower() == ".zip":
        fmt = "zip"
        base = out_path.with_suffix("")
    logger.info("Creating archive %s ...", out_path)
    shutil.make_archive(str(base), fmt, root_dir=str(model_root.parent), base_dir=model_root.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build offline Model Pack under ./models.")
    parser.add_argument("--model-root", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--include-diarization", action="store_true")
    parser.add_argument("--no-openvino-export", action="store_true")
    parser.add_argument("--archive", default="", help="Optional output archive path (zip or tar.gz).")
    args = parser.parse_args()

    model_root = Path(args.model_root).resolve()
    spec = PackSpec(
        include_diarization=bool(args.include_diarization),
        export_openvino=not bool(args.no_openvino_export),
    )

    _configure_model_root(model_root)
    _set_online_mode()

    from engines.model_cache import configured_asr_model_ids, has_cached_model_file

    model_ids = list(configured_asr_model_ids())
    if spec.include_diarization:
        model_ids.extend(DEFAULT_DIARIZATION_MODELS)

    if not os.getenv("HF_TOKEN") and any(m.startswith("pyannote/") or "typhoon-" in m for m in model_ids):
        logger.warning("HF_TOKEN is not set; gated models may fail to download.")

    _download_models(model_ids)

    missing = [m for m in model_ids if not has_cached_model_file(m)]
    if missing:
        raise RuntimeError(f"Cache still incomplete after download: {missing}")

    if spec.export_openvino:
        _export_openvino_ir()

    manifest_path = _write_manifest(model_root, model_ids, include_diarization=spec.include_diarization)
    logger.info("Wrote manifest: %s", manifest_path)

    # Flip back to strict offline defaults so subsequent runtime uses verify-only.
    _set_release_offline_mode()

    if args.archive:
        _make_archive(model_root, Path(args.archive).resolve())

    logger.info("Model Pack ready at %s", model_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

