#!/usr/bin/env python3
"""Package a clean offline release zip (source + preloaded models) for GitHub.

Creates under release/vX.Y.Z/:
  LocalTranscriptApp-vX.Y.Z-offline.zip   (or .001/.002 split parts)
  install.bat / install.sh
  SHA256SUMS.txt
  join_offline_zip.ps1 / join_offline_zip.sh  (when split)

HF hub caches on Windows often use reparse points; this script materializes
them into real files before zipping so the archive is portable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".cache",
    ".nv",
    ".cursor",
    ".claude",
    ".vscode",
    "intel",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "release",
}
EXCLUDE_DIR_PATHS = {
    "storage/audio",
    "storage/input",
    "storage/jobs",
    "storage/transcripts",
    "storage/logs",
    "storage/acceptance_output",
    "tests/output",
    "models/hf_cache/xet",
    "models/hf_cache/hub/.locks",
}
EXCLUDE_FILE_NAMES = {
    ".env",
    ".coverage",
    "coverage.xml",
    "Thumbs.db",
    ".DS_Store",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".log", ".lock", ".incomplete"}


def _rel(path: Path, root: Path = PROJECT_ROOT) -> str:
    return path.relative_to(root).as_posix()


def _should_skip_source(path: Path) -> bool:
    rel = _rel(path)
    if path.name in EXCLUDE_FILE_NAMES:
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    parts = Path(rel).parts
    if any(part in EXCLUDE_DIR_NAMES for part in parts):
        return True
    # Models are staged separately (materialized).
    if rel == "models" or rel.startswith("models/"):
        return True
    for blocked in EXCLUDE_DIR_PATHS:
        if rel == blocked or rel.startswith(blocked + "/"):
            return True
    return False


def _is_reparse_point(path: Path) -> bool:
    try:
        return bool(os.lstat(path).st_file_attributes & 0x400)  # type: ignore[attr-defined]
    except AttributeError:
        # Non-Windows or older Python: fall back to pathlib.
        try:
            return path.is_symlink()
        except OSError:
            return False
    except OSError:
        return False


def _reparse_relative_target(path: Path) -> str | None:
    """Parse `fsutil reparsepoint query` output for a relative target path."""
    try:
        proc = subprocess.run(
            ["fsutil", "reparsepoint", "query", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    bytes_list: list[int] = []
    for line in proc.stdout.splitlines():
        match = re.match(r"^\s*[0-9a-fA-F]{4}:\s+((?:[0-9a-fA-F]{2}\s+){1,16})", line)
        if not match:
            continue
        for hex_byte in match.group(1).split():
            bytes_list.append(int(hex_byte, 16))
    if len(bytes_list) <= 4:
        return None
    raw = bytes(bytes_list[4:]).decode("ascii", errors="ignore")
    # Trim NULs / junk after first path-looking token.
    raw = raw.split("\x00", 1)[0].strip()
    return raw.replace("/", "\\") if raw else None


def _resolve_portable_source(path: Path) -> Path | None:
    """Return a readable real file path for packaging, or None if unreadable."""
    if _is_reparse_point(path):
        rel_target = _reparse_relative_target(path)
        if not rel_target:
            return None
        target = (path.parent / rel_target).resolve()
        if not target.is_file():
            return None
        return target
    try:
        with path.open("rb"):
            return path
    except OSError:
        return None


def _read_ref_commit(repo_dir: Path) -> str | None:
    ref = repo_dir / "refs" / "main"
    if not ref.is_file():
        refs = list((repo_dir / "refs").rglob("*")) if (repo_dir / "refs").is_dir() else []
        files = [p for p in refs if p.is_file()]
        if not files:
            return None
        ref = files[0]
    try:
        return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _materialize_hf_hub(src_hub: Path, dst_hub: Path) -> None:
    """Copy HF hub repos with reparse points turned into real files.

    Only the refs/main snapshot is materialized (skips stale broken snapshots).
    Blob directories are omitted — snapshots contain full file copies.
    """
    dst_hub.mkdir(parents=True, exist_ok=True)
    if not src_hub.is_dir():
        return
    for repo in sorted(src_hub.iterdir()):
        if not repo.is_dir() or repo.name in {".locks", ".cache"}:
            continue
        commit = _read_ref_commit(repo)
        dest_repo = dst_hub / repo.name
        dest_repo.mkdir(parents=True, exist_ok=True)

        # refs
        refs_src = repo / "refs"
        if refs_src.is_dir():
            shutil.copytree(refs_src, dest_repo / "refs", dirs_exist_ok=True)

        # snapshots/<current>
        snap_root = repo / "snapshots"
        if commit and (snap_root / commit).is_dir():
            snap_dirs = [snap_root / commit]
        else:
            snap_dirs = [p for p in snap_root.iterdir() if p.is_dir()] if snap_root.is_dir() else []

        for snap in snap_dirs:
            dest_snap = dest_repo / "snapshots" / snap.name
            dest_snap.mkdir(parents=True, exist_ok=True)
            for child in snap.iterdir():
                if child.is_dir():
                    continue
                resolved = _resolve_portable_source(child)
                if resolved is None:
                    print(f"[package] skip unreadable: {_rel(child)}")
                    continue
                shutil.copy2(resolved, dest_snap / child.name)

        # Keep refs only; no blobs needed after materialization.


def _copy_tree_files(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        # prune caches/logs
        dirs[:] = [
            d
            for d in dirs
            if d not in EXCLUDE_DIR_NAMES and d != "logs" and not d.startswith(".")
        ]
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            if Path(name).suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            src_file = root_path / name
            if _is_reparse_point(src_file):
                resolved = _resolve_portable_source(src_file)
                if resolved is None:
                    print(f"[package] skip unreadable: {src_file}")
                    continue
                shutil.copy2(resolved, target_dir / name)
            else:
                try:
                    shutil.copy2(src_file, target_dir / name)
                except OSError as exc:
                    print(f"[package] skip {src_file}: {exc}")


def _write_manifest(version: str) -> Path:
    model_root = PROJECT_ROOT / "models"
    os.environ["APP_MODEL_ROOT"] = str(model_root)
    os.environ["HF_HOME"] = str(model_root / "hf_cache")
    os.environ["HF_HUB_CACHE"] = str(model_root / "hf_cache" / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["TORCH_HOME"] = str(model_root / "torch")
    os.environ["OV_CACHE_DIR"] = str(model_root / "ov_cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["APP_AUTO_DOWNLOAD_MISSING_MODELS"] = "false"

    from engines.model_cache import configured_asr_model_ids, has_cached_model_file
    from scripts.build_model_pack import DEFAULT_DIARIZATION_MODELS, _write_manifest

    model_ids = list(configured_asr_model_ids()) + list(DEFAULT_DIARIZATION_MODELS)
    missing = [m for m in model_ids if not has_cached_model_file(m)]
    if missing:
        raise SystemExit(f"Model cache incomplete; cannot package offline release: {missing}")
    path = _write_manifest(model_root, model_ids, include_diarization=True)
    (model_root / "PACK_INFO.txt").write_text(
        f"Local Transcript App offline Model Pack\nversion={version}\ninclude_diarization=true\n",
        encoding="utf-8",
    )
    print(f"[package] Wrote {path}")
    return path


def _stage_source(stage_root: Path) -> None:
    for root, dirs, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)
        keep: list[str] = []
        for name in dirs:
            candidate = root_path / name
            if _should_skip_source(candidate):
                continue
            keep.append(name)
        dirs[:] = keep
        for name in files:
            path = root_path / name
            if _should_skip_source(path):
                continue
            rel = _rel(path)
            dest = stage_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


def _stage_models(stage_root: Path) -> None:
    src_models = PROJECT_ROOT / "models"
    dst_models = stage_root / "models"
    dst_models.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "PACK_INFO.txt"):
        src = src_models / name
        if src.is_file():
            shutil.copy2(src, dst_models / name)
    # Slim pack: HF hub models + torch only (skip ov_cache ~13 GiB).
    _materialize_hf_hub(src_models / "hf_cache" / "hub", dst_models / "hf_cache" / "hub")
    _copy_tree_files(src_models / "torch", dst_models / "torch")
    (dst_models / "ov_cache").mkdir(parents=True, exist_ok=True)
    (dst_models / "ov_cache" / "README.txt").write_text(
        "OpenVINO IR cache is not included in the slim offline pack.\n"
        "Export locally with: python scripts/build_model_pack.py\n"
        "Or run once with APP_FORCE_BACKEND=openvino to populate this folder.\n",
        encoding="utf-8",
    )
    (dst_models / "hf_cache").mkdir(parents=True, exist_ok=True)
    for rel in (
        "storage/audio",
        "storage/input",
        "storage/jobs",
        "storage/transcripts",
        "storage/logs",
    ):
        (stage_root / rel).mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_file(path: Path, chunk_bytes: int) -> list[Path]:
    size = path.stat().st_size
    if size <= chunk_bytes:
        return [path]
    parts: list[Path] = []
    with path.open("rb") as src:
        index = 1
        while True:
            data = src.read(chunk_bytes)
            if not data:
                break
            part = path.with_name(f"{path.name}.{index:03d}")
            part.write_bytes(data)
            parts.append(part)
            print(f"[package] Wrote part {part.name} ({len(data) / (1024**3):.2f} GiB)")
            index += 1
    path.unlink()
    (path.parent / "join_offline_zip.ps1").write_text(
        f"""$ErrorActionPreference = 'Stop'
$out = Join-Path $PSScriptRoot '{path.name}'
if (Test-Path $out) {{ Remove-Item $out -Force }}
Get-ChildItem $PSScriptRoot -Filter '{path.name}.*' |
  Where-Object {{ $_.Name -match '\\.\\d{{3}}$' }} |
  Sort-Object Name |
  ForEach-Object {{
    Write-Host "Appending $($_.Name)"
    $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
    $fs = [System.IO.File]::Open($out, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write)
    $fs.Write($bytes, 0, $bytes.Length)
    $fs.Close()
  }}
Write-Host "Created $out"
""",
        encoding="utf-8",
    )
    (path.parent / "join_offline_zip.sh").write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
OUT='{path.name}'
rm -f "$OUT"
cat $(ls {path.name}.* | grep -E '\\.[0-9]{{3}}$' | sort) > "$OUT"
echo "Created $OUT"
""",
        encoding="utf-8",
        newline="\n",
    )
    return parts


def _zip_stage(stage_root: Path, zip_path: Path, prefix: str) -> None:
    files = [p for p in stage_root.rglob("*") if p.is_file()]
    print(f"[package] Archiving {len(files)} files -> {zip_path.name}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in files:
            arcname = f"{prefix}/{path.relative_to(stage_root).as_posix()}"
            zf.write(path, arcname)


def main() -> int:
    parser = argparse.ArgumentParser(description="Package offline release zip + installers.")
    parser.add_argument("--version", default="1.2.7")
    parser.add_argument("--max-part-mib", type=int, default=1900)
    parser.add_argument("--keep-stage", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()

    version = args.version.lstrip("v")
    out_dir = PROJECT_ROOT / "release" / f"v{version}"
    stage_root = out_dir / "_stage" / f"LocalTranscriptApp-v{version}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    stage_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_manifest:
        _write_manifest(version)

    for name in ("install.bat", "install.sh"):
        src = PROJECT_ROOT / "installer" / name
        if not src.is_file():
            raise SystemExit(f"Missing installer: {src}")
        shutil.copy2(src, out_dir / name)
        print(f"[package] Copied {name}")

    print("[package] Staging source...")
    _stage_source(stage_root)
    print("[package] Materializing models (HF reparse -> real files)...")
    _stage_models(stage_root)

    # Ship installers inside the zip too.
    for name in ("install.bat", "install.sh"):
        shutil.copy2(PROJECT_ROOT / "installer" / name, stage_root / "installer" / name)

    zip_path = out_dir / f"LocalTranscriptApp-v{version}-offline.zip"
    _zip_stage(stage_root, zip_path, f"LocalTranscriptApp-v{version}")
    size_gb = zip_path.stat().st_size / (1024**3)
    print(f"[package] Zip size: {size_gb:.2f} GiB")

    chunk = max(64, args.max_part_mib) * 1024 * 1024
    artifacts = _split_file(zip_path, chunk)
    if len(artifacts) > 1:
        print(f"[package] Split into {len(artifacts)} part(s) for GitHub upload")

    if not args.keep_stage:
        shutil.rmtree(out_dir / "_stage", ignore_errors=True)

    sums = out_dir / "SHA256SUMS.txt"
    lines: list[str] = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            digest = _sha256_file(path)
            lines.append(f"{digest}  {path.name}")
            print(f"[package] sha256 {path.name} = {digest[:12]}...")
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[package] Wrote {sums}")
    print(f"[package] Done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
