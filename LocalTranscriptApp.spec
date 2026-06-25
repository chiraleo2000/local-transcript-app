# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — native desktop bundle (Windows/Linux/macOS).
#
# Build:
#   pip install pyinstaller>=6.10
#   pyinstaller --noconfirm --clean LocalTranscriptApp.spec
#
# Output: dist/LocalTranscriptApp/LocalTranscriptApp.exe (+ _internal/)

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH)

block_cipher = None

# --- Data files shipped beside bytecode -----------------------------------
datas = [
    (str(ROOT / "config"), "config"),
    (str(ROOT / ".env.production"), "."),
    (str(ROOT / "sitecustomize.py"), "."),
    (str(ROOT / "app.py"), "."),
]

# Optional vendor native binaries (ffmpeg, platform-specific libs) — if a
# `vendor/` directory exists at the project root, ship its contents into a
# top-level `bin/` directory next to the executable so runtime subprocesses
# can find bundled tools without relying on system-installed packages.
vendor_root = ROOT / "vendor"
if vendor_root.exists() and vendor_root.is_dir():
    for p in vendor_root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(vendor_root)
            datas.append((str(p), str(Path("bin") / rel.parent)))

# --- Collect only core runtime packages by default to avoid brittle
# optional package collection failures on build hosts that don't have
# large ML packages pre-installed. Add additional packages here only when
# you've verified they exist in the build venv and need to be bundled.
_collect_packages = [
    "gradio",
    "fastapi",
    "starlette",
    "dotenv",
    "webview",
    "torch",
]

binaries = []
hiddenimports = [
    "app",
    "sitecustomize",
    "backend",
    "backend.pipeline",
    "backend.progress",
    "backend.storage",
    "backend.paths",
    "backend.services.asr_local",
    "backend.services.hardware_policy",
    "backend.services.media_pipeline",
    "engines",
    "engines.typhoon_asr",
    "engines.pathumma_asr",
    "engines.timestamps",
    "engines.diarization",
    "engines.preprocess",
    "engines.model_cache",
    "torchcodec",
    "torchcodec.decoders",
    "torchcodec.encoders",
]

for pkg in _collect_packages:
    try:
        tmp_ret = collect_all(pkg)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
    except Exception as exc:  # noqa: BLE001 — optional on minimal build hosts
        print(f"[spec] collect_all skipped for {pkg}: {exc}", file=sys.stderr)

hiddenimports += collect_submodules("backend")
hiddenimports += collect_submodules("engines")
hiddenimports = sorted(set(hiddenimports))

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LocalTranscriptApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LocalTranscriptApp",
)
