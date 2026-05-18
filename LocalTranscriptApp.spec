# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for LocalTranscriptApp (GUI installer release).

This bundles the launcher together with the full app (app.py + backend +
engines + torchcodec stub + config), all required dynamic imports, and a
windowed (no-console) executable suitable for end-user distribution.
"""

import os
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

block_cipher = None

# --- Bundled data: app code + assets + production env template ---------------
_datas = [
    ('app.py', '.'),
    ('sitecustomize.py', '.'),
    ('backend', 'backend'),
    ('engines', 'engines'),
    ('torchcodec', 'torchcodec'),
    ('config', 'config'),
    ('scripts', 'scripts'),
]
if os.path.exists('.env.production'):
    _datas.append(('.env.production', '.'))
if os.path.exists('.env.example'):
    _datas.append(('.env.example', '.'))

# Ship third-party package data files (Gradio assets, openvino plugin xml, etc.)
for _pkg in (
    'gradio',
    'gradio_client',
    'openvino',
    'pyannote',
    'pyannote.audio',
    'transformers',
    'huggingface_hub',
    'librosa',
    'soundfile',
    'noisereduce',
    'pedalboard',
):
    try:
        _datas += collect_data_files(_pkg, include_py_files=False)
    except Exception:  # pylint: disable=broad-except
        pass

_binaries = []
for _pkg in ('torch', 'openvino', 'onnxruntime', 'soundfile'):
    try:
        _binaries += collect_dynamic_libs(_pkg)
    except Exception:  # pylint: disable=broad-except
        pass

_hidden = []
for _pkg in (
    'gradio',
    'gradio.themes',
    'gradio_client',
    'torch',
    'transformers',
    'pyannote.audio',
    'openvino',
    'librosa',
    'soundfile',
    'noisereduce',
    'pedalboard',
    'webview',
    'psutil',
    'dotenv',
):
    try:
        _hidden += collect_submodules(_pkg)
    except Exception:  # pylint: disable=broad-except
        pass

_hidden += [
    'backend.pipeline',
    'backend.storage',
    'backend.services.asr_local',
    'backend.services.correction_local',
    'backend.services.hardware_policy',
    'backend.services.media_pipeline',
    'engines.diarization',
    'engines.hardware',
    'engines.model_cache',
    'engines.pathumma_asr',
    'engines.preprocess',
    'engines.timestamps',
    'engines.typhoon_asr',
    'torchcodec',
    'torchcodec.decoders',
    'torchcodec.encoders',
    'torchcodec.samplers',
    'torchcodec.transforms',
]

# torch_directml is optional; include if installed.
try:
    import torch_directml  # noqa: F401
    _hidden.append('torch_directml')
except ImportError:
    pass


a = Analysis(
    ['launcher.py'],
    pathex=['.'],
    binaries=_binaries,
    datas=_datas,
    hiddenimports=_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib.tests', 'numpy.tests'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='LocalTranscriptApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
