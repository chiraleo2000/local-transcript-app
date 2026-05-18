# Release notes — v1.2.0

## Highlights

This release replaces the Docker-only deployment model with a **native GUI installer** for Windows, Linux, and macOS. The app now runs on a wider hardware matrix (NVIDIA CUDA, AMD ROCm, Intel/AMD via DirectML, Intel NPU/GPU via OpenVINO, or pure CPU) and no longer requires the end user to hold a Hugging Face token.

## What's new

### No-token runtime
- End-user runtime is fully offline: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `APP_AUTO_DOWNLOAD_MISSING_MODELS=false`.
- Gated models (Typhoon Whisper, `pyannote/speaker-diarization-community-1`) are pre-cached by the maintainer at build time and bundled into the installer payload.
- New `.env.production` carries the no-token configuration; the existing dev `.env` is unaffected.
- `engines/model_cache.py` now logs a clear remediation message when an offline cache miss occurs.
- `scripts/bootstrap_models.py` rewritten as a build-time-only prefetch step (forces online mode, warns when `HF_TOKEN` is missing).

### Native GUI launcher
- `launcher.py` no longer attempts Docker when launched from a frozen PyInstaller exe or with `APP_FORCE_DIRECT=1`.
- pywebview window opens directly against the local Gradio server on `127.0.0.1:7896`.

### Five-backend hardware policy
`backend/services/hardware_policy.py` now detects and selects from the following backends in priority order:
1. NVIDIA CUDA (≥ 6 GB VRAM)
2. AMD ROCm (via `torch.version.hip`)
3. Intel OpenVINO NPU / GPU
4. Windows DirectML (`torch_directml`)
5. Intel OpenVINO CPU
6. Pure CPU (with 16 GB system RAM minimum check via `psutil`)

`APP_FORCE_BACKEND={cuda|rocm|openvino|directml|cpu}` lets you override the selection. The hardware summary now reports system RAM, ROCm, and DirectML status alongside CUDA and OpenVINO.

### Packaging
- `LocalTranscriptApp.spec` rewritten to bundle `app.py`, `backend/`, `engines/`, `torchcodec/`, `config/`, `scripts/`, and `.env.production`, plus `collect_data_files` / `collect_dynamic_libs` / `collect_submodules` for gradio, torch, openvino, pyannote, transformers, librosa, soundfile, noisereduce, pedalboard, webview, and psutil.
- `installer/LocalTranscriptApp.iss` (Inno Setup) now ships the prebuilt `LocalTranscriptApp.exe` plus the pre-cached `models/` payload. No `pip install` at install time. Per-user install (`PrivilegesRequired=lowest`).
- New `installer/install.sh` for Linux: copies the source tree to `~/.local/share/local-transcript-app/`, creates a venv, installs deps, registers a `.desktop` entry and a `local-transcript-app` CLI shim. Optional `--with-rocm` flag installs the ROCm PyTorch wheel.

### Dependencies
- Added `psutil>=5.9` (RAM detection) and `pywebview>=5.0` (native window).
- Documented optional installs: `torch-directml` (Windows), ROCm PyTorch wheel (Linux), `pyinstaller>=6.10` (maintainer-only).

## Upgrade notes

- The dev workflow still uses `.env` with your `HF_TOKEN`; nothing changes for contributors.
- To produce a release installer:
  1. `HF_TOKEN=hf_... python scripts/bootstrap_models.py`
  2. `pip install pyinstaller && pyinstaller LocalTranscriptApp.spec`
  3. Compile `installer/LocalTranscriptApp.iss` with Inno Setup 6 to produce `LocalTranscriptAppSetup.exe`.
- Linux end users: `chmod +x install.sh && ./install.sh`.

## Known limitations
- DirectML and ROCm code paths share the existing CUDA execution path (HIP is binary-compatible). Dedicated `torch_directml.device()` loaders for Typhoon and Pathumma engines are tracked for v1.3.
- macOS `.app` bundle workflow is not yet automated.
