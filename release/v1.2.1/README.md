# Local Transcript App - v1.2.1 release assets

## Windows installer
Download all `LocalTranscriptAppSetup*` files into the same folder, then run:

```text
LocalTranscriptAppSetup.exe
```

This is a real Windows GUI installer built with Inno Setup. The extra `.bin` files, if present, are installer payload slices created because GitHub release assets have a per-file size limit. They are not archives the user extracts manually; keep them beside the setup exe and run the exe.

The installer includes:
- Native Windows GUI runtime (`LocalTranscriptApp.exe` + PyInstaller `_internal` dependencies)
- Production offline config (`.env`, no `HF_TOKEN`)
- Pre-cached Hugging Face model cache from `models/hf_cache` for no-token runtime loading
- README / run instructions

It intentionally excludes duplicate `models/_archive` data and generated `models/ov_cache` files.

## Linux installer
From an extracted source archive:

```bash
chmod +x installer/install.sh
./installer/install.sh
```

Optional AMD ROCm setup:

```bash
./installer/install.sh --with-rocm
```

## Hardware backends
Auto-detected priority:

```text
CUDA (>=6 GB VRAM) -> ROCm -> OpenVINO NPU/GPU -> DirectML -> OpenVINO CPU -> CPU (>=16 GB RAM recommended)
```

Override with:

```text
APP_FORCE_BACKEND=cuda|rocm|openvino|directml|cpu
```
