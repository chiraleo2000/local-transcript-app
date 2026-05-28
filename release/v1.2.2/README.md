# Local Transcript App - v1.2.2 release assets

## Windows installer
Download `LocalTranscriptApp-1.2.2.exe` and run it. This is the PyInstaller desktop bundle (native GUI window via pywebview).

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

## Docker — OpenVINO / CPU AI (Intel Core Ultra, AMD AI CPU, ARM)

```bash
docker compose -f docker-compose.openvino.yml up -d --build
```

## Docker — NVIDIA CUDA

```bash
docker compose -f docker-compose.gpu.yml up -d --build
```

## Hardware backends
Auto-detected priority:

```text
CUDA (>=6 GB VRAM) -> ROCm -> OpenVINO NPU/GPU -> DirectML -> OpenVINO CPU -> CPU
```

Override with:

```text
APP_FORCE_BACKEND=cuda|rocm|openvino|directml|cpu
```
