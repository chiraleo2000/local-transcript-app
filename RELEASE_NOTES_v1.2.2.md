# Release notes — v1.2.2

## Summary

Multi-accelerator release: same Windows `.exe` / Linux `install.sh` installers as v1.2.1, plus a dedicated **OpenVINO Docker** image for Intel Core Ultra (NPU/GPU), AMD AI CPU, and ARM64 hosts. NVIDIA CUDA and AMD ROCm remain first-class on bare metal and in `docker-compose.gpu.yml`.

---

## New features

### OpenVINO Docker stack

- `Dockerfile.openvino` — CPU PyTorch + OpenVINO 2026.1, no CUDA runtime
- `docker-compose.openvino.yml` — `APP_FORCE_BACKEND=openvino`, `OV_DEVICE=AUTO`
- `requirements-docker-openvino.txt` — dependency set for the OpenVINO image

```bash
docker compose -f docker-compose.openvino.yml up -d --build
```

### Hardware / ASR routing

- **NVIDIA CUDA** — unchanged; `docker-compose.gpu.yml` or auto-detect
- **AMD ROCm** — ASR now correctly uses the PyTorch CUDA/HIP path (`backend=rocm`)
- **Intel OpenVINO** — NPU/GPU/CPU via `optimum-intel` IR cache under `models/ov_cache`
- **Windows DirectML** — detected; ASR falls back to CPU PyTorch when DirectML is selected
- **ARM64** — prefers OpenVINO when installed

`engines/runtime_backend.py` centralises pipeline selection for Typhoon and Pathumma.

### Launcher / run scripts

- `launcher.py` and `./run.sh docker` pick `docker-compose.gpu.yml` → `docker-compose.openvino.yml` → `docker-compose.yml`
- Packaged `.exe` / `install.sh` installers are unchanged in layout; only the app version string bumps to 1.2.2

---

## Files added

- `Dockerfile.openvino`, `docker-compose.openvino.yml`, `requirements-docker-openvino.txt`
- `engines/runtime_backend.py`, `RELEASE_NOTES_v1.2.2.md`

## Files changed

- `backend/services/hardware_policy.py` — ARM arch detection
- `engines/typhoon_asr.py`, `engines/pathumma_asr.py` — ROCm/OpenVINO/CPU routing
- `launcher.py`, `run.sh`, `installer/install.sh`, `installer/LocalTranscriptApp.iss`

---

## v1.2.2 GPU Docker stability (8 GB class)

### 16-core CPU for enhancement and diarization preprocessing

- `docker-compose.gpu.yml`: `cpus: "16"`, `APP_CPU_THREADS=16`, OMP/MKL/OpenBLAS thread caps
- `backend/cpu_limits.py` — applied at app startup
- FFmpeg `-threads` in `engines/preprocess.py`

### CUDA stability (`cudaErrorUnknown`)

- `backend/vram_state.py` — `cuda_device_healthy()` probe and `recover_cuda()`
- Diarization double-guard: 8 GB class and unhealthy CUDA force CPU inference
- ASR stays loaded when `ASR_KEEP_PRELOADED=true` (no diarization `release_after_job` eviction)
- ASR reload skip when model resident and CUDA healthy
- CUDA load retry in Typhoon/Pathumma on `cudaErrorUnknown`

### Image tags

- `local-transcript-app:1.2.2`, `APP_VERSION=1.2.2`
