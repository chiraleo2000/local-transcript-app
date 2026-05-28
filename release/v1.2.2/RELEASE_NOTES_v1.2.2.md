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
