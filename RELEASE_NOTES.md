# Local Transcript App — release notes

**Current version: 1.2.7**

See [README.md](README.md) for setup. Docker stacks live under [`deploy/docker/`](deploy/docker/).

---

## v1.2.7

### Summary

Cleaner Docker deployment layout for **CUDA 12.4 / 12.6 / latest (13.3)** and **OpenVINO**, plus one-click `Deploy-Docker.bat`, WiFi-safe public access helpers, and faster GPU diarization settings that keep accuracy locks.

### Docker stacks (`deploy/docker/`)

| Stack | CUDA / backend | UI |
|-------|----------------|----|
| `latest` | CUDA **13.3** + PyTorch cu130 (recommended) | `:7988` |
| `cuda126` | CUDA **12.6** + cu126 | `:7988` |
| `cuda124` | CUDA **12.4** + cu124 (minimum) | `:7988` |
| `openvino` | CPU / Intel iGPU OpenVINO | `:7987` |

```bat
Deploy-Docker.bat gpu -Build
Deploy-Docker.bat gpu -CudaStack cuda126 -Build
Deploy-Docker.bat gpu -CudaStack cuda124 -Build
Deploy-Docker.bat openvino -Build
```

`.env`: `DEPLOY_BACKEND=auto|gpu|openvino`, `DEPLOY_CUDA_STACK=latest|cuda126|cuda124`

Root `docker-compose.*.yml` / `Dockerfile*` remain compatibility shims.

### Faster diarization (accuracy retained)

Shared policy: [`deploy/docker/gpu-app.env`](deploy/docker/gpu-app.env)

- `DIARIZATION_ACCURACY_MODE=true`, locked thresholds, no multi-sample
- Larger diar windows / smaller overlap; turn-guided ASR up to 28s
- Beams **4**, `ASR_FAST_MODE=true`

### Public / travel access

- [`deploy/SETUP.md`](deploy/SETUP.md) — Cloudflare Tunnel + nginx guide
- `Setup-TravelTunnel.ps1` / `Setup-PublicAccess.ps1` / `Open-PublicFirewall.ps1`
- Host driver requirement for NVIDIA stacks: **CUDA >= 12.4**

### Removed / cleaned

- Unused IIS deploy helpers
- Secrets and generated nginx certs gitignored

---

## v1.2.6

Multi-platform local transcription baseline: NVIDIA CUDA, OpenVINO, DirectML/ROCm, public proxy samples, workstation queue (`UI_MAX_CONCURRENT_JOBS=1`, concurrency 4), beams 4, warm GPU start, host floor 4 threads / 8 GB RAM.

```powershell
docker compose -f docker-compose.gpu.yml up -d --build      # :7988
docker compose -f docker-compose.openvino.yml up -d --build # :7987
```
