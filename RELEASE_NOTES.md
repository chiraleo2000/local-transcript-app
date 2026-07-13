# Local Transcript App — release notes

**Current version: 1.2.6**

Multi-platform local transcription: **NVIDIA CUDA**, **OpenVINO** (Intel Core Ultra / Arc / NPU, AMD AI CPU, ARM), **DirectML** / **ROCm** for AMD GPUs. Minimum host **4 CPU threads / 8 GB RAM** (NVIDIA still needs ≥ 8 GB VRAM). Public proxy, workstation queue, beams **4**, warm GPU start.

See [README.md](README.md) for setup, configuration, and deployment.

---

## v1.2.6

### Summary

Public/LAN Docker access (nginx/IIS), workstation multi-user queue, modest ASR speed (beams **4**), warm GPU start, and a lowered host floor: **4 CPU threads / 8 GB RAM** for OpenVINO and CPU paths. NVIDIA CUDA still targets **≥ 8 GB VRAM**.

### Supported environments

| Target | How to run |
|--------|------------|
| NVIDIA GPU (≥ 8 GB VRAM) | `docker compose -f docker-compose.gpu.yml up -d --build` → `:7988` |
| Intel Core Ultra / Arc / iGPU / NPU | `docker-compose.openvino.yml` → `:7987` or `run.bat ov-gpu` / `OV_DEVICE=NPU` |
| AMD AI CPU / x86 CPU | OpenVINO compose or `APP_FORCE_BACKEND=openvino` `OV_DEVICE=CPU` |
| AMD GPU (Windows) | DirectML (`torch-directml`) or OpenVINO CPU |
| AMD GPU (Linux ROCm) | PyTorch HIP / `APP_FORCE_BACKEND=rocm` |
| ARM64 | OpenVINO compose / native OpenVINO |
| Intel GPU (Linux Docker) | OpenVINO compose + `/dev/dri` passthrough |

**Minimum host:** `MIN_CPU_THREADS=4`, `MIN_SYSTEM_RAM_MB=8192`. Set `APP_CPU_THREADS=0` (auto) or override (OpenVINO compose defaults to **4**).

### Public access + workstation queue

- Gradio binds `0.0.0.0` (GPU `:7988`, OpenVINO `:7987`)
- Optional: `GRADIO_ROOT_PATH`, auth, `GRADIO_MAX_FILE_SIZE`, `APP_PUBLIC_BASE_URL`
- Proxy samples: `deploy/nginx/`, `deploy/iis/` — see `deploy/README.md`
- `UI_MAX_CONCURRENT_JOBS=1` on 8 GB; `UI_GRADIO_TRANSCRIBE_CONCURRENCY=4` queues users
- Per-IP history (`UI_HISTORY_PER_CLIENT_IP`); cancel frees GPU cache for the next job

### ASR / diarization speed (GPU)

| Knob | v1.2.5 (cal15) | v1.2.6 |
|------|----------------|--------|
| `ASR_NUM_BEAMS` / `MAX` | 6 | **4** |
| Merge gap / max turn | 0.35 / 20 | unchanged |
| `DIARIZATION_KEEP_PRELOADED` | false | **true** |
| `ASR_CLEAR_VRAM_AFTER_JOB` | true | **false** |
| Warm job start | unload always | **keep weights** |

`models/ov_cache` is **OpenVINO-only**. NVIDIA GPU Docker does not need it.

```powershell
docker compose -f docker-compose.gpu.yml up -d --build      # :7988
docker compose -f docker-compose.openvino.yml up -d --build # :7987
```

---

## Earlier — v1.2.5

Enterprise GPU Docker acceptance, offline-only model policy, cal15 sample01 baseline (≈99.4% content / 100% speakers / ~177s), meeting309 profile, and slim offline pack assets on the GitHub release. Canonical knobs live in `backend/enterprise_config.py`.
