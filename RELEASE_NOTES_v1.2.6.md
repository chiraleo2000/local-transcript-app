# Release notes — v1.2.6

## Summary

Public/LAN Docker access (nginx/IIS), workstation multi-user queue, modest ASR speed (beams **4**), warm GPU start, and a lowered host floor: **4 CPU threads / 8 GB RAM** for OpenVINO and CPU paths. NVIDIA CUDA still targets **≥ 8 GB VRAM**.

---

## Supported environments

| Target | How to run |
|--------|------------|
| NVIDIA GPU (≥ 8 GB VRAM) | `docker compose -f docker-compose.gpu.yml up -d --build` → `:7988` |
| Intel Core Ultra / Arc / iGPU / NPU | `docker compose -f docker-compose.openvino.yml up -d --build` → `:7987` or native `run.bat ov-gpu` / `OV_DEVICE=NPU` |
| AMD AI CPU / x86 CPU | OpenVINO compose or `APP_FORCE_BACKEND=openvino` `OV_DEVICE=CPU` |
| AMD GPU (Windows) | DirectML (`torch-directml`) or OpenVINO CPU |
| AMD GPU (Linux ROCm) | PyTorch HIP / `APP_FORCE_BACKEND=rocm` |
| ARM64 | OpenVINO compose / native OpenVINO |
| Intel GPU (Linux Docker) | OpenVINO compose + `/dev/dri` passthrough |

**Minimum host:** `MIN_CPU_THREADS=4`, `MIN_SYSTEM_RAM_MB=8192`. Set `APP_CPU_THREADS=0` (auto) or override (OpenVINO compose defaults to **4**).

---

## Why acceptance looked slower (~345s vs ~177s)

Docker acceptance forces full VRAM clears. Production warm-starts when `ASR_KEEP_PRELOADED=true` and `ASR_CLEAR_VRAM_AFTER_JOB=false`. **Beams 7/8 are slower**, not faster.

---

## Public access (Docker + reverse proxy)

- Gradio binds `0.0.0.0` (GPU `:7988`, OpenVINO `:7987`)
- Optional: `GRADIO_ROOT_PATH`, auth, `GRADIO_MAX_FILE_SIZE`, `APP_PUBLIC_BASE_URL`
- Configs: `deploy/nginx/`, `deploy/iis/` — see `deploy/README.md`

---

## Workstation multi-user (8 GB NVIDIA)

| Knob | Value | Meaning |
|------|-------|---------|
| `UI_MAX_CONCURRENT_JOBS` | **1** | One Whisper+diar GPU job |
| `UI_GRADIO_TRANSCRIBE_CONCURRENCY` | **4** | Up to 4 users can submit; extras queue |
| `UI_HISTORY_PER_CLIENT_IP` | **true** | History filtered by client IP |
| `UI_CANCEL_FREES_GPU_FOR_QUEUE` | **true** | Cancel clears CUDA cache for next job |

---

## ASR / diarization speed (GPU)

| Knob | v1.2.5 (cal15) | v1.2.6 |
|------|----------------|--------|
| `ASR_NUM_BEAMS` / `MAX` | 6 | **4** |
| `ASR_TURN_GUIDED_MERGE_GAP_S` | 0.35 | 0.35 |
| `ASR_TURN_GUIDED_MAX_TURN_S` | 20 | 20 |
| `DIARIZATION_KEEP_PRELOADED` | false | **true** |
| `ASR_CLEAR_VRAM_AFTER_JOB` | true | **false** |
| `ASR_CLEAR_VRAM_ON_MEDIA_CHANGE` | true | **false** |
| `ASR_FAST_MODE` | false | **true** |
| Warm job start | unload always | **keep weights** |

## OpenVINO note

`models/ov_cache` is **OpenVINO-only**. NVIDIA GPU Docker does not need it. Slim offline packs ship HF models only.

## Deploy

```powershell
# NVIDIA
docker compose -f docker-compose.gpu.yml up -d --build
# http://localhost:7988

# Intel / AMD AI / ARM (OpenVINO)
docker compose -f docker-compose.openvino.yml up -d --build
# http://localhost:7987
```
