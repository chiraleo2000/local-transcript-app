# Release notes — v1.2.4

## Summary

Offline-first OpenVINO runtime, fast ASR engine switching, Model Pack builder with manifest verification, and profile-based Docker deploy for NVIDIA and OpenVINO backends.

---

## Offline-only runtime

- Models load strictly from local `./models/` — no Hugging Face downloads at deploy or runtime
- `scripts/ensure_model_cache.py` verifies cache via `models/manifest.json` when present
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `APP_AUTO_DOWNLOAD_MISSING_MODELS=false` in compose defaults
- Diarization skipped gracefully when models are not cached (ASR continues)

## OpenVINO fixes and hardware

- Whisper `decoder_input_ids` compatibility shim for `transformers` 4.57+ (`engines/openvino_compat.py`)
- OpenVINO prefers Intel GPU before NPU when both are available (`hardware_policy.py`)
- Native Windows modes: `run.bat ov`, `ov-gpu`, `ov-npu` for Arc GPU / NPU acceleration
- Docker OpenVINO image includes Intel GPU runtime libs (CPU-only inside Windows Docker Desktop unless devices are passed through)

## Fast ASR switching

- `ASR_KEEP_PRELOADED=true` keeps Typhoon + Pathumma resident in memory
- OpenVINO startup preloads both engines when preloading is enabled
- Engine switch no longer unloads the other model

## Model Pack and release tooling

- `scripts/build_model_pack.py` — maintainer-only builder (HF downloads, OV IR export, `manifest.json`, optional archive)
- `scripts/stage_model_pack.py` — stages Model Pack into `dist/` for Full installer bundles
- `scripts/build_desktop.ps1` / `scripts/build_desktop.sh` — desktop build helpers
- `docker-compose.profiles.yml` — `--profile openvino` (port 7987) and `--profile nvidia` (port 7988)

## Policy

- `MIN_NVIDIA_VRAM_MB=8192` — 8 GB VRAM minimum for NVIDIA GPU profile

## Deploy

```powershell
# OpenVINO (offline, local models mount)
docker compose -f docker-compose.profiles.yml --profile openvino up -d --build
# http://localhost:7987

# NVIDIA GPU (offline, local models mount)
docker compose -f docker-compose.profiles.yml --profile nvidia up -d --build
# http://localhost:7988

# Native Windows OpenVINO (Intel Arc / NPU)
run.bat ov-gpu
```
