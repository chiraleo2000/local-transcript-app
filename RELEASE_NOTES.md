# Local Transcript App — v1.2.4

Offline-first OpenVINO + NVIDIA deploy with Model Pack verification, fast ASR switching, and 8 GB VRAM minimum for GPU profile.

See [RELEASE_NOTES_v1.2.4.md](RELEASE_NOTES_v1.2.4.md) for full details.

---

## Quick deploy

```powershell
# OpenVINO (port 7987)
docker compose -f docker-compose.profiles.yml --profile openvino up -d --build

# NVIDIA GPU (port 7988)
docker compose -f docker-compose.profiles.yml --profile nvidia up -d --build

# Native Windows OpenVINO (Intel Arc / NPU)
run.bat ov-gpu
```
