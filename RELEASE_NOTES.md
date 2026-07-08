# Local Transcript App — v1.2.5

Enterprise GPU Docker acceptance with offline-only model policy, cal15 sample01 baseline, and validation tooling for short dialogue and long meeting fixtures.

**Enterprise GPU acceptance** (production sign-off): Docker-only validation of `sample01` + `meeting309` via `scripts/run_docker_acceptance.py`. Tiered perf: ≤10 min for audio &lt;20 min, half realtime for longer. VRAM locked at 0.92 with batch=1.

See [RELEASE_NOTES_v1.2.5.md](RELEASE_NOTES_v1.2.5.md) for full details.

---

## Quick deploy

```powershell
# NVIDIA GPU enterprise (port 7988)
docker compose -f docker-compose.gpu.yml up -d --build

# OpenVINO (port 7987)
docker compose -f docker-compose.profiles.yml --profile openvino up -d --build

# Native Windows OpenVINO (Intel Arc / NPU)
run.bat ov-gpu
```
