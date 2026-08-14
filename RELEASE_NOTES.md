# Local Transcript App — release notes

**Current version: 1.2.13**

See [README.md](README.md) for setup. Docker stacks live under [`deploy/docker/`](deploy/docker/).

---

## v1.2.13

### Summary

Tesla P4 / Pascal path: keep RTX 4060 accuracy knobs on Ampere+, but auto-switch P4 to CUDA 12.4 + FP32 + 2-beam decode so jobs do not take many times longer.

### GPU / deploy

- Detect Tesla P4 (compute 6.1) and apply `deploy/docker/gpu-p4.env` knobs at runtime
- `Deploy-Docker.ps1` selects the `cuda124` stack (CUDA 13 dropped Pascal kernels)
- RTX 4060 is unchanged (`ASR_GPU_PROFILE=auto`)

---

## v1.2.12

### Summary

Extend idle login timeout to 60 minutes and name the Docker Compose project `local-transcript-app` (replacing the `latest_default` network label).

### Session / deploy

- `APP_SESSION_TTL_S=3600` in production env and code defaults
- Compose `name: local-transcript-app` on all stacks; deploy script passes `-p local-transcript-app`

---

## v1.2.11

### Summary

SonarQube clean-up (complexity / security / duplication), friendlier Gradio UI (guided steps + clearer download/history), and re-verified unit gates before redeploy.

### SonarQube / quality

- Reduced cognitive complexity in session, recover, job API, ASR switch, timestamps helpers
- Fixed world-writable `/tmp` staging (private `lta_job_audio` dir with `0700`)
- Deduplicated Thai ASR variant literals + safer speaker-prefix regex
- E2E stopwatch waits use `expect.poll` instead of fixed timeouts

### UI

- Clear 1 → 2 → 3 upload / transcribe / download flow
- Tip banner for Previous transcripts after re-login
- Stronger primary Download actions and clearer history copy

---

## v1.2.10

### Summary

Harden the accuracy/performance improve loop: unify cal15 gates across golden/enterprise/docs, align production turn knobs to the verified sample01 lock, and add a CPU-only transcript scorer for fast regression checks.

### Accuracy / performance

- Docker sample01 acceptance re-verified: **99.3% content, 100% speaker, 67.5% ts, 66.8% strict, 9 mismatched, ~299s / 600s budget**
- Production `gpu-app.env` turn pad/merge/max-turn aligned to cal15 fixture overlay
- Golden gates matched to enterprise cal15 ceilings (no more impossible 99/98/98 chase)
- Acceptance env check uses VRAM **0.75** (was stale 0.92)
- `scripts/score_transcript.py` for CPU-only re-score without GPU
- Fixed `recording47` golden automation mapping; Docker stop/start uses container name across compose projects

---

## v1.2.9

### Summary

Fix **Download .txt** after long transcriptions: idle login no longer kills the session mid-job or right after completion, and re-login restores the finished transcript instead of an empty Output panel.

### Session / download

- Keepalive + tab-id scripts run via Gradio `head=` (previously injected with `gr.HTML` and never executed)
- Idle timeout skips forced logout while a job is in flight, and for a short grace window after completion so Download is not 401’d
- Completed jobs are recovered into Output / Download after refresh or re-login (`last_completed_job_id` + tab history)
- Durable Gradio-auth download route: `/ui/download/{job_id}` (also listed under Job Info)
- `allowed_paths` includes transcript/job storage so file serve stays reliable

### Notes

- Transcripts were always saved under `storage/transcripts` / **Previous transcripts**; the UI session was what looked “gone”

---

## v1.2.8

### Summary

Fix **Load into editor** for queued/running jobs so the UI streams live status (same as an online transcription) instead of blank “Done” output. Harden session recover after refresh / brief network drops, and tighten turn-guided ASR pads for content + timestamp quality (beams=5 retained).

### Job status recover

- Loading an in-flight Previous transcript polls the durable manifest / live API progress until completion, then shows the transcript
- Page recover re-attaches by `tab_id` / active job id and loads results when the worker finishes
- Manifest writes keep `status=running` sticky (avoids throttled sync dropping the running flag)

### Accuracy

- Production turn pad / boundary / merge gaps aligned closer to the cal15 sample01 lock
- `ASR_NUM_BEAMS=5` unchanged

### Auth (from 1.2.7+)

- Public `/register`, 15‑minute idle login timeout, Log out control, headless job API

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
