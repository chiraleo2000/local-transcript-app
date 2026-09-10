# Local Transcript App — release notes

**Current version: 2.0.2**

See [README.md](README.md) for setup. Docker stacks live under [`deploy/docker/`](deploy/docker/).

---

## v2.0.2

### Summary

Separate the Docker **app image** from durable **storage** so rebuilds do not wipe users or transcripts.

### Deploy

- Removed full-repo bind mount `.:/app` from GPU compose stacks (code comes from the image)
- Keep host binds: `storage/` → `/app/storage`, `models/` → `/app/models`, `config/` → `/app/config`
- `.dockerignore` excludes `storage/` and `models/` from the build context
- Dockerfile `VOLUME` markers + deploy script ensures host dirs exist before `compose up`

---

## v2.0.1

### Summary

Fix Gradio multi-file upload error: `stat: path should be string … not list` when selecting files.

### Fix

- Normalize Gradio list / FileData uploads before preview (`backend/media_paths.py`)
- `_on_media_upload` and UI limits reject non-path values instead of calling `os.stat` on a list

---

## v2.0.0

### Summary

Major release: per-user SQLite job history, multi-file fire-and-forget queue in the UI, and hardware-aware concurrency defaults (safe on Tesla P4 / 8 GB; scales on modern high-VRAM GPUs).

### Breaking / operator notes

- UI **Queue for processing** enqueues jobs and returns immediately — you do not need to keep the browser open. Retrieve outputs under **Previous transcripts** (or `GET /api/jobs` / transcript download).
- Job list/history prefers **`storage/jobs.db`**. Existing `storage/jobs/*.json` manifests are imported on first 2.x startup (no re-transcribe). JSON dual-write continues for one major cycle.
- Transcript text remains in `storage/transcripts/*.txt` (not stored as SQLite BLOBs).
- Queue caps: if `UI_MAX_CONCURRENT_JOBS` / `API_MAX_QUEUED_JOBS` / `UI_MAX_BATCH_FILES` are **unset**, startup applies tier defaults. Explicit env (including Docker) always wins.

### Hardware tiers

| Tier | Hardware | GPU slots | Queue / batch (when env unset) |
| --- | --- | --- | --- |
| A | Tesla P4 / Pascal / ~8 GB | 1 | 4 / 3 |
| B | Ampere+ under parallel VRAM threshold | 1 | 4 / 3 |
| C | VRAM ≥ `ASR_PARALLEL_MIN_VRAM_MB` (default 12288) | 2 | 8 / 5 |

Docker GPU / Tesla P4 overlays keep **1** GPU slot explicitly.

### Features

- `backend/jobs_db.py` — SQLite index with `schema_version`
- `backend/job_enqueue.py` — shared UI/API enqueue + resume interrupted jobs on startup
- `backend/queue_policy.py` — tier detection and auto env fill
- Gradio multi-file upload (`UI_MAX_BATCH_FILES`)

### Migration from 1.2.x

1. Deploy 2.0.0 with the same `storage/` volume.
2. On first start, JSON job manifests are upserted into `jobs.db`.
3. Sign in and open **Previous transcripts** → Refresh list.

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
