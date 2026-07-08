# Release notes — v1.2.5

## Summary

Enterprise GPU Docker acceptance, offline-only model policy, cal15 sample01 baseline tuning, and production validation tooling for short dialogue and long meeting fixtures.

---

## Enterprise GPU acceptance

- **Two fixtures:** `sample01` (~3.7 min, 4 speakers) and `meeting309` (~90 min, 11 speakers) — run via `scripts/run_docker_acceptance.py`
- **Tiered performance gates:** ≤10 min wall time for audio &lt;20 min; half realtime for longer audio
- **Canonical config:** `backend/enterprise_config.py` — shared VRAM 0.92 profile plus per-fixture overrides
- **sample01 cal15 baseline (locked):** 99.4% content, 100% speaker sequence, 4/4 speakers, ~177s, offline verified
- **meeting309 m310 profile (locked):** 11-speaker exact-count diarization; validation pending full gate pass

## Offline-only runtime

- Models load strictly from local `./models/` — no Hugging Face Hub downloads at deploy or runtime
- `APP_REQUIRE_DIARIZATION_MODELS=true` in GPU compose; diarization models must be cached before acceptance runs
- `scripts/_bootstrap.py` sets offline cache defaults; `scripts/ensure_model_cache.py` preflight in Docker validation
- Offline loaders in diarization sweep/probe scripts (`scripts/sweep_309_vbx.py`, `scripts/probe_309_centroids.py`)

## ASR and diarization improvements

- Turn-guided ASR with diarization timestamps (Whisper word timestamps disabled for stability on 8 GB)
- CUDA VRAM staging, OOM recovery, and job-level memory guards (`backend/vram_state.py`, `backend/asr_performance.py`)
- Diarization: VBx tuning, short-audio mega-turn refine (CPU sub-pass), centroid merge controls, stability logging
- Overlap-aware golden scoring (`tests/golden/accuracy.py`, `tests/golden/meeting_eval.py`)

## Validation and quality tooling

- `scripts/run_docker_acceptance.py` — Docker build, env verify, fixture runs, offline log scan
- `scripts/run_enterprise_validation.py` — queue-based enterprise validation with cache preflight
- `scripts/run_sonar_scan.py` + `sonar-project.properties` — SonarQube quality gate
- New unit tests: ASR quality, CUDA recovery, model cache offline, meeting eval, whisper decode

## Policy

- `MIN_NVIDIA_VRAM_MB=8192` — 8 GB VRAM minimum for NVIDIA GPU profile
- `ASR_CUDA_MEMORY_FRACTION=0.92` with batch=1, beams=6, single concurrent job

## Deploy

```powershell
# NVIDIA GPU enterprise profile (port 7988)
docker compose -f docker-compose.gpu.yml up -d --build

# Run acceptance (requires cached models under ./models/)
python scripts/run_docker_acceptance.py --tag release

# OpenVINO (v1.2.4 profile, port 7987)
docker compose -f docker-compose.profiles.yml --profile openvino up -d --build
```
