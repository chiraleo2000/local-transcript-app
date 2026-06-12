# Release notes — v1.2.3

## Summary

GPU co-resident ASR + diarization when VRAM allows, split model lifecycles, and a single **Output** tab in the UI.

---

## GPU co-resident diarization

- `DIARIZATION_DEVICE=auto` (default in `docker-compose.gpu.yml`)
- `DIARIZATION_GPU_CO_RESIDENT=true` — never unload ASR to make room for diarization
- CUDA diarization when free VRAM ≥ `DIARIZATION_CUDA_MIN_FREE_MB` (default 3072) with ASR loaded
- Falls back to CPU inference when VRAM is low; ASR unchanged
- `DIARIZATION_KEEP_PRELOADED=true` — diarization weights retained across jobs and engine changes

## Split model lifecycles

- `backend/model_registry.py` — `unload_asr_models()`, `unload_diarization_model()`, `unload_all_models()`
- Engine change in UI swaps ASR only; diarization retained
- Cancel keeps models when `ASR_UNLOAD_ON_CANCEL=false`

## UI: single Output tab

- Replaces separate Typhoon/Pathumma output tabs
- `#output-transcript`, `#output-elapsed` — engine selector unchanged
- E2E selectors updated in `tests/e2e/transcription.spec.ts` and `real_audio.spec.ts`

## Image

- `local-transcript-app:1.2.3`, `APP_VERSION=1.2.3`

```powershell
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml up -d
```
