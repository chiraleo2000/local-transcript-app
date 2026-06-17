# Release notes — v1.2.3

## Summary

GPU co-resident ASR + diarization when VRAM allows, split model lifecycles, a single **Output** tab, and improved multi-SR speaker diarization with faster 9-pass sampling.

---

## GPU co-resident diarization

- `DIARIZATION_DEVICE=auto` (default in `docker-compose.gpu.yml`)
- `DIARIZATION_GPU_CO_RESIDENT=true` — ASR stays loaded on GPU during diarization when configured
- CUDA diarization when free VRAM allows; falls back to CPU when tight
- `DIARIZATION_KEEP_PRELOADED=true` — diarization weights retained across jobs

## Multi-SR diarization (9-pass budget)

- `DIARIZATION_MULTI_SAMPLE_SR=16000,22050,44100` — compare three preprocess rates
- `DIARIZATION_MULTI_SAMPLE_MAX_TOTAL=9` — 3 rates × 3 curated hyperparameter configs
- Early stop when score ≥ 0.85; ASR models no longer unloaded on 8 GB GPUs when `ASR_UNLOAD_FOR_DIARIZATION=false`

## Transcript completeness

- Turn-centric speaker assignment uses full audio duration (not diarization end only)
- Gap-fill for speech between diarization turns; Thai character-level text slicing
- Fixes dropped transcript segments and missing speaker turns

## Docker GPU deploy

- Host port **7988** → container 7896
- Image: `local-transcript-app:1.2.3`, `APP_VERSION=1.2.3`

```powershell
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml up -d
```

Open **http://localhost:7988**

## Split model lifecycles

- `backend/model_registry.py` — `unload_asr_models()`, `unload_diarization_model()`, `unload_all_models()`
- Engine change in UI swaps ASR only; diarization retained
- Cancel keeps models when `ASR_UNLOAD_ON_CANCEL=false`

## UI: single Output tab

- Replaces separate Typhoon/Pathumma output tabs
- `#output-transcript`, `#output-elapsed` — engine selector unchanged
