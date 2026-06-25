# Release notes — v1.2.4

## Summary

GPU-staged pipeline for 8 GB cards: CUDA speaker diarization (single pass) + turn-guided Typhoon ASR on CUDA. Golden automation on `test-sample01` reaches **≥95%** transcript accuracy.

---

## GPU staging (8 GB default)

- `DIARIZATION_DEVICE=cuda` with `ASR_UNLOAD_FOR_DIARIZATION=true`
- ASR staged off GPU → CUDA pyannote diarization → CUDA turn-guided Typhoon ASR
- `DIARIZATION_MULTI_SAMPLE=false` — single fast GPU pass (~6s embeddings vs ~90s CPU)
- `ASR_NUM_BEAMS=8`, turn-guided merge gap `0.35s`, chunk length `45s`

## Quality & cleanup

- Thai ASR variant fixes (`พูลวิลล่า`, `เช็ค`, `ลิสต์`, etc.) in `engines/text_cleanup.py`
- Golden scoring normalizes the same variants for fair comparison
- 8 GB safety profile respects explicit `DIARIZATION_DEVICE=cuda` staging

## Golden automation

```powershell
python scripts/run_golden_automation.py --deploy
```

Tests `tests/test-sample01.m4a` against `tests/test-sample01.txt` inside an exclusive GPU container (≥95% threshold).

## Docker GPU deploy

- Host port **7988** → container 7896
- Image: `local-transcript-app:latest`, `APP_VERSION=1.2.4`

```powershell
docker compose -f docker-compose.gpu.yml up -d --build
```

Open **http://localhost:7988**
