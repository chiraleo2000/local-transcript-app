# Local Transcript App — v1.1.1

Golden automation, adaptive GPU performance budgets, and long-audio CUDA diarization fixes.

---

## Deploy profiles

### NVIDIA GPU (CUDA) — recommended

```powershell
docker compose -f docker-compose.gpu.yml up -d --build
```

- UI: **http://localhost:7988**
- Adaptive beams/merge by audio length; CUDA diar + Typhoon ASR on 8 GB GPUs
- Golden automation: `python scripts/run_golden_automation.py --deploy`

### OpenVINO / CPU AI

```powershell
docker compose -f docker-compose.openvino.yml up -d --build
```

- UI: **http://localhost:7987**

---

## Golden automation

```powershell
# Accuracy (≥95%) on test-sample01.m4a — under 10 min wall time
python scripts/run_golden_automation.py --deploy --fixtures sample01

# Long-audio performance smoke on Recording 172.wav (~77 min, target ~1/4 realtime)
python scripts/run_golden_automation.py --deploy --fixtures recording172

# Both
python scripts/run_golden_automation.py --deploy
```

**Performance targets**

| Audio length | Wall-time target |
|--------------|------------------|
| Short (&lt;20 min) | ≤10 min |
| Medium (20 min–1 h) | ≤30 min |
| 1 h+ | min(30 min, duration ÷ 4) |

Short clips use turn-guided ASR (beams 6–8). Files ≥1 h switch to segmented CUDA diar + windowed long-form ASR for speed.

---

## Highlights

- `backend/asr_performance.py` — per-job adaptive tuning
- `tests/golden/` — deploy-aware harness with sample01 accuracy + recording172 perf fixtures
- Segmented diarization CUDA recovery for 1 h+ files
- Thai variant normalization in golden scoring
