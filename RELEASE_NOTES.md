# Local Transcript App — v1.1.2

Four-fixture golden automation with verified GPU performance on RTX 4060 8GB.

---

## Verified results (Docker GPU deploy)

| Fixture | Audio | Accuracy / perf | Elapsed | Target |
|---------|-------|-----------------|---------|--------|
| `sample01` | test-sample01.m4a (3.6 min) | **98.0%** | 6.0 min | ≤10 min |
| `recording172` | Recording 172.wav (77 min) | GPU smoke ✓ | 13.0 min | ≤19.3 min (¼ RT) |
| `recording19` | Recording 19.wav (95 min) | GPU smoke ✓ | 16.5 min | ≤23.8 min (¼ RT) |
| `sample47` | 47.m4a (88 min) | GPU smoke ✓ | 21.7 min | ≤21.9 min (¼ RT) |

---

## Golden automation

```powershell
docker compose -f docker-compose.gpu.yml up -d --build
python scripts/run_golden_automation.py --deploy          # all fixtures
python scripts/run_golden_automation.py --deploy --skip-long   # sample01 only
```

Long files (≥1 h) use segmented CUDA diarization + 40-min ASR windows (`DIARIZATION_MAX_ASR_WINDOW_S=2400`, beams=1).

---

## Deploy

- **GPU:** `docker compose -f docker-compose.gpu.yml up -d --build` → http://localhost:7988
- **OpenVINO:** `docker compose -f docker-compose.openvino.yml up -d --build` → http://localhost:7987
