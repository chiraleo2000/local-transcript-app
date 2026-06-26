# Local Transcript App — v1.1.6

GPU-first pipeline for 8 GB VRAM: CUDA diarization + Typhoon ASR, turn-guided accuracy, fast cached Docker builds.

---

## Verified results (Docker GPU, RTX 4060 8 GB)

| Fixture | Audio | Content | Speaker / timestamp | Elapsed | Target |
|---------|-------|---------|---------------------|---------|--------|
| `sample01` | test-sample01.m4a (3.6 min) | **96.2%** | **100%** | 2.4 min | ≤10 min |
| `recording172` | Recording 172.wav (77 min) | GPU smoke ✓ | CUDA diar + ASR | 7.7 min | ≤19 min (¼ RT) |
| `recording19` | Recording 19.wav (95 min) | GPU smoke ✓ | CUDA diar + ASR | 10.7 min | ≤24 min (¼ RT) |
| `sample47` | 47.m4a (88 min) | GPU smoke ✓ | CUDA diar + ASR | 11.5 min | ≤22 min (¼ RT) |

Pass criteria for `sample01`: content ≥90%, speaker + timestamp ≥98%.

---

## Golden automation

```powershell
# Fast path: reuse cached image + bind-mounted code (no rebuild)
python scripts/run_golden_automation.py --deploy

# Force image rebuild only when Dockerfile/requirements change
python scripts/run_golden_automation.py --deploy --rebuild

# Accuracy only (skip long perf fixtures)
python scripts/run_golden_automation.py --deploy --fixtures sample01
```

---

## Deploy

```powershell
$env:DOCKER_BUILDKIT = "1"
docker compose -f docker-compose.gpu.yml build    # cached layers
docker compose -f docker-compose.gpu.yml up -d  # http://localhost:7988
```

**GPU profile highlights:** `DIARIZATION_DEVICE=cuda`, ASR staging off GPU before diar, turn-guided ASR with boundary pad/rebalance, enhancement off by default.
