# Local Transcript App — v1.0.0

First stable release: local Thai/English transcription with speaker diarization, two Docker profiles, and golden-file automation.

---

## Deploy profiles

### NVIDIA GPU (CUDA) — recommended for speed

```powershell
docker compose -f docker-compose.gpu.yml up -d --build
```

- UI: **http://localhost:7988**
- CUDA diarization staging + turn-guided Typhoon ASR on 8 GB GPUs
- Golden test: `python scripts/run_golden_automation.py --deploy` (≥95% on `test-sample01`)

### OpenVINO / CPU AI — Intel, AMD, ARM, no NVIDIA GPU

```powershell
docker compose -f docker-compose.openvino.yml up -d --build
```

- UI: **http://localhost:7987**
- CPU pyannote diarization + OpenVINO Pathumma/Typhoon ASR
- Same turn-guided transcript quality settings; no CUDA required

### Auto-detect (Windows)

```bat
run.bat docker
```

Picks GPU compose when NVIDIA is available in Docker, otherwise OpenVINO.

---

## Highlights

- Typhoon Whisper + Pathumma Whisper (local, no cloud)
- `pyannote/speaker-diarization-community-1` speaker diarization
- Turn-guided ASR aligned to diarization turns
- Thai ASR variant cleanup in transcript output
- Gradio UI + optional native desktop window (`run.bat gui`)
