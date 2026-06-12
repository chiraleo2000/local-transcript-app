# How To Run Local Transcript App

This file is the practical run guide for the root app in `local-transcript-app`.

## 1. Open The App Folder

On Windows PowerShell:

```powershell
cd C:\Users\chira\Documents\visualstudiocode\Work\local-transcript-app
```

The runnable app files are in this main folder:

```text
app.py
backend/
engines/
scripts/bootstrap_models.py
requirements.txt
setup.bat
run.bat
```

The old `speaker-aware-ai/` and `test-transcript-service/` folders are reference/legacy folders. You do not need to enter them to run the app.

## 2. Install Dependencies

Windows:

```bat
setup.bat
```

Linux/macOS:

```bash
./setup.sh
```

## 3. Download And Prepare Models

Windows:

```bat
venv\Scripts\activate
python scripts\bootstrap_models.py
```

Linux/macOS:

```bash
source venv/bin/activate
python scripts/bootstrap_models.py
```

This step checks hardware, selects the backend, downloads/exports models, and writes config under:

```text
config/app_config.json
models/
```

## 4. Run The App

Windows:

```bat
run.bat
```

Linux/macOS:

```bash
./run.sh
```

Open this URL in your browser:

```text
Direct / GUI:  http://localhost:7896
Docker test:   http://localhost:7987
```

## 5. Hardware Rules

- NVIDIA GPU with CUDA and at least 6 GB VRAM: CUDA path.
- NVIDIA GPU below 6 GB VRAM: fallback to OpenVINO/CPU.
- Intel GPU/NPU: OpenVINO path when available.
- AMD GPU: CPU/OpenVINO fallback in v1.
- CPU only: CPU/OpenVINO fallback.

On 6-8 GB-class NVIDIA GPUs the app uses strict memory-safe mode:

- Default engine: `Pathumma Whisper` for the fastest 8 GB path.
- Quality-first default: `ASR_QUALITY_PROFILE=high` (300s ASR windows, full diarization grid, enhance-on for diarization). Use `balanced` only for emergency low-VRAM. See README Quality profiles table.
- One GPU ASR model at a time; both engines can be selected, but they run sequentially.
- Up to **4 pipeline job slots** (`UI_MAX_CONCURRENT_JOBS=4` in GPU compose). Extra tabs queue; only **one CUDA ASR inference** runs at a time (`ASR_MAX_CONCURRENT_INFERENCE=1`). Restart the container after changing job slots.
- **Cancel & Reset** joins the worker and unloads models (`ASR_UNLOAD_ON_CANCEL=true`) so the next job starts with clean VRAM.
- Forced parallel ASR is ignored on 8 GB-class GPUs when `ASR_HARD_MEMORY_SAFE=true`.
- Pyannote diarization runs on CUDA in the GPU compose profile with accuracy-first tuning (multi-sample grid, 44.1 kHz preprocess).
- CUDA batch size defaults to **4** on GPU compose (capped by `ASR_8GB_MAX_BATCH_SIZE`); long-form ASR uses 300-second windows with 45-second overlap by default.
- Diarization uses `pyannote/speaker-diarization-community-1` by default (Sep 2025, 30-50% lower DER than the legacy 3.1 pipeline). Accept the model terms on Hugging Face and set `HF_TOKEN`. Set `DIARIZATION_MODEL_ID` if you need the legacy `pyannote/speaker-diarization-3.1`.
- The pyannote runtime package is pinned to `pyannote.audio==4.0.4`. For offline setup, the only release asset you need is `pyannote_audio-4.0.4-py3-none-any.whl`; source archives are not required.
- Audio Enhancement is on by default in quality-first mode and always applied when diarization is enabled.

The app records a speed target in each job manifest: 3 minutes for audio under 9 minutes, otherwise one-third of the audio duration. Accuracy-first diarization is slower by design.

Refresh the browser during a long job to recover progress; finished jobs appear in **Previous transcripts**.

## 6. Output Files

The app stores outputs in the main app folder:

```text
storage/transcripts/
storage/jobs/          # job history manifests (running + completed)
storage/input/         # archived uploads per job
storage/audio/
storage/logs/
```

No database is required. Use the **Previous transcripts** panel in the UI to reload or download past jobs.

If the local LLM is not available, transcription still works and correction is skipped.

## 7. Docker GPU startup errors (Windows / WSL2)

If `docker compose -f docker-compose.gpu.yml up` fails **before** the app starts, with:

```text
error running prestart hook #0: exit status 127
Inconsistency detected by ld.so: dl-setup_hash.c: ... Assertion failed
```

that is a **Docker Desktop NVIDIA runtime** problem on the host, not this Python app. Plain containers work; any `--gpus` container fails the same way.

**Try in order:**

1. **Restart WSL and Docker Desktop** (fixes most regressions after updates):
   ```powershell
   wsl --shutdown
   ```
   Quit Docker Desktop completely, start it again, wait until it is running.

2. **Docker Desktop → Settings → Resources → GPU** — turn GPU support **off**, Apply, then **on**, Apply & Restart.

3. **Update the Windows NVIDIA driver** to the latest Game Ready / Studio driver for your GPU, then repeat step 1.

4. **Update Docker Desktop** to the latest release (you are on Docker 29.x; match NVIDIA Container Toolkit compatibility).

5. **Verify GPU passthrough** after each step:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
   ```
   This must print your GPU table. Only then run:
   ```powershell
   docker compose -f docker-compose.gpu.yml up -d
   ```

6. If it still fails: **Docker Desktop → Troubleshoot → Reset to factory defaults** (removes local images/containers; your `./models` and `./storage` bind mounts are safe on disk).

**Temporary workaround (no Docker GPU):** run locally with `run.bat` / `python launcher.py` on Windows if CUDA works outside Docker, or use `docker-compose.yml` / `docker-compose.openvino.yml` for CPU/OpenVINO (much slower, no NVIDIA hook).

**Note:** Earlier in this project the GPU container did start successfully (`Pyannote diarization pipeline ready on cpu`). If it worked before and fails now, a Windows driver, WSL, or Docker Desktop update is the usual cause — step 1 often restores it.
