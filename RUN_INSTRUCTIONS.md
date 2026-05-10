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
http://localhost:7896
```

## 5. Hardware Rules

- NVIDIA GPU with CUDA and at least 6 GB VRAM: CUDA path.
- NVIDIA GPU below 6 GB VRAM: fallback to OpenVINO/CPU.
- Intel GPU/NPU: OpenVINO path when available.
- AMD GPU: CPU/OpenVINO fallback in v1.
- CPU only: CPU/OpenVINO fallback.

## 6. Output Files

The app stores outputs in the main app folder:

```text
storage/transcripts/
storage/jobs/
storage/audio/
storage/logs/
```

No database is required.

## 7. Optional Local LLM Correction

If you use Ollama locally, set these optional values in `.env`:

```text
OLLAMA_ENDPOINT=http://127.0.0.1:11434/api/generate
LOCAL_LLM_MODEL=llama3.1:8b
```

If the local LLM is not available, transcription still works and correction is skipped.
