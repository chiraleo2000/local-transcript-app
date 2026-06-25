@echo off
cd /d "%~dp0"
echo ============================================================
echo  Transcription Service - Run (Windows)
echo  Usage:  run.bat          ^<-- run app directly (default)
echo          run.bat gui      ^<-- native desktop window (pywebview)
echo          run.bat docker   ^<-- Docker (GPU :7988 or OpenVINO :7987)
echo ============================================================
echo.

if /i "%1"=="gui"    goto GUI
if /i "%1"=="docker" goto DOCKER

REM ============================================================
REM  DIRECT RUN MODE
REM ============================================================

call :SET_MODEL_ENV

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo [1/5] Checking Python...
venv\Scripts\python.exe --version
if errorlevel 1 ( echo [ERROR] Python missing in venv. && pause && exit /b 1 )

echo [2/5] Checking GPU...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo       No NVIDIA GPU detected — app will use OpenVINO/CPU.
) else (
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
)

echo [3/5] Checking FFmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ffmpeg not found. Install: choco install ffmpeg
) else (
    echo       FFmpeg OK
)

echo [4/5] Verifying local model cache under .\models\hf_cache\hub ...
call venv\Scripts\activate
python scripts\ensure_model_cache.py
if errorlevel 1 (
    echo [ERROR] Local model cache is incomplete. Check HF_TOKEN in .env and retry.
    pause
    exit /b 1
)

echo [5/5] Starting local transcript app on http://localhost:7896 ...
echo.
set "PYTHONPATH=%CD%;%PYTHONPATH%"
python app.py
goto END

REM ============================================================
REM  GUI / DESKTOP WINDOW MODE
REM ============================================================
:GUI
call :SET_MODEL_ENV
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
echo Verifying local model cache under .\models\hf_cache\hub ...
set "PYTHONPATH=%CD%;%PYTHONPATH%"
call venv\Scripts\activate
python scripts\ensure_model_cache.py
if errorlevel 1 (
    echo [ERROR] Local model cache is incomplete. Check HF_TOKEN in .env and retry.
    pause
    exit /b 1
)
echo Starting Local Transcript App in native desktop window...
echo.
python launcher.py
goto END

REM ============================================================
REM  DOCKER RUN MODE
REM ============================================================
:DOCKER
echo [1/3] Checking Docker...
where docker >nul 2>&1
if errorlevel 1 ( echo [ERROR] Docker not installed. && pause && exit /b 1 )
docker info >nul 2>&1
if errorlevel 1 ( echo [ERROR] Docker Desktop is not running. Start it first. && pause && exit /b 1 )

echo [2/3] Detecting accelerator support in Docker...
set COMPOSE_FILES=-f docker-compose.openvino.yml
docker run --rm --gpus all nvidia/cuda:13.0.0-runtime-ubuntu24.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    if exist docker-compose.openvino.yml (
        echo       No NVIDIA GPU in Docker — using OpenVINO/CPU AI compose.
    ) else (
        echo       GPU not available — using generic CPU compose.
        set COMPOSE_FILES=-f docker-compose.yml
        echo       To enable NVIDIA GPU: Docker Desktop ^> Settings ^> Resources ^> GPU ^> Enable
    )
) else (
    echo       NVIDIA GPU available — using CUDA compose.
    set COMPOSE_FILES=-f docker-compose.gpu.yml
)

echo [3/3] Docker deployment...
echo.
docker compose %COMPOSE_FILES% up --build -d
if "%COMPOSE_FILES%"=="-f docker-compose.gpu.yml" (
    echo GPU stack: http://localhost:7988
    docker logs -f transcription-service
) else (
    echo OpenVINO/CPU stack: http://localhost:7987
    docker logs -f transcription-service-openvino
)
goto END

:SET_MODEL_ENV
set "APP_MODEL_ROOT=%CD%\models"
set "HF_HOME=%CD%\models\hf_cache"
set "HF_HUB_CACHE=%CD%\models\hf_cache\hub"
set "HUGGINGFACE_HUB_CACHE=%CD%\models\hf_cache\hub"
set "TORCH_HOME=%CD%\models\torch"
set "OV_CACHE_DIR=%CD%\models\ov_cache"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "TRANSFORMERS_OFFLINE=0"
set "HF_HUB_OFFLINE=0"
set "APP_AUTO_DOWNLOAD_MISSING_MODELS=1"
set "ASR_ADAPTIVE_PERFORMANCE=1"
set "ASR_TARGET_SHORT_MAX_S=600"
set "ASR_TARGET_LONG_MAX_S=1800"
set "ASR_TARGET_LONG_AUDIO_S=3600"
set "ASR_TARGET_RT_RATIO_LONG=4"
set "ASR_NUM_BEAMS_MAX=8"
set "ASR_NUM_BEAMS_MIN=4"
set "ASR_NUM_BEAMS=6"
set "ASR_QUALITY_PROFILE=high"
set "ASR_KEEP_PRELOADED=1"
set "ASR_TURN_GUIDED_MERGE_GAP_S=0.55"
set "DIARIZATION_PRELOAD_MODE=eager"
set "DIARIZATION_KEEP_PRELOADED=1"
set "DIARIZATION_PREPROCESS_SR=16000"
set "DIARIZATION_GPU_CO_RESIDENT=0"
set "DIARIZATION_DEVICE=cuda"
set "DIARIZATION_PRELOAD_DEVICE=cpu"
set "DIARIZATION_ALLOW_8GB_CUDA=1"
set "DIARIZATION_CUDA_MIN_FREE_MB=768"
set "DIARIZATION_CUDA_RUN_MIN_FREE_MB=512"
set "ASR_UNLOAD_FOR_DIARIZATION=1"
set "ASR_DEFAULT_ENGINES=Auto"
set "ASR_AUTO_POLICY=quality"
exit /b 0

:END
