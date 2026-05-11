@echo off
echo ============================================================
echo  Transcription Service - Run (Windows)
echo  Usage:  run.bat          ^<-- run app directly (default)
echo          run.bat gui      ^<-- native desktop window (pywebview)
echo          run.bat docker   ^<-- run via Docker on port 7896
echo ============================================================
echo.

if /i "%1"=="gui"    goto GUI
if /i "%1"=="docker" goto DOCKER

REM ============================================================
REM  DIRECT RUN MODE
REM ============================================================

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo [1/4] Checking Python...
venv\Scripts\python.exe --version
if errorlevel 1 ( echo [ERROR] Python missing in venv. && pause && exit /b 1 )

echo [2/4] Checking GPU...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo       No NVIDIA GPU detected — app will use OpenVINO/CPU.
) else (
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
)

echo [3/4] Checking FFmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ffmpeg not found. Install: choco install ffmpeg
) else (
    echo       FFmpeg OK
)

echo [4/4] Starting local transcript app on http://localhost:7896 ...
echo.
set "PYTHONPATH=%CD%;%PYTHONPATH%"
call venv\Scripts\activate
python app.py
goto END

REM ============================================================
REM  GUI / DESKTOP WINDOW MODE
REM ============================================================
:GUI
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
echo Starting Local Transcript App in native desktop window...
echo.
set "PYTHONPATH=%CD%;%PYTHONPATH%"
call venv\Scripts\activate
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

echo [2/3] Detecting GPU support in Docker...
set COMPOSE_FILES=-f docker-compose.yml
docker run --rm --gpus all nvidia/cuda:13.0.0-runtime-ubuntu24.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo       GPU not available in Docker — using CPU/OpenVINO mode.
    echo       To enable GPU: Docker Desktop ^> Settings ^> Resources ^> GPU ^> Enable
) else (
    echo       GPU available in Docker — using CUDA mode.
    set COMPOSE_FILES=-f docker-compose.gpu.yml
)

echo [3/3] Starting Docker container on http://localhost:7896 ...
echo.
docker compose %COMPOSE_FILES% up --build -d
docker logs -f transcription-service
goto END

:END
