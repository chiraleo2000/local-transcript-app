@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ---------------------------------------------------------------------------
REM Local Transcript App — Windows installer (GUI / direct run, no Docker)
REM ---------------------------------------------------------------------------
REM Installs under %%LOCALAPPDATA%%\LocalTranscriptApp by default, creates a
REM venv, installs Python deps, writes .env with HF token + resource settings,
REM and creates Start Menu + Desktop shortcuts.
REM
REM Usage:
REM   install.bat
REM   install.bat --prefix "D:\Apps\LocalTranscriptApp"
REM   install.bat --hf-token hf_xxxx --cpu-threads 4 --force-backend openvino
REM   install.bat --uninstall
REM
REM Env fallbacks: HF_TOKEN, APP_CPU_THREADS, APP_FORCE_BACKEND, OV_DEVICE
REM ---------------------------------------------------------------------------

set "APP_ID=LocalTranscriptApp"
set "APP_NAME=Local Transcript App"
set "APP_VERSION=1.2.6"
set "DEFAULT_PREFIX=%LOCALAPPDATA%\%APP_ID%"
set "PREFIX=%DEFAULT_PREFIX%"
set "DO_UNINSTALL=0"
set "SKIP_DEPS=0"
set "HF_TOKEN_ARG="
set "CPU_THREADS_ARG="
set "FORCE_BACKEND_ARG="
set "OV_DEVICE_ARG="
set "MIN_RAM_ARG=8192"
set "MIN_CPU_ARG=4"
set "MIN_VRAM_ARG=8192"
set "UI_MAX_JOBS_ARG=1"
set "UI_CONCURRENCY_ARG=4"

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--prefix" (
  set "PREFIX=%~2"
  shift & shift & goto parse_args
)
if /i "%~1"=="--uninstall" (
  set "DO_UNINSTALL=1"
  shift & goto parse_args
)
if /i "%~1"=="--skip-deps" (
  set "SKIP_DEPS=1"
  shift & goto parse_args
)
if /i "%~1"=="--hf-token" (
  set "HF_TOKEN_ARG=%~2"
  shift & shift & goto parse_args
)
if /i "%~1"=="--cpu-threads" (
  set "CPU_THREADS_ARG=%~2"
  shift & shift & goto parse_args
)
if /i "%~1"=="--force-backend" (
  set "FORCE_BACKEND_ARG=%~2"
  shift & shift & goto parse_args
)
if /i "%~1"=="--ov-device" (
  set "OV_DEVICE_ARG=%~2"
  shift & shift & goto parse_args
)
if /i "%~1"=="--min-ram-mb" (
  set "MIN_RAM_ARG=%~2"
  shift & shift & goto parse_args
)
if /i "%~1"=="--min-cpu-threads" (
  set "MIN_CPU_ARG=%~2"
  shift & shift & goto parse_args
)
if /i "%~1"=="--min-vram-mb" (
  set "MIN_VRAM_ARG=%~2"
  shift & shift & goto parse_args
)
if /i "%~1"=="-h" goto help
if /i "%~1"=="--help" goto help
echo [error] Unknown option: %~1
exit /b 1

:help
echo Usage: install.bat [--prefix DIR] [--hf-token TOKEN] [--cpu-threads N]
echo                    [--force-backend cuda^|rocm^|openvino^|directml^|cpu]
echo                    [--ov-device GPU^|NPU^|CPU] [--min-ram-mb 8192]
echo                    [--uninstall] [--skip-deps]
exit /b 0

:args_done
if not defined HF_TOKEN_ARG if defined HF_TOKEN set "HF_TOKEN_ARG=%HF_TOKEN%"
if not defined CPU_THREADS_ARG if defined APP_CPU_THREADS set "CPU_THREADS_ARG=%APP_CPU_THREADS%"
if not defined FORCE_BACKEND_ARG if defined APP_FORCE_BACKEND set "FORCE_BACKEND_ARG=%APP_FORCE_BACKEND%"
if not defined OV_DEVICE_ARG if defined OV_DEVICE set "OV_DEVICE_ARG=%OV_DEVICE%"
if not defined CPU_THREADS_ARG set "CPU_THREADS_ARG=0"

set "SCRIPT_DIR=%~dp0"
set "SRC_DIR=%SCRIPT_DIR%.."
for %%I in ("%SRC_DIR%") do set "SRC_DIR=%%~fI"
for %%I in ("%PREFIX%") do set "PREFIX=%%~fI"

if "%DO_UNINSTALL%"=="1" (
  echo [install] Uninstalling %APP_NAME% from %PREFIX%
  if exist "%PREFIX%" rmdir /s /q "%PREFIX%"
  del /f /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\%APP_NAME%.lnk" 2>nul
  del /f /q "%USERPROFILE%\Desktop\%APP_NAME%.lnk" 2>nul
  echo [install] Uninstall complete.
  exit /b 0
)

where python >nul 2>&1
if errorlevel 1 (
  echo [error] python not found. Install Python 3.10+ and retry.
  exit /b 1
)

echo [install] Installing %APP_NAME% %APP_VERSION%
echo [install]   source : %SRC_DIR%
echo [install]   prefix : %PREFIX%

if not exist "%PREFIX%" mkdir "%PREFIX%"

echo [install] Copying application files...
robocopy "%SRC_DIR%" "%PREFIX%" /E /XD venv .venv .git __pycache__ build dist .cursor .claude storage\audio storage\input storage\jobs storage\transcripts storage\logs storage\acceptance_output tests\output release /XF .env *.pyc *.log .coverage coverage.xml /NFL /NDL /NJH /NJS /nc /ns /np
if errorlevel 8 (
  echo [error] robocopy failed with code %ERRORLEVEL%
  exit /b 1
)

if exist "%PREFIX%\.env.production" if not exist "%PREFIX%\.env" (
  copy /y "%PREFIX%\.env.production" "%PREFIX%\.env" >nul
)
if not exist "%PREFIX%\.env" if exist "%PREFIX%\.env.example" (
  copy /y "%PREFIX%\.env.example" "%PREFIX%\.env" >nul
)

echo [install] Writing HF token + resource settings into .env ...
set "WRITE_ENV=%PREFIX%\installer\write_runtime_env.py"
if not exist "%WRITE_ENV%" set "WRITE_ENV=%SRC_DIR%\installer\write_runtime_env.py"
python "%WRITE_ENV%" --env-path "%PREFIX%\.env" --hf-token "%HF_TOKEN_ARG%" --cpu-threads "%CPU_THREADS_ARG%" --force-backend "%FORCE_BACKEND_ARG%" --ov-device "%OV_DEVICE_ARG%" --min-ram-mb "%MIN_RAM_ARG%" --min-cpu-threads "%MIN_CPU_ARG%" --min-vram-mb "%MIN_VRAM_ARG%" --ui-max-jobs "%UI_MAX_JOBS_ARG%" --ui-gradio-concurrency "%UI_CONCURRENCY_ARG%"
if errorlevel 1 (
  echo [error] Failed to write .env resource settings.
  exit /b 1
)

if "%HF_TOKEN_ARG%"=="" (
  echo [install] NOTE: No HF_TOKEN set. Required only if models\ is missing — accept gated model terms on Hugging Face, then re-run with --hf-token or edit %PREFIX%\.env
) else (
  echo [install] HF_TOKEN written to .env
)

if "%SKIP_DEPS%"=="1" goto shortcuts

echo [install] Creating virtualenv...
python -m venv "%PREFIX%\venv"
if errorlevel 1 (
  echo [error] Failed to create venv.
  exit /b 1
)
call "%PREFIX%\venv\Scripts\activate.bat"
python -m pip install --upgrade pip wheel
echo [install] Installing Python dependencies (this can take several minutes)...
pip install -r "%PREFIX%\requirements.txt"
pip uninstall torchcodec -y 2>nul
python -c "import os,sys; d=os.path.join(sys.prefix,'Lib','site-packages','torchcodec-0.0.1.dist-info'); os.makedirs(d,exist_ok=True); open(os.path.join(d,'METADATA'),'w').write('Metadata-Version: 2.1\nName: torchcodec\nVersion: 0.0.1\n'); open(os.path.join(d,'RECORD'),'w').write(''); open(os.path.join(d,'INSTALLER'),'w').write('pip'); print('torchcodec stub created.')"
pip install pywebview
call deactivate

:shortcuts
echo [install] Creating shortcuts...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell; $dir='%PREFIX%'; $exe=Join-Path $dir 'run.bat'; $sm=Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\%APP_NAME%.lnk'; $s=$ws.CreateShortcut($sm); $s.TargetPath=$exe; $s.WorkingDirectory=$dir; $s.Save(); $desk=Join-Path $env:USERPROFILE 'Desktop\%APP_NAME%.lnk'; $d=$ws.CreateShortcut($desk); $d.TargetPath=$exe; $d.WorkingDirectory=$dir; $d.Save()"

echo.
echo [install] Installation complete.
echo [install]   Launch: %PREFIX%\run.bat
echo [install]   GUI:    %PREFIX%\run.bat gui
echo [install]   Config: %PREFIX%\.env  (token + MIN_* / APP_CPU_THREADS / APP_FORCE_BACKEND)
echo.
echo [install] If models\ is missing, bootstrap once:
echo [install]   %PREFIX%\venv\Scripts\python.exe %PREFIX%\scripts\bootstrap_models.py
exit /b 0
