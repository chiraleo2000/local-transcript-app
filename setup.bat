@echo off
echo ============================================
echo  Local Transcript App - Setup (Windows)
echo ============================================
echo.

REM Step 1: Create virtual environment
echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create venv. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

REM Step 2: Activate virtual environment
echo [2/5] Activating virtual environment...
call venv\Scripts\activate

REM Step 3: Upgrade pip
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

REM Step 4: Install OpenVINO first (pinned version)
echo [4/5] Installing OpenVINO 2026.1.0...
pip install openvino==2026.1.0

REM Step 5: Install remaining dependencies
echo [5/5] Installing remaining dependencies...
pip install -r requirements.txt

REM torchcodec is incompatible on Windows without FFmpeg full-shared DLLs.
REM Transformers 4.57+ tries to import it for ASR pipelines; remove it so
REM transformers falls back to soundfile/librosa (which works correctly).
pip uninstall torchcodec -y 2>nul

echo.
echo ============================================
echo  Setup complete!
echo.
echo  To run the app:
echo    venv\Scripts\activate
echo    run.bat
echo.
echo  Prepare/download local models:
echo    python scripts\bootstrap_models.py
echo ============================================
pause
