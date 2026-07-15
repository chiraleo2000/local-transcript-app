@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM ============================================================
REM  Docker auto-deploy (reorganized under deploy/docker/)
REM  GPU: latest (CUDA 13.3) | cuda126 | cuda124
REM  CPU/iGPU: openvino
REM
REM  Deploy-Docker.bat
REM  Deploy-Docker.bat gpu -Build
REM  Deploy-Docker.bat gpu -CudaStack cuda126 -Build
REM  Deploy-Docker.bat gpu -CudaStack cuda124 -Build
REM  Deploy-Docker.bat openvino -Build
REM ============================================================

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\scripts\Deploy-Docker.ps1" %*
set ERR=%ERRORLEVEL%
if %ERR% neq 0 (
  echo.
  echo [ERROR] Deploy failed with code %ERR%
  pause
  exit /b %ERR%
)
echo.
pause
endlocal
exit /b 0
