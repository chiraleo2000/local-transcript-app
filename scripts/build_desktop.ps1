# Build LocalTranscriptApp native desktop bundle (PyInstaller onedir).
# Prerequisites: venv with requirements.txt + pyinstaller installed (run setup.bat first).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "venv not found. Run setup.bat first, then: pip install pyinstaller>=6.10"
}

Write-Host "[build] Using $Python"
& $Python -m pip install "pyinstaller>=6.10" --quiet

Write-Host "[build] PyInstaller onedir -> dist\LocalTranscriptApp\"
& $Python -m PyInstaller --noconfirm --clean LocalTranscriptApp.spec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Writable runtime dirs next to the exe (not inside _internal).
$Dist = Join-Path $Root "dist\LocalTranscriptApp"
foreach ($dir in @("models", "storage", "config")) {
    $path = Join-Path $Dist $dir
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

# Copy optional vendor binaries into dist\LocalTranscriptApp\bin (ffmpeg, etc.)
$Vendor = Join-Path $Root "vendor"
if (Test-Path $Vendor) {
    $Bin = Join-Path $Dist "bin"
    New-Item -ItemType Directory -Force -Path $Bin | Out-Null
    Write-Host "[build] Copying vendor binaries to $Bin"
    Copy-Item -Recurse -Force (Join-Path $Vendor "*") $Bin
}

# Production env for end users (no HF token).
Copy-Item -Force (Join-Path $Root ".env.production") (Join-Path $Dist ".env")

Write-Host "[build] Done: $Dist\LocalTranscriptApp.exe"
Write-Host "[build] Copy pre-cached models into $Dist\models before distributing."
