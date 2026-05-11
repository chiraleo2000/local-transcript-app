<#
.SYNOPSIS
    Creates a "Local Transcript App" desktop shortcut on Windows.

.DESCRIPTION
    Generates a .lnk file on the current user's Desktop that launches
    launcher.py (or LocalTranscriptApp.exe if the release .exe exists)
    inside a native pywebview desktop window — no browser needed.

    Run once after cloning / installing:
        powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1

    Optional flags:
        -StartMenu    Also add a Start Menu entry
        -Exe          Force use of LocalTranscriptApp.exe even when launcher.py exists
#>
param(
    [switch]$StartMenu,
    [switch]$Exe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Resolve paths ──────────────────────────────────────────────────────────
$AppRoot   = (Resolve-Path "$PSScriptRoot\..").Path
$ExePath   = Join-Path $AppRoot "release\LocalTranscriptApp.exe"
$LauncherPy= Join-Path $AppRoot "launcher.py"
$VenvPython= Join-Path $AppRoot "venv\Scripts\pythonw.exe"  # no console window
$IconDir   = Join-Path $AppRoot "assets"
$IconFile  = Join-Path $IconDir  "icon.ico"

# ── Choose target executable ───────────────────────────────────────────────
if ($Exe -and (Test-Path $ExePath)) {
    $Target    = $ExePath
    $Arguments = ""
    Write-Host "[shortcut] Using release .exe: $ExePath"
} elseif (Test-Path $LauncherPy) {
    if (Test-Path $VenvPython) {
        $Target    = $VenvPython
    } else {
        $Target    = (Get-Command python -ErrorAction SilentlyContinue)?.Source
        if (-not $Target) {
            $Target = (Get-Command python3 -ErrorAction SilentlyContinue)?.Source
        }
        if (-not $Target) {
            Write-Error "Python not found. Run setup.bat first, or activate your venv."
            exit 1
        }
    }
    $Arguments = "`"$LauncherPy`""
    Write-Host "[shortcut] Using launcher.py with: $Target"
} else {
    Write-Error "Neither LocalTranscriptApp.exe nor launcher.py was found at: $AppRoot"
    exit 1
}

# ── Placeholder icon (text file) if real icon missing ─────────────────────
if (-not (Test-Path $IconFile)) {
    New-Item -ItemType Directory -Path $IconDir -Force | Out-Null
    # Use the Python executable icon as fallback
    $IconFile = $Target
}

# ── Helper: create one .lnk ───────────────────────────────────────────────
function New-Shortcut {
    param([string]$Destination)

    $WshShell  = New-Object -ComObject WScript.Shell
    $Shortcut  = $WshShell.CreateShortcut($Destination)
    $Shortcut.TargetPath       = $Target
    $Shortcut.Arguments        = $Arguments
    $Shortcut.WorkingDirectory = $AppRoot
    $Shortcut.WindowStyle      = 1          # Normal window
    $Shortcut.Description      = "Local Transcript App — GPU-accelerated transcription"
    $Shortcut.IconLocation     = "$IconFile,0"
    $Shortcut.Save()
    Write-Host "[shortcut] Created: $Destination"
}

# ── Desktop shortcut ───────────────────────────────────────────────────────
$DesktopPath = [Environment]::GetFolderPath("Desktop")
New-Shortcut (Join-Path $DesktopPath "Local Transcript App.lnk")

# ── Optional Start Menu shortcut ──────────────────────────────────────────
if ($StartMenu) {
    $StartMenuDir = Join-Path ([Environment]::GetFolderPath("Programs")) "Local Transcript App"
    New-Item -ItemType Directory -Path $StartMenuDir -Force | Out-Null
    New-Shortcut (Join-Path $StartMenuDir "Local Transcript App.lnk")
    New-Shortcut (Join-Path $StartMenuDir "Uninstall (remove shortcut).lnk")
    Write-Host "[shortcut] Start Menu entry created: $StartMenuDir"
}

Write-Host ""
Write-Host "Done! Double-click 'Local Transcript App' on your Desktop to launch."
