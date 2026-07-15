#Requires -Version 5.1
<#
.SYNOPSIS
  Auto-deploy Local Transcript App to Docker using reorganized stacks under deploy/docker/.

.DESCRIPTION
  Backends: gpu | openvino | auto
  CUDA stacks (GPU): latest (13.3) | cuda126 | cuda124
  Config from .env: DEPLOY_BACKEND, DEPLOY_CUDA_STACK, DEPLOY_LOOPBACK

.EXAMPLE
  .\Deploy-Docker.bat
  .\Deploy-Docker.bat gpu -Build
  .\Deploy-Docker.bat gpu -CudaStack cuda126 -Build
  .\Deploy-Docker.bat gpu -CudaStack cuda124 -Build
  .\Deploy-Docker.bat openvino -Build
#>
[CmdletBinding()]
param(
    [ValidateSet("auto", "gpu", "openvino")]
    [string]$Backend = "",

    [ValidateSet("latest", "cuda126", "cuda124")]
    [string]$CudaStack = "",

    [switch]$Build,
    [switch]$ClearCache,
    [switch]$Loopback,
    [switch]$FollowLogs
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Get-EnvValue([string]$Key) {
    $envFile = Join-Path $RepoRoot ".env"
    if (-not (Test-Path $envFile)) { return $null }
    $line = Select-String -Path $envFile -Pattern ("^\s*" + [regex]::Escape($Key) + "\s*=\s*(.*)$") -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $line) { return $null }
    return $line.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
}

function Test-DockerReady {
    try {
        docker info 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Test-NvidiaDocker {
    foreach ($img in @(
            "nvidia/cuda:12.4.1-base-ubuntu22.04",
            "nvidia/cuda:12.6.3-base-ubuntu24.04",
            "nvidia/cuda:13.3.0-base-ubuntu24.04"
        )) {
        try {
            docker run --rm --gpus all $img nvidia-smi 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { return $true }
        } catch { }
    }
    return $false
}

function Clear-LocalCaches {
    Write-Step "Clearing local caches (models/ untouched)"
    foreach ($d in @(
            (Join-Path $RepoRoot ".cache"),
            (Join-Path $RepoRoot ".pytest_cache"),
            (Join-Path $RepoRoot "htmlcov"),
            (Join-Path $RepoRoot "deploy\nginx\runtime\logs"),
            (Join-Path $RepoRoot "deploy\nginx\runtime\temp")
        )) {
        if (Test-Path $d) {
            Get-ChildItem $d -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "  cleaned $d"
        }
    }
    Get-ChildItem -Path $RepoRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\venv\\' } |
        ForEach-Object { Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
    if (Test-DockerReady) {
        docker builder prune -f 2>$null | Out-Null
        docker image prune -f 2>$null | Out-Null
    }
}

$envPath = Join-Path $RepoRoot ".env"
if (-not (Test-Path $envPath)) {
    Copy-Item (Join-Path $RepoRoot ".env.example") $envPath
    Write-Host "Created .env from .env.example"
}

if (-not $Backend) {
    $fromEnv = Get-EnvValue "DEPLOY_BACKEND"
    $Backend = if ($fromEnv -in @("auto", "gpu", "openvino")) { $fromEnv } else { "auto" }
}
if (-not $CudaStack) {
    $fromCuda = Get-EnvValue "DEPLOY_CUDA_STACK"
    $CudaStack = if ($fromCuda -in @("latest", "cuda126", "cuda124")) { $fromCuda } else { "latest" }
}

Write-Step "Docker deploy backend=$Backend cuda=$CudaStack"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker not found. Install Docker Desktop and retry."
}
if (-not (Test-DockerReady)) {
    Write-Host "Starting Docker Desktop..."
    $dd = @(
        "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
        "${env:ProgramFiles(x86)}\Docker\Docker\Docker Desktop.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($dd) {
        Start-Process $dd
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Seconds 3
            if (Test-DockerReady) { break }
        }
    }
    if (-not (Test-DockerReady)) { throw "Docker Engine is not reachable." }
}

if ($ClearCache) { Clear-LocalCaches }

$resolved = $Backend
if ($Backend -eq "auto") {
    Write-Step "Auto-detect NVIDIA Docker GPU"
    if (Test-NvidiaDocker) { $resolved = "gpu"; Write-Host "  -> GPU" }
    else { $resolved = "openvino"; Write-Host "  -> OpenVINO" }
}

$proxyOverride = Join-Path $RepoRoot "deploy\docker\compose.proxy-override.yml"
$stacks = @(
    "deploy/docker/latest/compose.yml",
    "deploy/docker/cuda126/compose.yml",
    "deploy/docker/cuda124/compose.yml",
    "deploy/docker/openvino/compose.yml"
)

Write-Step "Stopping conflicting containers"
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
foreach ($f in $stacks) {
    docker compose -f $f down 2>&1 | Out-Null
}
if (Test-Path $proxyOverride) {
    docker compose -f deploy/docker/latest/compose.yml -f $proxyOverride down 2>&1 | Out-Null
}
$ErrorActionPreference = $prevEap

if ($resolved -eq "gpu") {
    $composeFile = "deploy/docker/$CudaStack/compose.yml"
    if (-not (Test-Path $composeFile)) { throw "Missing $composeFile" }
    $composeArgs = @("-f", $composeFile)
    Write-Host "  stack file: $composeFile"
    if ($Loopback -or ((Get-EnvValue "DEPLOY_LOOPBACK") -eq "1")) {
        $composeArgs += @("-f", "deploy/docker/compose.proxy-override.yml")
        Write-Host "  loopback: 127.0.0.1:7988"
    }
    $url = if ($Loopback -or ((Get-EnvValue "DEPLOY_LOOPBACK") -eq "1")) { "http://127.0.0.1:7988" } else { "http://localhost:7988" }
    $name = "transcription-service"
} else {
    $composeArgs = @("-f", "deploy/docker/openvino/compose.yml")
    $url = "http://localhost:7987"
    $name = "transcription-service-openvino"
}

if ($Build) {
    Write-Step "Building image"
    & docker compose @composeArgs build
    if ($LASTEXITCODE -ne 0) { throw "docker compose build failed" }
}

Write-Step "Starting $resolved"
& docker compose @composeArgs up -d
if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }

Write-Step "Waiting for health ($name)"
$ok = $false
$healthUrl = if ($resolved -eq "gpu") { "http://127.0.0.1:7988/startup-events" } else { "http://127.0.0.1:7987/startup-events" }
for ($i = 0; $i -lt 90; $i++) {
    try {
        $r = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 5
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { Start-Sleep -Seconds 2 }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Deployed: $resolved / $CudaStack" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host " UI:        $url"
Write-Host " Container: $name"
Write-Host " Ports:     $(docker port $name 2>$null)"
Write-Host " Health:    $(if ($ok) { 'OK' } else { 'starting' })"
Write-Host " Layout:    deploy/docker/"
Write-Host ""

if ($FollowLogs) { docker logs -f $name }
