#Requires -Version 5.1
<#
.SYNOPSIS
  Start GPU Docker (loopback) + Cloudflare Tunnel for travel-proof public hostname.

.DESCRIPTION
  Keeps https://asrservice.demotoday.th reachable on any router/WiFi without
  port-forwarding or changing this PC's WiFi/DNS/routes.

  Requires CLOUDFLARE_TUNNEL_TOKEN in .env (from Cloudflare Zero Trust).

.EXAMPLE
  .\deploy\scripts\Setup-TravelTunnel.ps1
#>
[CmdletBinding()]
param(
    [string]$PublicHost = "asrservice.demotoday.th",
    [switch]$SkipDocker
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Set-EnvKey {
    param([string]$Path, [string]$Key, [string]$Value)
    $lines = @()
    if (Test-Path $Path) { $lines = Get-Content -Path $Path -Encoding UTF8 }
    $found = $false
    $out = foreach ($line in $lines) {
        if ($line -match ("^\s*#?\s*" + [regex]::Escape($Key) + "\s*=")) {
            $found = $true
            "$Key=$Value"
        } else { $line }
    }
    if (-not $found) {
        $out = @($out) + @("", "# Travel tunnel (Setup-TravelTunnel.ps1)", "$Key=$Value")
    }
    Set-Content -Path $Path -Value $out -Encoding UTF8
}

$envPath = Join-Path $RepoRoot ".env"
if (-not (Test-Path $envPath)) {
    Copy-Item (Join-Path $RepoRoot ".env.example") $envPath
}

$tokenLine = Select-String -Path $envPath -Pattern '^\s*CLOUDFLARE_TUNNEL_TOKEN=(.+)$' -ErrorAction SilentlyContinue |
    Select-Object -First 1
$token = $null
if ($tokenLine) { $token = $tokenLine.Matches[0].Groups[1].Value.Trim() }

if (-not $token) {
    Write-Host @"

CLOUDFLARE_TUNNEL_TOKEN is missing from .env

One-time setup (any location afterward just re-runs this script):
  1. Open https://one.dash.cloudflare.com/ -> Zero Trust -> Networks -> Tunnels
  2. Create tunnel (e.g. local-transcript), copy the token
  3. Add to .env:
       CLOUDFLARE_TUNNEL_TOKEN=eyJ...
  4. Public hostname in Cloudflare:
       asrservice.demotoday.th  ->  http://host.docker.internal:7988
       (Service type: HTTP)
  5. DNS for demotoday.th: CNAME asrservice -> <tunnel-id>.cfargotunnel.com (Proxied/orange cloud)
  6. Re-run: .\deploy\scripts\Setup-TravelTunnel.ps1

"@ -ForegroundColor Yellow
    throw "Set CLOUDFLARE_TUNNEL_TOKEN in .env then re-run."
}

$publicBase = "https://$PublicHost"
Set-EnvKey -Path $envPath -Key "APP_PUBLIC_BASE_URL" -Value $publicBase
$env:CLOUDFLARE_TUNNEL_TOKEN = $token
$env:APP_PUBLIC_BASE_URL = $publicBase

if (-not $SkipDocker) {
    Write-Step "Starting GPU Docker on 127.0.0.1:7988 (WiFi-safe loopback)"
    docker compose -f deploy/docker/latest/compose.yml -f deploy/docker/compose.proxy-override.yml up -d
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:7988/startup-events" -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -eq 200) { $ok = $true; break }
        } catch { Start-Sleep -Seconds 2 }
    }
    if (-not $ok) { throw "App not healthy on 127.0.0.1:7988" }
    $ports = docker port transcription-service 2>$null
    if ("$ports" -match "0\.0\.0\.0:7988") {
        throw "WiFi-safe check failed: Docker published 0.0.0.0:7988"
    }
    Write-Host "Upstream OK: $ports"
}

Write-Step "Starting Cloudflare Tunnel (outbound only -- no router port-forward)"
docker compose `
    -f deploy/docker/latest/compose.yml `
    -f deploy/docker/compose.proxy-override.yml `
    -f deploy/docker/compose.tunnel.yml `
    up -d cloudflared

Start-Sleep -Seconds 3
docker ps --filter "name=local-transcript-tunnel" --format "table {{.Names}}\t{{.Status}}"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Travel tunnel ready" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host " Share URL:  $publicBase"
Write-Host " Upstream:   http://127.0.0.1:7988 (loopback)"
Write-Host " Tunnel:     cloudflared -> host.docker.internal:7988"
Write-Host " WiFi-safe:  no adapter/DNS/route/ICS/port-forward on this PC"
Write-Host " At a new hotel/office: connect WiFi, run this script again (token stays in .env)"
Write-Host ""
