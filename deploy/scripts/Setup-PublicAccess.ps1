#Requires -Version 5.1
<#
.SYNOPSIS
  Deploy Local Transcript App for LAN/public access behind nginx (WiFi-safe).

.DESCRIPTION
  WiFi-safe public path (default for internet):
  - Docker UI only on 127.0.0.1:7988 (never 0.0.0.0:7988)
  - nginx terminates :80/:443 for PublicHost (default asrservice.demotoday.th)
  - Firewall: append-only LocalTranscript rules for 80/443; never opens 7988
  - Does NOT change WiFi adapter IP, DNS, gateway, metric, ICS, or routes

.EXAMPLE
  .\deploy\scripts\Setup-PublicAccess.ps1 -Audience internet -PublicHost asrservice.demotoday.th -EnableTls
  .\deploy\scripts\Setup-PublicAccess.ps1 -Audience lan
#>
[CmdletBinding()]
param(
    [ValidateSet("nginx", "iis")]
    [string]$Proxy = "nginx",

    [ValidateSet("lan", "internet")]
    [string]$Audience = "internet",

    [string]$PublicHost = "asrservice.demotoday.th",

    [string]$AuthUser = "admin",

    [string]$AuthPassword = "",

    [switch]$EnableTls,

    [switch]$SkipDocker,
    [switch]$SkipFirewall,
    [switch]$SkipNginxInstall
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Get-LanIPv4 {
    $addrs = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -notlike "127.*" -and
            $_.InterfaceAlias -notmatch "WSL|vEthernet|Loopback|Docker|Hyper-V" -and
            $_.PrefixOrigin -ne "WellKnown"
        } |
        Sort-Object -Property InterfaceMetric
    if ($addrs) { return $addrs[0].IPAddress }
    return "127.0.0.1"
}

function Set-EnvKey {
    param([string]$Path, [string]$Key, [string]$Value)
    $lines = @()
    if (Test-Path $Path) {
        $lines = Get-Content -Path $Path -Encoding UTF8
    }
    $found = $false
    $out = foreach ($line in $lines) {
        if ($line -match ("^\s*#?\s*" + [regex]::Escape($Key) + "\s*=")) {
            $found = $true
            "$Key=$Value"
        } else {
            $line
        }
    }
    if (-not $found) {
        $out = @($out) + @("", "# Public access (deploy/scripts/Setup-PublicAccess.ps1)", "$Key=$Value")
    }
    Set-Content -Path $Path -Value $out -Encoding UTF8
}

function New-RandomPassword([int]$Length = 20) {
    $chars = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789!@#$%"
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $bytes = New-Object byte[] $Length
    $rng.GetBytes($bytes)
    -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
}

function Find-OpenSsl {
    $cmd = Get-Command openssl.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "C:\Program Files\OpenSSL-Win64\bin\openssl.exe",
        "C:\Program Files\OpenSSL-Win32\bin\openssl.exe",
        "C:\Program Files\FireDaemon OpenSSL 3\bin\openssl.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    $winget = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter openssl.exe -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($winget) { return $winget.FullName }
    return $null
}

function Ensure-SelfSignedTls {
    param([string]$CertDir, [string]$DnsName)
    New-Item -ItemType Directory -Force -Path $CertDir | Out-Null
    $certPath = Join-Path $CertDir "fullchain.pem"
    $keyPath = Join-Path $CertDir "privkey.pem"
    if ((Test-Path $certPath) -and (Test-Path $keyPath)) {
        Write-Host "TLS certs already present in $CertDir"
        return
    }
    Write-Step "Generating self-signed TLS certificate for $DnsName"

    $openssl = Find-OpenSsl
    if ($openssl) {
        $conf = Join-Path $CertDir "openssl.cnf"
        @"
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_req

[dn]
CN = $DnsName

[v3_req]
subjectAltName = @alt_names
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth

[alt_names]
DNS.1 = $DnsName
DNS.2 = localhost
"@ | Set-Content -Path $conf -Encoding ASCII
        & $openssl req -x509 -nodes -newkey rsa:2048 -keyout $keyPath -out $certPath -days 825 -config $conf 2>$null
        if ((Test-Path $certPath) -and (Test-Path $keyPath)) {
            Write-Host "Wrote $certPath and $keyPath via $openssl"
            return
        }
    }

    # Fallback: openssl in a short-lived container (no WiFi/adapter changes)
    Write-Host "Local openssl missing; generating certs via Docker alpine/openssl"
    $mount = ($CertDir -replace "\\", "/")
    docker run --rm -v "${CertDir}:/certs" alpine/openssl req -x509 -nodes -newkey rsa:2048 `
        -keyout /certs/privkey.pem -out /certs/fullchain.pem -days 825 `
        -subj "/CN=$DnsName" `
        -addext "subjectAltName=DNS:$DnsName,DNS:localhost"
    if (-not ((Test-Path $certPath) -and (Test-Path $keyPath))) {
        throw "Failed to generate TLS PEM files in $CertDir"
    }
    Write-Host "Wrote $certPath and $keyPath via Docker openssl"
}

function Find-NginxRoot {
    $candidates = @(
        "$env:ProgramFiles\nginx",
        "${env:ProgramFiles(x86)}\nginx",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $exe = Get-ChildItem -Path $c -Filter nginx.exe -Recurse -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($exe) { return $exe.Directory.FullName }
        }
    }
    $cmd = Get-Command nginx.exe -ErrorAction SilentlyContinue
    if ($cmd) { return (Split-Path $cmd.Source -Parent) }
    return $null
}

function Copy-NoBom {
    param([string]$Src, [string]$Dst)
    $bytes = [System.IO.File]::ReadAllBytes($Src)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        $bytes = $bytes[3..($bytes.Length - 1)]
    }
    [System.IO.File]::WriteAllBytes($Dst, $bytes)
}

Write-Step "Defaults: Proxy=$Proxy Audience=$Audience PublicHost=$PublicHost (WiFi-safe: no adapter/DNS/route changes)"

$lanIp = Get-LanIPv4
if (-not $PublicHost) {
    if ($Audience -eq "lan") { $PublicHost = $lanIp }
    else { $PublicHost = "asrservice.demotoday.th" }
}
if ($Audience -eq "internet") { $EnableTls = $true }
$scheme = if ($EnableTls) { "https" } else { "http" }
$publicBase = "${scheme}://${PublicHost}"
if (-not $AuthPassword) {
    $envPathProbe = Join-Path $RepoRoot ".env"
    if (Test-Path $envPathProbe) {
        $existing = Select-String -Path $envPathProbe -Pattern '^\s*GRADIO_AUTH_PASSWORD=(.+)$' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($existing) { $AuthPassword = $existing.Matches[0].Groups[1].Value.Trim() }
    }
    if (-not $AuthPassword) { $AuthPassword = New-RandomPassword 18 }
}

$envPath = Join-Path $RepoRoot ".env"
if (-not (Test-Path $envPath)) {
    Copy-Item (Join-Path $RepoRoot ".env.example") $envPath
}

Write-Step "Writing public auth + URL into .env"
Set-EnvKey -Path $envPath -Key "GRADIO_AUTH_USER" -Value $AuthUser
Set-EnvKey -Path $envPath -Key "GRADIO_AUTH_PASSWORD" -Value $AuthPassword
Set-EnvKey -Path $envPath -Key "APP_PUBLIC_BASE_URL" -Value $publicBase

$credFile = Join-Path $RepoRoot "deploy\.public-credentials.txt"
@"
# Generated $(Get-Date -Format o) -- do not commit
# WiFi-safe deploy: Docker on 127.0.0.1:7988 only; nginx on 80/443
APP_PUBLIC_BASE_URL=$publicBase
GRADIO_AUTH_USER=$AuthUser
GRADIO_AUTH_PASSWORD=$AuthPassword
PROXY=$Proxy
AUDIENCE=$Audience
PUBLIC_HOST=$PublicHost
WIFI_HINT_IP=$lanIp
"@ | Set-Content -Path $credFile -Encoding UTF8
Write-Host "Credentials saved to deploy\.public-credentials.txt (gitignored)"

if (-not $SkipDocker) {
    Write-Step "Starting GPU Docker with proxy override (127.0.0.1:7988 only -- WiFi-safe)"
    $env:GRADIO_AUTH_USER = $AuthUser
    $env:GRADIO_AUTH_PASSWORD = $AuthPassword
    $env:APP_PUBLIC_BASE_URL = $publicBase
    docker compose -f deploy/docker/latest/compose.yml -f deploy/docker/compose.proxy-override.yml up -d
    Write-Step "Verifying http://127.0.0.1:7988/startup-events"
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:7988/startup-events" -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -eq 200) { $ok = $true; break }
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $ok) { throw "App did not become healthy on 127.0.0.1:7988" }
    $ports = docker port transcription-service 2>$null
    if ("$ports" -match "0\.0\.0\.0:7988") {
        throw "WiFi-safe check failed: Docker published 0.0.0.0:7988. Use deploy/docker/compose.proxy-override.yml"
    }
    Write-Host "Docker UI healthy on loopback :7988 ($ports)"
}

if ($Proxy -eq "nginx") {
    Write-Step "Configuring nginx (repo-local prefix deploy/nginx/runtime)"
    if (-not $SkipNginxInstall) {
        $nginxBinRoot = Find-NginxRoot
        if (-not $nginxBinRoot) {
            Write-Host "Installing nginx via winget..."
            winget install --id nginxinc.nginx -e --accept-package-agreements --accept-source-agreements
            Start-Sleep -Seconds 3
            $nginxBinRoot = Find-NginxRoot
        }
        if (-not $nginxBinRoot) { throw "nginx.exe not found after install. Install manually and re-run." }
        Write-Host "nginx.exe from: $nginxBinRoot"
    } else {
        $nginxBinRoot = Find-NginxRoot
        if (-not $nginxBinRoot) { throw "nginx not found and -SkipNginxInstall set" }
    }

    $nginxExe = Join-Path $nginxBinRoot "nginx.exe"
    $prefix = Join-Path $RepoRoot "deploy\nginx\runtime"
    $confDir = Join-Path $prefix "conf"
    $confD = Join-Path $confDir "conf.d"
    $runtimeCerts = Join-Path $prefix "certs"
    New-Item -ItemType Directory -Force -Path $confD | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $prefix "logs") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $prefix "temp") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $prefix "html") | Out-Null
    New-Item -ItemType Directory -Force -Path $runtimeCerts | Out-Null

    $mimeSrc = Join-Path $nginxBinRoot "conf\mime.types"
    if (Test-Path $mimeSrc) {
        Copy-Item $mimeSrc (Join-Path $confDir "mime.types") -Force
    }

    $mainConf = @"
worker_processes  1;
error_log  logs/error.log;
pid        logs/nginx.pid;

events {
    worker_connections  1024;
}

http {
    include       mime.types;
    default_type  application/octet-stream;
    sendfile        on;
    keepalive_timeout  65;
    include conf.d/*.conf;
}
"@
    [System.IO.File]::WriteAllText((Join-Path $confDir "nginx.conf"), $mainConf, (New-Object System.Text.UTF8Encoding $false))

    # Clear old site configs so LAN catch-all does not compete with public vhost
    Get-ChildItem $confD -Filter "*.conf" -ErrorAction SilentlyContinue | Remove-Item -Force

    $domainConfSrc = Join-Path $RepoRoot "deploy\nginx\$PublicHost.conf"
    $genericPublicSrc = Join-Path $RepoRoot "deploy\nginx\asrservice.demotoday.th.conf"
    $siteDst = Join-Path $confD "site.conf"

    if ($Audience -eq "internet" -or $EnableTls) {
        $certDir = Join-Path $RepoRoot "deploy\nginx\certs"
        Ensure-SelfSignedTls -CertDir $certDir -DnsName $PublicHost
        if ((Test-Path (Join-Path $certDir "fullchain.pem")) -and (Test-Path (Join-Path $certDir "privkey.pem"))) {
            Copy-Item (Join-Path $certDir "fullchain.pem") (Join-Path $runtimeCerts "fullchain.pem") -Force
            Copy-Item (Join-Path $certDir "privkey.pem") (Join-Path $runtimeCerts "privkey.pem") -Force
        } else {
            throw "TLS PEM files missing under deploy/nginx/certs (need fullchain.pem + privkey.pem)"
        }

        if (Test-Path $domainConfSrc) {
            Copy-NoBom -Src $domainConfSrc -Dst $siteDst
            Write-Host "Using domain nginx conf: $domainConfSrc"
        } elseif ($PublicHost -eq "asrservice.demotoday.th" -and (Test-Path $genericPublicSrc)) {
            Copy-NoBom -Src $genericPublicSrc -Dst $siteDst
            Write-Host "Using asrservice.demotoday.th nginx conf"
        } else {
            # Generate a public vhost from template paths relative to nginx -p prefix
            $generated = @"
upstream local_transcript_gpu {
    server 127.0.0.1:7988;
    keepalive 8;
}

map `$http_upgrade `$connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 80;
    listen [::]:80;
    server_name $PublicHost;

    location ^~ /.well-known/acme-challenge/ {
        default_type "text/plain";
        root html;
    }

    location / {
        return 301 https://`$host`$request_uri;
    }
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name $PublicHost;

    ssl_certificate     certs/fullchain.pem;
    ssl_certificate_key certs/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    client_max_body_size 512m;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
    proxy_connect_timeout 60s;

    location / {
        proxy_pass http://local_transcript_gpu;
        proxy_http_version 1.1;
        proxy_set_header Host `$host;
        proxy_set_header X-Real-IP `$remote_addr;
        proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto `$scheme;
        proxy_set_header Upgrade `$http_upgrade;
        proxy_set_header Connection `$connection_upgrade;
        proxy_buffering off;
    }
}
"@
            [System.IO.File]::WriteAllText($siteDst, $generated, (New-Object System.Text.UTF8Encoding $false))
            Write-Host "Generated public nginx vhost for $PublicHost"
        }

        # Windows nginx resolves relative SSL paths from the conf file dir; force absolute paths
        $certAbs = ((Join-Path $runtimeCerts "fullchain.pem") -replace "\\", "/")
        $keyAbs = ((Join-Path $runtimeCerts "privkey.pem") -replace "\\", "/")
        $siteText = [System.IO.File]::ReadAllText($siteDst)
        $siteText = $siteText -replace 'ssl_certificate\s+\S+;', "ssl_certificate     $certAbs;"
        $siteText = $siteText -replace 'ssl_certificate_key\s+\S+;', "ssl_certificate_key $keyAbs;"
        [System.IO.File]::WriteAllText($siteDst, $siteText, (New-Object System.Text.UTF8Encoding $false))
        Write-Host "SSL paths: $certAbs"
    } else {
        $lanConfSrc = Join-Path $RepoRoot "deploy\nginx\local-transcript.lan.conf"
        Copy-NoBom -Src $lanConfSrc -Dst $siteDst
        Write-Host "Using LAN HTTP nginx conf"
    }

    Get-Process nginx -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1

    & $nginxExe -p $prefix -c conf/nginx.conf -t
    if ($LASTEXITCODE -ne 0) { throw "nginx config test failed" }
    Start-Process -FilePath $nginxExe -ArgumentList @("-p", $prefix, "-c", "conf/nginx.conf") -WorkingDirectory $prefix -WindowStyle Hidden
    Write-Host "nginx started with prefix $prefix"

    Write-Step "Verifying proxy (Host: $PublicHost)"
    Start-Sleep -Seconds 2
    $headers = @{
        Host          = $PublicHost
        Authorization = ("Basic " + [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("${AuthUser}:${AuthPassword}")))
    }
    try {
        # HTTP should redirect to HTTPS for internet vhost
        $pr = Invoke-WebRequest -Uri "http://127.0.0.1/" -Headers @{ Host = $PublicHost } -MaximumRedirection 0 -UseBasicParsing -TimeoutSec 15 -ErrorAction Stop
        Write-Host "HTTP proxy status $($pr.StatusCode)"
    } catch {
        if ($_.Exception.Response -and [int]$_.Exception.Response.StatusCode -in 301, 302) {
            Write-Host "HTTP->HTTPS redirect OK"
        } else {
            Write-Warning "HTTP proxy check: $_"
        }
    }
    try {
        add-type @"
using System.Net;
using System.Security.Cryptography.X509Certificates;
public class TrustAllCertsPolicy : ICertificatePolicy {
    public bool CheckValidationResult(ServicePoint sp, X509Certificate cert, WebRequest req, int problem) { return true; }
}
"@
        [System.Net.ServicePointManager]::CertificatePolicy = New-Object TrustAllCertsPolicy
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
        $https = Invoke-WebRequest -Uri "https://127.0.0.1/" -Headers $headers -UseBasicParsing -TimeoutSec 20
        Write-Host "HTTPS proxy OK (HTTP $($https.StatusCode))"
    } catch {
        if ($_.Exception.Response -and [int]$_.Exception.Response.StatusCode -in 401, 302) {
            Write-Host "HTTPS proxy reachable (auth challenge) -- OK"
        } else {
            Write-Warning "HTTPS proxy check: $_"
        }
    }
}
elseif ($Proxy -eq "iis") {
    Write-Warning "IIS helper was removed. Use nginx (default) or Cloudflare Tunnel — see deploy/SETUP.md"
    throw "IIS deploy is no longer supported in this repo path. Use -Proxy nginx."
}

if (-not $SkipFirewall) {
    Write-Step "Configuring Windows Firewall (append-only 80/443; WiFi rules untouched)"
    $fwOk = $true
    foreach ($rule in @(
            @{ Name = "LocalTranscript-HTTP-80"; Port = 80 },
            @{ Name = "LocalTranscript-HTTPS-443"; Port = 443 }
        )) {
        try {
            $existing = Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue
            if ($existing) {
                Set-NetFirewallRule -DisplayName $rule.Name -Enabled True -ErrorAction Stop
            } else {
                New-NetFirewallRule -DisplayName $rule.Name -Direction Inbound -Action Allow -Protocol TCP -LocalPort $rule.Port -ErrorAction Stop | Out-Null
            }
            Write-Host "Firewall allow TCP $($rule.Port)"
        } catch {
            $fwOk = $false
            Write-Warning "Firewall rule for port $($rule.Port) needs elevation: $_"
        }
    }
    Get-NetFirewallRule -DisplayName "LocalTranscript-7988*" -ErrorAction SilentlyContinue |
        Remove-NetFirewallRule -ErrorAction SilentlyContinue
    if (-not $fwOk) {
        $fwScript = Join-Path $RepoRoot "deploy\scripts\Open-PublicFirewall.ps1"
        Write-Warning "Re-run elevated: powershell -ExecutionPolicy Bypass -File `"$fwScript`""
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Public access ready ($Proxy / $Audience)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host " Share URL:  $publicBase"
Write-Host " Auth user:  $AuthUser"
Write-Host " Auth pass:  (see deploy\.public-credentials.txt)"
Write-Host " Upstream:   http://127.0.0.1:7988 (loopback only)"
Write-Host " WiFi IP hint (router forward target): $lanIp"
Write-Host " WiFi-safe: no adapter/DNS/route/ICS changes"
if ($Audience -eq "internet") {
    Write-Host " DNS: A record $PublicHost -> public WAN IP"
    Write-Host " Router: forward TCP 80+443 -> $lanIp (DHCP reservation recommended)"
}
Write-Host ""
