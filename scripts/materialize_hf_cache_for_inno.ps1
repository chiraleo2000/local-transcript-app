param(
    [string]$SourceHub = "models\hf_cache\hub",
    [string]$DestinationRoot = "C:\lta-installer-stage-real"
)

$ErrorActionPreference = "Stop"

$sourceHubPath = (Resolve-Path $SourceHub).Path
$destinationHubPath = Join-Path $DestinationRoot "models\hf_cache\hub"

if (Test-Path $DestinationRoot) {
    Remove-Item $DestinationRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $destinationHubPath | Out-Null

function Get-ReparseTargetText {
    param([string]$Path)

    $bytes = New-Object System.Collections.Generic.List[byte]
    $lines = fsutil reparsepoint query $Path 2>$null
    foreach ($line in $lines) {
        if ($line -match '^\s*[0-9a-fA-F]{4}:\s+(?<hex>(?:[0-9a-fA-F]{2}\s+){1,16})') {
            foreach ($hexByte in ($Matches['hex'] -split '\s+' | Where-Object { $_ })) {
                $bytes.Add([Convert]::ToByte($hexByte, 16))
            }
        }
    }
    if ($bytes.Count -le 4) {
        throw "Could not parse reparse target: $Path"
    }
    [Text.Encoding]::ASCII.GetString($bytes.ToArray(), 4, $bytes.Count - 4)
}

function Copy-ResolvedTree {
    param(
        [string]$Source,
        [string]$Destination
    )

    $item = Get-Item -LiteralPath $Source -Force
    $isReparsePoint = (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)

    if ($item.PSIsContainer -and -not $isReparsePoint) {
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
        foreach ($child in Get-ChildItem -LiteralPath $Source -Force) {
            Copy-ResolvedTree -Source $child.FullName -Destination (Join-Path $Destination $child.Name)
        }
        return
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    if ($isReparsePoint) {
        $relativeTarget = (Get-ReparseTargetText -Path $Source).Replace('/', '\')
        $target = [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $Source) $relativeTarget))
        if (!(Test-Path -LiteralPath $target)) {
            throw "Missing reparse target for $Source -> $target"
        }
        Copy-Item -LiteralPath $target -Destination $Destination -Force
    } else {
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

foreach ($repo in Get-ChildItem -LiteralPath $sourceHubPath -Force -Directory) {
    if ($repo.Name -eq ".locks") {
        continue
    }

    $destinationRepo = Join-Path $destinationHubPath $repo.Name
    New-Item -ItemType Directory -Force -Path $destinationRepo | Out-Null

    foreach ($child in Get-ChildItem -LiteralPath $repo.FullName -Force) {
        if ($child.Name -eq "blobs") {
            continue
        }
        Copy-ResolvedTree -Source $child.FullName -Destination (Join-Path $destinationRepo $child.Name)
    }
}

$totalBytes = (Get-ChildItem (Join-Path $DestinationRoot "models") -Recurse -ErrorAction SilentlyContinue |
    Measure-Object Length -Sum).Sum
Write-Host ("Portable model stage: {0:N2} GB at {1}" -f ($totalBytes / 1GB), $DestinationRoot)