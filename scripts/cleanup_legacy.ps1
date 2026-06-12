# Remove legacy release binaries and duplicate docs (dry-run by default).
param(
    [switch]$Apply
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Remove-LegacyPath {
    param([string]$Target)
    if (-not (Test-Path $Target)) { return }
    if (-not $Apply) {
        Write-Host "[dry-run] would remove: $Target"
        return
    }
    Remove-Item -Recurse -Force $Target
    Write-Host "removed: $Target"
}

Remove-LegacyPath "release/v1.2.1"
Get-ChildItem "release/v1.2.2" -Filter *.exe -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-LegacyPath $_.FullName
}
Get-ChildItem "release/v1.2.2" -Filter *.zip -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-LegacyPath $_.FullName
}
Remove-LegacyPath "RELEASE_NOTES_v1.2.0.md"
Remove-LegacyPath "RELEASE_NOTES_v1.2.1.md"

if (-not $Apply) {
    Write-Host "Dry run complete. Re-run with -Apply to delete."
}
