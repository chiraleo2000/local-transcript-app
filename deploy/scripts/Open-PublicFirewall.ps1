#Requires -RunAsAdministrator
# WiFi-safe: append-only inbound allow for nginx 80/443.
# Does not change WiFi adapter, DNS, routes, or firewall profiles.
# Does not expose Docker upstream 7988.

$ErrorActionPreference = "Stop"
foreach ($port in 80, 443) {
    $name = if ($port -eq 80) { "LocalTranscript-HTTP-80" } else { "LocalTranscript-HTTPS-443" }
    $existing = Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
    if ($existing) {
        Set-NetFirewallRule -DisplayName $name -Enabled True
    } else {
        New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port | Out-Null
    }
    Write-Host "Firewall allow TCP $port ($name)"
}
Get-NetFirewallRule -DisplayName "LocalTranscript-7988*" -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule -ErrorAction SilentlyContinue
Write-Host "Done. Port 7988 is not opened publicly. WiFi settings were not modified."
