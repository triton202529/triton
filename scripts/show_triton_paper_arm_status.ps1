# scripts/show_triton_paper_arm_status.ps1
# Phase 196Q - Read-only paper ARM / operator latch status (no writes).

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    $py = "python"
}

Write-Host "=== TRITON Paper ARM Status (Phase 196Q) ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "READ-ONLY status display (session policy may auto-renew arm during market hours)" -ForegroundColor Yellow
Write-Host ""

$jsonOut = & $py -m services.paper_arm_status --json 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to query paper arm status." -ForegroundColor Red
    exit 1
}
if (-not $jsonOut) {
    Write-Host "Empty response from paper_arm_status." -ForegroundColor Red
    exit 1
}

try {
    $st = ($jsonOut | Out-String).Trim() | ConvertFrom-Json
} catch {
    Write-Host "Invalid JSON from paper_arm_status:" -ForegroundColor Red
    Write-Host $jsonOut
    exit 1
}

Write-Host ("ARM mode: " + $st.mode)
Write-Host ("ARM config: " + $st.arm_config_path)
Write-Host ("Paper arm confirm path: " + $st.paper_arm_confirm_path)
Write-Host ("Confirmation exists: " + $st.exists)
Write-Host ("Status: " + $st.status)
if ($st.timestamp) { Write-Host ("Timestamp: " + $st.timestamp) }
if ($st.expires_at) { Write-Host ("Expires at: " + $st.expires_at) }
if ($null -ne $st.minutes_remaining) {
    Write-Host ("Minutes remaining: " + $st.minutes_remaining)
} elseif ($st.status -eq "EXPIRED") {
    Write-Host "Minutes remaining: 0 (expired)"
}
Write-Host ("Operator latch required: " + $st.operator_required)
Write-Host ("Paper entry armed: " + $st.paper_entry_armed)
Write-Host ("Live execution blocked: " + $st.live_execution_blocked)
Write-Host ""
Write-Host "--- Live confirm (separate from paper arm) ---" -ForegroundColor Cyan
Write-Host ("Live confirm path: " + $st.live_confirm_path)
Write-Host ("Live confirm status: " + $st.live_confirm_status)
Write-Host ""
if ($st.arm_block_reasons -and $st.arm_block_reasons.Count -gt 0) {
    Write-Host ("ARM block reasons: " + ($st.arm_block_reasons -join ", "))
}
if ($st.notes) {
    Write-Host ""
    Write-Host "--- Notes ---" -ForegroundColor Cyan
    foreach ($n in $st.notes) {
        Write-Host ("  " + $n)
    }
}
Write-Host ""
Write-Host "To arm paper (operator only, after explicit approval):" -ForegroundColor DarkGray
Write-Host "  python -m services.set_paper_arm_confirm --ttl MINUTES" -ForegroundColor DarkGray
Write-Host "This is NOT live execution and NOT a broker order." -ForegroundColor DarkGray
Write-Host ""
Write-Host "=== End status ===" -ForegroundColor Cyan

exit 0
