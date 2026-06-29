# scripts/show_phase_197m_fix_live_block_detection_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197m_fix_live_block_detection_validation.json"

Write-Host "=== TRITON Phase 197M-FIX Live Block Detection ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197m_fix_live_block_detection_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Live block reason: " + $r.live_block_reason)
Write-Host ("Stale live confirm count: " + $r.stale_live_confirm_count)
Write-Host ("Valid live confirm count: " + $r.valid_live_confirm_count)
Write-Host ("Paper entry gate status: " + $r.paper_entry_gate_status)
Write-Host ("Paper entry gate reason: " + $r.paper_entry_gate_reason)
Write-Host ("Market open: " + $r.market_open)
Write-Host ("Readiness: " + $r.readiness_status)
Write-Host ("Governance authorized: " + $r.governance_authorized)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Broker snapshot ok: " + $r.broker_snapshot_ok)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Positions: " + $r.positions_count)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ""
Write-Host "Live confirm paths:" -ForegroundColor DarkGray
foreach ($p in $r.live_confirm_paths_found) {
    Write-Host ("  " + $p)
}
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
