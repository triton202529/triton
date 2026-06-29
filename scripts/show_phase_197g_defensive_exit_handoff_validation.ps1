# scripts/show_phase_197g_defensive_exit_handoff_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197g_defensive_exit_handoff_validation.json"

Write-Host "=== TRITON Phase 197G Defensive Exit Handoff ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197g_defensive_exit_handoff_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Root cause: " + $r.root_cause)
Write-Host ("Fix: " + $r.fix)
Write-Host ("Broker snapshot ok: " + $r.broker_snapshot_ok)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Planned exit count: " + $r.planned_exit_count)
Write-Host ("Planned exit symbols: " + ($r.planned_exit_symbols -join ", "))
Write-Host ("Manage handoff rows: " + $r.manage_orders_rows)
Write-Host ("Handoff path: " + $r.manage_orders_handoff_path)
Write-Host ("Handoff status: " + $r.handoff_status)
Write-Host ("Handoff block reason: " + $r.handoff_block_reason)
Write-Host ("Batch empty explained: " + $r.batch_empty_explained)
Write-Host ("Readiness status: " + $r.readiness_status)
Write-Host ("Placement dry-run ok: " + $r.placement_dry_run_ok)
Write-Host ("Placement maps exit rows: " + $r.placement_maps_exit_rows)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Git branch: " + $r.git_branch)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
