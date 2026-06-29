# scripts/show_phase_197i_defensive_exit_governance_sync_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197i_defensive_exit_governance_sync_validation.json"

Write-Host "=== TRITON Phase 197I Defensive Exit Governance Sync ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197i_defensive_exit_governance_sync_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Approval sync status: " + $r.approval_sync_status)
Write-Host ("Approval block reason: " + $r.approval_block_reason)
Write-Host ("Approval source: " + $r.approval_source)
Write-Host ("Preview exists: " + $r.preview_exists)
Write-Host ("Partial batch authorized: " + $r.partial_batch_authorized)
Write-Host ("Capped preview symbols: " + ($r.capped_preview_symbols -join ", "))
Write-Host ("Deferred symbols: " + ($r.deferred_symbols -join ", "))
Write-Host ("Capped notional: " + $r.capped_preview_notional)
Write-Host ("Configured cap: " + $r.configured_cap)
Write-Host ("Readiness: " + $r.readiness_status)
Write-Host ("Governance authorized: " + $r.governance_authorized)
Write-Host ("Operator authorized (global): " + $r.operator_authorized)
Write-Host ("Eligible for paper execution: " + $r.eligible_for_paper_execution)
Write-Host ("Paper entry gate ok: " + $r.paper_entry_gate_ok)
Write-Host ("Paper entry gate reason: " + $r.paper_entry_gate_reason)
Write-Host ("Placement maps exit rows: " + $r.placement_maps_exit_rows)
Write-Host ("Broker snapshot ok: " + $r.broker_snapshot_ok)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
