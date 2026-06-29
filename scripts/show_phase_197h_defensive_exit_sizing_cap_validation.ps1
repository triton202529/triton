# scripts/show_phase_197h_defensive_exit_sizing_cap_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197h_defensive_exit_sizing_cap_validation.json"

Write-Host "=== TRITON Phase 197H Defensive Exit Sizing Cap ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197h_defensive_exit_sizing_cap_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Root cause: " + $r.root_cause)
Write-Host ("Fix: " + $r.fix)
Write-Host ("Handoff rows: " + $r.handoff_rows)
Write-Host ("Placement maps exit rows: " + $r.placement_maps_exit_rows)
Write-Host ("Placement block reason: " + $r.placement_block_reason)
Write-Host ("Handoff status: " + $r.handoff_status)
Write-Host ("Batch notional estimate: " + $r.batch_notional_estimate)
Write-Host ("Configured cap (CLI dry-run): " + $r.configured_batch_cap_cli_default)
Write-Host ("Configured cap (manage_positions): " + $r.configured_batch_cap_manage_positions)
Write-Host ("Cap excess: " + $r.cap_excess_amount)
Write-Host ("Capped preview symbols: " + ($r.capped_preview_symbols -join ", "))
Write-Host ("Deferred symbols: " + ($r.deferred_symbols -join ", "))
Write-Host ("Partial batch authorized: " + $r.partial_batch_authorized)
Write-Host ("Partial batch status: " + $r.partial_batch_status)
Write-Host ("Broker snapshot ok: " + $r.broker_snapshot_ok)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Readiness: " + $r.readiness_status)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
