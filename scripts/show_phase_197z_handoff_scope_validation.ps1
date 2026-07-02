# scripts/show_phase_197z_handoff_scope_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197z_handoff_scope_validation.json"

Write-Host "=== TRITON Phase 197Z-HS Handoff Scope Validation ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197z_handoff_scope_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Bridge active: " + $r.bridge_active)
Write-Host ("Scope valid: " + $r.scope_valid)
Write-Host ("Pre handoff symbols: " + ($r.pre_handoff_symbols -join ", "))
Write-Host ("Excluded symbols: " + ($r.excluded_from_handoff_symbols -join ", "))
Write-Host ("Final handoff symbols: " + ($r.final_handoff_symbols -join ", "))
Write-Host ("Final handoff notional: " + $r.final_handoff_notional)
Write-Host ("Projected 197X notional: " + $r.projected_197x_notional)
Write-Host ("Batch cap: " + $r.batch_notional_cap)
Write-Host ("Cap passed: " + $r.cap_passed)
Write-Host ("Outside scope reached placement: " + $r.outside_scope_rows_reached_placement)
Write-Host ("Dispatch interlock: " + $r.dispatch_interlock_state)
Write-Host ("Dispatch permitted: " + $r.dispatch_permitted)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Open orders written: " + $r.open_orders_written)

Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
