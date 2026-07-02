# scripts/show_phase_197z_batch_notional_scope_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197z_batch_notional_scope_validation.json"

Write-Host "=== TRITON Phase 197Z-CAP Batch Notional Scope Validation ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197z_batch_notional_scope_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Bridge active: " + $r.bridge_active)
Write-Host ("197Y gate ready: " + $r.'197y_gate_ready')
Write-Host ("Pre-scope planned notional: " + $r.pre_scope_planned_notional)
Write-Host ("Excluded outside 197X notional: " + $r.excluded_outside_197x_scope_notional)
Write-Host ("Approved scope notional: " + $r.approved_scope_notional)
Write-Host ("Final cap-checked notional: " + $r.final_cap_checked_notional)
Write-Host ("Batch notional cap: " + $r.batch_notional_cap)
Write-Host ("Cap passed: " + $r.cap_passed)
Write-Host ("Cap basis: " + $r.cap_basis)
Write-Host ("Excluded symbols: " + ($r.excluded_symbols -join ", "))
Write-Host ("Approved scope: " + ($r.approved_scope_symbols -join ", "))
Write-Host ("Dispatch interlock: " + $r.dispatch_interlock_state)
Write-Host ("Dispatch reason: " + $r.dispatch_interlock_reason)
Write-Host ("Dispatch permitted: " + $r.dispatch_permitted)
Write-Host ("Blocked by batch cap: " + $r.dispatch_blocked_batch_notional_cap)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Open orders: " + $r.open_orders)

Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
