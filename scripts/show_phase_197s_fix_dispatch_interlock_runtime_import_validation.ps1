# scripts/show_phase_197s_fix_dispatch_interlock_runtime_import_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197s_fix_dispatch_interlock_runtime_import_validation.json"

Write-Host "=== TRITON Phase 197S-FIX2 Dispatch Interlock Import Repair ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197s_fix_dispatch_interlock_runtime_import_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Import check manage_positions: " + $r.import_check_manage_positions)
Write-Host ("Import check place_live_orders: " + $r.import_check_place_live_orders)
Write-Host ("Interlock module import OK: " + $r.interlock_module_import_ok)
Write-Host ("is_defensive_exit_manage_handoff available: " + $r.is_defensive_exit_manage_handoff_available)
Write-Host ("dispatch interlock function available: " + $r.dispatch_interlock_function_available)
Write-Host ("manage_positions static reference OK: " + $r.manage_positions_static_reference_ok)
Write-Host ("place_live_orders static reference OK: " + $r.place_live_orders_static_reference_ok)
Write-Host ("Latest runtime crash detected: " + $r.latest_runtime_crash_detected)
Write-Host ("Latest runtime crash signature: " + $r.latest_runtime_crash_signature)
Write-Host ("Crash repaired: " + $r.crash_repaired)
Write-Host ("Simulated interlock state: " + $r.simulated_missing_approval_interlock_state)
Write-Host ("Simulated marker: " + $r.simulated_missing_approval_marker)
Write-Host ("Latest market open log: " + $r.latest_market_open_log)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
