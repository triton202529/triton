# scripts/show_phase_197z_readiness_bridge_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197z_readiness_bridge_validation.json"

Write-Host "=== TRITON Phase 197Z-RB Readiness Bridge Validation ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197z_readiness_bridge_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Bridge active: " + $r.bridge_active)
Write-Host ("Bridge state: " + $r.bridge_state)
Write-Host ("Baseline readiness: " + $r.baseline_readiness_status)
Write-Host ("Effective readiness: " + $r.effective_readiness_status)
Write-Host ("Execution readiness bridged: " + $r.execution_readiness_bridged)
Write-Host ("197W valid: " + $r.paper_approval_valid_current)
Write-Host ("197Z valid: " + $r.execution_authorization_valid_current)
Write-Host ("197Y gate ready: " + $r.'197y_gate_ready')
Write-Host ("197Y eligible: " + $r.'197y_eligible_for_execution')
Write-Host ("MasterExecutionGate: " + $r.master_execution_gate_status)
Write-Host ("Dispatch interlock: " + $r.dispatch_interlock_state)
Write-Host ("Dispatch reason: " + $r.dispatch_interlock_reason)
Write-Host ("Dispatch permitted: " + $r.dispatch_permitted)
Write-Host ("Blocked by readiness: " + $r.dispatch_blocked_readiness)
Write-Host ("Approved scope: " + ($r.approved_scope_symbols -join ", "))
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Open orders: " + $r.open_orders)
Write-Host ("Positions written: " + $r.positions_written)

Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
