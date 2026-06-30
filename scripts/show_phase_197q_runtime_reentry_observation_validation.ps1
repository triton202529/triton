# scripts/show_phase_197q_runtime_reentry_observation_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197q_runtime_reentry_observation_validation.json"

Write-Host "=== TRITON Phase 197Q Runtime Re-Entry Observation ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197q_runtime_reentry_observation_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Runtime re-entry state: " + $r.runtime_reentry_state)
Write-Host ("Runtime re-entry reason: " + $r.runtime_reentry_reason)
Write-Host ("Continuous loop running: " + $r.continuous_loop_running)
Write-Host ("Watchdog loop running: " + $r.watchdog_loop_running)
Write-Host ("Continuous lock exists: " + $r.continuous_lock_exists)
Write-Host ("Watchdog lock exists: " + $r.watchdog_lock_exists)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Live block reason: " + $r.live_block_reason)
Write-Host ("Order guard authority state: " + $r.order_guard_authority_state)
Write-Host ("Broker snapshot OK: " + $r.broker_snapshot_ok)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Positions: " + $r.positions_count)
Write-Host ("Market open: " + $r.market_open)
Write-Host ("Market hours recheck state: " + $r.market_hours_recheck_state)
Write-Host ("Governance review state: " + $r.governance_review_state)
Write-Host ("Governance authorized: " + $r.governance_authorized)
Write-Host ("Readiness: " + $r.readiness_status)
Write-Host ("Paper entry gate status: " + $r.paper_entry_gate_status)
Write-Host ("Paper entry gate reason: " + $r.paper_entry_gate_reason)
Write-Host ("Operator approval status: " + $r.operator_approval_status)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Capped preview exists: " + $r.capped_preview_exists)
Write-Host ("Capped symbols: " + ($r.capped_symbols -join ", "))
Write-Host ("Deferred symbols: " + ($r.deferred_symbols -join ", "))
Write-Host ("Eligible for operator decision: " + $r.eligible_for_operator_decision)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Restart recommendation: " + $r.restart_recommendation)
Write-Host ("Restart command: " + $r.restart_command)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
