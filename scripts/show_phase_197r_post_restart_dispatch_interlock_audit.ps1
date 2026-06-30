# scripts/show_phase_197r_post_restart_dispatch_interlock_audit.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197r_post_restart_dispatch_interlock_audit.json"

Write-Host "=== TRITON Phase 197R Post-Restart Dispatch Interlock Audit ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197r_post_restart_dispatch_interlock_audit" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Continuous loop seen: " + $r.continuous_loop_seen)
Write-Host ("Watchdog running: " + $r.watchdog_loop_running)
Write-Host ("Continuous running: " + $r.continuous_loop_running)
Write-Host ("Latest continuous log: " + $r.latest_continuous_log)
Write-Host ("Latest market open log: " + $r.latest_market_open_log)
Write-Host ("Cycle completed: " + $r.cycle_completed)
Write-Host ("Manage positions dry_run: " + $r.manage_positions_dry_run)
Write-Host ("Placement dry_run: " + $r.placement_dry_run)
Write-Host ("Mapped exit rows: " + $r.mapped_exit_rows)
Write-Host ("Batch notional status: " + $r.batch_notional_status)
Write-Host ("Batch notional: " + $r.batch_notional)
Write-Host ("Batch notional cap: " + $r.batch_notional_cap)
Write-Host ("Exits executed: " + $r.exits_executed)
Write-Host ("Trims executed: " + $r.trims_executed)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Dispatch interlock state: " + $r.dispatch_interlock_state)
Write-Host ("Dispatch interlock reason: " + $r.dispatch_interlock_reason)
Write-Host ("Dispatch interlock in log: " + $r.dispatch_interlock_seen_in_log)
Write-Host ("Paper entry gate status: " + $r.paper_entry_gate_status)
Write-Host ("Paper entry gate reason: " + $r.paper_entry_gate_reason)
Write-Host ("Operator approval status: " + $r.operator_approval_status)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Governance authorized: " + $r.governance_authorized)
Write-Host ("Readiness: " + $r.readiness_status)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Order guard state: " + $r.order_guard_authority_state)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("No execution confirmed: " + $r.no_execution_confirmed)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
