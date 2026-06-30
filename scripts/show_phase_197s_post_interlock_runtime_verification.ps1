# scripts/show_phase_197s_post_interlock_runtime_verification.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197s_post_interlock_runtime_verification.json"

Write-Host "=== TRITON Phase 197S Post-Interlock Runtime Verification ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197s_post_interlock_runtime_verification" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Post-interlock runtime state: " + $r.post_interlock_runtime_state)
Write-Host ("Post-interlock runtime reason: " + $r.post_interlock_runtime_reason)
Write-Host ("Continuous loop running: " + $r.continuous_loop_running)
Write-Host ("Watchdog loop running: " + $r.watchdog_loop_running)
Write-Host ("Latest continuous log: " + $r.latest_continuous_log)
Write-Host ("Latest market open log: " + $r.latest_market_open_log)
Write-Host ("Cycle start: " + $r.cycle_start_timestamp)
Write-Host ("Cycle completed: " + $r.cycle_completed)
Write-Host ("Dispatch interlock seen: " + $r.dispatch_interlock_seen)
Write-Host ("Dispatch interlock state: " + $r.dispatch_interlock_state)
Write-Host ("Dispatch interlock stage: " + $r.dispatch_interlock_stage)
Write-Host ("Safe block marker present: " + $r.safe_block_marker_present)
Write-Host ("Market sequence abort: " + $r.market_sequence_abort)
Write-Host ("Market sequence complete: " + $r.market_sequence_complete)
Write-Host ("Placement after interlock: " + $r.placement_proceeded_after_interlock)
Write-Host ("Notional cap after interlock: " + $r.notional_cap_block_seen_after_interlock)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Exits executed: " + $r.exits_executed)
Write-Host ("Trims executed: " + $r.trims_executed)
Write-Host ("Paper entry gate status: " + $r.paper_entry_gate_status)
Write-Host ("Paper entry gate reason: " + $r.paper_entry_gate_reason)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Live block reason: " + $r.live_block_reason)
Write-Host ("Order guard state: " + $r.order_guard_authority_state)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("Restart required: " + $r.restart_required)
Write-Host ("Observation command: " + $r.observation_command)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
