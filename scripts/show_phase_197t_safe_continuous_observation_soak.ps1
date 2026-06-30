# scripts/show_phase_197t_safe_continuous_observation_soak.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197t_safe_continuous_observation_soak.json"

Write-Host "=== TRITON Phase 197T Safe Continuous Observation Soak ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197t_safe_continuous_observation_soak" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Soak state: " + $r.soak_state)
Write-Host ("Soak reason: " + $r.soak_reason)
Write-Host ("Cycles requested: " + $r.cycles_requested)
Write-Host ("Cycles inspected: " + $r.cycles_inspected)
Write-Host ("Watchdog running: " + $r.watchdog_loop_running)
Write-Host ("Continuous running: " + $r.continuous_loop_running)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Guard snapshot refreshed: " + $r.guard_snapshot_refreshed)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Live block reason: " + $r.live_block_reason)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("Paper entry gate: " + $r.paper_entry_gate_status)
Write-Host ("Paper entry reason: " + $r.paper_entry_gate_reason)
Write-Host ("Order guard state: " + $r.order_guard_authority_state)
Write-Host ""

if ($r.cycles -and $r.cycles.Count -gt 0) {
    Write-Host "=== Inspected cycles ===" -ForegroundColor Cyan
    foreach ($c in $r.cycles) {
        Write-Host ""
        Write-Host ("Cycle " + $c.cycle_number + " @ " + $c.cycle_start_timestamp)
        Write-Host ("  Continuous log: " + $c.continuous_log_path)
        Write-Host ("  Market open log: " + $c.market_open_log_path)
        Write-Host ("  Cycle OK: " + $c.cycle_ok)
        Write-Host ("  Cycle fail: " + $c.cycle_fail)
        Write-Host ("  Market sequence complete: " + $c.market_sequence_complete)
        Write-Host ("  Market sequence abort: " + $c.market_sequence_abort)
        Write-Host ("  Dispatch interlock: " + $c.dispatch_interlock_state)
        Write-Host ("  Safe block: " + $c.safe_block_state)
        Write-Host ("  Broker orders submitted: " + $c.broker_orders_submitted)
        Write-Host ("  Exits executed: " + $c.exits_executed)
        Write-Host ("  Trims executed: " + $c.trims_executed)
        Write-Host ("  Paper entry gate: " + $c.paper_entry_gate_status)
        Write-Host ("  Post-197S safe-block cycle: " + $c.post_197s_safe_block_cycle)
    }
}

Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
