# scripts/show_phase_197s_fix_dispatch_interlock_safe_block_exit_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197s_fix_dispatch_interlock_safe_block_exit_validation.json"

Write-Host "=== TRITON Phase 197S-FIX3 Safe-Block Exit Validation ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197s_fix_dispatch_interlock_safe_block_exit_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Safe block state: " + $r.safe_block_state)
Write-Host ("Manage positions expected exit code: " + $r.manage_positions_expected_exit_code)
Write-Host ("Market sequence expected outcome: " + $r.market_sequence_expected_outcome)
Write-Host ("Continuous cycle expected outcome: " + $r.continuous_cycle_expected_outcome)
Write-Host ("Dispatch interlock state: " + $r.dispatch_interlock_state)
Write-Host ("Dispatch interlock stage: " + $r.dispatch_interlock_stage)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Exits executed: " + $r.exits_executed)
Write-Host ("Trims executed: " + $r.trims_executed)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("Latest market open log: " + $r.latest_market_open_log)
Write-Host ("Latest continuous log: " + $r.latest_continuous_log)
Write-Host ("Latest runtime had abort: " + $r.latest_runtime_had_abort)
Write-Host ("Latest runtime abort signature: " + $r.latest_runtime_abort_signature)
Write-Host ("Safe block marker required: " + $r.safe_block_marker_required)
Write-Host ("Safe block marker present: " + $r.safe_block_marker_present)
Write-Host ("Pre-fix abort evidence: " + $r.pre_fix_abort_evidence)
Write-Host ("Continuous cycle fail: " + $r.continuous_cycle_fail)
Write-Host ("Continuous cycle ok: " + $r.continuous_cycle_ok)
Write-Host ("Static safe-block exit ok: " + $r.static_safe_block_exit_ok)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
