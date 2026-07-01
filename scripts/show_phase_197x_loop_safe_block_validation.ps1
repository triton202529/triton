# scripts/show_phase_197x_loop_safe_block_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197x_loop_safe_block_validation.json"

Write-Host "=== TRITON Phase 197X-LOOP-FIX Safe-Block Validation ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197x_loop_safe_block_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Approval valid: " + $r.approval_artifact_valid)
Write-Host ("Approval state: " + $r.approval_artifact_state)
Write-Host ("manage_positions exit: " + $r.manage_positions_exit_code)
Write-Host ("Dispatch interlock: " + $r.dispatch_interlock_state)
Write-Host ("Dispatch reason: " + $r.dispatch_interlock_reason)
Write-Host ("Safe block: " + $r.safe_block)
Write-Host ("Safe block state: " + $r.safe_block_state)
Write-Host ("Scope source: " + $r.paper_scope_source)
Write-Host ("Scope symbols: " + ($r.paper_scope_symbols -join ", "))
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Exits executed: " + $r.exits_executed)
Write-Host ("Trims executed: " + $r.trims_executed)
Write-Host ("Operator execution required: " + $r.operator_execution_required)
Write-Host ("Cycle should continue: " + $r.cycle_should_continue)
Write-Host ("Open orders: " + $r.open_orders)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Paper entry status: " + $r.paper_entry_dry_run_status)
Write-Host ("Paper entry reason: " + $r.paper_entry_dry_run_reason)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
