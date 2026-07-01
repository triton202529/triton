# scripts/show_phase_197y_loop_mediated_paper_execution_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197y_loop_mediated_paper_execution_validation.json"
$gatePath = Join-Path $RepoRoot "data\results\phase_197y_loop_mediated_paper_execution_gate.json"

Write-Host "=== TRITON Phase 197Y Loop-Mediated Paper Execution Validation ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197y_loop_mediated_paper_execution_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Final state: " + $r.final_state)
Write-Host ("Gate reason: " + $r.gate_reason)
Write-Host ("Execution auth exists: " + $r.execution_authorization_artifact_exists)
Write-Host ("Execution auth state: " + $r.execution_authorization_state)
Write-Host ("Paper approval state: " + $r.paper_approval_state)
Write-Host ("manage_positions exit: " + $r.manage_positions_exit_code)
Write-Host ("Dispatch interlock: " + $r.dispatch_interlock_state)
Write-Host ("Dispatch reason: " + $r.dispatch_interlock_reason)
Write-Host ("Safe block: " + $r.safe_block)
Write-Host ("Safe block state: " + $r.safe_block_state)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Exits executed: " + $r.exits_executed)
Write-Host ("Trims executed: " + $r.trims_executed)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("Operator execution required: " + $r.operator_execution_required)
Write-Host ("Open orders: " + $r.open_orders)
Write-Host ("Positions written: " + $r.positions_written)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Paper entry status: " + $r.paper_entry_dry_run_status)
Write-Host ("Paper entry reason: " + $r.paper_entry_dry_run_reason)

if (Test-Path -LiteralPath $gatePath -PathType Leaf) {
    Write-Host ""
    Write-Host "Gate artifact:" -ForegroundColor DarkGray
    Write-Host ("  " + $gatePath)
}

Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
