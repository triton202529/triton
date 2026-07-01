# scripts/show_phase_197x_paper_broker_execution_final_dry_run.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197x_paper_broker_execution_final_dry_run.json"
$csvPath = Join-Path $RepoRoot "data\results\phase_197x_projected_paper_orders.csv"

Write-Host "=== TRITON Phase 197X Paper Broker Execution Final Dry-Run ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197x_paper_broker_execution_final_dry_run" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Final state: " + $r.final_state)
Write-Host ("Approval valid: " + $r.approval_artifact_valid)
Write-Host ("Approval state: " + $r.approval_artifact_state)
Write-Host ("Approval packet: " + $r.approval_packet_id)
Write-Host ("Approval expires: " + $r.approval_expires_at)
Write-Host ("Approved symbols: " + ($r.approved_symbols -join ", "))
Write-Host ("Deferred symbols: " + ($r.deferred_symbols -join ", "))
Write-Host ("Projected orders: " + $r.projected_order_count)
Write-Host ("Batch notional estimate: " + $r.batch_notional_estimate)
Write-Host ("Open orders: " + $r.open_orders)
Write-Host ("Positions count: " + $r.positions_count)
Write-Host ("Market open: " + $r.market_open)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Exits executed: " + $r.exits_executed)
Write-Host ("Trims executed: " + $r.trims_executed)
Write-Host ("Operator execution required: " + $r.operator_execution_required)
Write-Host ("Projected CSV: " + $r.projected_orders_path)
if ($r.block_reason) {
    Write-Host ("Block reason: " + $r.block_reason)
}
if (Test-Path -LiteralPath $csvPath -PathType Leaf) {
    Write-Host ""
    Write-Host "Projected paper orders:" -ForegroundColor DarkGray
    Import-Csv -LiteralPath $csvPath | Format-Table symbol, side, quantity, notional_estimate, reason -AutoSize
}
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
