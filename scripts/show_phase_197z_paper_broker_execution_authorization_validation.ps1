# scripts/show_phase_197z_paper_broker_execution_authorization_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197z_paper_broker_execution_authorization_validation.json"
$creatorPath = Join-Path $RepoRoot "data\results\phase_197z_paper_broker_execution_authorization_creator.json"

Write-Host "=== TRITON Phase 197Z Paper Broker Execution Authorization Validation ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197z_paper_broker_execution_authorization_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Final state: " + $r.final_state)
Write-Host ("Creator status: " + $r.creator_status)
Write-Host ("Would create authorization: " + $r.would_create_authorization)
Write-Host ("Real authorization created: " + $r.real_authorization_created)
Write-Host ("Execution auth artifact exists: " + $r.execution_authorization_artifact_exists)
Write-Host ("Execution auth state: " + $r.execution_authorization_state)
Write-Host ("Paper approval state: " + $r.paper_approval_state)
Write-Host ("197Y gate final state: " + $r.'197y_gate_final_state')
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Exits executed: " + $r.exits_executed)
Write-Host ("Trims executed: " + $r.trims_executed)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Open orders: " + $r.open_orders)
Write-Host ("Positions written: " + $r.positions_written)
Write-Host ("Paper entry status: " + $r.paper_entry_dry_run_status)
Write-Host ("Paper entry reason: " + $r.paper_entry_dry_run_reason)

if (Test-Path -LiteralPath $creatorPath -PathType Leaf) {
    Write-Host ""
    Write-Host "Creator artifact:" -ForegroundColor DarkGray
    Write-Host ("  " + $creatorPath)
}

Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
