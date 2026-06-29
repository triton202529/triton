# scripts/show_phase_197o_market_hours_governance_recheck_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197o_market_hours_governance_recheck_validation.json"

Write-Host "=== TRITON Phase 197O Market-Hours Governance Recheck ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197o_market_hours_governance_recheck_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Market hours recheck state: " + $r.market_hours_recheck_state)
Write-Host ("Market hours recheck reason: " + $r.market_hours_recheck_reason)
Write-Host ("Market open: " + $r.market_open)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Live block reason: " + $r.live_block_reason)
Write-Host ("Governance authorized: " + $r.governance_authorized)
Write-Host ("Governance review state: " + $r.governance_review_state)
Write-Host ("Readiness: " + $r.readiness_status)
Write-Host ("Paper entry gate status: " + $r.paper_entry_gate_status)
Write-Host ("Paper entry gate reason: " + $r.paper_entry_gate_reason)
Write-Host ("Master execution gate: " + $r.master_execution_gate_status)
Write-Host ("Guard snapshot age (min): " + $r.guard_snapshot_age_minutes)
Write-Host ("Guard snapshot fresh: " + $r.guard_snapshot_fresh)
Write-Host ("Open orders snapshot fresh: " + $r.open_orders_snapshot_fresh)
Write-Host ("Gate freshness note: " + $r.gate_freshness_note)
Write-Host ("Capped preview exists: " + $r.capped_preview_exists)
Write-Host ("Capped symbols: " + ($r.capped_symbols -join ", "))
Write-Host ("Deferred symbols: " + ($r.deferred_symbols -join ", "))
Write-Host ("Eligible for operator decision: " + $r.eligible_for_operator_decision)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Positions: " + $r.positions_count)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
