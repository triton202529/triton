# scripts/show_phase_197n_governance_review_criteria_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197n_governance_review_criteria_validation.json"

Write-Host "=== TRITON Phase 197N Governance Review Criteria ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197n_governance_review_criteria_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Governance review state: " + $r.governance_review_state)
Write-Host ("Governance review reason: " + $r.governance_review_reason)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Live block reason: " + $r.live_block_reason)
Write-Host ("Paper mode: " + $r.paper_mode)
Write-Host ("Market open: " + $r.market_open)
Write-Host ("Readiness: " + $r.readiness_status)
Write-Host ("Management review required: " + $r.management_review_required)
Write-Host ("Capped preview exists: " + $r.capped_preview_exists)
Write-Host ("Capped symbols: " + ($r.capped_symbols -join ", "))
Write-Host ("Deferred symbols: " + ($r.deferred_symbols -join ", "))
Write-Host ("Capped notional: " + $r.capped_notional)
Write-Host ("Configured cap: " + $r.configured_cap)
Write-Host ("Notional cap status: " + $r.notional_cap_status)
Write-Host ("Operator packet exists: " + $r.operator_packet_exists)
Write-Host ("Operator packet current: " + $r.operator_packet_current)
Write-Host ("Source preview hash: " + $r.source_preview_hash)
Write-Host ("Evidence complete: " + $r.evidence_complete)
Write-Host ("Missing evidence: " + ($r.missing_evidence -join ", "))
Write-Host ("Eligible for operator decision: " + $r.eligible_for_operator_decision)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("Paper entry gate status: " + $r.paper_entry_gate_status)
Write-Host ("Paper entry gate reason: " + $r.paper_entry_gate_reason)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Positions: " + $r.positions_count)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
