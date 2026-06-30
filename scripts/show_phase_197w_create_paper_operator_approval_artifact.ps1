# scripts/show_phase_197w_create_paper_operator_approval_artifact.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197w_create_paper_operator_approval_artifact.json"

Write-Host "=== TRITON Phase 197W Paper Operator Approval Artifact ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197w_create_paper_operator_approval_artifact" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Final state: " + $r.final_state)
Write-Host ("Approval created: " + $r.approval_artifact_created)
Write-Host ("Approval path: " + $r.approval_artifact_path)
Write-Host ("Approval valid: " + $r.approval_artifact_valid)
Write-Host ("Approval status: " + $r.approval_status)
Write-Host ("Approval action: " + $r.approval_action)
Write-Host ("Approval scope: " + $r.approval_scope)
Write-Host ("Approved symbols: " + ($r.approved_symbols -join ", "))
Write-Host ("Deferred symbols: " + ($r.deferred_symbols -join ", "))
Write-Host ("Packet ID: " + $r.packet_id)
Write-Host ("Preview hash: " + $r.source_preview_hash)
Write-Host ("Batch notional: " + $r.batch_notional)
Write-Host ("Batch cap: " + $r.batch_notional_cap)
Write-Host ("Expires at: " + $r.expires_at)
Write-Host ("TTL minutes: " + $r.ttl_minutes)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("Governance state after: " + $r.governance_review_state_after)
Write-Host ("Paper entry dry-run after: " + $r.paper_entry_dry_run_status_after_approval)
Write-Host ("Paper entry reason after: " + $r.paper_entry_dry_run_reason_after_approval)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Exits executed: " + $r.exits_executed)
Write-Host ("Trims executed: " + $r.trims_executed)
Write-Host ""
Write-Host ("Warning: " + $r.warning)
Write-Host ("Next phase: " + $r.required_next_phase_for_execution)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
