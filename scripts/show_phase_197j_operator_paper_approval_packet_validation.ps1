# scripts/show_phase_197j_operator_paper_approval_packet_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197j_operator_paper_approval_packet_validation.json"

Write-Host "=== TRITON Phase 197J Operator Paper Approval Packet ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197j_operator_paper_approval_packet_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Packet created: " + $r.packet_created)
Write-Host ("Packet ID: " + $r.packet_id)
Write-Host ("Packet expires at: " + $r.packet_expires_at)
Write-Host ("Approval contract path: " + $r.approval_contract_path)
Write-Host ("Approval exists: " + $r.approval_exists)
Write-Host ("Approval status: " + $r.approval_status)
Write-Host ("Approval validation status: " + $r.approval_validation_status)
Write-Host ("Approval validation reason: " + $r.approval_validation_reason)
Write-Host ("Governance sync status: " + $r.governance_sync_status)
Write-Host ("Capped symbols: " + ($r.capped_symbols -join ", "))
Write-Host ("Deferred symbols: " + ($r.deferred_symbols -join ", "))
Write-Host ("Capped notional: " + $r.capped_notional)
Write-Host ("Configured cap: " + $r.configured_cap)
Write-Host ("Readiness: " + $r.readiness_status)
Write-Host ("Governance authorized: " + $r.governance_authorized)
Write-Host ("Eligible for paper execution: " + $r.eligible_for_paper_execution)
Write-Host ("Paper entry gate ok: " + $r.paper_entry_gate_ok)
Write-Host ("Paper entry gate reason: " + $r.paper_entry_gate_reason)
Write-Host ("Broker snapshot ok: " + $r.broker_snapshot_ok)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Positions: " + $r.positions_count)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
