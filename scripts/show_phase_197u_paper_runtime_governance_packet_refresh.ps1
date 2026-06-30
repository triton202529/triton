# scripts/show_phase_197u_paper_runtime_governance_packet_refresh.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197u_paper_runtime_governance_packet_refresh.json"

Write-Host "=== TRITON Phase 197U Paper Governance Packet Refresh ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197u_paper_runtime_governance_packet_refresh" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Final governance packet state: " + $r.final_governance_packet_state)
Write-Host ("Final governance packet reason: " + $r.final_governance_packet_reason)
Write-Host ("Packet refreshed: " + $r.packet_refreshed)
Write-Host ("Packet refresh reason: " + $r.packet_refresh_reason)
Write-Host ("Packet before current: " + $r.packet_before_current)
Write-Host ("Packet after current: " + $r.packet_after_current)
Write-Host ("Current packet path: " + $r.current_packet_path)
Write-Host ("Current packet age (hours): " + $r.current_packet_age_hours)
Write-Host ("Capped batch preview path: " + $r.capped_batch_preview_path)
Write-Host ("Capped batch preview age (hours): " + $r.capped_batch_preview_age_hours)
Write-Host ("Market open: " + $r.market_open)
Write-Host ("Soak status: " + $r.soak_status)
Write-Host ("Soak state: " + $r.soak_state)
Write-Host ("Post-interlock runtime state: " + $r.post_interlock_runtime_state)
Write-Host ("Governance review state: " + $r.governance_review_state)
Write-Host ("Dispatch interlock state: " + $r.dispatch_interlock_state)
Write-Host ("Safe block state: " + $r.safe_block_state)
Write-Host ("Planned exits: " + $r.planned_exits_count)
Write-Host ("Planned trims: " + $r.planned_trims_count)
Write-Host ("Batch notional: " + $r.batch_notional)
Write-Host ("Batch notional cap: " + $r.batch_notional_cap)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Guard snapshot refreshed: " + $r.guard_snapshot_refreshed)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Exits executed: " + $r.exits_executed)
Write-Host ("Trims executed: " + $r.trims_executed)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("Execution readiness: " + $r.execution_readiness)
Write-Host ("Paper entry gate: " + $r.paper_entry_gate_status)
Write-Host ("Operator packet id: " + $r.operator_packet_id)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
