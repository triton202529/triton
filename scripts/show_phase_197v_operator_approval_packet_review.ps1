# scripts/show_phase_197v_operator_approval_packet_review.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197v_operator_approval_packet_review.json"
$mdPath = Join-Path $RepoRoot "data\results\phase_197v_operator_approval_packet_review.md"

Write-Host "=== TRITON Phase 197V Operator Approval Packet Review ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197v_operator_approval_packet_review" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Review state: " + $r.review_state)
Write-Host ("Market open: " + $r.market_open)
Write-Host ("Governance packet state (197U): " + $r.governance_packet_state)
Write-Host ("197N state: " + $r.governance_review_state_197n)
Write-Host ("197O state: " + $r.market_hours_recheck_state_197o)
Write-Host ("197T soak: " + $r.phase_197t_status + " / " + $r.soak_state)
Write-Host ("197S runtime: " + $r.phase_197s_status + " / " + $r.post_interlock_runtime_state)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("Operator approval required: " + $r.operator_approval_required)
Write-Host ("Planned exits: " + $r.planned_exits_count)
Write-Host ("Planned trims: " + $r.planned_trims_count)
Write-Host ("Symbols to exit: " + ($r.symbols_to_exit -join ", "))
Write-Host ("Deferred symbols: " + ($r.deferred_symbols -join ", "))
Write-Host ("Batch notional: " + $r.batch_notional)
Write-Host ("Batch cap: " + $r.batch_notional_cap)
Write-Host ("Cap status: " + $r.cap_status)
Write-Host ("Highest drag symbol: " + $r.highest_drag_symbol)
Write-Host ("Largest notional symbol: " + $r.largest_notional_symbol)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Exits executed: " + $r.exits_executed)
Write-Host ("Trims executed: " + $r.trims_executed)
Write-Host ("Packet ID: " + $r.operator_packet_id)
Write-Host ""
Write-Host ("Final instruction: " + $r.final_operator_instruction)
Write-Host ""
if (Test-Path -LiteralPath $mdPath -PathType Leaf) {
    Write-Host ("Full review markdown: " + $r.review_markdown_path) -ForegroundColor DarkGray
}
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
