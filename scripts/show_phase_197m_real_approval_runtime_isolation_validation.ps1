# scripts/show_phase_197m_real_approval_runtime_isolation_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197m_real_approval_runtime_isolation_validation.json"

Write-Host "=== TRITON Phase 197M Real Approval Runtime Isolation ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197m_real_approval_runtime_isolation_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Isolated approval path: " + $r.isolated_approval_path)
Write-Host ("Isolated validation status: " + $r.isolated_approval_validation_status)
Write-Host ("Real approval path: " + $r.real_approval_contract_path)
Write-Host ("Real approval exists before: " + $r.real_approval_exists_before)
Write-Host ("Real approval exists after: " + $r.real_approval_exists_after)
Write-Host ("Real runtime approval status: " + $r.real_runtime_approval_status)
Write-Host ("Simulated classification: " + $r.simulated_approval_runtime_classification)
Write-Host ("Governance authorized: " + $r.governance_authorized)
Write-Host ("Governance block reason: " + $r.governance_block_reason)
Write-Host ("Readiness: " + $r.readiness_status)
Write-Host ("Paper entry gate ok: " + $r.paper_entry_gate_ok)
Write-Host ("Paper entry gate reason: " + $r.paper_entry_gate_reason)
Write-Host ("Master execution gate: " + $r.master_execution_gate_status)
Write-Host ("Notional cap status: " + $r.notional_cap_status)
Write-Host ("Eligible paper exec (real): " + $r.eligible_for_paper_execution_real_runtime)
Write-Host ("Eligible paper exec (simulated): " + $r.eligible_for_paper_execution_simulated)
Write-Host ("Runtime authority unchanged: " + $r.runtime_authority_unchanged)
Write-Host ("Broker snapshot ok: " + $r.broker_snapshot_ok)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Positions: " + $r.positions_count)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
