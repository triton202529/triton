# scripts/show_phase_197k_approval_contract_dry_run_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197k_approval_contract_dry_run_validation.json"

Write-Host "=== TRITON Phase 197K Approval Contract Dry-Run ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197k_approval_contract_dry_run_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Real approval path: " + $r.real_approval_contract_path)
Write-Host ("Real approval exists before: " + $r.real_approval_exists_before)
Write-Host ("Real approval exists after: " + $r.real_approval_exists_after)
Write-Host ("Real runtime approval status: " + $r.real_runtime_approval_status)
Write-Host ("Fixture dir: " + $r.fixture_dir)
Write-Host ("Cases tested: " + ($r.cases_tested -join ", "))
Write-Host ("Cases passed: " + $r.cases_passed)
Write-Host ("Cases failed: " + $r.cases_failed)
Write-Host ("Valid fixture status: " + $r.valid_fixture_status)
Write-Host ("Governance sync: " + $r.governance_sync_status)
Write-Host ("Readiness: " + $r.readiness_status)
Write-Host ("Governance authorized: " + $r.governance_authorized)
Write-Host ("Eligible for paper execution (real): " + $r.eligible_for_paper_execution_real_runtime)
Write-Host ("Paper entry gate ok: " + $r.paper_entry_gate_ok)
Write-Host ("Paper entry gate reason: " + $r.paper_entry_gate_reason)
Write-Host ("Broker snapshot ok: " + $r.broker_snapshot_ok)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Positions: " + $r.positions_count)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ""
Write-Host "Invalid fixture rejections:" -ForegroundColor DarkGray
foreach ($rej in $r.invalid_fixture_rejections) {
    Write-Host ("  " + $rej.case + ": " + $rej.reason)
}
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
