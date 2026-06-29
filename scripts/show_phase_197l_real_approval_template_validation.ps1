# scripts/show_phase_197l_real_approval_template_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197l_real_approval_template_validation.json"

Write-Host "=== TRITON Phase 197L Real Approval Template ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197l_real_approval_template_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Dry-run template created: " + $r.dry_run_template_created)
Write-Host ("Template path: " + $r.template_path)
Write-Host ("Real approval path: " + $r.real_approval_contract_path)
Write-Host ("Real approval exists before: " + $r.real_approval_exists_before)
Write-Host ("Real approval exists after: " + $r.real_approval_exists_after)
Write-Host ("Real approval written: " + $r.real_approval_written)
Write-Host ("Write real blocked without flag: " + $r.write_real_approval_blocked_without_flag)
Write-Host ("Template validates as non-runtime: " + $r.template_validates_as_non_runtime)
Write-Host ("Simulated real contract validates: " + $r.simulated_real_contract_validates)
Write-Host ("Real runtime approval status: " + $r.approval_status_real_runtime)
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
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
