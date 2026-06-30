# scripts/show_phase_197w_fix_post_approval_validator_repair.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197w_fix_post_approval_validator_repair.json"

Write-Host "=== TRITON Phase 197W-FIX Post-Approval Validator Repair ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197w_fix_post_approval_validator_repair" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Final state: " + $r.final_state)
Write-Host ("Approval artifact state: " + $r.approval_artifact_state)
Write-Host ("Approval valid: " + $r.approval_valid)
Write-Host ("Approval expired: " + $r.approval_expired)
Write-Host ("Cleanup required: " + $r.cleanup_required)
Write-Host ("Open orders: " + $r.open_orders)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Exits executed: " + $r.exits_executed)
Write-Host ("Trims executed: " + $r.trims_executed)
Write-Host ("197U state: " + $r.'197u_state')
Write-Host ("197N state: " + $r.'197n_state')
Write-Host ("197O state: " + $r.'197o_state')
Write-Host ("Paper entry dry-run status: " + $r.paper_entry_dry_run_status)
Write-Host ("Paper entry dry-run reason: " + $r.paper_entry_dry_run_reason)
if ($r.cleanup_guidance) {
    Write-Host ("Cleanup guidance: " + $r.cleanup_guidance)
}
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
