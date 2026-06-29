# scripts/show_phase_197f_premarket_broker_auth_quarantine_repair_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197f_premarket_broker_auth_quarantine_repair_validation.json"

Write-Host "=== TRITON Phase 197F Pre-Market Broker Auth + Quarantine Repair ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197f_premarket_broker_auth_quarantine_repair_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Branch: " + $r.branch)
Write-Host ("Broker auth root cause: " + $r.broker_auth_root_cause)
Write-Host ("Broker snapshot ok: " + $r.broker_snapshot_ok)
Write-Host ("Broker SAFE_AUTH_BLOCK: " + $r.broker_snapshot_safe_auth_block)
Write-Host ("Preflight run1 / run2: " + $r.preflight_status_run1 + " / " + $r.preflight_status_run2)
Write-Host ("Quarantined symbols: " + ($r.quarantined_symbols -join ", "))
Write-Host ("SPCX only quarantined: " + $r.spcx_only_quarantined)
Write-Host ("Artifact regression symbols count: " + ($r.artifact_regression_symbols | Measure-Object).Count)
Write-Host ("MU hydrated: " + $r.mu_hydrated)
Write-Host ("Registry git clean: " + $r.registry_git_clean)
Write-Host ("Registry bytes unchanged: " + $r.registry_bytes_unchanged)
Write-Host ("Live blocked: " + $r.live_execution_blocked)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Continuous loop running: " + $r.continuous_loop_running)
Write-Host ("Watchdog running: " + $r.watchdog_running)
Write-Host ""
Write-Host ("Next: " + $r.recommended_next_action) -ForegroundColor Yellow
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
