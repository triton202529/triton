# scripts/show_phase_197e_quarantine_registry_write_discipline_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197e_quarantine_registry_write_discipline_validation.json"

Write-Host "=== TRITON Phase 197E Quarantine Registry Write Discipline ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197e_quarantine_registry_write_discipline_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Root cause: " + $r.root_cause)
Write-Host ("Fix: " + $r.fix)
Write-Host ("Preflight run1 / run2: " + $r.preflight_status_run1 + " / " + $r.preflight_status_run2)
Write-Host ("Quarantined symbols: " + ($r.quarantined_symbols -join ", "))
Write-Host ("Failed symbols: " + ($r.failed_symbols -join ", "))
Write-Host ("MU hydrated: " + $r.mu_hydrated)
Write-Host ("Registry bytes unchanged: " + $r.registry_write_discipline.registry_bytes_unchanged)
Write-Host ("Registry git clean: " + $r.registry_write_discipline.registry_git_clean)
Write-Host ("Run2 registry skipped: " + $r.registry_write_discipline.run2.skipped)
Write-Host ("Run2 skip reason: " + $r.registry_write_discipline.run2.skip_reason)
Write-Host ("Preflight audit updated: " + $r.preflight_audit_updated)
Write-Host ("Live blocked: " + $r.live_execution_blocked)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Continuous loop running: " + $r.continuous_loop_running)
Write-Host ("Watchdog running: " + $r.watchdog_running)
Write-Host ("Loop interval minutes: " + $r.continuous_loop_interval_minutes)
Write-Host ""
Write-Host ("Next: " + $r.recommended_next_action) -ForegroundColor Yellow
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
