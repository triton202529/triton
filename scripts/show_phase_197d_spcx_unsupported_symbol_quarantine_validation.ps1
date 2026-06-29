# scripts/show_phase_197d_spcx_unsupported_symbol_quarantine_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197d_spcx_unsupported_symbol_quarantine_validation.json"
$csvPath = Join-Path $RepoRoot "data\results\phase_197d_spcx_quarantine_rows.csv"

Write-Host "=== TRITON Phase 197D SPCX Unsupported Symbol Quarantine ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197d_spcx_unsupported_symbol_quarantine_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Root cause: " + $r.root_cause)
Write-Host ("SPCX classification: " + $r.spcx_classification)
Write-Host ("Preflight before / after: " + $r.preflight_status_before + " -> " + $r.preflight_status_after)
Write-Host ("Preflight reason: " + $r.preflight_reason)
Write-Host ("Configured / processed / lifecycle / allocation: " + $r.configured_symbol_count + " / " + $r.processed_symbol_count + " / " + $r.lifecycle_symbol_count + " / " + $r.allocation_symbol_count)
Write-Host ("Stranded symbols: " + ($r.stranded_symbols -join ", "))
Write-Host ("Quarantined symbols: " + ($r.quarantined_symbols -join ", "))
Write-Host ("Failed symbols: " + ($r.failed_symbols -join ", "))
Write-Host ("Allocation refresh only: " + ($r.allocation_refresh_only_symbols -join ", "))
Write-Host ("MU hydrated: " + $r.mu_hydrated)
Write-Host ("Live blocked: " + $r.live_execution_blocked)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Continuous loop running: " + $r.continuous_loop_running)
Write-Host ("Watchdog running: " + $r.watchdog_running)
Write-Host ("Loop interval minutes: " + $r.continuous_loop_interval_minutes)
Write-Host ""
if (Test-Path -LiteralPath $csvPath -PathType Leaf) {
    Write-Host ("Quarantine rows CSV: " + $csvPath) -ForegroundColor DarkGray
}
Write-Host ("Next: " + $r.recommended_next_action) -ForegroundColor Yellow
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
