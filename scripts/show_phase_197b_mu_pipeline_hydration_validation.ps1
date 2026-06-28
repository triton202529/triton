# scripts/show_phase_197b_mu_pipeline_hydration_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197b_mu_pipeline_hydration_validation.json"

Write-Host "=== TRITON Phase 197B MU Pipeline Hydration ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197b_mu_pipeline_hydration_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Root cause: " + $r.root_cause)
Write-Host ("MU in config: " + $r.mu_in_config_universe)
Write-Host ("Processed parquet: " + $r.processed_parquet_exists)
Write-Host ("Processed rows: " + $r.processed_rows)
Write-Host ("MU in signals: " + $r.mu_in_signals)
Write-Host ("MU in effective lifecycle: " + $r.mu_in_effective_lifecycle)
Write-Host ("MU in allocation: " + $r.mu_in_allocation)
Write-Host ("Allocation action: " + $r.allocation_action)
Write-Host ("Lifecycle outcome: " + $r.lifecycle_outcome)
Write-Host ("Allocation universe before/after: " + $r.universe_counts_before.allocation + " / " + $r.universe_counts_after.allocation)
Write-Host ("Lifecycle universe before/after: " + $r.universe_counts_before.lifecycle_effective + " / " + $r.universe_counts_after.lifecycle_effective)
Write-Host ("Gate reason: " + $r.gate_reason)
Write-Host ("Live blocked: " + $r.live_execution_blocked)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Continuous loop running: " + $r.continuous_loop_running)
Write-Host ""
Write-Host ("Next: " + $r.recommended_next_action) -ForegroundColor Yellow
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
