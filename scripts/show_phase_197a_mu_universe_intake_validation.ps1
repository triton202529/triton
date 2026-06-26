# scripts/show_phase_197a_mu_universe_intake_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197a_mu_universe_intake_validation.json"

Write-Host "=== TRITON Phase 197A MU Universe Intake ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197a_mu_universe_intake_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("MU in universe: " + $r.mu_in_universe)
Write-Host ("MU duplicate: " + $r.mu_duplicate)
Write-Host ("Universe count: " + $r.universe_count)
Write-Host ("Data availability: " + $r.data_availability)
Write-Host ("Pipeline status: " + $r.pipeline_status)
Write-Host ("Lifecycle outcome: " + $r.lifecycle_outcome)
Write-Host ("Allocation action: " + $r.allocation_action)
Write-Host ("MU in final selection: " + $r.mu_in_final_selection)
Write-Host ("Gate reason: " + $r.gate_reason)
Write-Host ("Live blocked: " + $r.live_execution_blocked)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Continuous loop running: " + $r.continuous_loop_running)
Write-Host ""
Write-Host ("Next: " + $r.recommended_next_action) -ForegroundColor Yellow
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
