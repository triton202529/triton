# scripts/show_phase_196z_planner_empty_safe_classification_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_196z_planner_empty_safe_classification_validation.json"

Write-Host "=== TRITON Phase 196Z Planner-Empty Safe Classification ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_196z_planner_empty_safe_classification_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Continuous loop running: " + $r.continuous_loop_running)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Positions: " + ($r.positions -join ", "))
Write-Host ("Eligible allocation: " + ($r.eligible_allocation_symbols -join ", "))
Write-Host ("Planner considered: " + ($r.planner_considered_symbols -join ", "))
Write-Host ("Planner selected: " + ($r.planner_selected_symbols -join ", "))
Write-Host ("Contract status: " + $r.contract_status)
Write-Host ("Contract mismatch: " + $r.contract_mismatch_detected)
Write-Host ("Planner empty valid: " + $r.planner_empty_valid)
Write-Host ("Gate pass: " + $r.gate_pass)
Write-Host ("Gate reason: " + $r.gate_reason)
Write-Host ("Contract mismatch resolved: " + $r.contract_mismatch_resolved)
Write-Host ""
Write-Host ("Next: " + $r.recommended_next_action) -ForegroundColor Yellow
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
