# scripts/show_phase_196w_planner_allocation_alignment_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_196w_planner_allocation_alignment_validation.json"

Write-Host "=== TRITON Phase 196W Planner Allocation Alignment ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_196w_planner_allocation_alignment_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Market open: " + $r.market_open)
Write-Host ("Planner input mode: " + $r.planner_input_mode)
Write-Host ("Eligible allocation: " + ($r.eligible_allocation_symbols -join ", "))
Write-Host ("Planner considered: " + ($r.planner_considered_symbols -join ", "))
Write-Host ("Planner selected: " + ($r.planner_selected_symbols -join ", "))
Write-Host ("Contract mismatch: " + $r.contract_mismatch_detected)
Write-Host ("UNG outside allocation: " + $r.ung_selected_outside_allocation)
Write-Host ("Would execute if open: " + $r.would_execute_if_market_open)
Write-Host ("Gate reason: " + $r.gate_reason)
Write-Host ""
Write-Host ("Next: " + $r.recommended_next_action) -ForegroundColor Yellow
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
