# scripts/show_phase_196x_governed_execution_dispatch_alignment_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_196x_governed_execution_dispatch_alignment_validation.json"

Write-Host "=== TRITON Phase 196X Governed Execution Dispatch ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_196x_governed_execution_dispatch_alignment_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Dispatch contract created: " + $r.dispatch_contract_created)
Write-Host ("Dispatch symbols: " + ($r.dispatch_symbols -join ", "))
Write-Host ("Execute considered: " + ($r.execute_trades_considered_symbols -join ", "))
Write-Host ("Execute selected: " + ($r.execute_trades_selected_symbols -join ", "))
Write-Host ("Dispatch mode: " + $r.execute_trades_dispatch_mode)
Write-Host ("UNG outside dispatch: " + $r.ung_selected_outside_dispatch)
Write-Host ("Contract mismatch: " + $r.contract_mismatch_detected)
Write-Host ("Orders submitted: " + $r.orders_submitted)
Write-Host ("Loop remain stopped: " + $r.continuous_loop_should_remain_stopped)
Write-Host ""
Write-Host ("Next: " + $r.recommended_next_action) -ForegroundColor Yellow
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
