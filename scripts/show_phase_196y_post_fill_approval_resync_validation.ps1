# scripts/show_phase_196y_post_fill_approval_resync_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_196y_post_fill_approval_resync_validation.json"

Write-Host "=== TRITON Phase 196Y Post-Fill Approval Resync ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_196y_post_fill_approval_resync_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Continuous loop running: " + $r.continuous_loop_running)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Positions: " + ($r.positions -join ", "))
Write-Host ("Final selected: " + ($r.final_selected_symbols -join ", "))
Write-Host ("Approved symbols: " + ($r.approved_symbols -join ", "))
Write-Host ("Approval match mode: " + $r.approval_match_mode)
Write-Host ("Approval covers final: " + $r.approval_covers_final_selection)
Write-Host ("Selection sync: " + $r.selection_approval_sync)
Write-Host ("Gate pass: " + $r.gate_pass)
Write-Host ("Gate reason: " + $r.gate_reason)
Write-Host ("FINAL_SELECTION_NOT_APPROVED resolved: " + $r.final_selection_not_approved_resolved)
Write-Host ""
Write-Host ("Next: " + $r.recommended_next_action) -ForegroundColor Yellow
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
