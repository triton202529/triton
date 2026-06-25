# scripts/show_phase_196v_allocation_to_paper_entry_handoff_validation.ps1
# Phase 196V — Display allocation-to-paper-entry handoff validation report.

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_196v_allocation_to_paper_entry_handoff_validation.json"
$csvPath = Join-Path $RepoRoot "data\results\phase_196v_candidate_handoff_rows.csv"

Write-Host "=== TRITON Phase 196V Allocation-to-Paper-Entry Handoff ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Validation report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_196v_allocation_to_paper_entry_handoff_validation" -ForegroundColor DarkGray
    exit 1
}

try {
    $r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
} catch {
    Write-Host "Invalid JSON in validation report." -ForegroundColor Red
    exit 1
}

Write-Host ("Status: " + $r.status)
Write-Host ("Market open: " + $r.market_open)
Write-Host ("Paper arm valid: " + $r.paper_arm_valid)
Write-Host ("Execution readiness ready: " + $r.execution_readiness_ready)
Write-Host ("Allocation OPEN_NEW count: " + $r.allocation_open_new_count)
Write-Host ("Allocation ADD count: " + $r.allocation_add_count)
Write-Host ("Eligible allocation symbols: " + ($r.eligible_allocation_symbols -join ", "))
Write-Host ("Planner preview symbols: " + ($r.planner_preview_symbols -join ", "))
Write-Host ("Contract mismatch: " + $r.contract_mismatch_detected)
Write-Host ("Would execute if market open: " + $r.would_execute_if_market_open)
Write-Host ("Orders submitted: " + $r.orders_submitted)
Write-Host ("Live execution blocked: " + $r.live_execution_blocked)
Write-Host ("Stale cache used for execution: " + $r.stale_candidate_cache_used_for_execution)
Write-Host ("Gate reason: " + $r.gate_reason)
Write-Host ""
Write-Host ("Recommended next: " + $r.recommended_next_action) -ForegroundColor Yellow
Write-Host ""

if (Test-Path -LiteralPath $csvPath -PathType Leaf) {
    Write-Host ("Handoff rows CSV: " + $csvPath) -ForegroundColor DarkGray
}

if ($r.notes) {
    Write-Host "--- Notes ---" -ForegroundColor Cyan
    foreach ($n in $r.notes) {
        Write-Host ("  " + $n)
    }
}

Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
