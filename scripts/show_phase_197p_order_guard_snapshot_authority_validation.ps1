# scripts/show_phase_197p_order_guard_snapshot_authority_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197p_order_guard_snapshot_authority_validation.json"

Write-Host "=== TRITON Phase 197P Order Guard Snapshot Authority ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197p_order_guard_snapshot_authority_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Order guard authority state: " + $r.order_guard_authority_state)
Write-Host ("Order guard authority reason: " + $r.order_guard_authority_reason)
Write-Host ("Guard snapshot path: " + $r.guard_snapshot_path)
Write-Host ("Guard snapshot exists: " + $r.guard_snapshot_exists)
Write-Host ("Guard snapshot age (min): " + $r.guard_snapshot_age_minutes)
Write-Host ("Guard snapshot fresh: " + $r.guard_snapshot_fresh)
Write-Host ("Broker snapshot OK: " + $r.broker_snapshot_ok)
Write-Host ("Open orders snapshot path: " + $r.open_orders_snapshot_path)
Write-Host ("Open orders snapshot exists: " + $r.open_orders_snapshot_exists)
Write-Host ("Open orders snapshot age (min): " + $r.open_orders_snapshot_age_minutes)
Write-Host ("Open orders snapshot fresh: " + $r.open_orders_snapshot_fresh)
Write-Host ("Open orders count: " + $r.open_orders_count)
Write-Host ("Recent orders count: " + $r.recent_orders_count)
Write-Host ("Updated guard snapshot: " + $r.updated_guard_snapshot)
Write-Host ("Update mode: " + $r.update_mode)
Write-Host ("Source authority: " + $r.source_authority)
Write-Host ("Gate freshness note: " + $r.gate_freshness_note)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Live block reason: " + $r.live_block_reason)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Market open: " + $r.market_open)
Write-Host ("Paper entry gate status: " + $r.paper_entry_gate_status)
Write-Host ("Paper entry gate reason: " + $r.paper_entry_gate_reason)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("Broker orders submitted: " + $r.broker_orders_submitted)
Write-Host ("Recheck order guard state: " + $r.recheck_order_guard_authority_state)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
