# Quick check: paper trade cycle artifacts (scheduled wrapper + summary + log tail)
$ErrorActionPreference = "Continue"
$root = "C:\Users\akimw\triton"
$wrapper = Join-Path $root "run_paper_trade_cycle_scheduled.ps1"
$log = Join-Path $root "data\results\paper_trade_cycle_scheduled.log"
$summary = Join-Path $root "data\results\paper_trade_cycle_summary.json"
$cycleLog = Join-Path $root "data\results\paper_trade_cycle_log.csv"

Write-Host "=== Scheduled wrapper ===" -ForegroundColor Cyan
Write-Host "Exists: $wrapper -> $(Test-Path $wrapper)"

Write-Host "`n=== Scheduled log (last 15 lines) ===" -ForegroundColor Cyan
if (Test-Path $log) { Get-Content $log -Tail 15 } else { Write-Host "(missing)" }

Write-Host "`n=== paper_trade_cycle_summary.json ===" -ForegroundColor Cyan
if (Test-Path $summary) { Get-Content $summary -Raw } else { Write-Host "(missing)" }

Write-Host "`n=== paper_trade_cycle_log.csv (last 5 lines) ===" -ForegroundColor Cyan
if (Test-Path $cycleLog) { Get-Content $cycleLog -Tail 5 } else { Write-Host "(missing)" }
