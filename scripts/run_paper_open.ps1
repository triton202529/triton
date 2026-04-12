# scripts/run_paper_open.ps1
# ------------------------------------------------------------
# TRITON — Paper Market-Open Runner (safe, repeatable)
# - Cancels any open orders first (reset)
# - Runs preflight_place with UNIQUE placement session
# - Uses stable log-session for polling
# - Writes state to data\live\last_open_run.json
#
# Usage:
#   .\.venv\Scripts\Activate.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_paper_open.ps1
#
# Optional flags:
#   -OrdersPath "data\live\orders_today.csv"
#   -LogSession "2026-02-02_OPEN"
#   -BaseSession "2026-02-02_OPEN"
#   -MaxBatchNotional 4000
#   -PollLoops 0    # set >0 to poll after placing
#   -PollSleepSec 60

param(
  [string]$OrdersPath = "data\live\orders_today.csv",
  [string]$LogSession = "2026-02-02_OPEN",
  [string]$BaseSession = "2026-02-02_OPEN",
  [double]$MaxBatchNotional = 4000,
  [int]$PollLoops = 0,
  [int]$PollSleepSec = 60
)

$ErrorActionPreference = "Stop"

function Require-File($path) {
  if (-not (Test-Path $path)) {
    throw "Missing required file: $path"
  }
}

function Run-Cmd([string]$cmd) {
  Write-Host ""
  Write-Host ">>> $cmd" -ForegroundColor Cyan
  iex $cmd
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed (exit $LASTEXITCODE): $cmd"
  }
}

# Ensure we are in repo root (script can be called from anywhere)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

Require-File ".\.venv\Scripts\python.exe"
Require-File $OrdersPath

# Unique placement session suffix (prevents Alpaca client_order_id collisions)
$ts = Get-Date -Format "HHmmss"
$placementSession = "${LogSession}_R0930_${ts}"

# Persist state so you can reference what ran
$statePath = "data\live\last_open_run.json"
New-Item -ItemType Directory -Force -Path "data\live" | Out-Null
$state = @{
  mode = "paper"
  orders = (Resolve-Path $OrdersPath).Path
  base_session = $BaseSession
  log_session = $LogSession
  placement_session = $placementSession
  max_batch_notional = $MaxBatchNotional
  ran_at_local = (Get-Date).ToString("s")
} | ConvertTo-Json -Depth 4
Set-Content -Path $statePath -Value $state -Encoding UTF8

Write-Host ""
Write-Host "TRITON Paper Open Runner" -ForegroundColor Green
Write-Host "RepoRoot          : $repoRoot"
Write-Host "Orders            : $OrdersPath"
Write-Host "BaseSession       : $BaseSession"
Write-Host "LogSession        : $LogSession"
Write-Host "PlacementSession  : $placementSession"
Write-Host "MaxBatchNotional  : $MaxBatchNotional"
Write-Host "StateFile         : $statePath"
Write-Host ""

# 1) Cancel open orders (NO placing)
Run-Cmd "python -m services.execute_cycle --mode paper --cancel-open --session $BaseSession --refresh-orders --verbose"

# 2) Intent preview (read-only)
Run-Cmd "python -m services.intent_preview --mode paper --orders $OrdersPath --top 50 --warn-large-qty 10"

# 3) Preflight + place (only placement path)
Run-Cmd @"
python -m services.preflight_place `
  --mode paper `
  --orders $OrdersPath `
  --session $placementSession `
  --log-session $LogSession `
  --require-marketdata `
  --drop-illegal-sells `
  --cancel-duplicates `
  --max-batch-notional $MaxBatchNotional `
  --verbose
"@

# 4) Optional polling loop (poll by LOG-SESSION)
if ($PollLoops -gt 0) {
  for ($i = 1; $i -le $PollLoops; $i++) {
    Write-Host ""
    Write-Host "Polling ($i/$PollLoops) — log-session=$LogSession" -ForegroundColor Yellow
    Run-Cmd "python -m services.poll_order_status --mode paper --session $LogSession --refresh"
    if ($i -lt $PollLoops) { Start-Sleep -Seconds $PollSleepSec }
  }
} else {
  Write-Host ""
  Write-Host "Done. To poll now:" -ForegroundColor Green
  Write-Host "python -m services.poll_order_status --mode paper --session $LogSession --refresh"
}
