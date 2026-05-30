# scripts/run_market_open.ps1
# ------------------------------------------------------------
# TRITON -- Market-Open Runner (full safe sequence)
#
# Sequence:
#   A. Poll order status      (capture overnight changes)
#   B. Manage open orders     (smart reprice / cancellations)
#   C. Poll order status      (re-poll after adjustments)
#   D. Snapshot live state    (broker positions + open orders)
#   E. Rebuild intelligence   (outcomes, pnl, perf, risk overlay, allocation)
#   F. Run position management (drives exits / rotation)
#
# Behaviour:
#   - Pins working directory to the Triton repo root regardless of where
#     the script is invoked from (the script lives in scripts/ and walks
#     one level up).
#   - Stops on the first failed step. Prints the failed step label and
#     the underlying exit code so the operator knows exactly where to
#     look without scrolling through a partial pipeline.
#   - Honours external command exit codes (NOT just PowerShell errors)
#     by checking $LASTEXITCODE after each python invocation, since
#     $ErrorActionPreference = "Stop" only catches native PowerShell
#     errors -- python.exe returning rc=1 is silent without this.
#   - Writes a timestamped transcript of every run to
#     data/results/scheduled_runs/market_open_YYYYMMDD_HHMMSS.log so
#     scheduled (unattended) invocations leave an auditable trail.
#     Terminal output is preserved for manual runs.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_market_open.ps1
#
# Safety:
#   - Read-only invocation of broker / signal / analytics services.
#   - Does NOT modify any Python trading logic.
#   - Does NOT place orders directly; manage_positions handles all order
#     placement through its own existing safety guards.

$ErrorActionPreference = "Stop"

# ── Pin to repo root so the script works from any cwd ────────────────
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

# ── Set up per-run log file under data/results/scheduled_runs ────────
# Start-Transcript mirrors host output (Write-Host + native-command
# stdout/stderr) into the file while leaving the terminal stream
# untouched, so manual runs still see colours / progress live.
$logDir = Join-Path $repoRoot "data\results\scheduled_runs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile  = Join-Path $logDir ("market_open_" + $runStamp + ".log")

Start-Transcript -Path $logFile -Append | Out-Null

# ── Run-Step: invoke a python module and abort on non-zero exit ──────
function Run-Step {
    param(
        [Parameter(Mandatory = $true)] [string]   $Label,
        [Parameter(Mandatory = $true)] [string[]] $PyArgs
    )
    Write-Host ""
    Write-Host "[$Label]" -ForegroundColor Yellow
    Write-Host (">>> python " + ($PyArgs -join " ")) -ForegroundColor Cyan
    & python @PyArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host ("[FAIL] Step '" + $Label + "' failed (exit $LASTEXITCODE).") -ForegroundColor Red
        Write-Host ("[FAIL] Command: python " + ($PyArgs -join " ")) -ForegroundColor Red
        Write-Host "[FAIL] Aborting market-open sequence." -ForegroundColor Red
        # Use throw (not `exit`) so the outer try/finally can stop the
        # transcript cleanly while still preserving the python rc.
        throw ("STEP_FAILED:" + $Label + ":" + $LASTEXITCODE)
    }
}

$exitCode = 0
try {
    Write-Host "=== TRITON MARKET OPEN SEQUENCE START ===" -ForegroundColor Green
    Write-Host ("RepoRoot : " + $repoRoot)
    Write-Host ("LogFile  : " + $logFile)
    Write-Host ("StartedAt: " + (Get-Date).ToString("s"))

    # ── A. Poll order status (initial) ───────────────────────────────────
    Run-Step -Label "A. POLL ORDER STATUS (initial)" `
             -PyArgs @("-m","services.poll_order_status","--mode","paper","--refresh")

    # ── B. Manage open orders (smart reprice / cancel) ───────────────────
    Run-Step -Label "B. MANAGE OPEN ORDERS" `
             -PyArgs @("-m","services.manage_open_orders","--mode","paper","--smart-manage","--execute-smart","--verbose")

    # ── C. Poll order status (post-adjustments) ──────────────────────────
    Run-Step -Label "C. POLL ORDER STATUS (post-adjustments)" `
             -PyArgs @("-m","services.poll_order_status","--mode","paper","--refresh")

    # ── D. Snapshot broker / live state ──────────────────────────────────
    Run-Step -Label "D. SNAPSHOT LIVE ORDERS" `
             -PyArgs @("-m","services.snapshot_live_orders","--mode","paper")

    # ── E. Rebuild intelligence layers (5 read-only analytics builders) ──
    Run-Step -Label "E1. BUILD TRADE OUTCOMES" `
             -PyArgs @("-m","services.build_trade_outcomes")
    Run-Step -Label "E2. BUILD PNL DIAGNOSTICS" `
             -PyArgs @("-m","services.build_pnl_diagnostics")
    Run-Step -Label "E3. BUILD PERFORMANCE INTELLIGENCE" `
             -PyArgs @("-m","services.build_performance_intelligence")
    Run-Step -Label "E4. PERFORMANCE RISK OVERLAY" `
             -PyArgs @("-m","services.performance_risk_overlay")
    Run-Step -Label "E5. PORTFOLIO ALLOCATION ENGINE" `
             -PyArgs @("-m","services.portfolio_allocation_engine")

    # ── F. Position management (exits / trims / rotation) ────────────────
    Run-Step -Label "F. POSITION MANAGEMENT" `
             -PyArgs @("-m","services.manage_positions","--mode","paper","--use-performance-risk-overlay","--max-rotation-exits","3","--verbose")

    Write-Host ""
    Write-Host "=== TRITON MARKET OPEN SEQUENCE COMPLETE ===" -ForegroundColor Green
    Write-Host ("FinishedAt: " + (Get-Date).ToString("s"))
    Write-Host ("LogFile   : " + $logFile)
}
catch {
    # Run-Step throws "STEP_FAILED:<label>:<rc>" on the first non-zero
    # python exit. Decode it and surface the original rc so schedulers
    # (Task Scheduler, supervisord-style wrappers) see a real failure.
    $msg = $_.Exception.Message
    if ($msg -match '^STEP_FAILED:(.+):(\d+)$') {
        $failedLabel = $Matches[1]
        $exitCode    = [int]$Matches[2]
        Write-Host ""
        Write-Host ("[ABORT] Market-open sequence aborted at step: " + $failedLabel) -ForegroundColor Red
        Write-Host ("[ABORT] Exit code: " + $exitCode) -ForegroundColor Red
    } else {
        Write-Host ""
        Write-Host ("[ABORT] Unexpected error: " + $msg) -ForegroundColor Red
        $exitCode = 1
    }
    Write-Host ("FinishedAt: " + (Get-Date).ToString("s"))
    Write-Host ("LogFile   : " + $logFile)
}
finally {
    Stop-Transcript | Out-Null
}

if ($exitCode -ne 0) {
    exit $exitCode
}
