# scripts/run_market_open.ps1
# ------------------------------------------------------------
# TRITON -- Market-Open Runner (full safe sequence)
#
# Sequence:
#   A. Poll order status      (capture overnight changes)
#   B. Manage open orders     (smart reprice / cancellations)
#   C. Poll order status      (re-poll after adjustments)
#   D. Snapshot live state    (broker positions + open orders)
#   D1. Signal + lifecycle refresh (generate_signals, apply_signal_lifecycle,
#       build_effective_lifecycle) BEFORE analytics/allocation/readiness so
#       readiness/certification never evaluate stale inputs (Phase 198W).
#       Soft-fails to a safe-block: no orders, no artifacts, snapshot kept.
#   E0. Universe hydration preflight (stranded symbol sync; no orders)
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

. (Join-Path $scriptDir "triton_env_bootstrap.ps1") -RepoRoot $repoRoot

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

$env:TRITON_MARKET_OPEN_INVOCATION_ID = (
    "mo-" + $runStamp + "-" + [guid]::NewGuid().ToString("N")
)

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

# ── Run-RefreshStep: invoke a signal/lifecycle refresh module WITHOUT ─
# aborting the whole sequence on failure. Returns $true on success and
# $false on failure so the caller can safe-block (skip risk-acting stages,
# still snapshot/report) instead of placing orders on stale inputs.
#
# Capture python stdout/stderr before returning the boolean. Otherwise
# Write-Host-only logging is fine, but native python success-stream output
# is included in the function's return value; `if (-not (Run-RefreshStep))`
# then treats a non-empty array as $true and silently ignores the failure
# (observed: D1a REFRESH_FAIL followed by "refresh complete" then E0 abort).
function Run-RefreshStep {
    param(
        [Parameter(Mandatory = $true)] [string]   $Label,
        [Parameter(Mandatory = $true)] [string[]] $PyArgs
    )
    Write-Host ""
    Write-Host "[$Label]" -ForegroundColor Yellow
    Write-Host (">>> python " + ($PyArgs -join " ")) -ForegroundColor Cyan
    # Capture stdout/stderr so the boolean return is not polluted by python
    # output. Under Windows PowerShell 5.1, `2>&1` turns stderr lines into
    # ErrorRecords; with $ErrorActionPreference=Stop those become terminating
    # errors (Cycle 3 / D1a: "Unexpected error: Traceback...") before we can
    # apply REFRESH_SAFE_BLOCK. Temporarily Continue for the native invoke only.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & python @PyArgs 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($null -eq $code) { $code = 1 }
    $script:LastRefreshStepExitCode = [int]$code
    $script:LastRefreshStepStderrHint = ""
    foreach ($line in @($output)) {
        $text = if ($null -eq $line) { "" } else { [string]$line }
        Write-Host $text
        if (-not $script:LastRefreshStepStderrHint -and $text -match 'Traceback|Error|Exception|FAIL') {
            $script:LastRefreshStepStderrHint = $text.Substring(0, [Math]::Min(160, $text.Length))
        }
    }
    if ($code -ne 0) {
        Write-Host ""
        Write-Host ("[REFRESH_FAIL] Step '" + $Label + "' failed (exit $code).") -ForegroundColor Red
        Write-Host ("[REFRESH_FAIL] Command: python " + ($PyArgs -join " ")) -ForegroundColor Red
        return $false
    }
    return $true
}

function Write-MarketOpenSequenceStatus {
    param(
        [Parameter(Mandatory = $true)] [string] $Status,
        [string] $FailedStep = "",
        [string] $Reason = "",
        [int] $ExitCode = 0
    )
    $statusDir = Join-Path $repoRoot "data\results"
    if (-not (Test-Path $statusDir)) {
        New-Item -ItemType Directory -Path $statusDir -Force | Out-Null
    }
    $statusPath = Join-Path $statusDir "market_open_sequence_status.json"
    $payload = [ordered]@{
        generated_at             = (Get-Date).ToUniversalTime().ToString("o")
        status                   = $Status
        failed_step              = $FailedStep
        reason                   = $Reason
        exit_code                = $ExitCode
        broker_orders_submitted  = $false
        live_execution_permitted = $false
    }
    ($payload | ConvertTo-Json) | Set-Content -Path $statusPath -Encoding UTF8
}

$exitCode = 0
try {
    Write-Host "=== TRITON MARKET OPEN SEQUENCE START ===" -ForegroundColor Green
    Write-Host ("RepoRoot : " + $repoRoot)
    Write-Host ("LogFile  : " + $logFile)
    Write-Host ("StartedAt: " + (Get-Date).ToString("s"))

    # ── 0. Paper ARM validation (read-only; never create/renew/extend) ───
    # Explicit operator arm is required beforehand:
    #   python -m services.set_paper_arm_confirm --ttl <minutes>
    # Invalid/expired/missing/malformed arm fails closed and aborts before
    # order-capable stages below.
    Run-Step -Label "0. PAPER ARM VALIDATE (READ-ONLY)" `
             -PyArgs @("-m","services.paper_session_arm_policy","--validate-only","--json")

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

    # ── D1. Signal + lifecycle refresh (Phase 198W) ──────────────────────
    # Refresh market-data inputs BEFORE analytics / allocation / position
    # management / readiness / certification so those stages never evaluate
    # stale signals or lifecycle artifacts (Phase 198V: REFRESH_PIPELINE_GAP).
    # Soft-fail: if any refresh step fails we safe-block the risk-acting
    # stages (no broker orders, no approval/auth artifacts), still complete
    # the broker snapshot (already done above) and heartbeat/reporting, and
    # exit non-zero so the continuous loop blocks paper entry this cycle.
    $refreshOk = $true
    $refreshFailReason = ""
    $refreshFailRc = 0
    $script:LastRefreshStepExitCode = 0
    $script:LastRefreshStepStderrHint = ""
    Write-Host ""
    Write-Host "[D1. SIGNAL + LIFECYCLE REFRESH]" -ForegroundColor Yellow
    # Compare explicitly to $true so a polluted multi-value return can never
    # be treated as success (see Run-RefreshStep capture contract).
    if ($refreshOk) {
        $stepOk = Run-RefreshStep -Label "D1a. GENERATE SIGNALS" -PyArgs @("-m","services.generate_signals")
        if ($stepOk -ne $true) {
            $refreshOk = $false
            $refreshFailReason = "generate_signals_failed"
            $refreshFailRc = [int]$script:LastRefreshStepExitCode
        }
    }
    if ($refreshOk) {
        $stepOk = Run-RefreshStep -Label "D1b. APPLY SIGNAL LIFECYCLE" -PyArgs @("-m","services.apply_signal_lifecycle")
        if ($stepOk -ne $true) {
            $refreshOk = $false
            $refreshFailReason = "apply_signal_lifecycle_failed"
            $refreshFailRc = [int]$script:LastRefreshStepExitCode
        }
    }
    if ($refreshOk) {
        $stepOk = Run-RefreshStep -Label "D1c. BUILD EFFECTIVE LIFECYCLE" -PyArgs @("-m","services.build_effective_lifecycle")
        if ($stepOk -ne $true) {
            $refreshOk = $false
            $refreshFailReason = "build_effective_lifecycle_failed"
            $refreshFailRc = [int]$script:LastRefreshStepExitCode
        }
    }

    if ($refreshOk -ne $true) {
        if ($refreshFailRc -eq 0) { $refreshFailRc = 1 }
        $hint = [string]$script:LastRefreshStepStderrHint
        if (-not $hint) { $hint = $refreshFailReason }
        Write-Host ""
        Write-Host ("[REFRESH_SAFE_BLOCK] reason=" + $refreshFailReason + " rc=" + $refreshFailRc) -ForegroundColor Red
        Write-Host "[REFRESH_SAFE_BLOCK] readiness=SAFE_BLOCKED; skipping allocation / position management." -ForegroundColor Red
        Write-Host "[REFRESH_SAFE_BLOCK] No broker orders, no approval/auth artifacts. Broker snapshot already captured." -ForegroundColor Red
        Write-MarketOpenSequenceStatus -Status "REFRESH_SAFE_BLOCKED" `
            -FailedStep ("D1. SIGNAL + LIFECYCLE REFRESH (" + $refreshFailReason + ")") `
            -Reason $hint -ExitCode $refreshFailRc
        # Still touch heartbeat so dashboards report the (safe-blocked) cycle.
        Run-Step -Label "G. TOUCH PIPELINE HEARTBEAT (post safe-block)" `
                 -PyArgs @("-m","services.touch_pipeline_heartbeat")
        # Surface the failure so the outer catch records it and the loop
        # blocks paper entry. Broker safety snapshot (D) already completed.
        throw ("STEP_FAILED:D1. SIGNAL + LIFECYCLE REFRESH (" + $refreshFailReason + "):" + $refreshFailRc)
    }

    Write-Host "[D1. SIGNAL + LIFECYCLE REFRESH] complete (signals + lifecycle refreshed)." -ForegroundColor Green

    # ── E0. Universe hydration preflight (before allocation; no broker orders) ─
    Run-Step -Label "E0. UNIVERSE HYDRATION PREFLIGHT" `
             -PyArgs @("-m","services.runtime_universe_hydration_preflight","--verbose")

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

    # ── F2. Paper SELL readiness (cycle-bound; fail-closed; no dispatch/arm) ──
    # Non-zero F2 codes 10-14 block BUY/ADD via the exit-before-entry interlock
    # but must not abort heartbeat or crash the continuous loop. Do not use
    # Run-Step here: a pending-arm cycle is a successful fail-closed market-open.
    Write-Host ""
    Write-Host "[F2. PAPER SELL READINESS PACKET]" -ForegroundColor Yellow
    Write-Host ">>> python -m services.defensive_exit_approval_readiness" -ForegroundColor Cyan
    & python -m services.defensive_exit_approval_readiness
    $f2Rc = 0
    if ($null -ne $LASTEXITCODE) { $f2Rc = [int]$LASTEXITCODE }
    Write-Host ("[F2] exit_code=" + $f2Rc) -ForegroundColor Cyan
    if (@(0, 10, 11, 12, 13, 14) -notcontains $f2Rc) {
        Write-Host ("[FAIL] Step 'F2. PAPER SELL READINESS PACKET' failed (exit $f2Rc).") -ForegroundColor Red
        throw ("STEP_FAILED:F2. PAPER SELL READINESS PACKET:" + $f2Rc)
    }

    # ── G. Pipeline heartbeat (dashboard freshness; status writer only) ──
    Run-Step -Label "G. TOUCH PIPELINE HEARTBEAT" `
             -PyArgs @("-m","services.touch_pipeline_heartbeat")

    Write-Host ""
    Write-Host "=== TRITON MARKET OPEN SEQUENCE COMPLETE ===" -ForegroundColor Green
    Write-Host ("FinishedAt: " + (Get-Date).ToString("s"))
    Write-Host ("LogFile   : " + $logFile)
    Write-MarketOpenSequenceStatus -Status "COMPLETE" -FailedStep "" -Reason "ok" -ExitCode 0
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
        Write-MarketOpenSequenceStatus -Status "ABORTED" -FailedStep $failedLabel `
            -Reason "step_failed" -ExitCode $exitCode
    } else {
        Write-Host ""
        Write-Host ("[ABORT] Unexpected error: " + $msg) -ForegroundColor Red
        $exitCode = 1
        Write-MarketOpenSequenceStatus -Status "ABORTED" -FailedStep "UNEXPECTED" `
            -Reason $msg -ExitCode $exitCode
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
