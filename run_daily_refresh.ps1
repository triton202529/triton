# run_daily_refresh.ps1
# ------------------------------------------------------------
# TRITON — Daily Prediction Refresh Owner (TRITON_DailyRefresh)
#
# Canonical stages (strict order; stop on first failure):
#   0) capture current-run baseline (pre-fetch)
#   1) fetch_raw_data
#   2) stale_data_gate
#   3) preprocess
#   4) train_model
#   5) daily_prediction_refresh_validate  (symbol-set + current-run advancement)
#
# Safety:
#   - Market-data network only during fetch_raw_data.
#   - No broker / account / order / execution / paper-arm / runtime-control calls.
#   - Resolves repo root from this script's location (caller WD independent).
#   - Uses .venv\Scripts\python.exe when present.
#   - Propagates the first native nonzero exit code.
#   - Exit 0 without current-run output advancement fails validation.
# ------------------------------------------------------------

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding
try { chcp 65001 | Out-Null } catch { }

$repoRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$repoRoot = (Resolve-Path -LiteralPath $repoRoot).Path
Set-Location $repoRoot

$py = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw "Virtual-environment interpreter not found: $py"
}

$logsDir = Join-Path $repoRoot "logs"
$resultsDir = Join-Path $repoRoot "data\results"
New-Item -ItemType Directory -Force -Path $logsDir, $resultsDir | Out-Null

$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runId = "daily_prediction_refresh_" + $runStamp
$runStartedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$logPath = Join-Path $logsDir ("daily_refresh_" + $runStamp + ".log")
$summaryPath = Join-Path $resultsDir "daily_prediction_refresh_summary.json"
$baselinePath = Join-Path $resultsDir ("daily_prediction_refresh_baseline_" + $runId + ".json")
$receiptsPath = Join-Path $resultsDir ("daily_prediction_refresh_stage_receipts_" + $runId + ".json")

$script:StageReceipts = @()

function Log([string]$msg) {
    $line = "$(Get-Date -Format s)  $msg"
    Write-Host $line
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

function Write-StageReceipts {
    $payload = @($script:StageReceipts)
    ($payload | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $receiptsPath -Encoding UTF8
}

function Run-PyStep {
    param(
        [Parameter(Mandatory = $true)] [string] $Name,
        [Parameter(Mandatory = $true)] [string[]] $PyArgs
    )
    $started = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Log (">> " + $Name)
    Log ("CMD: " + $py + " " + ($PyArgs -join " "))
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $py @PyArgs 2>&1 | ForEach-Object {
            $text = if ($null -eq $_) { "" } else { [string]$_ }
            Write-Host $text
            Add-Content -LiteralPath $logPath -Value $text -Encoding UTF8
        }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($null -eq $code) { $code = 1 }
    $ended = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Log ("EXIT: " + $Name + " code=" + $code)
    $script:StageReceipts += [ordered]@{
        name       = $Name
        started_at = $started
        ended_at   = $ended
        exit_code  = [int]$code
        outcome    = $(if ([int]$code -eq 0) { "ok" } else { "failed" })
    }
    Write-StageReceipts
    if ([int]$code -ne 0) {
        Log ("FAILED: " + $Name + " (exit=" + $code + "). See log: " + $logPath)
        throw ("STEP_FAILED:" + $Name + ":" + $code)
    }
}

$exitCode = 0
try {
    Log "=== TRITON Daily Prediction Refresh START ==="
    Log ("repo_root=" + $repoRoot)
    Log ("python=" + $py)
    Log ("run_id=" + $runId)
    Log ("run_started_at_utc=" + $runStartedAt)
    Log ("log_path=" + $logPath)
    Log ("baseline_path=" + $baselinePath)
    Log ("summary_path=" + $summaryPath)

    $env:TRITON_ENABLE_PAPER_EXECUTION = "0"
    $env:TRITON_ENABLE_AUTO_PAPER_ENTRIES = "0"
    $env:TRITON_ENABLE_AUTO_PAPER_APPROVAL = "0"
    $env:TRITON_ENABLE_LIVE_EXECUTION = "0"
    $env:TRITON_LIVE_EXECUTION = "0"

    # 0) Pre-fetch baseline — binds later validation to this run.
    Run-PyStep -Name "capture_baseline" -PyArgs @(
        "-m", "services.daily_prediction_refresh_validate",
        "--capture-baseline",
        "--repo-root", $repoRoot,
        "--run-id", $runId,
        "--run-started-at", $runStartedAt,
        "--baseline-out", $baselinePath,
        "--json"
    )
    # Baseline is not part of the refresh pipeline receipt order for validator.
    $script:StageReceipts = @()
    Write-StageReceipts

    Run-PyStep -Name "fetch_raw_data" -PyArgs @("-m", "services.fetch_raw_data", "--verbose", "--min-ok", "10")
    Run-PyStep -Name "stale_data_gate" -PyArgs @("-m", "services.stale_data_gate")
    Run-PyStep -Name "preprocess" -PyArgs @("-m", "services.preprocess")
    Run-PyStep -Name "train_model" -PyArgs @("-m", "services.train_model", "--min-rows", "30")
    Run-PyStep -Name "daily_prediction_refresh_validate" -PyArgs @(
        "-m", "services.daily_prediction_refresh_validate",
        "--repo-root", $repoRoot,
        "--run-id", $runId,
        "--run-started-at", $runStartedAt,
        "--baseline-in", $baselinePath,
        "--stage-receipts", $receiptsPath,
        "--summary-out", $summaryPath,
        "--json"
    )

    Log "=== TRITON Daily Prediction Refresh OK ==="
    $exitCode = 0
}
catch {
    $msg = [string]$_.Exception.Message
    Log ("[ABORT] " + $msg)
    if ($msg -match '^STEP_FAILED:.+:(\d+)$') {
        $exitCode = [int]$Matches[1]
    } else {
        $exitCode = 1
    }
    if ($exitCode -eq 0) { $exitCode = 1 }
}

Log ("FinishedAt=" + (Get-Date -Format s) + " exit=" + $exitCode)
exit $exitCode
