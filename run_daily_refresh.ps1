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

if ($MyInvocation.InvocationName -ne '.') {
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
}

function Log([string]$msg) {
    $line = "$(Get-Date -Format s)  $msg"
    Write-Host $line
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

function Write-StageReceipts {
    param([string] $Destination = $receiptsPath)
    if ([string]::IsNullOrWhiteSpace($Destination)) { throw "STAGE_RECEIPT_DESTINATION_EMPTY" }
    $dest = [System.IO.Path]::GetFullPath($Destination)
    $parent = [System.IO.Path]::GetDirectoryName($dest)
    if ([string]::IsNullOrWhiteSpace($parent)) { throw "STAGE_RECEIPT_PARENT_MISSING" }
    [void][System.IO.Directory]::CreateDirectory($parent)
    $payload = foreach ($row in @($script:StageReceipts)) { $row }
    $json = @($payload) | ConvertTo-Json -Depth 8
    if ([string]::IsNullOrWhiteSpace($json)) { $json = "[]" }
    elseif (-not $json.TrimStart().StartsWith("[")) { $json = "[`r`n" + $json + "`r`n]" }
    $text = $json + "`r`n"
    $tmp = Join-Path $parent ("." + [System.IO.Path]::GetFileName($dest) + "." + [guid]::NewGuid().ToString("N") + ".tmp")
    $utf8 = New-Object System.Text.UTF8Encoding $false
    $stream = New-Object System.IO.FileStream($tmp, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        $writer = New-Object System.IO.StreamWriter($stream, $utf8)
        try {
            $writer.Write($text)
            $writer.Flush()
            $stream.Flush($true)
        } finally {
            $writer.Dispose()
            $stream = $null
        }
        if ([System.IO.File]::Exists($dest)) {
            try {
                [System.IO.File]::Replace($tmp, $dest, [NullString]::Value)
            } catch {
                if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
                throw
            }
        } else {
            [System.IO.File]::Move($tmp, $dest)
        }
    } catch {
        if ($null -ne $stream) { $stream.Dispose() }
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
        throw
    }
}

function Read-StableBytes([string]$Path) {
    $first = [System.IO.File]::ReadAllBytes($Path)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { $hash1 = ([BitConverter]::ToString($sha.ComputeHash($first))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
    $written = [System.IO.File]::GetLastWriteTimeUtc($Path)
    $second = [System.IO.File]::ReadAllBytes($Path)
    $sha2 = [System.Security.Cryptography.SHA256]::Create()
    try { $hash2 = ([BitConverter]::ToString($sha2.ComputeHash($second))).Replace("-", "").ToLowerInvariant() }
    finally { $sha2.Dispose() }
    if ($hash1 -ne $hash2 -or $first.Length -ne $second.Length) {
        throw "FETCH_ARTIFACT_UNSTABLE"
    }
    return @{ Bytes = $first; Sha256 = $hash1; WrittenUtc = $written }
}

function Bind-FetchArtifact {
    $fetchPath = Join-Path $resultsDir "daily_refresh_fetch_receipt.json"
    if (-not (Test-Path -LiteralPath $fetchPath -PathType Leaf)) { throw "FETCH_ARTIFACT_MISSING" }
    $stable = Read-StableBytes -Path $fetchPath
    $text = [System.Text.Encoding]::UTF8.GetString($stable.Bytes)
    if ($text.StartsWith([char]0xFEFF)) { $text = $text.Substring(1) }
    try { $receipt = $text | ConvertFrom-Json } catch { throw "FETCH_ARTIFACT_MALFORMED" }
    if ($null -eq $receipt.passed -or [bool]$receipt.passed -ne $true) { throw "FETCH_ARTIFACT_FAILED" }
    $ratio = 0.0
    if ($null -ne $receipt.tradable_coverage -and $null -ne $receipt.tradable_coverage.ratio) {
        $ratio = [double]$receipt.tradable_coverage.ratio
    }
    if ($ratio + 0.000000000001 -lt 0.90) { throw "FETCH_ARTIFACT_COVERAGE" }
    $session = [string]$receipt.target_session
    if (-not $session) { $session = [string]$receipt.completed_session }
    $activeSession = (
        & $py -c "from services.market_calendar import last_completed_nyse_session; print(last_completed_nyse_session().isoformat())"
    ).Trim()
    if (-not $session -or $session -ne $activeSession) { throw "FETCH_ARTIFACT_WRONG_SESSION" }
    $baseline = Get-Content -LiteralPath $baselinePath -Raw -Encoding UTF8 | ConvertFrom-Json
    $captured = [string]$baseline.captured_at_utc
    if (-not $captured) { throw "BASELINE_CAPTURE_TIME_MISSING" }
    $capturedAt = [datetime]::Parse($captured).ToUniversalTime()
    $prior = $null
    if ($null -ne $baseline.artifacts) {
        $prior = $baseline.artifacts.'data/results/daily_refresh_fetch_receipt.json'
    }
    if ($null -ne $prior -and $prior.exists -eq $true -and [string]$prior.sha256 -eq $stable.Sha256) {
        throw "FETCH_ARTIFACT_UNCHANGED_BEFORE_BASELINE"
    }
    if ($stable.WrittenUtc -le $capturedAt) { throw "FETCH_ARTIFACT_PREDATED_BASELINE" }
    $fetch = $script:StageReceipts[-1]
    $script:StageReceipts[-1] = [ordered]@{
        name                     = "fetch_raw_data"
        run_id                   = $runId
        exit_code                = 0
        outcome                  = "ok"
        status                   = "COMPLETE"
        artifact_path            = "data/results/daily_refresh_fetch_receipt.json"
        artifact_sha256          = $stable.Sha256
        artifact_size_bytes      = [int]$stable.Bytes.Length
        target_session           = $session
        artifact_written_at_utc  = $stable.WrittenUtc.ToString("yyyy-MM-ddTHH:mm:ssZ")
        baseline_captured_at_utc = $captured
        started_at               = [string]$fetch.started_at
        ended_at                 = [string]$fetch.ended_at
        artifact_after_baseline  = $true
    }
    Write-StageReceipts
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

if ($MyInvocation.InvocationName -ne '.') {
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
    Bind-FetchArtifact
    Run-PyStep -Name "stale_data_gate" -PyArgs @("-m", "services.stale_data_gate")
    Run-PyStep -Name "preprocess" -PyArgs @("-m", "services.preprocess")
    $completedSession = (
        & $py -c "from services.market_calendar import last_completed_nyse_session; print(last_completed_nyse_session().isoformat())"
    ).Trim()
    Run-PyStep -Name "train_model" -PyArgs @(
        "-m", "services.train_model",
        "--min-rows", "30",
        "--pipeline",
        "--run-id", $runId,
        "--completed-session", $completedSession
    )
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
}
