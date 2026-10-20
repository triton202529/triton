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

$script:DailyRefreshScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
. (Join-Path $script:DailyRefreshScriptRoot "scripts\loop_lock_utils.ps1")

function ConvertTo-UtcRoundTripText {
    param([datetime]$Value)
    $utc = [datetime]::SpecifyKind($Value.ToUniversalTime(), [datetimekind]::Utc)
    $text = $utc.ToString("o")
    if ($text.EndsWith(".0000000Z")) {
        return $utc.ToString("yyyy-MM-ddTHH:mm:ss'Z'")
    }
    return $text
}


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
$runStartedAt = ConvertTo-UtcRoundTripText (Get-Date)
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

function Get-Sha256Hex {
    param([byte[]] $Bytes)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Get-ShortFailure {
    param([string] $Message)
    $flat = ([string]$Message -replace "\s+", " ").Trim()
    if ($flat.Length -gt 160) { return $flat.Substring(0, 160) }
    return $flat
}

function Test-JsonNumber {
    param($Value)
    return (
        $Value -is [byte] -or
        $Value -is [int] -or
        $Value -is [long] -or
        $Value -is [double] -or
        $Value -is [decimal] -or
        $Value -is [single]
    )
}

function Read-JsonObjectBytes {
    param(
        [byte[]] $Bytes,
        [string] $Failure = "FETCH_ARTIFACT_JSON_PARSE"
    )
    $text = [System.Text.Encoding]::UTF8.GetString($Bytes)
    if ($text.Length -gt 0 -and [int][char]$text[0] -eq 0xFEFF) {
        $text = $text.Substring(1)
    }
    try {
        return ConvertFrom-Json -InputObject $text
    } catch {
        throw ($Failure + ": " + (Get-ShortFailure $_.Exception.Message))
    }
}

function ConvertFrom-UtcRoundTripText {
    param([string]$Text)
    $styles = [System.Globalization.DateTimeStyles]::RoundtripKind
    $parsed = [datetime]::Parse(
        $Text,
        [System.Globalization.CultureInfo]::InvariantCulture,
        $styles
    )
    return [datetime]::SpecifyKind($parsed.ToUniversalTime(), [datetimekind]::Utc)
}

function Enter-DailyRefreshMutationLock {
    param([Parameter(Mandatory = $true)][string]$LiveDir)
    New-Item -ItemType Directory -Force -Path $LiveDir | Out-Null
    $own = Join-Path $LiveDir "daily_refresh.lock"
    $other = Join-Path $LiveDir "data_freshness_refresh.lock"
    Clear-OrphanLoopLock -LockFile $own -ScriptName "run_daily_refresh.ps1" | Out-Null
    if (Test-Path -LiteralPath $own) { return $false }
    Clear-OrphanLoopLock -LockFile $other -ScriptName "run_data_freshness_refresh.ps1" | Out-Null
    if (Test-Path -LiteralPath $other) { return $false }
    try {
        $stream = New-Object System.IO.FileStream(
            $own,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        $stream.Dispose()
    } catch {
        return $false
    }
    Update-LoopLockMetadata -LockFile $own -Source "daily_refresh" -Cycle 1
    return $true
}

function Exit-DailyRefreshMutationLock {
    param([Parameter(Mandatory = $true)][string]$LiveDir)
    $own = Join-Path $LiveDir "daily_refresh.lock"
    if (-not (Test-Path -LiteralPath $own)) { return }
    try {
        $content = Get-Content -LiteralPath $own -Raw -ErrorAction Stop
    } catch {
        return
    }
    if ($content -match 'pid=(\d+)' -and [int]$Matches[1] -eq $PID) {
        Remove-Item -LiteralPath $own -Force
    }
}

function Invoke-LockedDailyRefreshWindow {
    param(
        [Parameter(Mandatory = $true)][string]$LiveDir,
        [Parameter(Mandatory = $true)][scriptblock]$Body
    )
    if (-not (Enter-DailyRefreshMutationLock -LiveDir $LiveDir)) {
        throw "DAILY_REFRESH_LOCK_REFUSED"
    }
    try {
        & $Body
    } finally {
        Exit-DailyRefreshMutationLock -LiveDir $LiveDir
    }
}

function Bind-FetchArtifact {
    $fetchPath = Join-Path $resultsDir "daily_refresh_fetch_receipt.json"
    if (-not (Test-Path -LiteralPath $fetchPath -PathType Leaf)) { throw "FETCH_ARTIFACT_MISSING" }
    $script:FetchBindingPublishAttempts = 0
    $first = [System.IO.File]::ReadAllBytes($fetchPath)
    $hash1 = Get-Sha256Hex -Bytes $first
    if ($null -ne $script:FetchBindingAfterRead) { & $script:FetchBindingAfterRead $fetchPath }
    $second = [System.IO.File]::ReadAllBytes($fetchPath)
    $hash2 = Get-Sha256Hex -Bytes $second
    if ($hash1 -ne $hash2 -or $first.Length -ne $second.Length) { throw "FETCH_ARTIFACT_CHANGED" }
    $script:FetchBindingReadAt = [datetime]::SpecifyKind((Get-Date).ToUniversalTime(), [datetimekind]::Utc)
    $written = [datetime]::SpecifyKind([System.IO.File]::GetLastWriteTimeUtc($fetchPath), [datetimekind]::Utc)
    $receipt = Read-JsonObjectBytes -Bytes $first
    if ($receipt -isnot [System.Management.Automation.PSCustomObject]) { throw "FETCH_ARTIFACT_NOT_OBJECT" }
    $passedProp = $receipt.PSObject.Properties["passed"]
    if ($null -eq $passedProp) { throw "FETCH_ARTIFACT_PASSED_MISSING" }
    if ($passedProp.Value -isnot [bool]) { throw "FETCH_ARTIFACT_PASSED_NOT_BOOLEAN" }
    if ($passedProp.Value -ne $true) { throw "FETCH_ARTIFACT_FAILED" }
    $coverageProp = $receipt.PSObject.Properties["tradable_coverage"]
    if ($null -eq $coverageProp -or $coverageProp.Value -isnot [System.Management.Automation.PSCustomObject]) {
        throw "FETCH_ARTIFACT_COVERAGE_SHAPE"
    }
    $coverage = $coverageProp.Value
    foreach ($name in @("ratio", "covered", "denominator")) {
        $prop = $coverage.PSObject.Properties[$name]
        if ($null -eq $prop -or -not (Test-JsonNumber $prop.Value)) { throw "FETCH_ARTIFACT_COVERAGE_SHAPE" }
    }
    $denominator = [double]$coverage.denominator
    $covered = [double]$coverage.covered
    $ratio = [double]$coverage.ratio
    if ($denominator -le 0 -or [math]::Abs(($covered / $denominator) - $ratio) -gt 0.0000001) {
        throw "FETCH_ARTIFACT_COVERAGE_SHAPE"
    }
    if ($ratio + 0.000000000001 -lt 0.90) { throw "FETCH_ARTIFACT_COVERAGE" }
    $decisionProp = $receipt.PSObject.Properties["decisions"]
    if ($null -eq $decisionProp -or $null -eq $decisionProp.Value) { throw "FETCH_ARTIFACT_DECISIONS_SHAPE" }
    $decisions = @($decisionProp.Value)
    if ($decisions.Count -lt 1) { throw "FETCH_ARTIFACT_DECISIONS_SHAPE" }
    foreach ($row in $decisions) {
        if ($row -isnot [System.Management.Automation.PSCustomObject]) { throw "FETCH_ARTIFACT_DECISIONS_SHAPE" }
        $symbolProp = $row.PSObject.Properties["symbol"]
        if ($null -eq $symbolProp -or [string]::IsNullOrWhiteSpace([string]$symbolProp.Value)) {
            throw "FETCH_ARTIFACT_DECISIONS_SHAPE"
        }
    }
    $sessionProp = $receipt.PSObject.Properties["target_session"]
    if ($null -eq $sessionProp -or [string]::IsNullOrWhiteSpace([string]$sessionProp.Value)) {
        $sessionProp = $receipt.PSObject.Properties["completed_session"]
    }
    if ($null -eq $sessionProp -or [string]::IsNullOrWhiteSpace([string]$sessionProp.Value)) {
        throw "FETCH_ARTIFACT_SESSION_MISSING"
    }
    $session = [string]$sessionProp.Value
    $script:TargetSession = $session
    $activeSession = (
        & $py -c "from services.market_calendar import last_completed_nyse_session; print(last_completed_nyse_session().isoformat())"
    )
    if ($null -eq $activeSession) { throw "FETCH_ARTIFACT_SESSION_UNAVAILABLE" }
    $activeSession = $activeSession.Trim()
    if (-not $activeSession) { throw "FETCH_ARTIFACT_SESSION_UNAVAILABLE" }
    if ($session -ne $activeSession) { throw "FETCH_ARTIFACT_WRONG_SESSION" }
    if (-not (Test-Path -LiteralPath $baselinePath -PathType Leaf)) { throw "FETCH_ARTIFACT_BASELINE_MISSING" }
    $baselineBytes = [System.IO.File]::ReadAllBytes($baselinePath)
    $baseline = Read-JsonObjectBytes -Bytes $baselineBytes -Failure "FETCH_ARTIFACT_BASELINE_JSON"
    $baselineText = [System.Text.Encoding]::UTF8.GetString($baselineBytes)
    $capturedMatch = [regex]::Match($baselineText, '"captured_at_utc"\s*:\s*"([^"]+)"')
    if (-not $capturedMatch.Success) { throw "BASELINE_CAPTURE_TIME_MISSING" }
    $captured = $capturedMatch.Groups[1].Value
    $capturedAt = ConvertFrom-UtcRoundTripText $captured
    $prior = $null
    if ($null -ne $baseline.PSObject.Properties["artifacts"]) {
        $prior = $baseline.artifacts.'data/results/daily_refresh_fetch_receipt.json'
    }
    if ($null -ne $prior -and $prior.exists -eq $true -and [string]$prior.sha256 -eq $hash1) {
        throw "FETCH_ARTIFACT_UNCHANGED_BEFORE_BASELINE"
    }
    if ($written.Year -lt 2000) { throw "FETCH_ARTIFACT_PROVENANCE_MISSING" }
    if ($written -le $capturedAt) { throw "FETCH_ARTIFACT_PREDATED_BASELINE" }
    if ($null -eq $script:StageReceipts -or @($script:StageReceipts).Count -lt 1) {
        throw "FETCH_ARTIFACT_STAGE_MISSING"
    }
    $fetch = @($script:StageReceipts)[-1]
    $started = ConvertFrom-UtcRoundTripText ([string]$fetch.started_at)
    $ended = ConvertFrom-UtcRoundTripText ([string]$fetch.ended_at)
    if ($written -lt $started -or $written -gt $ended -or $written -gt $script:FetchBindingReadAt) {
        throw "FETCH_ARTIFACT_PROVENANCE_WINDOW"
    }
    $bound = [ordered]@{
        name                     = "fetch_raw_data"
        run_id                   = $runId
        exit_code                = 0
        outcome                  = "ok"
        status                   = "COMPLETE"
        artifact_path            = "data/results/daily_refresh_fetch_receipt.json"
        artifact_sha256          = $hash1
        artifact_size_bytes      = [int]$first.Length
        target_session           = $session
        artifact_written_at_utc  = (ConvertTo-UtcRoundTripText $written)
        baseline_captured_at_utc = $captured
        started_at               = [string]$fetch.started_at
        ended_at                 = [string]$fetch.ended_at
        artifact_after_baseline  = $true
    }
    $previous = @($script:StageReceipts)
    $updated = @($script:StageReceipts)
    $updated[-1] = $bound
    $script:StageReceipts = $updated
    $script:FetchBindingPublishAttempts = 1
    try {
        Write-StageReceipts
    } catch {
        $script:StageReceipts = $previous
        throw
    }
}
function Run-PyStep {
    param(
        [Parameter(Mandatory = $true)] [string] $Name,
        [Parameter(Mandatory = $true)] [string[]] $PyArgs
    )
    $started = ConvertTo-UtcRoundTripText (Get-Date)
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
    $ended = ConvertTo-UtcRoundTripText (Get-Date)
    Log ("EXIT: " + $Name + " code=" + $code)
    $script:StageReceipts += [ordered]@{
        name            = $Name
        run_id          = $runId
        target_session  = [string]$script:TargetSession
        started_at      = $started
        ended_at        = $ended
        exit_code       = [int]$code
        outcome         = $(if ([int]$code -eq 0) { "ok" } else { "failed" })
        status          = $(if ([int]$code -eq 0) { "COMPLETE" } else { "FAILED" })
    }
    Write-StageReceipts
    if ([int]$code -ne 0) {
        Log ("FAILED: " + $Name + " (exit=" + $code + "). See log: " + $logPath)
        throw ("STEP_FAILED:" + $Name + ":" + $code)
    }
}

if ($MyInvocation.InvocationName -ne '.') {
$liveDir = Join-Path $repoRoot "data\live"
$exitCode = 0
try {
    Invoke-LockedDailyRefreshWindow -LiveDir $liveDir -Body {
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
