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
#   E. Rebuild intelligence   (outcomes, pnl, perf, risk overlay,
#       current-session opportunities + edge, allocation)
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
#
# -FunctionsOnly defines the handoff writer and returns before Set-Location,
# transcript, invocation-id creation, PENDING/READY writes, and broker stages.

param(
    [switch]$FunctionsOnly
)

$ErrorActionPreference = "Stop"

function Get-FileSha256Hex {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $fs = [System.IO.File]::OpenRead($Path)
        try {
            $hash = $sha.ComputeHash($fs)
            return ([BitConverter]::ToString($hash) -replace "-", "").ToLowerInvariant()
        } finally {
            $fs.Dispose()
        }
    } finally {
        $sha.Dispose()
    }
}

function Test-HandoffIsolatedRoot {
    param([Parameter(Mandatory = $true)][string]$Root)
    $full = [System.IO.Path]::GetFullPath($Root)
    $temp = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    if (-not $temp.EndsWith('\')) { $temp = $temp + '\' }
    if ($full.StartsWith($temp, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    foreach ($seg in ($full -split '[\\/]')) {
        if ($seg -like '*isolated*') { return $true }
    }
    return $false
}

function Assert-HandoffPathUnderRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $rootFull = [System.IO.Path]::GetFullPath($Root)
    if (-not $rootFull.EndsWith('\')) { $rootFull = $rootFull + '\' }
    $full = [System.IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "HANDOFF_OUTPUT_OUTSIDE_ROOT"
    }
}
function Resolve-HandoffResultsRoot {
    param(
        [string]$RepoRoot = "",
        [string]$ResultsRoot = ""
    )
    $repoBound = -not [string]::IsNullOrWhiteSpace($RepoRoot)
    $resultsBound = -not [string]::IsNullOrWhiteSpace($ResultsRoot)
    if (-not $repoBound -and -not $resultsBound) {
        return $null
    }
    $derived = $null
    if ($repoBound) {
        $derived = [System.IO.Path]::GetFullPath(
            (Join-Path ([System.IO.Path]::GetFullPath($RepoRoot)) "data\results")
        )
    }
    if ($resultsBound) {
        $given = [System.IO.Path]::GetFullPath($ResultsRoot)
        if ($null -ne $derived) {
            $left = $derived.TrimEnd('\')
            $right = $given.TrimEnd('\')
            if (-not $left.Equals($right, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "HANDOFF_ROOT_CONFLICT"
            }
        }
        else {
            $derived = $given
        }
    }
    return $derived
}

function Write-MarketOpenSequenceStatus {
    param(
        [Parameter(Mandatory = $true)] [string] $Status,
        [string] $FailedStep = "",
        [string] $Reason = "",
        [int] $ExitCode = 0,
        [string] $RepoRoot = "",
        [string] $ResultsRoot = ""
    )
    $repoBound = $PSBoundParameters.ContainsKey("RepoRoot") -and -not [string]::IsNullOrWhiteSpace($RepoRoot)
    $resultsBound = $PSBoundParameters.ContainsKey("ResultsRoot") -and -not [string]::IsNullOrWhiteSpace($ResultsRoot)
    if ($repoBound -or $resultsBound) {
        $checkRoot = $(if ($repoBound) { $RepoRoot } else { $ResultsRoot })
        if (-not (Test-HandoffIsolatedRoot -Root $checkRoot)) {
            throw "HANDOFF_ISOLATED_ROOT_REFUSED"
        }
        $statusDir = Resolve-HandoffResultsRoot -RepoRoot $(if ($repoBound) { $RepoRoot } else { "" }) -ResultsRoot $(if ($resultsBound) { $ResultsRoot } else { "" })
    }
    else {
        $statusDir = Join-Path $script:repoRoot "data\results"
    }
    if (-not (Test-Path $statusDir)) {
        New-Item -ItemType Directory -Path $statusDir -Force | Out-Null
    }
    $statusPath = Join-Path $statusDir "market_open_sequence_status.json"
    if ($repoBound -or $resultsBound) {
        Assert-HandoffPathUnderRoot -Root $statusDir -Path $statusPath
    }
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

function Write-Utf8NoBomAtomicFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $tmp = $Path + ".tmp"
    $encoding = New-Object System.Text.UTF8Encoding($false)
    $bytes = $encoding.GetBytes($Text)
    $fs = $null
    try {
        $fs = New-Object System.IO.FileStream(
            $tmp,
            [System.IO.FileMode]::Create,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        $fs.Write($bytes, 0, $bytes.Length)
        $fs.Flush($true)
    } finally {
        if ($null -ne $fs) { $fs.Dispose() }
    }
    # Refuse to replace dest with an incomplete tmp payload.
    if (-not (Test-Path -LiteralPath $tmp)) {
        throw "HANDOFF_TMP_MISSING"
    }
    $tmpLen = (Get-Item -LiteralPath $tmp).Length
    if ($tmpLen -le 0 -or $tmpLen -ne $bytes.Length) {
        throw "HANDOFF_TMP_INCOMPLETE"
    }
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}

function Write-InvocationHandoff {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("PENDING", "READY")][string]$Status,
        [string]$InterlockSha256 = "",
        [string]$FBindingSha256 = "",
        [string]$RepoRoot = "",
        [string]$ResultsRoot = "",
        [string]$InvocationId = "",
        [string]$SessionDate = "",
        [string]$IssuedAt = "",
        [string]$ReadyAt = "",
        [string]$GeneratedAt = "",
        [string]$HandoffPath = "",
        [string]$InterlockPath = "",
        [string]$ReadinessPath = "",
        [string]$ReadinessSha256 = "",
        [hashtable]$SourceArtifactHashes = $null,
        [string]$Mode = "",
        [object]$LiveExecutionPermitted = $null,
        [switch]$Isolated
    )

    $repoBound = $PSBoundParameters.ContainsKey("RepoRoot") -and -not [string]::IsNullOrWhiteSpace($RepoRoot)
    $resultsBound = $PSBoundParameters.ContainsKey("ResultsRoot") -and -not [string]::IsNullOrWhiteSpace($ResultsRoot)
    $explicitRoot = $repoBound -or $resultsBound
    $isolatedCall = [bool]$Isolated -or $explicitRoot
    $readyStamp = ""
    $generatedStamp = ""
    if ([string]::IsNullOrWhiteSpace($InvocationId)) {
        if ($isolatedCall) { throw "HANDOFF_ISOLATED_INVOCATION_ID_REQUIRED" }
        throw "HANDOFF_INVOCATION_ID_REQUIRED"
    }
    $invocation = $InvocationId

    if ($isolatedCall) {
        if (-not $explicitRoot) { throw "HANDOFF_ISOLATED_REPO_ROOT_REQUIRED" }
        if (-not $PSBoundParameters.ContainsKey("SessionDate") -or [string]::IsNullOrWhiteSpace($SessionDate)) {
            throw "HANDOFF_ISOLATED_SESSION_REQUIRED"
        }
        if (-not $PSBoundParameters.ContainsKey("IssuedAt") -or [string]::IsNullOrWhiteSpace($IssuedAt)) {
            throw "HANDOFF_ISOLATED_ISSUED_AT_REQUIRED"
        }
        $checkRoot = $(if ($repoBound) { $RepoRoot } else { $ResultsRoot })
        if (-not (Test-HandoffIsolatedRoot -Root $checkRoot)) {
            throw "HANDOFF_ISOLATED_ROOT_REFUSED"
        }
        if (-not $PSBoundParameters.ContainsKey("LiveExecutionPermitted")) {
            throw "HANDOFF_LIVE_EVIDENCE_MISSING"
        }
        if (($LiveExecutionPermitted -isnot [bool]) -or ($LiveExecutionPermitted -ne $false)) {
            throw "HANDOFF_LIVE_EVIDENCE_AMBIGUOUS"
        }
        $modeNorm = ""
        if ($null -ne $Mode) { $modeNorm = $Mode.Trim().ToUpperInvariant() }
        if (($modeNorm -ne "PAPER") -and ($modeNorm -ne "PAPER_ONLY")) {
            throw "HANDOFF_MODE_NOT_PAPER"
        }
        $resolvedResults = Resolve-HandoffResultsRoot -RepoRoot $(if ($repoBound) { $RepoRoot } else { "" }) -ResultsRoot $(if ($resultsBound) { $ResultsRoot } else { "" })
        $dest = Join-Path $resolvedResults "market_open_invocation_handoff.json"
        if ($PSBoundParameters.ContainsKey("HandoffPath") -and -not [string]::IsNullOrWhiteSpace($HandoffPath)) {
            $dest = $HandoffPath
        }
        Assert-HandoffPathUnderRoot -Root $resolvedResults -Path $dest
        if ($Status -eq "READY") {
            if ([string]::IsNullOrWhiteSpace($InterlockPath) -or [string]::IsNullOrWhiteSpace($InterlockSha256)) {
                throw "HANDOFF_INTERLOCK_REQUIRED"
            }
            if ((-not $PSBoundParameters.ContainsKey("ReadyAt")) -and (-not $PSBoundParameters.ContainsKey("GeneratedAt"))) {
                throw "HANDOFF_ISOLATED_TIME_REQUIRED"
            }
        }
        if (-not [string]::IsNullOrWhiteSpace($InterlockPath)) {
            Assert-HandoffPathUnderRoot -Root $resolvedResults -Path $InterlockPath
            $actualInterlock = Get-FileSha256Hex -Path $InterlockPath
            if ([string]::IsNullOrWhiteSpace($actualInterlock)) { throw "HANDOFF_INTERLOCK_UNREADABLE" }
            if (-not [string]::IsNullOrWhiteSpace($InterlockSha256)) {
                if ($actualInterlock.ToLowerInvariant() -ne $InterlockSha256.Trim().ToLowerInvariant()) {
                    throw "HANDOFF_INTERLOCK_HASH_MISMATCH"
                }
            }
        }
        if ($PSBoundParameters.ContainsKey("ReadinessPath") -or $PSBoundParameters.ContainsKey("ReadinessSha256")) {
            if ([string]::IsNullOrWhiteSpace($ReadinessPath) -or [string]::IsNullOrWhiteSpace($ReadinessSha256)) {
                throw "HANDOFF_READINESS_HASH_MISMATCH"
            }
            Assert-HandoffPathUnderRoot -Root $resolvedResults -Path $ReadinessPath
            $actualReady = Get-FileSha256Hex -Path $ReadinessPath
            if ([string]::IsNullOrWhiteSpace($actualReady)) { throw "HANDOFF_READINESS_HASH_MISMATCH" }
            if ($actualReady.ToLowerInvariant() -ne $ReadinessSha256.Trim().ToLowerInvariant()) {
                throw "HANDOFF_READINESS_HASH_MISMATCH"
            }
        }
        if ($null -ne $SourceArtifactHashes) {
            foreach ($key in @($SourceArtifactHashes.Keys)) {
                Assert-HandoffPathUnderRoot -Root $resolvedResults -Path ([string]$key)
                $actualSrc = Get-FileSha256Hex -Path ([string]$key)
                $expected = ([string]$SourceArtifactHashes[$key]).Trim().ToLowerInvariant()
                if ([string]::IsNullOrWhiteSpace($actualSrc) -or ($actualSrc.ToLowerInvariant() -ne $expected)) {
                    throw "HANDOFF_SOURCE_HASH_MISMATCH"
                }
            }
        }
        $session = $SessionDate
        $issued = $IssuedAt
        if (-not [string]::IsNullOrWhiteSpace($ReadyAt)) { $readyStamp = $ReadyAt }
        elseif (-not [string]::IsNullOrWhiteSpace($GeneratedAt)) { $readyStamp = $GeneratedAt }
        if (-not [string]::IsNullOrWhiteSpace($GeneratedAt)) { $generatedStamp = $GeneratedAt }
        elseif (-not [string]::IsNullOrWhiteSpace($ReadyAt)) { $generatedStamp = $ReadyAt }
        $destDir = Split-Path -Parent $dest
        if (-not (Test-Path -LiteralPath $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
    }
    else {
        $session = $script:HandoffSessionDate
        if ($PSBoundParameters.ContainsKey("SessionDate") -and -not [string]::IsNullOrWhiteSpace($SessionDate)) {
            $session = $SessionDate
        }
        $issued = $script:HandoffIssuedAt
        if ($PSBoundParameters.ContainsKey("IssuedAt") -and -not [string]::IsNullOrWhiteSpace($IssuedAt)) {
            $issued = $IssuedAt
        }
        $dest = $script:HandoffPath
        if ($PSBoundParameters.ContainsKey("HandoffPath") -and -not [string]::IsNullOrWhiteSpace($HandoffPath)) {
            $dest = $HandoffPath
        }
    }

    $payload = [ordered]@{
        schema_version            = 2
        status                    = $Status
        market_open_invocation_id = $invocation
        session_date              = $session
        issued_at                 = $issued
        source                    = "scripts/run_market_open.ps1"
        creates_no_authority      = $true
        live_execution_permitted  = $false
        broker_orders_submitted   = $false
    }
    if ($Status -eq "READY") {
        if ($Isolated) {
            $payload["ready_at"] = $readyStamp
            $payload["generated_at"] = $generatedStamp
        }
        else {
            $stamp = (Get-Date).ToUniversalTime().ToString("o")
            if ($PSBoundParameters.ContainsKey("ReadyAt") -and -not [string]::IsNullOrWhiteSpace($ReadyAt)) {
                $stamp = $ReadyAt
            }
            $payload["ready_at"] = $stamp
            $payload["generated_at"] = $stamp
            if ($PSBoundParameters.ContainsKey("GeneratedAt") -and -not [string]::IsNullOrWhiteSpace($GeneratedAt)) {
                $payload["generated_at"] = $GeneratedAt
            }
        }
        $payload["interlock_sha256"] = $InterlockSha256
        $payload["f_binding_sha256"] = $FBindingSha256
    }
    else {
        $payload["generated_at"] = $issued
        $payload["ready_at"] = $null
        $payload["interlock_sha256"] = $null
        $payload["f_binding_sha256"] = $null
    }
    $json = $payload | ConvertTo-Json
    try {
        Write-Utf8NoBomAtomicFile -Path $dest -Text $json
    }
    catch {
        Write-Host ("[INVOCATION] write_failed status=" + $Status + " err=" + $_.Exception.Message) -ForegroundColor Red
        throw
    }
    Write-Host ("[INVOCATION] status=" + $Status + " id=" + $invocation) -ForegroundColor Cyan
}

# Supersede any prior READY immediately — PENDING is never entry-usable.
if ($FunctionsOnly) {
    return
}

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

# Current-cycle invocation ID for F / F2 (inherited by child python in this process).
# Continuous-loop parent cannot inherit this env after this script exits, so also
# write an explicit two-phase handoff for paper_entry_loop_stage (Phase 198K-A/B):
#   PENDING at start (never entry-usable; supersedes prior READY)
#   READY after F2 (entry-usable only with matching F/F2 binding)
$script:MarketOpenInvocationId = (
    "mo-" + $runStamp + "-" + [guid]::NewGuid().ToString("N")
)
$env:TRITON_MARKET_OPEN_INVOCATION_ID = $script:MarketOpenInvocationId
$script:HandoffIssuedAt = (Get-Date).ToUniversalTime().ToString("o")
try {
    $tzEt = [TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
    $script:HandoffSessionDate = [TimeZoneInfo]::ConvertTimeFromUtc(
        [DateTime]::UtcNow, $tzEt
    ).ToString("yyyy-MM-dd")
} catch {
    $script:HandoffSessionDate = (Get-Date).ToString("yyyy-MM-dd")
}

$handoffDir = Join-Path $repoRoot "data\results"
if (-not (Test-Path $handoffDir)) {
    New-Item -ItemType Directory -Path $handoffDir -Force | Out-Null
}
$script:HandoffPath = Join-Path $handoffDir "market_open_invocation_handoff.json"
Write-InvocationHandoff -Status "PENDING" -InvocationId $script:MarketOpenInvocationId
Write-Host ("[INVOCATION] handoff=" + $script:HandoffPath) -ForegroundColor Cyan

# ── Run-Step: invoke a python module and abort on non-zero exit ──────
function Run-Step {
    param(
        [Parameter(Mandatory = $true)] [string]   $Label,
        [Parameter(Mandatory = $true)] [string[]] $PyArgs
    )
    Write-Host ""
    Write-Host "[$Label]" -ForegroundColor Yellow
    Write-Host (">>> python " + ($PyArgs -join " ")) -ForegroundColor Cyan
    # Capture stdout/stderr so a nonzero python exit (especially F) retains
    # the inner transcript. Windows PowerShell 5.1 turns 2>&1 ErrorRecords
    # into terminating errors under $ErrorActionPreference=Stop.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & python @PyArgs 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($null -eq $code) { $code = 1 }
    foreach ($line in @($output)) {
        $text = if ($null -eq $line) { "" } else { [string]$line }
        if ($text -match '(?i)APCA-API|SECRET[_-]?KEY|api[_-]?secret|BEGIN (RSA |OPENSSH )?PRIVATE') {
            $text = "[REDACTED]"
        }
        Write-Host $text
    }
    if ([int]$code -ne 0) {
        Write-Host ""
        Write-Host ("[FAIL] Step '" + $Label + "' failed (exit $code).") -ForegroundColor Red
        Write-Host ("[FAIL] Command: python " + ($PyArgs -join " ")) -ForegroundColor Red
        Write-Host "[FAIL] Aborting market-open sequence." -ForegroundColor Red
        # Use throw (not `exit`) so the outer try/finally can stop the
        # transcript cleanly while still preserving the python rc.
        throw ("STEP_FAILED:" + $Label + ":" + $code)
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

    # ── E4.5 / E4.6. Current-session opportunities + edge before allocation ──
    # These producers must not abort the sequence: EXIT/TRIM still need E5.
    # A failed or incomplete producer replaces any prior edge file so the June
    # CSV cannot remain current evidence.
    Write-Host ""
    Write-Host "[E4.5 BUILD TRADE OPPORTUNITIES]" -ForegroundColor Yellow
    Write-Host ">>> python -m services.build_trade_opportunities" -ForegroundColor Cyan
    $prevEapOpps = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $oppsOut = & python -m services.build_trade_opportunities 2>&1
        $oppsRc = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEapOpps
    }
    if ($null -eq $oppsRc) { $oppsRc = 1 }
    foreach ($line in @($oppsOut)) {
        $text = if ($null -eq $line) { "" } else { [string]$line }
        Write-Host $text
    }
    $edgeRc = 2
    if ([int]$oppsRc -eq 0) {
        Write-Host ""
        Write-Host "[E4.6 EDGE SIZING ENGINE]" -ForegroundColor Yellow
        Write-Host ">>> python -m services.edge_sizing_engine" -ForegroundColor Cyan
        $prevEapEdge = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $edgeOut = & python -m services.edge_sizing_engine 2>&1
            $edgeRc = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $prevEapEdge
        }
        if ($null -eq $edgeRc) { $edgeRc = 1 }
        foreach ($line in @($edgeOut)) {
            $text = if ($null -eq $line) { "" } else { [string]$line }
            Write-Host $text
        }
    } else {
        Write-Host ""
        Write-Host "[E4.6 EDGE SIZING ENGINE] skipped (opportunities producer failed); invalidating prior edge evidence" -ForegroundColor Yellow
    }
    if (([int]$oppsRc -ne 0) -or ([int]$edgeRc -ne 0)) {
        Write-Host "[FAIL-CLOSED] replacing prior edge file so it is not current evidence" -ForegroundColor Yellow
        $prevEapInv = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & python -m services.edge_sizing_engine --invalidate-current-evidence 2>&1 | ForEach-Object { Write-Host $_ }
        } finally {
            $ErrorActionPreference = $prevEapInv
        }
    }

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

    # Finalize invocation handoff only after F/F2 evidence exists for this ID.
    $interlockPath = Join-Path $repoRoot "data\results\paper_sell_exit_before_entry.json"
    $fBindingPath = Join-Path $repoRoot "data\results\manage_positions_cycle_binding.json"
    $interlockSha = Get-FileSha256Hex -Path $interlockPath
    $fBindingSha = Get-FileSha256Hex -Path $fBindingPath
    if (-not $interlockSha) {
        throw "STEP_FAILED:F2. PAPER SELL READINESS PACKET:HANDOFF_INTERLOCK_MISSING"
    }
    Write-InvocationHandoff -Status "READY" `
        -InvocationId $script:MarketOpenInvocationId `
        -InterlockSha256 $interlockSha `
        -FBindingSha256 $(if ($fBindingSha) { $fBindingSha } else { "" })

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
