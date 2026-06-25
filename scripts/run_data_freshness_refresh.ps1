# scripts/run_data_freshness_refresh.ps1
# Phase 196K — Read-only dashboard data freshness refresh (no trading/execution).

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton",
    [switch]$ForceLightOnly
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repo root not found: $RepoRoot"
}
$RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
Set-Location $RepoRoot

$scriptDir = Join-Path $RepoRoot "scripts"
. (Join-Path $scriptDir "loop_lock_utils.ps1")

$liveDir = Join-Path $RepoRoot "data\live"
$resultsDir = Join-Path $RepoRoot "data\results"
$lockFile = Join-Path $liveDir "data_freshness_refresh.lock"
$auditPath = Join-Path $resultsDir "data_freshness_refresh_audit.json"
$logDir = Join-Path $resultsDir "data_freshness_refresh"
$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    $py = "python"
}

New-Item -ItemType Directory -Force -Path $liveDir, $resultsDir, $logDir | Out-Null

$scriptName = "run_data_freshness_refresh.ps1"
Clear-OrphanLoopLock -LockFile $lockFile -ScriptName $scriptName | Out-Null

if (Test-Path -LiteralPath $lockFile) {
    if (Test-LoopProcessRunning -ScriptName $scriptName) {
        Write-Host "[DATA_FRESHNESS] blocked: another refresh is running" -ForegroundColor Yellow
        exit 0
    }
    Write-Host "[DATA_FRESHNESS] stale lock without process; clearing" -ForegroundColor Yellow
    Remove-Item -LiteralPath $lockFile -Force
}

Update-LoopLockMetadata -LockFile $lockFile -Source "data_freshness_refresh" -Cycle 1

$startedAt = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir ("data_freshness_refresh_" + $startedAt + ".log")

function Log-Line($msg) {
    $line = "$(Get-Date -Format s) $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

function Invoke-ReadOnlyStep {
    param(
        [Parameter(Mandatory = $true)] [string]$Label,
        [Parameter(Mandatory = $true)] [string[]]$PyArgs
    )
    Log-Line "[$Label] python $($PyArgs -join ' ')"
    & $py @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw ("STEP_FAILED:" + $Label + ":" + $LASTEXITCODE)
    }
}

function Test-ContinuousLoopActive {
    param([int]$RecentMinutes = 20)
    $clLock = Join-Path $liveDir "continuous_loop.lock"
    if (-not (Test-Path -LiteralPath $clLock)) {
        return $false
    }
    try {
        $content = Get-Content -LiteralPath $clLock -Raw
        if ($content -match 'refreshed_at=([^\r\n]+)') {
            $refreshed = [datetime]::Parse($Matches[1].Trim())
            $ageMin = ((Get-Date) - $refreshed).TotalMinutes
            return ($ageMin -le $RecentMinutes)
        }
    } catch {
        return $false
    }
    return $false
}

$stepsRun = @()
$errors = @()
$lightOnly = [bool]$ForceLightOnly.IsPresent
$exitCode = 0

try {
    Log-Line "=== TRITON DATA FRESHNESS REFRESH START ==="
    Log-Line "RepoRoot=$RepoRoot"

    if (Test-ContinuousLoopActive -RecentMinutes 20) {
        $lightOnly = $true
        Log-Line "[MODE] light_only=true (continuous loop recently active)"
    }

    Invoke-ReadOnlyStep -Label "SNAPSHOT_LIVE_ORDERS" `
        -PyArgs @("-m", "services.snapshot_live_orders", "--mode", "paper")
    $stepsRun += "snapshot_live_orders"

    Invoke-ReadOnlyStep -Label "TOUCH_PIPELINE_HEARTBEAT" `
        -PyArgs @("-m", "services.touch_pipeline_heartbeat")
    $stepsRun += "touch_pipeline_heartbeat"

    try {
        Log-Line "[PAPER_SESSION_ARM] bounded renew if session window permits"
        & $py -m services.paper_session_arm_policy --json 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Log-Line "[PAPER_SESSION_ARM] non-fatal exit=$LASTEXITCODE"
        } else {
            $stepsRun += "paper_session_arm_policy"
        }
    } catch {
        Log-Line "[PAPER_SESSION_ARM] skipped: $($_.Exception.Message)"
    }

    try {
        Log-Line "[PORTFOLIO_HISTORY_APPEND_IF_STALE] append if positions newer than history"
        $appendOut = & $py -c "from services.portfolio_history_sync import append_portfolio_history_if_stale; import json; print(json.dumps(append_portfolio_history_if_stale()))"
        if ($LASTEXITCODE -ne 0) {
            throw "STEP_FAILED:PORTFOLIO_HISTORY_APPEND:$LASTEXITCODE"
        }
        if ($appendOut) { Log-Line "[PORTFOLIO_HISTORY] $appendOut" }
        $stepsRun += "portfolio_history_append_if_stale"
    } catch {
        Log-Line "[PORTFOLIO_HISTORY] skipped: $($_.Exception.Message)"
        $errors += "portfolio_history:$($_.Exception.Message)"
    }

    if (-not $lightOnly) {
        foreach ($step in @(
            @{ Label = "BUILD_TRADE_OUTCOMES"; Args = @("-m", "services.build_trade_outcomes") },
            @{ Label = "BUILD_PNL_DIAGNOSTICS"; Args = @("-m", "services.build_pnl_diagnostics") },
            @{ Label = "BUILD_PERFORMANCE_INTELLIGENCE"; Args = @("-m", "services.build_performance_intelligence") }
        )) {
            try {
                Invoke-ReadOnlyStep -Label $step.Label -PyArgs $step.Args
                $stepsRun += $step.Label.ToLower()
            } catch {
                Log-Line "[$($step.Label)] skipped: $($_.Exception.Message)"
                $errors += "$($step.Label):$($_.Exception.Message)"
            }
        }
    } else {
        Log-Line "[MODE] skipped optional diagnostics (light_only)"
    }

    Log-Line "=== TRITON DATA FRESHNESS REFRESH COMPLETE ==="
    $exitCode = 0
}
catch {
    $msg = $_.Exception.Message
    if ($msg -match '^STEP_FAILED:(.+):(\d+)$') {
        $errors += "failed_step=$($Matches[1]) rc=$($Matches[2])"
        Log-Line "[FAIL] step=$($Matches[1]) exit=$($Matches[2])"
    } else {
        $errors += "exception=$msg"
        Log-Line "[FAIL] $msg"
    }
    $exitCode = 1
}
finally {
    if (Test-Path -LiteralPath $lockFile) {
        Remove-Item -LiteralPath $lockFile -Force
    }
}

$audit = [ordered]@{
    timestamp          = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    phase              = "196K"
    repo_root          = $RepoRoot
    light_only         = $lightOnly
    steps_run          = @($stepsRun)
    errors             = @($errors)
    log_file           = $logFile
    live_execution     = $false
    execute_trades_run = $false
}
$audit | ConvertTo-Json -Depth 4 | Set-Content -Path $auditPath -Encoding UTF8
Log-Line "[AUDIT] $auditPath"

if ($exitCode -ne 0) {
    exit $exitCode
}
exit 0
