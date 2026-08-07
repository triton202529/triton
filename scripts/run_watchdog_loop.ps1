# scripts/run_watchdog_loop.ps1
# TRITON Risk Watchdog loop — read-only monitoring between intelligence cycles.
param(
    [int]$IntervalMinutes = 1,
    [int]$MaxCycles = 0
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

. (Join-Path $scriptDir "loop_lock_utils.ps1")

$loopLogDir = "data\results\watchdog_logs"
New-Item -ItemType Directory -Force -Path $loopLogDir | Out-Null
New-Item -ItemType Directory -Force -Path "data\live" | Out-Null
New-Item -ItemType Directory -Force -Path "data\results" | Out-Null

$startedAt = Get-Date -Format "yyyyMMdd_HHmmss"
$loopLog = Join-Path $loopLogDir "watchdog_loop_$startedAt.log"
$lockFile = "data\live\watchdog_loop.lock"

Clear-OrphanLoopLock -LockFile $lockFile -ScriptName "run_watchdog_loop.ps1" | Out-Null

if (Test-Path $lockFile) {
    Write-Host "[BLOCKED] Watchdog loop lock already exists: $lockFile" -ForegroundColor Red
    Write-Host "Delete it only if you are sure no watchdog loop is running."
    exit 1
}

Update-LoopLockMetadata -LockFile $lockFile -Source "watchdog_loop" -Cycle 0

function Log-Line($msg) {
    # Best-effort logging only. With $ErrorActionPreference=Stop, a sharing
    # violation on Add-Content previously aborted the entire watchdog after a
    # successful SLEEP_POLL (2026-08-06 PID 14352). Never let log I/O kill the loop.
    $line = "$(Get-Date -Format s) $msg"
    Write-Host $line
    try {
        Add-Content -Path $loopLog -Value $line -ErrorAction Stop
    } catch {
        try {
            Add-Content -Path ($loopLog + ".fallback") -Value $line -ErrorAction SilentlyContinue
        } catch {
            # ignore fallback failures
        }
        Write-Host ("[LOG_WRITE_FAILED] " + $_.Exception.Message) -ForegroundColor Yellow
    }
}

function Write-TerminalReason {
    param(
        [Parameter(Mandatory = $true)] [string] $Reason,
        [string] $Detail = ""
    )
    if ($Detail) {
        Log-Line "[TERMINAL_REASON] reason=$Reason detail=$Detail"
    } else {
        Log-Line "[TERMINAL_REASON] reason=$Reason"
    }
}

# DO-004: cooperative shutdown helper (Python). Exit code 10 = accepted, exit now.
# Capture helper stdout/stderr before returning the exit code so diagnostic
# output cannot pollute the caller's integer comparison ($code -eq 10).
function Test-CooperativeShutdown {
    param([string]$CycleState = "BETWEEN_CYCLES")
    $out = python -m services.operator_cooperative_shutdown `
        --component watchdog --pid $PID --mode check --cycle-state $CycleState 2>&1
    $code = $LASTEXITCODE
    Log-Line "[SHUTDOWN_POLL] cycle_state=$CycleState exit_code=$code output=$out"
    # Return exactly one integer (unary comma avoids accidental array unwrap quirks).
    return ,$code
}

function Invoke-InterruptibleSleep {
    param([int]$TotalSeconds)
    # Capture exit code before logging: a pipeline into ForEach can clobber $LASTEXITCODE.
    Log-Line "[SLEEP] interruptible wait ${TotalSeconds}s (check every ~5s for cooperative shutdown)"
    $sleepOut = python -m services.operator_cooperative_shutdown `
        --component watchdog --pid $PID --mode sleep --seconds $TotalSeconds 2>&1
    $code = $LASTEXITCODE
    foreach ($line in @($sleepOut)) { Log-Line "[SLEEP_POLL] $line" }
    return ,$code
}

function Remove-OwnLoopLock {
    if (-not (Test-Path $lockFile)) { return }
    try {
        $text = Get-Content -Path $lockFile -Raw -ErrorAction SilentlyContinue
        if ($text -match 'pid=(\d+)' -and [int]$Matches[1] -ne $PID) {
            Log-Line "[LOCK_SKIP] lock owned by pid=$($Matches[1]); not removing (this pid=$PID)"
            return
        }
    } catch {
        Log-Line "[LOCK_CHECK] could not parse lock ownership: $($_.Exception.Message)"
    }
    Remove-Item $lockFile -Force
    Log-Line "[LOCK_RELEASE] released own watchdog_loop.lock pid=$PID"
}

try {
    Log-Line "=== TRITON RISK WATCHDOG LOOP START ==="
    Log-Line "RepoRoot=$repoRoot"
    Log-Line "IntervalMinutes=$IntervalMinutes"
    Log-Line "MaxCycles=$MaxCycles"
    Log-Line "LoopLog=$loopLog"
    Log-Line "[SAFE_BOUNDARY] PRE_CYCLE | POST_CYCLE_PRE_SLEEP | INTERRUPTIBLE_WAIT (DO-004)"

    $graceMinutes = [Math]::Max(50, $IntervalMinutes + 10)
    $hbRefresh = python -m services.loop_safety --refresh --source watchdog_loop `
        --grace-minutes $graceMinutes 2>&1
    Log-Line "[HEARTBEAT_REFRESH] grace_minutes=$graceMinutes output=$hbRefresh"

    $cycle = 0
    $shutdownAccepted = $false

    $terminalReason = ""
    while ($true) {
        # Safe boundary: before beginning a new watchdog cycle
        # Normalize helper return to a single int (guards against array coercion).
        $pollCode = @((Test-CooperativeShutdown -CycleState "BETWEEN_CYCLES"))[-1]
        if ($pollCode -eq 10) {
            Log-Line "[SHUTDOWN_ACCEPTED] at PRE_CYCLE safe boundary; no new cycle will start"
            $shutdownAccepted = $true
            $terminalReason = "COOPERATIVE_SHUTDOWN_PRE_CYCLE"
            break
        }
        # 0 = no request, 11 = refused/expired request (continue). Any other code is helper
        # failure — refuse a new monitoring cycle and wait boundedly (do not treat as accept).
        if (($null -eq $pollCode) -or (($pollCode -ne 0) -and ($pollCode -ne 11))) {
            Log-Line "[SHUTDOWN_HELPER_DEGRADED] pre_cycle exit_code=$pollCode; refusing new cycle; wait 30s"
            Start-Sleep -Seconds 30
            continue
        }

        $cycle++
        Log-Line ""
        Log-Line "[CYCLE_START] cycle=$cycle"
        Update-LoopLockMetadata -LockFile $lockFile -Source "watchdog_loop" -Cycle $cycle

        python -m services.risk_watchdog --expected-interval-minutes $IntervalMinutes
        $rc = $LASTEXITCODE

        $activeAlerts = 0
        $alertsFile = "data\results\watchdog_alerts.json"
        if (Test-Path $alertsFile) {
            try {
                $alertsJson = Get-Content $alertsFile -Raw | ConvertFrom-Json
                if ($alertsJson.active_alerts) {
                    $activeAlerts = @($alertsJson.active_alerts).Count
                }
            } catch {
                $activeAlerts = -1
            }
        }

        if ($rc -ne 0) {
            Log-Line "[CYCLE_FAIL] cycle=$cycle exit_code=$rc active_alerts=$activeAlerts"
        } else {
            Log-Line "[CYCLE_OK] cycle=$cycle active_alerts=$activeAlerts"
        }

        # Safe boundary: after bounded status/alert persistence; before sleep
        $pollCode = @((Test-CooperativeShutdown -CycleState "POST_CYCLE"))[-1]
        if ($pollCode -eq 10) {
            Log-Line "[SHUTDOWN_ACCEPTED] at POST_CYCLE_PRE_SLEEP safe boundary; skipping further cycles"
            $shutdownAccepted = $true
            $terminalReason = "COOPERATIVE_SHUTDOWN_POST_CYCLE"
            break
        }

        if ($MaxCycles -gt 0 -and $cycle -ge $MaxCycles) {
            Log-Line "[LOOP_COMPLETE] reached MaxCycles=$MaxCycles"
            $terminalReason = "MAX_CYCLES"
            break
        }

        $sleepCode = @((Invoke-InterruptibleSleep -TotalSeconds ($IntervalMinutes * 60)))[-1]
        if ($sleepCode -eq 10) {
            Log-Line "[SHUTDOWN_ACCEPTED] during INTERRUPTIBLE_WAIT; exiting without new cycle"
            $shutdownAccepted = $true
            $terminalReason = "COOPERATIVE_SHUTDOWN_INTERRUPTIBLE_WAIT"
            break
        }
        # Sleep mode emits 0 or 10 only. Any other code (incl. 11) is treated as helper
        # failure so an early non-zero return cannot open a new cycle / shorten cadence.
        if (($null -eq $sleepCode) -or ($sleepCode -ne 0)) {
            Log-Line "[SHUTDOWN_HELPER_DEGRADED] sleep exit_code=$sleepCode; refusing new cycle; wait 30s"
            Start-Sleep -Seconds 30
        }
    }

    if ($shutdownAccepted) {
        Log-Line "[SHUTDOWN_EXIT] cooperative shutdown complete for watchdog loop"
    }
    if (-not $terminalReason) {
        $terminalReason = "LOOP_ENDED_WITHOUT_MARKED_REASON"
    }
    Write-TerminalReason -Reason $terminalReason
}
catch {
    $abortDetail = $_.Exception.Message
    Log-Line "[LOOP_ABORT] $abortDetail"
    Write-TerminalReason -Reason "LOOP_ABORT" -Detail $abortDetail
    exit 1
}
finally {
    Remove-OwnLoopLock
    Log-Line "=== TRITON RISK WATCHDOG LOOP END ==="
}
