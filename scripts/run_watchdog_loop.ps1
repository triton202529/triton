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
    $line = "$(Get-Date -Format s) $msg"
    Write-Host $line
    Add-Content -Path $loopLog -Value $line
}

# DO-004: cooperative shutdown helper (Python). Exit code 10 = accepted, exit now.
function Test-CooperativeShutdown {
    param([string]$CycleState = "BETWEEN_CYCLES")
    $out = python -m services.operator_cooperative_shutdown `
        --component watchdog --pid $PID --mode check --cycle-state $CycleState 2>&1
    $code = $LASTEXITCODE
    Log-Line "[SHUTDOWN_POLL] cycle_state=$CycleState exit_code=$code output=$out"
    return $code
}

function Invoke-InterruptibleSleep {
    param([int]$TotalSeconds)
    Log-Line "[SLEEP] interruptible wait ${TotalSeconds}s (check every ~5s for cooperative shutdown)"
    python -m services.operator_cooperative_shutdown `
        --component watchdog --pid $PID --mode sleep --seconds $TotalSeconds 2>&1 | ForEach-Object { Log-Line "[SLEEP_POLL] $_" }
    return $LASTEXITCODE
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

    while ($true) {
        # Safe boundary: before beginning a new watchdog cycle
        $pollCode = Test-CooperativeShutdown -CycleState "BETWEEN_CYCLES"
        if ($pollCode -eq 10) {
            Log-Line "[SHUTDOWN_ACCEPTED] at PRE_CYCLE safe boundary; no new cycle will start"
            $shutdownAccepted = $true
            break
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
        $pollCode = Test-CooperativeShutdown -CycleState "POST_CYCLE"
        if ($pollCode -eq 10) {
            Log-Line "[SHUTDOWN_ACCEPTED] at POST_CYCLE_PRE_SLEEP safe boundary; skipping further cycles"
            $shutdownAccepted = $true
            break
        }

        if ($MaxCycles -gt 0 -and $cycle -ge $MaxCycles) {
            Log-Line "[LOOP_COMPLETE] reached MaxCycles=$MaxCycles"
            break
        }

        $sleepCode = Invoke-InterruptibleSleep -TotalSeconds ($IntervalMinutes * 60)
        if ($sleepCode -eq 10) {
            Log-Line "[SHUTDOWN_ACCEPTED] during INTERRUPTIBLE_WAIT; exiting without new cycle"
            $shutdownAccepted = $true
            break
        }
    }

    if ($shutdownAccepted) {
        Log-Line "[SHUTDOWN_EXIT] cooperative shutdown complete for watchdog loop"
    }
}
catch {
    Log-Line "[LOOP_ABORT] $($_.Exception.Message)"
    exit 1
}
finally {
    Remove-OwnLoopLock
    Log-Line "=== TRITON RISK WATCHDOG LOOP END ==="
}
