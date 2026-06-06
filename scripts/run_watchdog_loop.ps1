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

Set-Content -Path $lockFile -Value "started_at=$(Get-Date -Format s)" -Encoding UTF8

function Log-Line($msg) {
    $line = "$(Get-Date -Format s) $msg"
    Write-Host $line
    Add-Content -Path $loopLog -Value $line
}

try {
    Log-Line "=== TRITON RISK WATCHDOG LOOP START ==="
    Log-Line "RepoRoot=$repoRoot"
    Log-Line "IntervalMinutes=$IntervalMinutes"
    Log-Line "MaxCycles=$MaxCycles"
    Log-Line "LoopLog=$loopLog"

    $graceMinutes = [Math]::Max(50, $IntervalMinutes + 10)
    $hbRefresh = python -m services.loop_safety --refresh --source watchdog_loop `
        --grace-minutes $graceMinutes 2>&1
    Log-Line "[HEARTBEAT_REFRESH] grace_minutes=$graceMinutes output=$hbRefresh"

    $cycle = 0

    while ($true) {
        $cycle++
        Log-Line ""
        Log-Line "[CYCLE_START] cycle=$cycle"

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

        if ($MaxCycles -gt 0 -and $cycle -ge $MaxCycles) {
            Log-Line "[LOOP_COMPLETE] reached MaxCycles=$MaxCycles"
            break
        }

        Log-Line "[SLEEP] waiting $IntervalMinutes minutes before next cycle"
        Start-Sleep -Seconds ($IntervalMinutes * 60)
    }
}
catch {
    Log-Line "[LOOP_ABORT] $($_.Exception.Message)"
    exit 1
}
finally {
    if (Test-Path $lockFile) {
        Remove-Item $lockFile -Force
    }
    Log-Line "=== TRITON RISK WATCHDOG LOOP END ==="
}
