# scripts/run_continuous_loop.ps1
param(
    [int]$IntervalMinutes = 45,
    [int]$MaxCycles = 0
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

. (Join-Path $scriptDir "triton_env_bootstrap.ps1") -RepoRoot $repoRoot

. (Join-Path $scriptDir "loop_lock_utils.ps1")
. (Join-Path $scriptDir "loop_log_utils.ps1")

$loopLogDir = "data\results\continuous_loop"
New-Item -ItemType Directory -Force -Path $loopLogDir | Out-Null

$startedAt = Get-Date -Format "yyyyMMdd_HHmmss"
$loopLog = Join-Path $loopLogDir "continuous_loop_$startedAt.log"
$lockFile = "data\live\continuous_loop.lock"

New-Item -ItemType Directory -Force -Path "data\live" | Out-Null

Clear-OrphanLoopLock -LockFile $lockFile -ScriptName "run_continuous_loop.ps1" | Out-Null

if (Test-Path $lockFile) {
    Write-Host "[BLOCKED] Continuous loop lock already exists: $lockFile" -ForegroundColor Red
    Write-Host "Delete it only if you are sure no loop is running."
    exit 1
}

Update-LoopLockMetadata -LockFile $lockFile -Source "continuous_loop" -Cycle 0

function Log-Line($msg) {
    # Host first so a sharing violation cannot hide the audit line from the console.
    $line = "$(Get-Date -Format s) $msg"
    Write-Host $line
    # Bounded retries for transient sharing violations; fail closed if exhausted.
    Write-TritonLoopLogLine -Path $loopLog -Line $line | Out-Null
}

# DO-004: cooperative shutdown helper (Python). Exit code 10 = accepted, exit now.
function Test-CooperativeShutdown {
    param([string]$CycleState = "BETWEEN_CYCLES")
    $out = python -m services.operator_cooperative_shutdown `
        --component continuous --pid $PID --mode check --cycle-state $CycleState 2>&1
    $code = $LASTEXITCODE
    Log-Line "[SHUTDOWN_POLL] cycle_state=$CycleState exit_code=$code output=$out"
    return $code
}

function Invoke-InterruptibleSleep {
    param([int]$TotalSeconds)
    # Capture exit code before logging: a pipeline into ForEach can clobber $LASTEXITCODE.
    Log-Line "[SLEEP] interruptible wait ${TotalSeconds}s (check every ~10s for cooperative shutdown)"
    $sleepOut = python -m services.operator_cooperative_shutdown `
        --component continuous --pid $PID --mode sleep --seconds $TotalSeconds 2>&1
    $code = $LASTEXITCODE
    foreach ($line in @($sleepOut)) { Log-Line "[SLEEP_POLL] $line" }
    return $code
}

function Remove-OwnLoopLock {
    if (-not (Test-Path $lockFile)) { return }
    try {
        $text = Get-Content -Path $lockFile -Raw -ErrorAction SilentlyContinue
        if ($text -match 'pid=(\d+)' -and [int]$Matches[1] -ne $PID) {
            try {
                Log-Line "[LOCK_SKIP] lock owned by pid=$($Matches[1]); not removing (this pid=$PID)"
            }
            catch {
                Write-Host "$(Get-Date -Format s) [LOCK_SKIP] lock owned by pid=$($Matches[1]); not removing (this pid=$PID)"
            }
            return
        }
    } catch {
        try {
            Log-Line "[LOCK_CHECK] could not parse lock ownership: $($_.Exception.Message)"
        }
        catch {
            Write-Host "$(Get-Date -Format s) [LOCK_CHECK] could not parse lock ownership: $($_.Exception.Message)"
        }
    }
    Remove-Item $lockFile -Force
    try {
        Log-Line "[LOCK_RELEASE] released own continuous_loop.lock pid=$PID"
    }
    catch {
        Write-Host "$(Get-Date -Format s) [LOCK_RELEASE] released own continuous_loop.lock pid=$PID"
    }
}

try {
    Log-Line "=== TRITON CONTINUOUS LOOP START ==="
    Log-Line "RepoRoot=$repoRoot"
    Log-Line "IntervalMinutes=$IntervalMinutes"
    Log-Line "MaxCycles=$MaxCycles"
    Log-Line "LoopLog=$loopLog"
    Log-Line "[SAFE_BOUNDARY] PRE_CYCLE | POST_CYCLE_PRE_SLEEP | INTERRUPTIBLE_WAIT (DO-004)"

    $graceMinutes = [Math]::Max(50, $IntervalMinutes + 10)
    $hbRefresh = python -m services.loop_safety --refresh --source continuous_loop `
        --grace-minutes $graceMinutes --continuous-interval-minutes $IntervalMinutes 2>&1
    Log-Line "[HEARTBEAT_REFRESH] grace_minutes=$graceMinutes output=$hbRefresh"

    $env:TRITON_ENABLE_PAPER_EXECUTION = "1"
    $env:TRITON_ENABLE_AUTO_PAPER_ENTRIES = "1"
    $env:TRITON_ENABLE_AUTO_PAPER_APPROVAL = "1"
    Log-Line "[LOOP_ENV] TRITON_ENABLE_PAPER_EXECUTION=1"
    Log-Line "[LOOP_ENV] TRITON_ENABLE_AUTO_PAPER_ENTRIES=1"
    Log-Line "[LOOP_ENV] TRITON_ENABLE_AUTO_PAPER_APPROVAL=1"

    $cycle = 0
    $shutdownAccepted = $false

    while ($true) {
        # Safe boundary: before beginning a new cycle
        $pollCode = Test-CooperativeShutdown -CycleState "BETWEEN_CYCLES"
        if ($pollCode -eq 10) {
            Log-Line "[SHUTDOWN_ACCEPTED] at PRE_CYCLE safe boundary; no new cycle will start"
            $shutdownAccepted = $true
            break
        }
        # 0 = no request, 11 = refused request (continue). Any other code is helper
        # failure — refuse a new trading cycle and wait boundedly (do not treat as accept).
        if (($null -eq $pollCode) -or (($pollCode -ne 0) -and ($pollCode -ne 11))) {
            Log-Line "[SHUTDOWN_HELPER_DEGRADED] pre_cycle exit_code=$pollCode; refusing new cycle; wait 30s"
            Start-Sleep -Seconds 30
            continue
        }

        $cycle++
        $cycleRc = 0
        $intervalSeconds = [int]($IntervalMinutes * 60)

        try {
            Log-Line ""
            Log-Line "[CYCLE_START] cycle=$cycle"
            Update-LoopLockMetadata -LockFile $lockFile -Source "continuous_loop" -Cycle $cycle
            $cadenceStart = python -m services.continuous_loop_cadence --record-start --interval-seconds $intervalSeconds --cycle $cycle --json 2>&1
            Log-Line "[CADENCE_START] $cadenceStart"

            $env:TRITON_ENABLE_PAPER_EXECUTION = "1"
            $env:TRITON_ENABLE_AUTO_PAPER_ENTRIES = "1"
            $env:TRITON_ENABLE_AUTO_PAPER_APPROVAL = "1"
            Log-Line "[LOOP_ENV] TRITON_ENABLE_PAPER_EXECUTION=1"
            Log-Line "[LOOP_ENV] TRITON_ENABLE_AUTO_PAPER_ENTRIES=1"
            Log-Line "[LOOP_ENV] TRITON_ENABLE_AUTO_PAPER_APPROVAL=1"

            $sessionOpen = $false
            $sessionState = "UNKNOWN"
            $sessionReason = "session_check_failed"
            $sessionAction = ""
            $sessionRaw = python -m services.paper_runtime_session_gate --check --json
            $sessionCode = $LASTEXITCODE
            Log-Line "[SESSION_CHECK] exit_code=$sessionCode output=$sessionRaw"
            if ($sessionCode -eq 0 -and $sessionRaw) {
                try {
                    $sessionText = ($sessionRaw | Out-String)
                    $sessionObj = $sessionText | ConvertFrom-Json
                    if ($sessionObj.session_state) { $sessionState = [string]$sessionObj.session_state }
                    if ($sessionObj.action) { $sessionAction = [string]$sessionObj.action }
                    $sessionOpen = ($sessionState -eq "OPEN")
                    if ($sessionObj.reason) { $sessionReason = [string]$sessionObj.reason }
                } catch {
                    $sessionOpen = $false
                    $sessionState = "UNKNOWN"
                    $sessionAction = ""
                    $sessionReason = "session_json_unreadable"
                }
            }

            if ($sessionState -ne "OPEN" -and $sessionState -ne "CLOSED") {
                Log-Line "[PAPER_ENTRY_BLOCKED_SESSION_UNKNOWN] reason=$sessionReason session_state=$sessionState session_open=false"
                python -m services.paper_runtime_session_gate --record-cycle --outcome session_unknown --reason "PAPER_ENTRY_BLOCKED_SESSION_UNKNOWN" --session-open false | Out-Null
                Log-Line "[CYCLE_BLOCKED_SESSION_UNKNOWN] cycle=$cycle reason=$sessionReason"
            } elseif ($sessionAction -eq "skip_near_close") {
                Log-Line "[PAPER_ENTRY_SKIPPED_NEAR_CLOSE] reason=$sessionReason remaining=$($sessionObj.remaining_runway_seconds) required=$($sessionObj.required_runway_seconds) close=$($sessionObj.session_close_et) session_open=true broker_orders_submitted=false live_execution_permitted=false"
                python -m services.paper_runtime_session_gate --record-cycle --outcome skipped_near_close --reason "CYCLE_SKIPPED_NEAR_CLOSE" --session-open true | Out-Null
                Log-Line "[CYCLE_SKIPPED_NEAR_CLOSE] cycle=$cycle reason=$sessionReason remaining=$($sessionObj.remaining_runway_seconds) required=181"
            } elseif (-not $sessionOpen) {
                Log-Line "[PAPER_ENTRY_SKIPPED_MARKET_CLOSED] reason=$sessionReason session_open=false"
                python -m services.paper_runtime_session_gate --record-cycle --outcome skipped_market_closed --reason $sessionReason --session-open false | Out-Null
                Log-Line "[CYCLE_SKIPPED_MARKET_CLOSED] cycle=$cycle reason=$sessionReason"
            } else {
            powershell -ExecutionPolicy Bypass -File .\scripts\run_market_open.ps1
            $rc = $LASTEXITCODE
            if ($rc -ne 0) { $cycleRc = $rc }

            if ($rc -eq 0) {
                Log-Line "[PAPER_ENTRY_STAGE] evaluating controlled paper entry gates"
                python -m services.paper_entry_loop_stage --verbose
                $paperEntryRc = $LASTEXITCODE
                Log-Line "[PAPER_ENTRY_STAGE] exit_code=$paperEntryRc"
                if ($paperEntryRc -ne 0) { $cycleRc = $paperEntryRc }
            } else {
                # Prefer sanitized failed-step detail from the sequence status
                # artifact over the generic failure label alone.
                $failedStep = ""
                $failReason = "market_open_sequence_failed"
                $statusFile = "data\results\market_open_sequence_status.json"
                if (Test-Path $statusFile) {
                    try {
                        $st = Get-Content $statusFile -Raw | ConvertFrom-Json
                        if ($st.failed_step) { $failedStep = [string]$st.failed_step }
                        if ($st.reason) { $failReason = [string]$st.reason }
                    } catch {
                        $failedStep = ""
                    }
                }
                if ($failedStep) {
                    Log-Line "[PAPER_ENTRY_EXECUTION_BLOCKED] reason=market_open_sequence_failed failed_step=$failedStep detail=$failReason exit_code=$rc"
                } else {
                    Log-Line "[PAPER_ENTRY_EXECUTION_BLOCKED] reason=market_open_sequence_failed exit_code=$rc"
                }
            }

            if ($cycleRc -ne 0) {
                python -m services.paper_runtime_session_gate --record-cycle --outcome fail --reason "market_hours_cycle_fail" --session-open true | Out-Null
                Log-Line "[CYCLE_FAIL] cycle=$cycle exit_code=$cycleRc (continuing after sleep)"
            } else {
                python -m services.paper_runtime_session_gate --record-cycle --outcome ok --reason "market_hours_cycle_ok" --session-open true | Out-Null
                Log-Line "[CYCLE_OK] cycle=$cycle"
            }
            } # end regular-session entry path
        }
        catch {
            Log-Line "[CYCLE_ERROR] cycle=$cycle error=$($_.Exception.Message) (continuing after sleep)"
        }

        # Safe boundary: after current cycle work; before sleep / next cycle
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

        # Start-to-start cadence: remaining_wait = max(0, (start + interval) - now).
        $cadenceWaitRaw = python -m services.continuous_loop_cadence --remaining-wait --interval-seconds $intervalSeconds --cycle $cycle --json 2>&1
        Log-Line "[CADENCE_WAIT] $cadenceWaitRaw"
        $waitSeconds = $intervalSeconds
        try {
            $cadenceWaitObj = ($cadenceWaitRaw | Out-String) | ConvertFrom-Json
            if ($null -ne $cadenceWaitObj.remaining_wait_seconds) {
                $waitSeconds = [int]$cadenceWaitObj.remaining_wait_seconds
            }
            if ($cadenceWaitObj.next_due_at_utc) {
                Log-Line "[CADENCE_NEXT_DUE_UTC] $($cadenceWaitObj.next_due_at_utc)"
            }
        } catch {
            Log-Line "[CADENCE_WAIT_PARSE] fallback_full_interval reason=$($_.Exception.Message)"
            $waitSeconds = $intervalSeconds
        }
        $sleepCode = Invoke-InterruptibleSleep -TotalSeconds $waitSeconds
        if ($sleepCode -eq 10) {
            Log-Line "[SHUTDOWN_ACCEPTED] during INTERRUPTIBLE_WAIT; exiting without new cycle"
            $shutdownAccepted = $true
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
        Log-Line "[SHUTDOWN_EXIT] cooperative shutdown complete for continuous loop"
    }
}
catch {
    $abortMsg = $_.Exception.Message
    try {
        Log-Line "[LOOP_ABORT] $abortMsg"
    }
    catch {
        # Independent channel if the loop log remains contended.
        Write-Host "$(Get-Date -Format s) [LOOP_ABORT] $abortMsg"
        Write-Host "$(Get-Date -Format s) [LOG_WRITE_FATAL] abort marker not persisted: $($_.Exception.Message)"
    }
    exit 1
}
finally {
    # Lock release must run even when logging is impaired.
    Remove-OwnLoopLock
    try {
        Log-Line "=== TRITON CONTINUOUS LOOP END ==="
    }
    catch {
        Write-Host "$(Get-Date -Format s) === TRITON CONTINUOUS LOOP END ==="
        Write-Host "$(Get-Date -Format s) [LOG_WRITE_FATAL] end marker not persisted: $($_.Exception.Message)"
    }
}
