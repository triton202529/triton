# scripts/start_triton_runtime_931.ps1
# Daily market-open runtime autostart (watchdog + continuous loops only).
# Does NOT run execute_trades. Does NOT enable live execution.
# Hardened: truthful postconditions, -PassThru child tracking, bounded confirm.

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton",
    [int]$ConfirmTimeoutSeconds = 25,
    [int]$ConfirmPollMilliseconds = 1000,
    # Isolated validation / dependency injection (never used by Scheduled Task).
    [string]$AuditPath = "",
    [scriptblock]$ProcessRunningProbe = $null,
    [scriptblock]$LockExistsProbe = $null,
    [scriptblock]$StartProcessOverride = $null,
    [Nullable[DateTime]]$NowOverride = $null,
    # Validation hook: return exit code instead of calling exit (keeps caller alive).
    [switch]$ReturnInsteadOfExit
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repo root not found: $RepoRoot"
}
$RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
Set-Location $RepoRoot

$scriptDir = Join-Path $RepoRoot "scripts"
. (Join-Path $scriptDir "loop_lock_utils.ps1")

if ([string]::IsNullOrWhiteSpace($AuditPath)) {
    $AuditPath = Join-Path $RepoRoot "data\results\runtime_autostart_931.json"
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $AuditPath) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "data\live") | Out-Null

# Prefer a non-colliding name (avoid PowerShell automatic $Error).
$startupErrors = New-Object System.Collections.Generic.List[string]

function Write-AuditLine([string]$Message) {
    Write-Host $Message
}

function Get-RuntimeNow {
    if ($null -ne $NowOverride) { return [DateTime]$NowOverride }
    return Get-Date
}

function Test-EligibleMarketSession {
    param([DateTime]$When)
    # Monday=1 .. Friday=5 in .NET DayOfWeek when cast carefully:
    # Sunday=0, Monday=1, ..., Saturday=6
    $dow = [int]$When.DayOfWeek
    return ($dow -ge 1 -and $dow -le 5)
}

function Test-ProcessRunningSafe {
    param([Parameter(Mandatory = $true)][string]$ScriptName)
    if ($null -ne $ProcessRunningProbe) {
        return [bool](& $ProcessRunningProbe -ScriptName $ScriptName)
    }
    return [bool](Test-LoopProcessRunning -ScriptName $ScriptName)
}

function Test-LockConfirmedSafe {
    param(
        [Parameter(Mandatory = $true)][string]$LockFile,
        [Parameter(Mandatory = $true)][string]$ScriptName
    )
    if ($null -ne $LockExistsProbe) {
        return [bool](& $LockExistsProbe -LockFile $LockFile -ScriptName $ScriptName)
    }
    if (-not (Test-Path -LiteralPath $LockFile -PathType Leaf)) {
        return $false
    }
    # Lock must belong to a living process of the expected script (never treat orphan lock as success).
    return [bool](Test-LoopProcessRunning -ScriptName $ScriptName)
}

function Start-LoopProcessTracked {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $true)][int]$IntervalMinutes,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    if ($null -ne $StartProcessOverride) {
        return & $StartProcessOverride -ScriptPath $ScriptPath -IntervalMinutes $IntervalMinutes -WorkingDirectory $WorkingDirectory
    }

    $argList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $ScriptPath,
        "-IntervalMinutes", "$IntervalMinutes"
    )
    $proc = Start-Process -FilePath "powershell.exe" `
        -ArgumentList $argList `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -PassThru
    return $proc
}

function Wait-LoopStartupConfirmation {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptName,
        [Parameter(Mandatory = $true)][string]$LockFile,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
        [Parameter(Mandatory = $true)][int]$PollMilliseconds,
        [object]$ChildProcess = $null
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $processConfirmed = $false
    $lockConfirmed = $false
    $childExitedEarly = $false
    $childExitCode = $null

    while ((Get-Date) -lt $deadline) {
        if ($null -ne $ChildProcess) {
            try { $ChildProcess.Refresh() } catch { }
            try {
                if ($ChildProcess.HasExited) {
                    $childExitedEarly = $true
                    try { $childExitCode = $ChildProcess.ExitCode } catch { $childExitCode = $null }
                    break
                }
            } catch {
                # Child handle may become invalid; continue probing by script name.
            }
        }

        $processConfirmed = Test-ProcessRunningSafe -ScriptName $ScriptName
        $lockConfirmed = Test-LockConfirmedSafe -LockFile $LockFile -ScriptName $ScriptName
        if ($processConfirmed -and $lockConfirmed) {
            break
        }
        Start-Sleep -Milliseconds $PollMilliseconds
    }

    if (-not $processConfirmed) {
        $processConfirmed = Test-ProcessRunningSafe -ScriptName $ScriptName
    }
    if (-not $lockConfirmed) {
        $lockConfirmed = Test-LockConfirmedSafe -LockFile $LockFile -ScriptName $ScriptName
    }

    if ($null -ne $ChildProcess) {
        try { $ChildProcess.Refresh() } catch { }
        try {
            if ($ChildProcess.HasExited) {
                $childExitedEarly = $true
                try { $childExitCode = $ChildProcess.ExitCode } catch { $childExitCode = $null }
            }
        } catch { }
    }

    return [pscustomobject]@{
        process_confirmed   = [bool]$processConfirmed
        lock_confirmed      = [bool]$lockConfirmed
        child_exited_early  = [bool]$childExitedEarly
        child_exit_code     = $childExitCode
    }
}

# ---------------------------------------------------------------------------
# Paper-only authority (explicit live block)
# ---------------------------------------------------------------------------
$env:TRITON_ENABLE_PAPER_EXECUTION = "1"
$env:TRITON_ENABLE_AUTO_PAPER_ENTRIES = "1"
$env:TRITON_ENABLE_AUTO_PAPER_APPROVAL = "1"
$env:TRITON_ENABLE_LIVE_EXECUTION = "0"
$env:TRITON_LIVE_EXECUTION = "0"

$paperExecutionEnv = $env:TRITON_ENABLE_PAPER_EXECUTION
$autoPaperEntriesEnv = $env:TRITON_ENABLE_AUTO_PAPER_ENTRIES
$autoPaperApprovalEnv = $env:TRITON_ENABLE_AUTO_PAPER_APPROVAL
$enableLiveExecutionEnv = $env:TRITON_ENABLE_LIVE_EXECUTION
$liveExecutionEnv = $env:TRITON_LIVE_EXECUTION

$watchdogScript = Join-Path $scriptDir "run_watchdog_loop.ps1"
$continuousScript = Join-Path $scriptDir "run_continuous_loop.ps1"
$watchdogLock = Join-Path $RepoRoot "data\live\watchdog_loop.lock"
$continuousLock = Join-Path $RepoRoot "data\live\continuous_loop.lock"

foreach ($path in @($watchdogScript, $continuousScript)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $startupErrors.Add("missing_script=$path") | Out-Null
    }
}

$nowLocal = Get-RuntimeNow
$eligibleSession = Test-EligibleMarketSession -When $nowLocal
$startRefusedReason = $null

$watchdogWasRunning = Test-ProcessRunningSafe -ScriptName "run_watchdog_loop.ps1"
$continuousWasRunning = Test-ProcessRunningSafe -ScriptName "run_continuous_loop.ps1"

$watchdogStartAttempted = $false
$continuousStartAttempted = $false
$watchdogStarted = $false
$continuousStarted = $false
$watchdogProcessConfirmed = $false
$watchdogLockConfirmed = $false
$continuousProcessConfirmed = $false
$continuousLockConfirmed = $false
$continuousChildPid = $null
$continuousChildExitedEarly = $false
$continuousChildExitCode = $null
$watchdogChildPid = $null

try {
    if (-not $eligibleSession) {
        $startRefusedReason = "ineligible_session_day=$($nowLocal.DayOfWeek)"
        Write-AuditLine "[AUTOSTART_931] start refused: $startRefusedReason"
    }
    else {
        # Watchdog singleton
        if ($watchdogWasRunning) {
            Write-AuditLine "[AUTOSTART_931] watchdog already running - skip duplicate start"
            $watchdogProcessConfirmed = $true
            $watchdogLockConfirmed = Test-LockConfirmedSafe -LockFile $watchdogLock -ScriptName "run_watchdog_loop.ps1"
            if (-not $watchdogLockConfirmed) {
                $startupErrors.Add("watchdog_running_but_lock_unconfirmed") | Out-Null
            }
        }
        elseif ($startupErrors.Count -eq 0) {
            $watchdogStartAttempted = $true
            Write-AuditLine "[AUTOSTART_931] starting watchdog loop IntervalMinutes=1"
            $wdProc = Start-LoopProcessTracked -ScriptPath $watchdogScript -IntervalMinutes 1 -WorkingDirectory $RepoRoot
            if ($null -ne $wdProc) { $watchdogChildPid = $wdProc.Id }
            $wdConfirm = Wait-LoopStartupConfirmation `
                -ScriptName "run_watchdog_loop.ps1" `
                -LockFile $watchdogLock `
                -TimeoutSeconds $ConfirmTimeoutSeconds `
                -PollMilliseconds $ConfirmPollMilliseconds `
                -ChildProcess $wdProc
            $watchdogProcessConfirmed = [bool]$wdConfirm.process_confirmed
            $watchdogLockConfirmed = [bool]$wdConfirm.lock_confirmed
            $watchdogStarted = $watchdogProcessConfirmed -and $watchdogLockConfirmed
            if (-not $watchdogStarted) {
                $startupErrors.Add("watchdog_start_unconfirmed") | Out-Null
                if ($wdConfirm.child_exited_early) {
                    $startupErrors.Add("watchdog_child_exited_early exit_code=$($wdConfirm.child_exit_code)") | Out-Null
                }
            }
        }

        # Continuous singleton
        if ($continuousWasRunning) {
            Write-AuditLine "[AUTOSTART_931] continuous loop already running - skip duplicate start"
            $continuousProcessConfirmed = $true
            $continuousLockConfirmed = Test-LockConfirmedSafe -LockFile $continuousLock -ScriptName "run_continuous_loop.ps1"
            if (-not $continuousLockConfirmed) {
                $startupErrors.Add("continuous_running_but_lock_unconfirmed") | Out-Null
            }
            else {
                $continuousStarted = $true
            }
        }
        elseif ($startupErrors.Count -eq 0) {
            $continuousStartAttempted = $true
            Write-AuditLine "[AUTOSTART_931] starting continuous loop IntervalMinutes=45"
            $cProc = Start-LoopProcessTracked -ScriptPath $continuousScript -IntervalMinutes 45 -WorkingDirectory $RepoRoot
            if ($null -ne $cProc) { $continuousChildPid = $cProc.Id }
            $cConfirm = Wait-LoopStartupConfirmation `
                -ScriptName "run_continuous_loop.ps1" `
                -LockFile $continuousLock `
                -TimeoutSeconds $ConfirmTimeoutSeconds `
                -PollMilliseconds $ConfirmPollMilliseconds `
                -ChildProcess $cProc
            $continuousProcessConfirmed = [bool]$cConfirm.process_confirmed
            $continuousLockConfirmed = [bool]$cConfirm.lock_confirmed
            $continuousChildExitedEarly = [bool]$cConfirm.child_exited_early
            $continuousChildExitCode = $cConfirm.child_exit_code
            $continuousStarted = $continuousProcessConfirmed -and $continuousLockConfirmed
            if (-not $continuousStarted) {
                $startupErrors.Add("continuous_start_unconfirmed") | Out-Null
                if ($continuousChildExitedEarly) {
                    $startupErrors.Add("continuous_child_exited_early exit_code=$continuousChildExitCode") | Out-Null
                }
                if ($continuousProcessConfirmed -and -not $continuousLockConfirmed) {
                    $startupErrors.Add("continuous_process_without_lock") | Out-Null
                }
                if ($continuousLockConfirmed -and -not $continuousProcessConfirmed) {
                    $startupErrors.Add("continuous_lock_without_process") | Out-Null
                }
            }
        }
    }
}
catch {
    $startupErrors.Add("exception=$($_.Exception.Message)") | Out-Null
}

# Final postcondition probes (eligible sessions only).
if ($eligibleSession) {
    if (-not $watchdogProcessConfirmed) {
        $watchdogProcessConfirmed = Test-ProcessRunningSafe -ScriptName "run_watchdog_loop.ps1"
    }
    if (-not $watchdogLockConfirmed) {
        $watchdogLockConfirmed = Test-LockConfirmedSafe -LockFile $watchdogLock -ScriptName "run_watchdog_loop.ps1"
    }
    if (-not $continuousProcessConfirmed) {
        $continuousProcessConfirmed = Test-ProcessRunningSafe -ScriptName "run_continuous_loop.ps1"
    }
    if (-not $continuousLockConfirmed) {
        $continuousLockConfirmed = Test-LockConfirmedSafe -LockFile $continuousLock -ScriptName "run_continuous_loop.ps1"
    }
}

$liveExecutionEnabled = (
    $enableLiveExecutionEnv -eq "1" -or
    $liveExecutionEnv -eq "1" -or
    $env:TRITON_ENABLE_LIVE_EXECUTION -eq "1" -or
    $env:TRITON_LIVE_EXECUTION -eq "1"
)

$paperFlagsOk = (
    $paperExecutionEnv -eq "1" -and
    $autoPaperEntriesEnv -eq "1" -and
    $autoPaperApprovalEnv -eq "1"
)
$liveFlagsOk = (
    $enableLiveExecutionEnv -eq "0" -and
    $liveExecutionEnv -eq "0" -and
    -not $liveExecutionEnabled
)

if ($eligibleSession) {
    $postconditionsPassed = (
        $watchdogProcessConfirmed -and
        $watchdogLockConfirmed -and
        $continuousProcessConfirmed -and
        $continuousLockConfirmed -and
        $paperFlagsOk -and
        $liveFlagsOk -and
        ($startupErrors.Count -eq 0)
    )
    if (-not $postconditionsPassed) {
        if (-not $watchdogProcessConfirmed) { $startupErrors.Add("postcondition_watchdog_process_missing") | Out-Null }
        if (-not $watchdogLockConfirmed) { $startupErrors.Add("postcondition_watchdog_lock_missing") | Out-Null }
        if (-not $continuousProcessConfirmed) { $startupErrors.Add("postcondition_continuous_process_missing") | Out-Null }
        if (-not $continuousLockConfirmed) { $startupErrors.Add("postcondition_continuous_lock_missing") | Out-Null }
        if (-not $paperFlagsOk) { $startupErrors.Add("postcondition_paper_flags_invalid") | Out-Null }
        if (-not $liveFlagsOk) { $startupErrors.Add("postcondition_live_flags_not_blocked") | Out-Null }
        # De-duplicate while preserving order
        $dedup = New-Object System.Collections.Generic.List[string]
        foreach ($e in $startupErrors) {
            if (-not $dedup.Contains($e)) { $dedup.Add($e) | Out-Null }
        }
        $startupErrors = $dedup
        $postconditionsPassed = $false
    }
}
else {
    # Deliberate weekend/ineligible refusal is success only when no start was attempted.
    $postconditionsPassed = (
        -not $watchdogStartAttempted -and
        -not $continuousStartAttempted -and
        $paperFlagsOk -and
        $liveFlagsOk -and
        ($startupErrors.Count -eq 0)
    )
}

$audit = [ordered]@{
    timestamp                     = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    repo_root                     = $RepoRoot
    scheduled_time                = "09:31"
    eligible_session              = [bool]$eligibleSession
    start_refused_reason          = $startRefusedReason
    watchdog_was_running          = [bool]$watchdogWasRunning
    watchdog_start_attempted      = [bool]$watchdogStartAttempted
    watchdog_started              = [bool]$watchdogStarted
    watchdog_child_pid            = $watchdogChildPid
    watchdog_process_confirmed    = [bool]$watchdogProcessConfirmed
    watchdog_lock_confirmed       = [bool]$watchdogLockConfirmed
    continuous_was_running        = [bool]$continuousWasRunning
    continuous_start_attempted    = [bool]$continuousStartAttempted
    continuous_child_pid          = $continuousChildPid
    continuous_child_exited_early = [bool]$continuousChildExitedEarly
    continuous_child_exit_code    = $continuousChildExitCode
    continuous_started            = [bool]$continuousStarted
    continuous_process_confirmed  = [bool]$continuousProcessConfirmed
    continuous_lock_confirmed     = [bool]$continuousLockConfirmed
    paper_execution_env           = $paperExecutionEnv
    auto_paper_entries_env        = $autoPaperEntriesEnv
    auto_paper_approval_env       = $autoPaperApprovalEnv
    enable_live_execution_env     = $enableLiveExecutionEnv
    live_execution_env            = $liveExecutionEnv
    live_execution_enabled        = [bool]$liveExecutionEnabled
    postconditions_passed         = [bool]$postconditionsPassed
    errors                        = @($startupErrors)
}

$audit | ConvertTo-Json -Depth 6 | Set-Content -Path $AuditPath -Encoding UTF8
Write-AuditLine "[AUTOSTART_931] audit written: $AuditPath"

function Emit-AutostartExitCode {
    param([int]$Code)
    # Ensure callers using -ReturnInsteadOfExit receive a single int on the pipeline.
    Write-Output $Code
}

if (-not $postconditionsPassed) {
    Write-Host "[AUTOSTART_931] FAILED postconditions/errors: $($startupErrors -join '; ')" -ForegroundColor Red
    if ($ReturnInsteadOfExit) {
        Emit-AutostartExitCode -Code 1
        return
    }
    exit 1
}

if (-not $eligibleSession) {
    Write-Host "[AUTOSTART_931] completed with deliberate ineligible-session refusal" -ForegroundColor Green
    if ($ReturnInsteadOfExit) {
        Emit-AutostartExitCode -Code 0
        return
    }
    exit 0
}

Write-Host "[AUTOSTART_931] completed successfully" -ForegroundColor Green
if ($ReturnInsteadOfExit) {
    Emit-AutostartExitCode -Code 0
    return
}
exit 0
