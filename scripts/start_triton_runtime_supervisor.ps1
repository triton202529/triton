# scripts/start_triton_runtime_supervisor.ps1
# Logon/startup runtime supervisor (watchdog + continuous loops only).
# Does NOT run execute_trades. Does NOT enable live execution.
# Hardened to match 931 truthful-postcondition startup contract.

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton",
    [int]$IntervalMinutes = 45,
    [int]$ConfirmTimeoutSeconds = 25,
    [int]$ConfirmPollMilliseconds = 1000,
    [string]$AuditPath = "",
    [scriptblock]$ProcessRunningProbe = $null,
    [scriptblock]$LockExistsProbe = $null,
    [scriptblock]$StartProcessOverride = $null
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repo root not found: $RepoRoot"
}
$RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
Set-Location $RepoRoot

. (Join-Path $RepoRoot "scripts\triton_env_bootstrap.ps1") -RepoRoot $RepoRoot

$scriptDir = Join-Path $RepoRoot "scripts"
. (Join-Path $scriptDir "loop_lock_utils.ps1")

if ([string]::IsNullOrWhiteSpace($AuditPath)) {
    $AuditPath = Join-Path $RepoRoot "data\results\runtime_supervisor_autostart.json"
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $AuditPath) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "data\live") | Out-Null

$startupErrors = New-Object System.Collections.Generic.List[string]

function Write-SupervisorLine([string]$Message) {
    Write-Host $Message
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
    return Start-Process -FilePath "powershell.exe" `
        -ArgumentList $argList `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -PassThru
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
            } catch { }
        }
        $processConfirmed = Test-ProcessRunningSafe -ScriptName $ScriptName
        $lockConfirmed = Test-LockConfirmedSafe -LockFile $LockFile -ScriptName $ScriptName
        if ($processConfirmed -and $lockConfirmed) { break }
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
        process_confirmed  = [bool]$processConfirmed
        lock_confirmed     = [bool]$lockConfirmed
        child_exited_early = [bool]$childExitedEarly
        child_exit_code    = $childExitCode
    }
}

$env:TRITON_ENABLE_PAPER_EXECUTION = "1"
$env:TRITON_ENABLE_AUTO_PAPER_ENTRIES = "1"
$env:TRITON_ENABLE_AUTO_PAPER_APPROVAL = "1"
$env:TRITON_ENABLE_LIVE_EXECUTION = "0"
$env:TRITON_LIVE_EXECUTION = "0"

$watchdogScript = Join-Path $scriptDir "run_watchdog_loop.ps1"
$continuousScript = Join-Path $scriptDir "run_continuous_loop.ps1"
$watchdogLock = Join-Path $RepoRoot "data\live\watchdog_loop.lock"
$continuousLock = Join-Path $RepoRoot "data\live\continuous_loop.lock"

foreach ($path in @($watchdogScript, $continuousScript)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $startupErrors.Add("missing_script=$path") | Out-Null
    }
}

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

try {
    if ($watchdogWasRunning) {
        Write-SupervisorLine "[RUNTIME_SUPERVISOR] watchdog already running - skip duplicate start"
        $watchdogProcessConfirmed = $true
        $watchdogLockConfirmed = Test-LockConfirmedSafe -LockFile $watchdogLock -ScriptName "run_watchdog_loop.ps1"
        if (-not $watchdogLockConfirmed) {
            $startupErrors.Add("watchdog_running_but_lock_unconfirmed") | Out-Null
        }
    }
    elseif ($startupErrors.Count -eq 0) {
        $watchdogStartAttempted = $true
        Write-SupervisorLine "[RUNTIME_SUPERVISOR] starting watchdog loop IntervalMinutes=1"
        $wdProc = Start-LoopProcessTracked -ScriptPath $watchdogScript -IntervalMinutes 1 -WorkingDirectory $RepoRoot
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
        }
    }

    if ($continuousWasRunning) {
        Write-SupervisorLine "[RUNTIME_SUPERVISOR] continuous loop already running - skip duplicate start"
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
        Write-SupervisorLine "[RUNTIME_SUPERVISOR] starting continuous loop IntervalMinutes=$IntervalMinutes"
        $cProc = Start-LoopProcessTracked -ScriptPath $continuousScript -IntervalMinutes $IntervalMinutes -WorkingDirectory $RepoRoot
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
catch {
    $startupErrors.Add("exception=$($_.Exception.Message)") | Out-Null
}

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

$liveExecutionEnabled = (
    $env:TRITON_ENABLE_LIVE_EXECUTION -eq "1" -or
    $env:TRITON_LIVE_EXECUTION -eq "1"
)
$paperFlagsOk = (
    $env:TRITON_ENABLE_PAPER_EXECUTION -eq "1" -and
    $env:TRITON_ENABLE_AUTO_PAPER_ENTRIES -eq "1" -and
    $env:TRITON_ENABLE_AUTO_PAPER_APPROVAL -eq "1"
)
$liveFlagsOk = (
    $env:TRITON_ENABLE_LIVE_EXECUTION -eq "0" -and
    $env:TRITON_LIVE_EXECUTION -eq "0" -and
    -not $liveExecutionEnabled
)

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
}

$audit = [ordered]@{
    timestamp                     = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    phase                         = "196H"
    repo_root                     = $RepoRoot
    interval_minutes              = $IntervalMinutes
    watchdog_was_running          = [bool]$watchdogWasRunning
    watchdog_start_attempted      = [bool]$watchdogStartAttempted
    watchdog_started              = [bool]$watchdogStarted
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
    paper_execution_env           = $env:TRITON_ENABLE_PAPER_EXECUTION
    auto_paper_entries_env        = $env:TRITON_ENABLE_AUTO_PAPER_ENTRIES
    auto_paper_approval_env       = $env:TRITON_ENABLE_AUTO_PAPER_APPROVAL
    enable_live_execution_env     = $env:TRITON_ENABLE_LIVE_EXECUTION
    live_execution_env            = $env:TRITON_LIVE_EXECUTION
    live_execution_enabled        = [bool]$liveExecutionEnabled
    postconditions_passed         = [bool]$postconditionsPassed
    errors                        = @($startupErrors)
}

$audit | ConvertTo-Json -Depth 6 | Set-Content -Path $AuditPath -Encoding UTF8
Write-SupervisorLine "[RUNTIME_SUPERVISOR] audit written: $AuditPath"

if (-not $postconditionsPassed) {
    Write-Host "[RUNTIME_SUPERVISOR] FAILED: $($startupErrors -join '; ')" -ForegroundColor Red
    exit 1
}

Write-Host "[RUNTIME_SUPERVISOR] completed successfully" -ForegroundColor Green
exit 0
