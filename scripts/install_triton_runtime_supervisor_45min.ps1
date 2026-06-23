# scripts/install_triton_runtime_supervisor_45min.ps1
# Phase 196H — Register single TRITON_Runtime_Supervisor_45Min scheduled task.
# Pattern A: persistent loop at logon/startup (task does NOT repeat every 45 minutes).

[CmdletBinding()]
param(
    [string]$TaskName = "TRITON_Runtime_Supervisor_45Min",
    [string]$RepoRoot = "C:\Users\akimw\triton",
    [int]$IntervalMinutes = 45,
    [string]$LauncherScript = $null
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($LauncherScript)) {
    $LauncherScript = Join-Path $RepoRoot "scripts\start_triton_runtime_supervisor.ps1"
}

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repo root not found: $RepoRoot"
}
if (-not (Test-Path -LiteralPath $LauncherScript -PathType Leaf)) {
    throw "Launcher script not found: $LauncherScript"
}

$RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
$LauncherScript = (Resolve-Path -LiteralPath $LauncherScript).Path

$argString = (
    '-NoProfile -ExecutionPolicy Bypass -File "' + $LauncherScript + '" ' +
    '-RepoRoot "' + $RepoRoot + '" -IntervalMinutes ' + $IntervalMinutes
)

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $argString `
    -WorkingDirectory $RepoRoot

if ([string]::IsNullOrWhiteSpace($action.WorkingDirectory) -or
    $action.WorkingDirectory -ne $RepoRoot) {
    $action.WorkingDirectory = $RepoRoot
}

$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

$description = (
    "Phase 196H: TRITON runtime supervisor - starts watchdog + persistent continuous loop " +
    "($IntervalMinutes min interval) at logon/startup. Does not call execute_trades or live execution."
)

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $existing) {
    Write-Host "[TASK_INSTALL] replacing existing task: $TaskName" -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $description `
        -Action $action `
        -Trigger $logonTrigger `
        -Settings $settings `
        -Principal $principal `
        -Force | Out-Null
}
catch {
    Write-Host "[TASK_INSTALL] Register-ScheduledTask failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Try running this install script from an elevated PowerShell session." -ForegroundColor Yellow
    exit 1
}

$verify = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
if ($verify.State -eq "Disabled") {
    Enable-ScheduledTask -TaskName $TaskName | Out-Null
    $verify = Get-ScheduledTask -TaskName $TaskName
}

$persistedWd = $verify.Actions[0].WorkingDirectory
if ([string]::IsNullOrWhiteSpace($persistedWd) -or $persistedWd -ne $RepoRoot) {
    throw "Post-registration WorkingDirectory verification failed: '$persistedWd'"
}

$enabledSupervisorTasks = @(
    Get-ScheduledTask |
        Where-Object {
            $_.State -eq "Ready" -and
            $_.TaskName -match "Runtime_Supervisor|FullPipeline_45Min"
        }
)

Write-Host ""
Write-Host "[TASK_INSTALL]" -ForegroundColor Green
Write-Host ("  name=" + $TaskName)
Write-Host ("  pattern=PERSISTENT_LOOP_AT_LOGON")
Write-Host ("  launcher=" + $LauncherScript)
Write-Host ("  continuous_script=" + (Join-Path $RepoRoot "scripts\run_continuous_loop.ps1"))
Write-Host ("  interval_minutes=" + $IntervalMinutes)
Write-Host ("  working_dir=" + $persistedWd)
Write-Host ("  state=" + $verify.State)
Write-Host ("  triggers=AtLogOn")
Write-Host ("  enabled_supervisor_tasks=" + ($enabledSupervisorTasks.TaskName -join ", "))
Write-Host ""
Write-Host "Old TRITON scheduled tasks were NOT restored or enabled." -ForegroundColor Green
Write-Host ("Status: powershell -File scripts\show_triton_runtime_supervisor_status.ps1")
Write-Host ("Run now: Start-ScheduledTask -TaskName " + $TaskName)

if ($enabledSupervisorTasks.Count -gt 1) {
    Write-Host "[WARN] Multiple supervisor-style tasks are enabled." -ForegroundColor Yellow
    exit 2
}

exit 0
