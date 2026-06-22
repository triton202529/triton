# scripts/register_triton_runtime_931_task.ps1
# Phase 196C — Register TRITON_Runtime_931AM scheduled task (runtime loops only).

[CmdletBinding()]
param(
    [string]$TaskName  = "TRITON_Runtime_931AM",
    [string]$RepoRoot   = "C:\Users\akimw\triton",
    [string]$ScriptPath = $null,
    [string]$TimeOfDay  = "09:31"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ScriptPath)) {
    $ScriptPath = Join-Path $RepoRoot "scripts\start_triton_runtime_931.ps1"
}

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repo root not found: $RepoRoot"
}
if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
    throw "Startup script not found: $ScriptPath"
}

$RepoRoot   = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
$ScriptPath = (Resolve-Path -LiteralPath $ScriptPath).Path

$timeParts = $TimeOfDay -split ":"
if ($timeParts.Count -ne 2) {
    throw "Invalid -TimeOfDay '$TimeOfDay' (expected HH:mm, e.g. '09:31')."
}
[int]$hour   = $timeParts[0]
[int]$minute = $timeParts[1]
if ($hour   -lt 0 -or $hour   -gt 23) { throw "Invalid hour in -TimeOfDay: $hour" }
if ($minute -lt 0 -or $minute -gt 59) { throw "Invalid minute in -TimeOfDay: $minute" }

$startAt = Get-Date -Hour $hour -Minute $minute -Second 0

$argString = '-NoProfile -ExecutionPolicy Bypass -File "' + $ScriptPath + '"'

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $argString `
    -WorkingDirectory $RepoRoot

if ([string]::IsNullOrWhiteSpace($action.WorkingDirectory) -or
    $action.WorkingDirectory -ne $RepoRoot) {
    $action.WorkingDirectory = $RepoRoot
}

$trigger = New-ScheduledTaskTrigger -Daily -At $startAt

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew

# Interactive user at Limited run level (no admin required for registration).
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

$description = "Phase 196C: start TRITON watchdog and continuous runtime loops at market open. Loops only; no direct execute_trades or live execution."

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $existing) {
    Write-Host "[TASK_REGISTER] replacing existing task: $TaskName" -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName    $TaskName `
    -Description $description `
    -Action      $action `
    -Trigger     $trigger `
    -Settings    $settings `
    -Principal   $principal `
    -Force | Out-Null

$verify = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
$persistedWd = $verify.Actions[0].WorkingDirectory
if ([string]::IsNullOrWhiteSpace($persistedWd) -or $persistedWd -ne $RepoRoot) {
    throw "Post-registration WorkingDirectory verification failed: '$persistedWd'"
}

$scheduleSummary = ("Daily " + ("{0:D2}:{1:D2}" -f $hour, $minute) + " local")

Write-Host ""
Write-Host "[TASK_REGISTER]" -ForegroundColor Green
Write-Host ("  name="        + $TaskName)
Write-Host ("  schedule="    + $scheduleSummary)
Write-Host ("  script="      + $ScriptPath)
Write-Host ("  working_dir=" + $persistedWd)
Write-Host ("  state="       + $verify.State)
Write-Host ""
Write-Host "Only $TaskName was created/updated. Old TRITON tasks were not enabled or modified." -ForegroundColor Green
Write-Host ("Inspect: Get-ScheduledTask -TaskName " + $TaskName)
Write-Host ("Run now: Start-ScheduledTask -TaskName " + $TaskName)
