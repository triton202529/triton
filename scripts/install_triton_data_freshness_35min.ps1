# scripts/install_triton_data_freshness_35min.ps1
# Phase 196K — Register TRITON_Data_Freshness_35Min (read-only, every 35 minutes).

[CmdletBinding()]
param(
    [string]$TaskName = "TRITON_Data_Freshness_35Min",
    [string]$RepoRoot = "C:\Users\akimw\triton",
    [int]$IntervalMinutes = 35,
    [string]$RefreshScript = $null
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RefreshScript)) {
    $RefreshScript = Join-Path $RepoRoot "scripts\run_data_freshness_refresh.ps1"
}

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repo root not found: $RepoRoot"
}
if (-not (Test-Path -LiteralPath $RefreshScript -PathType Leaf)) {
    throw "Refresh script not found: $RefreshScript"
}

$RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
$RefreshScript = (Resolve-Path -LiteralPath $RefreshScript).Path

$argString = (
    '-NoProfile -ExecutionPolicy Bypass -File "' + $RefreshScript + '" ' +
    '-RepoRoot "' + $RepoRoot + '"'
)

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $argString `
    -WorkingDirectory $RepoRoot

if ([string]::IsNullOrWhiteSpace($action.WorkingDirectory) -or
    $action.WorkingDirectory -ne $RepoRoot) {
    $action.WorkingDirectory = $RepoRoot
}

$startAt = (Get-Date).AddMinutes(2)
$trigger = New-ScheduledTaskTrigger -Daily -At $startAt
$repetitionSource = New-ScheduledTaskTrigger `
    -Once `
    -At $startAt.Date `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 1)
$trigger.Repetition = $repetitionSource.Repetition

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 25) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

$description = (
    "Phase 196K: read-only TRITON data freshness refresh every $IntervalMinutes minutes. " +
    "Snapshots + heartbeat only; no execute_trades, no approvals, no order placement."
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
        -Trigger $trigger `
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

$enabledSupervisors = @(
    Get-ScheduledTask |
        Where-Object {
            $_.State -eq "Ready" -and
            $_.TaskName -match "Runtime_Supervisor|FullPipeline_45Min"
        }
)

Write-Host ""
Write-Host "[TASK_INSTALL]" -ForegroundColor Green
Write-Host ("  name=" + $TaskName)
Write-Host ("  pattern=READ_ONLY_EVERY_" + $IntervalMinutes + "_MIN")
Write-Host ("  refresh_script=" + $RefreshScript)
Write-Host ("  working_dir=" + $persistedWd)
Write-Host ("  state=" + $verify.State)
Write-Host ("  first_run=" + $startAt.ToString("s"))
Write-Host ("  repetition_minutes=" + $IntervalMinutes)
Write-Host ("  runtime_supervisor_tasks=" + ($enabledSupervisors.TaskName -join ", "))
Write-Host ""
Write-Host "Does NOT start continuous loop or call execute_trades." -ForegroundColor Green
Write-Host ("Status: powershell -File scripts\show_triton_data_freshness_status.ps1")
Write-Host ("Run now: Start-ScheduledTask -TaskName " + $TaskName)

if ($enabledSupervisors.Count -gt 1) {
    Write-Host "[WARN] Multiple runtime supervisor tasks enabled." -ForegroundColor Yellow
    exit 2
}

exit 0
