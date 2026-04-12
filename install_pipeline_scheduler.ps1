#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"

$taskName = "TRITON_FullPipeline_45Min"
$scriptPath = "C:\Users\akimw\triton\run_pipeline_scheduled.ps1"

if (-not (Test-Path $scriptPath)) {
    throw "Missing scheduled runner script: $scriptPath"
}

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument '-ExecutionPolicy Bypass -File "C:\Users\akimw\triton\run_pipeline_scheduled.ps1"'

$trigger = New-ScheduledTaskTrigger -Daily -At "09:00"
$repetitionSource = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes 45) `
    -RepetitionDuration (New-TimeSpan -Days 1)
$trigger.Repetition = $repetitionSource.Repetition

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

$principal = New-ScheduledTaskPrincipal `
    -UserId "NT AUTHORITY\SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Runs TRITON full pipeline every 45 minutes (daily window, repeats for 1 day; no AC-power requirement)."

Write-Host "Success: Scheduled task '$taskName' is registered and ready."
