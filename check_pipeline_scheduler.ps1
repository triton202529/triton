$ErrorActionPreference = "Continue"

$taskName = "TRITON_FullPipeline_45Min"
$wrapperPath = "C:\Users\akimw\triton\run_pipeline_scheduled.ps1"
$logPath = "C:\Users\akimw\triton\data\results\pipeline_scheduled.log"
$lockPath = "C:\Users\akimw\triton\data\results\pipeline_scheduled.lock"

Write-Host "=== Scheduled task: $taskName ===" -ForegroundColor Cyan
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "Task not found."
    $info = $null
} else {
    $info = Get-ScheduledTaskInfo -InputObject $task
    Write-Host "State:           $($task.State)"
    Write-Host "Last run time:   $(if ($info.LastRunTime) { $info.LastRunTime } else { '(never)' })"
    Write-Host "Next run time:   $(if ($info.NextRunTime) { $info.NextRunTime } else { '(none)' })"
    Write-Host "Last task result: $($info.LastTaskResult)  (0 = success)"
}

Write-Host ""
Write-Host "=== Files ===" -ForegroundColor Cyan
$wrapperExists = Test-Path -LiteralPath $wrapperPath
$logExists = Test-Path -LiteralPath $logPath
$lockExists = Test-Path -LiteralPath $lockPath
Write-Host "Wrapper script: $wrapperPath  -> $(if ($wrapperExists) { 'OK' } else { 'MISSING' })"
Write-Host "Log file:       $logPath  -> $(if ($logExists) { 'OK' } else { 'MISSING' })"
Write-Host "Lock file:      $lockPath  -> $(if ($lockExists) { 'present' } else { 'absent' })"

Write-Host ""
Write-Host "=== Last 20 log lines ===" -ForegroundColor Cyan
if ($logExists) {
    Get-Content -LiteralPath $logPath -Tail 20 -ErrorAction SilentlyContinue
} else {
    Write-Host "(log not found)"
}

Write-Host ""
Write-Host "=== Lock age ===" -ForegroundColor Cyan
if ($lockExists) {
    $ageMinutes = [math]::Round(((Get-Date) - (Get-Item -LiteralPath $lockPath).LastWriteTime).TotalMinutes, 2)
    Write-Host "Lock age: $ageMinutes minutes (by LastWriteTime)"
} else {
    Write-Host "(no lock file)"
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
$schedulerOk = ($null -ne $task) -and ($task.State -ne "Disabled")
$wrapperOk = $wrapperExists
$logOk = $logExists
Write-Host "scheduler_ok = $(if ($schedulerOk) { 'true' } else { 'false' })"
Write-Host "wrapper_ok   = $(if ($wrapperOk) { 'true' } else { 'false' })"
Write-Host "log_ok       = $(if ($logOk) { 'true' } else { 'false' })"
