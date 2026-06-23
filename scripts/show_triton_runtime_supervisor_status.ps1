# scripts/show_triton_runtime_supervisor_status.ps1
# Phase 196H — Show TRITON runtime supervisor task and loop health.

[CmdletBinding()]
param(
    [string]$TaskName = "TRITON_Runtime_Supervisor_45Min",
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$scriptDir = Join-Path $RepoRoot "scripts"
if (Test-Path -LiteralPath (Join-Path $scriptDir "loop_lock_utils.ps1") -PathType Leaf) {
    . (Join-Path $scriptDir "loop_lock_utils.ps1")
}

Write-Host "=== TRITON Runtime Supervisor Status (Phase 196H) ===" -ForegroundColor Cyan
Write-Host ""

# Scheduled task
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$taskExists = $null -ne $task
Write-Host ("Task exists: " + $taskExists)

if ($taskExists) {
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Host ("Task state: " + $task.State)
    Write-Host ("Last run time: " + $info.LastRunTime)
    Write-Host ("Last task result: " + $info.LastTaskResult)
    Write-Host ("Next run time: " + $info.NextRunTime)
    Write-Host ("Task action: " + $task.Actions[0].Execute + " " + $task.Actions[0].Arguments)
    Write-Host ("Working directory: " + $task.Actions[0].WorkingDirectory)
} else {
    Write-Host "Task state: N/A"
}

Write-Host ""
Write-Host "--- Enabled TRITON supervisor-style tasks ---" -ForegroundColor Cyan
Get-ScheduledTask |
    Where-Object { $_.TaskName -match "TRITON|Triton|triton" } |
    Where-Object { $_.State -eq "Ready" } |
    Select-Object TaskName, State |
    Format-Table -AutoSize

Write-Host "--- Loop processes ---" -ForegroundColor Cyan
$filter = "Name = 'powershell.exe' OR Name = 'pwsh.exe'"
$procs = Get-CimInstance Win32_Process -Filter $filter -ErrorAction SilentlyContinue
$loopProcs = @()
foreach ($proc in @($procs)) {
    $cmd = $proc.CommandLine
    if ($cmd -and ($cmd -match "run_continuous_loop|run_watchdog_loop|start_triton_runtime_supervisor")) {
        $loopProcs += [pscustomobject]@{
            ProcessId   = $proc.ProcessId
            CommandLine = $cmd
        }
    }
}
if ($loopProcs.Count -eq 0) {
    Write-Host "No matching TRITON loop processes found."
} else {
    $loopProcs | Format-Table -AutoSize -Wrap
}

$watchdogRunning = $false
$continuousRunning = $false
if (Get-Command Test-LoopProcessRunning -ErrorAction SilentlyContinue) {
    $watchdogRunning = Test-LoopProcessRunning -ScriptName "run_watchdog_loop.ps1"
    $continuousRunning = Test-LoopProcessRunning -ScriptName "run_continuous_loop.ps1"
}
Write-Host ("Watchdog process running: " + $watchdogRunning)
Write-Host ("Continuous loop process running: " + $continuousRunning)

Write-Host ""
Write-Host "--- Lock files ---" -ForegroundColor Cyan
$lockPaths = @(
    (Join-Path $RepoRoot "data\live\continuous_loop.lock"),
    (Join-Path $RepoRoot "data\live\watchdog_loop.lock")
)
foreach ($lock in $lockPaths) {
    if (Test-Path -LiteralPath $lock) {
        $item = Get-Item -LiteralPath $lock
        Write-Host ("{0} | modified {1} | bytes {2}" -f $item.Name, $item.LastWriteTime, $item.Length)
        try {
            Write-Host (Get-Content -LiteralPath $lock -Raw)
        } catch {
            Write-Host "(could not read lock content)"
        }
    } else {
        Write-Host ("{0} | missing" -f (Split-Path $lock -Leaf))
    }
}

Write-Host ""
Write-Host "--- Heartbeat / status artifacts ---" -ForegroundColor Cyan
$artifactPaths = @(
    (Join-Path $RepoRoot "data\results\heartbeat.json"),
    (Join-Path $RepoRoot "data\results\watchdog_status.json"),
    (Join-Path $RepoRoot "data\results\runtime_supervisor_autostart.json")
)
foreach ($path in $artifactPaths) {
    $name = Split-Path $path -Leaf
    if (Test-Path -LiteralPath $path) {
        $item = Get-Item -LiteralPath $path
        $ageMin = ((Get-Date) - $item.LastWriteTime).TotalMinutes
        Write-Host ("{0} | modified {1} | age ~{2:N1} min" -f $name, $item.LastWriteTime, $ageMin)
    } else {
        Write-Host ("{0} | missing" -f $name)
    }
}

try {
    python -m services.dashboard_status_sync 2>$null | Out-Null
} catch {
    # optional
}

$hbJson = Join-Path $RepoRoot "data\results\heartbeat.json"
if (Test-Path -LiteralPath $hbJson) {
    try {
        $hb = Get-Content -LiteralPath $hbJson -Raw | ConvertFrom-Json
        $ts = $hb.timestamp
        if ($ts) {
            $parsed = [datetime]::Parse($ts).ToUniversalTime()
            $age = ((Get-Date).ToUniversalTime() - $parsed).TotalMinutes
            Write-Host ("Pipeline heartbeat timestamp: {0} (~{1:N1} min ago)" -f $ts, $age)
        }
    } catch {
        Write-Host "Could not parse heartbeat.json timestamp."
    }
}

Write-Host ""
Write-Host "=== End status ===" -ForegroundColor Cyan
