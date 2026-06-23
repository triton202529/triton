# scripts/show_triton_data_freshness_status.ps1
# Phase 196K — Show TRITON data freshness task and artifact ages.

[CmdletBinding()]
param(
    [string]$TaskName = "TRITON_Data_Freshness_35Min",
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

function Get-ArtifactAgeMinutes($path) {
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    $item = Get-Item -LiteralPath $path
    return ((Get-Date) - $item.LastWriteTime).TotalMinutes
}

Write-Host "=== TRITON Data Freshness Status (Phase 196K) ===" -ForegroundColor Cyan
Write-Host ""

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
Write-Host ("Task exists: " + ($null -ne $task))
if ($task) {
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Host ("Task enabled: " + ($task.State -eq "Ready"))
    Write-Host ("Task state: " + $task.State)
    Write-Host ("Last run time: " + $info.LastRunTime)
    Write-Host ("Last task result: " + $info.LastTaskResult)
    Write-Host ("Next run time: " + $info.NextRunTime)
    Write-Host ("Task action: " + $task.Actions[0].Execute + " " + $task.Actions[0].Arguments)
} else {
    Write-Host "Task enabled: false"
}

Write-Host ""
Write-Host "--- Enabled freshness / supervisor tasks ---" -ForegroundColor Cyan
Get-ScheduledTask |
    Where-Object { $_.TaskName -match "TRITON|Triton|triton" } |
    Where-Object { $_.State -eq "Ready" } |
    Where-Object { $_.TaskName -match "Data_Freshness|Runtime_Supervisor|FullPipeline" } |
    Select-Object TaskName, State |
    Format-Table -AutoSize

Write-Host "--- Data freshness process ---" -ForegroundColor Cyan
$running = $false
if (Get-Command Test-LoopProcessRunning -ErrorAction SilentlyContinue) {
    $running = Test-LoopProcessRunning -ScriptName "run_data_freshness_refresh.ps1"
}
Write-Host ("Data freshness refresh running: " + $running)

$lockPath = Join-Path $RepoRoot "data\live\data_freshness_refresh.lock"
if (Test-Path -LiteralPath $lockPath) {
    $lock = Get-Item -LiteralPath $lockPath
    Write-Host ("data_freshness_refresh.lock: present | modified " + $lock.LastWriteTime)
    try { Write-Host (Get-Content -LiteralPath $lockPath -Raw) } catch {}
} else {
    Write-Host "data_freshness_refresh.lock: missing"
}

Write-Host ""
Write-Host "--- Artifact ages (minutes) ---" -ForegroundColor Cyan
$artifacts = @{
    "heartbeat.json"            = Join-Path $RepoRoot "data\results\heartbeat.json"
    "positions_snapshot.csv"    = Join-Path $RepoRoot "data\results\positions_snapshot.csv"
    "open_orders_snapshot.csv"  = Join-Path $RepoRoot "data\results\open_orders_snapshot.csv"
    "portfolio_history.csv"     = Join-Path $RepoRoot "data\results\portfolio_history.csv"
    "data_freshness_audit.json" = Join-Path $RepoRoot "data\results\data_freshness_refresh_audit.json"
}
foreach ($name in $artifacts.Keys) {
    $age = Get-ArtifactAgeMinutes $artifacts[$name]
    if ($null -eq $age) {
        Write-Host ("{0}: missing" -f $name)
    } else {
        Write-Host ("{0}: ~{1:N1} min" -f $name, $age)
    }
}

Write-Host ""
Write-Host "--- Broker snapshot summary ---" -ForegroundColor Cyan
try {
    $py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $py)) { $py = "python" }
    $summary = & $py -c @"
from services.dashboard_status_sync import count_open_orders, load_positions_summary
p = load_positions_summary()
print('positions', p.get('positions'))
print('open_orders', count_open_orders())
"@ 2>$null
    if ($summary) { $summary | ForEach-Object { Write-Host $_ } }
} catch {
    Write-Host "Could not load positions summary."
}

Write-Host ""
Write-Host "=== End status ===" -ForegroundColor Cyan
