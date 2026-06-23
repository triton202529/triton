# scripts/start_triton_runtime_supervisor.ps1
# Phase 196H — Logon/startup runtime supervisor (watchdog + continuous loops only).
# Does NOT run execute_trades directly. Does NOT enable live execution.

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton",
    [int]$IntervalMinutes = 45
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repo root not found: $RepoRoot"
}
$RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
Set-Location $RepoRoot

$scriptDir = Join-Path $RepoRoot "scripts"
. (Join-Path $scriptDir "loop_lock_utils.ps1")

$auditPath = Join-Path $RepoRoot "data\results\runtime_supervisor_autostart.json"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "data\results") | Out-Null

$errors = @()

function Write-SupervisorLine($msg) {
    Write-Host $msg
}

# Paper-only env flags (never set live execution flags).
$env:TRITON_ENABLE_PAPER_EXECUTION = "1"
$env:TRITON_ENABLE_AUTO_PAPER_ENTRIES = "1"
$env:TRITON_ENABLE_AUTO_PAPER_APPROVAL = "1"

$watchdogScript = Join-Path $scriptDir "run_watchdog_loop.ps1"
$continuousScript = Join-Path $scriptDir "run_continuous_loop.ps1"

foreach ($path in @($watchdogScript, $continuousScript)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $errors += "missing_script=$path"
    }
}

$watchdogWasRunning = Test-LoopProcessRunning -ScriptName "run_watchdog_loop.ps1"
$continuousWasRunning = Test-LoopProcessRunning -ScriptName "run_continuous_loop.ps1"

$watchdogStarted = $false
$continuousStarted = $false

try {
    if (-not $watchdogWasRunning) {
        if (Test-Path -LiteralPath $watchdogScript -PathType Leaf) {
            Write-SupervisorLine "[RUNTIME_SUPERVISOR] starting watchdog loop IntervalMinutes=1"
            Start-Process -FilePath "powershell.exe" `
                -ArgumentList @(
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", $watchdogScript,
                    "-IntervalMinutes", "1"
                ) `
                -WorkingDirectory $RepoRoot `
                -WindowStyle Minimized
            Start-Sleep -Seconds 2
            $watchdogStarted = Test-LoopProcessRunning -ScriptName "run_watchdog_loop.ps1"
            if (-not $watchdogStarted) {
                $errors += "watchdog_start_unconfirmed"
            }
        }
    } else {
        Write-SupervisorLine "[RUNTIME_SUPERVISOR] watchdog already running - skip duplicate start"
    }

    if (-not $continuousWasRunning) {
        if (Test-Path -LiteralPath $continuousScript -PathType Leaf) {
            Write-SupervisorLine "[RUNTIME_SUPERVISOR] starting continuous loop IntervalMinutes=$IntervalMinutes"
            Start-Process -FilePath "powershell.exe" `
                -ArgumentList @(
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", $continuousScript,
                    "-IntervalMinutes", "$IntervalMinutes"
                ) `
                -WorkingDirectory $RepoRoot `
                -WindowStyle Minimized
            Start-Sleep -Seconds 2
            $continuousStarted = Test-LoopProcessRunning -ScriptName "run_continuous_loop.ps1"
            if (-not $continuousStarted) {
                $errors += "continuous_start_unconfirmed"
            }
        }
    } else {
        Write-SupervisorLine "[RUNTIME_SUPERVISOR] continuous loop already running - skip duplicate start"
    }
}
catch {
    $errors += "exception=$($_.Exception.Message)"
}

$audit = [ordered]@{
    timestamp              = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    phase                  = "196H"
    repo_root              = $RepoRoot
    interval_minutes       = $IntervalMinutes
    watchdog_was_running   = [bool]$watchdogWasRunning
    continuous_was_running = [bool]$continuousWasRunning
    watchdog_started       = [bool]$watchdogStarted
    continuous_started     = [bool]$continuousStarted
    paper_execution_env    = $env:TRITON_ENABLE_PAPER_EXECUTION
    live_execution_enabled = $false
    errors                 = @($errors)
}

$audit | ConvertTo-Json -Depth 4 | Set-Content -Path $auditPath -Encoding UTF8
Write-SupervisorLine "[RUNTIME_SUPERVISOR] audit written: $auditPath"

if ($errors.Count -gt 0) {
    Write-Host "[RUNTIME_SUPERVISOR] completed with errors: $($errors -join '; ')" -ForegroundColor Yellow
    exit 1
}

Write-Host "[RUNTIME_SUPERVISOR] completed successfully" -ForegroundColor Green
exit 0
