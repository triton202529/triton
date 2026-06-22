# scripts/start_triton_runtime_931.ps1
# Phase 196C — Daily market-open runtime autostart (watchdog + continuous loops only).
# Does NOT run execute_trades directly. Does NOT enable live execution.

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\akimw\triton"
if (-not (Test-Path -LiteralPath $repoRoot -PathType Container)) {
    throw "Repo root not found: $repoRoot"
}
$repoRoot = ((Resolve-Path -LiteralPath $repoRoot).Path).TrimEnd('\', '/')
Set-Location $repoRoot

$scriptDir = Join-Path $repoRoot "scripts"
. (Join-Path $scriptDir "loop_lock_utils.ps1")

$auditPath = Join-Path $repoRoot "data\results\runtime_autostart_931.json"
New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "data\results") | Out-Null

$errors = @()

function Write-AuditLine($msg) {
    Write-Host $msg
}

# Paper-only env flags (never set live execution flags).
$env:TRITON_ENABLE_PAPER_EXECUTION = "1"
$env:TRITON_ENABLE_AUTO_PAPER_ENTRIES = "1"
$env:TRITON_ENABLE_AUTO_PAPER_APPROVAL = "1"

$paperExecutionEnv = $env:TRITON_ENABLE_PAPER_EXECUTION
$autoPaperEntriesEnv = $env:TRITON_ENABLE_AUTO_PAPER_ENTRIES
$autoPaperApprovalEnv = $env:TRITON_ENABLE_AUTO_PAPER_APPROVAL

$watchdogScript = Join-Path $scriptDir "run_watchdog_loop.ps1"
$continuousScript = Join-Path $scriptDir "run_continuous_loop.ps1"

if (-not (Test-Path -LiteralPath $watchdogScript -PathType Leaf)) {
    $errors += "missing_script=$watchdogScript"
}
if (-not (Test-Path -LiteralPath $continuousScript -PathType Leaf)) {
    $errors += "missing_script=$continuousScript"
}

$watchdogWasRunning = Test-LoopProcessRunning -ScriptName "run_watchdog_loop.ps1"
$continuousWasRunning = Test-LoopProcessRunning -ScriptName "run_continuous_loop.ps1"

$watchdogStarted = $false
$continuousStarted = $false

try {
    if (-not $watchdogWasRunning) {
        if (Test-Path -LiteralPath $watchdogScript -PathType Leaf) {
            Write-AuditLine "[AUTOSTART_931] starting watchdog loop IntervalMinutes=1"
            Start-Process -FilePath "powershell.exe" `
                -ArgumentList @(
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", $watchdogScript,
                    "-IntervalMinutes", "1"
                ) `
                -WorkingDirectory $repoRoot `
                -WindowStyle Minimized
            Start-Sleep -Seconds 2
            $watchdogStarted = Test-LoopProcessRunning -ScriptName "run_watchdog_loop.ps1"
            if (-not $watchdogStarted) {
                $errors += "watchdog_start_unconfirmed"
            }
        }
    } else {
        Write-AuditLine "[AUTOSTART_931] watchdog already running — skip duplicate start"
    }

    if (-not $continuousWasRunning) {
        if (Test-Path -LiteralPath $continuousScript -PathType Leaf) {
            Write-AuditLine "[AUTOSTART_931] starting continuous loop IntervalMinutes=45"
            Start-Process -FilePath "powershell.exe" `
                -ArgumentList @(
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", $continuousScript,
                    "-IntervalMinutes", "45"
                ) `
                -WorkingDirectory $repoRoot `
                -WindowStyle Minimized
            Start-Sleep -Seconds 2
            $continuousStarted = Test-LoopProcessRunning -ScriptName "run_continuous_loop.ps1"
            if (-not $continuousStarted) {
                $errors += "continuous_start_unconfirmed"
            }
        }
    } else {
        Write-AuditLine "[AUTOSTART_931] continuous loop already running — skip duplicate start"
    }
}
catch {
    $errors += "exception=$($_.Exception.Message)"
}

$audit = [ordered]@{
    timestamp                  = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    repo_root                  = $repoRoot
    scheduled_time             = "09:31"
    watchdog_was_running       = [bool]$watchdogWasRunning
    continuous_was_running     = [bool]$continuousWasRunning
    watchdog_started           = [bool]$watchdogStarted
    continuous_started         = [bool]$continuousStarted
    paper_execution_env        = $paperExecutionEnv
    auto_paper_entries_env     = $autoPaperEntriesEnv
    auto_paper_approval_env    = $autoPaperApprovalEnv
    live_execution_enabled     = $false
    errors                     = @($errors)
}

$audit | ConvertTo-Json -Depth 4 | Set-Content -Path $auditPath -Encoding UTF8
Write-AuditLine "[AUTOSTART_931] audit written: $auditPath"

if ($errors.Count -gt 0) {
    Write-Host "[AUTOSTART_931] completed with errors: $($errors -join '; ')" -ForegroundColor Yellow
    exit 1
}

Write-Host "[AUTOSTART_931] completed successfully" -ForegroundColor Green
exit 0
