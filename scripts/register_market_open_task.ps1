# scripts/register_market_open_task.ps1
# ------------------------------------------------------------
# TRITON -- Windows Task Scheduler setup for the market-open runner.
#
# Registers (or refreshes) a scheduled task that launches
#   scripts/run_market_open.ps1
# every weekday at 9:31 AM local machine time. Idempotent: if the task
# already exists, it is unregistered and re-created so re-running this
# script never produces duplicate triggers or stale settings.
#
# Resulting task action (matches the spec exactly):
#   Execute          : powershell.exe
#   Argument         : -ExecutionPolicy Bypass -File "<RepoRoot>\scripts\run_market_open.ps1"
#   WorkingDirectory : <RepoRoot>           <-- pinned so the runner can
#                                                use repo-relative paths
#                                                (python -m services....)
#
# Original failure this script defends against:
#   LastTaskResult = 2147942667 (= 0x8007010B = ERROR_DIRECTORY)
# Task Scheduler emits this BEFORE the action launches when the
# persisted <WorkingDirectory> element in the task XML is empty or
# unresolvable. This script:
#   1. Canonicalises $RepoRoot via Resolve-Path + TrimEnd to remove
#      trailing slashes / relative components / mixed-case quirks.
#   2. Re-asserts $action.WorkingDirectory after New-ScheduledTaskAction
#      to defend against the cmdlet's known silent-parameter-drop bug.
#   3. Re-reads the persisted task XML via Get-ScheduledTask after
#      registration and THROWS if WorkingDirectory did not survive
#      the round-trip -- so a broken registration can never appear
#      successful.
#
# Why a separate setup script (not just `schtasks /create`)?
#   - The ScheduledTasks PowerShell module gives us strongly-typed
#     New-ScheduledTask{Action,Trigger,Principal,Settings} objects so the
#     resulting task XML is consistent across Windows builds.
#   - Re-registering with Register-ScheduledTask -Force replaces atomically
#     instead of leaving partially-updated state if a re-create fails.
#   - We can pre-validate paths AND post-verify the persisted XML, then
#     emit a single structured [TASK_REGISTER] log line for tooling.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\register_market_open_task.ps1
#
# Optional parameters (defaults match the spec exactly):
#   -TaskName    "TRITON_MarketOpen"
#   -RepoRoot    "C:\Users\akimw\triton"
#   -ScriptPath  "<RepoRoot>\scripts\run_market_open.ps1"
#   -TimeOfDay   "09:31"
#
# Safety:
#   - Does NOT modify any Python trading logic; this is a pure OS-level
#     scheduler entry that invokes an existing PowerShell script.
#   - Runs the task as the current interactive user at the default
#     (Limited) run level, so no admin elevation is required and the
#     task inherits the user's normal environment / venv access.
#   - The task only fires when the user is logged in (LogonType Interactive),
#     matching the existing manual-run model for paper trading.

[CmdletBinding()]
param(
    [string]$TaskName   = "TRITON_MarketOpen",
    [string]$RepoRoot   = "C:\Users\akimw\triton",
    [string]$ScriptPath = $null,
    [string]$TimeOfDay  = "09:31"
)

$ErrorActionPreference = "Stop"

# ── Resolve defaults that depend on other params ─────────────────────
if ([string]::IsNullOrWhiteSpace($ScriptPath)) {
    $ScriptPath = Join-Path $RepoRoot "scripts\run_market_open.ps1"
}

# ── Validate inputs before touching the scheduler ────────────────────
if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "Repo root not found: $RepoRoot"
}
if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
    throw "Runner script not found: $ScriptPath"
}

# Canonicalise paths BEFORE handing them to the Task Scheduler API.
# The original failure (LastTaskResult = 2147942667 = 0x8007010B =
# ERROR_DIRECTORY) is emitted by Task Scheduler when the persisted
# <WorkingDirectory> element in the task XML is empty or unresolvable.
# Common ways this happens:
#   - The path had a trailing backslash that some Windows builds reject.
#   - The path was relative / contained '.' or '..'.
#   - The path was mixed-case / mixed-slash and the cmdlet silently
#     dropped it during XML serialisation.
# Resolve-Path returns an absolute, slash-normalised, case-correct
# string; TrimEnd then removes any stray trailing separator. This is
# the single most important fix in this version of the script.
$RepoRoot   = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\','/')
$ScriptPath =  (Resolve-Path -LiteralPath $ScriptPath).Path

# Parse the time string into hour/minute. Accept both 'HH:mm' and
# 'H:mm' so the user can pass '9:31' as well as '09:31'.
$timeParts = $TimeOfDay -split ":"
if ($timeParts.Count -ne 2) {
    throw "Invalid -TimeOfDay '$TimeOfDay' (expected HH:mm, e.g. '09:31')."
}
[int]$hour   = $timeParts[0]
[int]$minute = $timeParts[1]
if ($hour   -lt 0 -or $hour   -gt 23) { throw "Invalid hour in -TimeOfDay: $hour" }
if ($minute -lt 0 -or $minute -gt 59) { throw "Invalid minute in -TimeOfDay: $minute" }

# ScheduledTasks cmdlet expects a DateTime; the date portion is irrelevant
# for a recurring weekly trigger -- only the time-of-day is used.
$startAt = Get-Date -Hour $hour -Minute $minute -Second 0

# ── Build task components ────────────────────────────────────────────
# Use powershell.exe (Windows PowerShell 5.x) for max compatibility
# with the existing run_market_open.ps1 (which itself targets the
# bundled powershell.exe).
#
# Argument string is built to match the spec exactly:
#   -ExecutionPolicy Bypass -File "<absolute path to runner>"
# The runner itself is responsible for any further hardening (e.g.
# pinning $PSScriptRoot, checking the venv).
$argString = '-ExecutionPolicy Bypass -File "' + $ScriptPath + '"'

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $argString `
    -WorkingDirectory $RepoRoot

# Belt-and-suspenders: defend against the silent-parameter-drop quirk
# in New-ScheduledTaskAction that produced the original ERROR_DIRECTORY
# failure. If the cmdlet returned an action whose WorkingDirectory is
# empty/blank or different from what we asked for, force-set it on the
# CIM instance directly. The post-registration verify block below then
# confirms the value actually made it into the persisted task XML.
if ([string]::IsNullOrWhiteSpace($action.WorkingDirectory) -or
    $action.WorkingDirectory -ne $RepoRoot) {
    Write-Host ("[TASK_REGISTER] correcting Action.WorkingDirectory '" +
                $action.WorkingDirectory + "' -> '" + $RepoRoot + "'") `
               -ForegroundColor Yellow
    $action.WorkingDirectory = $RepoRoot
}

$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At $startAt

# Sensible production settings:
#   - StartWhenAvailable: if the machine was asleep at 9:31, run as soon
#     as it wakes (catch-up trigger).
#   - DontStopIfGoingOnBatteries / AllowStartIfOnBatteries: laptop-safe.
#   - ExecutionTimeLimit 1h: hard cap so a hung run cannot block tomorrow.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -MultipleInstances IgnoreNew

# Run as the current interactive user, no elevation. Matches the
# manual-run model: the user is logged in during US market hours.
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

$description = "Runs Triton paper trading market-open workflow: poll, manage orders, rebuild intelligence, manage positions."

# ── Idempotent replace ───────────────────────────────────────────────
# Detect existing registration first so we can give the operator a clear
# "replaced" vs "created" line. Register-ScheduledTask -Force would also
# overwrite, but the explicit Unregister keeps the audit trail honest.
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $existing) {
    Write-Host ("[TASK_REGISTER] existing task found; replacing: " + $TaskName) -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# ── Register ─────────────────────────────────────────────────────────
# Discard the returned task object; we re-fetch via Get-ScheduledTask
# below so the verify step exercises the OS-side state, not just the
# return value of Register-ScheduledTask.
Register-ScheduledTask `
    -TaskName    $TaskName `
    -Description $description `
    -Action      $action `
    -Trigger     $trigger `
    -Settings    $settings `
    -Principal   $principal `
    -Force | Out-Null

# ── Verify and emit the spec-mandated structured log line ────────────
# Re-read the task from the OS so we exercise the *persisted* XML, not
# just the in-memory action object. This is the verification step that
# would have caught the original 0x8007010B failure.
$verify = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
if ($null -eq $verify) {
    throw "Post-registration Get-ScheduledTask returned null for '$TaskName'."
}

$persistedExec = $verify.Actions[0].Execute
$persistedArgs = $verify.Actions[0].Arguments
$persistedWd   = $verify.Actions[0].WorkingDirectory

if ([string]::IsNullOrWhiteSpace($persistedWd)) {
    throw ("Post-registration verification FAILED: persisted " +
           "Action.WorkingDirectory is empty. This is the exact " +
           "condition that produces LastTaskResult = 2147942667 " +
           "(ERROR_DIRECTORY / 0x8007010B). Aborting.")
}
if ($persistedWd -ne $RepoRoot) {
    throw ("Post-registration verification FAILED: persisted " +
           "Action.WorkingDirectory='" + $persistedWd +
           "', expected='" + $RepoRoot + "'. Aborting.")
}

$scheduleSummary = ("Mon-Fri " + ("{0:D2}:{1:D2}" -f $hour, $minute) + " local")

Write-Host ""
Write-Host "[TASK_REGISTER]"             -ForegroundColor Green
Write-Host ("  name="        + $TaskName)
Write-Host ("  schedule="    + $scheduleSummary)
Write-Host ("  script="      + $ScriptPath)
Write-Host ("  working_dir=" + $persistedWd)
Write-Host ("  exec="        + $persistedExec)
Write-Host ("  args="        + $persistedArgs)
Write-Host ("  user="        + $env:USERNAME)
Write-Host ("  state="       + $verify.State)

Write-Host ""
Write-Host "Task registered successfully." -ForegroundColor Green
Write-Host "Inspect / edit in Task Scheduler -> Task Scheduler Library -> $TaskName"
Write-Host ("Run on demand:  Start-ScheduledTask -TaskName " + $TaskName)
Write-Host ("Remove:         Unregister-ScheduledTask -TaskName " + $TaskName + " -Confirm:" + '$false')
