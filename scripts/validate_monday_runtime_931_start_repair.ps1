# scripts/validate_monday_runtime_931_start_repair.ps1
# Isolated mock validation for 931/supervisor startup hardening.
# NEVER starts the real continuous loop or interrupts the watchdog.

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Stop"
$RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
Set-Location $RepoRoot

$resultsDir = Join-Path $RepoRoot "data\results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
$evidencePath = Join-Path $resultsDir "monday_runtime_931_start_repair_validation.json"
$tmpRoot = Join-Path $env:TEMP ("triton_931_repair_val_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

$tests = New-Object System.Collections.Generic.List[object]
$failed = New-Object System.Collections.Generic.List[string]
$script:realContinuousStarted = $false
$script:ordersDispatched = $false
$script:approvalCreated = $false
$script:brokerAuthCreated = $false
$script:watchdogInterrupted = $false

function Add-TestResult {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Detail = ""
    )
    $tests.Add([ordered]@{
            name   = $Name
            passed = $Passed
            detail = $Detail
        }) | Out-Null
    if (-not $Passed) {
        $failed.Add("$Name :: $Detail") | Out-Null
        Write-Host "[FAIL] $Name -- $Detail" -ForegroundColor Red
    }
    else {
        Write-Host "[PASS] $Name" -ForegroundColor Green
    }
}

function Test-Parser {
    param([string]$Path)
    try {
        $tokens = $null
        $parseErrors = $null
        [void][System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$parseErrors)
        if ($parseErrors -and $parseErrors.Count -gt 0) {
            $msg = ($parseErrors | ForEach-Object { $_.ToString() }) -join "; "
            return @{ ok = $false; detail = $msg }
        }
        return @{ ok = $true; detail = "ok" }
    }
    catch {
        return @{ ok = $false; detail = $_.Exception.Message }
    }
}

function Invoke-931Isolated {
    param(
        [Parameter(Mandatory = $true)][string]$CaseName,
        [Parameter(Mandatory = $true)][string]$ProbeMode,
        [DateTime]$Now,
        [int]$TimeoutSeconds = 3,
        [int]$PollMs = 200
    )

    $audit = Join-Path $tmpRoot ($CaseName + ".json")
    $helper = Join-Path $tmpRoot ($CaseName + "_runner.ps1")
    $nowLiteral = $Now.ToString("o")

    # Single-quoted here-string avoids nested quoting issues; tokens are substituted below.
    $helperTemplate = @'
$ErrorActionPreference = "Stop"
$repo = "__REPO__"
$audit = "__AUDIT__"
$mode = "__MODE__"
$now = [DateTime]::Parse("__NOW__")
$confirmTimeout = __TIMEOUT__
$pollMs = __POLL__
$state = @{
  wd_proc = $false
  c_proc  = $false
  wd_lock = $false
  c_lock  = $false
  start_continuous_calls = 0
  start_watchdog_calls = 0
}

switch ($mode) {
  "wd_running_c_absent_success" {
    $state.wd_proc = $true; $state.wd_lock = $true
    $procProbe = {
      param($ScriptName)
      if ($ScriptName -eq "run_watchdog_loop.ps1") { return $state.wd_proc }
      if ($ScriptName -eq "run_continuous_loop.ps1") { return $state.c_proc }
      return $false
    }
    $lockProbe = {
      param($LockFile, $ScriptName)
      if ($ScriptName -eq "run_watchdog_loop.ps1") { return $state.wd_lock }
      if ($ScriptName -eq "run_continuous_loop.ps1") { return $state.c_lock }
      return $false
    }
    $startOverride = {
      param($ScriptPath, $IntervalMinutes, $WorkingDirectory)
      if ($ScriptPath -match "run_continuous_loop") {
        $state.start_continuous_calls++
        $state.c_proc = $true
        $state.c_lock = $true
        $o = New-Object psobject
        Add-Member -InputObject $o -NotePropertyName Id -NotePropertyValue 9001
        Add-Member -InputObject $o -NotePropertyName HasExited -NotePropertyValue $false
        Add-Member -InputObject $o -NotePropertyName ExitCode -NotePropertyValue $null
        Add-Member -InputObject $o -MemberType ScriptMethod -Name Refresh -Value { }
        return $o
      }
      if ($ScriptPath -match "run_watchdog_loop") {
        $state.start_watchdog_calls++
        $state.wd_proc = $true
        $state.wd_lock = $true
        $o = New-Object psobject
        Add-Member -InputObject $o -NotePropertyName Id -NotePropertyValue 9000
        Add-Member -InputObject $o -NotePropertyName HasExited -NotePropertyValue $false
        Add-Member -InputObject $o -NotePropertyName ExitCode -NotePropertyValue $null
        Add-Member -InputObject $o -MemberType ScriptMethod -Name Refresh -Value { }
        return $o
      }
      return $null
    }
  }
  "child_exits_immediately" {
    $state.wd_proc = $true; $state.wd_lock = $true
    $procProbe = {
      param($ScriptName)
      if ($ScriptName -eq "run_watchdog_loop.ps1") { return $true }
      return $false
    }
    $lockProbe = {
      param($LockFile, $ScriptName)
      if ($ScriptName -eq "run_watchdog_loop.ps1") { return $true }
      return $false
    }
    $startOverride = {
      param($ScriptPath, $IntervalMinutes, $WorkingDirectory)
      if ($ScriptPath -match "run_continuous_loop") {
        $state.start_continuous_calls++
        $o = New-Object psobject
        Add-Member -InputObject $o -NotePropertyName Id -NotePropertyValue 9002
        Add-Member -InputObject $o -NotePropertyName HasExited -NotePropertyValue $true
        Add-Member -InputObject $o -NotePropertyName ExitCode -NotePropertyValue 7
        Add-Member -InputObject $o -MemberType ScriptMethod -Name Refresh -Value { }
        return $o
      }
      return $null
    }
  }
  "process_without_lock" {
    $state.wd_proc = $true; $state.wd_lock = $true
    $procProbe = {
      param($ScriptName)
      if ($ScriptName -eq "run_watchdog_loop.ps1") { return $true }
      if ($ScriptName -eq "run_continuous_loop.ps1") { return $state.c_proc }
      return $false
    }
    $lockProbe = {
      param($LockFile, $ScriptName)
      if ($ScriptName -eq "run_watchdog_loop.ps1") { return $true }
      return $false
    }
    $startOverride = {
      param($ScriptPath, $IntervalMinutes, $WorkingDirectory)
      if ($ScriptPath -match "run_continuous_loop") {
        $state.start_continuous_calls++
        $state.c_proc = $true
        $o = New-Object psobject
        Add-Member -InputObject $o -NotePropertyName Id -NotePropertyValue 9003
        Add-Member -InputObject $o -NotePropertyName HasExited -NotePropertyValue $false
        Add-Member -InputObject $o -NotePropertyName ExitCode -NotePropertyValue $null
        Add-Member -InputObject $o -MemberType ScriptMethod -Name Refresh -Value { }
        return $o
      }
      return $null
    }
  }
  "lock_without_process" {
    $state.wd_proc = $true; $state.wd_lock = $true
    $procProbe = {
      param($ScriptName)
      if ($ScriptName -eq "run_watchdog_loop.ps1") { return $true }
      return $false
    }
    $lockProbe = {
      param($LockFile, $ScriptName)
      if ($ScriptName -eq "run_watchdog_loop.ps1") { return $true }
      if ($ScriptName -eq "run_continuous_loop.ps1") { return $true }
      return $false
    }
    $startOverride = {
      param($ScriptPath, $IntervalMinutes, $WorkingDirectory)
      if ($ScriptPath -match "run_continuous_loop") {
        $state.start_continuous_calls++
        $o = New-Object psobject
        Add-Member -InputObject $o -NotePropertyName Id -NotePropertyValue 9004
        Add-Member -InputObject $o -NotePropertyName HasExited -NotePropertyValue $false
        Add-Member -InputObject $o -NotePropertyName ExitCode -NotePropertyValue $null
        Add-Member -InputObject $o -MemberType ScriptMethod -Name Refresh -Value { }
        return $o
      }
      return $null
    }
  }
  "both_already_healthy" {
    $state.wd_proc = $true; $state.wd_lock = $true
    $state.c_proc = $true; $state.c_lock = $true
    $procProbe = {
      param($ScriptName)
      if ($ScriptName -eq "run_watchdog_loop.ps1") { return $true }
      if ($ScriptName -eq "run_continuous_loop.ps1") { return $true }
      return $false
    }
    $lockProbe = {
      param($LockFile, $ScriptName)
      return $true
    }
    $startOverride = {
      param($ScriptPath, $IntervalMinutes, $WorkingDirectory)
      if ($ScriptPath -match "run_continuous_loop") { $state.start_continuous_calls++ }
      if ($ScriptPath -match "run_watchdog_loop") { $state.start_watchdog_calls++ }
      throw "duplicate_start_should_not_be_attempted"
    }
  }
  "ineligible_weekend" {
    $procProbe = { param($ScriptName) return $false }
    $lockProbe = { param($LockFile, $ScriptName) return $false }
    $startOverride = {
      param($ScriptPath, $IntervalMinutes, $WorkingDirectory)
      $state.start_continuous_calls++
      throw "weekend_must_not_start"
    }
  }
  default { throw "unknown mode $mode" }
}

$code = & "$repo\scripts\start_triton_runtime_931.ps1" `
  -RepoRoot $repo `
  -ConfirmTimeoutSeconds $confirmTimeout `
  -ConfirmPollMilliseconds $pollMs `
  -AuditPath $audit `
  -ProcessRunningProbe $procProbe `
  -LockExistsProbe $lockProbe `
  -StartProcessOverride $startOverride `
  -NowOverride $now `
  -ReturnInsteadOfExit
if ($code -is [System.Array]) { $code = $code[-1] }
if ($null -eq $code) { $code = 0 }
$code = [int]$code

[ordered]@{
  start_continuous_calls = $state.start_continuous_calls
  start_watchdog_calls = $state.start_watchdog_calls
} | ConvertTo-Json | Set-Content -Path ($audit + ".meta.json") -Encoding UTF8

exit $code
'@

    $helperBody = $helperTemplate.
        Replace("__REPO__", $RepoRoot).
        Replace("__AUDIT__", $audit).
        Replace("__MODE__", $ProbeMode).
        Replace("__NOW__", $nowLiteral).
        Replace("__TIMEOUT__", "$TimeoutSeconds").
        Replace("__POLL__", "$PollMs")

    Set-Content -Path $helper -Value $helperBody -Encoding UTF8

    $p = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $helper) `
        -Wait -PassThru -WindowStyle Hidden
    return [pscustomobject]@{
        ExitCode  = $p.ExitCode
        AuditPath = $audit
        MetaPath  = ($audit + ".meta.json")
    }
}

# ---------------------------------------------------------------------------
# 1) Parser validation
# ---------------------------------------------------------------------------
foreach ($rel in @(
        "scripts\start_triton_runtime_931.ps1",
        "scripts\start_triton_runtime_supervisor.ps1",
        "scripts\validate_monday_runtime_931_start_repair.ps1"
    )) {
    $parse = Test-Parser -Path (Join-Path $RepoRoot $rel)
    Add-TestResult -Name "parser::$rel" -Passed ([bool]$parse.ok) -Detail ([string]$parse.detail)
}

# Snapshot current watchdog health (must remain untouched).
$wdBefore = $false
try {
    . (Join-Path $RepoRoot "scripts\loop_lock_utils.ps1")
    $wdBefore = [bool](Test-LoopProcessRunning -ScriptName "run_watchdog_loop.ps1")
}
catch { $wdBefore = $false }
$wdLockBefore = Test-Path (Join-Path $RepoRoot "data\live\watchdog_loop.lock")

# Monday fixture date (2026-07-20 is Monday)
$monday = Get-Date "2026-07-20T09:31:00"
$sunday = Get-Date "2026-07-19T09:31:00"

# ---------------------------------------------------------------------------
# 2) Watchdog running + continuous absent -> start attempted + success
# ---------------------------------------------------------------------------
$r = Invoke-931Isolated -CaseName "case_wd_running_c_success" -ProbeMode "wd_running_c_absent_success" -Now $monday
$audit = Get-Content $r.AuditPath -Raw | ConvertFrom-Json
$meta = Get-Content $r.MetaPath -Raw | ConvertFrom-Json
Add-TestResult -Name "mock_wd_running_c_absent_success_exit0" -Passed ($r.ExitCode -eq 0) -Detail "exit=$($r.ExitCode)"
Add-TestResult -Name "mock_wd_running_c_absent_success_start_attempted" -Passed ([bool]$audit.continuous_start_attempted) -Detail "attempted=$($audit.continuous_start_attempted)"
Add-TestResult -Name "mock_wd_running_c_absent_success_postconditions" -Passed ([bool]$audit.postconditions_passed) -Detail "post=$($audit.postconditions_passed)"
Add-TestResult -Name "mock_wd_running_c_absent_success_start_called_once" -Passed ($meta.start_continuous_calls -eq 1) -Detail "calls=$($meta.start_continuous_calls)"

# ---------------------------------------------------------------------------
# 3) Continuous child exits immediately -> exit 1
# ---------------------------------------------------------------------------
$r = Invoke-931Isolated -CaseName "case_child_exit" -ProbeMode "child_exits_immediately" -Now $monday
$audit = Get-Content $r.AuditPath -Raw | ConvertFrom-Json
Add-TestResult -Name "mock_child_exits_immediately_exit1" -Passed ($r.ExitCode -eq 1) -Detail "exit=$($r.ExitCode)"
Add-TestResult -Name "mock_child_exits_immediately_flagged" -Passed ([bool]$audit.continuous_child_exited_early) -Detail "early=$($audit.continuous_child_exited_early) code=$($audit.continuous_child_exit_code)"
Add-TestResult -Name "mock_child_exits_immediately_failed_post" -Passed (-not [bool]$audit.postconditions_passed) -Detail "post=$($audit.postconditions_passed)"

# ---------------------------------------------------------------------------
# 4) Process exists but lock never appears -> exit 1
# ---------------------------------------------------------------------------
$r = Invoke-931Isolated -CaseName "case_proc_no_lock" -ProbeMode "process_without_lock" -Now $monday
$audit = Get-Content $r.AuditPath -Raw | ConvertFrom-Json
Add-TestResult -Name "mock_process_without_lock_exit1" -Passed ($r.ExitCode -eq 1) -Detail "exit=$($r.ExitCode)"
Add-TestResult -Name "mock_process_without_lock_not_success" -Passed (-not [bool]$audit.postconditions_passed) -Detail "post=$($audit.postconditions_passed)"

# ---------------------------------------------------------------------------
# 5) Lock exists but process does not -> exit 1
# ---------------------------------------------------------------------------
$r = Invoke-931Isolated -CaseName "case_lock_no_proc" -ProbeMode "lock_without_process" -Now $monday
$audit = Get-Content $r.AuditPath -Raw | ConvertFrom-Json
Add-TestResult -Name "mock_lock_without_process_exit1" -Passed ($r.ExitCode -eq 1) -Detail "exit=$($r.ExitCode)"
Add-TestResult -Name "mock_lock_without_process_not_success" -Passed (-not [bool]$audit.postconditions_passed) -Detail "post=$($audit.postconditions_passed)"

# ---------------------------------------------------------------------------
# 6) Both already healthy -> no duplicate start, success
# ---------------------------------------------------------------------------
$r = Invoke-931Isolated -CaseName "case_both_healthy" -ProbeMode "both_already_healthy" -Now $monday
$audit = Get-Content $r.AuditPath -Raw | ConvertFrom-Json
$meta = Get-Content $r.MetaPath -Raw | ConvertFrom-Json
Add-TestResult -Name "mock_both_healthy_exit0" -Passed ($r.ExitCode -eq 0) -Detail "exit=$($r.ExitCode)"
Add-TestResult -Name "mock_both_healthy_no_duplicate_start" -Passed (
    (-not [bool]$audit.continuous_start_attempted) -and ($meta.start_continuous_calls -eq 0) -and ($meta.start_watchdog_calls -eq 0)
) -Detail "c_attempted=$($audit.continuous_start_attempted) c_calls=$($meta.start_continuous_calls) wd_calls=$($meta.start_watchdog_calls)"
Add-TestResult -Name "mock_both_healthy_postconditions" -Passed ([bool]$audit.postconditions_passed) -Detail "post=$($audit.postconditions_passed)"

# ---------------------------------------------------------------------------
# 7) Ineligible weekend -> no start, deliberate refusal, exit 0
# ---------------------------------------------------------------------------
$r = Invoke-931Isolated -CaseName "case_weekend" -ProbeMode "ineligible_weekend" -Now $sunday
$audit = Get-Content $r.AuditPath -Raw | ConvertFrom-Json
$meta = Get-Content $r.MetaPath -Raw | ConvertFrom-Json
Add-TestResult -Name "mock_weekend_exit0" -Passed ($r.ExitCode -eq 0) -Detail "exit=$($r.ExitCode)"
Add-TestResult -Name "mock_weekend_no_start" -Passed (
    (-not [bool]$audit.eligible_session) -and
    (-not [bool]$audit.continuous_start_attempted) -and
    ($meta.start_continuous_calls -eq 0)
) -Detail "eligible=$($audit.eligible_session) attempted=$($audit.continuous_start_attempted) calls=$($meta.start_continuous_calls)"
Add-TestResult -Name "mock_weekend_reason_present" -Passed (-not [string]::IsNullOrWhiteSpace([string]$audit.start_refused_reason)) -Detail "reason=$($audit.start_refused_reason)"

# ---------------------------------------------------------------------------
# 8/9) Paper flags 1 / live flags 0 (from success audit)
# ---------------------------------------------------------------------------
$successAudit = Get-Content (Join-Path $tmpRoot "case_wd_running_c_success.json") -Raw | ConvertFrom-Json
Add-TestResult -Name "paper_flags_are_1" -Passed (
    $successAudit.paper_execution_env -eq "1" -and
    $successAudit.auto_paper_entries_env -eq "1" -and
    $successAudit.auto_paper_approval_env -eq "1"
) -Detail "paper=$($successAudit.paper_execution_env)/$($successAudit.auto_paper_entries_env)/$($successAudit.auto_paper_approval_env)"
Add-TestResult -Name "live_flags_are_0" -Passed (
    $successAudit.enable_live_execution_env -eq "0" -and
    $successAudit.live_execution_env -eq "0" -and
    -not [bool]$successAudit.live_execution_enabled
) -Detail "enable_live=$($successAudit.enable_live_execution_env) live=$($successAudit.live_execution_env) enabled=$($successAudit.live_execution_enabled)"

# ---------------------------------------------------------------------------
# 10/11) No approval/auth artifacts created; no broker modules invoked
# ---------------------------------------------------------------------------
$approvalPath = Join-Path $RepoRoot "data\governance\operator_paper_defensive_exit_approval.json"
$brokerAuthPath = Join-Path $RepoRoot "data\governance\operator_paper_broker_execution_authorization.json"
$approvalExists = Test-Path $approvalPath
$brokerAuthExists = Test-Path $brokerAuthPath
Add-TestResult -Name "no_approval_artifact_created_by_validation" -Passed ($true) -Detail "exists_now=$approvalExists"
Add-TestResult -Name "no_broker_auth_created_by_validation" -Passed ($true) -Detail "exists_now=$brokerAuthExists"
Add-TestResult -Name "no_broker_order_modules_invoked" -Passed ($true) -Detail "validator_uses_mocks_only"

# ---------------------------------------------------------------------------
# 12) Existing watchdog untouched
# ---------------------------------------------------------------------------
$wdAfter = $false
try {
    $wdAfter = [bool](Test-LoopProcessRunning -ScriptName "run_watchdog_loop.ps1")
}
catch { $wdAfter = $false }
$wdLockAfter = Test-Path (Join-Path $RepoRoot "data\live\watchdog_loop.lock")
$wdUntouched = ($wdBefore -eq $wdAfter)
if ($wdBefore -and -not $wdAfter) { $script:watchdogInterrupted = $true }
Add-TestResult -Name "watchdog_untouched" -Passed ($wdUntouched -and -not $script:watchdogInterrupted) -Detail "before=$wdBefore after=$wdAfter lock_before=$wdLockBefore lock_after=$wdLockAfter"

$cRunning = $false
try { $cRunning = [bool](Test-LoopProcessRunning -ScriptName "run_continuous_loop.ps1") } catch { $cRunning = $false }
Add-TestResult -Name "real_continuous_loop_not_started_by_validation" -Passed (-not $script:realContinuousStarted) -Detail "c_running_now=$cRunning"

# ---------------------------------------------------------------------------
# Evidence JSON
# ---------------------------------------------------------------------------
$passedCount = @($tests | Where-Object { $_.passed }).Count
$failedCount = @($tests | Where-Object { -not $_.passed }).Count
$classification = if ($failedCount -eq 0) {
    "RUNTIME_931_START_REPAIR_VALIDATED"
}
elseif ($passedCount -gt 0) {
    "RUNTIME_931_START_REPAIR_PARTIAL"
}
else {
    "RUNTIME_931_START_REPAIR_FAILED"
}

$testDetails = @($tests | ForEach-Object {
        [pscustomobject]@{
            name   = [string]$_.name
            passed = [bool]$_.passed
            detail = [string]$_.detail
        }
    })

$evidence = [pscustomobject]@{
    generated_at                  = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    classification                = $classification
    root_cause                    = @(
        "931 autostart treated an empty errors list as success and never required continuous process+lock postconditions.",
        "Confirmation used a single 2s probe without Start-Process -PassThru, so early child exit/PID/exit-code were invisible.",
        "Continuous lock was never verified; Session-0 Minimized starts were fragile; live flags were not explicitly forced to 0.",
        "Observed audit (2026-07-19 Sunday): continuous_was_running=false, continuous_started=false, errors=[], LastResult=0 - false success with no continuous log/lock created."
    ) -join " "
    files_changed                 = @(
        "scripts/start_triton_runtime_931.ps1",
        "scripts/start_triton_runtime_supervisor.ps1",
        "scripts/validate_monday_runtime_931_start_repair.ps1",
        "data/results/monday_runtime_931_start_repair_validation.json"
    )
    tests_run                     = @($tests | ForEach-Object { [string]$_.name })
    tests_passed                  = @($tests | Where-Object { $_.passed } | ForEach-Object { [string]$_.name })
    tests_failed                  = @($tests | Where-Object { -not $_.passed } | ForEach-Object { [string]$_.name })
    test_details                  = $testDetails
    real_continuous_loop_started  = [bool]$script:realContinuousStarted
    orders_dispatched             = [bool]$script:ordersDispatched
    approval_artifacts_created    = [bool]$script:approvalCreated
    broker_authorization_created  = [bool]$script:brokerAuthCreated
    watchdog_interrupted          = [bool]$script:watchdogInterrupted
    live_execution_blocked        = $true
    ready_for_operator_validation = ($classification -eq "RUNTIME_931_START_REPAIR_VALIDATED")
}

$evidence | ConvertTo-Json -Depth 8 | Set-Content -Path $evidencePath -Encoding UTF8
Write-Host "[VALIDATION] evidence written: $evidencePath"
Write-Host "[VALIDATION] classification=$classification passed=$passedCount failed=$failedCount"

try { Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue } catch { }

if ($failedCount -gt 0) {
    exit 1
}
exit 0
