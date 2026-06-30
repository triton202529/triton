# scripts/show_phase_197q_fix_runtime_process_and_env_authority_validation.ps1

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\akimw\triton"
)

$ErrorActionPreference = "Continue"

if (Test-Path -LiteralPath $RepoRoot -PathType Container) {
    $RepoRoot = ((Resolve-Path -LiteralPath $RepoRoot).Path).TrimEnd('\', '/')
}

$jsonPath = Join-Path $RepoRoot "data\results\phase_197q_fix_runtime_process_and_env_authority_validation.json"

Write-Host "=== TRITON Phase 197Q-FIX Runtime Process + Env Authority ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
    Write-Host "Report missing. Run:" -ForegroundColor Yellow
    Write-Host "  python -m services.phase_197q_fix_runtime_process_and_env_authority_validation" -ForegroundColor DarkGray
    exit 1
}

$r = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json

Write-Host ("Status: " + $r.status)
Write-Host ("Continuous loop running: " + $r.continuous_loop_running)
Write-Host ("Watchdog loop running: " + $r.watchdog_loop_running)
Write-Host ("Detected continuous PIDs: " + ($r.detected_continuous_process_ids -join ", "))
Write-Host ("Detected watchdog PIDs: " + ($r.detected_watchdog_process_ids -join ", "))
Write-Host ("Continuous lock exists: " + $r.continuous_lock_exists)
Write-Host ("Watchdog lock exists: " + $r.watchdog_lock_exists)
Write-Host ("Continuous process/lock state: " + $r.continuous_runtime_process_lock_state)
Write-Host ("Watchdog process/lock state: " + $r.watchdog_runtime_process_lock_state)
Write-Host ("Process detection confidence: " + $r.process_detection_confidence)
Write-Host ("Env authority state: " + $r.env_authority_state)
Write-Host ("Env source: " + $r.env_source)
Write-Host ("CLI validation env loaded: " + $r.cli_validation_env_loaded)
Write-Host ("Paper execution enabled: " + $r.paper_execution_enabled)
Write-Host ("Auto paper entries enabled: " + $r.auto_paper_entries_enabled)
Write-Host ("Auto paper approval enabled: " + $r.auto_paper_approval_enabled)
Write-Host ("Live execution enabled: " + $r.live_execution_enabled)
Write-Host ("Live force enabled: " + $r.live_force_enabled)
Write-Host ("Broker paper env present: " + $r.broker_paper_env_present)
Write-Host ("Secrets masked: " + $r.secrets_masked)
Write-Host ("Paper entry gate status: " + $r.paper_entry_gate_status)
Write-Host ("Paper entry gate reason: " + $r.paper_entry_gate_reason)
Write-Host ("197M status after env load: " + $r.phase_197m_status_after_env_load)
Write-Host ("197Q status after process fix: " + $r.phase_197q_status_after_process_fix)
Write-Host ("Live blocked: " + $r.live_blocked)
Write-Host ("Live block reason: " + $r.live_block_reason)
Write-Host ("Open orders: " + $r.open_orders_count)
Write-Host ("Real approval exists: " + $r.real_approval_artifact_exists)
Write-Host ("Eligible for execution: " + $r.eligible_for_execution)
Write-Host ("Restart recommendation: " + $r.restart_recommendation)
Write-Host ("Runtime re-entry state: " + $r.runtime_reentry_state)
Write-Host ""
Write-Host "=== End report ===" -ForegroundColor Cyan
exit 0
