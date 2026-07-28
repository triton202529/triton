# DO-001 — Runtime Control Audit

**Phase:** DO-001 (read-only)
**Scope:** Startup, shutdown, locks, scheduled tasks, paper-arm, live-block, dashboard control gaps.

---

## 1. Verdict

Start/health/lock detection is **script- and service-backed**. The dashboard **cannot** safely start/stop/restart loops today. Graceful stop is a **gap**. Live remains blocked by **separate gates**, not by loop lifecycle alone.

---

## 2. Core scripts and services

| Area | Path |
|------|------|
| Continuous loop | `scripts/run_continuous_loop.ps1` |
| Watchdog loop | `scripts/run_watchdog_loop.ps1` |
| Lock utils | `scripts/loop_lock_utils.ps1` (`Test-LoopProcessRunning`, `Clear-OrphanLoopLock`, `Update-LoopLockMetadata`) |
| 9:31 autostart | `scripts/start_triton_runtime_931.ps1` → task `TRITON_Runtime_931AM` via `scripts/register_triton_runtime_931_task.ps1` |
| Logon supervisor | `scripts/start_triton_runtime_supervisor.ps1` → task `TRITON_Runtime_Supervisor_45Min` |
| Process detect | `services/runtime_process_detector.py` |
| Lock classify | `services/runtime_lock_classifier.py` |
| Heartbeat refresh | `services/loop_safety.py` |
| Health / recover | `services/continuous_loop_recovery_manager.py`, `hung_loop_detector.py`, `hung_loop_recovery_manager.py` |
| Dashboard sync | `services/dashboard_status_sync.py` |
| Paper arm | `services/paper_arm_status.py`, `services/set_paper_arm_confirm.py` |
| Live block | `services/live_block_classifier.py`, readiness `live_execution_permitted` |
| Read-only UIs | `dashboard/continuous_loop_health_monitor.py`, `dashboard/hung_loop_recovery_monitor.py` |

---

## 3. Lock and status artifacts

| File | Role |
|------|------|
| `data/live/continuous_loop.lock` | Continuous singleton (`pid`, `started_at`, `refreshed_at`, `cycle`, `source`) |
| `data/live/watchdog_loop.lock` | Watchdog singleton |
| `data/live/paper_arm_confirm.json` | ASSISTED paper latch |
| `data/live/live_confirm.json` | Live confirm (separate; must stay unused for operator Start) |
| `data/results/runtime_autostart_931.json` | 931 startup audit |
| `data/results/runtime_supervisor_autostart.json` | Supervisor startup audit |
| `data/results/watchdog_status.json` | Watchdog heartbeat |
| `data/results/heartbeat.json` | Pipeline heartbeat |
| `data/results/continuous_loop_health.json` | Continuous health report |
| `data/results/continuous_loop/continuous_loop_*.log` | Continuous logs |
| `data/results/watchdog_logs/watchdog_loop_*.log` | Watchdog logs |

Lock health classes (`runtime_lock_classifier`):
`LOCK_HEALTHY_ACTIVE`, `LOCK_TIMESTAMP_STALE_BUT_PROCESS_ACTIVE`, `STALE_ACTIVE_LOCK`, `STALE_ORPHAN_LOCK`, `NO_LOCK_RUNNING`, `NO_PROCESS_NO_LOCK`, `CORRUPT_LOCK`, `MISSING_ACTIVE_LOCK`.

Grace windows: watchdog ~**5 min**; continuous ~**60–67.5 min** (`CONTINUOUS_GRACE_MINUTES`).

---

## 4. Exact safe paths (operator truth)

### 4.1 Detect healthy continuous loop

1. `detect_runtime_loop_processes()` → cmdline match for `run_continuous_loop.ps1`
2. `audit_runtime_locks()` / `classify_runtime_lock` → healthy lock class
3. Optional: `continuous_loop_health.json` → `HEALTHY`
4. CLI: `scripts/show_triton_runtime_supervisor_status.ps1`

### 4.2 Detect healthy watchdog

Same stack for `run_watchdog_loop.ps1` + `watchdog_loop.lock` + `watchdog_status.json` (5-minute grace).

### 4.3 Detect duplicate processes

- Start scripts: process already running → skip; lock after orphan-clear still present → `[BLOCKED]` exit 1
- Python: multiple PIDs for same script name
- Recovery: `_duplicate_protection()` → `RECOVERY_BLOCKED` if process and/or live lock present

### 4.4 Verify lock against real process

- PowerShell `Test-LockConfirmedSafe`: lock **file exists** **and** `Test-LoopProcessRunning` for expected script
- Python classifier: lock `pid` ∈ detected PIDs + heartbeat age

**Never** treat orphan lock or lock mtime alone as healthy.

### 4.5 Stale vs healthy lock

| Condition | Classification / action |
|-----------|-------------------------|
| Lock, no process | `STALE_ORPHAN_LOCK` — clear only via start-path `Clear-OrphanLoopLock` or gated recovery |
| Process, heartbeat beyond grace | `STALE_ACTIVE_LOCK` |
| Process + fresh heartbeat | `LOCK_HEALTHY_ACTIVE` |
| Process + stale timestamp but alive | `LOCK_TIMESTAMP_STALE_BUT_PROCESS_ACTIVE` |

### 4.6 Start each loop

| Loop | Safe path |
|------|-----------|
| Watchdog | `start_triton_runtime_931.ps1` / `start_triton_runtime_supervisor.ps1` → `Start-Process` `run_watchdog_loop.ps1 -IntervalMinutes 1`; or direct script |
| Continuous | Same starters with `-IntervalMinutes 45`; or direct `run_continuous_loop.ps1`; recovery `_restart_loop` (continuous only) |

Both: orphan-clear → create lock → `loop_safety --refresh` → cycle.

### 4.7 Stop each loop

| Path | Exists? |
|------|---------|
| Natural exit / `finally` removes lock | Yes |
| `-MaxCycles N` finite run | Yes |
| Forced continuous via hung recovery (`taskkill` when `SAFE_TERMINATION_CANDIDATE`) | Yes (gated CLI) |
| Dedicated graceful `stop_*.ps1` | **No** |
| Watchdog forced stop/recovery | **No** |
| Dashboard stop | **No** |

### 4.8 Preserve paper-only configuration

931 + supervisor set:

- `TRITON_ENABLE_PAPER_EXECUTION=1`
- `AUTO_PAPER_ENTRIES=1`
- `AUTO_PAPER_APPROVAL=1`
- `TRITON_ENABLE_LIVE_EXECUTION=0`
- `TRITON_LIVE_EXECUTION=0`

Postconditions require paper flags OK and live disabled.

### 4.9 Guarantee live execution blocked

Layered (independent of loops):

1. Startup env live=0 + audit `live_execution_enabled=false`
2. `execution_readiness.json` → `live_execution_permitted`
3. `live_block_classifier.evaluate_live_block_status(paper_mode=True)`
4. Paper arm ≠ live arm; `live_confirm.json` separate
5. Governance/execution validators refuse when live not blocked

### 4.10 Record runtime startup results

- `data/results/runtime_autostart_931.json`
- `data/results/runtime_supervisor_autostart.json`
Fields include eligibility, start attempted/confirmed, PIDs, env, errors, `postconditions_passed`.

### 4.11 Partial startup failure

- Watchdog-first, then continuous (continuous may be skipped if startup errors)
- Per-loop confirm within ~25s (process + lock)
- Failed postconditions → audit written → exit 1
- **Gap:** no automatic rollback of a successfully started peer loop

---

## 5. Scheduled tasks (backup authority)

| Task | Register script | Action |
|------|-----------------|--------|
| `TRITON_Runtime_931AM` | `register_triton_runtime_931_task.ps1` | Daily 09:31 → `start_triton_runtime_931.ps1` |
| `TRITON_Runtime_Supervisor_45Min` | `install_triton_runtime_supervisor_45min.ps1` | At logon → supervisor start |
| `TRITON_MarketOpen` | `register_market_open_task.ps1` | One-shot market-open cycle (not persistent loops) |
| `TRITON_Data_Freshness_35Min` | `install_triton_data_freshness_35min.ps1` | Freshness refresh |

**Dashboard controller must retain Task Scheduler as backup** and must not deregister 9:31.

---

## 6. Paper arm vs live arm

| Concern | Implementation |
|---------|----------------|
| Status | `python -m services.paper_arm_status` / `scripts/show_triton_paper_arm_status.ps1` |
| Arm | `python -m services.set_paper_arm_confirm --ttl <minutes>` → `paper_arm_confirm.json` |
| Live | Must remain blocked; no operator “enable live” control in proposed nav |

---

## 7. Existing dashboard runtime controls

| Control | Present? |
|---------|----------|
| Start continuous/watchdog | **No** |
| Stop either loop | **No** |
| Restart with confirmation | **No** |
| Health monitors | Read-only + CLI hints |
| Paper arm display | Sidebar / Execution Dashboard |
| Live Trading ARM/DISARM/KILL | Live-arm artifacts — **not** loop process control |

---

## 8. Tests / validators (lifecycle)

No classic `tests/test_*runtime*` suite. Coverage via phase validators:

- `scripts/validate_monday_runtime_931_start_repair.ps1`
- `services/phase_196c_runtime_autostart_validation.py`
- `phase_196h` / `phase_196i` supervisor validators
- `phase_195o` / `phase_195p` heartbeat/lock hardening
- `phase_197q` process/env authority
- Related live-block / loop-safe-block validators

---

## 9. Gaps (do not weaken safety to close them)

1. No dashboard start with paper-env + weekday + process/lock postconditions
2. No dashboard stop with healthy-vs-hung distinction
3. No graceful stop path for either loop
4. No watchdog recovery stop/restart symmetry
5. Partial 931 failure does not roll back peer
6. Recovery restart may not re-apply full 931 env block set (relies on loop + system env)
7. Dual authorities (931 + Supervisor + MarketOpen) — keep single intended authority clear
8. Live ARM UI must not be confused with runtime Start/Stop

---

## 10. Implications for control design

Any future controller must **wrap** existing start scripts and classifiers — not invent new process discovery — and must **add** a graceful stop design before exposing Stop/Restart in the UI. See `DO_001_OPERATOR_NAVIGATION_PROPOSAL.md` and `DO_001_IMPLEMENTATION_PLAN.md`.
