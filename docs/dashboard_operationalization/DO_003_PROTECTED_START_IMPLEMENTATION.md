# DO-003 — Protected Start Implementation

**Phase:** DO-003
**Baseline:** `cd991ca406cfbb9fbbf776467b8b9a971bddb1cb` (DO-002)
**Status:** Implemented — START_TRITON only

---

## Architecture

| Module | Role |
|--------|------|
| `services/operator_runtime_classifier.py` | Per-loop state + Start decision matrix |
| `services/operator_runtime_controller.py` | Allowlisted START_TRITON controller, guard, audit |
| `dashboard/operator_console.py` | Operations Start UI + Diagnostics eligibility |

Canonical launch (matches 931/supervisor pattern):

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_watchdog_loop.ps1 -IntervalMinutes 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_continuous_loop.ps1 -IntervalMinutes 45
```

Paper env overlay (not paper-arm mutation):

- `TRITON_ENABLE_PAPER_EXECUTION=1`
- `TRITON_ENABLE_LIVE_EXECUTION=0`
- `TRITON_LIVE_EXECUTION=0`

---

## Safety properties

- Frontend cannot supply command, PID, path, env dict, or shell argument
- No `shell=True` with user input
- No process termination / lock deletion from controller
- Stale lock without process → refuse (does not clear lock)
- Healthy loops → `ALREADY_RUNNING` (idempotent)
- Audit fail-safe: refuse launch if audit cannot be written
- 9:31 Scheduled Task unchanged (backup remains)

---

## Operator workflow

1. Open Operator Console → Operations
2. Review eligibility and live-blocked banner
3. Click **Start TRITON** → confirmation copy
4. Confirm Start → backend evaluates matrix, launches only missing loops, validates postconditions
5. Read structured result; expand technical evidence if needed

Stop/Restart remain unavailable placeholders.

---

## Recommended next phase

**DO-004 — Cooperative Shutdown Protocol** before enabling Stop/Restart.
