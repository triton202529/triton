# DO-001 — Implementation Plan & Control-Backend Design

**Phase:** DO-001 (design only — do not implement controls yet)

---

## 1. Recommended control architecture

**Local allowlisted Python runtime controller**, invoked only from Streamlit on localhost, wrapping existing PowerShell start scripts and process/lock classifiers.

```
[Streamlit Operator UI — localhost only]
        │
        │  action ∈ {START, STOP, RESTART}
        │  no user PIDs, paths, or shell strings
        ▼
[services/runtime_operator_controller.py]  ← NEW (future phase)
        │  validate local peer
        │  cooldown / idempotency token
        │  paper-mode enforce / live deny
        │  duplicate + lock classify
        ▼
[scripts/start_triton_runtime_*.ps1]  OR  [new graceful stop helpers]
        │
        ▼
[append-only data/results/operator_runtime_actions.jsonl]
        │
        ▼
[status refresh via runtime_process_detector + runtime_lock_classifier]
```

**Why this shape (Streamlit + Windows):**

- Matches current dashboard framework (no new web server required for v1)
- Reuses hardened 931/supervisor start paths and classifiers
- Avoids exposing a generic command runner to the frontend
- Task Scheduler (`TRITON_Runtime_931AM`) remains backup autostart

**Rejected approaches:**

- Arbitrary `subprocess` of user-supplied commands
- Remote HTTP control plane without auth
- Killing by operator-typed PID
- Dashboard editing of lock files directly
- Enabling live execution from Start

---

## 2. Action allowlist

| Action | Meaning | Confirmation | Cooldown |
|--------|---------|--------------|----------|
| `START_TRITON` | Start watchdog then continuous via supervisor/931-equivalent wrapper | Soft (show env posture) | Yes (e.g. 60s) |
| `STOP_TRITON` | Graceful stop both loops (new helper required) | **Hard confirm** | Yes |
| `RESTART_TRITON` | Stop then Start with postconditions | **Hard confirm** | Yes |

No other actions. No `RUN_SCRIPT`, `KILL_PID`, `CLEAR_LOCK`, `ENABLE_LIVE`.

---

## 3. Safety requirements (must hold)

| Requirement | Design |
|-------------|--------|
| Local-only access | Bind Streamlit to `127.0.0.1`; reject controller calls if peer is not local (best-effort headers + bind) |
| Request validation | Action enum only; server-side session cooldown token |
| Paper-mode enforcement | Force/verify paper env flags identical to 931/supervisor |
| Live-execution denial | Refuse if live env would be enabled; assert `live_execution_permitted` remains false after action |
| Duplicate-process detection | Use `runtime_process_detector` before start |
| PID + cmdline verification | Never trust lock PID alone; match script name in cmdline |
| Verified stale-lock handling | Orphan clear only via existing `Clear-OrphanLoopLock` / gated recovery semantics |
| Confirmation | Stop/Restart require typed confirm or Streamlit confirmation dialog + second click |
| Partial-failure recovery | If watchdog starts and continuous fails: record failure, offer guided STOP of partial set, do not claim success |
| Structured results | `{ok, action, started_at, pids, lock_classes, errors[], postconditions}` |
| Append-only audit | `operator_runtime_actions.jsonl` — never rewrite |
| Button cooldown / idempotency | Disable buttons N seconds; ignore duplicate action ids |
| Status refresh | Clear relevant caches; re-run detector/classifier after action |
| Task Scheduler backup | Documented; controller does not unregister 9:31 |
| No arbitrary shell | Fixed script paths under `scripts/` only |
| No user PIDs/paths | Hard-coded repo root from controller config |
| No generic command runner | Frontend sends action enum only |

---

## 4. START path (future)

1. Validate localhost + cooldown
2. Classify current locks/processes — if already healthy both loops → return `ALREADY_RUNNING` (idempotent success)
3. Set/confirm paper env; deny if live would enable
4. Invoke existing `start_triton_runtime_supervisor.ps1` **or** shared internal function extracted from 931/supervisor (prefer one code path)
5. Wait for process+lock confirmation (same ~25s pattern)
6. Write startup-compatible audit + operator JSONL row
7. Refresh dashboard status

Weekday eligibility: mirror 931 rules or explicitly allow operator override flag **only for START** with audit reason (decide in DO-003; default = same as 931).

---

## 5. STOP path (future — blocker until designed)

**Gap today:** no graceful stop script.

Proposed safe stop (to be implemented before UI Stop):

1. Detect PIDs via cmdline match only
2. Signal cooperative stop (preferred): write `data/live/continuous_loop.stop_request` / `watchdog_loop.stop_request` that loop scripts honor between cycles
3. Wait up to timeout for process exit + lock removal
4. If still running and classifier says hung + `SAFE_TERMINATION_CANDIDATE`, only then allow forced terminate (continuous parity with hung recovery)
5. Never delete a lock while a matching process is alive
6. Audit every step

Until cooperative stop exists, dashboard Stop must remain disabled or dry-run only.

---

## 6. RESTART path (future)

`STOP_TRITON` (confirmed) → verify both stopped → `START_TRITON` → full postconditions. Any mid failure returns `PARTIAL` with recovery guidance.

---

## 7. Paper Arm (related, not Start)

- Display + renew on Operations protected area
- Continues to use `set_paper_arm_confirm`
- Must not enable live confirm
- Separate cooldown from Start/Stop

---

## 8. Phased delivery after DO-001

### DO-002 — Operator scaffold + read-only Activity (recommended next)

- Six-page nav scaffold (dual-mode or feature flag; **do not delete** old pages yet)
- Operations status panel: running/stopped/degraded/stale using existing detectors
- Activity feed v1 (read-only parsers; no Start/Stop)
- Paper-arm + live-blocked status cards
- Localhost bind documentation / config

### DO-003 — Runtime controller (Start first)

- Implement `runtime_operator_controller` allowlist
- Wire **START** only behind confirmation
- Append-only operator audit
- Cooldown + postcondition refresh
- Still no Stop until cooperative stop ships

### DO-004 — Graceful Stop / Restart

- Loop stop-request protocol
- STOP + RESTART UI
- Partial-failure recovery UX
- Tests for lock/process classifiers + controller

### DO-005 — Nav consolidation

- Hide research/sim from default sidebar
- Move advanced pages under Diagnostics
- Retire broken/unused with dependency checklist

### DO-006 — Simulation page retirement (optional)

- Remove or quarantine sim pages per matrix
- Keep MUST_RETAIN services

---

## 9. Implementation readiness

| Item | Status |
|------|--------|
| Page inventory complete | Yes |
| Runtime safe paths documented | Yes |
| Activity sources mapped | Yes |
| Nav proposal complete | Yes |
| Control design complete | Yes (not implemented) |
| Blocker for full Stop UI | Graceful stop protocol missing |
| Blocker for controller hardening | Streamlit local-only not configured |
| Ready for DO-002? | **Yes** |

---

## 10. Non-goals (all phases until explicitly authorized)

- Live execution enablement
- Weakening gates / lifecycle / allocation / risk / governance / broker auth
- Removing duplicate-process or stale-lock protections
- Changing 9:31 scheduled startup behavior without a dedicated phase
- Committing runtime live artifacts
