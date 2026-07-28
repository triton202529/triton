# DO-002 — Operator Console Implementation

**Phase:** DO-002
**Status:** Implemented (observation only — no runtime Start/Stop/Arm controls)
**DO-001 commit:** `54f7c6d8` — `TRITON: complete dashboard operationalization audit`

---

## Summary

TRITON now opens in a focused **Operator Console** with six pages, while the full research dashboard remains available as **Advanced / Legacy Dashboard**.

| Item | Result |
|------|--------|
| Framework | Streamlit (`view_results.py`) |
| Default mode | Operator Console |
| Default page | Operations |
| Operator pages | Operations, Activity, Portfolio, Orders, Risk & Alerts, Diagnostics |
| Legacy pages | Intact via `PAGE_REGISTRY` + `SECTIONS` |
| Simulation in operator nav | Hidden |
| Runtime Start/Stop/Restart | **Not exposed** |
| Live enable control | **Not exposed** |
| Localhost bind | `.streamlit/config.toml` + `scripts/run_operator_console.ps1` |

---

## Entry commands

Preferred (localhost-only):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\akimw\triton\scripts\run_operator_console.ps1
```

Equivalent:

```text
streamlit run view_results.py --server.address 127.0.0.1 --server.port 8501
```

Defaults also live in `.streamlit/config.toml` (`server.address = "127.0.0.1"`).

---

## Architecture

| Module | Role |
|--------|------|
| `services/operator_console_status.py` | Shared read-only status / portfolio / orders / risk summaries |
| `services/operator_activity_feed.py` | Activity event normalization from real artifacts |
| `dashboard/operator_console.py` | Six operator page renderers |
| `view_results.py` | Mode switch (operator default / legacy) + intact legacy registry |

Status classification reuses `runtime_process_detector` and `runtime_lock_classifier` on the real root. Fixture roots never scan live machine processes.

---

## Operations status

Overall: `RUNNING` | `STOPPED` | `DEGRADED` | `STALE` | `UNKNOWN`

**RUNNING requires process evidence for both loops plus healthy locks.** A lock file alone is never sufficient.

Paper arm display: `ARMED` | `EXPIRED` | `MISSING` | `INVALID` | `UNKNOWN`
Live execution: prominent **LIVE EXECUTION BLOCKED** (critical warning if evidence says otherwise — still no enable button).

Disabled placeholder: *Runtime controls are being validated and are not yet available.*

---

## Activity feed

See `DO_002_ACTIVITY_EVENT_CONTRACT.md`.

---

## Safety boundaries honored

- No process start/stop/signal
- No lock mutation
- No order submit/cancel/approve
- No paper arm/disarm
- No live-execution control
- No arbitrary shell / user PIDs
- No 9:31 task changes
- No deletion of legacy/simulation pages or services

---

## Recommended next phase (DO-003)

Implement allowlisted **START_TRITON** only (localhost, confirmation, paper-env enforce, append-only audit). Keep Stop disabled until a cooperative graceful-stop protocol exists.
