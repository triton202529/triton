# Phase 197Q — Runtime Re-Entry Observation / Supervisor Integration Gate

## Problem

After the defensive exit governance hardening sprint (197G–197P), TRITON had no integrated
observation gate to confirm runtime supervisor components agree on safe state before restarting
the continuous loop.

## Fix

`services/runtime_reentry_observation_gate.py` — read-only aggregation of:

- Continuous/watchdog process and lock status
- Order guard authority (197P)
- Live block classifier
- Market-hours governance recheck (197O)
- Governance review criteria (197N)
- Paper entry gate safety
- Defensive exit capped preview and operator packet
- Real approval artifact absence

### Runtime re-entry states

| State | Meaning |
|-------|---------|
| `RUNTIME_REENTRY_WAITING_FOR_MARKET_OPEN` | Market closed; safe for observation-only posture |
| `RUNTIME_REENTRY_READY_FOR_OBSERVATION` | Market open; governance/readiness may block but loop may observe |
| `RUNTIME_REENTRY_BLOCKED_BY_LIVE_BLOCK_REGRESSION` | `live_blocked=false` — unsafe |
| `RUNTIME_REENTRY_BLOCKED_BY_OPEN_ORDERS` | In-flight orders |
| `RUNTIME_REENTRY_BLOCKED_BY_REAL_APPROVAL_ARTIFACT` | Unexpected real approval on disk |
| `RUNTIME_REENTRY_BLOCKED_BY_ORDER_GUARD` | Order guard not broker-safe |
| `RUNTIME_REENTRY_BLOCKED_BY_GOVERNANCE_RECHECK_FAILURE` | 197O/197N live-block regression |
| `RUNTIME_REENTRY_BLOCKED_BY_PAPER_ENTRY_UNSAFE` | Paper entry gate could permit execution |

### Restart recommendation

| Recommendation | Meaning |
|----------------|---------|
| `RESTART_CONTINUOUS_FOR_OBSERVATION_ONLY` | Safe to restart continuous loop via supervisor (observation only) |
| `DO_NOT_RESTART_CONTINUOUS` | Blocked — do not restart |

Restart command is **text only** (never executed by TRITON):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_triton_runtime_supervisor.ps1
```

## Current typical state (after hours)

- `RUNTIME_REENTRY_WAITING_FOR_MARKET_OPEN`
- `live_blocked=True`
- `order_guard_authority_state=ORDER_GUARD_FRESH`
- `watchdog_loop_running=True` (or explicitly reported)
- `continuous_loop_running=False` (acceptable)
- `restart_recommendation=RESTART_CONTINUOUS_FOR_OBSERVATION_ONLY`
- `eligible_for_execution=false`

## Validation

```powershell
python -m services.phase_197q_runtime_reentry_observation_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197q_runtime_reentry_observation_validation.ps1
python -m services.phase_197p_order_guard_snapshot_authority_validation
python -m services.phase_197o_market_hours_governance_recheck_validation
```

## Preserved

No approval artifact; no execution; no loop start/stop; no governance mutation; live blocked.
