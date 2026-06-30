# Phase 197R — Post-Restart Runtime Soak / Dispatch Approval Interlock Audit

## Problem

After observation-only continuous-loop restart, the first cycle ran `manage_positions` and
`place_live_orders` with `dry_run=False` after hours. Twelve exit rows were mapped and the
batch was blocked only by notional cap (`HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED`).
No broker orders were submitted, but the path reached executable placement mapping without
operator approval — too close to execution.

## Fix

### `services/defensive_exit_dispatch_interlock.py`

Central dispatch interlock evaluates before broker-order placement:

| State | Meaning |
|-------|---------|
| `DISPATCH_BLOCKED_APPROVAL_REQUIRED` | Real approval artifact absent or invalid |
| `DISPATCH_BLOCKED_GOVERNANCE` | Governance not authorized |
| `DISPATCH_BLOCKED_READINESS` | Readiness not READY |
| `DISPATCH_BLOCKED_MARKET_CLOSED` | Market closed |
| `DISPATCH_BLOCKED_BATCH_NOTIONAL_CAP` | Batch exceeds configured cap |
| `DISPATCH_BLOCKED_ORDER_GUARD` | Order guard not broker-safe |
| `DISPATCH_BLOCKED_LIVE_BLOCK` | Live block regression |
| `DISPATCH_BLOCKED_OPEN_ORDERS` | In-flight orders present |
| `DISPATCH_READY` | All gates satisfied (still does not auto-approve) |

Priority: live block → open orders → order guard → market closed → **approval required** →
governance → readiness → notional cap.

### Integration points

1. **`manage_positions.maybe_execute_management`** — blocks before `place_live_orders`
   subprocess when paper defensive exit handoff and interlock not permitted.
2. **`place_live_orders`** — defense-in-depth block on `manage_handoff` before placement mapping.

Batch notional cap remains; it is no longer the only guard before placement.

## Audit

`services/phase_197r_post_restart_dispatch_interlock_audit.py` parses:

- Latest `data/results/continuous_loop/continuous_loop_*.log`
- Latest `data/results/scheduled_runs/market_open_*.log`

Confirms no execution, explicit dispatch interlock, and real approval artifact absent.

## Validation

```powershell
python -m services.phase_197r_post_restart_dispatch_interlock_audit
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197r_post_restart_dispatch_interlock_audit.ps1
python -m services.phase_197q_runtime_reentry_observation_validation
```

## Preserved

No approval artifact created; no broker orders; live blocked; observation-only posture.
