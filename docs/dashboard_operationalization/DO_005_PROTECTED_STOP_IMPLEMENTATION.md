# DO-005 — Protected Stop Implementation

## Summary

Exposes confirmation-protected **Stop TRITON Safely** on Operations. Maps exclusively to DO-004 `PREPARE_COOPERATIVE_SHUTDOWN`. Restart remains unavailable.

## Components

| Piece | Path |
|-------|------|
| Stop action adapter | `services/operator_stop_action.py` |
| Operations / Diagnostics UI | `dashboard/operator_console.py` |
| Tests | `tests/test_operator_protected_stop_do005.py` |

## Allowed dashboard action

`STOP_TRITON` → backend `PREPARE_COOPERATIVE_SHUTDOWN`

Fixed dashboard target policy: `BOTH` (backend narrows via DO-004 matrix).

## Confirmation

- Checkbox required
- Exact phrase `STOP TRITON` (case- and whitespace-sensitive)
- Short-lived single-use preflight token bound to verified identities
- Backend revalidates before delivery

## Explicitly absent

Force stop, taskkill, Restart, frontend PID/path/command/env/timeout/target inputs, lock delete/repair, automatic Start after Stop.

## Baseline

`79509b43607c5ae214b717271266e8f43449c8f4` — DO-004 cooperative shutdown

## Next phase

**DO-006 — Protected Restart TRITON Orchestration** (Stop then fresh Start as separate verified transactions). Do not begin until DO-005 is committed.
