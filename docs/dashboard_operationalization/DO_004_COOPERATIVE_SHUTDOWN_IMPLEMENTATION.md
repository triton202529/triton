# DO-004 — Cooperative Shutdown Implementation

## Summary

Phase DO-004 installs a **request-and-acknowledgement cooperative shutdown protocol** for TRITON’s continuous and watchdog loops. The backend can prepare a verified shutdown request; loops exit only at documented safe boundaries through normal cleanup. **No clickable Stop or Restart** is exposed.

## Components

| Piece | Path |
|-------|------|
| Request/ack helpers + interruptible wait | `services/operator_cooperative_shutdown.py` |
| Prepare controller + decision matrix | `services/operator_shutdown_controller.py` |
| Continuous loop hooks | `scripts/run_continuous_loop.ps1` |
| Watchdog loop hooks | `scripts/run_watchdog_loop.ps1` |
| Start refuses during shutdown | `services/operator_runtime_controller.py` |
| Ops/Diagnostics read-only UI | `dashboard/operator_console.py` |
| Tests | `tests/test_operator_cooperative_shutdown_do004.py` |

## Fixed coordination paths

- `data/live/operator_shutdown_request.json`
- `data/live/operator_shutdown_ack_continuous.json`
- `data/live/operator_shutdown_ack_watchdog.json`
- `data/live/operator_shutdown_action.guard.json`
- Audit: `data/results/operator_shutdown_actions.jsonl` (gitignored via `data/results/**`)

## Allowed backend action

`PREPARE_COOPERATIVE_SHUTDOWN` only.

## Explicitly absent

Force stop, taskkill, Stop-Process, SIGKILL, controller lock deletion, paper arm mutation, live enablement, Stop/Restart buttons, browser-supplied PIDs/paths/commands/env.

## Baseline

- DO-002: `cd991ca406cfbb9fbbf776467b8b9a971bddb1cb`
- DO-003: `a56fe8285caf7a91823b426a9bf469dfb2f28800`

## Next phase

**DO-005 — Protected Stop TRITON Control** (confirmation UI over this protocol only). Do not begin until authorized.
