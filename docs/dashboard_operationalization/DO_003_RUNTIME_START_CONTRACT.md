# DO-003 — Runtime Start Contract

## Allowed action

`START_TRITON` only.

## Denied actions

`STOP_TRITON`, `RESTART_TRITON`, `FORCE_STOP`, `KILL_PROCESS`, `REPAIR_LOCK`, `REMOVE_LOCK`, `PAPER_ARM`, `DISARM_PAPER`, `APPROVE`, `AUTHORIZE`, `SUBMIT_ORDER`, `CANCEL_ORDER`, `ENABLE_LIVE`, `LIVE_ENABLE`

## Structured results

`STARTED` · `PARTIAL_START` · `ALREADY_RUNNING` · `START_IN_PROGRESS` · `START_REFUSED_UNSAFE_STATE` · `START_REFUSED_LIVE_SAFETY` · `START_REFUSED_EXISTING_UNHEALTHY_PROCESS` · `START_FAILED` · `POSTCONDITION_FAILED` · `UNKNOWN`

## Success postconditions

- Exactly one continuous process with matching script command line
- Exactly one watchdog process with matching script command line
- Each lock exists; lock PID agrees with process
- No duplicates
- Live execution remains blocked

Subprocess creation alone is never `STARTED`.

## Audit

Append-only: `data/results/operator_runtime_actions.jsonl` (gitignored under `data/results/**`)

Fail-safe: if audit cannot be written before launch → refuse Start (`AUDIT_UNAVAILABLE`).

## Guard

`data/live/operator_start_action.guard.json` — dashboard Start coordination only; never proof TRITON is running; never a loop lock.
