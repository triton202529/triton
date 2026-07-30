# DO-006 — Protected Restart TRITON Implementation

## Goal

Confirmation-protected **Restart TRITON Safely** on Operations.

Orchestrates two separately verified transactions:

1. DO-005 protected `STOP_TRITON` → DO-004 `PREPARE_COOPERATIVE_SHUTDOWN` (when Stop is required)
2. Fresh DO-003 `START_TRITON` only after independently verified full Stop completion

## Mapping

`RESTART_TRITON` → (optional) DO-005 Stop → fresh DO-003 Start

Restart never:

- force-terminates
- deletes/repairs locks
- reuses the Restart preflight token as Start authority
- accepts frontend PID/path/command/env/timeout/target/mode/force inputs
- starts before Stop postconditions complete

## Modules

- `services/operator_restart_action.py` — eligibility, preflight, orchestration, audit
- `dashboard/operator_console.py` — Operations Restart control + Diagnostics evidence

## Baseline

`aecd6f1c12a79b34d85cdd7051b590295ffada16` — TRITON: add protected cooperative stop control

## Recommended next phase

**DO-007 — Operator Runtime Recovery and Incident Guidance** (read-only/advisory unless separately authorized)
