# DO-007 — Runtime Recovery Guidance

## Summary

DO-007 adds a **read-only advisory** Operator Runtime Recovery and Incident Guidance layer on the Operations page.

It observes, classifies, explains, and recommends. It never Starts, Stops, Restarts, terminates, repairs, deletes, authorizes, approves, or modifies TRITON.

## Architecture

| Piece | Path |
|-------|------|
| Guidance engine | `services/operator_runtime_recovery_guidance.py` |
| Operations UI | `dashboard/operator_console.py` → `_render_runtime_recovery_guidance` |
| Tests | `tests/test_operator_runtime_recovery_guidance_do007.py` |

## Advisory-only contract

- `advisory_only=true`
- `execution_attempted=false`
- `guidance_id_is_authority=false`
- Fresh read-only evaluations of DO-003 / DO-005 / DO-006 eligibility
- No preflights, tokens, requests, acknowledgements, or control transactions

## Relationship to prior phases

| Phase | Role |
|-------|------|
| DO-003 | Protected Start remains the only Start authority |
| DO-004 | Cooperative shutdown protocol unchanged |
| DO-005 | Protected Stop remains the only Stop authority |
| DO-006 | Protected Restart remains the only Restart authority |
| DO-007 | Explains state and may *recommend considering* those controls |

## Required notice

> Recovery guidance is advisory only. It does not start, stop, restart, terminate, repair, authorize, or modify TRITON.
