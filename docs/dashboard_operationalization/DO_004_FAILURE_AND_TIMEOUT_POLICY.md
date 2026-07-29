# DO-004 — Failure and Timeout Policy

## Time budgets (production defaults)

| Phase | Default |
|-------|---------|
| Acknowledgement | 30s |
| Process exit | 60s |
| Lock release verification | 30s |

Tests inject short intervals and fake clocks/sleep.

## Partial shutdown

If one targeted component exits and another does not:

- Return `PARTIAL_SHUTDOWN`
- Do not restart the exited component
- Do not force-stop the remaining component
- Do not delete either runtime lock
- Preserve request + audit evidence
- Recommend Diagnostics / manual investigation
- Keep Start blocked while shutdown remains unresolved

## Timeouts

`SHUTDOWN_TIMED_OUT` never triggers forced termination. Message states forced termination is outside policy.

## Postconditions

- Exit with remaining lock → `POSTCONDITION_FAILED` (controller does not delete locks)
- Ack without exit → not completed
- Lock missing while process active → not completed

## Audit fail-safe

1. If **initial** audit cannot persist → refuse request delivery (`AUDIT_FAILED`).
2. If **final** audit fails after delivery → report `AUDIT_FAILED`, preserve request/ack evidence, no destructive rollback.

## Cleanup

Coordination files are retained/superseded atomically for audit. Cleanup never touches continuous/watchdog runtime locks and never stops a process.
