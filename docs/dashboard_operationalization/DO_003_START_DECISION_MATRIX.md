# DO-003 — Start Decision Matrix

| Continuous | Watchdog | Decision |
|------------|----------|----------|
| HEALTHY_RUNNING | HEALTHY_RUNNING | ALREADY_RUNNING |
| NOT_RUNNING | NOT_RUNNING | START_ALLOWED (both) |
| HEALTHY_RUNNING | NOT_RUNNING | PARTIAL_START_ALLOWED (watchdog only) |
| NOT_RUNNING | HEALTHY_RUNNING | PARTIAL_START_ALLOWED (continuous only) |
| STALE_OR_OVERDUE (either, verified process) | * | START_REFUSED_EXISTING_UNHEALTHY_PROCESS |
| STALE_LOCK_NO_PROCESS / PROCESS_WITHOUT_VALID_LOCK / PID_MISMATCH / COMMAND_MISMATCH / DUPLICATE_PROCESSES / AMBIGUOUS / UNKNOWN | * | START_REFUSED_UNSAFE_STATE |
| * | live not blocked | START_REFUSED_LIVE_SAFETY |

## Loop states

`HEALTHY_RUNNING` · `NOT_RUNNING` · `STARTING` · `STALE_LOCK_NO_PROCESS` · `PROCESS_WITHOUT_VALID_LOCK` · `PID_MISMATCH` · `COMMAND_MISMATCH` · `DUPLICATE_PROCESSES` · `STALE_OR_OVERDUE` · `AMBIGUOUS` · `UNKNOWN`

Rules: lock alone ≠ running; wrong command line ≠ running; duplicates refuse; healthy never restarted; stale locks not auto-removed in DO-003.
