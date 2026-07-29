# DO-004 — Stop / Shutdown Decision Matrix

| Condition | Result | Behavior |
|-----------|--------|----------|
| Both NOT_RUNNING, no conflicting locks | `ALREADY_STOPPED` | No request written |
| Verified continuous only | `SHUTDOWN_ELIGIBLE_CONTINUOUS_ONLY` | Target continuous |
| Verified watchdog only | `SHUTDOWN_ELIGIBLE_WATCHDOG_ONLY` | Target watchdog |
| Both verified healthy | `SHUTDOWN_ELIGIBLE_BOTH` | Target both |
| Verified but STALE_OR_OVERDUE + lock ownership OK | `SHUTDOWN_ELIGIBLE_UNHEALTHY_VERIFIED` | Cooperative request allowed; no force kill |
| PID/COMMAND mismatch, no valid lock, stale lock no process, duplicates, ambiguous, unknown | `SHUTDOWN_REFUSED_UNSAFE_STATE` | No request |
| Start guard active | `SHUTDOWN_REFUSED_START_IN_PROGRESS` | No request |
| Shutdown guard active | `SHUTDOWN_IN_PROGRESS` | No new request |
| Live safety violation | `SHUTDOWN_REFUSED_LIVE_SAFETY` | No request; escalate |

Eligible process states: `HEALTHY_RUNNING`, `STALE_OR_OVERDUE` (with matching process + lock).

## Completion results

`COOPERATIVE_SHUTDOWN_COMPLETED`, `PARTIAL_SHUTDOWN`, `ALREADY_STOPPED`, `SHUTDOWN_IN_PROGRESS`, `SHUTDOWN_REFUSED_*`, `SHUTDOWN_TIMED_OUT`, `POSTCONDITION_FAILED`, `AUDIT_FAILED`, `FAILED`, `UNKNOWN`.
