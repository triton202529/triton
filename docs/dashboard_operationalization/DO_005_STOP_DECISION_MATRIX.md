# DO-005 — Stop Decision Matrix

Maps DO-004 shutdown decisions to Stop-control results:

| Condition | Result | Stop enabled |
|-----------|--------|--------------|
| Both absent | `STOP_ALREADY_STOPPED` | No |
| Continuous only verified | `STOP_ELIGIBLE_CONTINUOUS_ONLY` | Yes |
| Watchdog only verified | `STOP_ELIGIBLE_WATCHDOG_ONLY` | Yes |
| Both verified | `STOP_ELIGIBLE_BOTH` | Yes |
| Verified stale/overdue | `STOP_ELIGIBLE_UNHEALTHY_VERIFIED` | Yes |
| Unsafe classifier states | `STOP_REFUSED_UNSAFE_STATE` | No |
| Start active | `STOP_REFUSED_START_IN_PROGRESS` | No |
| Shutdown guard active | `STOP_IN_PROGRESS` | No |
| Unresolved prior shutdown | `STOP_REFUSED_SHUTDOWN_UNRESOLVED` | No |
| Live safety violation | `STOP_REFUSED_LIVE_SAFETY` | No |
| Bad/expired confirmation | `STOP_REFUSED_CONFIRMATION` / `_EXPIRED` / `_REPLAY` | — |
| State changed after confirm | `STOP_REFUSED_STATE_CHANGED` | — |
| Full DO-004 completion | `STOP_COMPLETED` | — |
| Partial | `STOP_PARTIAL_COMPLETION` | — |
| Timeout | `STOP_TIMED_OUT` | — |
| Audit failure | `STOP_AUDIT_FAILED` | — |
