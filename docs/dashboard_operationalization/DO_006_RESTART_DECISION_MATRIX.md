# DO-006 — Restart Decision Matrix

| Decision | Restart enabled | Stop required | Notes |
|----------|-----------------|---------------|-------|
| RESTART_ELIGIBLE_RUNNING | yes | yes | both verified |
| RESTART_ELIGIBLE_CONTINUOUS_ONLY | yes | yes | continuous only |
| RESTART_ELIGIBLE_WATCHDOG_ONLY | yes | yes | watchdog only |
| RESTART_ALREADY_STOPPED | yes | no | fresh DO-003 Start eval still required |
| RESTART_IN_PROGRESS | no | — | guard active |
| RESTART_REFUSED_START_CONFLICT | no | — | Start guard active |
| RESTART_REFUSED_STOP_CONFLICT | no | — | shutdown guard/active |
| RESTART_REFUSED_UNSAFE_STATE | no | — | mismatch/duplicate/ambiguous/unknown |
| RESTART_REFUSED_LIVE_SAFETY | no | — | live appears enabled |
| RESTART_REFUSED_STATE_CHANGED | — | — | identity/eligibility drift |
| RESTART_REFUSED_STOP_INCOMPLETE | — | — | cannot advance to Start |
| RESTART_REFUSED_START_INELIGIBLE | — | done | Stop ok; Start refused |
| RESTART_TIMED_OUT | — | timed out | no Start |
| RESTART_PARTIAL_COMPLETION | — | partial | no Start / incomplete Start |
| RESTART_COMPLETED | — | verified | Start verified |
| RESTART_AUDIT_FAILED | — | preserved | no destructive rollback |
