# DO-007 — Incident Guidance

## Incident catalog

| Incident | Guidance |
|----------|----------|
| Missing process | Identify target; do not invent PIDs; consider protected Start only if fully stopped + DO-003 eligible |
| Unexpected extra process | Treat as unverified/unsafe; never force-terminate |
| Mismatched PID | Manual investigation; block automated recovery advice |
| Changed creation time | Unverified identity; investigate |
| Command fingerprint mismatch | Unverified identity; investigate |
| Missing owner lock | Report PROCESS_WITHOUT_VALID_LOCK; do not create locks |
| Stale lock | Manual investigation; **never delete or repair** |
| Conflicting lock owner | Block recovery advice; preserve ownership safeguards |
| Malformed runtime evidence | INSUFFICIENT_EVIDENCE; do not guess |
| Unavailable diagnostics | INSUFFICIENT_EVIDENCE |
| Active Stop | WAIT; block Start/Stop/Restart recommendations |
| Active Start | WAIT; ordinary in-progress ≠ timeout |
| Active Restart | WAIT; show Stop vs Start stage separately |
| Stop timeout | No force termination; no lock deletion |
| Start timeout | Report honestly; no automatic retry |
| Restart timeout | No force termination; no lock repair |
| Partial replacement verification | Not HEALTHY; investigate |
| Missing paper authorization | Separate prerequisite; do not create/extend |
| Live execution enabled | **CRITICAL** incident; escalate; do not remediate flags |

## Critical rule

If any evidence indicates live execution may be enabled, classify as a critical safety incident and require immediate manual investigation. DO-007 still must not modify execution flags.
