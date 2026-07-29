# DO-005 — Failure and Recovery Policy

| Outcome | Operator action |
|---------|-----------------|
| `STOP_REFUSED_*` | Re-open confirmation; do not force-kill |
| `STOP_TIMED_OUT` | Diagnostics; wait / investigate; forced termination outside policy |
| `STOP_PARTIAL_COMPLETION` | Component-specific investigation; do not delete locks |
| `STOP_AUDIT_FAILED` (pre-delivery) | No request delivered; retry later |
| `STOP_AUDIT_FAILED` (post-delivery) | Preserve request/ack evidence; investigate |
| `STOP_REFUSED_STATE_CHANGED` | Fresh preflight — runtime identity changed |

Never delete runtime locks, never force-terminate, never auto-restart.
