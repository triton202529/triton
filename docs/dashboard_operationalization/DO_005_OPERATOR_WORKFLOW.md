# DO-005 — Operator Workflow

1. Open Operations (localhost).
2. Review Start/Stop eligibility captions.
3. If Stop eligible, click **Stop TRITON Safely**.
4. Read consequences; check the confirmation box; type exactly `STOP TRITON`.
5. Click **Confirm Stop** (disabled until checkbox + exact phrase + valid token).
6. Wait for cooperative progress; do not force-kill processes.
7. Full stop requires process exit and owner lock release.
8. Restart remains unavailable.
9. Start may be re-evaluated only via fresh DO-003 rules after clean stop — never auto-started.

## Mutual exclusion

- Start disabled while Stop confirm/submit/unresolved shutdown is active.
- Stop disabled while Start is active.
