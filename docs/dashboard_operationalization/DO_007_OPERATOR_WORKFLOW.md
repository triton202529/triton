# DO-007 — Operator Workflow

1. Open Operations.
2. Read overall status metrics and LIVE EXECUTION BLOCKED banner.
3. Read **Runtime Recovery Guidance** (classification, severity, evidence, recommended safe action).
4. If guidance says WAIT — observe only; do not click Start/Stop/Restart.
5. If guidance says MANUAL_INVESTIGATION / ESCALATE — investigate Diagnostics; do not force-terminate or delete locks.
6. If guidance says CONSIDER_PROTECTED_* — scroll to the existing protected control and complete its confirmation workflow explicitly.
7. Never treat a guidance ID as Start/Stop/Restart authority.
8. Never expect page load, refresh, query params, fragments, or session restore to perform recovery.
