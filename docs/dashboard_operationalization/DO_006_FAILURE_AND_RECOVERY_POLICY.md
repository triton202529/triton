# DO-006 — Failure and Recovery Policy

## Stop incomplete / timeout / partial

- Return `RESTART_TIMED_OUT` or `RESTART_PARTIAL_COMPLETION`
- Do not Start
- Do not force-terminate
- Do not delete requests, acknowledgements, guards, or locks
- Preserve evidence for Diagnostics

## Start ineligible after Stop

- Return `RESTART_REFUSED_START_INELIGIBLE`
- Keep TRITON stopped
- Do not weaken DO-003 Start policy

## Audit failure

- Initial audit failure: refuse before Stop delivery
- Mid-transaction audit failure: preserve Stop evidence; do not Start
- Final audit failure: preserve Start/Stop evidence; return `RESTART_AUDIT_FAILED`

## Never

- Delete or repair runtime locks
- Force-terminate
- Auto-restart after unresolved Stop
- Clear unresolved guards silently
