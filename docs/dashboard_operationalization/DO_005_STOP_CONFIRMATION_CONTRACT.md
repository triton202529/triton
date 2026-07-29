# DO-005 — Stop Confirmation Contract

## Required

1. Backend eligibility is a safe shutdown-eligible result.
2. Operator checks the confirmation checkbox.
3. Operator enters exactly: `STOP TRITON`
4. Confirmation bound to current preflight token / snapshot.
5. Backend re-runs full safety preflight before delivery.
6. Backend refuses if identity or safety state changed.

## Checkbox text

> I understand that TRITON will finish its current safe unit of work before exiting and that this action will not force-terminate any process.

## Refused

- lowercase / whitespace variants / partial phrases
- unchecked box
- missing / expired / replayed token
- query params, URL fragments, refresh, session restoration as confirmation
- frontend-supplied targets, PIDs, paths, commands, env, timeouts
