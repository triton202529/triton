# DO-004 — Operator Workflow

## What operators see

1. **Operations** — Start TRITON remains the only clickable runtime mutation (when eligible).
2. **Stop / Restart** — disabled placeholders; not clickable.
3. **Cooperative Shutdown (read-only)** — eligibility, request state, acknowledgements, recommended action.
4. **Diagnostics** — read-only shutdown evidence and sanitized audit; no repair/kill/delete/Stop buttons.

Banner message:

> Cooperative shutdown protocol installed. Operator Stop remains unavailable until protected Stop control validation is complete.

## What operators must not do in DO-004

- Expect a Stop button to work
- Manually delete runtime locks as part of this protocol
- Force-kill loops to “help” the protocol
- Submit shutdown via URL/query/refresh (not wired)

## After a future DO-005 Stop

Operators will use confirmation-protected Stop that calls only `PREPARE_COOPERATIVE_SHUTDOWN`. Restart stays unavailable until Stop is validated in isolation.

## Start interaction

If cooperative shutdown is active or unresolved, Start returns `START_REFUSED_SHUTDOWN_IN_PROGRESS`.
