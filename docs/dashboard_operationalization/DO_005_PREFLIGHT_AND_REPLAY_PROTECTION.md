# DO-005 — Preflight and Replay Protection

## Preflight token

Server-generated (`secrets.token_urlsafe`), stored as SHA-256 hash under:

`data/live/operator_stop_preflight_tokens.json` (runtime; gitignored via `data/live/`)

Bound to: action `STOP_TRITON`, eligibility decision, target components, PID/lock/command fingerprints, live-block / guard states, issued/expiry times, nonce.

TTL default: 120 seconds. Single use. Invalid after eligibility/identity change, expiry, or prior use.

## Idempotency

`request_nonce` hashed into `data/live/operator_stop_idempotency.json`. Duplicate nonce returns prior result without a second DO-004 delivery.

## Diagnostics

May show `token_presented` boolean only — never the raw token.
