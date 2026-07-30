# DO-006 — Restart Transaction Model

## Stages

`RESTART_PREFLIGHT_CREATED` → `RESTART_CONFIRMED` → `RESTART_SUBMITTED` →
`STOP_PREFLIGHT_REVALIDATED` → `STOP_REQUEST_DELIVERED` → `STOP_COMPLETED` →
`START_FRESH_CLASSIFICATION` → `START_ELIGIBILITY_CONFIRMED` → `START_TRANSACTION_CREATED` →
`RESTART_COMPLETED` | `RESTART_PARTIAL_COMPLETION` | `RESTART_TIMED_OUT` | `RESTART_REFUSED` | `RESTART_AUDIT_FAILED`

## Authority separation

- Restart preflight token authorizes only `RESTART_TRITON`
- Stop uses a newly issued DO-005 Stop preflight and `STOP TRITON` phrase supplied server-side
- Start uses a fresh DO-003 idempotency key — never the Restart token

## Persistence

- Guard: `data/live/operator_restart_action.guard.json`
- Transaction: `data/live/operator_restart_transaction.json`
- Tokens: `data/live/operator_restart_preflight_tokens.json`
- Idempotency: `data/live/operator_restart_idempotency.json`
- Audit: `data/results/operator_restart_actions.jsonl` (gitignored)
