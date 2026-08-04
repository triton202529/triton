# DO-007 — Safety Boundaries

## Why guidance cannot act as authority

Guidance IDs are advisory correlation identifiers only. Protected controls require their own confirmation phrases, checkboxes, server-generated preflight tokens, and fresh eligibility.

## Why locks cannot be deleted or repaired

Runtime locks are owner-cleaned by loops at documented boundaries. Deleting or repairing locks from the dashboard would create false ownership and unsafe Start eligibility.

## Why force termination is prohibited

Forced termination bypasses cooperative shutdown and can leave inconsistent lock/process identity. DO-004/DO-005/DO-006 already forbid it; DO-007 must not recommend it.

## Why active or partial transactions prevent duplicate action

Mutual exclusion across Start, Stop, and Restart prevents racing control transactions and ambiguous identity.

## Why live execution remains blocked

Live enablement is outside operator recovery. Any evidence of enablement is a critical incident for investigation only.

## Incomplete evidence

When evidence is missing, malformed, or contradictory, DO-007 returns INSUFFICIENT_EVIDENCE and does not present a protected action as eligible.
