# DO-007 — Validation Report

## Status

**PASS** (advisory/read-only validation only; no real process or lock mutation)

## Baseline

`ec7c4546e50455936e92100012de142a11c027bd` — TRITON: add protected restart control

## Authoritative suite results

| Suite | Result | Pytest runtime | Outer PowerShell |
|-------|--------|----------------|------------------|
| DO-007 | 116 passed, 1 warning | 1.05s | 1.77377s |
| DO-006 | 134 passed, 1 warning | 6.19s | 6.846497s |
| DO-005 | 102 passed, 1 warning | 3.16s | 4.382412s |
| DO-004 | 73 passed, 1 warning | 4.07s | 4.628745s |
| DO-004 loop | 12 passed, 1 warning | 0.16s | 0.850397s |
| DO-003 | 31 passed, 1 warning | 26.33s | 27.617068s |
| DO-002 | 19 passed, 1 warning | 0.73s | 1.423544s |
| Combined | 487 passed, 1 warning | 38.19s | 38.806106s |

Production note: `controller_has_termination_paths()` was removed from the guidance service; source-absence checks live only in tests.

## Warning disposition

`PytestCacheWarning` from pytest cache ACL — environment only; non-blocking; not suppressed.

## Safety

- Advisory only; no Start/Stop/Restart execution
- Fresh read-only eligibility from DO-003/005/006
- Explicit Start timeout ≠ ordinary START_IN_PROGRESS
- No force termination / lock delete/repair / Scheduled Task
- Live remains blocked; paper unchanged
- No real runtime artifacts touched

## Recommended next phase

**DO-008 — Operator Runtime History and Read-Only Incident Timeline**
