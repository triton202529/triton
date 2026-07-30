# DO-006 — Validation Report

## Status

**PASS** (mock/fixture validation only; no real process or lock mutation)

## Baseline

`aecd6f1c12a79b34d85cdd7051b590295ffada16` — TRITON: add protected cooperative stop control

## Authoritative suite results (single invocation each)

| Suite | Result | Pytest runtime | Outer PowerShell |
|-------|--------|----------------|------------------|
| `tests/test_operator_protected_restart_do006.py` | 134 passed, 1 warning | 5.65s | 6.200007s |
| `tests/test_operator_protected_stop_do005.py` | 102 passed, 1 warning | 2.87s | 3.992867s |
| `tests/test_operator_cooperative_shutdown_do004.py` | 73 passed, 1 warning | 3.87s | 4.642242s |
| `tests/test_loop_cooperative_shutdown_regression_do004.py` | 12 passed, 1 warning | 0.16s | 0.771041s |
| `tests/test_operator_runtime_controller_do003.py` | 31 passed, 1 warning | 26.23s | 26.883128s |
| `tests/test_operator_console_do002.py` | 19 passed, 1 warning | 0.72s | 1.299228s |
| Combined (all six files) | 371 passed, 1 warning | 36.64s | 37.219765s |
| Additional: DO-006 + DO-002 | 153 passed, 1 warning | 5.52s | 6.089016s |

## Warning disposition

`PytestCacheWarning` from pytest cache ACL — environment only; non-blocking; not suppressed.

## Runtime discrepancy note

Pytest-reported `in Xs` and outer PowerShell stopwatch measure different scopes (pytest session vs process+import wall clock). Do not mix values across invocations or across measurement methods. Prior 27.75s vs 26.48s / 38.52s vs 37.88s mismatches were mixed outer vs pytest timings from different runs.

## Safety

- Exact phrase `RESTART TRITON` + checkbox
- Two-transaction Stop-then-fresh-Start
- Restart token never reused for Start
- Explicit Start timeout evidence → `RESTART_TIMED_OUT`; ordinary `START_IN_PROGRESS` → `RESTART_PARTIAL_COMPLETION`; no auto-retry
- Operations UI shows Stop/Start stages separately and does not claim immediate restart
- No forced termination / lock delete/repair / Scheduled Task
- Live remains blocked; paper unchanged
- No real runtime artifacts touched

## Recommended next phase

**DO-007 — Operator Runtime Recovery and Incident Guidance**
