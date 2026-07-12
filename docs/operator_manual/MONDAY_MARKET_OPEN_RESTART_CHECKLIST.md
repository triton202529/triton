# Monday Market-Open Restart Checklist

Operator checklist for the first market session after weekend / market-closed work.
Use this **before** any paper execution authority is considered.

Related:

- [Operator manual index](./README.md)
- [Weekend Safe Work Map](./WEEKEND_SAFE_WORK_MAP.md)
- [Partial GLD Defensive Path](./PARTIAL_GLD_DEFENSIVE_PATH.md)

---

## Warning

> **Market-open restart does not itself authorize paper execution.**

Completing this checklist verifies posture and evidence. It does **not** create approval
artifacts, broker authorization, or broker orders. Live execution must remain blocked.

---

## 1. Pre-open safety posture

Complete these before runtime broker or readiness commands that assume a clean session start.

| # | Check | Expected |
|---|--------|----------|
| 1.1 | Confirm market session timing (ET open / your intended window) | Operator knows whether market is open, pre-open, or closed |
| 1.2 | Confirm live execution remains blocked | `live_execution_permitted=false` / no live authorize path |
| 1.3 | Confirm no short-lived approval artifact exists | Path absent unless intentionally created later |
| 1.4 | Confirm no broker authorization artifact exists | Path absent unless intentionally created later |
| 1.5 | Confirm git status is clean before runtime checks | No unexpected source or authority-file dirty state |

```powershell
cd C:\Users\akimw\triton
git status --short

Test-Path data/governance/operator_paper_defensive_exit_approval.json
Test-Path data/governance/operator_paper_broker_execution_authorization.json
```

Both `Test-Path` results should be **False** at restart (no standing weekend authority).

**Stop** if live execution is not blocked, if either artifact exists unexpectedly, or if git is
dirty from unexpected source changes (see §5).

---

## 2. Broker state checks

Paper-mode broker observation only. These commands refresh visibility; they are **not**
paper-entry approval and **not** defensive-exit dispatch.

```powershell
python -m services.snapshot_live_orders --mode paper
python -m services.poll_order_status --mode paper --refresh
python -m services.manage_open_orders --mode paper --smart-manage --execute-smart --verbose
```

Operator expectations:

- Snapshot completes without implying live mode.
- Open-order count is understood; **stop if open orders exist** before considering new paper
  execution authority (see §5).
- Do not proceed to approval / broker-auth creation from this section alone.

---

## 3. Readiness checks

Refresh performance / allocation readiness surfaces (read-only relative to broker dispatch):

```powershell
python -m services.build_trade_outcomes
python -m services.build_pnl_diagnostics
python -m services.build_performance_intelligence
python -m services.performance_risk_overlay
python -m services.portfolio_allocation_engine
```

Review outputs for anomalies. Readiness refresh is **not** permission to create 197W / 197Z
artifacts or to run `manage_positions` dispatch.

Optional related (if used in your Monday runbook):

```powershell
python -m services.execution_readiness
```

---

## 4. Partial GLD review checks

Certified partial GLD path (Phases 198C–198K) — review only:

| Fact | Value |
|------|-------|
| Raw handoff may show | **195** |
| Authoritative governed partial dispatch qty | **48** |
| Remaining after partial (deferred) | **147** |
| Full GLD exit | **Deferred** |
| Execution allowed now (198K) | **false** |

Rules:

- Do **not** dispatch from raw `data/live/manage_orders.csv` alone.
- Always use **197X / 198J / 198K** governed quantity evidence for dispatch scope.
- Raw handoff quantity may represent the full defensive candidate; broker dispatch must use
  the governed projected dispatch quantity only (**48**).

Read-only display:

```powershell
python scripts/show_partial_gld_no_execution_certification.py
```

---

## 5. Required stop points

**Stop before** any of the following unless a separate, explicit paper-execution phase is in
progress with matching operator intent:

| Stop point | Why |
|------------|-----|
| Creating **197W** approval artifact | Creates paper execution authority |
| Creating **197Z** broker authorization | Creates broker execution authority |
| **`manage_positions` dispatch** | Can hand off / place along defensive path |
| **Open orders exist** | Do not add new paper authority on top of unsettled broker state |
| **Git dirty from unexpected source changes** | Unknown code/config drift before session |
| **Live execution is not blocked** | Live must remain blocked |

This checklist ends at observation and review. It does **not** continue into artifact creation
or dispatch.

---

## 6. Paper execution preconditions

Any **future** paper execution (separate explicit phase — not this checklist) requires all of:

1. Fresh broker snapshot
2. Open orders = **0**
3. Current readiness review
4. Operator approval artifact (197W path — short-lived, scope-matched)
5. Broker authorization artifact (197Z path — short-lived, scope-matched)
6. Exact paper-only authorization phrase (as required by the creating phase, e.g. 197W)
7. Final dry-run showing **GLD qty 48 only** (not raw 195)

Until every precondition is met under an intentional paper-execution plan:

- Do not create approval or broker-authorization artifacts from this restart checklist.
- Do not run manage_positions for dispatch.
- Do not place paper or live orders.

---

## 7. Sign-off (restart complete — still no execution authority)

| Item | Status |
|------|--------|
| Pre-open posture checked | ☐ |
| Broker state checked | ☐ |
| Readiness checked | ☐ |
| Partial GLD governed qty reviewed (48) | ☐ |
| Stop points respected (no 197W / 197Z / manage dispatch) | ☐ |
| Live still blocked | ☐ |
| Approval artifact absent (unless later intentional) | ☐ |
| Broker auth artifact absent (unless later intentional) | ☐ |

Remember: **Market-open restart does not itself authorize paper execution.**
