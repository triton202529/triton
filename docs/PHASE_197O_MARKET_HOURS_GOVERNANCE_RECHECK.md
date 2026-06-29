# Phase 197O — Market-Hours Governance Recheck / No-Execution Observation Gate

## Problem

197N defined governance review criteria, but TRITON lacked a market-hours observation gate
to classify capped defensive exit state during trading hours without executing.

## Fix

`services/defensive_exit_market_hours_governance_recheck.py` — read-only observation that
combines governance criteria, live block classifier, paper entry gate, master execution gate,
and guard/open-orders snapshot freshness.

### Market-hours recheck states

| State | Meaning |
|-------|---------|
| `MARKET_HOURS_RECHECK_WAITING_FOR_MARKET_OPEN` | Market closed (safe observation) |
| `MARKET_HOURS_RECHECK_BLOCKED_BY_GOVERNANCE` | Gov false, readiness READY |
| `MARKET_HOURS_RECHECK_BLOCKED_BY_READINESS` | Gov true, readiness not READY |
| `MARKET_HOURS_RECHECK_BLOCKED_BY_GOVERNANCE_AND_READINESS` | Both block |
| `MARKET_HOURS_RECHECK_BLOCKED_BY_OPEN_ORDERS` | In-flight orders |
| `MARKET_HOURS_RECHECK_BLOCKED_BY_LIVE_BLOCK_REGRESSION` | Unsafe live unblock |
| `MARKET_HOURS_RECHECK_READY_FOR_OPERATOR_DECISION` | Human decision only — not execution |

## Guard freshness

Reports `guard_snapshot_age_minutes` and `open_orders_snapshot_fresh`. Stale guard with
fresh open-orders snapshot and zero open orders is diagnostic only in paper mode.

## Current typical state (after hours)

- `MARKET_HOURS_RECHECK_WAITING_FOR_MARKET_OPEN`
- `live_blocked=True`
- `eligible_for_execution=false`

## Validation

```powershell
python -m services.phase_197o_market_hours_governance_recheck_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197o_market_hours_governance_recheck_validation.ps1
python -m services.phase_197n_governance_review_criteria_validation
```

## Preserved

No approval artifact; no execution; no governance/readiness mutation; live blocked.
