# Phase 197N — Governance Review Resolution Criteria

## Problem

TRITON blocked capped defensive exits on `MANAGEMENT_REVIEW_REQUIRED` but lacked formal,
auditable criteria defining what governance must inspect before any future paper-only
authorization can be considered.

## Fix

`services/defensive_exit_governance_review_criteria.py` defines:

### Required evidence (read-only collection)

- Live blocked confirmed; no valid live confirmation active
- Paper mode confirmed; broker snapshot OK; open orders = 0
- Real approval artifact absent; capped preview exists with symbols/deferred/cap
- Readiness and market-hours status recorded
- Operator packet current; preview hash recorded
- Management review reason and risk reduction intent recorded
- No broker orders; no cap increase; no live execution requested

### Governance decision states

| State | Meaning |
|-------|---------|
| `GOV_REVIEW_REQUIRED` | Management review still required |
| `GOV_REVIEW_READY_FOR_OPERATOR_DECISION` | Evidence complete; governance cleared |
| `GOV_REVIEW_BLOCKED_BY_LIVE_BLOCK_REGRESSION` | `live_blocked=false` (unsafe) |
| `GOV_REVIEW_BLOCKED_BY_OPEN_ORDERS` | In-flight orders present |
| `GOV_REVIEW_BLOCKED_BY_STALE_PACKET` | Operator packet expired or hash mismatch |
| `GOV_REVIEW_BLOCKED_BY_MISSING_PREVIEW` | No capped preview |
| `GOV_REVIEW_BLOCKED_BY_READINESS` | `execution_readiness != READY` |
| `GOV_REVIEW_BLOCKED_BY_MARKET_CLOSED` | Market closed (safe block) |

## Current typical state

- `GOV_REVIEW_REQUIRED` or `GOV_REVIEW_BLOCKED_BY_MARKET_CLOSED` (after hours)
- `eligible_for_execution=false` always in this phase
- Real approval artifact absent

## Validation

```powershell
python -m services.phase_197n_governance_review_criteria_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197n_governance_review_criteria_validation.ps1
```

## Preserved

No approval artifact; no execution; no governance/readiness mutation; live blocked; all gates intact.
