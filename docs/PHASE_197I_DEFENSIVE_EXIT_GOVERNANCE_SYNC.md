# Phase 197I — Defensive Exit Governance Approval Sync

## Problem

After 197H, capped defensive exit preview existed but was not connected to governance,
readiness, or operator approval layers. Runtime could not state whether a capped batch
was approved, blocked, or awaiting operator action.

## Fix

`services/defensive_exit_governance_sync.py` classifies capped defensive batches:

| Status | Meaning |
|--------|---------|
| `CAPPED_DEFENSIVE_BATCH_AVAILABLE` | Preview artifact present |
| `CAPPED_DEFENSIVE_BATCH_APPROVED` | All gates permit paper defensive execution |
| `CAPPED_DEFENSIVE_BATCH_AWAITING_OPERATOR_APPROVAL` | Human queue pending for symbols |
| `CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_READINESS` | `execution_readiness != READY` |
| `CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_GOVERNANCE` | `governance_authorized=false` |
| `CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_NOTIONAL_CAP` | No symbols fit under cap |
| `CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_IN_FLIGHT_ORDERS` | Duplicate sell orders open |

Read-only operator packet: `data/results/defensive_exit_operator_approval_packet.json`

## Current runtime (typical)

- Governance: `MANAGEMENT_REVIEW_REQUIRED` → `BLOCKED_BY_GOVERNANCE`
- Readiness: `PARTIALLY_READY` → paper entry remains blocked
- Partial batch: `PARTIAL_DEFENSIVE_BATCH_AUTHORIZED`
- No broker orders submitted by this phase

## Validation

```powershell
python -m services.phase_197i_defensive_exit_governance_sync_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197i_defensive_exit_governance_sync_validation.ps1
python -m services.paper_entry_loop_stage --verbose --dry-run
python -m services.snapshot_live_orders --mode paper
```

## Preserved

Paper-only; live blocked; no cap increase; no forced execution; all gates intact.
