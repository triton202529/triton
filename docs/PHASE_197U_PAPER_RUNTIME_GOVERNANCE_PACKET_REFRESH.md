# Phase 197U — Paper Runtime Governance Packet Refresh

## Problem

After 197S-FIX3 and 197T soak validation, TRITON runtime is stable and safe-blocking correctly.
Market-open execution remained blocked with:

- `GOV_REVIEW_BLOCKED_BY_STALE_PACKET`
- `execution_readiness=PARTIALLY_READY` (governance check only)
- `MARKET_HOURS_RECHECK_BLOCKED_BY_GOVERNANCE_AND_READINESS`

The operator approval packet hash no longer matched the current capped batch preview.

## Fix

Phase 197U refreshes `data/results/defensive_exit_operator_approval_packet.json` from current
runtime-safe evidence using existing sync/packet builders. It does **not** create:

`data/governance/operator_paper_defensive_exit_approval.json`

Governance criteria (197N) and market-hours recheck (197O) now distinguish:

| State | Meaning |
|-------|---------|
| `GOV_REVIEW_BLOCKED_BY_STALE_PACKET` | Packet hash/expiry mismatch — run 197U refresh |
| `GOV_REVIEW_CURRENT_OPERATOR_APPROVAL_REQUIRED` | Packet current; operator approval still required |
| `GOV_REVIEW_CURRENT_BUT_READINESS_BLOCKED` | Packet current; readiness not READY |
| `GOV_REVIEW_READY_FOR_OPERATOR_REVIEW` | Packet current; evidence complete for human review |

Market-hours recheck adds:

- `MARKET_HOURS_RECHECK_BLOCKED_BY_STALE_PACKET`
- `MARKET_HOURS_RECHECK_BLOCKED_BY_OPERATOR_APPROVAL`

`paper_entry_loop_stage` dry-run now reports `SAFE_BLOCK operator_approval_required` when
governance failure is due to missing defensive-exit operator approval with a current packet.

## Usage

```powershell
python -m services.phase_197u_paper_runtime_governance_packet_refresh
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197u_paper_runtime_governance_packet_refresh.ps1
python -m services.phase_197n_governance_review_criteria_validation
python -m services.phase_197o_market_hours_governance_recheck_validation
```

## Expected Post-Refresh

- 197U: `PASS`, `GOV_PACKET_REFRESHED_OPERATOR_REVIEW_REQUIRED` or `GOV_PACKET_CURRENT_OPERATOR_REVIEW_REQUIRED`
- 197N: `GOV_REVIEW_CURRENT_OPERATOR_APPROVAL_REQUIRED` (not stale packet)
- 197O: `MARKET_HOURS_RECHECK_BLOCKED_BY_OPERATOR_APPROVAL` when market open
- Paper entry dry-run: blocked on `operator_approval_required`, not stale packet
- No broker orders; live blocked; no real approval artifact

## Preserved

No approval artifact; no broker orders; no live_confirm refresh; no loop start/stop.
