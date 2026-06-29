# Phase 197J — Operator Paper Approval Packet / Governance Review Decision

## Problem

After 197I, TRITON could classify capped defensive batches against governance and readiness,
but lacked a formal operator review packet and explicit paper-only approval contract defining
what artifact must exist before capped defensive paper execution can be considered.

## Fix

`services/defensive_exit_operator_paper_approval.py` provides:

1. **Read-only operator packet** — `data/results/defensive_exit_operator_approval_packet.json`
2. **Approval contract path** — `data/governance/operator_paper_defensive_exit_approval.json`
   (never auto-created; operator must intentionally author)
3. **Contract validator** with explicit pass/fail reasons
4. **Decision options**:
   - `APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH`
   - `REJECT_CAPPED_PAPER_DEFENSIVE_BATCH`
   - `DEFER_REVIEW`
   - `REQUEST_REVISED_BATCH`

## Approval status classification

| Status | Meaning |
|--------|---------|
| `OPERATOR_APPROVAL_REQUIRED` | No contract file exists |
| `OPERATOR_APPROVAL_INVALID` | Contract exists but fails validation or operator rejected |
| `OPERATOR_APPROVAL_VALID_PAPER_ONLY` | Contract valid; paper-only; does not execute in this phase |

## Extended governance sync statuses (197J)

| Status | Meaning |
|--------|---------|
| `CAPPED_DEFENSIVE_BATCH_APPROVED_FOR_PAPER` | Valid operator contract + all gates permit |
| `CAPPED_DEFENSIVE_BATCH_REJECTED_BY_OPERATOR` | Operator decision = REJECT |

## Required approval contract fields

- `approval_id`, `approved_at`, `expires_at`, `approved_by`
- `approval_mode` = `PAPER_ONLY`
- `approved_symbols` (subset of current capped preview)
- `max_notional` (must not exceed configured cap)
- `source_packet_id` (must match latest packet)
- `source_preview_hash` (must match latest capped preview hash)
- `acknowledgement_live_blocked` = true
- `acknowledgement_no_live_execution` = true
- `acknowledgement_paper_only` = true
- `decision` = `APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH`

## Validation rules

Contract validation checks: file exists, approve decision, paper-only mode,
acknowledgements, not expired, symbols subset, notional within cap, packet ID match,
preview hash match, no in-flight sell conflicts, broker snapshot OK, live blocked.

## Current runtime (typical)

- `approval_status` = `OPERATOR_APPROVAL_REQUIRED`
- `governance_sync_status` = `CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_GOVERNANCE`
- `readiness_status` = `PARTIALLY_READY`
- No approval artifact auto-created
- No broker orders submitted

## Validation

```powershell
python -m services.phase_197j_operator_paper_approval_packet_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197j_operator_paper_approval_packet_validation.ps1
python -m services.phase_197i_defensive_exit_governance_sync_validation
python -m services.paper_entry_loop_stage --verbose --dry-run
python -m services.snapshot_live_orders --mode paper
```

## Preserved

Paper-only; live blocked; no auto-approval; no cap increase; no forced execution; all gates intact.
