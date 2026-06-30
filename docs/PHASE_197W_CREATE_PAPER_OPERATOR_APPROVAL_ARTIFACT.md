# Phase 197W — Paper-Only Operator Approval Artifact Creation

## Purpose

Create `data/governance/operator_paper_defensive_exit_approval.json` for the
current capped defensive batch reviewed in Phase 197V. **No broker execution.**

Operator authorization phrase required:

> I authorize Phase 197W to create the paper-only operator approval artifact
> for the current capped defensive batch. No broker orders yet.

## Usage

```powershell
python -m services.phase_197w_create_paper_operator_approval_artifact
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197w_create_paper_operator_approval_artifact.ps1
```

## Approval Artifact

- `approval_action`: `APPROVE_CAPPED_PAPER_DEFENSIVE_BATCH`
- `approval_scope`: `PAPER_ONLY`
- `live_authorized`: false
- `broker_execution_authorized`: false
- `raw_execution_authorized`: false
- TTL: 30 minutes (conservative)
- Tied to current `packet_id` and `source_preview_hash`
- Deferred symbols (GLD, GE) excluded from `approved_symbols`

## Final States

| State | Meaning |
|-------|---------|
| `PAPER_OPERATOR_APPROVAL_CREATED_EXECUTION_NOT_RUN` | Artifact created; execution separate |
| `PAPER_OPERATOR_APPROVAL_BLOCKED_EXISTING_ARTIFACT` | Valid artifact already exists |
| `PAPER_OPERATOR_APPROVAL_BLOCKED_SAFETY_EVIDENCE_FAILED` | 197V/197U/197T/197S preconditions failed |
| `PAPER_OPERATOR_APPROVAL_BLOCKED_PACKET_MISMATCH` | Packet/hash/symbol mismatch |

## Post-197W Governance States

- 197N: `GOV_REVIEW_CURRENT_PAPER_APPROVAL_PRESENT`
- 197O: `MARKET_HOURS_RECHECK_OPERATOR_APPROVAL_PRESENT_NO_EXECUTION`

## Warning

> This artifact does not submit orders. Broker execution requires a separate explicit operator phase.

## Preserved

No broker orders; no live_confirm; no raw `execute_trades`; no non-dry-run paper entry.
