# Phase 197W-FIX — Post-Approval Validator Repair

## Purpose

Repair paper-only operator approval artifact lifecycle handling after Phase 197W.
Expired or invalid approvals must classify explicitly — not as generic readiness or
artifact-present failures.

## Shared Classifier

`services/paper_operator_approval_artifact_state.py` classifies:

| State | Meaning |
|-------|---------|
| `PAPER_APPROVAL_ABSENT` | No artifact at known path |
| `PAPER_APPROVAL_VALID_CURRENT` | Valid, unexpired, packet-matched |
| `PAPER_APPROVAL_EXPIRED` | TTL elapsed |
| `PAPER_APPROVAL_MALFORMED` | JSON/field contract failure |
| `PAPER_APPROVAL_SCOPE_INVALID` | Not `PAPER_ONLY` |
| `PAPER_APPROVAL_PACKET_MISMATCH` | packet/hash/symbol mismatch |
| `PAPER_APPROVAL_LIVE_AUTH_INVALID` | live authorization present |
| `PAPER_APPROVAL_EXECUTION_AUTH_INVALID` | broker/raw execution authorized |

## Usage

```powershell
python -m services.phase_197w_fix_post_approval_validator_repair
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197w_fix_post_approval_validator_repair.ps1
```

## Downstream States (Expired Approval)

| Validator | State |
|-----------|-------|
| 197W | `PASS_SAFE_BLOCKED` / `PAPER_OPERATOR_APPROVAL_EXPIRED_CLEANUP_REQUIRED` |
| 197U | `GOV_PACKET_REFRESH_BLOCKED_APPROVAL_EXPIRED` |
| 197N | `GOV_REVIEW_BLOCKED_BY_EXPIRED_APPROVAL` |
| 197O | `MARKET_HOURS_RECHECK_BLOCKED_BY_EXPIRED_APPROVAL` |
| paper_entry dry-run | `SAFE_BLOCK` / `approval_expired` |

## Downstream States (Valid Current Approval)

| Validator | State |
|-----------|-------|
| 197W | `PASS` / `PAPER_OPERATOR_APPROVAL_VALID_CURRENT_EXECUTION_NOT_RUN` |
| 197U | `GOV_PACKET_CURRENT_PAPER_APPROVAL_PRESENT` |
| 197N | `GOV_REVIEW_CURRENT_PAPER_APPROVAL_PRESENT` |
| 197O | `MARKET_HOURS_RECHECK_OPERATOR_APPROVAL_PRESENT_NO_EXECUTION` |
| paper_entry dry-run | `SAFE_BLOCK` / `explicit_broker_execution_phase_required` |

## Guarded Cleanup

Expired or invalid artifacts may be removed only via:

```powershell
python -m services.phase_197w_fix_expired_approval_cleanup --execute
```

Safety proof required: `PAPER_ONLY`, no live auth, `open_orders=0`, `live_blocked=true`, no execution.

## Preserved

No broker orders; no live_confirm; no raw `execute_trades`; no non-dry-run paper entry.
