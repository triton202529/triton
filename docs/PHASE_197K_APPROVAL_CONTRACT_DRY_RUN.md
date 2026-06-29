# Phase 197K — Approval Contract Dry-Run / No-Execution Simulation

## Problem

197J defined the operator paper defensive exit approval contract, but TRITON had no safe
way to prove the validator rejects unsafe approvals without creating runtime authority or
submitting broker orders.

## Fix

`services/phase_197k_approval_contract_dry_run_validation.py` runs fixture-only contract
validation under `data/results/phase_197k_approval_contract_fixtures/`.

The validator accepts `contract_path` override (197K only) so fixtures never become
active runtime authority at `data/governance/operator_paper_defensive_exit_approval.json`.

## Fixture cases tested

| Case | Expected result |
|------|-----------------|
| `MISSING_APPROVAL` | `OPERATOR_APPROVAL_REQUIRED` |
| `EXPIRED_APPROVAL` | `OPERATOR_APPROVAL_INVALID` |
| `WRONG_DECISION` | `OPERATOR_APPROVAL_INVALID` |
| `NON_PAPER_MODE_APPROVAL` | `OPERATOR_APPROVAL_INVALID` |
| `MISSING_LIVE_BLOCK_ACKNOWLEDGEMENT` | `OPERATOR_APPROVAL_INVALID` |
| `MISSING_PAPER_ONLY_ACKNOWLEDGEMENT` | `OPERATOR_APPROVAL_INVALID` |
| `PACKET_ID_MISMATCH` | `OPERATOR_APPROVAL_INVALID` |
| `PREVIEW_HASH_MISMATCH` | `OPERATOR_APPROVAL_INVALID` |
| `SYMBOL_NOT_IN_PREVIEW` | `OPERATOR_APPROVAL_INVALID` |
| `MAX_NOTIONAL_EXCEEDS_CAP` | `OPERATOR_APPROVAL_INVALID` |
| `OPEN_ORDERS_CONFLICT` | `OPERATOR_APPROVAL_INVALID` |
| `VALID_PAPER_ONLY_APPROVAL` | `OPERATOR_APPROVAL_VALID_PAPER_ONLY` |

## Runtime authority preserved

- Real approval artifact is **not** created by 197K
- Real runtime status remains `OPERATOR_APPROVAL_REQUIRED`
- Valid fixture acceptance does **not** enable execution
- No broker orders submitted
- Live remains blocked
- Governance and readiness gates unchanged

## Validation

```powershell
python -m services.phase_197k_approval_contract_dry_run_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197k_approval_contract_dry_run_validation.ps1
python -m services.phase_197j_operator_paper_approval_packet_validation
python -m services.phase_197i_defensive_exit_governance_sync_validation
python -m services.paper_entry_loop_stage --verbose --dry-run
python -m services.snapshot_live_orders --mode paper
```

## Preserved

Paper-only; live blocked; no auto-approval; no cap increase; no forced execution; all gates intact.
