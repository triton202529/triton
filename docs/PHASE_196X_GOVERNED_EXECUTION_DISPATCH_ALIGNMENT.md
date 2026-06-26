# Phase 196X — Governed Execution Dispatch Alignment

## Problem

Phase 196W aligned planner preview with allocation candidates (BITO, TSLA, XLK). During a live paper-entry pass, `paper_entry_loop_stage` selected those symbols but the `execute_trades` subprocess rebuilt its plan from `trade_opportunities.csv` and selected UNG. MasterExecutionGate blocked submission, but dispatch drift was real.

## Repair

### Dispatch contract

`paper_entry_loop_stage` writes `data/live/paper_entry_dispatch_contract.json` before calling execute_trades:

- `final_selected_symbols` — governed dispatch set (e.g. BITO, TSLA, XLK)
- `allowed_symbols` — eligible allocation pool
- `approved_symbols` — governance-approved symbols
- TTL via `contract_expires_at_utc` (15 minutes)
- SHA-256 `checksum`

### execute_trades governed mode

```text
python -m services.execute_trades --mode paper --execute --dispatch-contract data/live/paper_entry_dispatch_contract.json
```

When `--dispatch-contract` is provided:

1. Load and validate contract (checksum, expiry, paper mode, no live execution)
2. Build planner intake **only** from `final_selected_symbols` via allocation bridge
3. **Never** fall back to full `trade_opportunities.csv`
4. After planning, enforce `EXECUTION_DISPATCH_CONTRACT_MISMATCH` if any buy symbol is outside dispatch set
5. Write `data/results/execute_trades_dispatch_audit.json`

## Validation (safe)

```powershell
python -m services.phase_196x_governed_execution_dispatch_alignment_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_196x_governed_execution_dispatch_alignment_validation.ps1
python -m services.paper_entry_loop_stage --verbose --dry-run
```

Do **not** restart continuous loop until validation PASS.

## Controlled paper validation (operator approval only)

```powershell
python -m services.paper_entry_loop_stage --verbose
python -m services.snapshot_live_orders --mode paper
Get-Content data/results/paper_entry_loop_stage_audit.json -Raw
```

## Restart after approval

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_triton_runtime_supervisor.ps1
```

## Safety (unchanged)

No live execution, no gate weakening, no stale candidate cache for execution.
