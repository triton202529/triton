# Phase 196V — Allocation-to-Paper-Entry Handoff Repair

## Problem

TRITON produced valid `OPEN_NEW` allocation candidates (e.g. BITO, XLK, TSLA, QQQ) but the controlled paper-entry stage sourced candidates from `trade_opportunities.csv` ENTRY rows instead of `portfolio_allocation_recommendations.csv`. The planner preview (`execute_trades`) still ranks from trade opportunities, so a risk-blocked symbol like UNG could appear as planner top while allocation OPEN_NEW symbols were ignored. After hours, the audit was overwritten with a sparse `market is closed` block and lost intraday decision detail.

## Repair

### Governed candidate source

`services/paper_entry_loop_stage` now treats **`portfolio_allocation_recommendations.csv`** as the authoritative paper-entry candidate source:

- Select rows with `recommended_action` = `OPEN_NEW` or `ADD`
- Support `ticker` and `symbol` columns
- Apply existing safety gates: risk overlay, held position, open/in-flight orders, duplicate protection, max positions
- `trade_opportunities.csv` is enrichment-only (edge/confidence), not an execution source
- `controlled_paper_entry_loop_candidates.csv` remains diagnostic-only

### Allocation ↔ planner contract

Before execution, the stage compares:

- **Eligible allocation symbols** (passed all entry gates)
- **Planner preview symbols** (`preview_planned_buy_symbols`)

If the planner selects symbols outside the eligible allocation set, the stage blocks with **`CONTRACT_MISMATCH`** instead of silently proceeding.

### Audit persistence

| File | Purpose |
|------|---------|
| `paper_entry_loop_stage_audit.json` | Latest run (including after-hours safe block) |
| `paper_entry_loop_stage_last_market_decision.json` | Last evaluation while market was open |

After-hours runs still populate allocation candidate detail in the audit but do not overwrite the last market-open decision file.

### Audit fields (flattened)

Each audit includes: `allocation_candidates_count`, `eligible_entry_symbols`, `rejected_entry_symbols`, `final_selected_symbols`, `final_selection_status`, `final_selection_block_reason`, `source_file_used`, `source_file_mtime`, `paper_arm_status`, `execution_readiness_status`, `market_open`, `open_orders_count`, `positions_count`, `dry_run`, `executed`, `contract_mismatch_detected`.

## What still blocks trades

- Market closed
- Paper arm expired / missing (ASSISTED mode)
- Execution readiness not READY
- Risk overlay `BLOCK_NEW_BUY` / `FORCE_EXIT`
- Duplicate / in-flight orders
- Governance approval missing
- Planner ↔ allocation contract mismatch
- Certification / lifecycle / allocation thresholds (unchanged)

## Validation (market closed)

```powershell
python -m services.phase_196v_allocation_to_paper_entry_handoff_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_196v_allocation_to_paper_entry_handoff_validation.ps1
python -m services.paper_entry_candidate_contract
python -m services.paper_entry_loop_stage
```

Expected when closed: `gate_reason=market is closed`, `executed=false`, allocation OPEN_NEW symbols recognized as eligible.

## Market-open observation

```powershell
$env:TRITON_ENABLE_PAPER_EXECUTION='1'
$env:TRITON_ENABLE_AUTO_PAPER_ENTRIES='1'
python -m services.paper_entry_loop_stage --verbose --dry-run
```

Review `data/results/paper_entry_loop_stage_audit.json` and `paper_entry_loop_stage_last_market_decision.json`.

## Safety doctrine (unchanged)

- No live execution
- No manual `execute_trades`
- No live_confirm
- No gate weakening
