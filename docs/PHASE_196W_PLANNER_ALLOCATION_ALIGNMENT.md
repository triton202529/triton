# Phase 196W — Planner Preview Alignment With Governed Allocation Candidates

## Problem (post-196V)

Phase 196V made `portfolio_allocation_recommendations.csv` the governed paper-entry source and detected `CONTRACT_MISMATCH` when the planner preview selected UNG from `trade_opportunities.csv` while allocation OPEN_NEW candidates were BITO, QQQ, TSLA, XLK.

## Repair

### Governed planner intake

`execute_trades.preview_planned_buy_symbols` supports allocation mode:

- `source="allocation"`
- `allowed_symbols=[...]` — eligible governed allocation symbols only

Intake builds rows via `build_governed_allocation_opportunities_df`:

1. Primary identity from allocation OPEN_NEW/ADD rows
2. Merge `trade_opportunities.csv` fields when a row exists
3. Otherwise synthesize planner intake from allocation edge/confidence/delta (no invented scores beyond allocation fields)

### Paper-entry integration

`paper_entry_loop_stage` calls planner preview with `allowed_symbols=eligible_entry_symbols` so UNG and other non-allocation symbols cannot be selected.

### Per-symbol rejection audit

When allocation candidates fail planner filters, `planner_rejected_symbols` maps symbol → explicit reason (final_score too low, sizing zero, sector cap, duplicate/in-flight, etc.).

## Validation (market closed)

```powershell
python -m services.phase_196w_planner_allocation_alignment_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_196w_planner_allocation_alignment_validation.ps1
python -m services.paper_entry_loop_stage
```

Expected:

- `gate_reason=market is closed`, `executed=false`
- `planner_input_mode=allocation`
- `ung_selected_outside_allocation=false`
- `contract_mismatch_detected=false` when planner output ⊆ eligible allocation

## Market-open observation

```powershell
$env:TRITON_ENABLE_PAPER_EXECUTION='1'
$env:TRITON_ENABLE_AUTO_PAPER_ENTRIES='1'
python -m services.paper_entry_loop_stage --verbose --dry-run
```

## Safety (unchanged)

No live execution, no gate weakening, no forced trades, no stale candidate cache for execution.
