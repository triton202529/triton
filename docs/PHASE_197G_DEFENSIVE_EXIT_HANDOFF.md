# Phase 197G — Defensive Exit Handoff / BATCH_EMPTY Repair

## Problem

`manage_positions_plan.csv` contained planned defensive EXIT rows (GLD, SLV, CVX, etc.)
and `data/live/manage_orders.csv` was written with handoff rows, but `place_live_orders`
mapped **zero** lifecycle rows and reported:

```
[PLACE_MANAGE_ORDERS] BATCH_EMPTY
[MANAGE_NOOP] reason=BATCH_EMPTY_AFTER_FILTERS
planned=15 executed=0
```

Note: `data/results/manage_orders.csv` is a **different file** (open-order repricing via
`manage_open_orders.py`). Defensive exit handoff uses `data/live/manage_orders.csv`.

## Root cause

`_placement_stance_from_row()` only returned `ROTATE_EXIT` from `effective_stance`; it did
not map `EXIT` / `TRIM` from `effective_stance` or `opportunity_type`. Manage handoff rows
therefore had empty stances and were filtered out before placement.

## Fix

1. **`place_live_orders.py`** — Map actionable stances from `effective_stance` and
   `opportunity_type`; recognize `manage_handoff` source; emit `HANDOFF_CONTRACT_MISMATCH`
   when handoff rows exist but zero rows map.

2. **`manage_positions_handoff_diagnostic.py`** — Contract diagnostic comparing planned exits,
   handoff rows, placement output, readiness/governance blocks.

3. **`manage_positions.py`** — On `BATCH_EMPTY` or master-gate block, write
   `data/results/manage_positions_handoff_audit.json` and print `[MANAGE_HANDOFF]` status.

## Validation

```powershell
python -m services.phase_197g_defensive_exit_handoff_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197g_defensive_exit_handoff_validation.ps1
python -m services.paper_entry_loop_stage --verbose --dry-run
python -m services.place_live_orders --mode paper --orders data/live/manage_orders.csv --dry-run
```

## Preserved

Paper-only; live blocked; no broker orders during validation; all governance/readiness gates intact.
