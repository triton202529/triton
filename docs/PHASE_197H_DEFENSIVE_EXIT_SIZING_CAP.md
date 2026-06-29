# Phase 197H — Defensive Exit Sizing Cap / Batch Notional Reconciliation

## Problem

After 197G fixed handoff mapping, placement dry-run failed with `BATCH_NOTIONAL_BLOCK`:
full defensive exit batch (~$95k) exceeds `place_live_orders` default cap ($5,000) and
even `manage_positions` configured cap ($50,000).

## Fix

1. **`defensive_exit_sizing_cap.py`** — Batch notional estimate, cap comparison, governed
   capped-batch preview (priority by plan rank / high-drag exits first).

2. **`place_live_orders.py`** — Emit `HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED` when manage
   handoff rows map but batch exceeds cap (no vague `BATCH_EMPTY`).

3. **`manage_positions_handoff_diagnostic.py`** — Integrate sizing analysis into handoff audit.

4. **`manage_positions.py`** — Treat `BATCH_NOTIONAL_BLOCK` as explicit handoff status (not
   batch-empty no-op); write capped preview to `defensive_exit_capped_batch_preview.json`.

## Caps (not raised)

| Source | Cap |
|--------|-----|
| `place_live_orders` CLI default | $5,000 |
| `config/manage_positions.json` | $50,000 |

Partial batches are **authorized** via `max_management_orders_per_run` (10) unless
`partial_defensive_batch_forbidden` is set.

## Validation

```powershell
python -m services.phase_197h_defensive_exit_sizing_cap_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197h_defensive_exit_sizing_cap_validation.ps1
python -m services.paper_entry_loop_stage --verbose --dry-run
```

## Preserved

Paper-only; live blocked; no broker orders during validation; all governance gates intact.
