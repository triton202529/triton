# Phase 197D — SPCX IPO / Unsupported Symbol Quarantine

## Problem

After Phase 197C integrated `runtime_universe_hydration_preflight`, every runtime cycle
reported:

- `status = HYDRATION_INCOMPLETE`
- `reason = preprocess_failed`
- `stranded_symbols = ["SPCX"]`
- `failed_symbols = ["SPCX"]`

SPCX is configured in the governed universe but cannot complete the signal chain because
it has insufficient price history for model training (IPO / new-listing edge case).

## Root cause (SPCX)

| Check | Result |
|-------|--------|
| In `tickers.txt` / `universe_assets.json` | Yes (SpaceX, ACTIVE) |
| Raw CSV rows | ~8 (dates 2026-06-12 → 2026-06-24) |
| Processed parquet rows | ~5 |
| `train_model --min-rows` | 30 (default) |
| Lifecycle / signals / allocation | Missing |

Classification: **`QUARANTINED_IPO_EDGE`** — insufficient history for hydration.

## Fix

`services/symbol_hydration_quarantine.py` classifies configured symbols that cannot yet
enter the active signal chain for acceptable reasons:

- `QUARANTINED_UNSUPPORTED`
- `QUARANTINED_INSUFFICIENT_HISTORY`
- `QUARANTINED_IPO_EDGE`
- `QUARANTINED_VENDOR_DATA_UNAVAILABLE`
- `QUARANTINED_NOT_TRADABLE`
- `QUARANTINED_PENDING_DATA`

`services/runtime_universe_hydration_preflight.py` applies quarantine before hydration
attempts. Quarantined symbols:

- Do **not** count as hydration failures
- Are removed from `stranded_symbols` and `failed_symbols`
- Set `hydration_needed = false` when only quarantined gaps remain
- Return `QUARANTINED_GAPS` instead of `HYDRATION_INCOMPLETE`

Registry: `data/governance/symbol_hydration_quarantine.json` (re-checkable later).

## BRK-B

Remains **`allocation_refresh_only_symbols`** — not a hydration failure. No forced
execution or allocation rule changes.

## Expected preflight outcome

```
status = QUARANTINED_GAPS
reason = unsupported_or_insufficient_history_symbols_quarantined
stranded_symbols = []
quarantined_symbols = ["SPCX"]
failed_symbols = []
hydration_needed = false
broker_orders_submitted = false
live_execution_permitted = false
```

## Validation

```powershell
python -m services.runtime_universe_hydration_preflight --verbose
python -m services.phase_197d_spcx_unsupported_symbol_quarantine_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197d_spcx_unsupported_symbol_quarantine_validation.ps1
python -m services.phase_197c_runtime_hydration_integration_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197c_runtime_hydration_integration_validation.ps1
python -m services.portfolio_allocation_engine
python -m services.paper_entry_loop_stage --verbose --dry-run
python -m compileall services scripts
```

## Preserved

Phases 196X–196Z, 197A–197C behavior; 45-minute cadence; runtime authority unchanged;
no live execution; no broker orders; all execution/safety gates intact.

## Restart

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_triton_runtime_supervisor.ps1
```
