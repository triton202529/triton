# Phase 197B — MU Pipeline Hydration / Universe-to-Signal Sync

## Problem

Phase 197A added `MU` to `data/config/tickers.txt` and fetched raw OHLCV, but after
continuous-loop cycles MU remained invisible:

- No `data/processed/MU.parquet`
- Not in `signal_lifecycle_effective.csv`
- Not in `portfolio_allocation_recommendations.csv`
- `portfolio_allocation_engine` reported `universe=64`

## Root cause

The continuous loop runs `run_market_open.ps1`, which rebuilds analytics
(outcomes, PnL, risk overlay, allocation, manage_positions) from **existing**
signal artifacts. It does **not** run the upstream signal chain:

```
fetch_raw → preprocess → fundamentals/scores → train → signals → lifecycle
```

Config-only universe expansion therefore stays stranded until the standard pipeline runs.

## Fix

1. **`services/universe_pipeline_hydration.py`** — assesses gaps and runs the existing
   signal chain for configured symbols (no forced execution).
2. **`services/score_stocks.py`** — reads momentum from `data/processed/*.parquet`
   (preferred) so newly processed symbols can be scored without legacy results parquets.

## Hydration command

```powershell
python -m services.universe_pipeline_hydration --symbols MU --verbose
```

## Validation

```powershell
python -m services.phase_197b_mu_pipeline_hydration_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197b_mu_pipeline_hydration_validation.ps1
python -m services.paper_entry_loop_stage --verbose --dry-run
```

## Preserved

No execution gates weakened. Phase 196X/196Y/196Z and 197A no-force onboarding intact.
