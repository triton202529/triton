# Phase 197A — MU Universe Intake / Governed No-Force Onboarding

## Goal

Add Micron Technology (`MU`) to TRITON's governed trading universe as a normal equity
candidate. No forced trades, manual buys, or gate bypasses.

## Changes

- `data/config/tickers.txt` — add `MU` (membership authority)
- `data/config/universe_assets.json` — optional metadata overlay (NASDAQ, semiconductors)

## Flow

MU enters the same pipeline as every other symbol:

1. `fetch_raw_data` → `data/raw/MU.csv`
2. Preprocess / signals / lifecycle (on next full cycle)
3. `portfolio_allocation_engine` advisory actions
4. Paper-entry gates (196X dispatch, 196Y approval, 196Z planner-empty)

## Validation

```powershell
python -m services.phase_197a_mu_universe_intake_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197a_mu_universe_intake_validation.ps1
python -m services.portfolio_allocation_engine
python -m services.paper_entry_loop_stage --verbose --dry-run
```

## Pass criteria

- `MU` present exactly once in `tickers.txt`
- Not an index/reference ticker (`^` prefix)
- Raw data fetch succeeds or reports clear availability status
- No forced MU execution in paper-entry dry-run
- `live_execution_permitted = false`
- No broker orders during validation
- Continuous loop stopped during onboarding

## Preserved

Phase 196X dispatch, 196Y approval sync, 196Z planner-empty classification — unchanged.
