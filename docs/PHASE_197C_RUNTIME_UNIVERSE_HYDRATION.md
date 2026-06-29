# Phase 197C — Universe Hydration Integration Into Runtime Supervisor

## Problem

Phase 197B proved that new governed-universe symbols can exist in `tickers.txt` and raw
data but remain missing from processed parquets, lifecycle, and allocation because the
continuous loop only runs `run_market_open.ps1` analytics — not the upstream signal chain.

## Fix

`services/runtime_universe_hydration_preflight.py` runs as **step E0** in
`scripts/run_market_open.ps1`, before portfolio allocation and paper-entry:

1. Fast set-based gap scan (configured vs processed / signals / lifecycle)
2. **SKIPPED_NO_GAPS** when all symbols are hydrated (typical steady state)
3. Cheap per-symbol raw fetch + preprocess when only processed data is missing
4. Full signal-chain hydration for stranded symbols when market is closed
5. **DEFERRED_MARKET_OPEN** when expensive hydration is needed during market hours
   (override with `TRITON_RUNTIME_HYDRATION_ALLOW_MARKET_OPEN=1`)

Audit written every evaluation: `data/results/runtime_universe_hydration_preflight.json`

## Runtime authority

Unchanged:

- `scripts/start_triton_runtime_supervisor.ps1`
- `scripts/run_continuous_loop.ps1` at **45 minutes**
- No new loops, no Scheduled Task changes, no broker orders

## Validation

```powershell
python -m services.runtime_universe_hydration_preflight --verbose
python -m services.phase_197c_runtime_hydration_integration_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197c_runtime_hydration_integration_validation.ps1
```

## Preserved

Phases 196X–196Z, 197A, 197B behavior and all execution/safety gates.
