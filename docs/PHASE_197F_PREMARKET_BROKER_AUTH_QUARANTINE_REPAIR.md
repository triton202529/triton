# Phase 197F — Pre-Market Broker Auth and Quarantine Regression Repair

## Problems

### 1. Broker auth 401

Root `.env` was removed from git history. Shell aliases (`ALPACA_KEY_ID`) were set but
`broker_alpaca.py` only read `ALPACA_API_KEY_ID` / `APCA_API_KEY_ID` and did not normalize
secret aliases (`ALPACA_SECRET_KEY`). 401 failures surfaced as raw tracebacks.

### 2. Quarantine overbreadth

After merge, `runtime_universe_hydration_preflight` quarantined 33+ mature symbols because:

- Gap detection treated **missing processed parquet** as stranded even when lifecycle/signals existed
- Classifier quarantined symbols with `proc_rows=0` despite mature `raw_rows` (git-untracked artifacts)

Only **SPCX** should be quarantined (`QUARANTINED_IPO_EDGE`, 8 raw rows).

## Fixes

### Broker (`services/triton_env_loader.py`, `services/broker_alpaca.py`)

- Load git-ignored local env: `.env.local`, `.env.paper.local`, `config/local/runtime_env.local`
- Normalize key/secret/base URL aliases across `ALPACA_*` and `APCA_*`
- Safe diagnostics: masked key, lengths, alias used, paper endpoint check
- `SAFE_AUTH_BLOCK` on missing/invalid auth — no secret values in logs
- `scripts/triton_env_bootstrap.ps1` dot-sourced by runtime scripts
- Template: `config/env.paper.example` (no real secrets)

### Quarantine (`symbol_hydration_quarantine.py`, `runtime_universe_hydration_preflight.py`)

- Stranded = missing **lifecycle OR signals** (not missing processed alone)
- `artifact_regression_symbols` for processed-only gaps (informational)
- Quarantine only: IPO edge (`raw_rows < train_min_rows`), vendor missing, unsupported
- Mature symbols (`raw_rows >= train_min_rows`) or pipeline-present symbols → not quarantined

## Validation

```powershell
python -m services.phase_197f_premarket_broker_auth_quarantine_repair_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197f_premarket_broker_auth_quarantine_repair_validation.ps1
python -m services.runtime_universe_hydration_preflight --verbose
python -m services.phase_197d_spcx_unsupported_symbol_quarantine_validation
python -m services.phase_197e_quarantine_registry_write_discipline_validation
python -m services.snapshot_live_orders --mode paper
```

Copy `config/env.paper.example` → `.env.local` with valid paper keys for live broker snapshot OK.

## Preserved

All Phase 196X–197E safety gates; no live execution; no broker orders during validation.

## Restart (only after broker snapshot + quarantine PASS)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_triton_runtime_supervisor.ps1
```
