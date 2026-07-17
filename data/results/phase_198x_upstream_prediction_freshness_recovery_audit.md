# TRITON Phase 198X — Upstream Prediction Freshness Recovery Audit

**Generated:** 2026-07-17T15:55:42.762014+00:00
**Status:** PASS
**Blocker classification:** `RAW_DATA_STALE`

## Executive summary

generate_signals FAIL: required_session=2026-07-16, latest_prediction=2026-07-10, lag_sessions=4. Upstream: raw=STALE@2026-07-10, processed=STALE@2026-07-10, predictions=STALE@2026-07-10. Primary classification=RAW_DATA_STALE. STRICT_ASOF correctly blocked=True. SPCX quarantine relevant=False. Market-open signal refresh present=True; upstream fetch/train in market-open=False. Safe: open_orders=0, live_blocked=True, authority_absent=True. Recommended: python -m services.fetch_raw_data; python -m services.preprocess; python -m services.train_model; python -m services.generate_signals; python -m services.apply_signal_lifecycle; python -m services.build_effective_lifecycle; python -m services.execution_readiness; python -m services.capital_preservation_certification --skip-stress-revalidation

## generate_signals AS_OF

| Field | Value |
|-------|-------|
| generate_signals_status | FAIL |
| failure_reason | AS_OF invariant: session=2026-07-16 max_prediction=2026-07-10 lag_sessions=4 coverage=0.0 strict=True |
| required_session_date | 2026-07-16 |
| latest_prediction_date | 2026-07-10 |
| prediction_age_sessions | 4 |
| prediction_artifact_path | `data/predictions/*_predictions.(parquet|csv)` |
| strict_asof_enabled | True |
| strict_asof_block_was_correct | True |

## Upstream layer freshness

| Layer | Status | Latest date | Missing |
|-------|--------|-------------|---------|
| raw | STALE | 2026-07-10 | 0 |
| processed | STALE | 2026-07-10 | — |
| model/predictions | STALE | 2026-07-10 | — |

## Pipeline requirements

| Field | Value |
|-------|-------|
| training_required | True |
| prediction_refresh_required | True |
| fetch_raw_data_required | True |
| preprocess_required | True |
| train_model_required | True |
| spcx_quarantine_relevant | False |
| market_open_calls_signal_refresh | True |
| continuous_loop_invokes_market_open | True |
| market_open_calls_upstream_fetch_train | False |

## Secondary classifications

- `STRICT_ASOF_CORRECTLY_BLOCKED`
- `PREDICTIONS_STALE`
- `PROCESSED_DATA_STALE`
- `RAW_DATA_STALE`
- `TRAINING_PIPELINE_REQUIRED`
- `MODEL_INPUT_STALE`

## Safety

| Field | Value |
|-------|-------|
| broker_connected | True |
| open_orders_count | 0 |
| paper_arm_valid | True |
| live_execution_blocked | True |
| approval_artifact_present | False |
| broker_auth_artifact_present | False |
| gld_qty / weight | 16 / 0.197232 |
| gld_target_satisfied | True |
| allocation open_new / add | 0 / 0 |
| broker_orders_submitted | False |

## Recommended refresh sequence

```
python -m services.fetch_raw_data; python -m services.preprocess; python -m services.train_model; python -m services.generate_signals; python -m services.apply_signal_lifecycle; python -m services.build_effective_lifecycle; python -m services.execution_readiness; python -m services.capital_preservation_certification --skip-stress-revalidation
```

## Recommended next phase

198Y — execute upstream prediction recovery (fetch_raw_data → preprocess → train_model → generate_signals) without loosening TRITON_STRICT_ASOF / MAX_LAG_SESSIONS; then re-check readiness/certification

---

*UPSTREAM_PREDICTION_FRESHNESS_RECOVERY_AUDIT_ONLY*
