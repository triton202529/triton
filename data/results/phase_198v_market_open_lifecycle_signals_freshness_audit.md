# TRITON Phase 198V — Market-Open Lifecycle / Signals Freshness Audit

**Generated:** 2026-07-17T15:25:35.931724+00:00
**Status:** PASS
**Blocker classification:** `REFRESH_PIPELINE_GAP`

## Executive summary

Market-open readiness is NOT_READY because failed_checks=['lifecycle', 'signals']. Lifecycle diagnostics FILE_STALE on signal_lifecycle.csv (age=178856.400346s > threshold=172800.0s). Signals check: signals.csv exceeds freshness window. Classification=REFRESH_PIPELINE_GAP. Broker safety clean=True; live blocked=True; paper arm valid=True. No broker orders submitted by this audit.

## Readiness / certification

| Field | Value |
|-------|-------|
| readiness_status | NOT_READY |
| failed_checks | ['lifecycle', 'signals'] |
| certification_status | NOT_CERTIFIED |
| failed_requirements | ['Readiness', 'veto:mandatory_preservation_gate_failed'] |

## Lifecycle freshness

| Field | Value |
|-------|-------|
| lifecycle_stale | True |
| lifecycle_file (diagnostics FILE_STALE) | `C:/Users/akimw/triton/data/results/signal_lifecycle.csv` |
| lifecycle_file_mtime | 2026-07-15T13:44:37Z |
| lifecycle_file_age_seconds | 178856.400346 |
| lifecycle_freshness_threshold_seconds | 172800.0 |
| lifecycle_rows | 65 |
| lifecycle_missing_columns | [] |
| readiness lifecycle path | `C:/Users/akimw/triton/data/results/signal_lifecycle_effective.csv` |
| readiness lifecycle age seconds | 178855.681 |

## Signals freshness

| Field | Value |
|-------|-------|
| signals_stale | True |
| signals_file | `C:/Users/akimw/triton/data/results/signals.csv` |
| signals_file_mtime | 2026-07-13T17:47:26Z |
| signals_file_age_seconds | 337087.465 |
| signals_freshness_threshold_seconds | 172800.0 |
| signals_snapshot_age_seconds | 2472865.197 |

## Market-open / loop refresh

| Field | Value |
|-------|-------|
| stale_files_expected_after_market_close | True |
| continuous_loop_refreshing_lifecycle_signals | False |
| market_open_refresh_pipeline_gap | True |
| missing refresh modules | ['services.generate_signals', 'services.apply_signal_lifecycle', 'services.build_effective_lifecycle'] |
| watchdog_vs_readiness_path_mismatch | False |
| certification_failing_from_stale_inputs_only | True |

## Safety / posture

| Field | Value |
|-------|-------|
| broker_connected | True |
| open_orders_count | 0 |
| positions_count | 10 |
| paper_arm_valid | True |
| live_execution_blocked | True |
| approval_artifact_present | False |
| broker_auth_artifact_present | False |
| GLD qty / weight | 16 / 0.197118 |
| gld_target_satisfied | True |
| allocation open_new / add / exit / trim | 0 / 0 / 3 / 0 |

## Recommendations

- **Refresh command:** `python -m services.generate_signals; python -m services.apply_signal_lifecycle; python -m services.build_effective_lifecycle; python -m services.execution_readiness; python -m services.capital_preservation_certification --skip-stress-revalidation`
- **Next phase:** 198W — refresh market-open / continuous-loop signal+lifecycle pipeline (generate_signals → apply_signal_lifecycle → build_effective_lifecycle) then re-evaluate readiness/certification; do not loosen freshness thresholds

## Classification notes

- run_market_open.ps1 does not invoke generate_signals / apply_signal_lifecycle / build_effective_lifecycle; stale inputs persist after market-open sequence
- broker/order safety is clean; blocker is stale market-data inputs

---

*MARKET_OPEN_LIFECYCLE_SIGNALS_FRESHNESS_AUDIT_ONLY*
