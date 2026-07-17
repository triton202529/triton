# TRITON Phase 198W — Market-Open Refresh Pipeline Validation

**Generated:** 2026-07-17T15:47:23.961490+00:00
**Status:** REVIEW_REQUIRED
**Refresh pipeline status:** `SAFE_BLOCKED`

## Executive summary

Refresh pipeline SAFE_BLOCKED: generate_signals=FAIL, apply_signal_lifecycle=PASS, build_effective_lifecycle=PASS. After refresh: signals_stale=True, lifecycle_stale=False, effective_stale=False. execution_readiness=NOT_READY (failed=['signals']); certification=NOT_CERTIFIED. Broker safe: open_orders=0, live_blocked=True, approval_absent=True, broker_auth_absent=True. GLD target satisfied=True. allocation open_new=0 add=0. No broker orders submitted by this phase.

## Refresh steps

- `services.generate_signals`: FAIL (exit=1)
- `services.apply_signal_lifecycle`: PASS (exit=0)
- `services.build_effective_lifecycle`: PASS (exit=0)

## Artifact freshness after refresh

| Artifact | Age (s) | Threshold (s) | Stale |
|----------|---------|---------------|-------|
| signals.csv | 338396.085 | 172800 | True |
| signal_lifecycle.csv | 1.607 | 172800 | False |
| signal_lifecycle_effective.csv | 0.871 | 172800 | False |

## Readiness / certification

| Field | Value |
|-------|-------|
| execution_readiness_status | NOT_READY |
| execution_readiness_failed_checks | ['signals'] |
| certification_status | NOT_CERTIFIED |
| certification_failed_requirements | ['Readiness', 'veto:mandatory_preservation_gate_failed'] |

## Safety / posture

| Field | Value |
|-------|-------|
| broker_orders_submitted | False |
| open_orders_count | 0 |
| paper_arm_valid | True |
| live_execution_blocked | True |
| approval_artifact_present | False |
| broker_auth_artifact_present | False |
| gld_qty / gld_weight | 16 / 0.197232 |
| gld_target_satisfied | True |
| allocation_open_new / add | 0 / 0 |

## Recommended runtime posture

SAFE_BLOCK — refresh step failed (generate_signals_failed: Fix upstream: ensure train_model fetch window includes the completed bar (end-date inclusivity).); readiness safe-blocked; investigate upstream predictions/model (fetch_raw_data/preprocess/train_model) before retrying. No broker orders, no authority artifacts; live blocked.

---

*MARKET_OPEN_REFRESH_PIPELINE_VALIDATION_ONLY*
