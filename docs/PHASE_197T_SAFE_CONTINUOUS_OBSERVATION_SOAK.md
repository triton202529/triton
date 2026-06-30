# Phase 197T — Safe Continuous Observation Soak

## Purpose

Formalize multi-cycle observation soak validation for TRITON continuous runtime after
197S-FIX3 safe-block semantics. Confirms runtime stability, safe-blocking, and
non-execution over one or more recent continuous-loop cycles.

Read-only: does not restart loops, approve trades, or submit broker orders.

## Soak States

| State | Meaning |
|-------|---------|
| `SOAK_PASS_SAFE_BLOCKING` | All inspected cycles completed with expected safe-blocks, no execution |
| `SOAK_PASS_NO_ACTION` | Cycles completed with no execution and no defensive dispatch path |
| `SOAK_READY_FOR_OBSERVATION` | No post-197S safe-block cycle in window; system safe and ready |
| `SOAK_FAIL_BROKER_ORDERS_SUBMITTED` | Broker order submission or exits detected |
| `SOAK_FAIL_CYCLE_FAILURE` | Unexplained `[CYCLE_FAIL]` |
| `SOAK_FAIL_MARKET_SEQUENCE_ABORT` | Market-open abort without recognized safe-block |
| `SOAK_FAIL_LIVE_NOT_BLOCKED` | Live block regression |
| `SOAK_FAIL_APPROVAL_ARTIFACT_PRESENT` | Real approval artifact exists unexpectedly |

## Usage

```powershell
# Default: inspect latest 1 cycle
python -m services.phase_197t_safe_continuous_observation_soak

# Multi-cycle soak (e.g. 2 cycles)
python -m services.phase_197t_safe_continuous_observation_soak --cycles 2

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197t_safe_continuous_observation_soak.ps1
```

## Per-Cycle Inspection

For each cycle the validator correlates:

- `data/results/continuous_loop/continuous_loop_*.log` (`[CYCLE_START]` / `[CYCLE_OK]` / `[CYCLE_FAIL]`)
- Matching `data/results/scheduled_runs/market_open_*.log` (by timestamp proximity)

Captured fields include dispatch interlock state, safe-block markers, broker submission flags,
exits/trims executed, market sequence outcome, and paper entry gate status.

## Expected Post-197S-FIX3 Cycle

```
[DISPATCH_INTERLOCK] status=DISPATCH_BLOCKED_APPROVAL_REQUIRED stage=manage_positions
[SAFE_BLOCK] state=SAFE_BLOCK_DISPATCH_APPROVAL_REQUIRED reason=DISPATCH_APPROVAL_REQUIRED
broker_orders_submitted=false
eligible_for_execution=false
exits_executed=0
=== TRITON MARKET OPEN SEQUENCE COMPLETE ===
[CYCLE_OK] cycle=N
```

## Preserved

No approval artifact; no broker orders; live blocked; no loop start/stop from this phase.
