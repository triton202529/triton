# Phase 197P — Order Guard Snapshot Authority Reconciliation

## Problem

`guard_snapshot.json` was stale (~15k minutes) while `open_orders_snapshot.csv` was fresh with
zero open orders. TRITON emitted ambiguous `[GATE_FRESHNESS]` warnings because
`snapshot_live_orders` refreshed broker CSV snapshots but never updated the guard artifact.

## Authority model

`services/order_guard_snapshot_authority.py` defines explicit order guard states:

| State | Meaning |
|-------|---------|
| `ORDER_GUARD_FRESH` | Guard snapshot within TTL |
| `ORDER_GUARD_STALE_BUT_BROKER_ZERO_OPEN_ORDERS` | Guard stale; broker/open-orders snapshot fresh with zero open orders — acceptable for paper observation |
| `ORDER_GUARD_STALE_AND_UNVERIFIED` | Guard stale and broker authority unverified — not safe |
| `ORDER_GUARD_OPEN_ORDERS_PRESENT` | In-flight orders — execution ineligible |
| `ORDER_GUARD_BROKER_SNAPSHOT_FAILED` | Broker snapshot failed — cannot verify |

### Writers / readers

**Writers:** `reconcile_state`, `guard_control`, `guard_set`, `guard_auto`, `write_guard_snapshot`,
`master_execution_gate.try_refresh_stale_guard_snapshot_if_clear`, and (197P)
`order_guard_snapshot_authority.refresh_guard_snapshot_from_broker_authority`.

**Readers:** `master_execution_gate`, `paper_entry_loop_stage`, `preflight_checks`, `safety_gate`,
`live_gate`, `defensive_exit_market_hours_governance_recheck`, dashboards, and others.

### Why snapshot did not refresh guard (historically)

`snapshot_live_orders` writes `open_orders_snapshot.csv`, `recent_orders.csv`, and
`positions_snapshot.csv` only. Guard snapshots were maintained by reconcile/guard pipelines.

## Fix (197P)

1. **Classification** — Central authority helper reads broker snapshot status, open orders CSV,
   guard snapshot, and classifies state.
2. **Optional refresh** — After successful `snapshot_live_orders --mode paper` with zero open
   orders and guard not blocked, refresh `guard_snapshot.json` as a read-only state artifact
   with `source_authority=BROKER_OPEN_ORDERS_SNAPSHOT` and `broker_orders_submitted=false`.
3. **197O recheck** — Reports `order_guard_authority_state` and explicit gate freshness note.
4. **Master gate (paper)** — Stale guard with broker-zero-open-orders suppresses ambiguous
   `STALE_GUARD_SNAPSHOT` warning; does not bypass execution guards.

## Validation

```powershell
python -m services.phase_197p_order_guard_snapshot_authority_validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\show_phase_197p_order_guard_snapshot_authority_validation.ps1
python -m services.phase_197o_market_hours_governance_recheck_validation
python -m services.snapshot_live_orders --mode paper
```

## Preserved

Paper mode only. No approval artifact. No execution. No live enablement. Live remains blocked.
`eligible_for_execution=false` always.
