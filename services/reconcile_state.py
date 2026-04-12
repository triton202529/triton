# services/reconcile_state.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from services.ledger import (
    LEDGER_PATH,
    load_ledger,
    save_ledger,
    ledger_from_broker_positions,
    index_by_symbol,
    utc_now_iso_z,
)

# Uses your existing broker
from services.broker_alpaca import AlpacaBroker  # type: ignore

# ✅ Equity snapshot (CPM feed)
from services.snapshot_equity import snapshot_equity as _snapshot_equity
from services.snapshot_equity import append_snapshot as _append_equity_snapshot


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "reconcile_state.json"

DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RECONCILE_REPORT_PATH = RESULTS_DIR / "reconcile_report.csv"
GUARD_SNAPSHOT_PATH = RESULTS_DIR / "guard_snapshot.json"


@dataclass
class ReconcileResult:
    ok: bool
    code: str
    message: str
    context: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "code": self.code, "message": self.message, "context": self.context}


def _default_cfg() -> Dict[str, Any]:
    return {
        "enabled": True,
        "freeze_on_mismatch": True,
        # Tolerances
        "qty_tolerance": 0,  # absolute shares tolerance
        "allow_shorts": False,  # freeze if any short position exists
        # What to reconcile
        "write_report": True,
        "overwrite_ledger_from_broker": True,  # broker is authoritative
        # Classify what triggers a freeze
        "freeze_on_missing_symbol": True,
        "freeze_on_qty_mismatch": True,
        "freeze_on_unexpected_open_orders": False,  # optional (can be noisy)
        # ✅ Snapshot control (best-effort; never blocks reconcile)
        "write_equity_snapshot": True,  # append equity snapshots pre/post reconcile
        # Safety meta
        "tag": "reconcile_state",
    }


def _load_cfg() -> Dict[str, Any]:
    cfg = _default_cfg()
    try:
        if CONFIG_PATH.exists() and CONFIG_PATH.stat().st_size > 0:
            user_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(user_cfg, dict):
                cfg.update(user_cfg)
    except Exception:
        pass
    return cfg


def _write_guard_snapshot(*, blocked: bool, code: str, message: str, extra: Dict[str, Any]) -> None:
    payload = {
        "updated_at": utc_now_iso_z(),
        "blocked": bool(blocked),
        "mode": "FREEZE" if blocked else "OK",
        "code": code,
        "reason": message,
        "message": message,
        "kill_switch": bool(blocked),  # manual_order_desk recognizes this
        "extra": extra or {},
    }
    try:
        GUARD_SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _open_orders_count_by_symbol(open_orders: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for o in open_orders or []:
        sym = str(o.get("symbol") or "").upper().strip()
        if not sym:
            continue
        out[sym] = out.get(sym, 0) + 1
    return out


def _safe_account_id(acct: Dict[str, Any]) -> str:
    return str(acct.get("id") or acct.get("account_number") or "").strip()


def _maybe_snapshot_equity(
    broker: AlpacaBroker,
    *,
    enabled: bool,
    phase: str,
    source: str,
    tag: str,
) -> None:
    """Best-effort: never throws."""
    if not enabled:
        return
    try:
        row = _snapshot_equity(broker)
        # add metadata (extra columns will be ignored by CSV writer if not in schema)
        row["reconcile_phase"] = phase
        row["reconcile_source"] = source
        row["reconcile_tag"] = tag
        _append_equity_snapshot(row)
    except Exception:
        pass


def _diff_ledger_vs_broker(
    *,
    ledger_df: pd.DataFrame,
    broker_positions: List[Dict[str, Any]],
    open_orders: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
    """
    Returns:
      diff_df: row-per-symbol diff view
      summary: counts + risk flags
      should_freeze: bool
    """
    qty_tol = int(cfg.get("qty_tolerance", 0) or 0)
    allow_shorts = bool(cfg.get("allow_shorts", False))

    # Build broker map (authoritative)
    open_counts = _open_orders_count_by_symbol(open_orders)
    broker_df = ledger_from_broker_positions(
        broker_positions,
        broker_account="",
        open_orders_by_symbol=open_counts,
        source="broker",
    )
    bmap = index_by_symbol(broker_df)
    lmap = index_by_symbol(ledger_df)

    symbols = sorted(set(bmap.keys()) | set(lmap.keys()))
    rows: List[Dict[str, Any]] = []

    should_freeze = False
    flags = {
        "short_positions": 0,
        "missing_in_ledger": 0,
        "missing_in_broker": 0,
        "qty_mismatch": 0,
        "open_orders_present": 0,
    }

    for sym in symbols:
        b = bmap.get(sym)
        l = lmap.get(sym)

        b_qty = int(b["qty"]) if b else 0
        l_qty = int(l["qty"]) if l else 0
        b_oo = int(b.get("open_orders_count", 0)) if b else 0
        l_oo = int(l.get("open_orders_count", 0)) if l else 0

        missing_in_ledger = (b is not None) and (l is None)
        missing_in_broker = (l is not None) and (b is None)

        qty_diff = b_qty - l_qty
        qty_mismatch = abs(qty_diff) > qty_tol

        short_flag = b_qty < 0

        if short_flag:
            flags["short_positions"] += 1
            if not allow_shorts:
                should_freeze = True

        if missing_in_ledger:
            flags["missing_in_ledger"] += 1
            if bool(cfg.get("freeze_on_missing_symbol", True)):
                should_freeze = True

        if missing_in_broker:
            flags["missing_in_broker"] += 1
            # Usually not fatal because ledger could be stale; broker is authoritative.
            # We *still* treat as mismatch if freeze_on_missing_symbol is true.
            if bool(cfg.get("freeze_on_missing_symbol", True)):
                should_freeze = True

        if qty_mismatch:
            flags["qty_mismatch"] += 1
            if bool(cfg.get("freeze_on_qty_mismatch", True)):
                should_freeze = True

        if b_oo > 0:
            flags["open_orders_present"] += 1
            if bool(cfg.get("freeze_on_unexpected_open_orders", False)):
                should_freeze = True

        rows.append(
            {
                "symbol": sym,
                "broker_qty": b_qty,
                "ledger_qty": l_qty,
                "qty_diff": qty_diff,
                "qty_tol": qty_tol,
                "qty_mismatch": bool(qty_mismatch),
                "broker_open_orders": b_oo,
                "ledger_open_orders": l_oo,
                "missing_in_ledger": bool(missing_in_ledger),
                "missing_in_broker": bool(missing_in_broker),
                "short_position": bool(short_flag),
            }
        )

    diff_df = pd.DataFrame(rows)
    summary = {
        "symbols": len(symbols),
        **flags,
        "qty_tolerance": qty_tol,
        "allow_shorts": allow_shorts,
    }
    return diff_df, summary, should_freeze


def reconcile_state(
    broker: AlpacaBroker,
    *,
    phase: str = "pre",
    source: str = "executor",
) -> ReconcileResult:
    """
    Reconcile broker state with TRITON ledger.

    - Reads ledger.parquet
    - Pulls broker positions + open orders
    - Writes reconcile_report.csv
    - Overwrites ledger from broker (authoritative) if enabled
    - Writes guard_snapshot.json (blocked or ok)
    - ✅ Appends equity snapshots pre + post reconcile (best-effort)
    """
    cfg = _load_cfg()
    tag = str(cfg.get("tag", "reconcile_state"))
    snapshot_enabled = bool(cfg.get("write_equity_snapshot", True))

    # ✅ Snapshot PRE (always best-effort)
    _maybe_snapshot_equity(
        broker, enabled=snapshot_enabled, phase=f"{phase}_start", source=source, tag=tag
    )

    if not bool(cfg.get("enabled", True)):
        _write_guard_snapshot(
            blocked=False,
            code="RECONCILE_DISABLED",
            message="Reconcile disabled in config.",
            extra={"phase": phase, "source": source, "tag": tag},
        )
        # ✅ Snapshot POST
        _maybe_snapshot_equity(
            broker, enabled=snapshot_enabled, phase=f"{phase}_end", source=source, tag=tag
        )
        return ReconcileResult(
            True,
            "RECONCILE_DISABLED",
            "Reconcile disabled in config.",
            {"phase": phase, "source": source},
        )

    # Load current ledger
    ledger_df = load_ledger(LEDGER_PATH)

    # Pull broker state
    try:
        acct = broker.get_account()
    except Exception:
        acct = {}

    try:
        positions = broker.get_positions()  # list[dict]
    except Exception as e:
        msg = f"Broker positions fetch failed: {e}"
        _write_guard_snapshot(
            blocked=True,
            code="BROKER_POS_FAIL",
            message=msg,
            extra={"phase": phase, "source": source, "tag": tag},
        )
        # ✅ Snapshot POST even on failure
        _maybe_snapshot_equity(
            broker, enabled=snapshot_enabled, phase=f"{phase}_end", source=source, tag=tag
        )
        return ReconcileResult(False, "BROKER_POS_FAIL", msg, {"phase": phase, "source": source})

    try:
        open_orders = broker.list_orders(status="open", nested=True, limit=500)  # list[dict]
        if not isinstance(open_orders, list):
            open_orders = []
    except Exception:
        open_orders = []

    # Diff + classify
    diff_df, summary, should_freeze = _diff_ledger_vs_broker(
        ledger_df=ledger_df,
        broker_positions=positions,
        open_orders=open_orders,
        cfg=cfg,
    )

    # Write report
    if bool(cfg.get("write_report", True)):
        try:
            diff_df2 = diff_df.copy()
            diff_df2.insert(0, "ts_utc", utc_now_iso_z())
            diff_df2.insert(1, "phase", phase)
            diff_df2.insert(2, "source", source)
            if RECONCILE_REPORT_PATH.exists() and RECONCILE_REPORT_PATH.stat().st_size > 0:
                diff_df2.to_csv(RECONCILE_REPORT_PATH, mode="a", header=False, index=False)
            else:
                diff_df2.to_csv(RECONCILE_REPORT_PATH, mode="w", header=True, index=False)
        except Exception:
            pass

    # Overwrite ledger from broker (authoritative)
    if bool(cfg.get("overwrite_ledger_from_broker", True)):
        try:
            open_counts = _open_orders_count_by_symbol(open_orders)
            broker_id = _safe_account_id(acct)
            new_ledger = ledger_from_broker_positions(
                positions,
                broker_account=broker_id,
                open_orders_by_symbol=open_counts,
                source="broker",
            )
            save_ledger(new_ledger, LEDGER_PATH)
        except Exception:
            # If ledger write fails, freeze (state authority cannot be maintained)
            should_freeze = True

    freeze = bool(cfg.get("freeze_on_mismatch", True)) and bool(should_freeze)
    if freeze:
        msg = (
            "RECONCILE FREEZE: mismatches detected "
            f"(missing={summary.get('missing_in_ledger',0)+summary.get('missing_in_broker',0)}, "
            f"qty_mismatch={summary.get('qty_mismatch',0)}, shorts={summary.get('short_positions',0)})."
        )
        _write_guard_snapshot(
            blocked=True,
            code="RECONCILE_FREEZE",
            message=msg,
            extra={
                "phase": phase,
                "source": source,
                "summary": summary,
                "tag": tag,
            },
        )
        # ✅ Snapshot POST even on freeze
        _maybe_snapshot_equity(
            broker, enabled=snapshot_enabled, phase=f"{phase}_end", source=source, tag=tag
        )
        return ReconcileResult(
            False, "RECONCILE_FREEZE", msg, {"phase": phase, "source": source, "summary": summary}
        )

    ok_msg = f"Reconcile OK: symbols={summary.get('symbols')} mismatches=0 (tol={summary.get('qty_tolerance')})."
    _write_guard_snapshot(
        blocked=False,
        code="RECONCILE_OK",
        message=ok_msg,
        extra={"phase": phase, "source": source, "summary": summary, "tag": tag},
    )

    # ✅ Snapshot POST
    _maybe_snapshot_equity(
        broker, enabled=snapshot_enabled, phase=f"{phase}_end", source=source, tag=tag
    )
    return ReconcileResult(
        True, "RECONCILE_OK", ok_msg, {"phase": phase, "source": source, "summary": summary}
    )


def reconcile_or_freeze(
    broker: AlpacaBroker,
    *,
    phase: str,
    source: str,
    hard_stop: bool = True,
) -> ReconcileResult:
    """
    Convenience wrapper:
      - Runs reconcile_state
      - If frozen and hard_stop=True -> raise RuntimeError to stop executors
    """
    res = reconcile_state(broker, phase=phase, source=source)
    if (not res.ok) and hard_stop:
        raise RuntimeError(res.message)
    return res
