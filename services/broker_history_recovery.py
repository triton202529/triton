"""
TRITON Phase 165B — Broker History Recovery & Ledger Healing.

Recovers broker-side order/fill truth (read-only API calls), compares it to
local execution records, reconstructs positions, breaks down the P&L
reconciliation residual, and scores ledger integrity.

AUDIT ONLY:
  - No orders are placed, modified, or canceled.
  - No trading / execution / lifecycle logic is touched.
  - Historical records are never rewritten; findings are reported as new
    artifacts for human review.

Outputs (data/results/):
    recovered_fills.csv
    recovered_positions.csv
    ledger_healing_report.csv
    position_reconciliation_summary.json
    residual_breakdown.csv
    ledger_integrity_report.json

Run:
    python -m services.broker_history_recovery
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.pnl_ledger_builder import (
    _build_lifecycle_map,
    _classify_exit_reason,
    _load_fill_ledger,
    _load_price_series,
    _norm_sym,
    _safe_read_csv,
    _to_float,
)

RESULTS = ROOT / "data" / "results"

OUT_RECOVERED_FILLS = RESULTS / "recovered_fills.csv"
OUT_RECOVERED_POSITIONS = RESULTS / "recovered_positions.csv"
OUT_HEALING_REPORT = RESULTS / "ledger_healing_report.csv"
OUT_POSITION_RECON = RESULTS / "position_reconciliation_summary.json"
OUT_RESIDUAL_BREAKDOWN = RESULTS / "residual_breakdown.csv"
OUT_INTEGRITY_REPORT = RESULTS / "ledger_integrity_report.json"

MAX_ACTIVITY_PAGES = 100
MAX_ORDER_PAGES = 60


def _log(msg: str) -> None:
    print(f"[broker_history_recovery] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(x: Any) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce", utc=True)


# ──────────────────────────────────────────────
# Broker pull (read-only; best-effort)
# ──────────────────────────────────────────────
def fetch_broker_history() -> Dict[str, Any]:
    """
    Pull broker truth using read-only endpoints:
      - /v2/account            (cash, equity, created_at)
      - /v2/account/activities (per-fill records — gold source)
      - /v2/orders?status=closed (order-level history, secondary)
      - /v2/positions          (current open positions)

    Returns {} values when the broker is unavailable so callers can fall back
    to local snapshots without crashing.
    """
    out: Dict[str, Any] = {
        "available": False,
        "account": {},
        "activities": [],
        "closed_orders": [],
        "positions": [],
        "error": "",
    }
    try:
        from services.broker_alpaca import AlpacaBroker

        broker = AlpacaBroker(mode="paper")
    except Exception as e:
        out["error"] = f"broker client unavailable: {type(e).__name__}: {e}"
        _log(out["error"])
        return out

    try:
        out["account"] = broker.get_account() or {}
    except Exception as e:
        out["error"] = f"account fetch failed: {type(e).__name__}: {e}"
        _log(out["error"])
        return out

    # FILL activities — paginated descending by activity id.
    activities: List[Dict[str, Any]] = []
    token: Optional[str] = None
    try:
        for _ in range(MAX_ACTIVITY_PAGES):
            params: Dict[str, Any] = {
                "activity_types": "FILL",
                "page_size": 100,
                "direction": "desc",
            }
            if token:
                params["page_token"] = token
            page = broker._get("/v2/account/activities", params=params)
            if not page:
                break
            activities.extend(page)
            token = page[-1].get("id")
            if len(page) < 100:
                break
    except Exception as e:
        _log(f"activities pagination stopped: {type(e).__name__}: {e}")
    out["activities"] = activities

    # Closed orders — paginated backwards via `until` cursor.
    closed: List[Dict[str, Any]] = []
    until: Optional[str] = None
    try:
        for _ in range(MAX_ORDER_PAGES):
            page = broker.list_orders(status="closed", limit=500, direction="desc", until=until)
            if not page:
                break
            closed.extend(page)
            until = page[-1].get("submitted_at") or page[-1].get("created_at")
            if len(page) < 500 or not until:
                break
    except Exception as e:
        _log(f"closed-order pagination stopped: {type(e).__name__}: {e}")
    out["closed_orders"] = closed

    try:
        out["positions"] = broker.get_positions() or []
    except Exception as e:
        _log(f"positions fetch failed: {type(e).__name__}: {e}")

    out["available"] = True
    _log(
        f"broker pull OK: {len(activities)} fill activities, "
        f"{len(closed)} closed orders, {len(out['positions'])} open positions"
    )
    return out


# ──────────────────────────────────────────────
# Canonical recovered fills
# ──────────────────────────────────────────────
def build_recovered_fills(broker: Dict[str, Any], local_fills: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical fill record. Primary source: broker FILL activities (one row per
    fill event, incremental qty). Fallback: local recent_orders/live_orders_log
    when the broker is unreachable. Session tags are joined from local records
    by order_id (the broker does not store TRITON session names).
    """
    cols = [
        "timestamp",
        "symbol",
        "side",
        "qty",
        "price",
        "order_id",
        "fill_id",
        "source",
        "session",
        "status",
    ]

    session_by_order: Dict[str, str] = {}
    if not local_fills.empty:
        for _, r in local_fills.iterrows():
            oid = str(r.get("order_id") or "")
            sess = str(r.get("session") or "")
            if oid and sess:
                session_by_order.setdefault(oid, sess)

    rows: List[Dict[str, Any]] = []
    for a in broker.get("activities") or []:
        sym = _norm_sym(a.get("symbol"))
        qty = _to_float(a.get("qty"))
        px = _to_float(a.get("price"))
        if not sym or not np.isfinite(qty) or not np.isfinite(px):
            continue
        oid = str(a.get("order_id") or "")
        rows.append(
            {
                "timestamp": a.get("transaction_time"),
                "symbol": sym,
                "side": str(a.get("side") or "").lower(),
                "qty": qty,
                "price": px,
                "order_id": oid,
                "fill_id": str(a.get("id") or ""),
                "source": "broker_activities",
                "session": session_by_order.get(oid, ""),
                "status": str(a.get("type") or "fill"),
            }
        )

    if not rows and not local_fills.empty:
        # Broker unavailable — degrade to the local fill ledger (clearly labeled).
        for _, r in local_fills.iterrows():
            rows.append(
                {
                    "timestamp": r.get("ts"),
                    "symbol": r.get("symbol"),
                    "side": r.get("side"),
                    "qty": r.get("qty"),
                    "price": r.get("price"),
                    "order_id": r.get("order_id"),
                    "fill_id": "",
                    "source": f"local_fallback:{r.get('source')}",
                    "session": r.get("session"),
                    "status": "filled",
                }
            )

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["timestamp"] = _parse_utc(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# Position reconstruction (FIFO over recovered fills)
# ──────────────────────────────────────────────
def reconstruct_positions(
    recovered: pd.DataFrame, broker_positions: List[Dict[str, Any]]
) -> pd.DataFrame:
    cols = [
        "symbol",
        "entry_date",
        "entry_price",
        "adds",
        "trims",
        "exits",
        "net_qty",
        "avg_cost",
        "market_value",
        "realized_pnl",
        "unrealized_pnl",
        "status",
    ]
    if recovered.empty:
        return pd.DataFrame(columns=cols)

    broker_px: Dict[str, float] = {}
    for p in broker_positions or []:
        broker_px[_norm_sym(p.get("symbol"))] = _to_float(p.get("current_price"))

    out_rows: List[Dict[str, Any]] = []
    for sym, grp in recovered.groupby("symbol"):
        grp = grp.sort_values("timestamp")
        lots: deque = deque()  # [qty, price, ts]
        realized = 0.0
        adds = trims = exits = 0
        first_entry_ts = None
        for _, r in grp.iterrows():
            q = float(r["qty"])
            px = float(r["price"])
            if r["side"] == "buy":
                if not lots:
                    first_entry_ts = r["timestamp"]
                else:
                    adds += 1
                lots.append([q, px, r["timestamp"]])
            else:
                left = q
                while left > 1e-9 and lots:
                    lot = lots[0]
                    take = min(left, lot[0])
                    realized += (px - lot[1]) * take
                    lot[0] -= take
                    left -= take
                    if lot[0] <= 1e-9:
                        lots.popleft()
                if lots:
                    trims += 1
                else:
                    exits += 1

        net_qty = sum(l[0] for l in lots)
        cost = sum(l[0] * l[1] for l in lots)
        avg_cost = cost / net_qty if net_qty > 1e-9 else np.nan
        cur_px = broker_px.get(sym, np.nan)
        if not np.isfinite(cur_px):
            prices = _load_price_series(sym)
            if not prices.empty:
                cur_px = float(prices["close"].iloc[-1])
        mv = net_qty * cur_px if net_qty > 1e-9 and np.isfinite(cur_px) else 0.0
        upl = (cur_px - avg_cost) * net_qty if net_qty > 1e-9 and np.isfinite(cur_px) else 0.0
        out_rows.append(
            {
                "symbol": sym,
                "entry_date": (lots[0][2] if lots else first_entry_ts),
                "entry_price": (lots[0][1] if lots else np.nan),
                "adds": adds,
                "trims": trims,
                "exits": exits,
                "net_qty": net_qty,
                "avg_cost": avg_cost,
                "market_value": mv,
                "realized_pnl": realized,
                "unrealized_pnl": upl,
                "status": "OPEN" if net_qty > 1e-9 else "FLAT",
            }
        )

    return pd.DataFrame(out_rows, columns=cols).sort_values("symbol").reset_index(drop=True)


# ──────────────────────────────────────────────
# Fill matching audit / healing report
# ──────────────────────────────────────────────
def _verify_orders_missing(order_ids: List[str], cap: int = 25) -> Dict[str, bool]:
    """
    Read-only per-order verification: True means the broker returned 404
    (order does not exist in the current account). Capped to avoid API abuse.
    """
    out: Dict[str, bool] = {}
    if not order_ids:
        return out
    try:
        from services.broker_alpaca import AlpacaBroker

        broker = AlpacaBroker(mode="paper")
    except Exception:
        return out
    for oid in order_ids[:cap]:
        if not oid:
            continue
        try:
            broker.get_order(oid)
            out[oid] = False
        except Exception as e:
            out[oid] = "404" in str(e) or "not found" in str(e).lower()
    return out


def _order_level(df: pd.DataFrame, qty_col: str, px_col: str) -> pd.DataFrame:
    """Aggregate fills to order level: total qty, vwap price, side, symbol."""
    if df.empty:
        return pd.DataFrame(columns=["order_id", "symbol", "side", "qty", "vwap", "notional"])
    g = df[df["order_id"].astype(bool)].copy()
    if g.empty:
        return pd.DataFrame(columns=["order_id", "symbol", "side", "qty", "vwap", "notional"])
    g["notional"] = g[qty_col] * g[px_col]
    agg = g.groupby("order_id").agg(
        symbol=("symbol", "first"),
        side=("side", "first"),
        qty=(qty_col, "sum"),
        notional=("notional", "sum"),
    )
    agg["vwap"] = agg["notional"] / agg["qty"].replace(0, np.nan)
    return agg.reset_index()


def build_healing_report(
    local_fills: pd.DataFrame,
    recovered: pd.DataFrame,
    account_created: pd.Timestamp,
    broker_positions: List[Dict[str, Any]],
) -> pd.DataFrame:
    cols = [
        "symbol",
        "issue_type",
        "local_record",
        "broker_record",
        "dollar_impact",
        "severity",
        "recommended_fix",
    ]
    rows: List[Dict[str, Any]] = []

    broker_is_live = not recovered.empty and recovered["source"].iloc[0] == "broker_activities"
    local_orders = _order_level(local_fills, "qty", "price")
    broker_orders = _order_level(recovered, "qty", "price") if broker_is_live else pd.DataFrame()

    def _sev(impact: float) -> str:
        a = abs(impact)
        if a >= 5000:
            return "CRITICAL"
        if a >= 1000:
            return "HIGH"
        if a >= 50:
            return "WARNING"
        return "INFO"

    # Epoch split: local fills that predate the current broker account.
    if not local_fills.empty and not pd.isna(account_created):
        pre = local_fills[local_fills["ts"] < account_created]
        for sym, grp in pre.groupby("symbol"):
            buys = float(grp.loc[grp["side"] == "buy", "notional"].sum())
            sells = float(grp.loc[grp["side"] == "sell", "notional"].sum())
            impact = buys - sells
            rows.append(
                {
                    "symbol": sym,
                    "issue_type": "ORPHAN_FILL_PREVIOUS_ACCOUNT",
                    "local_record": f"{len(grp)} fills before {account_created.date()} "
                    f"(buys ${buys:,.2f} / sells ${sells:,.2f})",
                    "broker_record": "account did not exist (paper account reset)",
                    "dollar_impact": impact,
                    "severity": _sev(impact),
                    "recommended_fix": "MANUAL_REVIEW",
                }
            )

    if broker_is_live and not broker_orders.empty:
        local_ids = set(local_orders["order_id"]) if not local_orders.empty else set()
        broker_ids = set(broker_orders["order_id"])

        # Broker fills missing from local records.
        for _, r in broker_orders[~broker_orders["order_id"].isin(local_ids)].iterrows():
            impact = float(r["notional"])
            rows.append(
                {
                    "symbol": r["symbol"],
                    "issue_type": ("MISSING_BUY" if r["side"] == "buy" else "MISSING_SELL"),
                    "local_record": "absent",
                    "broker_record": f"order {r['order_id'][:13]}… {r['side']} "
                    f"{r['qty']:g} @ {r['vwap']:.2f}",
                    "dollar_impact": impact if r["side"] == "buy" else -impact,
                    "severity": _sev(impact),
                    "recommended_fix": "IMPORT_MISSING_FILL",
                }
            )

        # Local fills (current epoch) absent at the broker. Verify each order
        # directly (read-only GET); a 404 proves the order belongs to a wiped
        # account epoch (paper reset) rather than a pagination gap.
        cur_local = local_orders.copy()
        if not local_fills.empty and not pd.isna(account_created):
            cur_ids = set(
                local_fills.loc[local_fills["ts"] >= account_created, "order_id"].astype(str)
            )
            cur_local = cur_local[cur_local["order_id"].isin(cur_ids)]
        orphans = cur_local[~cur_local["order_id"].isin(broker_ids)]
        verified_404 = _verify_orders_missing(list(orphans["order_id"].astype(str)))
        for _, r in orphans.iterrows():
            impact = float(r["notional"])
            oid = str(r["order_id"])
            confirmed = verified_404.get(oid)
            broker_note = (
                "verified 404 at broker — order erased by account reset"
                if confirmed
                else "not found in recovered broker history"
            )
            rows.append(
                {
                    "symbol": r["symbol"],
                    "issue_type": "ORPHAN_FILL",
                    "local_record": f"order {oid[:13]}… {r['side']} "
                    f"{r['qty']:g} @ {r['vwap']:.2f}",
                    "broker_record": broker_note,
                    "dollar_impact": impact if r["side"] == "buy" else -impact,
                    "severity": _sev(impact),
                    "recommended_fix": "MANUAL_REVIEW",
                }
            )

        # Quantity mismatches on shared orders (partial-fill divergence).
        shared = local_orders.merge(broker_orders, on="order_id", suffixes=("_local", "_broker"))
        for _, r in shared.iterrows():
            dq = float(r["qty_local"]) - float(r["qty_broker"])
            if abs(dq) < 1e-6:
                continue
            impact = dq * float(r["vwap_broker"] if np.isfinite(r["vwap_broker"]) else 0.0)
            rows.append(
                {
                    "symbol": r["symbol_broker"],
                    "issue_type": "PARTIAL_FILL_MISMATCH",
                    "local_record": f"qty {r['qty_local']:g}",
                    "broker_record": f"qty {r['qty_broker']:g}",
                    "dollar_impact": impact,
                    "severity": _sev(impact),
                    "recommended_fix": ("IMPORT_MISSING_FILL" if dq < 0 else "MERGE_DUPLICATE"),
                }
            )

    # Duplicate local rows (same order_id appearing with conflicting fills).
    if not local_fills.empty:
        with_id = local_fills[local_fills["order_id"].astype(bool)]
        dupes = with_id[with_id.duplicated(subset=["order_id"], keep=False)]
        for oid, grp in dupes.groupby("order_id"):
            if grp["qty"].nunique() <= 1:
                continue
            side = str(grp["side"].iloc[0])
            impact = float((grp["qty"].max() - grp["qty"].min()) * grp["price"].mean())
            rows.append(
                {
                    "symbol": grp["symbol"].iloc[0],
                    "issue_type": ("DUPLICATE_BUY" if side == "buy" else "DUPLICATE_SELL"),
                    "local_record": f"order {str(oid)[:13]}… {len(grp)} rows, "
                    f"qty {grp['qty'].min():g}–{grp['qty'].max():g}",
                    "broker_record": "single order at broker",
                    "dollar_impact": impact,
                    "severity": _sev(impact),
                    "recommended_fix": ("IGNORE_ZERO_IMPACT" if impact < 1 else "MERGE_DUPLICATE"),
                }
            )

    # Phantom open lots: local FIFO inventory vs broker positions.
    if not local_fills.empty:
        broker_qty = {
            _norm_sym(p.get("symbol")): _to_float(p.get("qty"), 0.0)
            for p in (broker_positions or [])
        }
        lots: Dict[str, deque] = defaultdict(deque)
        unmatched_sells: Dict[str, float] = defaultdict(float)
        for _, r in local_fills.sort_values("ts").iterrows():
            sym, q, px = str(r["symbol"]), float(r["qty"]), float(r["price"])
            if r["side"] == "buy":
                lots[sym].append([q, px])
            else:
                left = q
                while left > 1e-9 and lots[sym]:
                    lot = lots[sym][0]
                    take = min(left, lot[0])
                    lot[0] -= take
                    left -= take
                    if lot[0] <= 1e-9:
                        lots[sym].popleft()
                if left > 1e-9:
                    unmatched_sells[sym] += left * px

        for sym, dq in lots.items():
            local_open = sum(q for q, _ in dq)
            cost = sum(q * p for q, p in dq)
            broker_open = broker_qty.get(sym, 0.0)
            if abs(local_open - broker_open) < 1e-6 or cost < 1:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "issue_type": "PHANTOM_OPEN_LOT",
                    "local_record": f"FIFO open {local_open:g} sh (cost ${cost:,.2f})",
                    "broker_record": f"broker holds {broker_open:g} sh",
                    "dollar_impact": cost,
                    "severity": _sev(cost),
                    "recommended_fix": "RECONSTRUCT_CLOSE",
                }
            )

        for sym, notional in unmatched_sells.items():
            rows.append(
                {
                    "symbol": sym,
                    "issue_type": "UNMATCHED_SELL",
                    "local_record": f"sell fills without prior local buys (${notional:,.2f})",
                    "broker_record": "buy side missing from local ledger",
                    "dollar_impact": -notional,
                    "severity": _sev(notional),
                    "recommended_fix": "IMPORT_MISSING_FILL",
                }
            )

    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values("dollar_impact", key=lambda s: s.abs(), ascending=False)
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# Position reconciliation summary
# ──────────────────────────────────────────────
def build_position_reconciliation(recon_positions: pd.DataFrame) -> Dict[str, Any]:
    snap = _safe_read_csv(RESULTS / "positions_snapshot.csv")
    summary: Dict[str, Any] = {
        "generated_at": _now_iso(),
        "symbols_checked": 0,
        "perfect_matches": 0,
        "quantity_mismatches": 0,
        "cost_basis_mismatches": 0,
        "missing_positions": 0,
        "phantom_positions": 0,
        "details": [],
    }

    snap_map: Dict[str, Dict[str, float]] = {}
    if not snap.empty:
        snap.columns = [str(c).strip().lower() for c in snap.columns]
        sym_col = "ticker" if "ticker" in snap.columns else "symbol"
        for _, r in snap.iterrows():
            sym = _norm_sym(r.get(sym_col))
            if sym:
                snap_map[sym] = {
                    "qty": _to_float(r.get("qty"), 0.0),
                    "cost_basis": _to_float(r.get("cost_basis"), np.nan),
                }

    open_recon = (
        recon_positions[recon_positions["status"] == "OPEN"]
        if not recon_positions.empty
        else pd.DataFrame()
    )
    recon_map: Dict[str, Dict[str, float]] = {}
    if not open_recon.empty:
        for _, r in open_recon.iterrows():
            recon_map[str(r["symbol"])] = {
                "qty": _to_float(r.get("net_qty"), 0.0),
                "cost_basis": _to_float(r.get("net_qty"), 0.0)
                * _to_float(r.get("avg_cost"), np.nan),
            }

    all_syms = sorted(set(snap_map) | set(recon_map))
    summary["symbols_checked"] = len(all_syms)
    for sym in all_syms:
        s = snap_map.get(sym)
        r = recon_map.get(sym)
        if s and not r:
            summary["missing_positions"] += 1
            summary["details"].append(
                {"symbol": sym, "issue": "in snapshot, not in recovered fills"}
            )
            continue
        if r and not s:
            summary["phantom_positions"] += 1
            summary["details"].append(
                {"symbol": sym, "issue": "reconstructed open, not in snapshot"}
            )
            continue
        qty_ok = abs(s["qty"] - r["qty"]) < 1e-6
        cb_s, cb_r = s.get("cost_basis"), r.get("cost_basis")
        cb_ok = (
            np.isfinite(cb_s)
            and np.isfinite(cb_r)
            and abs(cb_s - cb_r) < max(1.0, 0.01 * abs(cb_s))
        )
        if qty_ok and cb_ok:
            summary["perfect_matches"] += 1
        else:
            if not qty_ok:
                summary["quantity_mismatches"] += 1
                summary["details"].append(
                    {
                        "symbol": sym,
                        "issue": f"qty snapshot {s['qty']:g} vs recovered {r['qty']:g}",
                    }
                )
            if not cb_ok:
                summary["cost_basis_mismatches"] += 1
                summary["details"].append(
                    {
                        "symbol": sym,
                        "issue": f"cost basis snapshot {cb_s} vs recovered {cb_r}",
                    }
                )
    return summary


# ──────────────────────────────────────────────
# Residual breakdown
# ──────────────────────────────────────────────
def build_residual_breakdown(
    local_fills: pd.DataFrame,
    healing: pd.DataFrame,
    account_created: pd.Timestamp,
) -> pd.DataFrame:
    cols = [
        "symbol",
        "date",
        "session",
        "strategy",
        "lifecycle_action",
        "issue_type",
        "dollar_impact",
        "is_root_cause",
    ]
    if healing.empty:
        return pd.DataFrame(columns=cols)

    # Phantom open lots / duplicates are downstream symptoms of the same missing
    # dollars as orphan/missing fills — flag root causes so sums don't double count.
    root_cause_types = {
        "ORPHAN_FILL_PREVIOUS_ACCOUNT",
        "ORPHAN_FILL",
        "UNMATCHED_SELL",
        "MISSING_BUY",
        "MISSING_SELL",
        "PARTIAL_FILL_MISMATCH",
    }

    lifecycle_map = _build_lifecycle_map()

    # Per-symbol session/date context from the local fill ledger.
    ctx: Dict[str, Dict[str, str]] = {}
    if not local_fills.empty:
        for sym, grp in local_fills.groupby("symbol"):
            grp = grp.sort_values("ts")
            last = grp.iloc[-1]
            ts = last["ts"]
            ctx[str(sym)] = {
                "date": ts.strftime("%Y-%m-%d") if not pd.isna(ts) else "",
                "session": str(last.get("session") or ""),
            }

    rows: List[Dict[str, Any]] = []
    for _, r in healing.iterrows():
        sym = str(r["symbol"])
        c = ctx.get(sym, {})
        sess = c.get("session", "")
        rows.append(
            {
                "symbol": sym,
                "date": c.get("date", ""),
                "session": sess,
                "strategy": _classify_exit_reason(sess) if sess else "—",
                "lifecycle_action": lifecycle_map.get(sym, "—"),
                "issue_type": r["issue_type"],
                "dollar_impact": _to_float(r["dollar_impact"], 0.0),
                "is_root_cause": str(r["issue_type"]) in root_cause_types,
            }
        )
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values("dollar_impact", key=lambda s: s.abs(), ascending=False).reset_index(
        drop=True
    )


# ──────────────────────────────────────────────
# Integrity report
# ──────────────────────────────────────────────
def build_integrity_report(
    broker: Dict[str, Any],
    local_fills: pd.DataFrame,
    recovered: pd.DataFrame,
    recon_positions: pd.DataFrame,
    pos_recon: Dict[str, Any],
    healing: pd.DataFrame,
    account_created: pd.Timestamp,
) -> Dict[str, Any]:
    reasons: List[str] = []
    broker_live = bool(broker.get("available")) and not recovered.empty

    # ---- Fill match: share of broker fill notional present in local records.
    fill_match = np.nan
    if broker_live:
        broker_orders = _order_level(recovered, "qty", "price")
        local_ids = set(local_fills["order_id"].astype(str)) if not local_fills.empty else set()
        total = float(broker_orders["notional"].sum())
        matched = float(
            broker_orders.loc[broker_orders["order_id"].isin(local_ids), "notional"].sum()
        )
        fill_match = (matched / total * 100.0) if total > 0 else 100.0
        if fill_match < 99.5:
            reasons.append(
                f"{100 - fill_match:.1f}% of broker fill notional is missing from local records"
            )

    # ---- Position match: recovered FIFO vs broker live positions.
    position_match = np.nan
    if broker_live:
        broker_pos = {
            _norm_sym(p.get("symbol")): _to_float(p.get("qty"), 0.0)
            for p in (broker.get("positions") or [])
        }
        recon_open = {
            str(r["symbol"]): _to_float(r["net_qty"], 0.0)
            for _, r in recon_positions.iterrows()
            if r["status"] == "OPEN"
        }
        syms = set(broker_pos) | set(recon_open)
        if syms:
            ok = sum(1 for s in syms if abs(broker_pos.get(s, 0.0) - recon_open.get(s, 0.0)) < 1e-6)
            position_match = ok / len(syms) * 100.0
        else:
            position_match = 100.0
        if position_match < 100.0:
            reasons.append("recovered FIFO positions do not fully match live broker positions")

    # ---- Cash match: reconstructed cash from recovered fills vs broker cash.
    cash_match = np.nan
    residual_difference = np.nan
    identity_ok = False
    if broker_live:
        acct = broker.get("account") or {}
        broker_cash = _to_float(acct.get("cash"))
        broker_equity = _to_float(acct.get("equity"))
        start_equity = 100000.0  # paper account default at reset
        buys = float(
            recovered.loc[recovered["side"] == "buy", "qty"]
            .mul(recovered.loc[recovered["side"] == "buy", "price"])
            .sum()
        )
        sells = float(
            recovered.loc[recovered["side"] == "sell", "qty"]
            .mul(recovered.loc[recovered["side"] == "sell", "price"])
            .sum()
        )
        recon_cash = start_equity - buys + sells
        if np.isfinite(broker_cash):
            residual_difference = recon_cash - broker_cash
            rel = abs(residual_difference) / max(abs(broker_cash), 1.0)
            cash_match = max(0.0, (1.0 - min(rel, 1.0))) * 100.0
            identity_ok = abs(residual_difference) < 50.0
            if not identity_ok:
                reasons.append(
                    f"reconstructed cash differs from broker cash by ${residual_difference:,.2f}"
                )

    # ---- P&L match: FIFO realized (recovered fills) vs identity-implied realized.
    pnl_match = np.nan
    if broker_live and not recon_positions.empty:
        fifo_realized = float(pd.to_numeric(recon_positions["realized_pnl"], errors="coerce").sum())
        open_cost = float(
            (
                pd.to_numeric(recon_positions["net_qty"], errors="coerce")
                * pd.to_numeric(recon_positions["avg_cost"], errors="coerce")
            )
            .fillna(0.0)
            .sum()
        )
        buys = float(
            recovered.loc[recovered["side"] == "buy", "qty"]
            .mul(recovered.loc[recovered["side"] == "buy", "price"])
            .sum()
        )
        sells = float(
            recovered.loc[recovered["side"] == "sell", "qty"]
            .mul(recovered.loc[recovered["side"] == "sell", "price"])
            .sum()
        )
        identity_realized = open_cost - (buys - sells)
        gap = abs(fifo_realized - identity_realized)
        pnl_match = max(0.0, 1.0 - min(gap / max(abs(fifo_realized), 100.0), 1.0)) * 100.0
        if gap > 5.0:
            reasons.append(
                f"FIFO realized (${fifo_realized:,.2f}) vs identity realized "
                f"(${identity_realized:,.2f}) differ by ${gap:,.2f}"
            )

    # ---- Trade match: current-epoch local exits matched to broker sell fills.
    trade_match = np.nan
    if broker_live and not local_fills.empty:
        cur = local_fills
        if not pd.isna(account_created):
            cur = local_fills[local_fills["ts"] >= account_created]
        local_sells = cur[cur["side"] == "sell"]
        broker_sell_ids = set(recovered.loc[recovered["side"] == "sell", "order_id"].astype(str))
        if len(local_sells):
            matched = local_sells["order_id"].astype(str).isin(broker_sell_ids).mean()
            trade_match = float(matched * 100.0)
        else:
            trade_match = 100.0
        if trade_match < 99.5:
            reasons.append("some local sell records have no matching broker fill")

    # ---- Epoch disclosure.
    pre_epoch_notional = 0.0
    if not local_fills.empty and not pd.isna(account_created):
        pre = local_fills[local_fills["ts"] < account_created]
        if len(pre):
            pre_buys = float(pre.loc[pre["side"] == "buy", "notional"].sum())
            pre_sells = float(pre.loc[pre["side"] == "sell", "notional"].sum())
            pre_epoch_notional = pre_buys - pre_sells
            reasons.append(
                f"{len(pre)} local fills predate the current paper account "
                f"(created {account_created.date()}); net ${pre_epoch_notional:,.2f} of "
                f"flow belongs to a previous account epoch and cannot be certified "
                f"against current broker history"
            )

    comps = {
        "position_match": position_match,
        "fill_match": fill_match,
        "cash_match": cash_match,
        "pnl_match": pnl_match,
        "trade_match": trade_match,
    }
    finite = [v for v in comps.values() if np.isfinite(_to_float(v, np.nan))]
    score = float(np.mean(finite)) if finite else 0.0

    if not broker_live:
        status = "NOT_CERTIFIED"
        reasons.append("broker history unavailable — certification requires broker truth")
    elif score >= 95.0 and identity_ok:
        status = "CERTIFIED"
    elif score >= 70.0:
        status = "PARTIALLY_CERTIFIED"
    else:
        status = "NOT_CERTIFIED"

    n_critical = int((healing["severity"] == "CRITICAL").sum()) if not healing.empty else 0
    n_high = int((healing["severity"] == "HIGH").sum()) if not healing.empty else 0

    return {
        "generated_at": _now_iso(),
        "broker_available": broker_live,
        "account_created": (str(account_created) if not pd.isna(account_created) else ""),
        "ledger_integrity_score": round(score, 2),
        "position_match": _round_or_none(position_match),
        "fill_match": _round_or_none(fill_match),
        "cash_match": _round_or_none(cash_match),
        "pnl_match": _round_or_none(pnl_match),
        "trade_match": _round_or_none(trade_match),
        "residual_difference": _round_or_none(residual_difference),
        "previous_epoch_net_flow": round(pre_epoch_notional, 2),
        "identity_ok": bool(identity_ok),
        "status": status,
        "healing_issues_critical": n_critical,
        "healing_issues_high": n_high,
        "reasons": reasons,
    }


def _round_or_none(v: Any) -> Optional[float]:
    f = _to_float(v, np.nan)
    return round(f, 2) if np.isfinite(f) else None


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)

    local_fills = _load_fill_ledger()
    _log(f"local fill ledger: {len(local_fills)} rows")

    broker = fetch_broker_history()
    account_created = _parse_utc((broker.get("account") or {}).get("created_at"))

    recovered = build_recovered_fills(broker, local_fills)
    recovered_out = recovered.copy()
    if not recovered_out.empty:
        recovered_out["timestamp"] = recovered_out["timestamp"].astype(str)
    recovered_out.to_csv(OUT_RECOVERED_FILLS, index=False, encoding="utf-8")
    _log(f"recovered_fills.csv: {len(recovered)} fills")

    recon_positions = reconstruct_positions(recovered, broker.get("positions") or [])
    rp_out = recon_positions.copy()
    if not rp_out.empty:
        rp_out["entry_date"] = rp_out["entry_date"].astype(str)
    rp_out.to_csv(OUT_RECOVERED_POSITIONS, index=False, encoding="utf-8")
    _log(f"recovered_positions.csv: {len(recon_positions)} symbols")

    healing = build_healing_report(
        local_fills, recovered, account_created, broker.get("positions") or []
    )
    healing.to_csv(OUT_HEALING_REPORT, index=False, encoding="utf-8")
    _log(f"ledger_healing_report.csv: {len(healing)} issues")

    pos_recon = build_position_reconciliation(recon_positions)
    OUT_POSITION_RECON.write_text(json.dumps(pos_recon, indent=2, default=str), encoding="utf-8")
    _log("position_reconciliation_summary.json written")

    residual = build_residual_breakdown(local_fills, healing, account_created)
    residual.to_csv(OUT_RESIDUAL_BREAKDOWN, index=False, encoding="utf-8")
    _log(f"residual_breakdown.csv: {len(residual)} rows")

    integrity = build_integrity_report(
        broker,
        local_fills,
        recovered,
        recon_positions,
        pos_recon,
        healing,
        account_created,
    )
    OUT_INTEGRITY_REPORT.write_text(json.dumps(integrity, indent=2, default=str), encoding="utf-8")
    _log(
        f"ledger_integrity_report.json: score {integrity['ledger_integrity_score']} "
        f"status {integrity['status']}"
    )
    return integrity


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, default=str))
