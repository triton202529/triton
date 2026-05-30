# services/build_trade_outcomes.py
"""
Standalone PnL / trade-outcome analytics from local Triton result CSVs only.
No broker calls; no trading-path imports.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

# --- Input files (if present) ---
PATH_RECENT_ORDERS = RESULTS / "recent_orders.csv"
PATH_OPEN_ORDERS_SNAP = RESULTS / "open_orders_snapshot.csv"
PATH_POSITIONS_SNAP = RESULTS / "positions_snapshot.csv"
PATH_LIVE_ORDERS_LOG = RESULTS / "live_orders_log.csv"
PATH_TRADE_OPPS = RESULTS / "trade_opportunities.csv"
PATH_LIFECYCLE = RESULTS / "signal_lifecycle_effective.csv"

# --- Output files ---
OUT_TRADE_OUTCOMES = RESULTS / "trade_outcomes.csv"
OUT_SUMMARY = RESULTS / "trade_outcomes_summary.json"
OUT_BY_SYMBOL = RESULTS / "trade_outcomes_by_symbol.csv"

LIVE_LOG_EXPECTED = (
    "timestamp",
    "session",
    "action",
    "symbol",
    "side",
    "qty",
    "type",
    "limit_price",
    "order_id",
    "status",
    "filled_qty",
    "filled_avg_price",
    "client_order_id",
    "tp_limit",
    "sl_stop",
)


def _warn(msg: str) -> None:
    print(f"[build_trade_outcomes] {msg}", file=sys.stderr)


def safe_read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        _warn(f"missing or empty: {path.name}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        _warn(f"could not read {label} ({path.name}): {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_read_csv_flex(path: Path, *, label: str) -> pd.DataFrame:
    """Tolerate malformed rows (e.g. legacy log corruption)."""
    if not path.is_file() or path.stat().st_size == 0:
        _warn(f"missing or empty: {path.name}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    except TypeError:
        try:
            df = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=False, engine="python")
        except Exception as e:
            _warn(f"could not read {label} ({path.name}): {e}")
            return pd.DataFrame()
    except Exception as e:
        _warn(f"could not read {label} ({path.name}): {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


def _to_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, str) and not str(x).strip()):
        return None
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _parse_time_ts(val: Any) -> Optional[pd.Timestamp]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nat", "nan", "none"):
        return None
    try:
        t = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(t):
            return None
        return t
    except Exception:
        return None


@dataclass
class _Lot:
    ts: Optional[pd.Timestamp]
    qty: float
    price: float


def _ledger_from_recent_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Map recent_orders -> normalized columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # Aliases
    if "id" in out.columns and "order_id" not in out.columns:
        out["order_id"] = out["id"]
    tcol = None
    for c in ("filled_at", "updated_at", "submitted_at", "created_at", "snapshot_ts"):
        if c in out.columns:
            tcol = c
            break
    out["_event_ts"] = out[tcol] if tcol else np.nan
    if "symbol" not in out.columns and "ticker" in out.columns:
        out["symbol"] = out["ticker"]
    for col in [
        "symbol",
        "side",
        "status",
        "order_id",
        "client_order_id",
        "filled_qty",
        "qty",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    out["source_file"] = "recent_orders.csv"
    out["session"] = out["subtag"].fillna("") if "subtag" in out.columns else ""
    return out


def _ledger_from_live_log(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = set(c.lower() for c in df.columns)
    if not cols.issuperset(set(x.lower() for x in ("symbol", "side", "order_id"))):
        return pd.DataFrame()
    h = [str(c).lower() for c in df.columns]
    if "snapshot_ts" in h and list(df.columns) != list(LIVE_LOG_EXPECTED):
        _warn("live_orders_log.csv: snapshot or unexpected schema; skipping for ledger merge")
        return pd.DataFrame()
    o = df.copy()
    o.columns = [str(c).strip() for c in o.columns]
    for c in LIVE_LOG_EXPECTED:
        if c not in o.columns:
            o[c] = ""
    o["_event_ts"] = o["timestamp"] if "timestamp" in o.columns else np.nan
    o["source_file"] = "live_orders_log.csv"
    o["order_id"] = o.get("order_id", "")
    o["client_order_id"] = o.get("client_order_id", "")
    o["status"] = o.get("status", "")
    o["side"] = o.get("side", "")
    o["symbol"] = o.get("symbol", "")
    o["session"] = o.get("session", "")
    if "qty" in o.columns and "filled_qty" not in o.columns:
        o["filled_qty"] = o["qty"]
    o["filled_avg_price"] = o.get("filled_avg_price", np.nan)
    return o


def _filled_mask(status: Any, filled_q: Any) -> bool:
    st = str(status or "").strip().lower()
    fq = _to_float(filled_q) or 0.0
    return st in ("filled", "partially_filled", "calculated", "closed") and fq > 1e-9


def _dedupe_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    d = ledger.copy()
    d["_event_ts_parsed"] = d["_event_ts"].map(_parse_time_ts)
    d["order_id_s"] = d["order_id"].astype(str).str.strip()
    d["symbol_s"] = d["symbol"].map(_norm_sym)
    # Prefer recent_orders over live log when the same (order, symbol) appears twice.
    d["_src_pri"] = d["source_file"].map(lambda x: 1 if "recent_orders" in str(x) else 0)
    d = d.sort_values(
        ["_event_ts_parsed", "order_id_s", "symbol_s", "_src_pri"],
        ascending=[True, True, True, True],
    )
    d = d.drop_duplicates(subset=["order_id_s", "symbol_s"], keep="last")
    return d


def _build_fill_events(ledger: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    One fill event per ledger row: buy/sell with qty and price, ordered by time.
    """
    rows: List[Dict[str, Any]] = []
    if ledger.empty:
        return rows
    d = ledger.copy()
    d["_t"] = d["_event_ts"].map(_parse_time_ts)
    for _, r in d.iterrows():
        sym = _norm_sym(r.get("symbol"))
        if not sym:
            continue
        if not _filled_mask(r.get("status"), r.get("filled_qty")):
            continue
        fq = _to_float(r.get("filled_qty")) or 0.0
        fpx = _to_float(r.get("filled_avg_price"))
        if fpx is None and "limit_price" in r:
            fpx = _to_float(r.get("limit_price"))
        if fpx is None or fq <= 0:
            continue
        side = str(r.get("side") or "").strip().lower()
        if side not in ("buy", "sell", "b", "s"):
            if "buy" in side:
                side = "buy"
            elif "sell" in side:
                side = "sell"
        if side in ("b", "buy", "long"):
            sdir = "buy"
        elif side in ("s", "sell", "short") or "close" in str(side):
            sdir = "sell"
        else:
            continue
        ord_id = str(r.get("order_id") or "").strip()
        cid = str(r.get("client_order_id") or "").strip()
        src = str(r.get("source_file") or "")
        sess = str(r.get("session") or "")
        stat = str(r.get("status") or "").strip()
        rows.append(
            {
                "timestamp": r.get("_event_ts"),
                "_t": r.get("_t"),
                "symbol": sym,
                "side": sdir,
                "qty": fq,
                "price": float(fpx),
                "order_id": ord_id,
                "client_order_id": cid,
                "status": stat,
                "source_file": src,
                "session": sess,
            }
        )
    # Fix _t from loop
    for row in rows:
        if row.get("_t") is None and row.get("timestamp") is not None:
            row["_t"] = _parse_time_ts(row["timestamp"])
    rows.sort(
        key=lambda x: (x.get("_t") is None, x.get("_t") or pd.Timestamp.min, x.get("order_id", ""))
    )
    return rows


def _fifo_realized(
    events: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[_Lot]]]:
    """
    Per-symbol FIFO: returns realized trade dicts and remaining lots.
    """
    from collections import defaultdict

    by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_sym[e["symbol"]].append(e)
    remain: Dict[str, List[_Lot]] = defaultdict(list)
    out_realized: List[Dict[str, Any]] = []

    for sym, evs in by_sym.items():
        evs = sorted(
            evs,
            key=lambda x: (
                x.get("_t") is None,
                x.get("_t") or pd.Timestamp.min,
                str(x.get("order_id", "")),
            ),
        )
        lotq: List[_Lot] = []
        for e in evs:
            s = e["side"]
            q = float(e["qty"])
            p = float(e["price"])
            t = e.get("_t")
            if s == "buy":
                lotq.append(_Lot(ts=t, qty=q, price=p))
                continue
            if s == "sell":
                sell_left = q
                while sell_left > 1e-9 and lotq:
                    lot = lotq[0]
                    take = min(sell_left, lot.qty)
                    cost = lot.price
                    pl = (p - cost) * take
                    opened = lot.ts
                    closed = t
                    out_realized.append(
                        {
                            "symbol": sym,
                            "outcome_type": "REALIZED",
                            "entry_price": cost,
                            "exit_price": p,
                            "qty": take,
                            "realized_pl": pl,
                            "unrealized_pl": np.nan,
                            "total_pl": pl,
                            "return_pct": (
                                (p - cost) / cost * 100.0 if cost and cost > 0 else np.nan
                            ),
                            "opened_at": opened,
                            "closed_at": closed,
                            "holding_status": "CLOSED",
                            "order_id_sell": e.get("order_id", ""),
                            "order_id": e.get("order_id", ""),
                            "source_file": e.get("source_file", ""),
                            "session": e.get("session", ""),
                            "notes": "FIFO match from order ledger (recent_orders / live_orders_log).",
                        }
                    )
                    lot.qty -= take
                    sell_left -= take
                    if lot.qty <= 1e-9:
                        lotq.pop(0)
                if sell_left > 1e-9:
                    out_realized.append(
                        {
                            "symbol": sym,
                            "outcome_type": "UNKNOWN",
                            "entry_price": np.nan,
                            "exit_price": p,
                            "qty": sell_left,
                            "realized_pl": np.nan,
                            "unrealized_pl": np.nan,
                            "total_pl": np.nan,
                            "return_pct": np.nan,
                            "opened_at": np.nan,
                            "closed_at": t,
                            "holding_status": "CLOSED",
                            "order_id_sell": e.get("order_id", ""),
                            "order_id": e.get("order_id", ""),
                            "source_file": e.get("source_file", ""),
                            "session": e.get("session", ""),
                            "notes": "Sell fill without prior buy qty in local ledger; cost basis not reconstructed.",
                        }
                    )
        remain[sym] = list(lotq)
    return out_realized, dict(remain)


def _position_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    c = {str(x).lower(): x for x in df.columns}
    tc = c.get("ticker") or c.get("symbol")
    if not tc:
        return []
    rows = []
    for _, r in df.iterrows():
        sym = _norm_sym(r.get(tc))
        if not sym:
            continue
        qty = _to_float(r.get("qty") or r.get("qty_available"))
        if qty is None or abs(qty) < 1e-9:
            continue
        avg = _to_float(r.get("avg_entry_price"))
        mkt = _to_float(r.get("current_price") or r.get("last") or r.get("mark"))
        mv = _to_float(r.get("market_value") or r.get("value"))
        upl = _to_float(r.get("unrealized_pl") or r.get("unrealized_pnl"))
        ulpc = _to_float(r.get("unrealized_plpc"))
        ret_pct: Optional[float] = None
        if ulpc is not None:
            ret_pct = ulpc * 100.0 if abs(ulpc) < 1.0 else ulpc
        elif mkt is not None and avg is not None and avg > 0:
            ret_pct = (mkt - avg) / avg * 100.0
        pl_total = upl if upl is not None else np.nan
        rows.append(
            {
                "symbol": sym,
                "outcome_type": "OPEN",
                "entry_price": avg,
                "exit_price": mkt,
                "qty": abs(qty),
                "realized_pl": np.nan,
                "unrealized_pl": upl,
                "total_pl": pl_total,
                "return_pct": ret_pct,
                "opened_at": r.get("snapshot_ts") or r.get("date"),
                "closed_at": "",
                "holding_status": "OPEN",
                "order_id": "",
                "source_file": "positions_snapshot.csv",
                "session": "",
                "market_value": mv,
                "notes": "From positions_snapshot; unrealized fields only when present in file.",
            }
        )
    return rows


def _partial_from_open_orders(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty or len(df.columns) < 2:
        return []
    if str(df.columns[0]).lower() == "snapshot_ts" and len(df) <= 1:
        return []
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        sym = _norm_sym(r.get("symbol") or r.get("ticker"))
        if not sym:
            continue
        st = str(r.get("status") or "").lower()
        fq = _to_float(r.get("filled_qty") or r.get("qty") or 0) or 0.0
        if st and "partial" in st and fq > 0:
            rows.append(
                {
                    "symbol": sym,
                    "outcome_type": "PARTIAL",
                    "entry_price": np.nan,
                    "exit_price": _to_float(r.get("filled_avg_price") or r.get("limit_price")),
                    "qty": fq,
                    "realized_pl": np.nan,
                    "unrealized_pl": np.nan,
                    "total_pl": np.nan,
                    "return_pct": np.nan,
                    "opened_at": r.get("created_at") or r.get("submitted_at"),
                    "closed_at": "",
                    "holding_status": "IN_FLIGHT",
                    "order_id": str(r.get("id") or r.get("order_id") or ""),
                    "source_file": "open_orders_snapshot.csv",
                    "session": "",
                    "notes": "Open order with partial fill; no P/L until closed.",
                }
            )
    return rows


def _merge_confidence(
    by_symbol: Set[str], lc: pd.DataFrame, opps: pd.DataFrame
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sym in by_symbol:
        c = None
        if not lc.empty and "ticker" in lc.columns and "confidence" in lc.columns:
            sub = lc[lc["ticker"].map(_norm_sym) == sym]
            if not sub.empty:
                c = _to_float(sub.iloc[-1].get("confidence"))
        if (
            c is None
            and not opps.empty
            and "ticker" in opps.columns
            and "confidence" in opps.columns
        ):
            sub = opps[opps["ticker"].map(_norm_sym) == sym]
            if not sub.empty:
                c = _to_float(sub.iloc[-1].get("confidence"))
        if c is not None:
            out[sym] = c
    return out


def _cell_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return ""
    if hasattr(x, "isoformat"):
        try:
            return str(x)
        except Exception:
            return ""
    if isinstance(x, (pd.Timestamp, datetime)):
        return str(x)
    return str(x)


def _fmt_out_df(rows: List[Dict[str, Any]], conf: Dict[str, float]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "outcome_type",
                "entry_price",
                "exit_price",
                "qty",
                "realized_pl",
                "unrealized_pl",
                "total_pl",
                "return_pct",
                "opened_at",
                "closed_at",
                "holding_status",
                "source_confidence",
                "notes",
            ]
        )
    rec: List[Dict[str, Any]] = []
    for r in rows:
        sym = r.get("symbol", "")
        rpl = r.get("realized_pl", "")
        upl = r.get("unrealized_pl", "")
        tpl = r.get("total_pl", "")
        rec.append(
            {
                "symbol": sym,
                "outcome_type": r.get("outcome_type", ""),
                "entry_price": r.get("entry_price", ""),
                "exit_price": r.get("exit_price", ""),
                "qty": r.get("qty", ""),
                "realized_pl": "" if (isinstance(rpl, float) and np.isnan(rpl)) else rpl,
                "unrealized_pl": "" if (isinstance(upl, float) and np.isnan(upl)) else upl,
                "total_pl": "" if (isinstance(tpl, float) and np.isnan(tpl)) else tpl,
                "return_pct": r.get("return_pct", ""),
                "opened_at": _cell_str(r.get("opened_at")),
                "closed_at": _cell_str(r.get("closed_at")),
                "holding_status": r.get("holding_status", ""),
                "source_confidence": conf.get(_norm_sym(sym), ""),
                "notes": r.get("notes", ""),
            }
        )
    return pd.DataFrame(rec)


def _safe_float_for_json(x: Any) -> Any:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x


def build() -> int:
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
    RESULTS.mkdir(parents=True, exist_ok=True)

    used: List[str] = []

    r_recent = safe_read_csv(PATH_RECENT_ORDERS, label="recent_orders")
    if not r_recent.empty:
        used.append("recent_orders.csv")
    r_live = safe_read_csv_flex(PATH_LIVE_ORDERS_LOG, label="live_orders_log")
    if not r_live.empty and set(str(c) for c in r_live.columns) != {"snapshot_ts"}:
        used.append("live_orders_log.csv")

    pos = safe_read_csv(PATH_POSITIONS_SNAP, label="positions")
    if not pos.empty:
        used.append("positions_snapshot.csv")
    oopen = safe_read_csv(PATH_OPEN_ORDERS_SNAP, label="open_orders")
    if not oopen.empty and not (str(oopen.columns[0]).lower() == "snapshot_ts" and len(oopen) <= 1):
        used.append("open_orders_snapshot.csv")

    lc = safe_read_csv(PATH_LIFECYCLE, label="lifecycle")
    if not lc.empty:
        used.append("signal_lifecycle_effective.csv")
    opps = safe_read_csv(PATH_TRADE_OPPS, label="opportunities")
    if not opps.empty:
        used.append("trade_opportunities.csv")

    led1 = _ledger_from_recent_orders(r_recent) if not r_recent.empty else pd.DataFrame()
    led2 = _ledger_from_live_log(r_live) if not r_live.empty else pd.DataFrame()
    if not led1.empty and not led2.empty:
        ledger = pd.concat([led1, led2], ignore_index=True, sort=False)
    elif not led1.empty:
        ledger = led1
    else:
        ledger = led2

    ledger = _dedupe_ledger(ledger) if not ledger.empty else ledger
    fill_events = _build_fill_events(ledger)
    realized, _rem = _fifo_realized(fill_events) if fill_events else ([], {})

    open_rows = _position_rows(pos)
    partial = _partial_from_open_orders(oopen)

    all_rows: List[Dict[str, Any]] = (
        [dict(x) for x in realized] + [dict(x) for x in open_rows] + [dict(x) for x in partial]
    )
    sym_set: Set[str] = set(_norm_sym(x.get("symbol", "")) for x in all_rows if x.get("symbol"))
    conf_map = _merge_confidence(sym_set, lc, opps)

    out_main = _fmt_out_df(all_rows, conf_map)
    if not out_main.empty:
        out_main.to_csv(OUT_TRADE_OUTCOMES, index=False, encoding="utf-8")
    else:
        out_main.to_csv(OUT_TRADE_OUTCOMES, index=False, encoding="utf-8")

    # --- by symbol ---
    by_s: Dict[str, Dict[str, Any]] = {}
    for r in all_rows:
        s = _norm_sym(r.get("symbol", ""))
        if not s:
            continue
        b = by_s.setdefault(
            s,
            {
                "symbol": s,
                "realized_pl": 0.0,
                "unrealized_pl": 0.0,
                "total_pl": 0.0,
                "open_qty": 0.0,
                "avg_entry_price": np.nan,
                "market_price": np.nan,
                "trade_rows": 0,
                "win_flag_if_known": "",
            },
        )
        b["trade_rows"] = int(b["trade_rows"]) + 1
        ot = str(r.get("outcome_type", "")).upper()
        rpl = _to_float(r.get("realized_pl"))
        upl = _to_float(r.get("unrealized_pl"))
        if ot == "REALIZED" and rpl is not None and np.isfinite(rpl):
            b["realized_pl"] = float(b["realized_pl"]) + rpl
        if ot == "OPEN" and upl is not None and np.isfinite(upl):
            b["unrealized_pl"] = float(b.get("unrealized_pl", 0.0)) + upl
        if ot == "OPEN":
            b["open_qty"] = _to_float(r.get("qty")) or 0.0
            b["avg_entry_price"] = _to_float(r.get("entry_price"))
            b["market_price"] = _to_float(r.get("exit_price"))

    for b in by_s.values():
        rp = float(b.get("realized_pl", 0.0) or 0.0)
        up = float(b.get("unrealized_pl", 0.0) or 0.0)
        b["total_pl"] = rp + up
        wf = b.get("win_flag_if_known", "")
        if wf != "":
            continue
        tot = b.get("total_pl", 0.0)
        if isinstance(tot, (int, float)) and np.isfinite(float(tot)) and float(tot) != 0.0:
            b["win_flag_if_known"] = 1 if float(tot) > 0 else 0
        elif b.get("open_qty", 0) and b.get("unrealized_pl") is not None:
            u = float(b.get("unrealized_pl", 0.0) or 0.0)
            if u > 0:
                b["win_flag_if_known"] = 1
            elif u < 0:
                b["win_flag_if_known"] = 0

    df_by = (
        pd.DataFrame(list(by_s.values()))
        if by_s
        else pd.DataFrame(
            columns=[
                "symbol",
                "realized_pl",
                "unrealized_pl",
                "total_pl",
                "open_qty",
                "avg_entry_price",
                "market_price",
                "trade_rows",
                "win_flag_if_known",
            ]
        )
    )
    if not df_by.empty:
        df_by.to_csv(OUT_BY_SYMBOL, index=False, encoding="utf-8")
    else:
        df_by.to_csv(OUT_BY_SYMBOL, index=False, encoding="utf-8")

    # Summary JSON
    n_real = sum(1 for r in all_rows if str(r.get("outcome_type", "")).upper() == "REALIZED")
    n_open = sum(1 for r in all_rows if str(r.get("outcome_type", "")).upper() == "OPEN")
    n_unkn = sum(1 for r in all_rows if str(r.get("outcome_type", "")).upper() == "UNKNOWN")
    open_pos = len({_norm_sym(r.get("symbol")) for r in open_rows})
    total_re = 0.0
    for r in all_rows:
        if str(r.get("outcome_type", "")).upper() != "REALIZED":
            continue
        v = _to_float(r.get("realized_pl"))
        if v is not None and np.isfinite(v):
            total_re += v
    total_un = 0.0
    un_ct = 0
    for r in open_rows:
        v = _to_float(r.get("unrealized_pl"))
        if v is not None and np.isfinite(v):
            total_un += v
            un_ct += 1
    has_r = n_real > 0
    has_u = un_ct > 0
    total_comb = total_re + total_un if (has_r or has_u) else 0.0

    winners = 0
    losers = 0
    for _, r in df_by.iterrows() if not df_by.empty else []:
        t = r.get("total_pl")
        try:
            t = float(t)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(t):
            continue
        if t > 0:
            winners += 1
        elif t < 0:
            losers += 1
    n_syms = len(df_by) if not df_by.empty else 0
    denom = winners + losers
    wr = (winners / denom) if denom > 0 else None
    best = None
    worst = None
    if not df_by.empty and "total_pl" in df_by.columns:
        try:
            x = df_by[df_by["total_pl"].apply(lambda v: v == v)]  # not nan
            if not x.empty:
                best = str(x.loc[x["total_pl"].idxmax()]["symbol"])
                worst = str(x.loc[x["total_pl"].idxmin()]["symbol"])
        except Exception:
            pass

    summary = {
        "total_symbols": n_syms,
        "open_positions": int(open_pos),
        "realized_trade_rows": int(n_real + n_unkn),
        "realized_paired_rows": int(n_real),
        "unknown_sell_rows": int(n_unkn),
        "total_realized_pl": _safe_float_for_json(total_re) if has_r else None,
        "total_unrealized_pl": _safe_float_for_json(total_un) if has_u else None,
        "total_combined_pl": _safe_float_for_json(total_comb) if (has_r or has_u) else None,
        "winners": int(winners),
        "losers": int(losers),
        "win_rate_if_known": wr,
        "best_symbol": best,
        "worst_symbol": worst,
        "inputs_used": used,
    }
    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    n_orders = len(ledger) if not ledger.empty else 0
    n_pos = len(pos) if not pos.empty else 0
    n_out = len(out_main)
    n_close_fills = n_real + n_unkn
    trp = f"{_safe_float_for_json(total_re) if has_r else 0.0}"
    tur = f"{_safe_float_for_json(total_un) if has_u else 0.0}"
    tcp = f"{_safe_float_for_json(total_comb) if (has_r or has_u) else 0.0}"
    print(
        "[TRADE_OUTCOMES] "
        f"orders_rows={n_orders} "
        f"positions_rows={n_pos} "
        f"outcome_rows={n_out} "
        f"realized_rows={n_close_fills} "
        f"open_rows={n_open} "
        f"total_realized_pl={trp} "
        f"total_unrealized_pl={tur} "
        f"total_combined_pl={tcp} "
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build trade outcomes and PnL summary from result CSVs."
    )
    p.parse_args(argv)
    return build()


if __name__ == "__main__":
    raise SystemExit(main())
