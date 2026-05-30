# services/manage_open_orders.py
"""
Open-order diagnostics (legacy --legacy-pressure) and Smart Order Management:

fetches Alpaca open orders + quotes, classifies STALE / VERY_STALE, reprices or
cancel+replaces limits per config/order_manager.json, writes data/results/manage_orders.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from services.broker_call_resilience import (
    call_with_transient_retry,
    is_transient_broker_error,
    transient_failure_kind,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OPEN_SNAPSHOT = RESULTS / "open_orders_snapshot.csv"
LIVE_LOG = RESULTS / "live_orders_log.csv"
OUT_JSON = RESULTS / "open_order_pressure.json"
OUT_CSV = RESULTS / "open_order_pressure.csv"
OUT_LOG = RESULTS / "open_order_pressure_log.csv"
STALE_CSV = RESULTS / "stale_open_orders.csv"
SMART_MANAGER_CONFIG = ROOT / "config" / "order_manager.json"
SMART_MANAGER_JSON = RESULTS / "smart_order_manager_last.json"
MANAGE_ORDERS_CSV = RESULTS / "manage_orders.csv"

DEFAULT_STALE_MINUTES = 30

# Statuses treated as open / in-flight for stale policy (lowercase)
OPEN_IN_FLIGHT_STATUSES = frozenset(
    {
        "accepted",
        "new",
        "pending_new",
        "partially_filled",
        "pending_replace",
        "accepted_for_bidding",
        "pending_cancel",
        "held",
        "open",
        "pending",
        "submitted",
    }
)

# Terminal / not eligible for stale cancel path
TERMINAL_STATUSES = frozenset(
    {
        "filled",
        "canceled",
        "cancelled",
        "expired",
        "rejected",
        "done_for_day",
        "replaced",
        "stopped",
        "suspended",
        "calculated",
    }
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(val: Any) -> Optional[datetime]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _norm_status(st: Any) -> str:
    return str(st or "").strip().lower()


def _is_open_manageable_status(st: str) -> bool:
    u = _norm_status(st)
    if not u or u == "nan":
        return False
    if u in TERMINAL_STATUSES or u == "filled":
        return False
    if u in OPEN_IN_FLIGHT_STATUSES:
        return True
    # Unknown: treat as open if not clearly terminal
    if u in TERMINAL_STATUSES:
        return False
    return u not in TERMINAL_STATUSES


def _is_buy_side(side: Any) -> bool:
    s = str(side or "").strip().lower()
    return "buy" in s


def _is_sell_side(side: Any) -> bool:
    s = str(side or "").strip().lower()
    return "sell" in s


def _qty_filled_filled(row: Dict[str, Any]) -> Tuple[float, float]:
    try:
        q = float(row.get("qty") or row.get("quantity") or 0)
    except Exception:
        q = 0.0
    try:
        fq = float(row.get("filled_qty") or 0)
    except Exception:
        fq = 0.0
    return q, fq


def _not_filled(row: Dict[str, Any]) -> bool:
    q, fq = _qty_filled_filled(row)
    if q <= 0:
        return fq <= 0
    return fq < q - 1e-6


def _order_ts(row: pd.Series) -> Optional[datetime]:
    for c in ("submitted_at", "created_at", "timestamp", "updated_at"):
        if c in row.index and pd.notna(row.get(c)):
            dt = _parse_ts(row.get(c))
            if dt:
                return dt
    return None


def _load_session_map_from_log() -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        if not LIVE_LOG.is_file() or LIVE_LOG.stat().st_size == 0:
            return out
        df = pd.read_csv(LIVE_LOG)
        if "order_id" not in df.columns or "session" not in df.columns:
            return out
        df["order_id"] = df["order_id"].astype(str).str.strip()
        df["session"] = df["session"].astype(str).str.strip()
        if "timestamp" in df.columns:
            df["_ts"] = df["timestamp"].astype(str).apply(_parse_ts)
            df = df.sort_values("_ts")
        last = df.groupby("order_id", as_index=False).last()
        for _, r in last.iterrows():
            oid = str(r.get("order_id") or "").strip()
            if oid:
                out[oid] = str(r.get("session") or "").strip()
    except Exception:
        pass
    return out


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "symbol" not in d.columns and "ticker" in d.columns:
        d["symbol"] = d["ticker"]
    if "ticker" not in d.columns and "symbol" in d.columns:
        d["ticker"] = d["symbol"]
    if "id" in d.columns and "order_id" not in d.columns:
        d["order_id"] = d["id"]
    if "quantity" in d.columns and "qty" not in d.columns:
        d["qty"] = d["quantity"]
    if "price" in d.columns and "limit_price" not in d.columns:
        d["limit_price"] = d["price"]
    return d


def load_open_orders_snapshot_or_broker(
    mode: str,
    verbose: bool,
) -> Tuple[pd.DataFrame, str]:
    """Returns (dataframe, source_label)."""
    if OPEN_SNAPSHOT.is_file() and OPEN_SNAPSHOT.stat().st_size > 0:
        try:
            df = pd.read_csv(OPEN_SNAPSHOT)
            df = _normalize_frame(df)
            if not df.empty:
                if verbose:
                    print(
                        f"[manage_open_orders] loaded {len(df)} rows from {OPEN_SNAPSHOT.name}",
                        flush=True,
                    )
                return df, "open_orders_snapshot.csv"
        except Exception as e:
            if verbose:
                print(f"[manage_open_orders] snapshot read failed: {e}", flush=True)
    try:
        from services.broker_alpaca import AlpacaBroker

        bco: Dict[str, int] = {"retried": 0}
        br = call_with_transient_retry(
            "broker_init", lambda: AlpacaBroker(mode=mode), out_counts=bco
        )
        raw = call_with_transient_retry(
            "get_open_orders", lambda: br.get_open_orders(limit=500), out_counts=bco
        )
        df = pd.DataFrame(raw) if raw else pd.DataFrame()
        df = _normalize_frame(df)
        if verbose:
            print(
                f"[manage_open_orders] loaded {len(df)} rows from broker.get_open_orders",
                flush=True,
            )
        return df, "broker"
    except Exception as e:
        if verbose:
            print(f"[manage_open_orders] broker fallback failed: {e}", flush=True)
        return pd.DataFrame(), "none"


def _age_minutes_from_ts(ts: Optional[datetime], now: datetime) -> Optional[float]:
    if not ts:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() / 60.0


def build_pressure_rows(
    df: pd.DataFrame,
    *,
    stale_minutes: float,
    now: datetime,
    session_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return rows

    for idx, r in df.iterrows():
        sym = str(r.get("symbol") or r.get("ticker") or "").strip().upper()
        oid = str(r.get("order_id") or r.get("id") or "").strip()
        st = _norm_status(r.get("status"))
        side = str(r.get("side") or "").strip()
        try:
            lp = r.get("limit_price")
            lp_out = "" if lp is None or (isinstance(lp, float) and pd.isna(lp)) else float(lp)
        except Exception:
            lp_out = ""
        try:
            qty = int(float(r.get("qty") or r.get("quantity") or 0))
        except Exception:
            qty = 0

        ts = _order_ts(r)
        created_s = ""
        if ts:
            created_s = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        age_min = _age_minutes_from_ts(ts, now)

        manageable = _is_open_manageable_status(st)
        nf = _not_filled(dict(r))
        is_stale = False
        reason = "below_stale_threshold"
        if not manageable:
            reason = "not_open_manageable"
        elif not nf:
            reason = "fully_filled_or_complete"
        elif age_min is None:
            reason = "unknown_age"
        elif age_min >= float(stale_minutes):
            is_stale = True
            reason = f"age>={stale_minutes:g}m_open_not_filled"
        else:
            reason = "below_stale_threshold"

        sess = session_map.get(oid, str(r.get("session") or ""))

        rows.append(
            {
                "timestamp": _utc_iso(),
                "order_id": oid,
                "symbol": sym,
                "side": side,
                "qty": qty,
                "limit_price": lp_out,
                "status": st,
                "created_at_or_timestamp": created_s,
                "age_minutes": round(age_min, 3) if age_min is not None else "",
                "is_stale": is_stale,
                "stale_reason": reason,
                "session": sess,
                "client_order_id": str(r.get("client_order_id") or ""),
            }
        )
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    open_total = len(rows)
    stale = [r for r in rows if r.get("is_stale")]
    fresh = [
        r
        for r in rows
        if (not r.get("is_stale")) and r.get("stale_reason") == "below_stale_threshold"
    ]
    stale_n = len(stale)
    fresh_n = len(fresh)
    buy_n = sum(1 for r in rows if _is_buy_side(r.get("side")))
    sell_n = sum(1 for r in rows if _is_sell_side(r.get("side")))
    stale_by_sym: Dict[str, int] = {}
    for r in stale:
        s = str(r.get("symbol") or "").strip().upper()
        if s:
            stale_by_sym[s] = stale_by_sym.get(s, 0) + 1
    oldest = None
    for r in rows:
        try:
            am = float(r.get("age_minutes"))
            if oldest is None or am > oldest:
                oldest = am
        except Exception:
            pass
    blocking = sorted(stale_by_sym.keys())
    return {
        "open_orders_total": open_total,
        "stale_orders_total": stale_n,
        "fresh_orders_total": fresh_n,
        "buy_open_orders": buy_n,
        "sell_open_orders": sell_n,
        "stale_by_symbol": stale_by_sym,
        "oldest_open_order_minutes": round(oldest, 3) if oldest is not None else None,
        "symbols_blocking_execution": blocking,
    }


def _append_log(summary: Dict[str, Any]) -> None:
    try:
        OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts_utc": summary.get("timestamp", _utc_iso()),
            "mode": summary.get("mode", ""),
            "stale_minutes": summary.get("stale_minutes", ""),
            "open_orders_total": summary.get("open_orders_total", ""),
            "stale_orders_total": summary.get("stale_orders_total", ""),
            "fresh_orders_total": summary.get("fresh_orders_total", ""),
            "dry_run": summary.get("dry_run", ""),
            "canceled_orders": summary.get("canceled_orders", ""),
            "notes": ";".join(str(x) for x in summary.get("notes", []) if x)[:4000],
        }
        new_file = not OUT_LOG.is_file() or OUT_LOG.stat().st_size == 0
        with OUT_LOG.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                w.writeheader()
            w.writerow(row)
    except Exception:
        pass


def _write_artifacts(
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    stale_only: List[Dict[str, Any]],
) -> None:
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass
    try:
        cols = [
            "timestamp",
            "order_id",
            "symbol",
            "side",
            "qty",
            "limit_price",
            "status",
            "created_at_or_timestamp",
            "age_minutes",
            "is_stale",
            "stale_reason",
            "session",
            "client_order_id",
        ]
        if rows:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        else:
            pd.DataFrame(columns=cols).to_csv(OUT_CSV, index=False)
    except Exception:
        pass
    try:
        if stale_only:
            pd.DataFrame(stale_only).to_csv(STALE_CSV, index=False)
        else:
            pd.DataFrame(
                columns=[
                    "timestamp",
                    "order_id",
                    "symbol",
                    "side",
                    "qty",
                    "limit_price",
                    "status",
                    "created_at_or_timestamp",
                    "age_minutes",
                    "is_stale",
                    "stale_reason",
                    "session",
                    "client_order_id",
                ]
            ).to_csv(STALE_CSV, index=False)
    except Exception:
        pass
    _append_log(summary)


def load_smart_order_manager_config() -> Dict[str, Any]:
    cfg = {
        "stale_after_seconds": 120.0,
        "very_stale_after_seconds": 300.0,
        # STALE only: reprice when abs(distance_to_market) exceeds this (e.g. 0.002 = 0.2%)
        "distance_reprice_abs": 0.002,
        # Fixed targets: BUY ask * mult, SELL bid * mult
        "reprice_buy_mult": 0.9985,
        "reprice_sell_mult": 1.0015,
        "max_slippage_pct": 0.005,
        "spread_no_cross_buffer_bps": 2.0,
        "min_tick": 0.01,
        "quote_max_age_seconds": 60.0,
    }
    try:
        if SMART_MANAGER_CONFIG.is_file():
            u = json.loads(SMART_MANAGER_CONFIG.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict):
                cfg.update(u)
    except Exception:
        pass
    # Back-compat: legacy minute-based keys
    if "stale_after_seconds" not in cfg and cfg.get("stale_minutes") is not None:
        try:
            cfg["stale_after_seconds"] = float(cfg["stale_minutes"]) * 60.0
        except Exception:
            pass
    if "very_stale_after_seconds" not in cfg and cfg.get("inactive_minutes") is not None:
        try:
            cfg["very_stale_after_seconds"] = float(cfg["inactive_minutes"]) * 60.0
        except Exception:
            pass
    if cfg.get("max_slippage_pct") is None and cfg.get("max_slippage_bps_per_action") is not None:
        try:
            # e.g. 40 bps -> 0.4 (% points for _max_slippage_frac)
            cfg["max_slippage_pct"] = float(cfg["max_slippage_bps_per_action"]) / 100.0
        except Exception:
            pass
    return cfg


def _max_slippage_frac(cfg: Dict[str, Any]) -> float:
    """
    Max relative move vs current limit per action.
    - 0.5 or 50: 0.5% of price (legacy)
    - 0.005: 0.5% as decimal fraction (spec-style)
    """
    p = float(cfg.get("max_slippage_pct", 0.5) or 0.0)
    if p <= 0:
        return 0.0
    if p < 0.05:
        return p
    if p > 1.0:
        return p / 100.0
    return p / 100.0


def _sf(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        v = float(x)
        return v if v > 0 else None
    except Exception:
        return None


def _remaining_qty_raw(o: Dict[str, Any]) -> int:
    try:
        q = float(o.get("qty") or o.get("quantity") or 0)
        fq = float(o.get("filled_qty") or 0)
        return max(0, int(q) - int(fq))
    except Exception:
        return 0


def _order_age_seconds(o: Dict[str, Any], now: datetime) -> Optional[float]:
    for k in ("submitted_at", "created_at", "updated_at"):
        dt = _parse_ts(o.get(k))
        if dt:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (now - dt).total_seconds())
    return None


def _round_px(p: float, cfg: Dict[str, Any]) -> float:
    mt = float(cfg.get("min_tick", 0.01) or 0.01)
    if p >= 1.0:
        return round(round(p / mt) * mt, 2)
    return round(round(p / mt) * mt, 4)


def _parse_quote_ts(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    s = str(ts).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _quote_is_stale(
    q: Dict[str, Any],
    *,
    now: datetime,
    max_age_s: float,
) -> Tuple[bool, str]:
    """Returns (is_stale, reason_if_stale)."""
    if max_age_s <= 0:
        return False, ""
    dt = _parse_quote_ts(q.get("ts"))
    if not dt:
        return True, "quote_no_timestamp"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age = (now - dt).total_seconds()
    if age > max_age_s:
        return True, "stale_quote"
    return False, ""


def distance_to_market(is_buy: bool, lp: float, bid: float, ask: float) -> float:
    """Spec: BUY (limit-ask)/ask; SELL (bid-limit)/bid."""
    if is_buy:
        return (float(lp) - float(ask)) / float(ask)
    return (float(bid) - float(lp)) / float(bid)


def _target_reprice_fixed(is_buy: bool, bid: float, ask: float, cfg: Dict[str, Any]) -> float:
    """BUY: ask * 0.9985; SELL: bid * 1.0015 (configurable)."""
    bm = float(cfg.get("reprice_buy_mult", 0.9985) or 0.9985)
    sm = float(cfg.get("reprice_sell_mult", 1.0015) or 1.0015)
    if is_buy:
        return float(ask) * bm
    return float(bid) * sm


def _clamp_reprice_target(
    is_buy: bool,
    lp: float,
    raw_tgt: float,
    bid: float,
    ask: float,
    buf: float,
    max_slip_frac: float,
) -> Tuple[Optional[float], str]:
    """Enforce no aggressive cross + max_slippage vs current limit."""
    if is_buy:
        cap = float(ask) * (1.0 - buf)
        tgt = min(float(raw_tgt), cap)
        if tgt <= lp + 1e-12:
            return None, "no_improvement"
        hi = lp * (1.0 + max_slip_frac)
        tgt = min(tgt, hi)
        if tgt <= lp + 1e-12:
            return None, "max_slippage"
        return tgt, "ok"
    floor = float(bid) * (1.0 + buf)
    tgt = max(float(raw_tgt), floor)
    if tgt >= lp - 1e-12:
        return None, "no_improvement"
    lo = lp * (1.0 - max_slip_frac)
    tgt = max(tgt, lo)
    if tgt >= lp - 1e-12:
        return None, "max_slippage"
    return tgt, "ok"


def _plan_smart_action(
    o: Dict[str, Any],
    *,
    cfg: Dict[str, Any],
    now: datetime,
) -> Tuple[str, str, Any]:
    """
    Returns (action_kind, reason_code, payload).
    action_kind: skip | reprice | replace_inactive
    payload: new limit price (float) when action_kind != skip
    """
    st = _norm_status(o.get("status"))
    if st in ("pending_cancel", "pending_replace"):
        return "skip", "pending_broker_state", None
    otype = str(o.get("type") or o.get("order_type") or "").strip().lower()
    if otype and otype not in ("limit",):
        return "skip", "not_limit_order", None
    rem = _remaining_qty_raw(o)
    if rem <= 0:
        return "skip", "fully_filled_or_zero_remainder", None
    lp = _sf(o.get("limit_price") or o.get("price"))
    if lp is None:
        return "skip", "no_limit_price", None

    side = str(o.get("side") or "").strip().lower()
    is_buy = _is_buy_side(side)
    if not is_buy and not _is_sell_side(side):
        return "skip", "unknown_side", None

    try:
        fq = int(float(o.get("filled_qty") or 0))
    except Exception:
        fq = 0

    age_sec = _order_age_seconds(o, now)
    if age_sec is None:
        return "skip", "unknown_age", None

    return (
        "need_broker",
        "",
        {
            "lp": lp,
            "rem": rem,
            "is_buy": is_buy,
            "age_sec": float(age_sec),
            "fq": fq,
        },
    )


def _plan_price_after_quote(
    broker: Any,
    *,
    o: Dict[str, Any],
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
    now: datetime,
    bcounts: Optional[Dict[str, int]] = None,
) -> Tuple[str, str, Optional[float]]:
    """
    Quote + decision engine:
    stale quote -> STALE_QUOTE; FRESH -> TOO_NEW; VERY_STALE -> replace;
    STALE + abs(distance) > threshold -> reprice.
    """
    lp = float(meta["lp"])
    is_buy = bool(meta["is_buy"])
    age_sec = float(meta["age_sec"])

    stale_s = float(cfg.get("stale_after_seconds", 120.0))
    very_s = float(cfg.get("very_stale_after_seconds", 300.0))
    dist_thr = float(cfg.get("distance_reprice_abs", 0.002) or 0.002)
    buf = float(cfg.get("spread_no_cross_buffer_bps", 2.0)) / 10000.0
    max_slip = _max_slippage_frac(cfg)
    qmax_age = float(cfg.get("quote_max_age_seconds", 60.0) or 60.0)

    sym = str(o.get("symbol") or "").strip().upper()
    try:
        q = call_with_transient_retry(
            f"get_latest_quote({sym})",
            lambda: broker.get_latest_quote(sym),
            out_counts=bcounts,
        )
    except Exception:
        q = None
    if not q:
        return "skip", "no_quote", None

    stale_q, _st_r = _quote_is_stale(q, now=now, max_age_s=qmax_age)
    if stale_q:
        return "skip", "STALE_QUOTE", None

    bid = _sf((q or {}).get("bid"))
    ask = _sf((q or {}).get("ask"))
    if not bid or not ask:
        return "skip", "incomplete_quote", None
    if float(bid) >= float(ask):
        return "skip", "inverted_quote", None

    dist = distance_to_market(is_buy, lp, float(bid), float(ask))
    if age_sec <= stale_s:
        return "skip", "TOO_NEW", None

    raw_tgt = _target_reprice_fixed(is_buy, float(bid), float(ask), cfg)

    if age_sec > very_s:
        tgt_u, why = _clamp_reprice_target(
            is_buy, lp, raw_tgt, float(bid), float(ask), buf, max_slip
        )
        if tgt_u is None:
            return "skip", f"very_stale_{why}", None
        tgt = _round_px(float(tgt_u), cfg)
        return "replace_very_stale", "very_stale", tgt

    # STALE: (120, 300]
    if abs(dist) <= dist_thr:
        return "skip", "NOT_FAR_ENOUGH", None

    tgt_u, why = _clamp_reprice_target(is_buy, lp, raw_tgt, float(bid), float(ask), buf, max_slip)
    if tgt_u is None:
        return "skip", why, None
    tgt = _round_px(float(tgt_u), cfg)
    return "reprice", "stale_reprice", tgt


def _log_order_skip(sym: str, reason: str) -> None:
    print(f"[ORDER_SKIP] symbol={sym} reason={reason}", flush=True)


def _log_order_reprice(
    sym: str,
    old_p: float,
    new_p: float,
    age_sec: float,
    *,
    dry_run: bool,
) -> None:
    suf = " dry_run" if dry_run else ""
    print(
        f"[ORDER_REPRICE] symbol={sym} old={old_p:.4f} new={new_p:.4f} " f"age={age_sec:.1f}s{suf}",
        flush=True,
    )


def _log_order_cancel(sym: str, reason: str, *, dry_run: bool) -> None:
    suf = " dry_run" if dry_run else ""
    print(f"[ORDER_CANCEL] symbol={sym} reason={reason}{suf}", flush=True)


def _print_order_manager_summary(
    *,
    total_open: int,
    fresh: int,
    stale: int,
    very_stale: int,
    repriced: int,
    cancelled: int,
    replaced: int,
    skipped: int,
    note: str = "",
) -> None:
    """Required summary block (one field per line after the tag)."""
    lines = [
        "[ORDER_MANAGER]",
        f"total_open={total_open}",
        f"fresh={fresh}",
        f"stale={stale}",
        f"very_stale={very_stale}",
        f"repriced={repriced}",
        f"cancelled={cancelled}",
        f"replaced={replaced}",
        f"skipped={skipped}",
    ]
    if note:
        lines.append(f"note={note}")
    print("\n".join(lines), flush=True)


def _write_manage_orders_csv(rows: List[Dict[str, Any]]) -> None:
    """Always writes data/results/manage_orders.csv (header + rows, possibly empty)."""
    fieldnames = ["symbol", "action", "old_price", "new_price", "age_seconds", "reason"]
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        with MANAGE_ORDERS_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
    except Exception:
        pass


def _row_csv(
    symbol: str,
    action: str,
    *,
    old_price: Any = "",
    new_price: Any = "",
    age_seconds: Any = "",
    reason: str = "",
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "action": action,
        "old_price": old_price,
        "new_price": new_price,
        "age_seconds": age_seconds,
        "reason": reason,
    }


def run_smart_order_manager(
    mode: str,
    *,
    dry_run: bool = True,
    verbose: bool = False,
    ignore_market_closed: bool = False,
) -> int:
    """
    Actively reprice or cancel+replace open limit orders using Alpaca quotes.
    Does not touch signals, sizing, execute_trades, sector caps, or lifecycle.
    """
    cfg = load_smart_order_manager_config()
    now = datetime.now(timezone.utc)
    reasons: Counter[str] = Counter()
    repriced = cancelled = replaced = skipped = 0
    stale_after = float(cfg.get("stale_after_seconds", 120.0))
    very_after = float(cfg.get("very_stale_after_seconds", 300.0))
    csv_rows: List[Dict[str, Any]] = []
    bcounts: Dict[str, int] = {"retried": 0}
    moo_stats = {
        "attempted": 0,
        "retried": 0,
        "degraded": 0,
        "actions_taken": 0,
        "network_failures": 0,
    }

    def _age_cell(o: Dict[str, Any]) -> Any:
        ag = _order_age_seconds(o, now)
        return "" if ag is None else round(float(ag), 3)

    def _managed_limit_candidate(o: Dict[str, Any]) -> bool:
        if _norm_status(o.get("status")) in ("pending_cancel", "pending_replace"):
            return False
        otype = str(o.get("type") or o.get("order_type") or "").strip().lower()
        if otype and otype != "limit":
            return False
        if _remaining_qty_raw(o) <= 0:
            return False
        if not _sf(o.get("limit_price") or o.get("price")):
            return False
        return True

    try:
        from services.broker_alpaca import AlpacaBroker

        broker = call_with_transient_retry(
            "broker_init", lambda: AlpacaBroker(mode=mode), out_counts=bcounts
        )
    except Exception as e:
        moo_stats["network_failures"] = 1
        moo_stats["degraded"] = 1
        rk = transient_failure_kind(e) if is_transient_broker_error(e) else "network"
        print(f"[ORDER_MANAGER] broker_init_failed={e}", flush=True)
        print(f"[MOO] DEGRADED reason={rk}", flush=True)
        _write_manage_orders_csv([])
        print(
            f"[MANAGE_OPEN_ORDERS_SUMMARY] attempted=0 retried={bcounts.get('retried', 0)} degraded=1 "
            f"actions_taken=0 network_failures=1",
            flush=True,
        )
        return 1

    try:
        moo_stats["attempted"] = 1
        raw = call_with_transient_retry(
            "list_orders",
            lambda: broker.list_orders(status="open", limit=500, nested=True) or [],
            out_counts=bcounts,
        )
        orders = [AlpacaBroker._normalize_order(o) for o in raw]
    except Exception as e:
        moo_stats["network_failures"] = 1
        moo_stats["degraded"] = 1
        if is_transient_broker_error(e):
            moo_stats["network_failures"] = 1
        rk = transient_failure_kind(e) if is_transient_broker_error(e) else "network"
        print(f"[ORDER_MANAGER] list_orders_failed={e}", flush=True)
        print(f"[MOO] DEGRADED reason={rk}", flush=True)
        _write_manage_orders_csv([])
        print(
            f"[MANAGE_OPEN_ORDERS_SUMMARY] attempted=1 retried={bcounts.get('retried', 0)} degraded=1 "
            f"actions_taken=0 network_failures={moo_stats['network_failures']}",
            flush=True,
        )
        return 1

    if not orders:
        print("[ORDER_MANAGER] no open orders", flush=True)
        _print_order_manager_summary(
            total_open=0,
            fresh=0,
            stale=0,
            very_stale=0,
            repriced=0,
            cancelled=0,
            replaced=0,
            skipped=0,
        )
        _write_manage_orders_csv([])
        print(
            f"[MANAGE_OPEN_ORDERS_SUMMARY] attempted=1 retried={bcounts.get('retried', 0)} degraded=0 "
            f"actions_taken=0 network_failures=0",
            flush=True,
        )
        return 0

    clock_open: Optional[bool] = None
    try:
        ck = call_with_transient_retry("get_clock", lambda: broker.get_clock(), out_counts=bcounts)
        if isinstance(ck, dict) and ck.get("is_open") is not None:
            clock_open = bool(ck.get("is_open"))
    except Exception:
        clock_open = None

    if clock_open is False and not ignore_market_closed:
        _print_order_manager_summary(
            total_open=len(orders),
            fresh=0,
            stale=0,
            very_stale=0,
            repriced=0,
            cancelled=0,
            replaced=0,
            skipped=0,
            note="market_closed",
        )
        _write_manage_orders_csv([])
        print(
            f"[MANAGE_OPEN_ORDERS_SUMMARY] attempted=1 retried={bcounts.get('retried', 0)} degraded=0 "
            f"actions_taken=0 network_failures=0",
            flush=True,
        )
        return 0

    total_open = len(orders)
    fresh_n = 0
    stale_n = 0
    very_stale_n = 0
    for o in orders:
        if not _managed_limit_candidate(o):
            continue
        ag = _order_age_seconds(o, now)
        if ag is None:
            continue
        if ag > very_after:
            very_stale_n += 1
        elif ag > stale_after:
            stale_n += 1
        else:
            fresh_n += 1

    for o in orders:
        oid = str(o.get("id") or o.get("order_id") or "").strip()
        sym = str(o.get("symbol") or "").strip().upper() or "?"
        age_cell = _age_cell(o)
        kind, code, payload = _plan_smart_action(o, cfg=cfg, now=now)
        if kind == "skip":
            skipped += 1
            reasons[code] += 1
            _log_order_skip(sym, str(code))
            csv_rows.append(
                _row_csv(
                    sym,
                    "skip",
                    old_price="",
                    new_price="",
                    age_seconds=age_cell,
                    reason=str(code),
                )
            )
            continue
        if kind != "need_broker" or not isinstance(payload, dict):
            skipped += 1
            reasons["plan_unexpected"] += 1
            _log_order_skip(sym, "plan_unexpected")
            csv_rows.append(
                _row_csv(
                    sym,
                    "skip",
                    old_price="",
                    new_price="",
                    age_seconds=age_cell,
                    reason="plan_unexpected",
                )
            )
            continue

        lp0 = float(payload.get("lp") or 0.0)
        age_sec = float(payload.get("age_sec") or 0.0)
        a, rsn, new_px = _plan_price_after_quote(
            broker, o=o, cfg=cfg, meta=payload, now=now, bcounts=bcounts
        )
        if a == "skip":
            skipped += 1
            reasons[str(rsn)] += 1
            _log_order_skip(sym, str(rsn))
            csv_rows.append(
                _row_csv(
                    sym,
                    "skip",
                    old_price=lp0,
                    new_price="",
                    age_seconds=round(age_sec, 3),
                    reason=str(rsn),
                )
            )
            continue

        side = str(o.get("side") or "").lower()
        rem = _remaining_qty_raw(o)
        tif = str(o.get("time_in_force") or "day").lower()
        npx = float(new_px or 0.0)

        if a == "reprice":
            _log_order_reprice(sym, lp0, npx, age_sec, dry_run=dry_run)
        elif a == "replace_very_stale":
            _log_order_cancel(sym, "very_stale", dry_run=dry_run)
            _log_order_reprice(sym, lp0, npx, age_sec, dry_run=dry_run)

        if dry_run:
            if a == "reprice":
                repriced += 1
            elif a == "replace_very_stale":
                replaced += 1
            reasons[f"dry_run_would_{a}"] += 1
            act = "reprice" if a == "reprice" else "replace"
            csv_rows.append(
                _row_csv(
                    sym,
                    act,
                    old_price=lp0,
                    new_price=npx,
                    age_seconds=round(age_sec, 3),
                    reason=str(rsn),
                )
            )
            if verbose:
                print(
                    f"[ORDER_MANAGER] DRY {a} sym={sym} id={oid} new_limit={new_px} detail={rsn}",
                    flush=True,
                )
            continue

        try:
            call_with_transient_retry(
                f"cancel_order({oid})", lambda: broker.cancel_order(oid), out_counts=bcounts
            )
        except Exception as ex:
            skipped += 1
            reasons["cancel_failed"] += 1
            if verbose:
                print(f"[ORDER_MANAGER] cancel_failed sym={sym} id={oid} err={ex}", flush=True)
            csv_rows.append(
                _row_csv(
                    sym,
                    "cancel_failed",
                    old_price=lp0,
                    new_price=npx,
                    age_seconds=round(age_sec, 3),
                    reason=str(ex),
                )
            )
            continue

        coid = f"triton-om{uuid.uuid4().hex[:22]}"
        try:
            resp = call_with_transient_retry(
                f"submit_order({sym})",
                lambda: broker.submit_order(
                    symbol=sym,
                    qty=rem,
                    side=side,
                    order_type="limit",
                    time_in_force=tif if tif else "day",
                    limit_price=npx,
                    client_order_id=coid,
                    extended_hours=False,
                ),
                out_counts=bcounts,
            )
            _ = str((resp or {}).get("id") or "").strip()
            if a == "reprice":
                repriced += 1
            elif a == "replace_very_stale":
                replaced += 1
            reasons["ok_" + a] += 1
            act = "reprice" if a == "reprice" else "replace"
            csv_rows.append(
                _row_csv(
                    sym,
                    act,
                    old_price=lp0,
                    new_price=npx,
                    age_seconds=round(age_sec, 3),
                    reason=str(rsn),
                )
            )
        except Exception as ex:
            cancelled += 1
            reasons["replace_submit_failed"] += 1
            if verbose:
                print(f"[ORDER_MANAGER] replace_submit_failed sym={sym} err={ex}", flush=True)
            csv_rows.append(
                _row_csv(
                    sym,
                    "replace_failed",
                    old_price=lp0,
                    new_price=npx,
                    age_seconds=round(age_sec, 3),
                    reason=str(ex),
                )
            )

    summary = {
        "timestamp": _utc_iso(),
        "epoch": time.time(),
        "mode": mode,
        "dry_run": dry_run,
        "total_open": total_open,
        "fresh": fresh_n,
        "stale": stale_n,
        "very_stale": very_stale_n,
        "repriced": repriced,
        "cancelled": cancelled,
        "replaced": replaced,
        "skipped": skipped,
        "reasons": dict(reasons),
        "ignore_market_closed": ignore_market_closed,
    }
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        SMART_MANAGER_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass

    _write_manage_orders_csv(csv_rows)

    _print_order_manager_summary(
        total_open=total_open,
        fresh=fresh_n,
        stale=stale_n,
        very_stale=very_stale_n,
        repriced=repriced,
        cancelled=cancelled,
        replaced=replaced,
        skipped=skipped,
    )
    rs = ",".join(f"{k}={v}" for k, v in sorted(reasons.items()) if v)[:400]
    if rs:
        print(f"[ORDER_MANAGER] reasons={rs}", flush=True)
    moo_stats["actions_taken"] = int(repriced + replaced)
    moo_stats["retried"] = int(bcounts.get("retried", 0))
    ret = 1 if int(moo_stats.get("degraded", 0)) else 0
    print(
        f"[MANAGE_OPEN_ORDERS_SUMMARY] attempted={moo_stats['attempted']} retried={moo_stats['retried']} "
        f"degraded={int(moo_stats.get('degraded', 0))} actions_taken={moo_stats['actions_taken']} "
        f"network_failures={int(moo_stats.get('network_failures', 0))}",
        flush=True,
    )
    return ret


def run(
    *,
    mode: str,
    stale_minutes: float,
    execute_cancel: bool,
    verbose: bool,
) -> int:
    now = datetime.now(timezone.utc)
    notes: List[str] = []
    dry_run = not execute_cancel

    df, src = load_open_orders_snapshot_or_broker(mode, verbose)
    if df.empty:
        notes.append(f"no open orders (source={src})")
        summary = {
            "timestamp": _utc_iso(),
            "mode": mode,
            "stale_minutes": stale_minutes,
            "open_orders_total": 0,
            "stale_orders_total": 0,
            "fresh_orders_total": 0,
            "buy_open_orders": 0,
            "sell_open_orders": 0,
            "stale_by_symbol": {},
            "oldest_open_order_minutes": None,
            "symbols_blocking_execution": [],
            "dry_run": True,
            "canceled_orders": 0,
            "cancel_failed": 0,
            "cancel_skipped": 0,
            "data_source": src,
            "notes": notes,
        }
        _write_artifacts([], summary, [])
        print(
            f"[manage_open_orders] mode={mode} dry_run=True open=0 stale=0 fresh=0 threshold={stale_minutes:g}m (no data)",
            flush=True,
        )
        return 0

    session_map = _load_session_map_from_log()
    rows = build_pressure_rows(df, stale_minutes=stale_minutes, now=now, session_map=session_map)
    meta = summarize(rows)
    stale_rows = [r for r in rows if r.get("is_stale")]

    canceled = failed = skipped = 0
    if execute_cancel:
        if mode.lower() != "paper":
            notes.append(
                "LIVE: --execute-cancel refused by policy; set mode=paper or use manual broker tools."
            )
            dry_run = True
            execute_cancel = False
        else:
            dry_run = False
            try:
                from services.broker_alpaca import AlpacaBroker

                bco = {"retried": 0}
                br = call_with_transient_retry(
                    "broker_init", lambda: AlpacaBroker(mode="paper"), out_counts=bco
                )
                for r in stale_rows:
                    oid = str(r.get("order_id") or "").strip()
                    if not oid:
                        skipped += 1
                        continue
                    try:
                        call_with_transient_retry(
                            f"cancel_order({oid})",
                            lambda: br.cancel_order(oid),
                            out_counts=bco,
                        )
                        canceled += 1
                        if verbose:
                            print(
                                f"[manage_open_orders] canceled order_id={oid} symbol={r.get('symbol')}",
                                flush=True,
                            )
                    except Exception as e:
                        failed += 1
                        if verbose:
                            print(
                                f"[manage_open_orders] cancel failed order_id={oid} err={e}",
                                flush=True,
                            )
            except Exception as e:
                notes.append(f"cancel path error: {e}")
                failed += len(stale_rows)

    summary = {
        "timestamp": _utc_iso(),
        "mode": mode,
        "stale_minutes": stale_minutes,
        **meta,
        "dry_run": dry_run,
        "canceled_orders": canceled,
        "cancel_failed": failed,
        "cancel_skipped": skipped,
        "data_source": src,
        "notes": notes,
    }
    _write_artifacts(rows, summary, stale_rows)

    print(
        f"[manage_open_orders] mode={mode} dry_run={dry_run} open={meta['open_orders_total']} "
        f"stale={meta['stale_orders_total']} fresh={meta['fresh_orders_total']} threshold={stale_minutes:g}m",
        flush=True,
    )
    if meta["symbols_blocking_execution"]:
        syms = ", ".join(meta["symbols_blocking_execution"][:40])
        print(f"[manage_open_orders] stale symbols: {syms}", flush=True)
    if dry_run:
        print(f"[manage_open_orders] would_cancel={meta['stale_orders_total']}", flush=True)
    else:
        print(
            f"[manage_open_orders] canceled={canceled} failed={failed} skipped={skipped}",
            flush=True,
        )

    print(
        f"[MANAGE_OPEN_ORDERS_SUMMARY] attempted=1 retried=0 degraded=0 "
        f"actions_taken={canceled} network_failures=0",
        flush=True,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TRITON Smart Order Management (default) or legacy stale-pressure diagnostics"
    )
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--stale-minutes", type=float, default=DEFAULT_STALE_MINUTES)
    ap.add_argument(
        "--execute-cancel",
        action="store_true",
        help="Cancel stale orders (paper only; live blocked)",
    )
    ap.add_argument(
        "--legacy-pressure",
        action="store_true",
        help="Diagnostics-only stale pressure path (open_order_pressure CSV). Default is smart manager.",
    )
    ap.add_argument(
        "--smart-manage",
        action="store_true",
        help="(Deprecated/no-op) Smart manager is the default; use --legacy-pressure for old behavior.",
    )
    ap.add_argument(
        "--execute-smart",
        action="store_true",
        help="Apply smart manager actions (default is dry-run preview)",
    )
    ap.add_argument(
        "--ignore-market-closed",
        action="store_true",
        help="Run smart manager when market is closed (normally blocked)",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.legacy_pressure:
        return run(
            mode=args.mode,
            stale_minutes=float(args.stale_minutes),
            execute_cancel=bool(args.execute_cancel),
            verbose=bool(args.verbose),
        )
    return run_smart_order_manager(
        args.mode,
        dry_run=not bool(args.execute_smart),
        verbose=bool(args.verbose),
        ignore_market_closed=bool(args.ignore_market_closed),
    )


if __name__ == "__main__":
    raise SystemExit(main())
