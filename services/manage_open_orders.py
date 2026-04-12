# services/manage_open_orders.py
"""Stale open-order diagnostics and optional paper cancellation. Best-effort; never raises to callers."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OPEN_SNAPSHOT = RESULTS / "open_orders_snapshot.csv"
LIVE_LOG = RESULTS / "live_orders_log.csv"
OUT_JSON = RESULTS / "open_order_pressure.json"
OUT_CSV = RESULTS / "open_order_pressure.csv"
OUT_LOG = RESULTS / "open_order_pressure_log.csv"
STALE_CSV = RESULTS / "stale_open_orders.csv"

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

        br = AlpacaBroker(mode=mode)
        raw = br.get_open_orders(limit=500)
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

                br = AlpacaBroker(mode="paper")
                for r in stale_rows:
                    oid = str(r.get("order_id") or "").strip()
                    if not oid:
                        skipped += 1
                        continue
                    try:
                        br.cancel_order(oid)
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

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TRITON stale open-order diagnostics (optional paper cancel)"
    )
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--stale-minutes", type=float, default=DEFAULT_STALE_MINUTES)
    ap.add_argument(
        "--execute-cancel",
        action="store_true",
        help="Cancel stale orders (paper only; live blocked)",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    return run(
        mode=args.mode,
        stale_minutes=float(args.stale_minutes),
        execute_cancel=bool(args.execute_cancel),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    raise SystemExit(main())
