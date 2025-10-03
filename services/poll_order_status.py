# services/poll_order_status.py
import argparse
import csv
import os
import sys
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# Path bootstrap
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from services.broker_alpaca import AlpacaBroker, AlpacaError
from services.notify import notify

RESULTS_DIR = os.path.join("data", "results")
LIVE_ORDERS_LOG = os.path.join(RESULTS_DIR, "live_orders.csv")

EXPECTED_COLS = [
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
]

TERMINAL = {
    "filled",
    "canceled",
    "expired",
    "rejected",
    "stopped",
    "suspended",
    "calculated",
    "replaced",
    "done_for_day",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_log():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(LIVE_ORDERS_LOG):
        with open(LIVE_ORDERS_LOG, "w", newline="") as f:
            csv.writer(f).writerow(EXPECTED_COLS)


def _upgrade_log_schema_if_needed():
    if not os.path.exists(LIVE_ORDERS_LOG):
        return
    try:
        with open(LIVE_ORDERS_LOG, "r", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            with open(LIVE_ORDERS_LOG, "w", newline="") as f:
                csv.writer(f).writerow(EXPECTED_COLS)
            return
        if rows[0] == EXPECTED_COLS:
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup = os.path.join(RESULTS_DIR, f"live_orders.backup.{ts}.csv")
        shutil.copyfile(LIVE_ORDERS_LOG, backup)
        with open(LIVE_ORDERS_LOG, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(EXPECTED_COLS)
            for row in rows[1:]:
                new_row = (row + [""] * len(EXPECTED_COLS))[: len(EXPECTED_COLS)]
                w.writerow(new_row)
        print("Upgraded live_orders.csv schema. Backup saved as:", backup)
    except Exception as e:
        print("Log schema upgrade skipped:", e)


def _read_log_df() -> pd.DataFrame:
    _ensure_log()
    try:
        df = pd.read_csv(LIVE_ORDERS_LOG)
    except Exception:
        # Robust reader to skip any malformed old lines
        df = pd.read_csv(LIVE_ORDERS_LOG, engine="python", on_bad_lines="skip")
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = "" if c not in ("qty", "limit_price", "filled_qty") else 0
    df = df[EXPECTED_COLS].copy()
    return df


def _latest_session_from_log(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    sub = df[df["action"] == "submit"]["session"].astype(str).str.strip()
    sub = sub[sub != ""]
    if not sub.empty:
        return str(sub.iloc[-1])
    s = df["session"].astype(str).str.strip()
    s = s[s != ""]
    return str(s.iloc[-1]) if not s.empty else None


def _minutes_to_market_close(broker: AlpacaBroker) -> Optional[float]:
    try:
        clk = broker.get_clock()
        next_close = clk.get("next_close")
        is_open = bool(clk.get("is_open", False))
        if not next_close or not is_open:
            return None
        close_dt = datetime.fromisoformat(next_close.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (close_dt - now).total_seconds() / 60.0
    except Exception:
        return None


def _build_session_map(df: pd.DataFrame) -> Dict[str, str]:
    """Map order_id -> last known non-empty session from history."""
    if df.empty:
        return {}
    df2 = df.dropna(subset=["order_id"]).copy()
    df2["session"] = df2["session"].astype(str)
    df2 = df2[df2["session"].str.strip() != ""]
    if df2.empty:
        return {}
    df2 = df2.sort_values("timestamp").groupby("order_id", as_index=False).last()
    return {str(r["order_id"]): str(r["session"]) for _, r in df2.iterrows()}


def _last_state(df: pd.DataFrame, order_id: str) -> Optional[pd.Series]:
    sub = df[df["order_id"].astype(str).eq(order_id)]
    if sub.empty:
        return None
    return sub.tail(1).iloc[0]


def _append_log_row(row: List[Any]):
    with open(LIVE_ORDERS_LOG, "a", newline="") as f:
        csv.writer(f).writerow(row)


def _write_poll_if_changed(session: str, o: Dict[str, Any], last: Optional[pd.Series]) -> bool:
    """Write a poll row only if status or filled_qty changed since the last log entry for this order."""
    symbol = o.get("symbol", "")
    side = o.get("side", "")
    qty = int(float(o.get("qty", 0) or 0))
    otype = (o.get("type") or o.get("order_type") or "").lower()
    limit_price = o.get("limit_price", "")
    status = o.get("status", "")
    order_id = o.get("id", "") or o.get("order_id", "") or ""
    filled_qty = int(float(o.get("filled_qty", 0) or 0))
    filled_avg_price = o.get("filled_avg_price", "") or ""
    client_order_id = o.get("client_order_id", "") or ""
    tp_limit = ""
    sl_stop = ""
    # Bracket legs (best-effort)
    legs = o.get("legs") or []
    for leg in legs:
        ltype = (leg.get("type") or "").lower()
        if ltype == "limit" and (leg.get("status") or "").lower() != "canceled":
            tp_limit = leg.get("limit_price", tp_limit)
        if ltype == "stop" and (leg.get("status") or "").lower() != "canceled":
            sl_stop = leg.get("stop_price", sl_stop)

    # Dedup unchanged
    if last is not None:
        same_status = str(last.get("status", "")) == status
        same_filled = int(float(last.get("filled_qty", 0) or 0)) == filled_qty
        if same_status and same_filled:
            return False

    _append_log_row(
        [
            _now_iso(),
            session,
            "poll",
            symbol,
            side,
            qty,
            otype,
            limit_price,
            order_id,
            status,
            filled_qty,
            filled_avg_price,
            client_order_id,
            tp_limit,
            sl_stop,
        ]
    )
    return True


def _summarize_session(df_all: pd.DataFrame, session: str):
    """Summarize by taking the last status for each order_id submitted in the session (from the full log)."""
    submits = df_all[
        (df_all["action"] == "submit") & (df_all["session"].astype(str).str.strip() == session)
    ].copy()
    if submits.empty:
        print(f"Session summary [{session}]: no submit rows found.")
        return

    iids = set(submits["order_id"].astype(str))
    last = (
        df_all[df_all["order_id"].astype(str).isin(iids)]
        .sort_values("timestamp")
        .groupby("order_id", as_index=False)
        .last()[["order_id", "status", "filled_qty", "filled_avg_price", "symbol", "qty"]]
    )
    last["qty"] = pd.to_numeric(last["qty"], errors="coerce").fillna(0).astype(int)
    last["filled_qty"] = pd.to_numeric(last["filled_qty"], errors="coerce").fillna(0).astype(int)

    total_qty = int(last["qty"].sum())
    total_filled = int(last["filled_qty"].sum())
    fill_pct = (100.0 * total_filled / total_qty) if total_qty else 0.0
    print(
        f"Session summary [{session}]: total_qty={total_qty} total_filled={total_filled} fill_pct={fill_pct:.2f}%"
    )

    # Per-symbol breakdown
    br = last.groupby("symbol", as_index=False).agg(
        qty=("qty", "sum"),
        filled=("filled_qty", "sum"),
        status=("status", "last"),
    )
    headers = ["Symbol", "Qty", "Filled", "Fill%", "LastStatus"]
    widths = [len(h) for h in headers]
    rows: List[List[str]] = []
    for _, r in br.iterrows():
        denom = max(1, int(r["qty"]))
        row = [
            r["symbol"],
            int(r["qty"]),
            int(r["filled"]),
            f"{(100*r['filled']/denom):.1f}%",
            r["status"],
        ]
        rows.append(row)
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join("{:%d}" % w for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))

    # Persist session summary snapshot
    out_csv = os.path.join(RESULTS_DIR, "order_session_summaries.csv")
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "session", "total_qty", "total_filled", "fill_pct"])
        w.writerow([_now_iso(), session, total_qty, total_filled, round(fill_pct, 4)])


def main():
    parser = argparse.ArgumentParser(
        description="Poll orders; tag sessions; dedup; EOD cancel/refresh; summarize."
    )
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument(
        "--session",
        default=None,
        help="Override session (used when no mapping is found).",
    )
    parser.add_argument(
        "--cancel-near-eod-min",
        type=float,
        default=5.0,
        help="If minutes to close <= this and order unfilled, cancel.",
    )
    parser.add_argument(
        "--refresh-on-cancel",
        action="store_true",
        help="After cancel, re-submit as MARKET.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not actually cancel/refresh, just log.",
    )
    args = parser.parse_args()

    _ensure_log()
    _upgrade_log_schema_if_needed()

    try:
        broker = AlpacaBroker(mode=args.mode)
    except Exception as e:
        print("Broker init failed:", e)
        return

    df_hist = _read_log_df()
    session_map = _build_session_map(df_hist)

    # 1) Poll OPEN orders (fast path)
    try:
        open_orders = broker.list_orders(status="open", limit=200)
    except AlpacaError:
        open_orders = []

    changes = 0
    for o in open_orders:
        oid = str(o.get("id", "") or "")
        sess = session_map.get(oid, args.session or "")
        last = _last_state(df_hist, oid)
        if _write_poll_if_changed(sess, o, last):
            changes += 1
            # Notifications on transitions
            prev_filled = int(float(last.get("filled_qty", 0) or 0)) if last is not None else 0
            prev_status = str(last.get("status", "")) if last is not None else ""
            filled_qty = int(float(o.get("filled_qty", 0) or 0))
            qty = int(float(o.get("qty", 0) or 0))
            symbol = o.get("symbol", "")
            side = o.get("side", "")
            otype = (o.get("type") or "").upper()
            filled_avg_price = o.get("filled_avg_price", "") or "avg"
            status = o.get("status", "")
            if filled_qty > prev_filled:
                ev = "filled" if filled_qty == qty else "partial_fill"
                notify(
                    ev,
                    f"{symbol} {side.upper()} {filled_qty}/{qty} @ {filled_avg_price} (status={status})",
                )
            elif status != prev_status and status in ("canceled", "rejected"):
                notify(
                    status,
                    f"{symbol} {side.upper()} {qty} {otype} status={status} id={oid}",
                )

    # 2) Poll OUTSTANDING orders (those not terminal by the last seen log row)
    if not df_hist.empty:
        last_by_id = df_hist.sort_values("timestamp").groupby("order_id", as_index=False).last()
        outstanding_ids: Set[str] = set(
            last_by_id[~last_by_id["status"].astype(str).str.lower().isin(TERMINAL)][
                "order_id"
            ].astype(str)
        )
    else:
        outstanding_ids = set()

    open_ids = {str(o.get("id", "")) for o in open_orders}
    for oid in sorted(outstanding_ids):
        if oid in open_ids:
            continue
        try:
            o = broker.get_order(oid)
        except AlpacaError:
            continue
        sess = session_map.get(oid, args.session or "")
        last = _last_state(df_hist, oid)
        if _write_poll_if_changed(sess, o, last):
            changes += 1
            # Transition notifications
            prev_filled = int(float(last.get("filled_qty", 0) or 0)) if last is not None else 0
            prev_status = str(last.get("status", "")) if last is not None else ""
            filled_qty = int(float(o.get("filled_qty", 0) or 0))
            qty = int(float(o.get("qty", 0) or 0))
            symbol = o.get("symbol", "")
            side = o.get("side", "")
            otype = (o.get("type") or "").upper()
            filled_avg_price = o.get("filled_avg_price", "") or "avg"
            status = o.get("status", "")
            if filled_qty > prev_filled:
                ev = "filled" if filled_qty == qty else "partial_fill"
                notify(
                    ev,
                    f"{symbol} {side.upper()} {filled_qty}/{qty} @ {filled_avg_price} (status={status})",
                )
            elif status != prev_status and status in ("canceled", "rejected"):
                notify(
                    status,
                    f"{symbol} {side.upper()} {qty} {otype} status={status} id={oid}",
                )

    # 3) Near EOD cancel/refresh (on current OPEN orders)
    minutes_to_close = _minutes_to_market_close(broker)
    if minutes_to_close is not None and minutes_to_close <= args.cancel_near_eod_min:
        for o in open_orders:
            oid = str(o.get("id", "") or "")
            symbol = o.get("symbol", "")
            side = o.get("side", "")
            qty = int(float(o.get("qty", 0) or 0))
            filled_qty = int(float(o.get("filled_qty", 0) or 0))
            remain = qty - filled_qty
            if remain <= 0:
                continue
            sess = session_map.get(oid, args.session or "")
            if args.dry_run:
                print(f"DRY EOD CANCEL: {symbol} remain={remain}")
                continue
            try:
                broker.cancel_order(oid)
                _append_log_row(
                    [
                        _now_iso(),
                        sess,
                        "cancel",
                        symbol,
                        side,
                        qty,
                        (o.get("type", "") or "").lower(),
                        o.get("limit_price", ""),
                        oid,
                        "canceled",
                        filled_qty,
                        o.get("filled_avg_price", ""),
                        o.get("client_order_id", ""),
                        "",
                        "",
                    ]
                )
                print(f"Canceled unfilled {symbol} (remain={remain}) near EOD.")
                notify(
                    "canceled",
                    f"{symbol} {side.upper()} canceled near EOD. remain={remain}",
                )

                if args.refresh_on_cancel:
                    resp = broker.submit_order(
                        symbol=symbol,
                        qty=remain,
                        side=side,
                        order_type="market",
                        time_in_force="day",
                        client_order_id=f"refresh-{oid}",
                    )
                    rid = resp.get("id", "")
                    rstatus = resp.get("status", "")
                    rfilled = int(float(resp.get("filled_qty", 0) or 0))
                    ravg = resp.get("filled_avg_price", "")
                    _append_log_row(
                        [
                            _now_iso(),
                            sess,
                            "submit",
                            symbol,
                            side,
                            remain,
                            "market",
                            "",
                            rid,
                            rstatus,
                            rfilled,
                            ravg,
                            f"refresh-{oid}",
                            "",
                            "",
                        ]
                    )
                    print(f"Refreshed {symbol} as MARKET for remain={remain}. status={rstatus}")
                    notify(
                        "refresh",
                        f"{symbol} {side.upper()} re-submitted MARKET for remain={remain}. status={rstatus}",
                    )
            except AlpacaError as e:
                print(f"Cancel/refresh failed for {symbol}: {e}")
                notify("error", f"Cancel/refresh failed {symbol}: {e}")

    # 4) Summary
    print(
        f"Polled {len(open_orders)} open + {len(outstanding_ids)} outstanding; wrote {changes} changes."
    )
    df_all = _read_log_df()
    chosen = args.session or _latest_session_from_log(df_all)
    if chosen:
        _summarize_session(df_all, chosen)
    else:
        print("No session found to summarize.")


if __name__ == "__main__":
    main()
