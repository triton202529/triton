# services/poll_order_status.py
"""
Poll Alpaca orders and append status changes to data/results/live_orders_log.csv.

CLEAN UPDATE (2026-01-29):
- ✅ Session stamping deterministic:
    CLI --session > session_map[order_id] > latest_session_from_log > ""
- ✅ Schema-safe CSV append aligned to EXPECTED_COLS
- ✅ Summary hardening:
    If requested session has NO rows:
      - Try LIVE summary from broker
      - If broker returns no orders, fallback to summarizing latest session in log
        (the "closest available" truth).

PATCH (2026-01-29 LATE):
- ✅ Fix session "double counting" when the same session label is reused:
    Default summary for a session is now LATEST SUBMIT BURST ONLY (not all submits ever).
    Use --summarize-all to restore old behavior.
- ✅ When --session is provided, outstanding polling can be scoped to that latest burst
    (prevents older non-terminal leftovers from polluting the view).

PATCH (2026-01-31):
- ✅ CRITICAL FIX: log schema upgrade is now SAFE:
    - upgrades by COLUMN NAMES (not positional slicing)
    - tolerates extra columns (e.g., legacy 'mode')
    - detects snapshot-only files ('snapshot_ts' header) and refuses to corrupt

PATCH (2026-02-01):
- ✅ Auto-repair known "submit-row shift" corruption patterns:
    Pattern A: status contains 'triton-*' and client_order_id empty -> move status -> client_order_id, blank status
    Pattern B: submit rows where order_id is numeric (actually limit_price), status blank, client_order_id is 'triton-*'
              -> move order_id -> limit_price, set order_id="", set status="submitted"
- ✅ Outstanding polling ignores non-UUID-like order_ids (prevents get_order('248.04') spam)

PATCH (2026-02-02):
- ✅ Add simple .lock file around log writes to avoid corruption if two processes write at once

PATCH (2026-02-02 MON AM):
- ✅ FIX NameError: define _append_log_row_dict used by _write_poll_if_changed

PATCH (2026-04-01):
- ✅ Add post-change reconciliation:
    - when polling writes changes, refresh broker-backed snapshots via services.snapshot_live_orders
    - best effort only; reconciliation failures warn but do not fail polling

PATCH (2026-04-01 lifecycle):
- ✅ After snapshot_live_orders (when changes > 0), best-effort run services.apply_signal_lifecycle
"""

import argparse
import csv
import os
import sys
import shutil
import time
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# Path bootstrap
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from services.broker_alpaca import AlpacaBroker, AlpacaError
from services.notify import notify  # noqa: F401  (kept for compatibility)

RESULTS_DIR = os.path.join("data", "results")
LIVE_ORDERS_LOG = os.path.join(RESULTS_DIR, "live_orders_log.csv")

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


# -----------------------------
# .lock guard for safe writes
# -----------------------------
@contextmanager
def _log_lock(path: str, timeout_sec: float = 5.0):
    """
    Very small cross-process lock using a sidecar file:
      live_orders_log.csv.lock

    Prevents concurrent writers (two pollers / poll+execute) from interleaving CSV writes.
    """
    lock_path = path + ".lock"
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > timeout_sec:
                raise RuntimeError(f"Timed out waiting for log lock: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except Exception:
            pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_log() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(LIVE_ORDERS_LOG):
        with _log_lock(LIVE_ORDERS_LOG):
            if not os.path.exists(LIVE_ORDERS_LOG):
                with open(LIVE_ORDERS_LOG, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(EXPECTED_COLS)


def _first_line(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (f.readline() or "").strip()
    except Exception:
        return ""


def _backup_file(path: str, tag: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = os.path.join(RESULTS_DIR, f"{os.path.basename(path)}.{tag}.{ts}.bak.csv")
    shutil.copyfile(path, backup)
    return backup


def _is_number_like(s: str) -> bool:
    try:
        float(str(s).strip())
        return True
    except Exception:
        return False


def _is_uuid_like(s: str) -> bool:
    s = str(s or "").strip()
    if len(s) < 20:
        return False
    # simple heuristic: typical UUID has 4 hyphens and hex-ish chars
    if s.count("-") >= 3 and any(c.isalpha() for c in s):
        return True
    # Alpaca order ids are UUIDs; be strict-ish
    return False


def _upgrade_log_schema_if_needed() -> None:
    """
    SAFE schema upgrade (by column names, not positions).
    Also refuses to "upgrade" snapshot-style files (snapshot_ts header),
    because those are a different artifact entirely.
    """
    if not os.path.exists(LIVE_ORDERS_LOG):
        return

    header = _first_line(LIVE_ORDERS_LOG)
    if not header:
        with _log_lock(LIVE_ORDERS_LOG):
            header2 = _first_line(LIVE_ORDERS_LOG)
            if not header2:
                with open(LIVE_ORDERS_LOG, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(EXPECTED_COLS)
        return

    # Guard: snapshot writer overwrote the event log
    if header.lower().startswith("snapshot_ts"):
        print(
            "[SCHEMA_GUARD] live_orders_log.csv looks like a snapshot file (snapshot_ts). Not upgrading."
        )
        print("[SCHEMA_GUARD] live_orders_log.csv must be an EVENT LOG with header:")
        print("              " + ",".join(EXPECTED_COLS))
        return

    # Already correct
    if header.split(",") == EXPECTED_COLS:
        return

    backup = _backup_file(LIVE_ORDERS_LOG, tag="pre_upgrade")

    try:
        df = pd.read_csv(backup, keep_default_na=False)
    except Exception:
        df = pd.read_csv(backup, engine="python", on_bad_lines="skip", keep_default_na=False)

    df.columns = [str(c).strip() for c in df.columns]

    # common aliases (legacy)
    rename = {}
    if "order_type" in df.columns and "type" not in df.columns:
        rename["order_type"] = "type"
    if "id" in df.columns and "order_id" not in df.columns:
        rename["id"] = "order_id"
    df = df.rename(columns=rename)

    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = ""

    df2 = df[EXPECTED_COLS].copy()

    with _log_lock(LIVE_ORDERS_LOG):
        df2.to_csv(LIVE_ORDERS_LOG, index=False, encoding="utf-8")

    print("Upgraded live_orders_log.csv schema safely. Backup saved as:", backup)


def _repair_known_corruptions(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Repairs ONLY known safe patterns. Returns (df, repaired_rows_count).

    Pattern A:
      status startswith 'triton-' AND client_order_id empty -> move status -> client_order_id, blank status

    Pattern B (your current corruption):
      action == 'submit'
      status empty
      client_order_id startswith 'triton-'
      order_id is numeric (actually limit_price)
      limit_price empty OR non-numeric
        -> move order_id -> limit_price
        -> order_id = ""
        -> status = "submitted"
    """
    if df.empty:
        return df, 0

    repaired = 0
    d = df.copy()

    # Normalize for matching
    action = d.get("action", "").astype(str).str.strip().str.lower()
    status = d.get("status", "").astype(str).str.strip()
    client = d.get("client_order_id", "").astype(str).str.strip()
    order_id = d.get("order_id", "").astype(str).str.strip()
    limit_price = d.get("limit_price", "").astype(str).str.strip()

    # Pattern A
    mA = status.str.startswith("triton-") & ((client == "") | (client.str.lower() == "nan"))
    if mA.any():
        d.loc[mA, "client_order_id"] = d.loc[mA, "status"]
        d.loc[mA, "status"] = ""
        repaired += int(mA.sum())

    # Refresh views after modifications
    status = d.get("status", "").astype(str).str.strip()
    client = d.get("client_order_id", "").astype(str).str.strip()
    order_id = d.get("order_id", "").astype(str).str.strip()
    limit_price = d.get("limit_price", "").astype(str).str.strip()
    action = d.get("action", "").astype(str).str.strip().str.lower()

    # Pattern B
    mB = (
        (action == "submit")
        & ((status == "") | (status.str.lower() == "nan"))
        & client.str.startswith("triton-")
        & order_id.apply(_is_number_like)
    )

    # Only apply if limit_price is empty OR not number-like (to avoid overwriting good data)
    if mB.any():
        lp_bad = (
            (limit_price == "")
            | (limit_price.str.lower() == "nan")
            | (~limit_price.apply(_is_number_like))
        )
        mB = mB & lp_bad

    if mB.any():
        d.loc[mB, "limit_price"] = d.loc[mB, "order_id"]
        d.loc[mB, "order_id"] = ""
        # we cannot know broker status; mark as submitted (not unknown/nan)
        d.loc[mB, "status"] = "submitted"
        repaired += int(mB.sum())

    return d, repaired


def _read_log_df(repair: bool = True) -> pd.DataFrame:
    _ensure_log()
    try:
        df = pd.read_csv(LIVE_ORDERS_LOG, keep_default_na=False)
    except Exception:
        df = pd.read_csv(
            LIVE_ORDERS_LOG, engine="python", on_bad_lines="skip", keep_default_na=False
        )

    # hard guard: snapshot-only file
    if list(df.columns) == ["snapshot_ts"]:
        raise RuntimeError(
            "live_orders_log.csv is a snapshot file (snapshot_ts only). "
            "It must be an EVENT LOG. Fix the writer that overwrote it."
        )

    # Ensure expected columns exist
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = "" if c not in ("qty", "limit_price", "filled_qty") else 0

    df = df[EXPECTED_COLS].copy()

    # Normalize strings
    df["action"] = df["action"].astype(str).str.strip().str.lower()
    df["session"] = df["session"].astype(str).str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["side"] = df["side"].astype(str).str.strip().str.lower()
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["order_id"] = df["order_id"].astype(str).str.strip()
    df["timestamp"] = df["timestamp"].astype(str).str.strip()
    df["client_order_id"] = df["client_order_id"].astype(str).str.strip()

    # Numerics
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    df["filled_qty"] = pd.to_numeric(df["filled_qty"], errors="coerce").fillna(0).astype(int)

    if repair:
        repaired_df, n = _repair_known_corruptions(df)
        if n > 0:
            # Write back safely with backup so the log stays clean going forward
            backup = _backup_file(LIVE_ORDERS_LOG, tag="pre_repair")
            with _log_lock(LIVE_ORDERS_LOG):
                repaired_df.to_csv(LIVE_ORDERS_LOG, index=False, encoding="utf-8")
            print(
                f"[REPAIR] Applied {n} auto-repairs to live_orders_log.csv. Backup saved as: {backup}"
            )
            df = repaired_df

    return df


# -----------------------------
# MISSING WRITER (FIX)
# -----------------------------
def _append_log_row_dict(row: Dict[str, Any]) -> None:
    """
    Append one row to live_orders_log.csv safely.

    - uses the .lock guard to avoid concurrent write corruption
    - enforces EXPECTED_COLS order (no positional shifting)
    - ensures the log exists + schema is upgraded first
    """
    _ensure_log()
    _upgrade_log_schema_if_needed()

    out = {c: row.get(c, "") for c in EXPECTED_COLS}

    with _log_lock(LIVE_ORDERS_LOG):
        with open(LIVE_ORDERS_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=EXPECTED_COLS)
            w.writerow(out)


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


def _build_session_map(df: pd.DataFrame) -> Dict[str, str]:
    if df.empty:
        return {}
    df2 = df.copy()
    df2 = df2[df2["order_id"].astype(str).str.strip() != ""]
    df2 = df2[df2["session"].astype(str).str.strip() != ""]
    if df2.empty:
        return {}
    df2 = df2.sort_values("timestamp").groupby("order_id", as_index=False).last()
    return {str(r["order_id"]): str(r["session"]) for _, r in df2.iterrows()}


def _last_state(df: pd.DataFrame, order_id: str) -> Optional[pd.Series]:
    oid = str(order_id).strip()
    if not oid or df.empty:
        return None
    sub = df[df["order_id"].astype(str).eq(oid)]
    if sub.empty:
        return None
    return sub.tail(1).iloc[0]


def _choose_session(
    cli_session: str, order_id: str, session_map: Dict[str, str], latest_session: str
) -> str:
    s = (cli_session or "").strip()
    if s:
        return s
    oid = str(order_id or "").strip()
    if oid:
        mapped = (session_map.get(oid) or "").strip()
        if mapped:
            return mapped
    ls = (latest_session or "").strip()
    return ls if ls else ""


def _extract_legs_tp_sl(o: Dict[str, Any]) -> Tuple[str, str]:
    tp_limit = ""
    sl_stop = ""
    legs = o.get("legs") or []
    for leg in legs:
        ltype = (leg.get("type") or "").lower()
        lstatus = (leg.get("status") or "").lower()
        if ltype == "limit" and lstatus != "canceled":
            tp_limit = leg.get("limit_price", tp_limit)
        if ltype == "stop" and lstatus != "canceled":
            sl_stop = leg.get("stop_price", sl_stop)
    return tp_limit, sl_stop


def _write_poll_if_changed(
    cli_session: str,
    session_map: Dict[str, str],
    latest_session: str,
    o: Dict[str, Any],
    last: Optional[pd.Series],
) -> bool:
    symbol = (o.get("symbol", "") or "").strip().upper()
    side = (o.get("side", "") or "").strip().lower()

    try:
        qty = int(float(o.get("qty", 0) or 0))
    except Exception:
        qty = 0

    otype = (o.get("type") or o.get("order_type") or "").strip().lower()
    limit_price = o.get("limit_price", "")
    status = (o.get("status", "") or "").strip().lower()
    order_id = (o.get("id", "") or o.get("order_id", "") or "").strip()

    try:
        filled_qty = int(float(o.get("filled_qty", 0) or 0))
    except Exception:
        filled_qty = 0

    filled_avg_price = o.get("filled_avg_price", "") or ""
    client_order_id = o.get("client_order_id", "") or ""
    tp_limit, sl_stop = _extract_legs_tp_sl(o)

    if last is not None:
        last_status = str(last.get("status", "")).strip().lower()
        try:
            prev_filled = int(float(last.get("filled_qty", 0) or 0))
        except Exception:
            prev_filled = 0

        # If last status is blank/unknown/submitted but broker has a real status, write the first real status.
        if (last_status in ("", "unknown", "submitted", "nan")) and status not in ("", "nan"):
            pass
        else:
            same_status = last_status == status
            if same_status and (prev_filled == filled_qty):
                return False

    sess = _choose_session(cli_session, order_id, session_map, latest_session)

    _append_log_row_dict(
        {
            "timestamp": _now_iso(),
            "session": sess,
            "action": "poll",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "type": otype,
            "limit_price": limit_price,
            "order_id": order_id,
            "status": status,
            "filled_qty": filled_qty,
            "filled_avg_price": filled_avg_price,
            "client_order_id": client_order_id,
            "tp_limit": tp_limit,
            "sl_stop": sl_stop,
        }
    )
    return True


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _latest_submit_burst_order_ids(
    df_all: pd.DataFrame, session: str, burst_window_sec: int
) -> Set[str]:
    """
    Returns order_ids for the latest submit burst inside a session.
    Burst = submit rows whose timestamps are within burst_window_sec of the latest submit timestamp.
    """
    sess = df_all["session"].astype(str).str.strip()
    df_sess = df_all[sess == session].copy()
    if df_sess.empty:
        return set()

    df_sub = df_sess[df_sess["action"] == "submit"].copy()
    if df_sub.empty:
        return set()

    df_sub["ts_dt"] = df_sub["timestamp"].astype(str).apply(_parse_ts)
    df_sub = df_sub[df_sub["ts_dt"].notna()].copy()
    if df_sub.empty:
        oids = df_sess[df_sess["action"] == "submit"]["order_id"].astype(str).str.strip()
        # IMPORTANT: only uuid-like order ids are usable for polling
        return set([x for x in oids.tail(50).tolist() if x and _is_uuid_like(x)])

    df_sub = df_sub.sort_values("ts_dt")
    latest_dt: datetime = df_sub["ts_dt"].iloc[-1]
    cutoff = latest_dt.timestamp() - float(max(1, burst_window_sec))

    in_burst = df_sub[df_sub["ts_dt"].apply(lambda d: d.timestamp() >= cutoff)]
    oids = in_burst["order_id"].astype(str).str.strip()
    return set([x for x in oids.tolist() if x and _is_uuid_like(x)])


def _summarize_order_ids(df_all: pd.DataFrame, order_ids: Set[str], label: str) -> bool:
    if not order_ids:
        print(f"{label}: no usable broker order_ids found.")
        return False

    last = (
        df_all[df_all["order_id"].astype(str).str.strip().isin(order_ids)]
        .sort_values("timestamp")
        .groupby("order_id", as_index=False)
        .last()[["order_id", "status", "filled_qty", "filled_avg_price", "symbol", "qty"]]
    )

    last["qty"] = pd.to_numeric(last["qty"], errors="coerce").fillna(0).astype(int)
    last["filled_qty"] = pd.to_numeric(last["filled_qty"], errors="coerce").fillna(0).astype(int)
    last["symbol"] = last["symbol"].astype(str).str.strip().str.upper()
    last["status"] = (
        last["status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "unknown", "nan": "unknown"})
    )

    total_qty = int(last["qty"].sum())
    total_filled = int(last["filled_qty"].sum())
    fill_pct = (100.0 * total_filled / total_qty) if total_qty else 0.0
    print(f"{label}: total_qty={total_qty} total_filled={total_filled} fill_pct={fill_pct:.2f}%")

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
            str(r["symbol"]),
            int(r["qty"]),
            int(r["filled"]),
            f"{(100 * r['filled'] / denom):.1f}%",
            str(r["status"] or "unknown"),
        ]
        rows.append(row)
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    fmt = "  ".join("{:%d}" % w for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))

    return True


def _summarize_session(
    df_all: pd.DataFrame, session: str, summarize_all: bool, burst_window_sec: int
) -> bool:
    sess = df_all["session"].astype(str).str.strip()
    df_sess = df_all[sess == session].copy()
    if df_sess.empty:
        print(f"Session summary [{session}]: no rows found.")
        return False

    df_sess["action"] = df_sess["action"].astype(str).str.strip().str.lower()

    if summarize_all:
        submits = df_sess[df_sess["action"] == "submit"].copy()
        if not submits.empty:
            order_ids = set(submits["order_id"].astype(str).str.strip())
            order_ids = {x for x in order_ids if x and _is_uuid_like(x)}
        else:
            df_oid = df_sess[df_sess["order_id"].astype(str).str.strip() != ""]
            order_ids = set(df_oid["order_id"].astype(str).str.strip())
            order_ids = {x for x in order_ids if x and _is_uuid_like(x)}
            if not order_ids:
                print(f"Session summary [{session}]: no usable broker order_id rows found.")
                return True
            print(f"Session summary [{session}]: no submit rows found (fallback used).")

        return _summarize_order_ids(df_all, order_ids, label=f"Session summary [{session}] (ALL)")

    burst_ids = _latest_submit_burst_order_ids(df_all, session, burst_window_sec=burst_window_sec)
    if not burst_ids:
        # We can still display something useful: symbol list from submit rows (even if order_id missing)
        df_sub = df_sess[df_sess["action"] == "submit"].copy()
        if not df_sub.empty:
            syms = sorted(set(df_sub["symbol"].astype(str).str.strip().str.upper().tolist()))
            print(f"Session summary [{session}] (LATEST_BURST): no usable broker order_ids found.")
            print(f"Symbols in submit burst (log-only): {', '.join([s for s in syms if s])}")
            return True

        print(f"Session summary [{session}]: no submit rows found (and no usable order_ids).")
        return True

    return _summarize_order_ids(
        df_all, burst_ids, label=f"Session summary [{session}] (LATEST_BURST)"
    )


def _live_summary(
    open_orders: List[Dict[str, Any]], outstanding_orders: List[Dict[str, Any]], label: str
) -> bool:
    all_orders: List[Dict[str, Any]] = []
    all_orders.extend(open_orders or [])
    all_orders.extend(outstanding_orders or [])

    if not all_orders:
        print(f"Live summary [{label}]: no orders returned by broker.")
        return False

    rows = []
    for o in all_orders:
        oid = str(o.get("id") or o.get("order_id") or "").strip()
        sym = str(o.get("symbol") or "").strip().upper()
        side = str(o.get("side") or "").strip().lower()
        st = str(o.get("status") or "").strip().lower()
        typ = str(o.get("type") or o.get("order_type") or "").strip().lower()
        try:
            qty = int(float(o.get("qty", 0) or 0))
        except Exception:
            qty = 0
        try:
            fq = int(float(o.get("filled_qty", 0) or 0))
        except Exception:
            fq = 0
        rows.append((sym, side, qty, fq, st, typ, oid))

    total_qty = sum(r[2] for r in rows)
    total_filled = sum(r[3] for r in rows)
    fill_pct = (100.0 * total_filled / total_qty) if total_qty else 0.0
    print(
        f"Live summary [{label}]: orders={len(rows)} total_qty={total_qty} total_filled={total_filled} fill_pct={fill_pct:.2f}%"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll orders; tag sessions; summarize.")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument(
        "--session", default=None, help="Prefer/force session label for this run's summary."
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Poll broker and append log rows if changed."
    )

    parser.add_argument(
        "--summarize-all",
        action="store_true",
        help="Summarize ALL submits in a session (old behavior).",
    )
    parser.add_argument(
        "--burst-window-sec",
        type=int,
        default=60,
        help="For latest-burst summary: include submit rows within this many seconds of latest submit.",
    )

    args = parser.parse_args()

    _ensure_log()
    _upgrade_log_schema_if_needed()

    try:
        broker = AlpacaBroker(mode=args.mode)
    except Exception as e:
        print("Broker init failed:", e)
        return

    try:
        df_hist = _read_log_df(repair=True)
    except Exception as e:
        print("[BLOCK]", e)
        return

    session_map = _build_session_map(df_hist)
    latest_session = _latest_session_from_log(df_hist) or ""
    cli_session = (args.session or "").strip()

    # Open orders
    try:
        open_orders = broker.list_orders(status="open", nested=True, limit=500) or []
    except AlpacaError:
        open_orders = []

    open_ids = {str(o.get("id", "") or "").strip() for o in open_orders}

    # Outstanding IDs
    outstanding_ids: Set[str] = set()
    if cli_session:
        burst_ids = _latest_submit_burst_order_ids(
            df_hist, cli_session, burst_window_sec=args.burst_window_sec
        )
        outstanding_ids = {
            oid for oid in burst_ids if oid and oid not in open_ids and _is_uuid_like(oid)
        }
    else:
        if not df_hist.empty:
            last_by_id = df_hist.sort_values("timestamp").groupby("order_id", as_index=False).last()
            outstanding_ids = set(
                last_by_id[~last_by_id["status"].astype(str).str.lower().isin(TERMINAL)][
                    "order_id"
                ].astype(str)
            )
            outstanding_ids = {
                str(x).strip() for x in outstanding_ids if str(x).strip() and _is_uuid_like(str(x))
            }
        outstanding_ids = {
            oid for oid in outstanding_ids if oid and oid not in open_ids and _is_uuid_like(oid)
        }

    changes = 0
    outstanding_orders: List[Dict[str, Any]] = []

    if args.refresh:
        for o in open_orders:
            oid = str(o.get("id", "") or "").strip()
            last = _last_state(df_hist, oid)
            if _write_poll_if_changed(cli_session, session_map, latest_session, o, last):
                changes += 1

        for oid in sorted(outstanding_ids):
            try:
                o = broker.get_order(oid)
            except AlpacaError:
                continue
            if isinstance(o, dict):
                outstanding_orders.append(o)
                last = _last_state(df_hist, oid)
                if _write_poll_if_changed(cli_session, session_map, latest_session, o, last):
                    changes += 1

    print(
        f"Polled {len(open_orders)} open + {len(outstanding_ids)} outstanding; wrote {changes} changes."
    )

    # ------------------------------------------------------------
    # ✅ RECONCILIATION STEP (post-fill / post-change refresh)
    # ------------------------------------------------------------
    if changes > 0:
        try:
            print("[RECON] refreshing broker snapshots after status changes...")

            cmd = [
                sys.executable,
                "-m",
                "services.snapshot_live_orders",
                "--mode",
                args.mode,
            ]

            result = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            if result.returncode == 0:
                print("[RECON] snapshot_live_orders complete")
            else:
                print("[RECON_WARN] snapshot_live_orders failed:")
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)

        except Exception as e:
            print(f"[RECON_WARN] snapshot_live_orders exception: {e}")

        try:
            print("[RECON] refreshing signal lifecycle after status changes...")
            lc_cmd = [sys.executable, "-m", "services.apply_signal_lifecycle"]
            lc_result = subprocess.run(
                lc_cmd,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            if lc_result.returncode == 0:
                print("[RECON] apply_signal_lifecycle complete")
            else:
                print("[RECON_WARN] apply_signal_lifecycle failed:")
                if lc_result.stdout:
                    print(lc_result.stdout)
                if lc_result.stderr:
                    print(lc_result.stderr)
        except Exception as e:
            print(f"[RECON_WARN] apply_signal_lifecycle exception: {e}")

    try:
        df_all = _read_log_df(repair=True)
    except Exception as e:
        print("[BLOCK]", e)
        return

    latest_in_log = (_latest_session_from_log(df_all) or "").strip()
    chosen = (cli_session or latest_in_log or "").strip()

    if cli_session and latest_in_log and cli_session != latest_in_log:
        print(f"[INFO] Latest session in log is '{latest_in_log}' (you requested '{cli_session}').")

    if not chosen:
        print("No session found to summarize.")
        return

    had_rows = _summarize_session(
        df_all,
        chosen,
        summarize_all=bool(args.summarize_all),
        burst_window_sec=int(args.burst_window_sec),
    )

    if cli_session and not had_rows:
        had_live = _live_summary(
            open_orders=open_orders, outstanding_orders=outstanding_orders, label=cli_session
        )

        if (not had_live) and latest_in_log and latest_in_log != cli_session:
            print("\n[NO_ACTIVITY] No lifecycle actions executed → no orders placed.")
            print(
                f"\n[FALLBACK] Showing last known session '{latest_in_log}' for reference only.\n"
            )
            _summarize_session(
                df_all,
                latest_in_log,
                summarize_all=bool(args.summarize_all),
                burst_window_sec=int(args.burst_window_sec),
            )


if __name__ == "__main__":
    main()
