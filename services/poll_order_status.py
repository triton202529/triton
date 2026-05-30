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

PATCH (2026-05-26 ghost reconciliation):
- ✅ Detect & resolve "ghost" pending orders whose latest local status is
    pending_new / accepted / new but which are NOT in broker open orders AND
    NOT in the recent-orders snapshot. When such an order has been quiet for
    longer than --stale-pending-hours (default 24h = 1 trading day) it is
    marked as `canceled_stale` in live_orders_log.csv so:
      * subsequent polls ignore it (canceled_stale is TERMINAL),
      * downstream lifecycle (apply_signal_lifecycle / build_effective_lifecycle)
        no longer treats the symbol as having an in-flight order, and
      * POSITION_NOT_FOUND noise from manage_positions stops being triggered
        by stale local execution state.
    Safety guarantees:
      * Never touches filled / partially_filled orders (now or in history).
      * Never calls the broker to cancel — purely local state reconciliation.
      * Skipped automatically if the broker open-orders fetch was degraded
        this run (we will not classify ghosts off an unreliable snapshot).
"""

import argparse
import csv
import json
import os
import sys
import shutil
import time
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# Path bootstrap
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from services.broker_alpaca import AlpacaBroker, AlpacaError
from services.broker_call_resilience import (
    call_with_transient_retry,
    is_transient_broker_error,
    transient_failure_kind,
)
from services.notify import notify  # noqa: F401  (kept for compatibility)


# -----------------------------------------------------------
# Lifecycle-sync helpers
# Ensures signal_lifecycle_effective.csv is always refreshed
# immediately after signal_lifecycle.csv, eliminating repeated
# STALE_EFFECTIVE / LIFECYCLE_GATE_BLOCK events.
# -----------------------------------------------------------
def _run_module_checked(
    module_name: str,
    label: str,
    extra_args: Optional[List[str]] = None,
    *,
    allow_fail: bool = False,
) -> bool:
    """Run `python -m <module_name>`; return False on failure if allow_fail else raise."""
    cmd = [sys.executable, "-m", module_name] + list(extra_args or [])
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        if allow_fail:
            print(
                f"[POLL_RECON] {label} failed rc={result.returncode} (degraded, local state preserved)",
                flush=True,
            )
            if result.stdout:
                print(result.stdout, flush=True)
            if result.stderr:
                print(result.stderr, flush=True)
            return False
        raise RuntimeError(
            f"{label} failed rc={result.returncode}\n" f"{result.stdout}\n{result.stderr}"
        )
    return True


def _refresh_effective_lifecycle(*, allow_fail: bool = False) -> bool:
    """Rebuild `signal_lifecycle_effective.csv` right after `signal_lifecycle.csv`."""
    print("[RECON] refreshing effective lifecycle...")
    ok = _run_module_checked(
        "services.build_effective_lifecycle", "build_effective_lifecycle", allow_fail=allow_fail
    )
    if ok:
        print("[RECON] effective lifecycle refreshed")
    return bool(ok)


RESULTS_DIR = os.path.join("data", "results")
LIVE_ORDERS_LOG = os.path.join(RESULTS_DIR, "live_orders_log.csv")
POLL_FAILED_CACHE = os.path.join(RESULTS_DIR, "poll_failed_order_cache.json")
RECENT_ORDERS_SNAPSHOT = os.path.join(RESULTS_DIR, "recent_orders.csv")

STALE_HOURS_DEFAULT = 24.0
SUPPRESS_AFTER_FAILS_DEFAULT = 5
MAX_DETAIL_FAILS_PER_RUN_DEFAULT = 5
CACHE_RETENTION_DAYS = 30

# Default ghost-cleanup window: an order quiet for ≥ this many hours
# AND missing from both the broker open list and the recent-orders snapshot
# is treated as a local-state ghost (broker has no record of it).
STALE_PENDING_HOURS_DEFAULT = 24.0

# Statuses considered "in-flight" locally. If the broker can't see the order
# at all and the row hasn't moved in STALE_PENDING_HOURS_DEFAULT, the row is
# a ghost left behind from a prior degraded broker round-trip.
STALE_PENDING_STATUSES: Set[str] = {"pending_new", "accepted", "new"}

# Statuses we must NEVER reconcile away — touching these would erase real fills.
NEVER_RECONCILE_STATUSES: Set[str] = {"filled", "partially_filled"}

# Broker statuses in recent_orders.csv that mean "the broker is still
# working this order" — reconciling pending_new → canceled_stale here
# would lie about live broker state.
BROKER_ALIVE_RECENT_STATUSES: Set[str] = {
    "new",
    "accepted",
    "pending_new",
    "pending_cancel",
    "pending_replace",
    "held",
    "open",
    "submitted",
}

# Broker statuses in recent_orders.csv that prove a fill exists. Never
# overwrite these with canceled_stale; they require a real poll/repair.
BROKER_FILL_RECENT_STATUSES: Set[str] = {"filled", "partially_filled"}

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
    "filled_reconciled",  # locally imported broker fill (see _reconcile_ghost_pending_orders)
    "canceled",
    "canceled_stale",  # locally-reconciled ghost (see _reconcile_ghost_pending_orders)
    "expired",
    "rejected",
    "stopped",
    "suspended",
    "calculated",
    "replaced",
    "done_for_day",
}

# ---- Session-summary status buckets ---------------------------------
# These govern ONLY the session-summary / reporting layer; broker calls,
# polling, and execution logic remain unchanged.
#
# An order is "active" in a session summary iff its LATEST row (by
# parsed timestamp, not csv position) has a status in this set. Anything
# else is filtered out so reconciled ghosts (canceled_stale) and other
# terminal noise can never resurface as a fake "GLD pending_new" row.
ACTIVE_SUMMARY_STATUSES: Set[str] = {
    "pending_new",
    "accepted",
    "new",
    "partially_filled",
    "filled",
}

# Latest status in this set => the order is done; drop it from active view.
TERMINAL_SUMMARY_STATUSES: Set[str] = {
    "canceled",
    "canceled_stale",
    "filled_reconciled",  # broker-confirmed fill imported into local log
    "expired",
    "rejected",
    "failed",
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


# -----------------------------------------------------------
# Failed-order cache + classification
#
# Persists across runs at data/results/poll_failed_order_cache.json
# Used to suppress hammering broker API for old/stale unresolved IDs.
# -----------------------------------------------------------
def _load_failed_cache() -> Dict[str, Dict[str, Any]]:
    try:
        if not os.path.exists(POLL_FAILED_CACHE):
            return {}
        with open(POLL_FAILED_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                out[str(k)] = v
        return out
    except Exception:
        return {}


def _prune_failed_cache(cache: Dict[str, Dict[str, Any]], retention_days: int) -> None:
    if not cache or retention_days <= 0:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    drop: List[str] = []
    for oid, e in cache.items():
        last = _parse_ts(str(e.get("last_failed_at") or ""))
        if last is None or last < cutoff:
            drop.append(oid)
    for oid in drop:
        cache.pop(oid, None)


def _save_failed_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        tmp = POLL_FAILED_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
        os.replace(tmp, POLL_FAILED_CACHE)
    except Exception:
        pass


def _classify_failure(exc: BaseException) -> str:
    """
    Categorize a get_order failure for compact summary logging.

    Returns one of:
      - not_found       (broker has no record / 404)
      - network         (timeouts, DNS, resets, connection issues)
      - broker_error    (5xx or other AlpacaError HTTP errors)
      - auth            (401/403; preserved as a real safety signal)
      - unknown         (anything else)
    """
    if exc is None:
        return "unknown"
    if is_transient_broker_error(exc):
        return "network"
    s = str(exc) or ""
    sl = s.lower()
    if (
        " 401" in s
        or " 403" in s
        or "unauthorized" in sl
        or "forbidden" in sl
        or "authentication" in sl
    ):
        return "auth"
    if " 404" in s or "not found" in sl or "not_found" in sl:
        return "not_found"
    if any(code in s for code in (" 500", " 502", " 503", " 504")):
        return "broker_error"
    if type(exc).__name__ == "AlpacaError":
        return "broker_error"
    return "unknown"


def _record_failure(
    cache: Dict[str, Dict[str, Any]], oid: str, reason: str, now_iso_str: str
) -> Dict[str, Any]:
    e = cache.get(oid) or {}
    if "first_failed_at" not in e or not str(e.get("first_failed_at") or "").strip():
        e["first_failed_at"] = now_iso_str
    e["last_failed_at"] = now_iso_str
    e["last_failure_reason"] = reason
    e["failure_count"] = int(e.get("failure_count", 0)) + 1
    cache[oid] = e
    return e


def _clear_failure(cache: Dict[str, Dict[str, Any]], oid: str) -> None:
    cache.pop(oid, None)


def _last_ts_for_oid_map(df_hist: pd.DataFrame) -> Dict[str, datetime]:
    """Map of order_id -> latest log timestamp (UTC datetime)."""
    if df_hist is None or df_hist.empty:
        return {}
    df = df_hist[df_hist["order_id"].astype(str).str.strip() != ""].copy()
    if df.empty:
        return {}
    df["__ts_dt"] = df["timestamp"].astype(str).apply(_parse_ts)
    df = df[df["__ts_dt"].notna()]
    if df.empty:
        return {}
    last = df.sort_values("__ts_dt").groupby("order_id", as_index=False).last()
    return {str(r["order_id"]).strip(): r["__ts_dt"] for _, r in last.iterrows()}


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


def _latest_row_per_order_id(df_all: pd.DataFrame, order_ids: Set[str]) -> pd.DataFrame:
    """
    Returns one row per order_id — the row with the MAX parsed timestamp.

    Using idxmax on a parsed datetime guarantees the true latest row wins,
    independent of CSV write order or pandas groupby.last() per-column
    skipna behaviour. This is what makes canceled_stale supersede an older
    pending_new row in the same order_id's history.
    """
    if not order_ids or df_all is None or df_all.empty:
        return pd.DataFrame(
            columns=["order_id", "status", "filled_qty", "filled_avg_price", "symbol", "qty"]
        )

    sub = df_all[df_all["order_id"].astype(str).str.strip().isin(order_ids)].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["order_id", "status", "filled_qty", "filled_avg_price", "symbol", "qty"]
        )

    sub["__ts_dt"] = sub["timestamp"].astype(str).apply(_parse_ts)
    sub = sub[sub["__ts_dt"].notna()].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["order_id", "status", "filled_qty", "filled_avg_price", "symbol", "qty"]
        )

    # idxmax across the parsed datetime picks the single latest row per oid.
    idx = sub.groupby("order_id")["__ts_dt"].idxmax()
    cols = ["order_id", "status", "filled_qty", "filled_avg_price", "symbol", "qty"]
    return sub.loc[idx, cols].reset_index(drop=True)


def _summarize_order_ids(df_all: pd.DataFrame, order_ids: Set[str], label: str) -> bool:
    """
    Render the per-session active-orders table.

    Filtering rules (summary/reporting layer only — broker untouched):
      * Use the LATEST row per order_id (by parsed timestamp).
      * Keep rows whose latest status is in ACTIVE_SUMMARY_STATUSES.
      * Drop rows whose latest status is in TERMINAL_SUMMARY_STATUSES
        (counted as `terminal_filtered`) — this is what hides reconciled
        ghosts like GLD canceled_stale from the LATEST_BURST view.
      * Drop everything else as `stale_filtered` (unknown / submitted /
        empty / etc.) so we never surface stale historical state as active.
    """
    if not order_ids:
        print(f"{label}: no usable broker order_ids found.")
        print(
            "[SESSION_SUMMARY_FILTER] active_rows=0 terminal_filtered=0 stale_filtered=0",
            flush=True,
        )
        return False

    last = _latest_row_per_order_id(df_all, order_ids)
    if last.empty:
        print(f"{label}: no rows match given order_ids.")
        print(
            "[SESSION_SUMMARY_FILTER] active_rows=0 terminal_filtered=0 stale_filtered=0",
            flush=True,
        )
        return True

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

    # ── Classify each order_id by its LATEST status ──────────────────
    is_active = last["status"].isin(ACTIVE_SUMMARY_STATUSES)
    is_terminal = last["status"].isin(TERMINAL_SUMMARY_STATUSES)
    is_stale = (~is_active) & (~is_terminal)

    active_rows = int(is_active.sum())
    terminal_filtered = int(is_terminal.sum())
    stale_filtered = int(is_stale.sum())

    print(
        f"[SESSION_SUMMARY_FILTER] active_rows={active_rows} "
        f"terminal_filtered={terminal_filtered} stale_filtered={stale_filtered}",
        flush=True,
    )

    last_active = last[is_active].copy()
    if last_active.empty:
        print(
            f"{label}: no active orders (all {terminal_filtered + stale_filtered} "
            f"order_ids are terminal/stale)."
        )
        return True

    total_qty = int(last_active["qty"].sum())
    total_filled = int(last_active["filled_qty"].sum())
    fill_pct = (100.0 * total_filled / total_qty) if total_qty else 0.0
    print(f"{label}: total_qty={total_qty} total_filled={total_filled} fill_pct={fill_pct:.2f}%")

    br = last_active.groupby("symbol", as_index=False).agg(
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


# -----------------------------------------------------------
# Ghost-order reconciliation
#
# An order is a "ghost" when:
#   - its latest row in live_orders_log.csv shows pending_new / accepted / new
#   - it is NOT in the broker's current open-orders list
#   - it is NOT in the recent-orders snapshot from snapshot_live_orders
#   - the row has been quiet for longer than the stale threshold
#
# These rows survive degraded round-trips (e.g. a submit got through but
# the broker confirmation never landed; or a manual cancel happened at the
# broker side). They cause manage_positions to chase POSITION_NOT_FOUND
# and pollute lifecycle gating. We resolve them locally by appending a
# terminal `canceled_stale` row — we never call the broker to cancel.
# -----------------------------------------------------------
def _load_recent_orders_broker_map() -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Returns {order_id: {"status", "filled_qty", "filled_avg_price"}} from
    recent_orders.csv (i.e. broker truth as captured by snapshot_live_orders),
    or None when the snapshot is missing / malformed so the caller can fall
    back to the broker `not_found` signal. An empty {} is a valid result —
    the snapshot is present and broker reported zero recent orders.

    The fill data is carried so a pending_new row whose oid appears in
    recent_orders.csv with broker_status=filled / partially_filled can be
    reconciled into a `filled_reconciled` row containing the broker's real
    filled_qty / filled_avg_price (see _reconcile_ghost_pending_orders).
    """
    if not os.path.exists(RECENT_ORDERS_SNAPSHOT):
        return None
    try:
        df = pd.read_csv(RECENT_ORDERS_SNAPSHOT, keep_default_na=False)
    except Exception:
        return None

    if df.empty:
        return {}

    # snapshot_live_orders writes 'id' or 'order_id' depending on the
    # broker payload shape; accept both.
    id_col = next((c for c in ("id", "order_id") if c in df.columns), None)
    if id_col is None or "status" not in df.columns:
        # Malformed snapshot: refuse to classify based on it.
        return None

    out: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        oid = str(r.get(id_col, "") or "").strip()
        if not oid or not _is_uuid_like(oid):
            continue
        out[oid] = {
            "status": str(r.get("status", "") or "").strip().lower(),
            "filled_qty": r.get("filled_qty", "") if "filled_qty" in df.columns else "",
            "filled_avg_price": (
                r.get("filled_avg_price", "") if "filled_avg_price" in df.columns else ""
            ),
        }
    return out


def _reconcile_ghost_pending_orders(
    *,
    df_hist: pd.DataFrame,
    open_ids: Set[str],
    recent_broker_map: Optional[Dict[str, Dict[str, Any]]],
    failed_cache: Dict[str, Dict[str, Any]],
    cli_session: str,
    session_map: Dict[str, str],
    latest_session: str,
    stale_hours: float,
    open_orders_fetched_ok: bool,
) -> int:
    """
    Append terminal-state reconciliation rows for ghost / contradicted
    pending orders. Returns total reconciliation rows written
    (canceled_stale + filled_reconciled).

    Two reconciliation flavors, both append-only (history is never edited):

      A. `canceled_stale` — broker has no live record of the order
         (either missing from recent_orders snapshot, or recent_orders shows
         a broker-terminal status like canceled/expired/rejected) AND the
         local row is older than --stale-pending-hours.

      B. `filled_reconciled` — local row says pending_new / accepted / new
         but recent_orders.csv proves the broker filled (or partially filled)
         this order_id. Local state is updated to reflect broker truth using
         the broker's filled_qty / filled_avg_price so downstream lifecycle
         and outcomes stop treating the symbol as in-flight.

    Logging:
      [STALE_ORDER_CLEANUP]      one line per canceled_stale reconciliation
      [FILLED_STATE_RECONCILE]   one line per filled_reconciled reconciliation
      [STALE_ORDER_CLEANUP_SKIP] one line per candidate that was skipped, with
                                 the concrete reason — e.g. broker_alive_in_recent,
                                 too_young, filled_in_history, etc.
      [STALE_ORDER_CLEANUP_SUMMARY] single end-of-pass summary with both counts.

    Safety:
      - Returns 0 if the broker open-orders fetch was degraded this run.
      - Skips any order_id whose ANY historical row was filled/partially_filled.
      - Never modifies broker state; never rewrites existing rows.
    """
    if not open_orders_fetched_ok:
        return 0
    if df_hist is None or df_hist.empty:
        return 0
    if stale_hours <= 0:
        return 0

    df = df_hist[df_hist["order_id"].astype(str).str.strip() != ""].copy()
    if df.empty:
        return 0
    df["__ts_dt"] = df["timestamp"].astype(str).apply(_parse_ts)
    df = df[df["__ts_dt"].notna()]
    if df.empty:
        return 0
    df_sorted = df.sort_values("__ts_dt")
    latest_per_oid = df_sorted.groupby("order_id", as_index=False).last()

    # Order_ids that EVER reached a fill state — sacred, never reconcile
    # even if their latest log row somehow regressed to a pending status.
    ever_filled_ids: Set[str] = set(
        df_sorted[df_sorted["status"].isin(NEVER_RECONCILE_STATUSES)]["order_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    now_dt = datetime.now(timezone.utc)
    ghosts_written = 0  # canceled_stale rows appended
    filled_reconciled_written = 0  # filled_reconciled rows appended

    def _emit_skip(symbol: str, oid: str, status: str, reason: str) -> None:
        print(
            f"[STALE_ORDER_CLEANUP_SKIP] symbol={symbol or '?'} "
            f"order_id={oid} "
            f"status={status} "
            f"reason={reason}",
            flush=True,
        )

    def _coerce_int(val: Any) -> int:
        try:
            return int(float(val))
        except Exception:
            return 0

    for _, row in latest_per_oid.iterrows():
        oid = str(row.get("order_id") or "").strip()
        if not oid:
            continue
        status = str(row.get("status") or "").strip().lower()
        # Only orders whose LATEST status is pending_new / accepted / new
        # are candidates. Everything else is silently skipped (not "skipped"
        # in the diagnostic sense — they were never candidates).
        if status not in STALE_PENDING_STATUSES:
            continue

        symbol = str(row.get("symbol") or "").strip().upper()

        if not _is_uuid_like(oid):
            _emit_skip(symbol, oid, status, "non_uuid_order_id")
            continue
        if oid in ever_filled_ids:
            _emit_skip(symbol, oid, status, "filled_in_history")
            continue
        if oid in open_ids:
            _emit_skip(symbol, oid, status, "currently_open_at_broker")
            continue

        # Defensive: never reconcile a pending row that already carries a fill.
        try:
            filled_qty = int(float(row.get("filled_qty", 0) or 0))
        except Exception:
            filled_qty = 0
        if filled_qty > 0:
            _emit_skip(symbol, oid, status, "filled_qty_nonzero")
            continue

        last_ts = row.get("__ts_dt")
        try:
            age_h = (now_dt - last_ts).total_seconds() / 3600.0
        except Exception:
            _emit_skip(symbol, oid, status, "unparseable_timestamp")
            continue
        if age_h < stale_hours:
            _emit_skip(
                symbol,
                oid,
                status,
                f"too_young age_hours={age_h:.2f} threshold={stale_hours:.2f}",
            )
            continue

        # ── Broker cross-check via recent_orders.csv ──────────────────
        # recent_orders.csv (status="all") legitimately contains every
        # broker-side state. Classify the broker's view of this oid:
        #   - alive   → skip (broker is still working it)
        #   - filled  → BROKER-CONFIRMED FILL: reconcile local pending_new
        #                to filled_reconciled, importing broker fill data
        #   - terminal-but-not-filled (canceled/expired/rejected/etc.)
        #     or missing-from-snapshot → reconcile to canceled_stale
        broker_status: Optional[str] = None
        broker_filled_qty: Optional[int] = None
        broker_filled_avg: str = ""
        broker_row = recent_broker_map.get(oid) if recent_broker_map is not None else None
        if broker_row is not None:
            broker_status = str(broker_row.get("status") or "").strip().lower() or None
            broker_filled_qty = _coerce_int(broker_row.get("filled_qty"))
            broker_filled_avg = str(broker_row.get("filled_avg_price") or "").strip()

        if recent_broker_map is None:
            # No snapshot to cross-check; require an explicit "not_found"
            # from a get_order attempt before we touch local state.
            cache_e = failed_cache.get(oid) or {}
            broker_says_not_found = (
                str(cache_e.get("last_failure_reason") or "").strip().lower() == "not_found"
            )
            if not broker_says_not_found:
                _emit_skip(
                    symbol,
                    oid,
                    status,
                    "no_recent_snapshot_and_no_not_found",
                )
                continue
            # Fall through to canceled_stale path below.
        else:
            if broker_status in BROKER_ALIVE_RECENT_STATUSES:
                _emit_skip(
                    symbol,
                    oid,
                    status,
                    f"broker_alive_in_recent broker_status={broker_status}",
                )
                continue

            if broker_status in BROKER_FILL_RECENT_STATUSES:
                # ── Filled-state reconciliation ─────────────────────
                # Broker truth says this order filled (or partially); local
                # row was stuck at pending_new. Append a terminal local row
                # that mirrors the broker's fill so lifecycle/outcomes stop
                # treating the symbol as in-flight. We DO NOT call the
                # broker, DO NOT rewrite the older pending_new row, and DO
                # NOT touch any other history.
                side_f = str(row.get("side") or "").strip().lower()
                try:
                    qty_f = int(float(row.get("qty", 0) or 0))
                except Exception:
                    qty_f = 0
                otype_f = str(row.get("type") or "").strip().lower()
                limit_price_f = row.get("limit_price", "")
                client_order_id_f = row.get("client_order_id", "") or ""
                tp_limit_f = row.get("tp_limit", "") or ""
                sl_stop_f = row.get("sl_stop", "") or ""

                # Prefer broker's filled_qty; fall back to the row's qty so the
                # reconciliation row is never misleadingly zero.
                fq_final = int(broker_filled_qty or 0)
                if fq_final <= 0 and broker_status == "filled":
                    fq_final = qty_f
                fap_final = broker_filled_avg or (row.get("filled_avg_price", "") or "")

                sess_f = _choose_session(cli_session, oid, session_map, latest_session)

                _append_log_row_dict(
                    {
                        "timestamp": _now_iso(),
                        "session": sess_f,
                        "action": "poll",
                        "symbol": symbol,
                        "side": side_f,
                        "qty": qty_f,
                        "type": otype_f,
                        "limit_price": limit_price_f,
                        "order_id": oid,
                        "status": "filled_reconciled",
                        "filled_qty": fq_final,
                        "filled_avg_price": fap_final,
                        "client_order_id": client_order_id_f,
                        "tp_limit": tp_limit_f,
                        "sl_stop": sl_stop_f,
                    }
                )
                filled_reconciled_written += 1
                print(
                    f"[FILLED_STATE_RECONCILE] symbol={symbol or '?'} "
                    f"order_id={oid} "
                    f"old_status={status} "
                    f"broker_status={broker_status} "
                    f"action=mark_filled_reconciled "
                    f"broker_filled_qty={fq_final} "
                    f"broker_filled_avg_price={fap_final or 'NA'} "
                    f"age_hours={age_h:.2f} "
                    f"reconciled_from_pending=True "
                    f"broker_confirmed_fill=True",
                    flush=True,
                )
                continue

            # broker_status is None (oid missing from snapshot) or a
            # broker-terminal status (canceled / expired / rejected /
            # done_for_day / replaced / suspended / stopped / calculated)
            # → safe to mark canceled_stale below.

        # ── Reconcile: append canceled_stale terminal row ─────────────
        side = str(row.get("side") or "").strip().lower()
        try:
            qty = int(float(row.get("qty", 0) or 0))
        except Exception:
            qty = 0
        otype = str(row.get("type") or "").strip().lower()
        limit_price = row.get("limit_price", "")
        filled_avg_price = row.get("filled_avg_price", "") or ""
        client_order_id = row.get("client_order_id", "") or ""
        tp_limit = row.get("tp_limit", "") or ""
        sl_stop = row.get("sl_stop", "") or ""

        sess = _choose_session(cli_session, oid, session_map, latest_session)

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
                "order_id": oid,
                "status": "canceled_stale",
                "filled_qty": filled_qty,
                "filled_avg_price": filled_avg_price,
                "client_order_id": client_order_id,
                "tp_limit": tp_limit,
                "sl_stop": sl_stop,
            }
        )
        ghosts_written += 1
        print(
            f"[STALE_ORDER_CLEANUP] symbol={symbol or '?'} "
            f"old_status={status} "
            f"action=mark_stale_cancelled "
            f"age_hours={age_h:.2f} "
            f"order_id={oid} "
            f"broker_recent_status={broker_status or 'not_in_snapshot'} "
            f"stale_missing_at_broker=True",
            flush=True,
        )

    total_written = ghosts_written + filled_reconciled_written
    if total_written > 0:
        print(
            f"[STALE_ORDER_CLEANUP_SUMMARY] canceled_stale={ghosts_written} "
            f"filled_reconciled={filled_reconciled_written} "
            f"threshold_hours={stale_hours:.2f}",
            flush=True,
        )
    return total_written


def main() -> int:
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
    parser.add_argument(
        "--stale-hours",
        type=float,
        default=STALE_HOURS_DEFAULT,
        help="Outstanding ids whose last log timestamp is older than this (and not currently open at broker) "
        "are suppressed from polling instead of being hammered.",
    )
    parser.add_argument(
        "--suppress-after-fails",
        type=int,
        default=SUPPRESS_AFTER_FAILS_DEFAULT,
        help="Once an outstanding id (not currently open at broker) has failed at least this many times across runs, "
        "stop calling get_order on it.",
    )
    parser.add_argument(
        "--detail-max-fails-per-run",
        type=int,
        default=MAX_DETAIL_FAILS_PER_RUN_DEFAULT,
        help="Print full per-id failure lines for at most this many failures per run; the rest go to the summary.",
    )
    parser.add_argument(
        "--stale-pending-hours",
        type=float,
        default=STALE_PENDING_HOURS_DEFAULT,
        help="Pending-state rows (pending_new/accepted/new) missing from broker open orders AND "
        "recent-orders snapshot for at least this many hours are reconciled to canceled_stale.",
    )
    parser.add_argument(
        "--no-cleanup-stale-pending",
        dest="cleanup_stale_pending",
        action="store_false",
        help="Disable ghost-order reconciliation for this run (default: enabled).",
    )
    parser.set_defaults(cleanup_stale_pending=True)

    args = parser.parse_args()

    rstats: Dict[str, int] = {
        "attempted": 0,
        "retried": 0,
        "degraded": 0,
        "updated": 0,
        "failed_ids": 0,
    }
    bcounts: Dict[str, int] = {"retried": 0}
    step_degraded: Optional[str] = None

    _ensure_log()
    _upgrade_log_schema_if_needed()

    broker: Any = None
    try:
        broker = call_with_transient_retry(
            "broker_init", lambda: AlpacaBroker(mode=args.mode), out_counts=bcounts
        )
    except Exception as e:
        rstats["degraded"] = 1
        if is_transient_broker_error(e):
            step_degraded = transient_failure_kind(e)
        else:
            step_degraded = "network"
        print(f"[POLL] broker init failed: {e}", flush=True)
        print(
            f"[POLL] DEGRADED reason={step_degraded} (broker init)",
            flush=True,
        )
        print(
            f"[POLL_STATUS_SUMMARY] attempted={rstats['attempted']} retried={bcounts.get('retried', 0)} "
            f"degraded=1 updated=0 failed_ids=0",
            flush=True,
        )
        return 1

    try:
        df_hist = _read_log_df(repair=True)
    except Exception as e:
        print(f"[BLOCK] {e}", flush=True)
        print(
            f"[POLL_STATUS_SUMMARY] attempted={rstats['attempted']} retried={bcounts.get('retried', 0)} "
            f"degraded=1 updated=0 failed_ids=0",
            flush=True,
        )
        return 2

    session_map = _build_session_map(df_hist)
    latest_session = _latest_session_from_log(df_hist) or ""
    cli_session = (args.session or "").strip()

    # Open orders
    rstats["attempted"] = 1
    open_orders: List[Dict[str, Any]] = []
    open_orders_fetched_ok = False
    try:
        open_orders = call_with_transient_retry(
            "list_orders",
            lambda: broker.list_orders(status="open", nested=True, limit=500) or [],
            out_counts=bcounts,
        )
        open_orders_fetched_ok = True
    except Exception as e:
        rstats["degraded"] = 1
        rstats["failed_ids"] = 0
        if is_transient_broker_error(e) and not step_degraded:
            step_degraded = transient_failure_kind(e)
        print(f"[POLL] list_orders failed after retries: {e}", flush=True)
        try:
            print(
                f"[POLL] step DEGRADED reason={step_degraded or 'broker_timeout'} (empty open set)",
                flush=True,
            )
        except Exception:
            pass
        open_orders = []

    open_ids = {str(o.get("id", "") or "").strip() for o in open_orders}
    open_ids = {x for x in open_ids if x}

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

    # ----------------------------------------------------------------
    # Stale-id suppression: do not hammer the broker for old IDs that
    # are not currently open and that have either aged out of our window
    # or failed repeatedly across prior runs. We KEEP the cache and KEEP
    # the log untouched; we only stop calling get_order on them.
    # ----------------------------------------------------------------
    failed_cache = _load_failed_cache()
    last_ts_map = _last_ts_for_oid_map(df_hist)
    now_dt = datetime.now(timezone.utc)
    stale_hours = float(args.stale_hours)
    suppress_after = int(args.suppress_after_fails)
    detail_budget = int(args.detail_max_fails_per_run)

    suppressed_ids: Set[str] = set()
    suppressed_reasons: Dict[str, str] = {}
    pollable_outstanding: List[str] = []

    for oid in sorted(outstanding_ids):
        cache_e = failed_cache.get(oid) or {}
        fcount = int(cache_e.get("failure_count", 0) or 0)
        last_log_ts = last_ts_map.get(oid)
        age_h: Optional[float] = None
        if last_log_ts is not None:
            try:
                age_h = (now_dt - last_log_ts).total_seconds() / 3600.0
            except Exception:
                age_h = None
        not_open_now = oid not in open_ids
        if not_open_now and fcount >= suppress_after:
            suppressed_ids.add(oid)
            suppressed_reasons[oid] = "repeated_failures"
            continue
        if not_open_now and age_h is not None and age_h >= stale_hours:
            suppressed_ids.add(oid)
            suppressed_reasons[oid] = "stale_old_order"
            continue
        pollable_outstanding.append(oid)

    changes = 0
    outstanding_orders: List[Dict[str, Any]] = []
    by_reason: Dict[str, int] = {}
    fresh_failed = 0
    stale_failed = 0
    auth_failed = 0
    detail_printed = 0

    if args.refresh:
        for o in open_orders:
            oid = str(o.get("id", "") or "").strip()
            last = _last_state(df_hist, oid)
            if _write_poll_if_changed(cli_session, session_map, latest_session, o, last):
                changes += 1
            # broker confirms this id; clear any prior failure history.
            if oid:
                _clear_failure(failed_cache, oid)

        for oid in pollable_outstanding:
            o: Optional[Dict[str, Any]] = None
            try:
                o = call_with_transient_retry(
                    f"get_order({oid})",
                    lambda oi=oid: broker.get_order(oi),
                    out_counts=bcounts,
                )
            except Exception as e:
                reason = _classify_failure(e)
                _record_failure(failed_cache, oid, reason, _now_iso())
                by_reason[reason] = by_reason.get(reason, 0) + 1
                rstats["failed_ids"] = int(rstats.get("failed_ids", 0)) + 1

                last_log_ts = last_ts_map.get(oid)
                age_h2: Optional[float] = None
                if last_log_ts is not None:
                    try:
                        age_h2 = (now_dt - last_log_ts).total_seconds() / 3600.0
                    except Exception:
                        age_h2 = None
                is_open_now = oid in open_ids
                is_fresh = is_open_now or (age_h2 is not None and age_h2 < stale_hours)

                # Auth errors always count as "fresh" because they reflect a real
                # broker-credential problem we must not hide.
                if reason == "auth":
                    auth_failed += 1
                    fresh_failed += 1
                elif is_fresh:
                    fresh_failed += 1
                else:
                    stale_failed += 1

                if is_transient_broker_error(e) and not step_degraded:
                    step_degraded = transient_failure_kind(e)

                if detail_printed < detail_budget:
                    print(
                        f"[POLL] get_order skipped id={oid} after retries: "
                        f"type={type(e).__name__} reason={reason} "
                        f"fresh={'1' if is_fresh else '0'} "
                        f"failure_count={int(failed_cache.get(oid, {}).get('failure_count', 0))}",
                        flush=True,
                    )
                    detail_printed += 1
                continue
            if isinstance(o, dict):
                outstanding_orders.append(o)
                _clear_failure(failed_cache, oid)
                last = _last_state(df_hist, oid)
                if _write_poll_if_changed(cli_session, session_map, latest_session, o, last):
                    changes += 1

    # Apply degradation rule: do NOT degrade just because old historical IDs
    # are gone. Only fresh/open failures (or auth errors) should degrade.
    if fresh_failed > 0 or auth_failed > 0:
        rstats["degraded"] = 1

    rstats["updated"] = changes
    rstats["retried"] = int(bcounts.get("retried", 0))

    # ----------------------------------------------------------------
    # Compact failure summary (replaces what used to be hundreds of
    # repeated [POLL] get_order skipped lines).
    # ----------------------------------------------------------------
    by_reason_sorted = dict(sorted(by_reason.items()))
    print(
        f"[POLL_FAILURE_SUMMARY] failed_ids={int(rstats.get('failed_ids', 0))} "
        f"suppressed_ids={len(suppressed_ids)} "
        f"fresh_failed={fresh_failed} stale_failed={stale_failed} "
        f"by_reason={by_reason_sorted}",
        flush=True,
    )
    if suppressed_ids:
        suppressed_breakdown: Dict[str, int] = {}
        for r in suppressed_reasons.values():
            suppressed_breakdown[r] = suppressed_breakdown.get(r, 0) + 1
        print(
            f"[POLL_SUPPRESSION] count={len(suppressed_ids)} "
            f"by_reason={dict(sorted(suppressed_breakdown.items()))} "
            f"stale_hours={stale_hours} suppress_after_fails={suppress_after}",
            flush=True,
        )

    # Trim cache to keep the file small over time and persist it.
    _prune_failed_cache(failed_cache, CACHE_RETENTION_DAYS)
    _save_failed_cache(failed_cache)

    print(
        f"Polled {len(open_orders)} open + {len(outstanding_ids)} outstanding "
        f"(suppressed={len(suppressed_ids)}, polled={len(pollable_outstanding)}); wrote {changes} changes."
    )
    if step_degraded and int(rstats.get("degraded", 0)):
        print(
            f"[POLL] DEGRADED reason={step_degraded} (broker/network — local log and prior state kept)",
            flush=True,
        )

    # ------------------------------------------------------------
    # GHOST-ORDER RECONCILIATION
    # Marks orphaned pending_new/accepted/new rows as canceled_stale
    # when the broker no longer knows about them. Runs BEFORE the
    # snapshot/lifecycle refresh below so the new terminal rows are
    # immediately reflected in apply_signal_lifecycle output.
    # ------------------------------------------------------------
    ghost_changes = 0
    if args.cleanup_stale_pending:
        try:
            df_for_ghosts = _read_log_df(repair=False)
        except Exception as e:
            print(f"[STALE_ORDER_CLEANUP] skipped: cannot reread log ({e})", flush=True)
            df_for_ghosts = None

        if df_for_ghosts is not None:
            recent_broker_map = _load_recent_orders_broker_map()
            if recent_broker_map is None:
                print(
                    "[STALE_ORDER_CLEANUP] recent_orders.csv unavailable/malformed — "
                    "falling back to broker not_found signal only",
                    flush=True,
                )
            ghost_changes = _reconcile_ghost_pending_orders(
                df_hist=df_for_ghosts,
                open_ids=open_ids,
                recent_broker_map=recent_broker_map,
                failed_cache=failed_cache,
                cli_session=cli_session,
                session_map=session_map,
                latest_session=latest_session,
                stale_hours=float(args.stale_pending_hours),
                open_orders_fetched_ok=open_orders_fetched_ok,
            )
            if ghost_changes > 0:
                # The function itself emits [STALE_ORDER_CLEANUP_SUMMARY] with
                # the per-flavor breakdown (canceled_stale vs filled_reconciled).
                changes += ghost_changes
                rstats["updated"] = changes

    # ------------------------------------------------------------
    # RECONCILIATION STEP (post-fill / post-change refresh)
    # ------------------------------------------------------------
    if changes > 0:
        print("[RECON] refreshing broker snapshots after status changes...")
        recon_degraded = not _run_module_checked(
            "services.snapshot_live_orders",
            "snapshot_live_orders",
            extra_args=["--mode", args.mode],
            allow_fail=True,
        )
        if not recon_degraded:
            print("[RECON] snapshot_live_orders complete")
        if recon_degraded:
            rstats["degraded"] = 1

        print("[RECON] refreshing signal lifecycle after status changes...")
        r2 = _run_module_checked(
            "services.apply_signal_lifecycle", "apply_signal_lifecycle", allow_fail=True
        )
        if r2:
            print("[RECON] apply_signal_lifecycle complete")
        if not r2:
            rstats["degraded"] = 1
        if not _refresh_effective_lifecycle(allow_fail=True):
            rstats["degraded"] = 1

    try:
        df_all = _read_log_df(repair=True)
    except Exception as e:
        print("[BLOCK]", e)
        print(
            f"[POLL_STATUS_SUMMARY] attempted={rstats['attempted']} retried={rstats.get('retried', 0)} "
            f"degraded=1 updated={rstats.get('updated', 0)} failed_ids={rstats.get('failed_ids', 0)}",
            flush=True,
        )
        return 2

    latest_in_log = (_latest_session_from_log(df_all) or "").strip()
    chosen = (cli_session or latest_in_log or "").strip()

    if cli_session and latest_in_log and cli_session != latest_in_log:
        print(f"[INFO] Latest session in log is '{latest_in_log}' (you requested '{cli_session}').")

    if not chosen:
        print("No session found to summarize.")
        ret = 1 if int(rstats.get("degraded", 0)) else 0
        print(
            f"[POLL_STATUS_SUMMARY] attempted={rstats['attempted']} retried={rstats.get('retried', 0)} "
            f"degraded={int(rstats.get('degraded', 0))} updated={rstats.get('updated', 0)} "
            f"failed_ids={rstats.get('failed_ids', 0)}",
            flush=True,
        )
        return ret

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

    ret = 1 if int(rstats.get("degraded", 0)) else 0
    print(
        f"[POLL_STATUS_SUMMARY] attempted={rstats['attempted']} retried={rstats.get('retried', 0)} "
        f"degraded={int(rstats.get('degraded', 0))} updated={rstats.get('updated', 0)} "
        f"failed_ids={rstats.get('failed_ids', 0)}",
        flush=True,
    )
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
