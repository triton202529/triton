# services/session_fill_pressure.py
"""Session-scoped fill stats from live_orders_log.csv + last_execution_session.json. Read-only; never raises to callers."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LAST_EXECUTION_SESSION_JSON = RESULTS / "last_execution_session.json"
SESSION_FILL_PRESSURE_JSON = RESULTS / "session_fill_pressure.json"
LIVE_ORDERS_LOG_CSV = RESULTS / "live_orders_log.csv"

# Terminal / open buckets (Alpaca-style lowercase)
_OPEN_STATUSES = frozenset(
    {
        "new",
        "accepted",
        "pending",
        "partially_filled",
        "submitted",
        "held",
        "open",
        "pending_new",
        "pending_cancel",
        "pending_replace",
    }
)


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts or not str(ts).strip():
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _is_uuid_like(s: str) -> bool:
    s = str(s or "").strip()
    if len(s) < 20:
        return False
    if s.count("-") >= 3 and any(c.isalpha() for c in s):
        return True
    return False


def write_last_execution_session_snapshot(
    *,
    session: str,
    mode: str,
    orders_planned: int,
    dry_run: bool,
    opportunities_seen: int,
) -> None:
    """Persist the latest execute_trades session id for session_fill_pressure (best-effort)."""
    try:
        payload = {
            "timestamp": utc_iso(),
            "session": str(session or "").strip(),
            "mode": str(mode or ""),
            "orders_planned": int(orders_planned),
            "dry_run": bool(dry_run),
            "opportunities_seen": int(opportunities_seen),
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        LAST_EXECUTION_SESSION_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _read_last_session() -> Optional[Dict[str, Any]]:
    try:
        if (
            not LAST_EXECUTION_SESSION_JSON.is_file()
            or LAST_EXECUTION_SESSION_JSON.stat().st_size == 0
        ):
            return None
        o = json.loads(LAST_EXECUTION_SESSION_JSON.read_text(encoding="utf-8", errors="replace"))
        return o if isinstance(o, dict) else None
    except Exception:
        return None


def _session_family(session: str) -> str:
    s = str(session or "").strip()
    if s.startswith("reprice_"):
        return "reprice"
    if s.startswith("exec_trades_"):
        return "exec_trades"
    return "other"


def _session_max_ts_map(df: pd.DataFrame) -> Dict[str, datetime]:
    """Per-session max(timestamp) for exec_trades_* and reprice_* rows only."""
    out: Dict[str, datetime] = {}
    if df is None or df.empty or "session" not in df.columns or "timestamp" not in df.columns:
        return out
    s = df["session"].astype(str).str.strip()
    m = s.str.startswith("exec_trades_", na=False) | s.str.startswith("reprice_", na=False)
    sub = df.loc[m].copy()
    if sub.empty:
        return out
    sub["_ts"] = sub["timestamp"].astype(str).apply(_parse_ts)
    sub = sub[sub["_ts"].notna()]
    if sub.empty:
        return out
    for sess, g in sub.groupby(sub["session"].astype(str).str.strip()):
        k = str(sess).strip()
        mx = g["_ts"].max()
        out[k] = mx
    return out


def _choose_session_auto(df: pd.DataFrame, meta: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    """
    Pick session for observability:
    - If last_execution_session.json points to exec_trades_* and that session's max_ts in log
      is strictly greater than the max timestamp across all reprice_* sessions, keep it.
    - Otherwise use the session with the latest max(timestamp) among exec_trades_* and reprice_*.
    """
    by_ts = _session_max_ts_map(df)
    if not by_ts:
        return "", "no_exec_trades_or_reprice_in_log", ""

    reprice_ts_map = {k: v for k, v in by_ts.items() if k.startswith("reprice_")}
    max_reprice = max(reprice_ts_map.values()) if reprice_ts_map else None

    meta_s = (meta.get("session") or "").strip() if isinstance(meta, dict) else ""
    if meta_s.startswith("exec_trades_") and meta_s in by_ts:
        meta_ts = by_ts[meta_s]
        if max_reprice is None or meta_ts > max_reprice:
            return meta_s, "last_execution_session_newer_than_reprice", "exec_trades"

    best_sess, _ = max(by_ts.items(), key=lambda kv: kv[1])
    stype = _session_family(best_sess)
    return best_sess, "latest_activity_in_log", stype if stype != "other" else "exec_trades"


def _empty_payload(
    sess: str,
    meta: Optional[Dict[str, Any]],
    selection_rule: str,
    session_type: str,
    notes: List[str],
) -> Dict[str, Any]:
    meta_session = (meta.get("session") or "").strip() if isinstance(meta, dict) else ""
    orders_planned = int(meta.get("orders_planned") or 0) if meta and meta_session == sess else 0
    mode = str(meta.get("mode") or "") if meta and meta_session == sess else ""
    dry_run = bool(meta.get("dry_run")) if meta and meta_session == sess else False
    return {
        "timestamp": utc_iso(),
        "session": sess,
        "session_type": session_type or (_session_family(sess) if sess else ""),
        "session_selection_rule": selection_rule,
        "mode": mode,
        "dry_run_plan": dry_run,
        "orders_planned": orders_planned,
        "orders_submitted": 0,
        "orders_filled": 0,
        "orders_open": 0,
        "orders_canceled": 0,
        "orders_rejected": 0,
        "fill_rate": 0.0,
        "symbols_submitted": [],
        "symbols_filled": [],
        "symbols_open": [],
        "notes": list(notes),
    }


def _classify_terminal(status: str) -> str:
    u = str(status or "").strip().lower()
    if u in ("", "nan", "unknown"):
        return "open"
    if u == "filled":
        return "filled"
    if u in ("canceled", "cancelled"):
        return "canceled"
    if u == "rejected":
        return "rejected"
    if u in ("expired",):
        return "canceled"
    if u in ("error", "failed"):
        return "rejected"
    if u in _OPEN_STATUSES:
        return "open"
    # done_for_day, replaced, etc. — treat as closed but not filled
    return "open"


def refresh_session_fill_pressure(session: Optional[str] = None) -> Dict[str, Any]:
    """
    Build session_fill_pressure.json for one placement session.
    Default session: auto among exec_trades_* and reprice_* (see _choose_session_auto), or CLI override.
    Counts only rows in live_orders_log.csv with matching `session` column.
    """
    meta = _read_last_session()
    selection_rule = ""
    session_type = ""

    try:
        if not LIVE_ORDERS_LOG_CSV.is_file() or LIVE_ORDERS_LOG_CSV.stat().st_size == 0:
            empty_out = _empty_payload("", meta, "", "", ["live_orders_log.csv missing or empty"])
            _write_json(empty_out)
            return empty_out
        df = pd.read_csv(LIVE_ORDERS_LOG_CSV)
    except Exception as e:
        empty_out = _empty_payload("", meta, "", "", [f"read live_orders_log failed: {e}"])
        _write_json(empty_out)
        return empty_out

    if "session" not in df.columns:
        empty_out = _empty_payload("", meta, "", "", ["live_orders_log.csv has no session column"])
        _write_json(empty_out)
        return empty_out

    sess_in = (session or "").strip()
    if sess_in:
        sess = sess_in
        selection_rule = "cli_override"
        session_type = _session_family(sess)
    else:
        sess, selection_rule, session_type = _choose_session_auto(df, meta)
        if not session_type and sess:
            session_type = _session_family(sess)

    meta_session = (meta.get("session") or "").strip() if isinstance(meta, dict) else ""
    orders_planned = int(meta.get("orders_planned") or 0) if meta and meta_session == sess else 0
    mode = str(meta.get("mode") or "") if meta and meta_session == sess else ""
    dry_run = bool(meta.get("dry_run")) if meta and meta_session == sess else False

    empty_out: Dict[str, Any] = {
        "timestamp": utc_iso(),
        "session": sess,
        "session_type": session_type or _session_family(sess),
        "session_selection_rule": selection_rule,
        "mode": mode,
        "dry_run_plan": dry_run,
        "orders_planned": orders_planned,
        "orders_submitted": 0,
        "orders_filled": 0,
        "orders_open": 0,
        "orders_canceled": 0,
        "orders_rejected": 0,
        "fill_rate": 0.0,
        "symbols_submitted": [],
        "symbols_filled": [],
        "symbols_open": [],
        "notes": [],
    }

    if not sess:
        empty_out["notes"] = ["no exec_trades_* or reprice_* session found in log"]
        _write_json(empty_out)
        return empty_out

    s_col = df["session"].astype(str).str.strip()
    df_sess = df.loc[s_col == sess].copy()
    if df_sess.empty:
        nlist: List[str] = []
        if dry_run and orders_planned > 0:
            nlist.append("dry_run plan: no broker submits for this session (expected).")
        else:
            nlist.append(f"no log rows for session={sess}")
        out = {**empty_out, "session": sess, "notes": nlist}
        _write_json(out)
        return out

    if "order_id" not in df_sess.columns:
        empty_out["notes"] = ["live_orders_log missing order_id"]
        _write_json(empty_out)
        return empty_out

    df_sess["order_id"] = df_sess["order_id"].astype(str).str.strip()
    df_sess = df_sess[df_sess["order_id"].apply(_is_uuid_like)].copy()
    if df_sess.empty:
        out = {
            **empty_out,
            "session": sess,
            "notes": [f"session={sess} has no uuid-like order_id rows"],
        }
        _write_json(out)
        return out

    df_sess["_ts"] = df_sess["timestamp"].astype(str).apply(_parse_ts)
    df_sess = df_sess.sort_values("_ts")
    latest = df_sess.groupby("order_id", as_index=False).last()

    n_sub = int(len(latest))
    sym_sub: Set[str] = set()
    sym_fill: Set[str] = set()
    sym_open: Set[str] = set()
    n_fill = n_open = n_canceled = n_rejected = 0

    for _, row in latest.iterrows():
        st = str(row.get("status") or "").strip().lower()
        sym = str(row.get("symbol") or "").strip().upper()
        if sym:
            sym_sub.add(sym)
        bucket = _classify_terminal(st)
        try:
            fq = float(row.get("filled_qty") or 0)
        except Exception:
            fq = 0.0
        try:
            q = float(row.get("qty") or 0)
        except Exception:
            q = 0.0
        if bucket == "filled":
            n_fill += 1
            if sym:
                sym_fill.add(sym)
        elif bucket == "canceled":
            n_canceled += 1
        elif bucket == "rejected":
            n_rejected += 1
        elif bucket == "open":
            n_open += 1
            if sym:
                sym_open.add(sym)
            if st == "partially_filled" and fq > 0 and q > 0 and fq >= q - 1e-9:
                n_fill += 1
                n_open -= 1
                if sym:
                    sym_fill.add(sym)
                    sym_open.discard(sym)
        else:
            n_open += 1
            if sym:
                sym_open.add(sym)

    fill_rate = (float(n_fill) / float(n_sub)) if n_sub > 0 else 0.0

    out: Dict[str, Any] = {
        "timestamp": utc_iso(),
        "session": sess,
        "session_type": session_type or _session_family(sess),
        "session_selection_rule": selection_rule,
        "mode": mode,
        "dry_run_plan": dry_run,
        "orders_planned": orders_planned,
        "orders_submitted": n_sub,
        "orders_filled": n_fill,
        "orders_open": max(0, n_open),
        "orders_canceled": n_canceled,
        "orders_rejected": n_rejected,
        "fill_rate": round(fill_rate, 6),
        "symbols_submitted": sorted(sym_sub),
        "symbols_filled": sorted(sym_fill),
        "symbols_open": sorted(sym_open),
        "notes": [],
    }
    if meta_session and meta_session != sess:
        out["notes"].append(
            f"last_execution_session.json points to {meta_session}; "
            "orders_planned/mode/dry_run apply only when that matches the viewed session."
        )
    if dry_run and n_sub == 0 and orders_planned > 0 and meta_session == sess:
        out["notes"].append("dry_run plan: no broker submits for this session (expected).")
    _write_json(out)
    return out


def _write_json(payload: Dict[str, Any]) -> None:
    try:
        SESSION_FILL_PRESSURE_JSON.parent.mkdir(parents=True, exist_ok=True)
        SESSION_FILL_PRESSURE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Session-scoped fill stats from live_orders_log.csv")
    ap.add_argument(
        "--session",
        type=str,
        default="",
        help="Override session id (e.g. exec_trades_* or reprice_*). Default: auto-detect from log.",
    )
    args = ap.parse_args()
    sess = str(args.session or "").strip() or None
    refresh_session_fill_pressure(session=sess)
    try:
        print(SESSION_FILL_PRESSURE_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
