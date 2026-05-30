"""
Lightweight order discipline: de-dupe per session, cooldowns, open same-side, reprice bypass.

Reads config/order_discipline.json, data/results/live_orders_log.csv, optional open_orders_snapshot.csv.
Writes data/results/order_discipline_diagnostics.json and optional order_discipline_log.csv.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from services.execution_quality import (
    effective_recent_fill_cooldown_minutes,
    load_positions_qty_map,
    log_cooldown_bypassed,
    log_cooldown_relaxed,
    median_confidence_buys,
    recent_submit_cooldown_should_bypass,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "order_discipline.json"
RESULTS = ROOT / "data" / "results"
DEFAULT_LOG_PATH = RESULTS / "live_orders_log.csv"
DEFAULT_OPEN_SNAPSHOT = RESULTS / "open_orders_snapshot.csv"
DIAG_JSON = RESULTS / "order_discipline_diagnostics.json"
DISCIPLINE_LOG_CSV = RESULTS / "order_discipline_log.csv"

_DEFAULT_CFG: Dict[str, Any] = {
    "enabled": True,
    "same_session_symbol_lock": True,
    "cross_session_cooldown_minutes": 30,
    "block_if_open_same_side_exists": True,
    "block_if_recent_filled_same_side_minutes": 20,
    "block_if_recent_canceled_same_side_minutes": 10,
    "block_if_recent_submitted_same_side_minutes": 15,
    "allow_reprice_replacements": True,
    "allow_exit_after_buy_fill": False,
    "allow_buy_after_exit_fill": True,
    "log_decisions": True,
}

_SYM_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_symbol(x: Any) -> str:
    s = str(x or "").strip().upper()
    return s


def normalize_side(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("buy", "b", "buy_to_open"):
        return "buy"
    if s in ("sell", "s", "sell_to_close"):
        return "sell"
    return None


def load_order_discipline_config() -> Dict[str, Any]:
    cfg = dict(_DEFAULT_CFG)
    try:
        if CONFIG_PATH.is_file():
            u = json.loads(CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict):
                cfg.update(u)
    except Exception:
        pass
    return cfg


def _parse_ts(val: Any) -> Optional[datetime]:
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return None
    if isinstance(val, datetime):
        dt = val
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    s = str(val).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    if "T" not in s and " " in s:
        s = s.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _looks_like_symbol(sym: str) -> bool:
    if not sym or len(sym) > 16:
        return False
    return bool(_SYM_RE.match(sym))


def read_recent_order_events(
    log_path: Path = DEFAULT_LOG_PATH,
    lookback_minutes: float = 60.0,
    *,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Return recent event dicts from live_orders_log (timestamp-filtered)."""
    now = now or _utc_now()
    if not log_path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        import pandas as pd

        df = pd.read_csv(log_path, engine="python", on_bad_lines="skip", keep_default_na=False)
    except Exception:
        try:
            import pandas as pd

            df = pd.read_csv(log_path, keep_default_na=False)
        except Exception:
            return []
    if df is None or df.empty:
        return []
    df.columns = [str(c).strip() for c in df.columns]
    if "timestamp" not in df.columns:
        return []
    cutoff = (
        (now.timestamp() - max(0.0, float(lookback_minutes)) * 60.0)
        if lookback_minutes < 1e6
        else 0.0
    )

    for _, r in df.iterrows():
        ts = _parse_ts(r.get("timestamp"))
        if ts is None:
            continue
        if lookback_minutes < 1e6 and ts.timestamp() < cutoff:
            continue
        sym = normalize_symbol(r.get("symbol"))
        if not _looks_like_symbol(sym):
            continue
        side = normalize_side(r.get("side"))
        if not side:
            continue
        action = str(r.get("action") or "").strip().lower()
        status = str(r.get("status") or "").strip().lower()
        try:
            fq = int(float(r.get("filled_qty") or 0))
        except Exception:
            fq = 0
        rows.append(
            {
                "timestamp": ts,
                "symbol": sym,
                "side": side,
                "action": action,
                "status": status,
                "filled_qty": fq,
                "session": str(r.get("session") or "").strip(),
            }
        )
    rows.sort(key=lambda x: x["timestamp"])
    return rows


def load_open_same_side_keys(
    snapshot_path: Path = DEFAULT_OPEN_SNAPSHOT,
    *,
    broker_open_keys: Optional[Set[Tuple[str, str]]] = None,
) -> Set[Tuple[str, str]]:
    """Union of open (symbol, side) keys from snapshot and optional broker index."""
    out: Set[Tuple[str, str]] = set()
    if broker_open_keys:
        out |= set(broker_open_keys)
    if not snapshot_path.is_file():
        return out
    terminal = {
        "filled",
        "canceled",
        "cancelled",
        "done_for_day",
        "expired",
        "replaced",
        "failed",
    }
    try:
        import pandas as pd

        df = pd.read_csv(snapshot_path, on_bad_lines="skip", keep_default_na=False)
    except Exception:
        return out
    if df is None or df.empty:
        return out
    df.columns = [str(c).strip() for c in df.columns]
    sym_col = "symbol" if "symbol" in df.columns else None
    side_col = "side" if "side" in df.columns else None
    stat_col = "status" if "status" in df.columns else None
    if not sym_col or not side_col:
        return out
    for _, r in df.iterrows():
        sym = normalize_symbol(r.get(sym_col))
        if not _looks_like_symbol(sym):
            continue
        side = normalize_side(r.get(side_col))
        if not side:
            continue
        if stat_col:
            st = str(r.get(stat_col) or "").strip().lower()
            if st in terminal:
                continue
        out.add((sym, side))
    return out


def build_event_indexes(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per (symbol, side) last timestamps for submit / fill / cancel; per symbol last buy/sell fill."""
    last_submit: Dict[Tuple[str, str], datetime] = {}
    last_fill: Dict[Tuple[str, str], datetime] = {}
    last_cancel: Dict[Tuple[str, str], datetime] = {}
    last_buy_fill: Dict[str, datetime] = {}
    last_sell_fill: Dict[str, datetime] = {}

    for e in events:
        sym = e["symbol"]
        side = e["side"]
        ts: datetime = e["timestamp"]
        key = (sym, side)
        act = e["action"]
        st = e["status"]
        fq = int(e.get("filled_qty") or 0)

        if act == "submit":
            last_submit[key] = max(ts, last_submit.get(key, ts))
        if act == "cancel" or st in ("canceled", "cancelled"):
            last_cancel[key] = max(ts, last_cancel.get(key, ts))
        is_fill = fq > 0 or st in ("filled", "partially_filled")
        if is_fill:
            last_fill[key] = max(ts, last_fill.get(key, ts))
            if side == "buy":
                last_buy_fill[sym] = max(ts, last_buy_fill.get(sym, ts))
            else:
                last_sell_fill[sym] = max(ts, last_sell_fill.get(sym, ts))

    return {
        "last_submit": last_submit,
        "last_fill": last_fill,
        "last_cancel": last_cancel,
        "last_buy_fill": last_buy_fill,
        "last_sell_fill": last_sell_fill,
    }


def get_symbol_order_state(
    symbol: str,
    side: str,
    lookback_minutes: float,
    events: Optional[List[Dict[str, Any]]] = None,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Summarize recent activity for symbol/side from events (optional)."""
    now = now or _utc_now()
    sym = normalize_symbol(symbol)
    sd = normalize_side(side) or "buy"
    ev = events or read_recent_order_events(lookback_minutes=lookback_minutes, now=now)
    idx = build_event_indexes(ev)
    key = (sym, sd)
    return {
        "symbol": sym,
        "side": sd,
        "last_submit": idx["last_submit"].get(key),
        "last_fill": idx["last_fill"].get(key),
        "last_cancel": idx["last_cancel"].get(key),
        "last_buy_fill": idx["last_buy_fill"].get(sym),
        "last_sell_fill": idx["last_sell_fill"].get(sym),
    }


def _batch_buy_confidence_median_dict_rows(rows: List[Dict[str, Any]]) -> Optional[float]:
    vals: List[float] = []
    for r in rows:
        if normalize_side(r.get("side")) != "buy":
            continue
        try:
            vals.append(float(r.get("confidence", 0) or 0.0))
        except Exception:
            vals.append(0.0)
    return median_confidence_buys(vals)


def _batch_buy_confidence_median_planned(planned: List[Any]) -> Optional[float]:
    vals: List[float] = []
    for p in planned:
        if normalize_side(getattr(p, "side", None)) != "buy":
            continue
        try:
            vals.append(float(getattr(p, "confidence", 0) or 0.0))
        except Exception:
            vals.append(0.0)
    return median_confidence_buys(vals)


def _row_confidence_from_dict(r: Dict[str, Any]) -> Optional[float]:
    c = r.get("confidence")
    if c is None:
        return None
    try:
        return float(c)
    except Exception:
        return None


def _row_confidence_from_planned(p: Any) -> Optional[float]:
    c = getattr(p, "confidence", None)
    if c is None:
        return None
    try:
        return float(c)
    except Exception:
        return None


def should_block_order(
    symbol: str,
    side: str,
    session: str,
    context: Optional[Dict[str, Any]],
    *,
    cfg: Optional[Dict[str, Any]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    event_index: Optional[Dict[str, Any]] = None,
    session_seen: Optional[Set[Tuple[str, str]]] = None,
    open_side_keys: Optional[Set[Tuple[str, str]]] = None,
    now: Optional[datetime] = None,
    row_confidence: Optional[float] = None,
    batch_buy_confidence_median: Optional[float] = None,
    positions_qty_map: Optional[Dict[str, float]] = None,
) -> Tuple[bool, str]:
    """
    Returns (blocked, reason). If blocked True, do not submit.
    reason may be REPRICE_ALLOWED or OK when not blocked.
    """
    cfg = cfg or load_order_discipline_config()
    ctx = dict(context or {})
    now = now or _utc_now()

    if not cfg.get("enabled", True):
        return False, "OK"

    sym = normalize_symbol(symbol)
    sd = normalize_side(side)
    if not sd:
        return False, "OK"

    if ctx.get("is_reprice_replacement") and cfg.get("allow_reprice_replacements", True):
        return False, "REPRICE_ALLOWED"

    idx = event_index if event_index is not None else build_event_indexes(events or [])
    key = (sym, sd)
    sub_m = float(cfg.get("block_if_recent_submitted_same_side_minutes", 15) or 15)
    fill_m = float(cfg.get("block_if_recent_filled_same_side_minutes", 20) or 20)
    can_m = float(cfg.get("block_if_recent_canceled_same_side_minutes", 10) or 10)

    if cfg.get("same_session_symbol_lock", True) and session_seen is not None:
        if key in session_seen:
            return True, "SAME_SESSION_DUPLICATE"

    if cfg.get("block_if_open_same_side_exists", True) and open_side_keys is not None:
        if key in open_side_keys:
            return True, "OPEN_SAME_SIDE_EXISTS"

    last_submit_ts = idx["last_submit"].get(key)
    if last_submit_ts is not None and (now - last_submit_ts).total_seconds() / 60.0 <= sub_m:
        bypass, detail = recent_submit_cooldown_should_bypass(
            symbol=sym,
            side=sd,
            last_submit_ts=last_submit_ts,
            events=events,
            positions_qty_map=positions_qty_map,
        )
        if bypass:
            log_cooldown_bypassed(sym, detail)
        else:
            return True, "RECENT_SUBMIT_COOLDOWN"
    last_fill_t = idx["last_fill"].get(key)
    if last_fill_t is not None:
        minutes_since = (now - last_fill_t).total_seconds() / 60.0
        if minutes_since <= fill_m:
            eff_m = effective_recent_fill_cooldown_minutes(fill_m)
            if minutes_since > eff_m:
                pass
            else:
                if (
                    sd == "buy"
                    and row_confidence is not None
                    and batch_buy_confidence_median is not None
                    and row_confidence > batch_buy_confidence_median
                ):
                    log_cooldown_relaxed(
                        sym,
                        "RECENT_FILL_COOLDOWN",
                        "buy_confidence_above_batch_median",
                    )
                else:
                    return True, "RECENT_FILL_COOLDOWN"
    last_cancel_ts = idx["last_cancel"].get(key)
    if last_cancel_ts is not None and (now - last_cancel_ts).total_seconds() / 60.0 <= can_m:
        bypass, detail = recent_submit_cooldown_should_bypass(
            symbol=sym,
            side=sd,
            last_submit_ts=last_submit_ts,
            events=events,
            positions_qty_map=positions_qty_map,
        )
        if bypass:
            log_cooldown_bypassed(sym, detail)
        else:
            return True, "RECENT_CANCEL_COOLDOWN"

    fill_m_seq = float(cfg.get("block_if_recent_filled_same_side_minutes", 20) or 20)
    if sd == "sell" and not cfg.get("allow_exit_after_buy_fill", False):
        lb = idx["last_buy_fill"].get(sym)
        if lb is not None and (now - lb).total_seconds() / 60.0 <= fill_m_seq:
            return True, "EXIT_AFTER_BUY_FILL_COOLDOWN"
    if sd == "buy" and not cfg.get("allow_buy_after_exit_fill", True):
        ls = idx["last_sell_fill"].get(sym)
        if ls is not None and (now - ls).total_seconds() / 60.0 <= fill_m_seq:
            return True, "BUY_AFTER_EXIT_FILL_COOLDOWN"

    return False, "OK"


_WARN_PREFIX = "[order_discipline]"


def _append_discipline_csv(
    row: Dict[str, Any],
    *,
    log_decisions: bool,
) -> None:
    if not log_decisions:
        return
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        fields = ["timestamp", "session", "symbol", "side", "decision", "reason", "source_module"]
        new_file = not DISCIPLINE_LOG_CSV.is_file() or DISCIPLINE_LOG_CSV.stat().st_size == 0
        with DISCIPLINE_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})
    except Exception as e:
        print(f"{_WARN_PREFIX} WARN could not write order_discipline_log.csv: {e}", flush=True)


def _write_diagnostics_json(payload: Dict[str, Any]) -> None:
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        DIAG_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"{_WARN_PREFIX} WARN could not write diagnostics: {e}", flush=True)


def annotate_plan_with_discipline(
    rows: Union[List[Dict[str, Any]], Any],
    *,
    session: str,
    source_module: str,
    mode: str = "paper",
    context: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    open_side_keys: Optional[Set[Tuple[str, str]]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    For list of dicts with symbol/side keys: add discipline_allowed, discipline_reason; drop blocked.
    Returns (kept_rows, meta).
    """
    cfg = cfg or load_order_discipline_config()
    if not cfg.get("enabled", True):
        out = []
        for r in rows:
            if isinstance(r, dict):
                rr = dict(r)
                rr["discipline_allowed"] = True
                rr["discipline_reason"] = "DISABLED"
                out.append(rr)
        n_out = len(out)
        _write_diagnostics_json(
            {
                "timestamp": _utc_iso(),
                "enabled": False,
                "mode": mode,
                "session": session,
                "source_module": source_module,
                "orders_seen": n_out,
                "orders_allowed": n_out,
                "orders_blocked": 0,
                "block_reasons": {},
                "symbols_blocked": [],
            }
        )
        return out, {
            "enabled": False,
            "orders_seen": n_out,
            "orders_allowed": n_out,
            "orders_blocked": 0,
        }

    lookback = float(cfg.get("cross_session_cooldown_minutes", 30) or 30)
    ev = events if events is not None else read_recent_order_events(lookback_minutes=lookback)
    idx = build_event_indexes(ev)
    opens = (
        open_side_keys
        if open_side_keys is not None
        else load_open_same_side_keys(DEFAULT_OPEN_SNAPSHOT)
    )
    session_seen: Set[Tuple[str, str]] = set()
    positions_qty_map = load_positions_qty_map()

    block_counts: Counter = Counter()
    symbols_blocked: List[str] = []
    seen_block: Set[str] = set()

    out: List[Dict[str, Any]] = []
    raw_list: List[Dict[str, Any]] = [dict(x) for x in rows] if isinstance(rows, list) else []
    m_buy = _batch_buy_confidence_median_dict_rows(raw_list)

    for r in raw_list:
        sym = normalize_symbol(r.get("symbol") or r.get("ticker"))
        sd = normalize_side(r.get("side"))
        if not sym or not sd:
            r["discipline_allowed"] = True
            r["discipline_reason"] = "OK"
            out.append(r)
            continue
        blocked, reason = should_block_order(
            sym,
            sd,
            session,
            context,
            cfg=cfg,
            events=ev,
            event_index=idx,
            session_seen=session_seen,
            open_side_keys=opens,
            row_confidence=_row_confidence_from_dict(r),
            batch_buy_confidence_median=m_buy,
            positions_qty_map=positions_qty_map,
        )
        if blocked:
            block_counts[reason] += 1
            if sym not in seen_block:
                seen_block.add(sym)
                symbols_blocked.append(sym)
            r["discipline_allowed"] = False
            r["discipline_reason"] = reason
            _ld = bool(cfg.get("log_decisions", True))
            _append_discipline_csv(
                {
                    "timestamp": _utc_iso(),
                    "session": session,
                    "symbol": sym,
                    "side": sd,
                    "decision": "blocked",
                    "reason": reason,
                    "source_module": source_module,
                },
                log_decisions=_ld,
            )
            continue
        r["discipline_allowed"] = True
        r["discipline_reason"] = reason
        session_seen.add((sym, sd))
        _ld = bool(cfg.get("log_decisions", True))
        _append_discipline_csv(
            {
                "timestamp": _utc_iso(),
                "session": session,
                "symbol": sym,
                "side": sd,
                "decision": "allowed",
                "reason": reason,
                "source_module": source_module,
            },
            log_decisions=_ld,
        )
        out.append(r)

    n_seen = len(raw_list)
    n_blocked = n_seen - len(out)
    meta = {
        "orders_seen": n_seen,
        "orders_allowed": len(out),
        "orders_blocked": n_blocked,
        "block_reasons": dict(block_counts),
        "symbols_blocked": symbols_blocked,
    }
    _write_diagnostics_json(
        {
            "timestamp": _utc_iso(),
            "enabled": True,
            "mode": mode,
            "session": session,
            "source_module": source_module,
            "orders_seen": n_seen,
            "orders_allowed": len(out),
            "orders_blocked": n_blocked,
            "block_reasons": dict(block_counts),
            "symbols_blocked": symbols_blocked,
        }
    )
    return out, meta


def apply_discipline_to_planned_generic(
    planned: List[Any],
    *,
    session: str,
    source_module: str,
    mode: str = "paper",
    context: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    open_side_keys: Optional[Set[Tuple[str, str]]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Filter a list of objects with .symbol and .side; set discipline_allowed / discipline_reason."""
    cfg = cfg or load_order_discipline_config()
    if not cfg.get("enabled", True):
        for p in planned:
            try:
                setattr(p, "discipline_allowed", True)
                setattr(p, "discipline_reason", "DISABLED")
            except Exception:
                pass
        n = len(planned)
        _write_diagnostics_json(
            {
                "timestamp": _utc_iso(),
                "enabled": False,
                "mode": mode,
                "session": session,
                "source_module": source_module,
                "orders_seen": n,
                "orders_allowed": n,
                "orders_blocked": 0,
                "block_reasons": {},
                "symbols_blocked": [],
            }
        )
        return planned, {
            "enabled": False,
            "orders_seen": n,
            "orders_allowed": n,
            "orders_blocked": 0,
            "block_reasons": {},
            "symbols_blocked": [],
        }

    lookback = float(cfg.get("cross_session_cooldown_minutes", 30) or 30)
    ev = events if events is not None else read_recent_order_events(lookback_minutes=lookback)
    idx = build_event_indexes(ev)
    opens = (
        open_side_keys
        if open_side_keys is not None
        else load_open_same_side_keys(DEFAULT_OPEN_SNAPSHOT)
    )
    session_seen: Set[Tuple[str, str]] = set()
    positions_qty_map = load_positions_qty_map()
    kept: List[Any] = []
    block_counts: Counter = Counter()
    symbols_blocked: List[str] = []
    seen_block: Set[str] = set()
    m_buy = _batch_buy_confidence_median_planned(planned)

    for p in planned:
        sym = normalize_symbol(getattr(p, "symbol", None))
        sd = normalize_side(getattr(p, "side", None))
        if not sym or not sd:
            try:
                setattr(p, "discipline_allowed", True)
                setattr(p, "discipline_reason", "OK")
            except Exception:
                pass
            kept.append(p)
            continue
        blocked, reason = should_block_order(
            sym,
            sd,
            session,
            context,
            cfg=cfg,
            events=ev,
            event_index=idx,
            session_seen=session_seen,
            open_side_keys=opens,
            row_confidence=_row_confidence_from_planned(p),
            batch_buy_confidence_median=m_buy,
            positions_qty_map=positions_qty_map,
        )
        if blocked:
            block_counts[reason] += 1
            if sym not in seen_block:
                seen_block.add(sym)
                symbols_blocked.append(sym)
            try:
                setattr(p, "discipline_allowed", False)
                setattr(p, "discipline_reason", reason)
            except Exception:
                pass
            _ld = bool(cfg.get("log_decisions", True))
            _append_discipline_csv(
                {
                    "timestamp": _utc_iso(),
                    "session": session,
                    "symbol": sym,
                    "side": sd,
                    "decision": "blocked",
                    "reason": reason,
                    "source_module": source_module,
                },
                log_decisions=_ld,
            )
            continue
        try:
            setattr(p, "discipline_allowed", True)
            setattr(p, "discipline_reason", reason)
        except Exception:
            pass
        session_seen.add((sym, sd))
        _ld = bool(cfg.get("log_decisions", True))
        _append_discipline_csv(
            {
                "timestamp": _utc_iso(),
                "session": session,
                "symbol": sym,
                "side": sd,
                "decision": "allowed",
                "reason": reason,
                "source_module": source_module,
            },
            log_decisions=_ld,
        )
        kept.append(p)

    n_seen = len(planned)
    n_blocked = n_seen - len(kept)
    _write_diagnostics_json(
        {
            "timestamp": _utc_iso(),
            "enabled": True,
            "mode": mode,
            "session": session,
            "source_module": source_module,
            "orders_seen": n_seen,
            "orders_allowed": len(kept),
            "orders_blocked": n_blocked,
            "block_reasons": dict(block_counts),
            "symbols_blocked": symbols_blocked,
        }
    )
    return kept, {
        "orders_seen": n_seen,
        "orders_allowed": len(kept),
        "orders_blocked": n_blocked,
        "block_reasons": dict(block_counts),
        "symbols_blocked": symbols_blocked,
    }
