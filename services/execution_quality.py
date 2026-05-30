"""
Execution quality: entry confirmation (MA / delta / confidence), delay weak entries, top-N ranking.
Exits unchanged; trims get optional minimum spacing via position_management_state.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "execution_quality.json"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
DEFAULT_POSITIONS_SNAPSHOT_PATH = RESULTS / "positions_snapshot.csv"

_DEFAULT: Dict[str, Any] = {
    "enabled": True,
    "min_confidence": 0.5,
    "min_delta_pct": 0.003,
    "top_n_entries": 8,
    "delay_weak_entries": True,
    "delayed_entry_soft_penalty_enabled": True,
    "delayed_entry_penalty_factor": 0.85,
    "short_ma_period": 20,
    "trim_min_interval_minutes": 60,
    "fill_cooldown_minutes_scale": 0.5,
    # State-aware cooldown bypass for post-cancel / no-position scenarios.
    # Cooldown still applies when a real position exists or when the previous
    # order is still actively working; this only relaxes the block when the
    # market clearly did not result in a filled position.
    "recent_submit_cooldown_state_aware": True,
    "recent_submit_cooldown_terminal_statuses": [
        "canceled",
        "cancelled",
        "rejected",
        "expired",
    ],
    "recent_submit_cooldown_position_qty_threshold": 0.0,
    "positions_snapshot_path": str(DEFAULT_POSITIONS_SNAPSHOT_PATH),
}


def load_execution_quality_config() -> Dict[str, Any]:
    cfg = dict(_DEFAULT)
    try:
        if CONFIG_PATH.is_file():
            u = json.loads(CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(u, dict):
                cfg.update(u)
    except Exception:
        pass
    return cfg


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(val: Any) -> Optional[datetime]:
    if val is None or str(val).strip() == "":
        return None
    s = str(val).strip().replace("Z", "+00:00")
    if "T" not in s and " " in s:
        s = s.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def short_ma_close(symbol: str, period: int) -> Optional[float]:
    """Rolling mean of last `period` closes from processed parquet; None if unavailable."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    p = PROCESSED / f"{sym}.parquet"
    if not p.is_file():
        return None
    try:
        df = pd.read_parquet(p)
        if df is None or df.empty or "close" not in df.columns:
            return None
        c = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(c) < period:
            return None
        return float(c.tail(period).mean())
    except Exception:
        return None


def entry_confirmation_ok(
    *,
    ref_price: float,
    confidence: float,
    delta_pct: float,
    min_confidence: float,
    min_delta_pct: float,
    ma_short: Optional[float],
) -> bool:
    """At least ONE: price above short MA, delta meets threshold, or confidence meets threshold."""
    ok_ma = ma_short is not None and ma_short > 0 and ref_price > ma_short
    ok_delta = delta_pct >= min_delta_pct
    ok_conf = confidence >= min_confidence
    return bool(ok_ma or ok_delta or ok_conf)


def entry_priority_score(confidence: float, delta_pct: float) -> float:
    return float(confidence) * float(delta_pct)


def _planned_entry_rank_score(po: Any) -> float:
    """Prefer execute_trades diversification final score when present; else confidence * delta_pct."""
    er = getattr(po, "execution_rank_score", None)
    if er is not None:
        try:
            v = float(er)
            if not math.isnan(v):
                return v
        except Exception:
            pass
    return entry_priority_score(float(po.confidence), float(po.delta_pct))


def should_delay_weak_entry(
    *,
    confidence: float,
    delta_pct: float,
    min_confidence: float,
    min_delta_pct: float,
    delay_weak_entries: bool,
) -> bool:
    """Weak = low confidence OR small delta vs configured floors (delay, do not execute this run)."""
    if not delay_weak_entries:
        return False
    return (confidence < min_confidence) or (delta_pct < min_delta_pct)


def trim_allowed_after_cooldown(
    _symbol: str,
    sym_state: Dict[str, Any],
    trim_min_interval_minutes: float,
) -> Tuple[bool, Optional[str]]:
    """Returns (allowed, skip_reason_if_blocked). Exits are not passed here."""
    if float(trim_min_interval_minutes or 0) <= 0:
        return True, None
    last = _parse_ts(sym_state.get("last_trim_ts_utc"))
    if last is None:
        return True, None
    age = (_utc_now() - last).total_seconds() / 60.0
    if age < float(trim_min_interval_minutes or 0):
        return False, "TRIM_COOLDOWN"
    return True, None


def effective_recent_fill_cooldown_minutes(configured_minutes: float) -> float:
    """Order discipline X → X * scale (default 0.5). Used only for RECENT_FILL same-side, not exit sequencing."""
    cfg = load_execution_quality_config()
    scale = float(cfg.get("fill_cooldown_minutes_scale", 0.5) or 0.5)
    scale = max(0.05, min(1.0, scale))
    base = float(configured_minutes or 0.0)
    return max(0.0, base * scale)


def median_confidence_buys(confidences: List[float]) -> Optional[float]:
    if not confidences:
        return None
    return float(statistics.median(confidences))


def log_cooldown_relaxed(
    symbol: str,
    original_block: str,
    override_reason: str,
) -> None:
    print(
        f"[COOLDOWN_RELAXED] symbol={symbol} original_block={original_block} override_reason={override_reason}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# State-aware submit/cancel cooldown bypass
#
# These helpers are imported by services/order_discipline.py to make the
# RECENT_SUBMIT_COOLDOWN / RECENT_CANCEL_COOLDOWN checks state-aware.
# A cooldown is only meaningful when there is a real position to protect or
# a working order still in flight. If the previous attempts were canceled,
# rejected, or expired (e.g. market-closed paper rejections), or no position
# was actually opened, we should allow re-submission of the new planned order.
# ---------------------------------------------------------------------------


def _normalize_symbol(x: Any) -> str:
    return str(x or "").strip().upper()


def load_positions_qty_map(
    path: Optional[Path] = None,
) -> Dict[str, float]:
    """Read positions_snapshot.csv defensively.

    Returns {symbol: qty} using the latest snapshot_ts row per symbol.
    Missing or malformed files yield an empty dict. Never raises.
    """
    if path is None:
        cfg = load_execution_quality_config()
        cfg_path = cfg.get("positions_snapshot_path")
        path = Path(cfg_path) if cfg_path else DEFAULT_POSITIONS_SNAPSHOT_PATH
    out: Dict[str, float] = {}
    try:
        p = Path(path)
        if not p.is_file():
            return out
        df = pd.read_csv(p, on_bad_lines="skip", keep_default_na=False)
    except Exception:
        return out
    if df is None or df.empty:
        return out
    df.columns = [str(c).strip() for c in df.columns]
    sym_col = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
    if not sym_col or "qty" not in df.columns:
        return out
    ts_col = "snapshot_ts" if "snapshot_ts" in df.columns else None
    # Build (symbol -> (ts, qty)) keeping the latest ts.
    latest: Dict[str, Tuple[Optional[datetime], float]] = {}
    for _, r in df.iterrows():
        sym = _normalize_symbol(r.get(sym_col))
        if not sym:
            continue
        try:
            qty = float(r.get("qty") or 0.0)
        except Exception:
            qty = 0.0
        ts = _parse_ts(r.get(ts_col)) if ts_col else None
        prev = latest.get(sym)
        if prev is None:
            latest[sym] = (ts, qty)
            continue
        prev_ts, _ = prev
        if ts is None and prev_ts is None:
            latest[sym] = (ts, qty)
        elif prev_ts is None and ts is not None:
            latest[sym] = (ts, qty)
        elif ts is not None and prev_ts is not None and ts >= prev_ts:
            latest[sym] = (ts, qty)
    for sym, (_ts, qty) in latest.items():
        out[sym] = qty
    return out


def position_is_active(
    qty_map: Optional[Dict[str, float]],
    symbol: str,
    qty_threshold: float = 0.0,
) -> bool:
    """True when the qty map records a non-zero position for the symbol."""
    if not qty_map:
        return False
    qty = qty_map.get(_normalize_symbol(symbol))
    if qty is None:
        return False
    try:
        return abs(float(qty)) > float(qty_threshold or 0.0)
    except Exception:
        return False


def last_status_after_submit(
    events: Optional[List[Dict[str, Any]]],
    symbol: str,
    side: str,
    last_submit_ts: Optional[datetime],
) -> Optional[str]:
    """Return the most recent non-empty status for (sym, side) at/after last_submit_ts.

    Used to detect whether the previous submission ended in a terminal state
    such as canceled/rejected/expired. Returns None when nothing relevant is
    recorded.
    """
    if not events:
        return None
    sym = _normalize_symbol(symbol)
    sd = str(side or "").strip().lower()
    best_ts: Optional[datetime] = None
    best_status: Optional[str] = None
    for e in events:
        try:
            e_sym = _normalize_symbol(e.get("symbol"))
            e_side = str(e.get("side") or "").strip().lower()
            if e_sym != sym or e_side != sd:
                continue
            ts = e.get("timestamp")
            if not isinstance(ts, datetime):
                continue
            if last_submit_ts is not None and ts < last_submit_ts:
                continue
            status = str(e.get("status") or "").strip().lower()
            if not status:
                continue
            if best_ts is None or ts >= best_ts:
                best_ts = ts
                best_status = status
        except Exception:
            continue
    return best_status


def recent_submit_cooldown_should_bypass(
    *,
    symbol: str,
    side: str,
    last_submit_ts: Optional[datetime],
    events: Optional[List[Dict[str, Any]]] = None,
    positions_qty_map: Optional[Dict[str, float]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Decide whether a RECENT_SUBMIT/CANCEL cooldown should be bypassed.

    Returns (True, detail_reason) when the cooldown is safe to skip:
      * the prior order on this side ended canceled/rejected/expired, OR
      * there is no active position (qty == 0 / symbol absent) and a
        positions snapshot is available to confirm it.

    Otherwise returns (False, ""). Never raises.
    """
    cfg = cfg or load_execution_quality_config()
    if not bool(cfg.get("recent_submit_cooldown_state_aware", True)):
        return False, ""
    sym = _normalize_symbol(symbol)
    sd = str(side or "").strip().lower()
    if not sym or not sd:
        return False, ""

    terminal_raw = cfg.get(
        "recent_submit_cooldown_terminal_statuses",
        ["canceled", "cancelled", "rejected", "expired"],
    )
    try:
        terminal_statuses = {str(s).strip().lower() for s in (terminal_raw or []) if str(s).strip()}
    except Exception:
        terminal_statuses = {"canceled", "cancelled", "rejected", "expired"}

    last_status = last_status_after_submit(events, sym, sd, last_submit_ts)
    if last_status and last_status in terminal_statuses:
        return True, f"last_order_{last_status}"

    if positions_qty_map is not None:
        try:
            qty_threshold = float(
                cfg.get("recent_submit_cooldown_position_qty_threshold", 0.0) or 0.0
            )
        except Exception:
            qty_threshold = 0.0
        if not position_is_active(positions_qty_map, sym, qty_threshold=qty_threshold):
            return True, "no_active_position"

    return False, ""


def log_cooldown_bypassed(symbol: str, detail: str) -> None:
    """Emit the user-facing cooldown bypass marker.

    The canonical reason field is fixed so it is easy to grep for. The
    `detail` field carries the specific bypass cause (e.g. last_order_canceled,
    no_active_position).
    """
    print(
        f"[COOLDOWN_BYPASSED] symbol={symbol} reason=no_active_position_or_order_canceled detail={detail}",
        flush=True,
    )


def apply_execution_quality_filters(
    planned: List[Any],
    plan_lines: List[Any],
    skip_reasons: Dict[str, int],
    *,
    broker: Any,
    eq_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Any], List[Any], Dict[str, int], Dict[str, int]]:
    """
    Filter BUY/ADD only: confirmation, delay weak, top-N by confidence*delta_pct.
    TRIM/EXIT rows pass through unchanged.
    When delay_weak_entries fires, optional soft penalty (reduce rank score) may replace hard skip.
    """
    eq_stats: Dict[str, int] = {"delayed_penalized": 0, "delayed_entry_skipped": 0}
    eq_cfg = eq_cfg or load_execution_quality_config()
    if not eq_cfg.get("enabled", True):
        return planned, plan_lines, skip_reasons, eq_stats

    def bump(code: str) -> None:
        skip_reasons[code] = skip_reasons.get(code, 0) + 1

    min_conf = float(eq_cfg.get("min_confidence", 0.55) or 0.0)
    min_delta = float(eq_cfg.get("min_delta_pct", 0.01) or 0.0)
    top_n = max(1, int(eq_cfg.get("top_n_entries", 5) or 5))
    delay_weak = bool(eq_cfg.get("delay_weak_entries", True))
    soft_delay = bool(eq_cfg.get("delayed_entry_soft_penalty_enabled", True))
    delay_penalty = float(eq_cfg.get("delayed_entry_penalty_factor", 0.85) or 0.85)
    delay_penalty = max(0.05, min(1.0, delay_penalty))
    min_exec_thr = float(eq_cfg.get("min_final_score_threshold", 0.0) or 0.0)
    ma_period = max(3, int(eq_cfg.get("short_ma_period", 20) or 20))

    # Indices of plan lines with BUY/ADD planned orders
    entry_indices: List[int] = []
    for i, pl in enumerate(plan_lines):
        if pl.status != "planned" or pl.planned is None:
            continue
        st = str(pl.planned.stance or "").strip().upper()
        if st in ("BUY", "ADD"):
            entry_indices.append(i)

    if not entry_indices:
        return planned, plan_lines, skip_reasons, eq_stats

    # First pass: confirmation + delay → mark skipped on plan_lines
    kept_entry_indices: List[int] = []
    for i in entry_indices:
        pl = plan_lines[i]
        po = pl.planned
        if po is None:
            continue
        sym = str(po.symbol).strip().upper()
        ref = None
        try:
            from services.execute_trades import _ref_price

            ref = _ref_price(broker, sym, "buy")
        except Exception:
            ref = None
        if ref is None or ref <= 0:
            pl.action = "skip"
            pl.status = "skipped"
            pl.skip_reason = "EXECUTION_QUALITY_NO_PRICE"
            pl.planned = None
            bump("EXECUTION_QUALITY_NO_PRICE")
            continue

        ma_s = short_ma_close(sym, ma_period)
        confirmed = entry_confirmation_ok(
            ref_price=float(ref),
            confidence=float(po.confidence),
            delta_pct=float(po.delta_pct),
            min_confidence=min_conf,
            min_delta_pct=min_delta,
            ma_short=ma_s,
        )
        if not confirmed:
            pl.action = "skip"
            pl.status = "skipped"
            pl.skip_reason = "EXECUTION_QUALITY_NO_CONFIRMATION"
            pl.planned = None
            bump("EXECUTION_QUALITY_NO_CONFIRMATION")
            continue

        if should_delay_weak_entry(
            confidence=float(po.confidence),
            delta_pct=float(po.delta_pct),
            min_confidence=min_conf,
            min_delta_pct=min_delta,
            delay_weak_entries=delay_weak,
        ):
            raw_rank = _planned_entry_rank_score(po)
            penalized = raw_rank * delay_penalty
            if soft_delay and (min_exec_thr <= 0 or penalized >= min_exec_thr):
                if getattr(po, "execution_rank_score", None) is not None:
                    try:
                        po.execution_rank_score = float(po.execution_rank_score) * delay_penalty
                    except Exception:
                        po.execution_rank_score = penalized
                else:
                    po.execution_rank_score = penalized
                eq_stats["delayed_penalized"] = int(eq_stats.get("delayed_penalized", 0) or 0) + 1
                kept_entry_indices.append(i)
                print(
                    f"[DELAYED_ENTRY_SOFT] symbol={sym} raw_rank={raw_rank:.6f} "
                    f"penalized={penalized:.6f} factor={delay_penalty:.2f} min_threshold={min_exec_thr:.6f}"
                )
                continue
            pl.action = "skip"
            pl.status = "skipped"
            pl.skip_reason = "DELAYED_ENTRY"
            pl.planned = None
            bump("DELAYED_ENTRY")
            eq_stats["delayed_entry_skipped"] = (
                int(eq_stats.get("delayed_entry_skipped", 0) or 0) + 1
            )
            continue

        kept_entry_indices.append(i)

    # Top-N by priority score among remaining entries
    if len(kept_entry_indices) > top_n:
        scored: List[Tuple[float, int]] = []
        for i in kept_entry_indices:
            po = plan_lines[i].planned
            if po is None:
                continue
            sc = _planned_entry_rank_score(po)
            scored.append((sc, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, i in scored[top_n:]:
            pl = plan_lines[i]
            pl.action = "skip"
            pl.status = "skipped"
            pl.skip_reason = "EXECUTION_QUALITY_TOP_N_CAP"
            pl.planned = None
            bump("EXECUTION_QUALITY_TOP_N_CAP")

    # Rebuild planned from plan_lines
    new_planned: List[Any] = []
    for pl in plan_lines:
        if pl.status == "planned" and pl.planned is not None:
            new_planned.append(pl.planned)

    return new_planned, plan_lines, skip_reasons, eq_stats
