"""
Execution quality: entry confirmation (MA / delta / confidence), delay weak entries, top-N ranking.
Exits unchanged; trims get optional minimum spacing via position_management_state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "execution_quality.json"
PROCESSED = ROOT / "data" / "processed"

_DEFAULT: Dict[str, Any] = {
    "enabled": True,
    "min_confidence": 0.5,
    "min_delta_pct": 0.003,
    "top_n_entries": 8,
    "delay_weak_entries": True,
    "short_ma_period": 20,
    "trim_min_interval_minutes": 60,
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


def apply_execution_quality_filters(
    planned: List[Any],
    plan_lines: List[Any],
    skip_reasons: Dict[str, int],
    *,
    broker: Any,
    eq_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Any], List[Any], Dict[str, int]]:
    """
    Filter BUY/ADD only: confirmation, delay weak, top-N by confidence*delta_pct.
    TRIM/EXIT rows pass through unchanged.
    """
    eq_cfg = eq_cfg or load_execution_quality_config()
    if not eq_cfg.get("enabled", True):
        return planned, plan_lines, skip_reasons

    def bump(code: str) -> None:
        skip_reasons[code] = skip_reasons.get(code, 0) + 1

    min_conf = float(eq_cfg.get("min_confidence", 0.55) or 0.0)
    min_delta = float(eq_cfg.get("min_delta_pct", 0.01) or 0.0)
    top_n = max(1, int(eq_cfg.get("top_n_entries", 5) or 5))
    delay_weak = bool(eq_cfg.get("delay_weak_entries", True))
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
        return planned, plan_lines, skip_reasons

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
            pl.action = "skip"
            pl.status = "skipped"
            pl.skip_reason = "DELAYED_ENTRY"
            pl.planned = None
            bump("DELAYED_ENTRY")
            continue

        kept_entry_indices.append(i)

    # Top-N by priority score among remaining entries
    if len(kept_entry_indices) > top_n:
        scored: List[Tuple[float, int]] = []
        for i in kept_entry_indices:
            po = plan_lines[i].planned
            if po is None:
                continue
            sc = entry_priority_score(float(po.confidence), float(po.delta_pct))
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

    return new_planned, plan_lines, skip_reasons
