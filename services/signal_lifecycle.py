from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_LOGIC_PATH = PROJECT_ROOT / "config" / "lifecycle_logic.json"

# Legacy effective-layer threshold when lifecycle_logic.enabled is false (matches prior build_effective default).
LEGACY_ADD_DELTA_THRESHOLD = 0.002


# ------------------------------------------------------------
# Lifecycle Config
# ------------------------------------------------------------


@dataclass
class LifecycleConfig:
    # Minimum days to hold a new LONG before allowing EXIT/TRIM
    min_hold_days: int = 1

    # Cooldown days after EXIT before allowing a new BUY
    cooldown_days_after_exit: int = 1

    # Thresholds based on delta_pct magnitude (e.g. 0.002 = 0.20%)
    buy_delta_pct: float = 0.002
    add_delta_pct: float = 0.002
    exit_delta_pct: float = -0.002
    trim_delta_pct: float = -0.004

    # If signal is HOLD and we are LONG -> HOLD
    hold_means_keep_position: bool = True

    # If we can't parse signal, do nothing
    default_action_when_unknown: str = "WAIT"


@dataclass
class LifecycleLogicConfig:
    """Thresholds for LONG-position stance quality (config/lifecycle_logic.json)."""

    enabled: bool = True
    add_confidence_min: float = 0.60
    add_delta_pct_min: float = 0.008
    hold_delta_floor: float = -0.002
    hold_delta_ceiling: float = 0.006
    trim_delta_pct_threshold: float = -0.001
    exit_delta_pct_threshold: float = -0.004
    exit_confidence_min: float = 0.55


# ------------------------------------------------------------
# State-aware decision engine thresholds
# ------------------------------------------------------------


@dataclass
class DecisionLogicConfig:
    """
    Thresholds for the state-aware decision_action layer.
    Independent from LifecycleLogicConfig so broker/execution behavior is unchanged.
    """

    entry_confidence_min: float = 0.68
    add_confidence_min: float = 0.74
    trim_confidence_floor: float = 0.55
    exit_confidence_max: float = 0.45
    confidence_delta_for_add: float = 0.04
    confidence_drop_for_trim: float = -0.08
    score_delta_for_add: float = 0.03
    score_drop_for_trim: float = -0.05


def default_decision_logic() -> DecisionLogicConfig:
    return DecisionLogicConfig()


def default_lifecycle_logic_dict() -> Dict[str, Any]:
    """Defaults match config/lifecycle_logic.json (single source for JSON + dict loaders)."""
    return {
        "enabled": True,
        "add_confidence_min": 0.60,
        "add_delta_pct_min": 0.008,
        "hold_delta_floor": -0.002,
        "hold_delta_ceiling": 0.006,
        "trim_delta_pct_threshold": -0.001,
        "exit_delta_pct_threshold": -0.004,
        "exit_confidence_min": 0.55,
    }


def lifecycle_logic_from_dict(d: Mapping[str, Any]) -> LifecycleLogicConfig:
    """Build LifecycleLogicConfig from merged dict (no hardcoded thresholds outside defaults)."""
    base = default_lifecycle_logic_dict()
    merged: Dict[str, Any] = dict(base)
    merged.update(dict(d))
    return LifecycleLogicConfig(
        enabled=bool(merged["enabled"]),
        add_confidence_min=float(merged["add_confidence_min"]),
        add_delta_pct_min=float(merged["add_delta_pct_min"]),
        hold_delta_floor=float(merged["hold_delta_floor"]),
        hold_delta_ceiling=float(merged["hold_delta_ceiling"]),
        trim_delta_pct_threshold=float(merged["trim_delta_pct_threshold"]),
        exit_delta_pct_threshold=float(merged["exit_delta_pct_threshold"]),
        exit_confidence_min=float(merged["exit_confidence_min"]),
    )


def load_lifecycle_logic_config(path: Optional[Path] = None) -> LifecycleLogicConfig:
    """Load lifecycle logic from JSON; same defaults as apply_signal_lifecycle.load_lifecycle_config()."""
    p = path or LIFECYCLE_LOGIC_PATH
    merged = default_lifecycle_logic_dict()
    if not p.exists():
        return lifecycle_logic_from_dict(merged)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            merged.update(raw)
    except Exception:
        pass
    return lifecycle_logic_from_dict(merged)


def _scalar_float(x: Any) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float("nan")
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


CONFIDENCE_CONFLICT_EPS = 1e-6
CONFIDENCE_REASON_OK = "confidence"
CONFIDENCE_REASON_SCORE_ALIAS = "confidence_score"
CONFIDENCE_REASON_MISSING = "missing_or_nonfinite_confidence"
CONFIDENCE_REASON_CONFLICT = "conflicting_confidence"


def resolve_row_confidence(row: Mapping[str, Any] | pd.Series) -> Tuple[Optional[float], str]:
    """Resolve one finite confidence or reject missing/nonfinite/conflicting evidence.

    ``signals_with_rationale.csv`` carries ``confidence``. Legacy ``signals.csv``
    renames the same proven score to ``confidence_score``. Prefer ``confidence``
    when it is finite. Accept ``confidence_score`` only when ``confidence`` is
    absent/nonfinite. If both are finite and disagree, reject — do not pick a
    winner and do not invent ``delta_pct`` or a strong edge.
    """
    if row is None:
        return None, CONFIDENCE_REASON_MISSING
    getter = row.get if hasattr(row, "get") else None
    if getter is None:
        return None, CONFIDENCE_REASON_MISSING
    primary = _scalar_float(getter("confidence"))
    alias = _scalar_float(getter("confidence_score"))
    primary_ok = math.isfinite(primary)
    alias_ok = math.isfinite(alias)
    if primary_ok and alias_ok and abs(primary - alias) > CONFIDENCE_CONFLICT_EPS:
        return None, CONFIDENCE_REASON_CONFLICT
    if primary_ok:
        return float(primary), CONFIDENCE_REASON_OK
    if alias_ok:
        return float(alias), CONFIDENCE_REASON_SCORE_ALIAS
    return None, CONFIDENCE_REASON_MISSING


def long_buy_qualifies_for_add(
    delta_pct: Any,
    confidence: Any,
    *,
    logic: LifecycleLogicConfig,
) -> bool:
    """ADD quality is never granted from Random Forest residual.

    Operational delta_pct is non-authoritative (NaN). A finite residual must
    not promote LONG+BUY to ADD.
    """
    _ = (delta_pct, confidence, logic)
    return False


def long_qualifies_for_exit_delta(
    delta_pct: Any,
    *,
    logic: LifecycleLogicConfig,
) -> bool:
    """Predicted-return residual must not create an EXIT."""
    _ = (delta_pct, logic)
    return False


def long_qualifies_for_trim(
    delta_pct: Any,
    *,
    logic: LifecycleLogicConfig,
) -> bool:
    """Predicted-return residual must not create a TRIM."""
    _ = (delta_pct, logic)
    return False


# ------------------------------------------------------------
# State I/O
# ------------------------------------------------------------


def _today_utc_date() -> date:
    return datetime.now(timezone.utc).date()


def load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists() or state_path.stat().st_size == 0:
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _parse_date_any(x: Any) -> Optional[date]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return None
        if hasattr(dt, "to_pydatetime"):
            return dt.to_pydatetime().date()
        return dt.date()
    except Exception:
        return None


# ------------------------------------------------------------
# Signal normalization
# ------------------------------------------------------------


def _normalize_signal_value(sig: Any) -> str:
    """
    Return one of: BUY, SELL, HOLD, UNKNOWN.
    Supports numeric (-1,0,1) and string signals.
    """
    if sig is None or (isinstance(sig, float) and np.isnan(sig)):
        return "UNKNOWN"

    # numeric signal
    if isinstance(sig, (int, float, np.integer, np.floating)):
        try:
            v = float(sig)
        except Exception:
            return "UNKNOWN"
        if v > 0:
            return "BUY"
        if v < 0:
            return "SELL"
        return "HOLD"

    s = str(sig).strip().upper()
    if s in ("BUY", "B", "LONG", "1"):
        return "BUY"
    if s in ("SELL", "S", "SHORT", "-1"):
        return "SELL"
    if s in ("HOLD", "H", "WAIT", "0"):
        return "HOLD"
    return "UNKNOWN"


def _coerce_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _ensure_delta_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Keep operational delta_pct non-authoritative.

    Random Forest predicted_close must never refill delta_pct. Missing,
    malformed, or empty values stay NaN so consumers fail closed instead of
    recovering the old residual gate.
    """
    if "delta_pct" not in df.columns:
        df["delta_pct"] = np.nan
        return df
    df["delta_pct"] = pd.to_numeric(df["delta_pct"], errors="coerce")
    return df


def _get_strength(row: pd.Series) -> float:
    """
    Return a strength measure used for thresholds.
    Prefer delta_pct. Else try confidence.
    """
    if "delta_pct" in row.index:
        try:
            v = float(row.get("delta_pct"))
            if np.isfinite(v):
                return v
        except Exception:
            pass

    if "confidence" in row.index:
        try:
            c = float(row.get("confidence"))
            if np.isfinite(c):
                return c
        except Exception:
            pass

    return float("nan")


# ------------------------------------------------------------
# Lifecycle transition logic
# ------------------------------------------------------------


def _days_between(d1: Optional[date], d2: Optional[date]) -> int:
    if not d1 or not d2:
        return 10_000
    return abs((d2 - d1).days)


def _get_ticker_state(state: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    t = state.get(ticker, {})
    if "position_state" not in t:
        t["position_state"] = "FLAT"  # FLAT or LONG
    if "last_action" not in t:
        t["last_action"] = "NONE"
    if "last_change_date" not in t:
        t["last_change_date"] = None
    if "cooldown_until" not in t:
        t["cooldown_until"] = None
    return t


def _set_ticker_state(state: Dict[str, Any], ticker: str, tstate: Dict[str, Any]) -> None:
    state[ticker] = tstate


def _rf_era_decide_action(
    *,
    cfg: LifecycleConfig,
    logic: LifecycleLogicConfig,
    signal: str,
    rf_delta_pct: float,
    confidence: float,
    pos_state: str,
    last_change_date: Optional[date],
    cooldown_until: Optional[date],
    as_of_date: Optional[date],
) -> Tuple[str, str]:
    """Pre-RF2 counterfactual: residual used as a hard gate. Diagnostic only."""
    if as_of_date is None:
        as_of_date = _today_utc_date()

    if cooldown_until and as_of_date < cooldown_until and signal == "BUY" and pos_state == "FLAT":
        return "WAIT", "cooldown_flat_buy_blocked"

    hold_age = _days_between(last_change_date, as_of_date)
    can_exit = hold_age >= max(0, cfg.min_hold_days)

    d = rf_delta_pct
    c = confidence
    if not math.isfinite(d):
        d = float("nan")
    if not math.isfinite(c):
        c = float("nan")

    if pos_state == "FLAT":
        if signal == "BUY":
            strength = d if math.isfinite(d) else (c if math.isfinite(c) else 0.0)
            if not math.isfinite(strength):
                strength = 0.0
            if strength >= cfg.buy_delta_pct:
                return "BUY", "flat_buy_to_buy"
            return "WAIT", "flat_buy_weak_to_wait"
        return "WAIT", "flat_non_buy_wait"

    if pos_state == "LONG":
        if logic.enabled:
            if can_exit:
                if math.isfinite(d) and d <= logic.exit_delta_pct_threshold:
                    return "EXIT", "long_exit_delta_threshold"
                if math.isfinite(d) and d <= logic.trim_delta_pct_threshold:
                    return "TRIM", "long_mild_deterioration_to_trim"
        else:
            if signal == "SELL":
                if not can_exit:
                    return "HOLD", "long_sell_blocked_min_hold"
                if math.isfinite(d) and d <= cfg.trim_delta_pct:
                    return "EXIT", "long_bearish_reversal_to_exit"
                if math.isfinite(d) and d <= cfg.exit_delta_pct:
                    return "TRIM", "long_mild_deterioration_to_trim"
                return "HOLD", "long_sell_but_not_strong"

        if signal == "BUY":
            if logic.enabled:
                d_ok = math.isfinite(d) and d >= logic.add_delta_pct_min
                c_ok = math.isfinite(c) and c >= logic.add_confidence_min
                if d_ok and c_ok:
                    return "ADD", "long_strong_buy_to_add"
                return "HOLD", "long_weak_buy_to_hold"
            strength = d if math.isfinite(d) else (c if math.isfinite(c) else 0.0)
            if not math.isfinite(strength):
                strength = 0.0
            if strength >= cfg.add_delta_pct:
                return "ADD", "long_strong_buy_to_add"
            return "HOLD", "long_weak_buy_to_hold"

        if signal == "SELL":
            if not can_exit:
                return "HOLD", "long_sell_blocked_min_hold"
            if logic.enabled:
                if math.isfinite(c) and c >= logic.exit_confidence_min:
                    return "EXIT", "long_bearish_reversal_to_exit"
                return "HOLD", "long_sell_but_not_strong"

        if signal == "HOLD":
            return (
                ("HOLD", "long_hold_signal_hold")
                if cfg.hold_means_keep_position
                else ("WAIT", "long_hold_to_wait")
            )

        return cfg.default_action_when_unknown, "unknown_signal_long"

    return cfg.default_action_when_unknown, "unknown_position_state"


def decide_action(
    *,
    cfg: LifecycleConfig,
    logic: LifecycleLogicConfig,
    signal: str,
    delta_pct: float,
    confidence: float,
    pos_state: str,
    last_change_date: Optional[date],
    cooldown_until: Optional[date],
    as_of_date: Optional[date],
) -> Tuple[str, str]:
    """
    Returns (lifecycle_action, lifecycle_decision_reason).

    Random Forest residual / operational delta_pct is never used to veto BUY,
    promote HOLD to TRIM/EXIT, or grant ADD. Feature signal, cooldown, min-hold,
    and explicit SELL conviction remain.
    """
    if as_of_date is None:
        as_of_date = _today_utc_date()

    if cooldown_until and as_of_date < cooldown_until and signal == "BUY" and pos_state == "FLAT":
        return "WAIT", "cooldown_flat_buy_blocked"

    hold_age = _days_between(last_change_date, as_of_date)
    can_exit = hold_age >= max(0, cfg.min_hold_days)

    c = confidence
    if not math.isfinite(c):
        c = float("nan")
    _ = delta_pct  # operational residual is non-authoritative

    if pos_state == "FLAT":
        if signal == "BUY":
            if not math.isfinite(c):
                return "WAIT", "flat_buy_missing_confidence"
            if c >= cfg.buy_delta_pct:
                return "BUY", "flat_buy_to_buy"
            return "WAIT", "flat_buy_weak_to_wait"
        return "WAIT", "flat_non_buy_wait"

    if pos_state == "LONG":
        if not logic.enabled:
            if signal == "SELL":
                if not can_exit:
                    return "HOLD", "long_sell_blocked_min_hold"
                return "EXIT", "long_bearish_reversal_to_exit"
            if signal == "BUY":
                return "HOLD", "long_weak_buy_to_hold"
            if signal == "HOLD":
                return (
                    ("HOLD", "long_hold_signal_hold")
                    if cfg.hold_means_keep_position
                    else ("WAIT", "long_hold_to_wait")
                )
            return cfg.default_action_when_unknown, "unknown_signal_long"

        if signal == "BUY":
            return "HOLD", "long_weak_buy_to_hold"

        if signal == "SELL":
            if not can_exit:
                return "HOLD", "long_sell_blocked_min_hold"
            if math.isfinite(c) and c >= logic.exit_confidence_min:
                return "EXIT", "long_bearish_reversal_to_exit"
            return "HOLD", "long_sell_but_not_strong"

        if signal == "HOLD":
            return (
                ("HOLD", "long_hold_signal_hold")
                if cfg.hold_means_keep_position
                else ("WAIT", "long_hold_to_wait")
            )

        return cfg.default_action_when_unknown, "unknown_signal_long"

    return cfg.default_action_when_unknown, "unknown_position_state"


# ------------------------------------------------------------
# State-aware decision engine (decision_action layer)
# ------------------------------------------------------------


_ALLOWED_DECISION_ACTIONS = {"BUY", "HOLD", "ADD", "TRIM", "EXIT", "WAIT"}


def _safe_number(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def decide_state_aware_action(
    *,
    signal: str,
    confidence: float,
    score: float,
    prior_signal: Optional[str],
    prior_confidence: float,
    prior_score: float,
    held_state: str,
    cfg: DecisionLogicConfig,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Decide one of {BUY, HOLD, ADD, TRIM, EXIT, WAIT} with a short reason, given:
      - current signal/confidence/score
      - prior signal/confidence/score (may be missing)
      - held_state: "FLAT" or "LONG"

    Returns (action, reason, diagnostics).
    diagnostics contains: confidence_change, score_change, signal_changed,
                          strengthening, weakening.

    No shorting logic.
    """
    sig = (signal or "").strip().upper()
    held = (held_state or "FLAT").strip().upper()
    if held not in ("FLAT", "LONG"):
        held = "FLAT"

    c_now = _safe_number(confidence)
    s_now = _safe_number(score)
    c_prev = _safe_number(prior_confidence)
    s_prev = _safe_number(prior_score)

    conf_change = (c_now - c_prev) if (math.isfinite(c_now) and math.isfinite(c_prev)) else 0.0
    score_change = (s_now - s_prev) if (math.isfinite(s_now) and math.isfinite(s_prev)) else 0.0

    prior_sig_norm = (prior_signal or "").strip().upper() if prior_signal else ""
    signal_changed = bool(prior_sig_norm) and prior_sig_norm != sig

    # If score feed is unavailable for either side, fall back to confidence-only for strengthening/weakening.
    score_available = math.isfinite(s_now) and math.isfinite(s_prev)
    if score_available:
        strengthening = (conf_change > cfg.confidence_delta_for_add) and (
            score_change > cfg.score_delta_for_add
        )
        weakening = (conf_change < cfg.confidence_drop_for_trim) and (
            score_change < cfg.score_drop_for_trim
        )
    else:
        strengthening = conf_change > cfg.confidence_delta_for_add
        weakening = conf_change < cfg.confidence_drop_for_trim

    diagnostics = {
        "confidence_change": float(conf_change),
        "score_change": float(score_change),
        "signal_changed": bool(signal_changed),
        "strengthening": bool(strengthening),
        "weakening": bool(weakening),
    }

    # --- Rule 3: FLAT + SELL => WAIT (no shorts) ---
    if held == "FLAT":
        if sig == "BUY" and math.isfinite(c_now) and c_now >= cfg.entry_confidence_min:
            return "BUY", "new bullish entry", diagnostics
        if sig == "BUY":
            return "WAIT", "bullish signal below entry confidence", diagnostics
        if sig == "SELL":
            return "WAIT", "no shorts: flat and bearish", diagnostics
        return "WAIT", "flat with non-actionable signal", diagnostics

    # --- held_state == LONG ---
    c_now_val = c_now if math.isfinite(c_now) else 0.0

    # EXIT must be checked before anything that keeps position.
    # Trigger on explicit SELL OR confidence collapsing below exit_confidence_max.
    if sig == "SELL" or c_now_val <= cfg.exit_confidence_max:
        return "EXIT", "bearish deterioration / broken setup", diagnostics

    if sig == "BUY":
        if math.isfinite(c_now) and c_now >= cfg.add_confidence_min and strengthening:
            return "ADD", "bullish signal strengthened while already long", diagnostics
        # Bullish but not materially stronger -> keep the position.
        return "HOLD", "bullish signal intact while already long", diagnostics

    if sig == "HOLD":
        # Weakening while long but not broken -> TRIM; else HOLD.
        if weakening and c_now_val < cfg.trim_confidence_floor:
            return "TRIM", "signal weakening while long", diagnostics
        if c_now_val < cfg.trim_confidence_floor and (conf_change < cfg.confidence_drop_for_trim):
            return "TRIM", "conviction fell below trim floor", diagnostics
        return "HOLD", "conviction softened but not enough to reduce", diagnostics

    # Unknown/missing signal while long -> keep position, flag as HOLD.
    return "HOLD", "unknown signal while long", diagnostics


# ------------------------------------------------------------
# Apply lifecycle over signals dataframe
# ------------------------------------------------------------


def apply_lifecycle(
    signals_df: pd.DataFrame,
    *,
    state_path: Path,
    cfg: LifecycleConfig,
    lifecycle_logic: Optional[LifecycleLogicConfig] = None,
    decision_state_path: Optional[Path] = None,
    decision_logic: Optional[DecisionLogicConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    logic = lifecycle_logic or load_lifecycle_logic_config()
    df = signals_df.copy()

    # Separate state file for the state-aware decision layer so we don't conflict
    # with other writers of signal_state.json (e.g. place_live_orders.py).
    if decision_state_path is None:
        decision_state_path = Path(state_path).parent / "decision_prior_state.json"
    prior_state = load_state(decision_state_path)

    if "ticker" not in df.columns:
        raise ValueError("signals_df must include 'ticker'")

    # dates
    if "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "as_of_date" in df.columns and df["as_of_date"].notna().any():
        df["_asof"] = df["as_of_date"]
    elif "date" in df.columns:
        df["_asof"] = df["date"]
    else:
        df["_asof"] = pd.NaT

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # ✅ new: ensure delta_pct exists if we can derive it
    df = _ensure_delta_pct(df)

    # normalize signal
    if "signal" in df.columns:
        df["_signal_norm"] = df["signal"].apply(_normalize_signal_value)
    else:
        df["_signal_norm"] = "UNKNOWN"

    # sort time order
    df = df.sort_values(
        ["_asof", "ticker"], ascending=[True, True], na_position="last"
    ).reset_index(drop=True)

    state = load_state(state_path)

    # State-aware decision layer config (kept separate from lifecycle thresholds
    # so broker/execution behavior is unchanged).
    decision_cfg = decision_logic or default_decision_logic()

    lifecycle_actions = []
    lifecycle_decision_reasons: list[str] = []
    position_states = []
    last_actions = []
    state_changed_flags = []

    # Parallel buffers for the new decision_action layer.
    decision_actions: list[str] = []
    decision_reasons: list[str] = []
    prior_signals: list[str] = []
    prior_confidences: list[float] = []
    prior_scores: list[float] = []
    confidence_changes: list[float] = []
    score_changes: list[float] = []
    signal_changed_flags: list[bool] = []
    held_states: list[str] = []
    state_transitions: list[str] = []
    rf_former_actions: list[str] = []
    rf_former_reasons: list[str] = []
    rf_would_have_changed: list[bool] = []

    for i in range(len(df)):
        row = df.iloc[i]
        ticker = row["ticker"]
        as_of = _parse_date_any(row["_asof"])

        tstate = _get_ticker_state(state, ticker)
        pos_state = str(tstate.get("position_state", "FLAT")).upper()
        last_action = str(tstate.get("last_action", "NONE")).upper()
        last_change_date = _parse_date_any(tstate.get("last_change_date"))
        cooldown_until = _parse_date_any(tstate.get("cooldown_until"))

        sig = row["_signal_norm"]
        delta_pct = _scalar_float(row.get("delta_pct"))
        resolved_conf, _conf_reason = resolve_row_confidence(row)
        confidence = float(resolved_conf) if resolved_conf is not None else float("nan")
        # Persist the proven score under ``confidence`` so effective lifecycle
        # and allocation see the same finite value. Do not invent delta_pct.
        persist = confidence if math.isfinite(confidence) else np.nan
        if "confidence" not in df.columns:
            df["confidence"] = np.nan
        df.iloc[i, df.columns.get_loc("confidence")] = persist
        score_val = _scalar_float(row.get("score"))

        action, decision_reason = decide_action(
            cfg=cfg,
            logic=logic,
            signal=sig,
            delta_pct=delta_pct,
            confidence=confidence,
            pos_state=pos_state,
            last_change_date=last_change_date,
            cooldown_until=cooldown_until,
            as_of_date=as_of,
        )
        rf_delta = _scalar_float(row.get("rf_delta_pct"))
        former_action, former_reason = _rf_era_decide_action(
            cfg=cfg,
            logic=logic,
            signal=sig,
            rf_delta_pct=rf_delta,
            confidence=confidence,
            pos_state=pos_state,
            last_change_date=last_change_date,
            cooldown_until=cooldown_until,
            as_of_date=as_of,
        )

        # --- State-aware decision_action layer (non-authoritative; parallel to lifecycle) ---
        pstate = prior_state.get(ticker, {}) if isinstance(prior_state, dict) else {}
        if not isinstance(pstate, dict):
            pstate = {}
        prior_sig = str(pstate.get("prior_signal") or "") or ""
        prior_conf = _scalar_float(pstate.get("prior_confidence"))
        prior_sc = _scalar_float(pstate.get("prior_score"))
        held_now = "LONG" if str(pos_state).upper() == "LONG" else "FLAT"

        decision_action, decision_reason_new, diag = decide_state_aware_action(
            signal=sig,
            confidence=confidence,
            score=score_val,
            prior_signal=prior_sig,
            prior_confidence=prior_conf,
            prior_score=prior_sc,
            held_state=held_now,
            cfg=decision_cfg,
        )

        # Derive state_transition label from held_state + decision_action.
        if held_now == "FLAT":
            if decision_action == "BUY":
                state_transition = "FLAT_TO_LONG"
            else:
                state_transition = "FLAT_WAIT"
        else:  # LONG
            if decision_action == "ADD":
                state_transition = "LONG_ADD"
            elif decision_action == "TRIM":
                state_transition = "LONG_TRIM"
            elif decision_action == "EXIT":
                state_transition = "LONG_EXIT"
            else:
                state_transition = "LONG_HOLD"

        changed = False

        # Apply transitions
        if action == "BUY":
            if pos_state == "FLAT":
                pos_state = "LONG"
                last_change_date = as_of or _today_utc_date()
                last_action = "BUY"
                cooldown_until = None
                changed = True

        elif action == "ADD":
            if pos_state == "LONG":
                last_action = "ADD"
                changed = True

        elif action == "TRIM":
            if pos_state == "LONG":
                last_action = "TRIM"
                changed = True

        elif action == "EXIT":
            if pos_state == "LONG":
                pos_state = "FLAT"
                last_action = "EXIT"
                last_change_date = as_of or _today_utc_date()
                cd = (as_of or _today_utc_date()) + pd.Timedelta(days=cfg.cooldown_days_after_exit)
                cooldown_until = cd.date() if hasattr(cd, "date") else None
                changed = True

        elif action == "HOLD":
            if pos_state == "LONG":
                last_action = "HOLD"

        # persist lifecycle state (unchanged, shared with other writers)
        tstate["position_state"] = pos_state
        tstate["last_action"] = last_action
        tstate["last_change_date"] = str(last_change_date) if last_change_date else None
        tstate["cooldown_until"] = str(cooldown_until) if cooldown_until else None
        _set_ticker_state(state, ticker, tstate)

        # Snapshot CURRENT row into the decision-layer state (separate file).
        prior_state[ticker] = {
            "prior_signal": sig,
            "prior_confidence": (
                float(confidence) if math.isfinite(_scalar_float(confidence)) else None
            ),
            "prior_score": (float(score_val) if math.isfinite(_scalar_float(score_val)) else None),
            "held_state": held_now,
            "last_updated_asof": str(as_of) if as_of else None,
        }

        lifecycle_actions.append(action)
        lifecycle_decision_reasons.append(decision_reason)
        position_states.append(pos_state)
        last_actions.append(last_action)
        state_changed_flags.append(bool(changed))

        decision_actions.append(decision_action)
        decision_reasons.append(decision_reason_new)
        prior_signals.append(prior_sig or "")
        prior_confidences.append(float(prior_conf) if math.isfinite(prior_conf) else float("nan"))
        prior_scores.append(float(prior_sc) if math.isfinite(prior_sc) else float("nan"))
        confidence_changes.append(float(diag["confidence_change"]))
        score_changes.append(float(diag["score_change"]))
        signal_changed_flags.append(bool(diag["signal_changed"]))
        held_states.append(held_now)
        state_transitions.append(state_transition)
        rf_former_actions.append(former_action)
        rf_former_reasons.append(former_reason)
        rf_would_have_changed.append(former_action != action)

    df["lifecycle_action"] = lifecycle_actions
    df["lifecycle_decision_reason"] = lifecycle_decision_reasons
    df["stance"] = df["lifecycle_action"]  # backwards compat
    df["position_state"] = position_states
    df["last_action"] = last_actions
    df["state_changed"] = state_changed_flags

    # --- New state-aware decision columns (safe, downstream may ignore) ---
    df["decision_action"] = decision_actions
    df["decision_reason"] = decision_reasons
    df["prior_signal"] = prior_signals
    df["prior_confidence"] = prior_confidences
    df["prior_score"] = prior_scores
    df["confidence_change"] = confidence_changes
    df["score_change"] = score_changes
    df["signal_changed"] = signal_changed_flags
    df["held_state"] = held_states
    df["state_transition"] = state_transitions
    df["rf_shadow_only"] = True
    df["rf_former_lifecycle_action"] = rf_former_actions
    df["rf_former_lifecycle_reason"] = rf_former_reasons
    df["rf_would_have_changed"] = rf_would_have_changed
    if "rf_delta_pct" not in df.columns:
        df["rf_delta_pct"] = np.nan
    if "rf_predicted_close" not in df.columns:
        df["rf_predicted_close"] = np.nan
    if "rf_evidence_status" not in df.columns:
        df["rf_evidence_status"] = "unavailable"
    if "rf_evidence_timestamp" not in df.columns:
        df["rf_evidence_timestamp"] = ""

    # Fill NaNs in new numeric columns with 0.0 (safe defaults for first-seen tickers).
    for _col in ("prior_confidence", "prior_score", "confidence_change", "score_change"):
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors="coerce").fillna(0.0)

    df = df.drop(columns=["_signal_norm", "_asof"], errors="ignore")

    # FINAL STATE: one row per ticker (authoritative lifecycle output)
    ts = None
    if "as_of_date" in df.columns:
        ts = pd.to_datetime(df["as_of_date"], errors="coerce")
    if "date" in df.columns:
        d2 = pd.to_datetime(df["date"], errors="coerce")
        if ts is None:
            ts = d2
        else:
            ts = ts.fillna(d2)
    if ts is not None:
        df = df.assign(_sort_ts=ts)
        df = df.sort_values(["ticker", "_sort_ts"], ascending=[True, False], na_position="last")
        df = df.drop(columns=["_sort_ts"])
    else:
        df = df.sort_values("ticker", ascending=True)

    df = df.drop_duplicates("ticker", keep="first").reset_index(drop=True)

    df = df.drop(columns=[c for c in df.columns if str(c).startswith("_")], errors="ignore")

    preserve_order = [
        "ticker",
        "stance",
        "lifecycle_action",
        "lifecycle_decision_reason",
        "decision_action",
        "decision_reason",
        "state_transition",
        "held_state",
        "position_state",
        "last_action",
        "confidence",
        "score",
        "prior_signal",
        "prior_confidence",
        "prior_score",
        "confidence_change",
        "score_change",
        "signal_changed",
        "delta_pct",
        "edge_pct",
        "as_of_date",
        "date",
    ]
    front = [c for c in preserve_order if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    # Persist decision-layer prior state (isolated file; no impact on signal_state.json).
    try:
        save_state(decision_state_path, prior_state)
    except Exception:
        pass

    return df, state
