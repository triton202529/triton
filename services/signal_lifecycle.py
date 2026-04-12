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
    add_confidence_min: float = 0.62
    add_delta_pct_min: float = 0.01
    hold_delta_floor: float = -0.002
    hold_delta_ceiling: float = 0.008
    trim_delta_pct_threshold: float = -0.002
    exit_delta_pct_threshold: float = -0.006
    exit_confidence_min: float = 0.58


def default_lifecycle_logic_dict() -> Dict[str, Any]:
    """Defaults match config/lifecycle_logic.json (single source for JSON + dict loaders)."""
    return {
        "enabled": True,
        "add_confidence_min": 0.62,
        "add_delta_pct_min": 0.01,
        "hold_delta_floor": -0.002,
        "hold_delta_ceiling": 0.008,
        "trim_delta_pct_threshold": -0.002,
        "exit_delta_pct_threshold": -0.006,
        "exit_confidence_min": 0.58,
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


def long_buy_qualifies_for_add(
    delta_pct: Any,
    confidence: Any,
    *,
    logic: LifecycleLogicConfig,
) -> bool:
    """Used by build_effective_lifecycle to align LONG+BUY→ADD with the same rules as decide_action."""
    d = _scalar_float(delta_pct)
    c = _scalar_float(confidence)
    if not logic.enabled:
        return math.isfinite(d) and d >= LEGACY_ADD_DELTA_THRESHOLD
    d_ok = math.isfinite(d) and d >= logic.add_delta_pct_min
    c_ok = math.isfinite(c) and c >= logic.add_confidence_min
    return d_ok and c_ok


def long_qualifies_for_exit_delta(
    delta_pct: Any,
    *,
    logic: LifecycleLogicConfig,
) -> bool:
    """Signal-agnostic EXIT on deterioration (same as first EXIT check in decide_action)."""
    if not logic.enabled:
        return False
    d = _scalar_float(delta_pct)
    return math.isfinite(d) and d <= logic.exit_delta_pct_threshold


def long_qualifies_for_trim(
    delta_pct: Any,
    *,
    logic: LifecycleLogicConfig,
) -> bool:
    """TRIM band for LONG (after EXIT checks)."""
    if not logic.enabled:
        return False
    d = _scalar_float(delta_pct)
    return math.isfinite(d) and d <= logic.trim_delta_pct_threshold


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
    """
    If delta_pct missing, derive it from predicted_close + close when possible:
      delta_pct = (predicted_close - close) / close
    Also supports alt column names.
    """
    if "delta_pct" in df.columns and df["delta_pct"].notna().any():
        return df

    # find close column
    close_col = None
    for c in ("close", "Close", "price", "last", "last_price"):
        if c in df.columns:
            close_col = c
            break

    # find predicted close column
    pred_col = None
    for c in ("predicted_close", "pred", "yhat", "pred_close", "predicted"):
        if c in df.columns:
            pred_col = c
            break

    if close_col and pred_col:
        c = _coerce_float(df[close_col])
        p = _coerce_float(df[pred_col])
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = (p - c) / c.replace(0.0, np.nan)
        df["delta_pct"] = delta
    else:
        # keep column present for downstream expectations
        df["delta_pct"] = np.nan

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
    LONG: EXIT/TRIM from delta/confidence (when enabled) run before ADD/HOLD for any signal.
    """
    if as_of_date is None:
        as_of_date = _today_utc_date()

    if cooldown_until and as_of_date < cooldown_until and signal == "BUY" and pos_state == "FLAT":
        return "WAIT", "cooldown_flat_buy_blocked"

    hold_age = _days_between(last_change_date, as_of_date)
    can_exit = hold_age >= max(0, cfg.min_hold_days)

    d = delta_pct
    c = confidence
    if not math.isfinite(d):
        d = float("nan")
    if not math.isfinite(c):
        c = float("nan")

    # FLAT — preserve prior behavior (strength = delta then confidence)
    if pos_state == "FLAT":
        if signal == "BUY":
            strength = d if math.isfinite(d) else (c if math.isfinite(c) else 0.0)
            if not math.isfinite(strength):
                strength = 0.0
            if strength >= cfg.buy_delta_pct:
                return "BUY", "flat_buy_to_buy"
            return "WAIT", "flat_buy_weak_to_wait"
        return "WAIT", "flat_non_buy_wait"

    # LONG — deterioration (EXIT/TRIM) is evaluated before ADD/HOLD for any signal.
    if pos_state == "LONG":
        if logic.enabled:
            if can_exit:
                # 1) EXIT — delta only (applies before ADD/HOLD for any signal)
                if math.isfinite(d) and d <= logic.exit_delta_pct_threshold:
                    return "EXIT", "long_exit_delta_threshold"
                # 2) TRIM
                if math.isfinite(d) and d <= logic.trim_delta_pct_threshold:
                    return "TRIM", "long_mild_deterioration_to_trim"
        else:
            # Legacy: trim/exit only on explicit SELL
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
                # 3) ADD — only after EXIT/TRIM bands did not fire
                if long_buy_qualifies_for_add(d, c, logic=logic):
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
                # 3b) EXIT on bearish conviction after delta bands (explicit SELL only)
                if math.isfinite(c) and c >= logic.exit_confidence_min:
                    return "EXIT", "long_bearish_reversal_to_exit"
                return "HOLD", "long_sell_but_not_strong"
            # logic.disabled + SELL handled in legacy branch above

        if signal == "HOLD":
            return (
                ("HOLD", "long_hold_signal_hold")
                if cfg.hold_means_keep_position
                else ("WAIT", "long_hold_to_wait")
            )

        return cfg.default_action_when_unknown, "unknown_signal_long"

    return cfg.default_action_when_unknown, "unknown_position_state"


# ------------------------------------------------------------
# Apply lifecycle over signals dataframe
# ------------------------------------------------------------


def apply_lifecycle(
    signals_df: pd.DataFrame,
    *,
    state_path: Path,
    cfg: LifecycleConfig,
    lifecycle_logic: Optional[LifecycleLogicConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    logic = lifecycle_logic or load_lifecycle_logic_config()
    df = signals_df.copy()

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

    lifecycle_actions = []
    lifecycle_decision_reasons: list[str] = []
    position_states = []
    last_actions = []
    state_changed_flags = []

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
        confidence = _scalar_float(row.get("confidence"))

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

        # persist
        tstate["position_state"] = pos_state
        tstate["last_action"] = last_action
        tstate["last_change_date"] = str(last_change_date) if last_change_date else None
        tstate["cooldown_until"] = str(cooldown_until) if cooldown_until else None
        _set_ticker_state(state, ticker, tstate)

        lifecycle_actions.append(action)
        lifecycle_decision_reasons.append(decision_reason)
        position_states.append(pos_state)
        last_actions.append(last_action)
        state_changed_flags.append(bool(changed))

    df["lifecycle_action"] = lifecycle_actions
    df["lifecycle_decision_reason"] = lifecycle_decision_reasons
    df["stance"] = df["lifecycle_action"]  # backwards compat
    df["position_state"] = position_states
    df["last_action"] = last_actions
    df["state_changed"] = state_changed_flags

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
        "position_state",
        "last_action",
        "confidence",
        "delta_pct",
        "edge_pct",
        "as_of_date",
        "date",
    ]
    front = [c for c in preserve_order if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    return df, state
