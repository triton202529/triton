# services/capital_preservation_engine.py
# ------------------------------------------------------------
# TRITON — Capital Preservation Engine (CPE)
# Prime Directive: NEVER permanently damage seed/client capital.
#
# HARDENED UPDATE:
# - If portfolio-risk signals are missing (drawdown/daily_loss/vol),
#   we fail-closed to DEFENSIVE/CPM for LIVE safety (configurable).
# - Manual override supported via data/results/cpm_override.json:
#     { "force_mode": "LOCKDOWN" | "CPM" | "DEFENSIVE" | "NORMAL",
#       "reason": "string",
#       "expires_at": "2026-01-11T12:00:00Z" }
# - If override is active, it is SUPREME.
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CPMConfig:
    # Seed capital (optional). You can set TRITON_INITIAL_CAPITAL env too.
    initial_capital: Optional[float] = None

    # Drawdown bands (from rolling peak equity)
    dd_defensive: float = 0.03
    dd_cpm: float = 0.06
    dd_lockdown: float = 0.10

    # Daily loss lock (from start-of-day equity)
    daily_loss_defensive: float = 0.005
    daily_loss_cpm: float = 0.010
    daily_loss_lockdown: float = 0.015

    # Loss-streak guard
    max_consecutive_losses_defensive: int = 2
    max_consecutive_losses_cpm: int = 3
    max_consecutive_losses_lockdown: int = 4

    # Volatility gating (equity curve daily returns std)
    vol_window_days: int = 20
    vol_defensive: float = 0.02
    vol_cpm: float = 0.03
    vol_lockdown: float = 0.05

    # Risk caps output
    risk_per_trade_normal: float = 0.0025
    risk_per_trade_defensive: float = 0.0015
    risk_per_trade_cpm: float = 0.0008
    risk_per_trade_lockdown: float = 0.0

    exposure_normal: float = 1.0
    exposure_defensive: float = 0.6
    exposure_cpm: float = 0.25
    exposure_lockdown: float = 0.0

    allow_pyramiding_normal: bool = True
    allow_pyramiding_defensive: bool = False
    allow_pyramiding_cpm: bool = False
    allow_pyramiding_lockdown: bool = False

    limit_only_cpm: bool = True
    cancel_open_orders_on_lockdown: bool = True

    # Data paths
    portfolio_history_csv: Optional[str] = "data/results/portfolio_history.csv"
    trade_log_csv: Optional[str] = "data/results/trade_log.csv"
    state_json_path: Optional[str] = "data/results/capital_preservation_state.json"

    # Override control file
    override_json_path: Optional[str] = "data/results/cpm_override.json"

    # Fail-closed behavior
    fail_closed: bool = True

    # NEW: if these risk signals are missing, degrade mode at least to this
    # (NORMAL, DEFENSIVE, CPM, LOCKDOWN)
    min_mode_when_signals_missing: str = "DEFENSIVE"


@dataclass
class CPMDecision:
    as_of: str
    mode: str
    cpi: float

    allow_new_trades: bool
    allow_increase: bool

    max_position_risk_pct: float
    exposure_multiplier: float

    limit_only: bool
    cancel_open_orders: bool

    equity: Optional[float]
    peak_equity: Optional[float]
    drawdown_pct: Optional[float]
    day_start_equity: Optional[float]
    daily_pnl_pct: Optional[float]
    vol_estimate: Optional[float]
    loss_streak: Optional[int]
    reasons: Dict[str, Any]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _read_csv_if_exists(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _infer_equity_from_portfolio_history(
    ph: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if ph is None or ph.empty:
        return None, None, None, None

    col_candidates = ["portfolio_value", "equity", "total_equity", "account_equity", "value"]
    date_candidates = ["date", "timestamp", "time", "as_of"]

    value_col = next((c for c in col_candidates if c in ph.columns), None)
    date_col = next((c for c in date_candidates if c in ph.columns), None)

    if value_col is None:
        return None, None, None, None

    df = ph.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df = df.sort_values(date_col)
    else:
        df = df.reset_index(drop=True)

    df = df.dropna(subset=[value_col])
    if df.empty:
        return None, None, None, None

    equity_latest = float(df[value_col].iloc[-1])
    peak_equity = float(df[value_col].max())

    drawdown_pct = None
    if peak_equity > 0:
        drawdown_pct = (peak_equity - equity_latest) / peak_equity

    day_start_equity = None
    if date_col:
        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()
        dfv = df.dropna(subset=[date_col])
        if not dfv.empty:
            same_day = dfv[dfv[date_col].dt.date == today]
            if not same_day.empty:
                day_start_equity = float(same_day[value_col].iloc[0])
            else:
                last_24h = dfv[dfv[date_col] >= (now_utc - pd.Timedelta(hours=24))]
                if not last_24h.empty:
                    day_start_equity = float(last_24h[value_col].iloc[0])

    return equity_latest, peak_equity, drawdown_pct, day_start_equity


def _estimate_volatility_from_portfolio_history(
    ph: pd.DataFrame, window_days: int
) -> Optional[float]:
    if ph is None or ph.empty:
        return None

    col_candidates = ["portfolio_value", "equity", "total_equity", "account_equity", "value"]
    date_candidates = ["date", "timestamp", "time", "as_of"]
    value_col = next((c for c in col_candidates if c in ph.columns), None)
    date_col = next((c for c in date_candidates if c in ph.columns), None)
    if value_col is None or date_col is None:
        return None

    df = ph.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if df.empty:
        return None

    df = df.set_index(date_col)
    daily = df[value_col].resample("1D").last().dropna()
    if len(daily) < 5:
        return None

    rets = daily.pct_change().dropna()
    if rets.empty:
        return None

    rets = rets.iloc[-window_days:]
    if len(rets) < 5:
        return None

    return float(rets.std())


def _compute_loss_streak_from_trade_log(tl: pd.DataFrame) -> Optional[int]:
    if tl is None or tl.empty:
        return None

    pnl_candidates = ["realized_pnl", "pnl", "profit", "profit_loss", "pl", "pnl_usd"]
    date_candidates = ["exit_time", "closed_at", "timestamp", "date", "time", "as_of"]

    pnl_col = next((c for c in pnl_candidates if c in tl.columns), None)
    if pnl_col is None:
        return None

    df = tl.copy()
    df[pnl_col] = pd.to_numeric(df[pnl_col], errors="coerce")

    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df = df.sort_values(date_col)

    df = df.dropna(subset=[pnl_col])
    if df.empty:
        return None

    streak = 0
    for v in reversed(df[pnl_col].tolist()):
        if v < 0:
            streak += 1
        else:
            break
    return streak


def _score_band(value: Optional[float], a: float, b: float, c: float) -> float:
    if value is None:
        return 0.35  # hardened: missing signal is slightly worse
    if value <= a:
        return 1.0
    if value <= b:
        return 0.7
    if value <= c:
        return 0.4
    return 0.1


def _mode_rank(mode: str) -> int:
    return {"NORMAL": 0, "DEFENSIVE": 1, "CPM": 2, "LOCKDOWN": 3}.get(mode.upper(), 2)


def _max_mode(a: str, b: str) -> str:
    return a if _mode_rank(a) >= _mode_rank(b) else b


def _mode_from_thresholds(
    drawdown: Optional[float],
    daily_loss: Optional[float],
    vol: Optional[float],
    loss_streak: Optional[int],
    cfg: CPMConfig,
) -> Tuple[str, Dict[str, Any]]:
    reasons: Dict[str, Any] = {}
    mode = "NORMAL"

    def escalate(new_mode: str, reason_key: str, reason_val: Any):
        nonlocal mode
        mode = _max_mode(mode, new_mode)
        reasons[reason_key] = reason_val

    # Drawdown
    if drawdown is not None:
        if drawdown >= cfg.dd_lockdown:
            escalate("LOCKDOWN", "drawdown_trigger", drawdown)
        elif drawdown >= cfg.dd_cpm:
            escalate("CPM", "drawdown_trigger", drawdown)
        elif drawdown >= cfg.dd_defensive:
            escalate("DEFENSIVE", "drawdown_trigger", drawdown)
    else:
        reasons["drawdown_missing"] = True

    # Daily loss
    if daily_loss is not None:
        if daily_loss >= cfg.daily_loss_lockdown:
            escalate("LOCKDOWN", "daily_loss_trigger", daily_loss)
        elif daily_loss >= cfg.daily_loss_cpm:
            escalate("CPM", "daily_loss_trigger", daily_loss)
        elif daily_loss >= cfg.daily_loss_defensive:
            escalate("DEFENSIVE", "daily_loss_trigger", daily_loss)
    else:
        reasons["daily_loss_missing"] = True

    # Volatility
    if vol is not None:
        if vol >= cfg.vol_lockdown:
            escalate("LOCKDOWN", "vol_trigger", vol)
        elif vol >= cfg.vol_cpm:
            escalate("CPM", "vol_trigger", vol)
        elif vol >= cfg.vol_defensive:
            escalate("DEFENSIVE", "vol_trigger", vol)
    else:
        reasons["vol_missing"] = True

    # Loss streak
    if loss_streak is not None:
        if loss_streak >= cfg.max_consecutive_losses_lockdown:
            escalate("LOCKDOWN", "loss_streak_trigger", loss_streak)
        elif loss_streak >= cfg.max_consecutive_losses_cpm:
            escalate("CPM", "loss_streak_trigger", loss_streak)
        elif loss_streak >= cfg.max_consecutive_losses_defensive:
            escalate("DEFENSIVE", "loss_streak_trigger", loss_streak)
    else:
        reasons["loss_streak_missing"] = True

    # HARDEN: if core signals are missing, enforce minimum mode
    core_missing = (drawdown is None) or (daily_loss is None) or (vol is None)
    if cfg.fail_closed and core_missing:
        mode = _max_mode(mode, cfg.min_mode_when_signals_missing)
        reasons["fail_closed_missing_core_signals"] = True
        reasons["min_mode_when_signals_missing"] = cfg.min_mode_when_signals_missing

    return mode, reasons


def _read_override(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None

        force_mode = (data.get("force_mode") or "").upper().strip()
        if force_mode not in ("NORMAL", "DEFENSIVE", "CPM", "LOCKDOWN"):
            return None

        # Optional expiry
        exp = data.get("expires_at")
        if exp:
            try:
                dt = pd.to_datetime(exp, utc=True)
                if dt is not pd.NaT:
                    now = pd.Timestamp.utcnow()
                    if now > dt:
                        return None
            except Exception:
                pass

        return data
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────


class CapitalPreservationEngine:
    def __init__(self, config: Optional[CPMConfig] = None):
        self.cfg = config or CPMConfig()

    def evaluate(
        self,
        *,
        equity: Optional[float] = None,
        broker: Optional[Any] = None,
        portfolio_history: Optional[pd.DataFrame] = None,
        trade_log: Optional[pd.DataFrame] = None,
    ) -> CPMDecision:
        cfg = self.cfg

        # Manual override (SUPREME)
        override = _read_override(cfg.override_json_path)
        override_mode = None
        override_reason = None
        if override:
            override_mode = (override.get("force_mode") or "").upper().strip()
            override_reason = override.get("reason") or "manual_override"

        ph = (
            portfolio_history
            if portfolio_history is not None
            else _read_csv_if_exists(cfg.portfolio_history_csv)
        )
        tl = trade_log if trade_log is not None else _read_csv_if_exists(cfg.trade_log_csv)

        equity_val = _safe_float(equity)

        if equity_val is None and broker is not None:
            try:
                acct = broker.get_account()
                equity_val = _safe_float(
                    acct.get("equity") or acct.get("portfolio_value") or acct.get("last_equity")
                )
            except Exception:
                equity_val = None

        eq_latest, peak_eq, dd_pct, day_start_eq = (
            _infer_equity_from_portfolio_history(ph) if ph is not None else (None, None, None, None)
        )
        if equity_val is None:
            equity_val = eq_latest

        peak_equity = peak_eq
        drawdown_pct = dd_pct

        daily_pnl_pct = None
        if equity_val is not None and day_start_eq is not None and day_start_eq > 0:
            pnl = (equity_val - day_start_eq) / day_start_eq
            daily_pnl_pct = -pnl if pnl < 0 else 0.0

        vol_est = (
            _estimate_volatility_from_portfolio_history(ph, cfg.vol_window_days)
            if ph is not None
            else None
        )
        loss_streak = _compute_loss_streak_from_trade_log(tl) if tl is not None else None

        # If equity missing entirely and fail_closed -> CPM
        reasons: Dict[str, Any] = {}
        mode, reasons = _mode_from_thresholds(
            drawdown=drawdown_pct,
            daily_loss=daily_pnl_pct,
            vol=vol_est,
            loss_streak=loss_streak,
            cfg=cfg,
        )

        if cfg.fail_closed and equity_val is None:
            mode = _max_mode(mode, "CPM")
            reasons["fail_closed_missing_equity"] = True

        # CPI computation
        dd_score = _score_band(drawdown_pct, cfg.dd_defensive, cfg.dd_cpm, cfg.dd_lockdown)
        dl_score = _score_band(
            daily_pnl_pct, cfg.daily_loss_defensive, cfg.daily_loss_cpm, cfg.daily_loss_lockdown
        )
        vol_score = _score_band(vol_est, cfg.vol_defensive, cfg.vol_cpm, cfg.vol_lockdown)

        if loss_streak is None:
            ls_score = 0.5
        else:
            if loss_streak <= cfg.max_consecutive_losses_defensive:
                ls_score = 0.9
            elif loss_streak <= cfg.max_consecutive_losses_cpm:
                ls_score = 0.6
            elif loss_streak <= cfg.max_consecutive_losses_lockdown:
                ls_score = 0.35
            else:
                ls_score = 0.15

        cpi = _clamp01(0.40 * dd_score + 0.25 * dl_score + 0.20 * vol_score + 0.15 * ls_score)

        # Apply override at the end (SUPREME)
        if override_mode:
            reasons["override_active"] = True
            reasons["override_mode"] = override_mode
            reasons["override_reason"] = override_reason
            mode = override_mode

        # Map mode -> outputs
        if mode == "NORMAL":
            allow_new = True
            allow_inc = cfg.allow_pyramiding_normal
            max_risk = cfg.risk_per_trade_normal
            expo = cfg.exposure_normal
            limit_only = False
            cancel_open = False
        elif mode == "DEFENSIVE":
            allow_new = True
            allow_inc = cfg.allow_pyramiding_defensive
            max_risk = cfg.risk_per_trade_defensive
            expo = cfg.exposure_defensive
            limit_only = False
            cancel_open = False
        elif mode == "CPM":
            allow_new = False
            allow_inc = cfg.allow_pyramiding_cpm
            max_risk = cfg.risk_per_trade_cpm
            expo = cfg.exposure_cpm
            limit_only = cfg.limit_only_cpm
            cancel_open = False
        else:  # LOCKDOWN
            allow_new = False
            allow_inc = cfg.allow_pyramiding_lockdown
            max_risk = cfg.risk_per_trade_lockdown
            expo = cfg.exposure_lockdown
            limit_only = True
            cancel_open = bool(cfg.cancel_open_orders_on_lockdown)

        decision = CPMDecision(
            as_of=_utc_now_iso(),
            mode=mode,
            cpi=float(cpi),
            allow_new_trades=bool(allow_new),
            allow_increase=bool(allow_inc),
            max_position_risk_pct=float(max_risk),
            exposure_multiplier=float(expo),
            limit_only=bool(limit_only),
            cancel_open_orders=bool(cancel_open),
            equity=equity_val,
            peak_equity=peak_equity,
            drawdown_pct=drawdown_pct,
            day_start_equity=day_start_eq,
            daily_pnl_pct=daily_pnl_pct,
            vol_estimate=vol_est,
            loss_streak=loss_streak,
            reasons=reasons,
        )

        self._persist_state(decision)
        return decision

    def _persist_state(self, decision: CPMDecision) -> None:
        path = self.cfg.state_json_path
        if not path:
            return
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(asdict(decision), indent=2), encoding="utf-8")
        except Exception:
            return


if __name__ == "__main__":
    # Example:
    #   python -m services.capital_preservation_engine
    try:
        from services.broker_alpaca import AlpacaBroker  # type: ignore

        b = AlpacaBroker(mode=os.getenv("CPM_MODE", "paper"))
    except Exception:
        b = None

    cfg = CPMConfig(initial_capital=_safe_float(os.getenv("TRITON_INITIAL_CAPITAL")))
    engine = CapitalPreservationEngine(cfg)
    d = engine.evaluate(broker=b)

    print(json.dumps(asdict(d), indent=2))
