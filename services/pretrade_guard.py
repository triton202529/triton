"""
pretrade_guard.py
Triton's global safety governor before placing ANY new buys.

Goal:
- Enforce Capital Preservation Doctrine.
- Refuse or downsize new risk if conditions are bad.

Modes:
    NORMAL     -> full sizing
    DEFENSIVE  -> cut sizing (half size for now)
    LOCKDOWN   -> block new buys

Signals considered (for now):
    - buying_power (after reserve)
    - min_cash_floor_usd (hard floor you never break)
    - est_portfolio_drawdown_pct (drawdown vs recent peak)

Environment overrides (optional):
    TRITON_GUARD_MAX_DD   : hard drawdown cap (default "0.15" = 15%)
    TRITON_GUARD_SOFT_DD  : soft drawdown cap (default "0.07" = 7%)
    TRITON_MIN_CASH_FLOOR : hard cash floor in USD (e.g. "500")
    TRITON_GUARD_OVERRIDE : force mode: NORMAL | DEFENSIVE | LOCKDOWN

Example (PowerShell):
    $env:TRITON_GUARD_MAX_DD   = "0.30"
    $env:TRITON_GUARD_SOFT_DD  = "0.10"
    $env:TRITON_MIN_CASH_FLOOR = "500"
    $env:TRITON_GUARD_OVERRIDE = "DEFENSIVE"
"""

from dataclasses import dataclass
import os


# ----------------------------
# Helpers & default config
# ----------------------------
def _env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name, "").strip()
        return float(v) if v != "" else default
    except Exception:
        return default


def _env_str(name: str) -> str | None:
    v = os.getenv(name)
    return v.strip().upper() if isinstance(v, str) and v.strip() else None


# Defaults (can be overridden by env)
_HARD_DD_DEFAULT = 0.15  # 15%
_SOFT_DD_DEFAULT = 0.07  # 7%

MAX_DRAWDOWN_HARD = _env_float("TRITON_GUARD_MAX_DD", _HARD_DD_DEFAULT)
MAX_DRAWDOWN_SOFT = _env_float("TRITON_GUARD_SOFT_DD", _SOFT_DD_DEFAULT)
FORCE_MODE = _env_str("TRITON_GUARD_OVERRIDE")  # NORMAL | DEFENSIVE | LOCKDOWN | None
ENV_MIN_CASH_FLOOR = os.getenv("TRITON_MIN_CASH_FLOOR")  # optional, numeric string


@dataclass
class GuardInputs:
    buying_power: float                 # current BP from broker
    reserve_pct: float                  # e.g. 0.05 means keep 5% untouched
    min_cash_floor_usd: float           # e.g. 1000 => never allow BP < $1k to be risked
    est_portfolio_drawdown_pct: float   # rough % drawdown, e.g. 0.12 = -12%


@dataclass
class GuardDecision:
    mode: str                # "NORMAL", "DEFENSIVE", "LOCKDOWN"
    scale_multiplier: float  # multiply all target notionals by this
    reason: str


def _forced_decision() -> GuardDecision | None:
    """
    If TRITON_GUARD_OVERRIDE is set to a valid mode, return a forced decision.
    """
    if FORCE_MODE in {"NORMAL", "DEFENSIVE", "LOCKDOWN"}:
        mult = 1.0 if FORCE_MODE == "NORMAL" else (0.5 if FORCE_MODE == "DEFENSIVE" else 0.0)
        return GuardDecision(
            mode=FORCE_MODE,
            scale_multiplier=mult,
            reason=f"forced via env TRITON_GUARD_OVERRIDE={FORCE_MODE}",
        )
    return None


def decide_guard(inputs: GuardInputs) -> GuardDecision:
    """
    Simple rule tree:

    0. Forced override via env (optional)
    1. Hard capital floor:
       If deployable BP after reserve < min_cash_floor_usd, LOCKDOWN.
    2. Big drawdown defense:
       If drawdown >= MAX_DRAWDOWN_HARD, LOCKDOWN.
       If drawdown >= MAX_DRAWDOWN_SOFT, DEFENSIVE.
    3. Otherwise NORMAL.
    """
    # (0) Forced override?
    forced = _forced_decision()
    if forced is not None:
        return forced

    # (1) Cash floor check (allow env override of the provided floor)
    bp_after_reserve = float(inputs.buying_power) * (1.0 - float(inputs.reserve_pct))

    min_floor = inputs.min_cash_floor_usd
    if ENV_MIN_CASH_FLOOR:
        try:
            env_floor_val = float(ENV_MIN_CASH_FLOOR)
            if env_floor_val > 0:
                min_floor = env_floor_val
        except Exception:
            pass

    if bp_after_reserve < float(min_floor):
        return GuardDecision(
            mode="LOCKDOWN",
            scale_multiplier=0.0,
            reason=f"bp_after_reserve ${bp_after_reserve:,.2f} < floor ${min_floor:,.2f}",
        )

    # (2) Drawdown logic (env-configurable thresholds)
    dd = float(inputs.est_portfolio_drawdown_pct or 0.0)

    if dd >= MAX_DRAWDOWN_HARD:
        return GuardDecision(
            mode="LOCKDOWN",
            scale_multiplier=0.0,
            reason=f"drawdown {dd:.1%} exceeds {MAX_DRAWDOWN_HARD:.0%} hard limit",
        )

    if dd >= MAX_DRAWDOWN_SOFT:
        return GuardDecision(
            mode="DEFENSIVE",
            scale_multiplier=0.5,
            reason=f"drawdown {dd:.1%} exceeds {MAX_DRAWDOWN_SOFT:.0%} soft limit",
        )

    # (3) Default healthy state
    return GuardDecision(
        mode="NORMAL",
        scale_multiplier=1.0,
        reason="within normal risk tolerances",
    )
