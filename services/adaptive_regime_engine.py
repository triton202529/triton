"""
Adaptive Regime Intelligence Engine — Step 10 (meta-policy layer).

Reads:
    data/results/investment_committee_summary.json
    data/results/performance_risk_overlay.csv
    data/results/opportunity_persistence_summary.json
    data/results/portfolio_construction_summary.json
    data/results/capital_deployment_summary.json
    data/results/signals_with_rationale.csv

Writes:
    data/results/adaptive_regime.json
    data/results/adaptive_policy.json

Purpose
-------
Steps 3-9 each answered an operational question. This engine answers
the meta-question:

    "What market regime are we in and how should Triton behave?"

It looks at the synthesised committee view (Step 9), the raw risk
overlay, the persistence funnel, the construction summary, the
deployment summary, and the live signal distribution; then classifies
the current operating regime (8 possible values) and emits a
*policy* — concrete numeric parameters that downstream engines
(construction, deployment, rebalance) can adopt to change Triton's
behaviour without code changes.

Field names in the policy are deliberately aligned with the
construction & deployment engines' existing constants (e.g.
``max_single_position_pct``, ``min_cash_reserve_pct``,
``min_deploy_confidence``) so the policy can be wired in as a single
JSON override without bespoke glue code.

Safety
------
* Read-only. No broker calls, no execution-state mutation.
* Missing inputs warn and continue — defaults degrade to NEUTRAL with a
  documented ``insufficient_data`` reason.
* Atomic writes (``.tmp`` + ``os.replace``).
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_COMMITTEE_JSON = RESULTS_DIR / "investment_committee_summary.json"
DEFAULT_RISK_OVERLAY_CSV = RESULTS_DIR / "performance_risk_overlay.csv"
DEFAULT_PERSISTENCE_JSON = RESULTS_DIR / "opportunity_persistence_summary.json"
DEFAULT_CONSTRUCTION_JSON = RESULTS_DIR / "portfolio_construction_summary.json"
DEFAULT_DEPLOYMENT_JSON = RESULTS_DIR / "capital_deployment_summary.json"
DEFAULT_SIGNALS_CSV = RESULTS_DIR / "signals_with_rationale.csv"

DEFAULT_OUT_REGIME = RESULTS_DIR / "adaptive_regime.json"
DEFAULT_OUT_POLICY = RESULTS_DIR / "adaptive_policy.json"

# -----------------------------------------------------------
# Regime taxonomy & priority
# -----------------------------------------------------------
# Priority order — first match wins. Special states (RISK_OFF,
# HIGH_VOLATILITY, ROTATION, MOMENTUM) can override the basic
# risk-axis classification when their evidence is strong.
REGIMES: Tuple[str, ...] = (
    "RISK_OFF",
    "DEFENSIVE",
    "HIGH_VOLATILITY",
    "ROTATION",
    "MOMENTUM",
    "AGGRESSIVE",
    "OPPORTUNISTIC",
    "NEUTRAL",
)

# Triggers for special / extreme regimes.
RISK_OFF_FORCE_EXIT = 4
RISK_OFF_BLOCK_NEW_BUY = 10
RISK_OFF_FLAG_RATIO = 0.50

DEFENSIVE_FORCE_EXIT = 1
DEFENSIVE_BLOCK_NEW_BUY = 5
DEFENSIVE_FLAG_RATIO = 0.30
DEFENSIVE_HEALTH_SCORE_MAX = 0.45

# HIGH_VOLATILITY thresholds (signal dispersion proxies).
HIGH_VOL_DELTA_STD = 0.012  # per-cycle delta std-dev
HIGH_VOL_CONFIDENCE_STD = 0.16  # cross-symbol confidence dispersion

# ROTATION: simultaneous selling AND buying activity.
ROTATION_MIN_EXECUTE_BUYS = 2
ROTATION_MIN_EXECUTE_SELLS = 1

# MOMENTUM: sustained strengthening.
MOMENTUM_PROMOTED_CONFIRMED = 3
MOMENTUM_AVG_PERSISTENCE = 0.65

# AGGRESSIVE: many buys, no blocks, clean risk.
AGGRESSIVE_MIN_EXECUTE_BUYS = 4

# OPPORTUNISTIC: at least one executable buy.
OPPORTUNISTIC_MIN_EXECUTE_BUYS = 1

# -----------------------------------------------------------
# Regime → Policy table
# -----------------------------------------------------------
# These map cleanly to the construction/deployment engines'
# existing constants (so they can be consumed as a JSON override).
REGIME_POLICY_TABLE: Dict[str, Dict[str, Any]] = {
    "RISK_OFF": {
        "max_position_pct": 3.0,
        "min_cash_reserve_pct": 30.0,
        "max_cash_reserve_pct": 50.0,
        "target_cash_pct": 35.0,
        "max_new_positions_per_cycle": 0,
        "max_sector_pct": 15.0,
        "max_cluster_pct": 20.0,
        "min_position_pct": 0.5,
        "deployment_threshold": 0.80,  # intent_score floor
        "confidence_threshold": 0.75,  # min_deploy_confidence
        "persistence_threshold": 0.85,  # deploy_persistence_floor
        "rebalance_frequency": "ON_RISK_CHANGE",
        "rotation_pressure": 0.85,  # strong push to recycle into safety
        "diversification_aggressiveness": 0.90,
        "risk_tolerance": 0.05,
        "block_strict_mode": True,
    },
    "DEFENSIVE": {
        "max_position_pct": 4.0,
        "min_cash_reserve_pct": 20.0,
        "max_cash_reserve_pct": 35.0,
        "target_cash_pct": 27.0,
        "max_new_positions_per_cycle": 1,
        "max_sector_pct": 18.0,
        "max_cluster_pct": 22.0,
        "min_position_pct": 0.5,
        "deployment_threshold": 0.70,
        "confidence_threshold": 0.65,
        "persistence_threshold": 0.75,
        "rebalance_frequency": "ON_RISK_CHANGE",
        "rotation_pressure": 0.70,
        "diversification_aggressiveness": 0.75,
        "risk_tolerance": 0.20,
        "block_strict_mode": True,
    },
    "HIGH_VOLATILITY": {
        "max_position_pct": 5.0,
        "min_cash_reserve_pct": 18.0,
        "max_cash_reserve_pct": 30.0,
        "target_cash_pct": 22.0,
        "max_new_positions_per_cycle": 2,
        "max_sector_pct": 20.0,
        "max_cluster_pct": 25.0,
        "min_position_pct": 0.5,
        "deployment_threshold": 0.65,
        "confidence_threshold": 0.60,
        "persistence_threshold": 0.70,
        "rebalance_frequency": "DAILY",
        "rotation_pressure": 0.55,
        "diversification_aggressiveness": 0.85,
        "risk_tolerance": 0.35,
        "block_strict_mode": True,
    },
    "ROTATION": {
        "max_position_pct": 6.0,
        "min_cash_reserve_pct": 12.0,
        "max_cash_reserve_pct": 22.0,
        "target_cash_pct": 17.0,
        "max_new_positions_per_cycle": 3,
        "max_sector_pct": 22.0,
        "max_cluster_pct": 28.0,
        "min_position_pct": 0.5,
        "deployment_threshold": 0.55,
        "confidence_threshold": 0.55,
        "persistence_threshold": 0.60,
        "rebalance_frequency": "DAILY",
        "rotation_pressure": 0.85,
        "diversification_aggressiveness": 0.70,
        "risk_tolerance": 0.55,
        "block_strict_mode": False,
    },
    "NEUTRAL": {
        "max_position_pct": 6.0,
        "min_cash_reserve_pct": 10.0,
        "max_cash_reserve_pct": 20.0,
        "target_cash_pct": 15.0,
        "max_new_positions_per_cycle": 3,
        "max_sector_pct": 25.0,
        "max_cluster_pct": 30.0,
        "min_position_pct": 0.5,
        "deployment_threshold": 0.55,
        "confidence_threshold": 0.55,
        "persistence_threshold": 0.60,
        "rebalance_frequency": "DAILY",
        "rotation_pressure": 0.50,
        "diversification_aggressiveness": 0.60,
        "risk_tolerance": 0.50,
        "block_strict_mode": False,
    },
    "MOMENTUM": {
        "max_position_pct": 7.0,
        "min_cash_reserve_pct": 8.0,
        "max_cash_reserve_pct": 15.0,
        "target_cash_pct": 12.0,
        "max_new_positions_per_cycle": 4,
        "max_sector_pct": 28.0,
        "max_cluster_pct": 32.0,
        "min_position_pct": 0.5,
        "deployment_threshold": 0.55,
        "confidence_threshold": 0.55,
        "persistence_threshold": 0.65,
        "rebalance_frequency": "DAILY",
        "rotation_pressure": 0.45,  # ride the winners
        "diversification_aggressiveness": 0.50,
        "risk_tolerance": 0.70,
        "block_strict_mode": False,
    },
    "OPPORTUNISTIC": {
        "max_position_pct": 7.0,
        "min_cash_reserve_pct": 8.0,
        "max_cash_reserve_pct": 15.0,
        "target_cash_pct": 12.0,
        "max_new_positions_per_cycle": 4,
        "max_sector_pct": 26.0,
        "max_cluster_pct": 30.0,
        "min_position_pct": 0.5,
        "deployment_threshold": 0.50,
        "confidence_threshold": 0.50,
        "persistence_threshold": 0.55,
        "rebalance_frequency": "DAILY",
        "rotation_pressure": 0.55,
        "diversification_aggressiveness": 0.65,
        "risk_tolerance": 0.65,
        "block_strict_mode": False,
    },
    "AGGRESSIVE": {
        "max_position_pct": 8.0,
        "min_cash_reserve_pct": 5.0,
        "max_cash_reserve_pct": 12.0,
        "target_cash_pct": 8.0,
        "max_new_positions_per_cycle": 5,
        "max_sector_pct": 30.0,
        "max_cluster_pct": 35.0,
        "min_position_pct": 0.5,
        "deployment_threshold": 0.50,
        "confidence_threshold": 0.50,
        "persistence_threshold": 0.50,
        "rebalance_frequency": "DAILY",
        "rotation_pressure": 0.40,
        "diversification_aggressiveness": 0.55,
        "risk_tolerance": 0.85,
        "block_strict_mode": False,
    },
}

# Risk tokens we recognise from the overlay.
RISK_FORCE_EXIT = "FORCE_EXIT"
RISK_TRIM_PRIORITY = "TRIM_PRIORITY"
RISK_BLOCK_NEW_BUY = "BLOCK_NEW_BUY"
RISK_OK = "OK"


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[REGIME_WARN] {msg}", flush=True)


def _safe_read_json(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        if not path.is_file():
            _warn(f"missing input: {label} ({path}); continuing without it")
            return {}
    except OSError as e:
        _warn(f"stat failed for {label} ({path}): {type(e).__name__}: {e}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        _warn(f"failed to parse {label} ({path}): {type(e).__name__}: {e}")
        return {}


def _safe_read_csv(path: Path, *, label: str) -> pd.DataFrame:
    try:
        if not path.is_file():
            _warn(f"missing input: {label} ({path}); continuing without it")
            return pd.DataFrame()
    except OSError as e:
        _warn(f"stat failed for {label} ({path}): {type(e).__name__}: {e}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception:
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip", keep_default_na=False)
        except Exception as e:
            _warn(f"failed to read {label} ({path}): {type(e).__name__}: {e}")
            return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_safe)
    os.replace(tmp, path)


def _json_safe(o: Any) -> Any:
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            return str(o)
    try:
        return float(o)
    except Exception:
        return str(o)


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# -----------------------------------------------------------
# Coercion
# -----------------------------------------------------------
def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _to_float_or_zero(x: Any) -> float:
    v = _to_float(x)
    return 0.0 if v is None else v


def _norm_upper(x: Any) -> str:
    return str(x or "").strip().upper()


def _norm_symbol(x: Any) -> str:
    return str(x or "").strip().upper()


def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _risk_components(risk_flag: str) -> List[str]:
    if not risk_flag:
        return []
    parts = [p.strip().upper() for p in str(risk_flag).split("|")]
    return [p for p in parts if p and p != RISK_OK]


# -----------------------------------------------------------
# Evidence extraction
# -----------------------------------------------------------
def _extract_risk_evidence(risk_df: pd.DataFrame) -> Dict[str, Any]:
    if risk_df is None or risk_df.empty:
        return {
            "n_symbols": 0,
            "n_force_exit": 0,
            "n_trim_priority": 0,
            "n_block_new_buy": 0,
            "n_flagged": 0,
            "flag_ratio": 0.0,
            "force_exit_symbols": [],
            "block_new_buy_symbols": [],
        }
    sym_col = _pick_first_present(risk_df, ("ticker", "symbol"))
    if not sym_col:
        return {
            "n_symbols": 0,
            "n_force_exit": 0,
            "n_trim_priority": 0,
            "n_block_new_buy": 0,
            "n_flagged": 0,
            "flag_ratio": 0.0,
            "force_exit_symbols": [],
            "block_new_buy_symbols": [],
        }
    n = 0
    fe: List[str] = []
    tp: List[str] = []
    bn: List[str] = []
    flagged = 0
    for _, r in risk_df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if not sym:
            continue
        n += 1
        comps = _risk_components(_norm_upper(r.get("risk_flag")))
        if comps:
            flagged += 1
        if RISK_FORCE_EXIT in comps:
            fe.append(sym)
        if RISK_TRIM_PRIORITY in comps:
            tp.append(sym)
        if RISK_BLOCK_NEW_BUY in comps:
            bn.append(sym)
    fe.sort()
    tp.sort()
    bn.sort()
    return {
        "n_symbols": n,
        "n_force_exit": len(fe),
        "n_trim_priority": len(tp),
        "n_block_new_buy": len(bn),
        "n_flagged": flagged,
        "flag_ratio": round(flagged / n, 6) if n else 0.0,
        "force_exit_symbols": fe,
        "block_new_buy_symbols": bn,
    }


def _extract_signal_evidence(signals_df: pd.DataFrame) -> Dict[str, Any]:
    """Volatility / dispersion proxies from live signals."""
    if signals_df is None or signals_df.empty:
        return {
            "n_signals": 0,
            "delta_std": None,
            "confidence_std": None,
            "abs_delta_mean": None,
            "signal_mix": {},
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
            "positive_signal_ratio": None,
            "directional_consensus": None,
        }
    sym_col = _pick_first_present(signals_df, ("ticker", "symbol"))
    delta_col = _pick_first_present(signals_df, ("delta_pct", "delta_pct_snapshot"))
    deltas: List[float] = []
    confs: List[float] = []
    mix: Counter = Counter()
    for _, r in signals_df.iterrows():
        if sym_col and not _norm_symbol(r.get(sym_col)):
            continue
        sig = _norm_upper(r.get("signal"))
        if sig:
            mix[sig] += 1
        c = _to_float(r.get("confidence"))
        if c is not None:
            confs.append(c)
        if delta_col:
            d = _to_float(r.get(delta_col))
            if d is not None:
                deltas.append(d)
    n = sum(mix.values())
    delta_std = float(statistics.pstdev(deltas)) if len(deltas) >= 2 else None
    conf_std = float(statistics.pstdev(confs)) if len(confs) >= 2 else None
    abs_delta_mean = float(sum(abs(d) for d in deltas) / len(deltas)) if deltas else None
    buy = int(mix.get("BUY", 0) + mix.get("ADD", 0))
    sell = int(mix.get("SELL", 0) + mix.get("EXIT", 0))
    hold = int(mix.get("HOLD", 0))
    positive_ratio = (buy / n) if n else None
    consensus = abs(buy - sell) / max(1, (buy + sell)) if (buy + sell) > 0 else None
    return {
        "n_signals": n,
        "delta_std": delta_std,
        "confidence_std": conf_std,
        "abs_delta_mean": abs_delta_mean,
        "signal_mix": dict(mix),
        "buy_count": buy,
        "sell_count": sell,
        "hold_count": hold,
        "positive_signal_ratio": positive_ratio,
        "directional_consensus": consensus,
    }


def _extract_committee_evidence(committee: Dict[str, Any]) -> Dict[str, Any]:
    scores = committee.get("scores") or {}
    counts = committee.get("intent_counts") or {}
    return {
        "market_posture": _norm_upper(committee.get("market_posture")) or "NEUTRAL",
        "deployment_stance": _norm_upper(committee.get("deployment_stance")) or "HOLD",
        "portfolio_conviction": _norm_upper(committee.get("portfolio_conviction")) or "WEAK",
        "portfolio_health_score": _to_float(scores.get("portfolio_health_score")),
        "deployment_readiness_score": _to_float(scores.get("deployment_readiness_score")),
        "conviction_score": _to_float(scores.get("conviction_score")),
        "diversification_score": _to_float(scores.get("diversification_score")),
        "governance_score": _to_float(scores.get("governance_score")),
        "execute_now": int(_to_float_or_zero(counts.get("execute_now"))),
        "delay": int(_to_float_or_zero(counts.get("delay"))),
        "block": int(_to_float_or_zero(counts.get("block"))),
        "skip": int(_to_float_or_zero(counts.get("skip"))),
        "execute_buys": int(_to_float_or_zero(committee.get("execute_buys"))),
        "execute_sells": int(_to_float_or_zero(committee.get("execute_sells"))),
        "cash_reserve_pct": _to_float(committee.get("cash_reserve_pct")),
        "deployable_capital_estimate": _to_float(committee.get("deployable_capital_estimate")),
        "construction_score": _to_float(committee.get("construction_score")),
        "concentration_warnings": list(committee.get("concentration_warnings") or []),
    }


def _extract_persistence_evidence(persistence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "watch_candidates": int(_to_float_or_zero(persistence.get("watch_candidates"))),
        "promoted_confirmed": int(_to_float_or_zero(persistence.get("promoted_confirmed"))),
        "kept_watch": int(_to_float_or_zero(persistence.get("kept_watch"))),
        "demoted": int(_to_float_or_zero(persistence.get("demoted"))),
        "rejected": int(_to_float_or_zero(persistence.get("rejected"))),
        "avg_persistence_score": _to_float(persistence.get("avg_persistence_score")),
    }


def _extract_construction_evidence(construction: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "construction_score": _to_float(construction.get("portfolio_construction_score")),
        "diversification_score": _to_float(construction.get("diversification_score")),
        "concentration_risk_pct": _to_float(construction.get("concentration_risk")),
        "cash_reserve_pct": _to_float(construction.get("cash_reserve_pct")),
        "target_cash_pct": _to_float(construction.get("target_cash_pct")),
        "total_portfolio_value": _to_float(construction.get("total_portfolio_value")),
        "current_positions": int(_to_float_or_zero(construction.get("current_positions"))),
        "max_positions": int(_to_float_or_zero(construction.get("max_positions"))),
    }


def _extract_deployment_evidence(deployment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "deploy_now_count": int(_to_float_or_zero(deployment.get("deploy_now_count"))),
        "shadow_watch_count": int(_to_float_or_zero(deployment.get("shadow_watch_count"))),
        "blocked_count": int(_to_float_or_zero(deployment.get("blocked_count"))),
        "deployable_capital_estimate": _to_float(deployment.get("deployable_capital_estimate")),
        "current_exposure_pct": _to_float(deployment.get("current_exposure_pct")),
    }


# -----------------------------------------------------------
# Regime classification
# -----------------------------------------------------------
def _classify_regime(
    *,
    risk_ev: Dict[str, Any],
    signal_ev: Dict[str, Any],
    committee_ev: Dict[str, Any],
    persistence_ev: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """
    Return (regime, list_of_reason_tags). Priority-ordered triggers:
        RISK_OFF > DEFENSIVE > HIGH_VOLATILITY > ROTATION > MOMENTUM
        > AGGRESSIVE > OPPORTUNISTIC > NEUTRAL
    """
    reasons: List[str] = []
    n_force = int(risk_ev["n_force_exit"])
    n_block = int(risk_ev["n_block_new_buy"])
    flag_ratio = float(risk_ev["flag_ratio"])
    health = committee_ev.get("portfolio_health_score") or 0.0
    delta_std = signal_ev.get("delta_std")
    conf_std = signal_ev.get("confidence_std")
    execute_buys = int(committee_ev["execute_buys"])
    execute_sells = int(committee_ev["execute_sells"])
    block_count = int(committee_ev["block"])
    promoted = int(persistence_ev["promoted_confirmed"])
    avg_pers = persistence_ev.get("avg_persistence_score") or 0.0

    # ── 1. RISK_OFF ──────────────────────────────────────────────
    if (
        n_force >= RISK_OFF_FORCE_EXIT
        or n_block >= RISK_OFF_BLOCK_NEW_BUY
        or flag_ratio >= RISK_OFF_FLAG_RATIO
    ):
        if n_force >= RISK_OFF_FORCE_EXIT:
            reasons.append(f"force_exit_count>={RISK_OFF_FORCE_EXIT} (got {n_force})")
        if n_block >= RISK_OFF_BLOCK_NEW_BUY:
            reasons.append(f"block_new_buy_count>={RISK_OFF_BLOCK_NEW_BUY} (got {n_block})")
        if flag_ratio >= RISK_OFF_FLAG_RATIO:
            reasons.append(f"risk_flag_ratio>={RISK_OFF_FLAG_RATIO} (got {flag_ratio:.2f})")
        return "RISK_OFF", reasons

    # ── 2. DEFENSIVE ─────────────────────────────────────────────
    defensive_hits: List[str] = []
    if n_force >= DEFENSIVE_FORCE_EXIT:
        defensive_hits.append(f"force_exit_count>={DEFENSIVE_FORCE_EXIT} (got {n_force})")
    if n_block >= DEFENSIVE_BLOCK_NEW_BUY:
        defensive_hits.append(f"block_new_buy_count>={DEFENSIVE_BLOCK_NEW_BUY} (got {n_block})")
    if flag_ratio >= DEFENSIVE_FLAG_RATIO:
        defensive_hits.append(f"risk_flag_ratio>={DEFENSIVE_FLAG_RATIO} (got {flag_ratio:.2f})")
    if health and health <= DEFENSIVE_HEALTH_SCORE_MAX:
        defensive_hits.append(
            f"portfolio_health_score<={DEFENSIVE_HEALTH_SCORE_MAX} (got {health:.2f})"
        )
    # Trigger DEFENSIVE only when multiple defensive signals agree, OR
    # when a single signal is already strong (force-exit or many blocks).
    if (
        len(defensive_hits) >= 2
        or n_force >= DEFENSIVE_FORCE_EXIT
        or n_block >= DEFENSIVE_BLOCK_NEW_BUY
    ):
        return "DEFENSIVE", defensive_hits

    # ── 3. HIGH_VOLATILITY ───────────────────────────────────────
    vol_hits: List[str] = []
    if delta_std is not None and delta_std >= HIGH_VOL_DELTA_STD:
        vol_hits.append(f"delta_std>={HIGH_VOL_DELTA_STD} (got {delta_std:.4f})")
    if conf_std is not None and conf_std >= HIGH_VOL_CONFIDENCE_STD:
        vol_hits.append(f"confidence_std>={HIGH_VOL_CONFIDENCE_STD} (got {conf_std:.3f})")
    if len(vol_hits) >= 2:
        return "HIGH_VOLATILITY", vol_hits

    # ── 4. ROTATION ──────────────────────────────────────────────
    if execute_buys >= ROTATION_MIN_EXECUTE_BUYS and execute_sells >= ROTATION_MIN_EXECUTE_SELLS:
        return "ROTATION", [
            f"execute_buys>={ROTATION_MIN_EXECUTE_BUYS} (got {execute_buys})",
            f"execute_sells>={ROTATION_MIN_EXECUTE_SELLS} (got {execute_sells})",
        ]

    # ── 5. MOMENTUM ──────────────────────────────────────────────
    if promoted >= MOMENTUM_PROMOTED_CONFIRMED and avg_pers >= MOMENTUM_AVG_PERSISTENCE:
        return "MOMENTUM", [
            f"promoted_confirmed>={MOMENTUM_PROMOTED_CONFIRMED} (got {promoted})",
            f"avg_persistence_score>={MOMENTUM_AVG_PERSISTENCE} (got {avg_pers:.2f})",
        ]

    # ── 6. AGGRESSIVE ────────────────────────────────────────────
    if (
        execute_buys >= AGGRESSIVE_MIN_EXECUTE_BUYS
        and block_count == 0
        and flag_ratio < DEFENSIVE_FLAG_RATIO
    ):
        return "AGGRESSIVE", [
            f"execute_buys>={AGGRESSIVE_MIN_EXECUTE_BUYS} (got {execute_buys})",
            "block_count=0",
            f"risk_flag_ratio<{DEFENSIVE_FLAG_RATIO} (got {flag_ratio:.2f})",
        ]

    # ── 7. OPPORTUNISTIC ─────────────────────────────────────────
    if execute_buys >= OPPORTUNISTIC_MIN_EXECUTE_BUYS:
        return "OPPORTUNISTIC", [
            f"execute_buys>={OPPORTUNISTIC_MIN_EXECUTE_BUYS} (got {execute_buys})",
        ]

    # ── 8. NEUTRAL (default) ─────────────────────────────────────
    return "NEUTRAL", ["no_strong_regime_signal"]


# -----------------------------------------------------------
# Policy derivation
# -----------------------------------------------------------
def _build_policy(regime: str) -> Dict[str, Any]:
    base = REGIME_POLICY_TABLE.get(regime) or REGIME_POLICY_TABLE["NEUTRAL"]
    return dict(base)


def _build_rationale(
    *,
    regime: str,
    reasons: List[str],
    risk_ev: Dict[str, Any],
    signal_ev: Dict[str, Any],
    committee_ev: Dict[str, Any],
    persistence_ev: Dict[str, Any],
    deployment_ev: Dict[str, Any],
) -> str:
    """Human-readable rationale for the regime classification."""
    parts: List[str] = []
    parts.append(
        f"Triton classified the environment as {regime} "
        f"(market posture {committee_ev['market_posture']}, "
        f"deployment stance {committee_ev['deployment_stance']}, "
        f"book conviction {committee_ev['portfolio_conviction']})."
    )
    if reasons:
        parts.append("Trigger evidence: " + "; ".join(reasons) + ".")

    diag_bits: List[str] = []
    if committee_ev.get("portfolio_health_score") is not None:
        diag_bits.append(f"health_score={committee_ev['portfolio_health_score']:.2f}")
    if committee_ev.get("deployment_readiness_score") is not None:
        diag_bits.append(f"readiness={committee_ev['deployment_readiness_score']:.2f}")
    if committee_ev.get("conviction_score") is not None:
        diag_bits.append(f"conviction={committee_ev['conviction_score']:.2f}")
    if committee_ev.get("diversification_score") is not None:
        diag_bits.append(f"diversification={committee_ev['diversification_score']:.2f}")
    if diag_bits:
        parts.append("Committee diagnostics: " + ", ".join(diag_bits) + ".")

    risk_bits: List[str] = [
        f"{risk_ev['n_force_exit']} FORCE_EXIT",
        f"{risk_ev['n_block_new_buy']} BLOCK_NEW_BUY",
        f"{risk_ev['n_flagged']}/{risk_ev['n_symbols']} symbols flagged",
    ]
    parts.append("Risk overlay: " + ", ".join(risk_bits) + ".")

    if signal_ev.get("delta_std") is not None or signal_ev.get("confidence_std") is not None:
        sig_bits: List[str] = []
        if signal_ev.get("delta_std") is not None:
            sig_bits.append(f"delta_std={signal_ev['delta_std']:.4f}")
        if signal_ev.get("confidence_std") is not None:
            sig_bits.append(f"confidence_std={signal_ev['confidence_std']:.3f}")
        if signal_ev.get("positive_signal_ratio") is not None:
            sig_bits.append(f"positive_signal_ratio={signal_ev['positive_signal_ratio']:.2f}")
        parts.append("Signal dispersion: " + ", ".join(sig_bits) + ".")

    if persistence_ev.get("watch_candidates"):
        parts.append(
            f"Persistence funnel: {persistence_ev['watch_candidates']} WATCH "
            f"({persistence_ev['promoted_confirmed']} promoted, "
            f"{persistence_ev['demoted']} demoted, "
            f"{persistence_ev['rejected']} rejected, "
            f"avg persistence "
            f"{persistence_ev.get('avg_persistence_score') or 0.0:.2f})."
        )

    if deployment_ev.get("deploy_now_count") or deployment_ev.get("blocked_count"):
        parts.append(
            f"Capital deployment: {deployment_ev['deploy_now_count']} DEPLOY_NOW, "
            f"{deployment_ev['shadow_watch_count']} SHADOW_WATCH, "
            f"{deployment_ev['blocked_count']} BLOCKED."
        )

    # Behaviour summary tied to the chosen regime.
    parts.append(_regime_behavior_blurb(regime))
    return " ".join(parts)


def _regime_behavior_blurb(regime: str) -> str:
    return {
        "RISK_OFF": (
            "Behavioural directive: full capital preservation -- halt new deploys, "
            "tighten position caps, raise cash to 30-50%, accept only the highest-"
            "conviction setups (confidence>=0.75, persistence>=0.85)."
        ),
        "DEFENSIVE": (
            "Behavioural directive: capital preservation dominates -- cash 20-35%, "
            "max position 4%, max 1 new position per cycle, confidence floor 0.65."
        ),
        "HIGH_VOLATILITY": (
            "Behavioural directive: trade smaller and more diversified to absorb "
            "dispersion -- cash 18-30%, max position 5%, daily rebalance cadence, "
            "block-strict mode on."
        ),
        "ROTATION": (
            "Behavioural directive: actively recycle capital from risk-flagged "
            "names into confirmed setups -- normal sizing, daily rebalance, "
            "rotation_pressure elevated."
        ),
        "MOMENTUM": (
            "Behavioural directive: ride confirmed strengthening -- larger positions "
            "(up to 7%), lower rotation pressure (let winners run), confidence "
            "floor relaxed to 0.55."
        ),
        "AGGRESSIVE": (
            "Behavioural directive: deploy widely -- cash 5-12%, max position 8%, "
            "up to 5 new positions per cycle, confidence floor 0.50."
        ),
        "OPPORTUNISTIC": (
            "Behavioural directive: selective deployment on confirmed setups -- "
            "cash 8-15%, max position 7%, up to 4 new positions per cycle."
        ),
        "NEUTRAL": (
            "Behavioural directive: hold standard discipline -- cash ~15%, max "
            "position 6%, 3 new positions per cycle, standard thresholds."
        ),
    }.get(regime, "")


# -----------------------------------------------------------
# Top-level builder
# -----------------------------------------------------------
def build_regime_report(
    *,
    committee_summary: Dict[str, Any],
    risk_df: pd.DataFrame,
    persistence_summary: Dict[str, Any],
    construction_summary: Dict[str, Any],
    deployment_summary: Dict[str, Any],
    signals_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pure planner -- returns (regime_report, policy_obj)."""
    risk_ev = _extract_risk_evidence(risk_df)
    signal_ev = _extract_signal_evidence(signals_df)
    committee_ev = _extract_committee_evidence(committee_summary)
    persistence_ev = _extract_persistence_evidence(persistence_summary)
    construction_ev = _extract_construction_evidence(construction_summary)
    deployment_ev = _extract_deployment_evidence(deployment_summary)

    # Composite presence check -- if every input is missing, snap to
    # NEUTRAL with an insufficient-data flag.
    have_any = any(
        [
            bool(committee_summary),
            bool(persistence_summary),
            bool(construction_summary),
            bool(deployment_summary),
            not risk_df.empty,
            not signals_df.empty,
        ]
    )
    if not have_any:
        regime = "NEUTRAL"
        reasons = ["insufficient_data: no inputs available, defaulting to NEUTRAL"]
    else:
        regime, reasons = _classify_regime(
            risk_ev=risk_ev,
            signal_ev=signal_ev,
            committee_ev=committee_ev,
            persistence_ev=persistence_ev,
        )

    rationale = _build_rationale(
        regime=regime,
        reasons=reasons,
        risk_ev=risk_ev,
        signal_ev=signal_ev,
        committee_ev=committee_ev,
        persistence_ev=persistence_ev,
        deployment_ev=deployment_ev,
    )

    policy = _build_policy(regime)

    # Add aliases that mirror downstream engine constants verbatim, so a
    # future override layer can splat the policy directly.
    policy_aliases = {
        "max_single_position_pct": policy["max_position_pct"],
        "min_deploy_confidence": policy["confidence_threshold"],
        "deploy_persistence_floor": policy["persistence_threshold"],
    }

    inputs_seen = {
        "committee_summary": bool(committee_summary),
        "persistence_summary": bool(persistence_summary),
        "construction_summary": bool(construction_summary),
        "deployment_summary": bool(deployment_summary),
        "risk_overlay_symbols": int(risk_ev["n_symbols"]),
        "signals_rows": int(signal_ev["n_signals"]),
    }

    regime_report: Dict[str, Any] = {
        "generated_at_utc": _now_iso_utc(),
        "engine": "adaptive_regime_engine",
        "engine_version": 1,
        "regime": regime,
        "regime_priority_order": list(REGIMES),
        "trigger_reasons": reasons,
        "rationale": rationale,
        "evidence": {
            "risk_overlay": risk_ev,
            "signals": signal_ev,
            "committee": committee_ev,
            "persistence": persistence_ev,
            "construction": construction_ev,
            "deployment": deployment_ev,
        },
        "inputs_seen": inputs_seen,
        "thresholds": {
            "risk_off_force_exit": RISK_OFF_FORCE_EXIT,
            "risk_off_block_new_buy": RISK_OFF_BLOCK_NEW_BUY,
            "risk_off_flag_ratio": RISK_OFF_FLAG_RATIO,
            "defensive_force_exit": DEFENSIVE_FORCE_EXIT,
            "defensive_block_new_buy": DEFENSIVE_BLOCK_NEW_BUY,
            "defensive_flag_ratio": DEFENSIVE_FLAG_RATIO,
            "defensive_health_score_max": DEFENSIVE_HEALTH_SCORE_MAX,
            "high_vol_delta_std": HIGH_VOL_DELTA_STD,
            "high_vol_confidence_std": HIGH_VOL_CONFIDENCE_STD,
            "rotation_min_execute_buys": ROTATION_MIN_EXECUTE_BUYS,
            "rotation_min_execute_sells": ROTATION_MIN_EXECUTE_SELLS,
            "momentum_promoted_confirmed": MOMENTUM_PROMOTED_CONFIRMED,
            "momentum_avg_persistence": MOMENTUM_AVG_PERSISTENCE,
            "aggressive_min_execute_buys": AGGRESSIVE_MIN_EXECUTE_BUYS,
            "opportunistic_min_execute_buys": OPPORTUNISTIC_MIN_EXECUTE_BUYS,
        },
    }

    policy_obj: Dict[str, Any] = {
        "generated_at_utc": regime_report["generated_at_utc"],
        "engine": "adaptive_regime_engine",
        "engine_version": 1,
        "regime": regime,
        "trigger_reasons": reasons,
        "rationale_short": _regime_behavior_blurb(regime),
        "policy": policy,
        "policy_aliases": policy_aliases,
        "inputs_seen": inputs_seen,
    }

    return regime_report, policy_obj


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only adaptive regime intelligence engine (Step 10). "
            "Classifies the current market regime and emits a dynamic policy "
            "that downstream engines (construction, deployment, rebalance) "
            "can consume as a JSON override."
        ),
    )
    p.add_argument("--committee", default=str(DEFAULT_COMMITTEE_JSON))
    p.add_argument("--risk-overlay", default=str(DEFAULT_RISK_OVERLAY_CSV))
    p.add_argument("--persistence", default=str(DEFAULT_PERSISTENCE_JSON))
    p.add_argument("--construction", default=str(DEFAULT_CONSTRUCTION_JSON))
    p.add_argument("--deployment", default=str(DEFAULT_DEPLOYMENT_JSON))
    p.add_argument("--signals", default=str(DEFAULT_SIGNALS_CSV))
    p.add_argument("--out-regime", default=str(DEFAULT_OUT_REGIME))
    p.add_argument("--out-policy", default=str(DEFAULT_OUT_POLICY))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[ADAPTIVE_REGIME] starting (read-only meta-policy engine)", flush=True)

    committee = _safe_read_json(Path(args.committee), label="investment_committee_summary.json")
    risk_df = _safe_read_csv(Path(args.risk_overlay), label="performance_risk_overlay.csv")
    persistence = _safe_read_json(
        Path(args.persistence), label="opportunity_persistence_summary.json"
    )
    construction = _safe_read_json(
        Path(args.construction), label="portfolio_construction_summary.json"
    )
    deployment = _safe_read_json(Path(args.deployment), label="capital_deployment_summary.json")
    signals_df = _safe_read_csv(Path(args.signals), label="signals_with_rationale.csv")

    regime_report, policy_obj = build_regime_report(
        committee_summary=committee,
        risk_df=risk_df,
        persistence_summary=persistence,
        construction_summary=construction,
        deployment_summary=deployment,
        signals_df=signals_df,
    )

    out_regime = Path(args.out_regime)
    out_policy = Path(args.out_policy)

    try:
        _atomic_write_json(regime_report, out_regime)
    except Exception as e:
        _warn(f"failed to write {out_regime}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(policy_obj, out_policy)
    except Exception as e:
        _warn(f"failed to write {out_policy}: {type(e).__name__}: {e}")
        return 2

    regime = regime_report["regime"]
    ev = regime_report["evidence"]
    p = policy_obj["policy"]
    print(
        "[ADAPTIVE_REGIME] "
        f"regime={regime} "
        f"deployment={ev['committee']['deployment_stance']} "
        f"conviction={ev['committee']['portfolio_conviction']} "
        f"risk_flag_ratio={ev['risk_overlay']['flag_ratio']:.2f}",
        flush=True,
    )
    print(
        "[REGIME_POLICY] "
        f"cash={p['target_cash_pct']:.1f}% "
        f"(range {p['min_cash_reserve_pct']:.1f}-{p['max_cash_reserve_pct']:.1f}%) "
        f"max_position={p['max_position_pct']:.1f}% "
        f"new_positions={p['max_new_positions_per_cycle']} "
        f"confidence>={p['confidence_threshold']:.2f} "
        f"persistence>={p['persistence_threshold']:.2f}",
        flush=True,
    )
    print(
        f"[ADAPTIVE_REGIME_OUT] regime={out_regime.as_posix()} " f"policy={out_policy.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
