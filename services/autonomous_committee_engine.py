"""
Autonomous Investment Committee Engine -- Step 15 (final decision layer).

Reads:
    data/results/investment_committee_report.json
    data/results/meta_decision_intelligence.json
    data/results/runtime_policy.json
    data/results/portfolio_execution_intents.csv
    data/results/portfolio_rebalance_summary.json
    data/results/adaptive_regime.json
    data/results/meta_policy_summary.json

Writes:
    data/results/autonomous_committee_decision.json
    data/results/autonomous_committee_report.md
    data/results/autonomous_committee_summary.json

Purpose
-------
Every prior layer is *advisory*. Step 15 is the capstone that says:

    "What should Triton actually recommend today?"

It collapses every upstream signal -- regime, meta-trust, portfolio
health, deployment readiness, executable opportunities, runtime
policy, governance score, risk overlay pressure -- into a single
categorical decision plus a 0-1 confidence score, then emits the
institutional CIO memo.

Decision states (spec section 1)
--------------------------------
    HOLD                    -- no actionable trades today
    DEPLOY_SELECTIVELY      -- execute a small number of high-conviction buys
    DEPLOY_AGGRESSIVELY     -- many EXECUTE_NOW buys + high self-trust + healthy book
    DELEVER                 -- risk pressure elevated, no executable buys
    DEFENSIVE_ROTATION      -- risk pressure elevated AND concrete sells ready
    CAPITAL_PRESERVATION    -- very low self-trust / collapsed health / RISK_OFF

The selector is a strict-precedence cascade: defensive states beat
neutral states beat deployment states, so risk-protective sells can
never be vetoed by buy-side enthusiasm. Confidence is independently
scored from input agreement.

Safety
------
* READ ONLY. No broker calls. No execution-state mutation. The
  written outputs are advisory artefacts only.
* Atomic writes (.tmp + os.replace) for every output.
* Missing inputs warn-and-continue; the engine always produces a
  defensible HOLD/CAPITAL_PRESERVATION recommendation when the
  upstream signal is thin.
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_COMMITTEE_JSON = RESULTS_DIR / "investment_committee_report.json"
DEFAULT_META_INTEL = RESULTS_DIR / "meta_decision_intelligence.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy.json"
DEFAULT_INTENTS_CSV = RESULTS_DIR / "portfolio_execution_intents.csv"
DEFAULT_REBALANCE_SUMMARY = RESULTS_DIR / "portfolio_rebalance_summary.json"
DEFAULT_REGIME_JSON = RESULTS_DIR / "adaptive_regime.json"
DEFAULT_META_POLICY_SUMMARY = RESULTS_DIR / "meta_policy_summary.json"

DEFAULT_OUT_DECISION = RESULTS_DIR / "autonomous_committee_decision.json"
DEFAULT_OUT_REPORT = RESULTS_DIR / "autonomous_committee_report.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "autonomous_committee_summary.json"

# -----------------------------------------------------------
# Decision states (spec section 1)
# -----------------------------------------------------------
DEC_HOLD = "HOLD"
DEC_DEPLOY_SELECTIVELY = "DEPLOY_SELECTIVELY"
DEC_DEPLOY_AGGRESSIVELY = "DEPLOY_AGGRESSIVELY"
DEC_DELEVER = "DELEVER"
DEC_DEFENSIVE_ROTATION = "DEFENSIVE_ROTATION"
DEC_CAPITAL_PRESERVATION = "CAPITAL_PRESERVATION"

ALL_DECISIONS: Tuple[str, ...] = (
    DEC_HOLD,
    DEC_DEPLOY_SELECTIVELY,
    DEC_DEPLOY_AGGRESSIVELY,
    DEC_DELEVER,
    DEC_DEFENSIVE_ROTATION,
    DEC_CAPITAL_PRESERVATION,
)

# Intent labels (mirror upstream).
INTENT_EXECUTE = "EXECUTE_NOW"
INTENT_DELAY = "DELAY"
INTENT_SKIP = "SKIP"
INTENT_BLOCK = "BLOCK"

BUY_ACTIONS = frozenset({"BUY_NEW", "ADD"})
SELL_ACTIONS = frozenset({"FULL_EXIT", "TRIM", "SELL"})

# Hard ceiling for the risk-overlay contribution (spec calls it
# "risk pressure"). Above this fraction-of-book the engine treats the
# overlay as the dominant signal.
RISK_PRESSURE_SEVERE = 0.50
RISK_PRESSURE_ELEVATED = 0.30

# Deployment-pressure bands.
DEPLOYMENT_AGGRESSIVE_FLOOR = 0.75
DEPLOYMENT_SELECTIVE_FLOOR = 0.50

# Defensive-pressure bands.
DEFENSIVE_CAPITAL_PRESERVATION_FLOOR = 0.70
DEFENSIVE_ROTATION_FLOOR = 0.55

# Self-trust labels needed to unlock the AGGRESSIVE state.
HIGH_TRUST_LABELS = frozenset({"HIGH", "VERY_HIGH"})

# Regimes that bias toward defensive posture.
RISK_REGIMES = frozenset({"RISK_OFF", "HIGH_VOLATILITY", "DEFENSIVE"})
BULL_REGIMES = frozenset({"OPPORTUNISTIC", "MOMENTUM", "AGGRESSIVE"})


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[AUTONOMOUS_COMMITTEE_WARN] {msg}", flush=True)


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


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
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
    if isinstance(x, bool):
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


def _to_float_or(x: Any, default: float) -> float:
    v = _to_float(x)
    return default if v is None else v


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm01(x: Any, default: float = 0.50) -> float:
    v = _to_float(x)
    if v is None:
        return default
    return _clamp(v, 0.0, 1.0)


def _norm_upper(x: Any) -> str:
    return str(x or "").strip().upper()


def _norm_symbol(x: Any) -> str:
    return str(x or "").strip().upper()


# -----------------------------------------------------------
# Evidence extraction
# -----------------------------------------------------------
def _extract_executable_breakdown(intents_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Bucket the execution intents into the four states plus buy/sell
    direction so the decision logic can reason about *what kind* of
    actions are pending, not just how many.
    """
    out: Dict[str, Any] = {
        "n_execute_now": 0,
        "n_delay": 0,
        "n_skip": 0,
        "n_block": 0,
        "n_execute_buys": 0,
        "n_execute_sells": 0,
        "n_delay_buys": 0,
        "n_delay_sells": 0,
        "n_block_buys": 0,
        "n_block_sells": 0,
        "execute_buy_tickers": [],
        "execute_sell_tickers": [],
        "delay_tickers": [],
        "block_tickers": [],
        "avg_execute_intent_score": 0.0,
        "max_execute_intent_score": 0.0,
        "total_notional_usd": 0.0,
    }
    if intents_df is None or intents_df.empty:
        return out
    sym_col = (
        "ticker"
        if "ticker" in intents_df.columns
        else ("symbol" if "symbol" in intents_df.columns else None)
    )
    if not sym_col:
        return out
    execute_scores: List[float] = []
    for _, r in intents_df.iterrows():
        intent = _norm_upper(r.get("execution_intent"))
        action = _norm_upper(r.get("rebalance_action"))
        sym = _norm_symbol(r.get(sym_col))
        amount = _to_float_or(r.get("rebalance_amount_usd"), 0.0)
        score = _to_float_or(r.get("intent_score"), 0.0)
        is_buy = action in BUY_ACTIONS
        is_sell = action in SELL_ACTIONS
        if intent == INTENT_EXECUTE:
            out["n_execute_now"] += 1
            out["total_notional_usd"] += abs(amount)
            execute_scores.append(score)
            if is_buy:
                out["n_execute_buys"] += 1
                out["execute_buy_tickers"].append(sym)
            elif is_sell:
                out["n_execute_sells"] += 1
                out["execute_sell_tickers"].append(sym)
        elif intent == INTENT_DELAY:
            out["n_delay"] += 1
            out["delay_tickers"].append(sym)
            if is_buy:
                out["n_delay_buys"] += 1
            elif is_sell:
                out["n_delay_sells"] += 1
        elif intent == INTENT_BLOCK:
            out["n_block"] += 1
            out["block_tickers"].append(sym)
            if is_buy:
                out["n_block_buys"] += 1
            elif is_sell:
                out["n_block_sells"] += 1
        elif intent == INTENT_SKIP:
            out["n_skip"] += 1
    if execute_scores:
        out["avg_execute_intent_score"] = sum(execute_scores) / len(execute_scores)
        out["max_execute_intent_score"] = max(execute_scores)
    return out


def _extract_risk_pressure(committee: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise the risk overlay signal into a 0-1 pressure scalar plus
    the underlying counts.
    """
    risk = (committee or {}).get("risk_notes") or {}
    n_force = len(risk.get("force_exit_symbols") or [])
    n_block = len(risk.get("block_new_buy_symbols") or [])
    n_trim = len(risk.get("trim_priority_symbols") or [])
    n_eval = int(risk.get("n_symbols_evaluated") or 0)
    n_concentration = len(risk.get("concentration_warnings") or [])
    # Weight force-exits most heavily, then block-new-buy, then trims.
    weighted = (n_force * 3.0) + (n_block * 1.5) + (n_trim * 0.5)
    denom = max(1, n_eval if n_eval > 0 else (n_force + n_block + n_trim))
    pressure = _clamp(weighted / max(1.0, denom * 1.0), 0.0, 1.0)
    return {
        "risk_pressure_score": round(pressure, 4),
        "n_force_exit": n_force,
        "n_block_new_buy": n_block,
        "n_trim_priority": n_trim,
        "n_symbols_evaluated": n_eval,
        "n_concentration_warnings": n_concentration,
        "risk_clean_ratio": _to_float_or(risk.get("risk_clean_ratio"), 0.0),
    }


def _extract_committee_scores(committee: Dict[str, Any]) -> Dict[str, float]:
    sc = (committee.get("executive_summary") or {}).get("scores") or {}
    return {
        "portfolio_health_score": _norm01(sc.get("portfolio_health_score")),
        "deployment_readiness_score": _norm01(sc.get("deployment_readiness_score")),
        "conviction_score": _norm01(sc.get("conviction_score")),
        "diversification_score": _norm01(sc.get("diversification_score")),
        "governance_score": _norm01(sc.get("governance_score")),
    }


def _extract_meta(meta_intel: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trust_level": _norm_upper(meta_intel.get("trust_level") or "MODERATE"),
        "self_confidence_score": _norm01(meta_intel.get("self_confidence_score")),
        "scores": meta_intel.get("scores") or {},
        "rationale_short": str(meta_intel.get("rationale_short") or ""),
    }


def _extract_policy(runtime_policy: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "regime": _norm_upper(runtime_policy.get("regime") or "UNKNOWN"),
        "target_cash_pct": _to_float_or(runtime_policy.get("target_cash_pct"), 15.0),
        "max_position_pct": _to_float_or(runtime_policy.get("max_position_pct"), 6.0),
        "max_new_positions_per_cycle": int(runtime_policy.get("max_new_positions_per_cycle") or 0),
        "confidence_threshold": _to_float_or(runtime_policy.get("confidence_threshold"), 0.55),
        "deployment_threshold": _to_float_or(runtime_policy.get("deployment_threshold"), 0.55),
        "engine": str(runtime_policy.get("engine") or "unknown"),
        "meta_overlay_applied": (
            str(runtime_policy.get("engine") or "") == "meta_policy_injection_engine"
        ),
    }


def _extract_regime(regime_json: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "regime": _norm_upper(regime_json.get("regime") or "UNKNOWN"),
        "rationale": str(regime_json.get("rationale") or ""),
    }


def _extract_turnover(rebalance_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "portfolio_turnover_pct": _to_float_or(
            rebalance_summary.get("portfolio_turnover_pct"), 0.0
        ),
        "estimated_capital_deployed": _to_float_or(
            rebalance_summary.get("estimated_capital_deployed"), 0.0
        ),
        "estimated_capital_freed": _to_float_or(
            rebalance_summary.get("estimated_capital_freed"), 0.0
        ),
        "total_rebalance_actions": int(rebalance_summary.get("total_rebalance_actions") or 0),
    }


# -----------------------------------------------------------
# Pressure scoring
# -----------------------------------------------------------
def _deployment_pressure(
    *,
    breakdown: Dict[str, Any],
    committee_scores: Dict[str, float],
    meta: Dict[str, Any],
    risk: Dict[str, Any],
    policy: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    How strongly should Triton DEPLOY today? Returns (score, components).

    Higher when:
        + many EXECUTE_NOW buys ready
        + high deployment readiness
        + high self-confidence
        + healthy portfolio
        + low risk pressure
        + bull regime
    """
    n_buys = int(breakdown.get("n_execute_buys") or 0)
    avg_score = float(breakdown.get("avg_execute_intent_score") or 0.0)
    # Normalise buy count: 1 buy = 0.30, 3 buys = 0.70, 5+ buys = 1.0
    buy_count_score = _clamp(0.30 + 0.20 * (n_buys - 1), 0.0, 1.0) if n_buys else 0.0
    components = {
        "executable_buys": round(buy_count_score, 4),
        "avg_intent_score": round(avg_score, 4),
        "deployment_readiness": round(committee_scores["deployment_readiness_score"], 4),
        "self_confidence": round(meta["self_confidence_score"], 4),
        "portfolio_health": round(committee_scores["portfolio_health_score"], 4),
        "risk_clearance": round(1.0 - risk["risk_pressure_score"], 4),
        "regime_bias": (
            0.75
            if policy["regime"] in BULL_REGIMES
            else (0.25 if policy["regime"] in RISK_REGIMES else 0.50)
        ),
    }
    weights = {
        "executable_buys": 0.25,
        "avg_intent_score": 0.15,
        "deployment_readiness": 0.15,
        "self_confidence": 0.15,
        "portfolio_health": 0.10,
        "risk_clearance": 0.10,
        "regime_bias": 0.10,
    }
    score = sum(weights[k] * components[k] for k in weights)
    return round(_clamp(score, 0.0, 1.0), 4), components


def _defensive_pressure(
    *,
    breakdown: Dict[str, Any],
    committee_scores: Dict[str, float],
    meta: Dict[str, Any],
    risk: Dict[str, Any],
    policy: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    How strongly should Triton GO DEFENSIVE today? Returns (score, components).

    Higher when:
        + risk overlay is firing on many symbols
        + portfolio health is weak
        + self-confidence is weak
        + governance score is weak
        + risk regime (DEFENSIVE / RISK_OFF / HIGH_VOL)
        + concentration warnings present
    """
    health = committee_scores["portfolio_health_score"]
    trust = meta["self_confidence_score"]
    gov = committee_scores["governance_score"]
    has_concentration = bool(risk["n_concentration_warnings"] > 0)
    components = {
        "risk_pressure": round(risk["risk_pressure_score"], 4),
        "health_weakness": round(1.0 - health, 4),
        "self_doubt": round(1.0 - trust, 4),
        "governance_weakness": round(1.0 - gov, 4),
        "concentration_flag": 1.0 if has_concentration else 0.0,
        "regime_bias": (
            0.85
            if policy["regime"] == "RISK_OFF"
            else (
                0.75
                if policy["regime"] == "HIGH_VOLATILITY"
                else (
                    0.60
                    if policy["regime"] == "DEFENSIVE"
                    else (0.25 if policy["regime"] in BULL_REGIMES else 0.50)
                )
            )
        ),
    }
    weights = {
        "risk_pressure": 0.25,
        "health_weakness": 0.20,
        "self_doubt": 0.15,
        "governance_weakness": 0.10,
        "concentration_flag": 0.10,
        "regime_bias": 0.20,
    }
    score = sum(weights[k] * components[k] for k in weights)
    return round(_clamp(score, 0.0, 1.0), 4), components


# -----------------------------------------------------------
# Decision selector
# -----------------------------------------------------------
def _select_decision(
    *,
    deployment_pressure: float,
    defensive_pressure: float,
    breakdown: Dict[str, Any],
    meta: Dict[str, Any],
    risk: Dict[str, Any],
) -> Tuple[str, List[str], str]:
    """
    Strict-precedence selector. Defensive states win over deployment
    states because risk-protective sells must always be allowed to
    fire. Returns (decision, rationale_tags, primary_reason).
    """
    tags: List[str] = []
    n_buys = int(breakdown["n_execute_buys"])
    n_sells = int(breakdown["n_execute_sells"])
    trust = meta["trust_level"]
    self_conf = float(meta["self_confidence_score"])

    # --- Defensive cascade ---
    if (
        defensive_pressure >= DEFENSIVE_CAPITAL_PRESERVATION_FLOOR
        or self_conf <= 0.20
        or risk["risk_pressure_score"] >= RISK_PRESSURE_SEVERE
    ):
        tags.append("capital_preservation_trigger")
        if defensive_pressure >= DEFENSIVE_CAPITAL_PRESERVATION_FLOOR:
            tags.append("defensive_pressure_severe")
        if self_conf <= 0.20:
            tags.append("self_trust_collapsed")
        if risk["risk_pressure_score"] >= RISK_PRESSURE_SEVERE:
            tags.append("risk_overlay_severe")
        reason = (
            "Capital preservation triggered by severe defensive pressure "
            f"({defensive_pressure:.2f}), risk overlay "
            f"({risk['risk_pressure_score']:.2f}), or collapsed self-trust "
            f"({self_conf:.2f})."
        )
        return DEC_CAPITAL_PRESERVATION, tags, reason

    if defensive_pressure >= DEFENSIVE_ROTATION_FLOOR and n_sells > 0:
        tags.extend(["defensive_pressure_elevated", "sells_ready", "rotation_path"])
        reason = (
            f"Defensive rotation: defensive pressure {defensive_pressure:.2f} "
            f"with {n_sells} sell action(s) ready to execute."
        )
        return DEC_DEFENSIVE_ROTATION, tags, reason

    if risk["risk_pressure_score"] >= RISK_PRESSURE_ELEVATED and n_buys == 0:
        tags.extend(["risk_pressure_elevated", "no_buys_ready"])
        reason = (
            f"Delever: risk overlay pressure {risk['risk_pressure_score']:.2f} "
            "elevated with no executable buys; tighten exposure where possible."
        )
        return DEC_DELEVER, tags, reason

    # --- Deployment cascade ---
    if (
        deployment_pressure >= DEPLOYMENT_AGGRESSIVE_FLOOR
        and n_buys >= 3
        and trust in HIGH_TRUST_LABELS
        and risk["risk_pressure_score"] < RISK_PRESSURE_ELEVATED
    ):
        tags.extend(
            [
                "deployment_pressure_high",
                "many_buys_ready",
                "trust_high",
                "risk_clear",
            ]
        )
        reason = (
            f"Aggressive deployment: deployment pressure {deployment_pressure:.2f}, "
            f"{n_buys} buys ready, self-trust {trust}, risk overlay clear."
        )
        return DEC_DEPLOY_AGGRESSIVELY, tags, reason

    if deployment_pressure >= DEPLOYMENT_SELECTIVE_FLOOR and n_buys >= 1:
        tags.extend(["deployment_pressure_moderate", "buys_ready"])
        reason = (
            f"Selective deployment: deployment pressure {deployment_pressure:.2f} "
            f"with {n_buys} high-conviction buy(s) ready."
        )
        return DEC_DEPLOY_SELECTIVELY, tags, reason

    # --- Neutral fallback ---
    if n_sells > 0 and n_buys == 0:
        tags.extend(["sells_only", "hold_with_exits"])
        reason = (
            f"Hold posture with {n_sells} risk-protective sell(s) authorised; "
            "no actionable buys today."
        )
        return DEC_HOLD, tags, reason
    if n_buys == 0 and n_sells == 0:
        tags.append("no_actionable_trades")
    else:
        tags.append("mixed_signals_neutral")
    reason = (
        "Hold: deployment pressure "
        f"({deployment_pressure:.2f}) below selective floor "
        f"({DEPLOYMENT_SELECTIVE_FLOOR:.2f}) and defensive pressure "
        f"({defensive_pressure:.2f}) below rotation floor "
        f"({DEFENSIVE_ROTATION_FLOOR:.2f})."
    )
    return DEC_HOLD, tags, reason


def _recommendation_confidence(
    *,
    decision: str,
    deployment_pressure: float,
    defensive_pressure: float,
    meta: Dict[str, Any],
    breakdown: Dict[str, Any],
) -> float:
    """
    Confidence in the recommendation itself (not in the underlying signals).

    Higher when:
        * The selected decision is well separated from the runner-up
          (one pressure clearly dominates).
        * Self-trust is high (the input signals are trustworthy).
        * For deployment decisions, the average intent score on the
          executable buys is high.
    """
    self_conf = float(meta["self_confidence_score"])
    separation = abs(deployment_pressure - defensive_pressure)
    avg_intent = float(breakdown.get("avg_execute_intent_score") or 0.0)
    n_exec = int(breakdown.get("n_execute_now") or 0)

    # Base = separation * self_trust (both 0-1).
    base = 0.5 * (separation + self_conf)

    # For deployment decisions, weight average intent score and breadth.
    if decision in (DEC_DEPLOY_SELECTIVELY, DEC_DEPLOY_AGGRESSIVELY):
        breadth = _clamp(0.30 + 0.20 * (n_exec - 1), 0.0, 1.0) if n_exec else 0.0
        base = 0.4 * base + 0.4 * avg_intent + 0.2 * breadth

    # For risk-defensive decisions, weight magnitude of defensive pressure.
    if decision in (DEC_DEFENSIVE_ROTATION, DEC_CAPITAL_PRESERVATION, DEC_DELEVER):
        base = 0.5 * base + 0.5 * defensive_pressure

    # HOLD with no actionable trades is high-confidence when self-trust is solid.
    if decision == DEC_HOLD and n_exec == 0:
        base = 0.5 * (base + self_conf)

    return round(_clamp(base, 0.0, 1.0), 4)


# -----------------------------------------------------------
# Action / report builders
# -----------------------------------------------------------
def _build_action_groups(
    intents_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build (recommended, delayed, blocked) lists from execution intents.
    Each list element is a small dict suitable for both JSON and markdown.
    """
    if intents_df is None or intents_df.empty:
        return [], [], []
    sym_col = (
        "ticker"
        if "ticker" in intents_df.columns
        else ("symbol" if "symbol" in intents_df.columns else None)
    )
    if not sym_col:
        return [], [], []
    recommended: List[Dict[str, Any]] = []
    delayed: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    for _, r in intents_df.iterrows():
        intent = _norm_upper(r.get("execution_intent"))
        row = {
            "ticker": _norm_symbol(r.get(sym_col)),
            "rebalance_action": _norm_upper(r.get("rebalance_action")),
            "rebalance_amount_usd": round(_to_float_or(r.get("rebalance_amount_usd"), 0.0), 2),
            "priority": round(_to_float_or(r.get("priority"), 0.0), 4),
            "intent_score": round(_to_float_or(r.get("intent_score"), 0.0), 4),
            "reason": str(r.get("reason") or "").strip(),
        }
        if intent == INTENT_EXECUTE:
            recommended.append(row)
        elif intent == INTENT_DELAY:
            delayed.append(row)
        elif intent == INTENT_BLOCK:
            blocked.append(row)
    recommended.sort(key=lambda d: d["intent_score"], reverse=True)
    delayed.sort(key=lambda d: d["intent_score"], reverse=True)
    blocked.sort(key=lambda d: d["intent_score"], reverse=True)
    return recommended, delayed, blocked


def _build_risk_notes_list(committee: Dict[str, Any]) -> List[Dict[str, Any]]:
    risk = (committee or {}).get("risk_notes") or {}
    out: List[Dict[str, Any]] = []
    force = risk.get("force_exit_symbols") or []
    if force:
        out.append(
            {
                "kind": "FORCE_EXIT",
                "symbols": list(force),
                "count": len(force),
                "summary": f"{len(force)} symbol(s) flagged for FORCE_EXIT.",
            }
        )
    block = risk.get("block_new_buy_symbols") or []
    if block:
        out.append(
            {
                "kind": "BLOCK_NEW_BUY",
                "symbols": list(block),
                "count": len(block),
                "summary": f"{len(block)} symbol(s) blocked from new buys.",
            }
        )
    trim = risk.get("trim_priority_symbols") or []
    if trim:
        out.append(
            {
                "kind": "TRIM_PRIORITY",
                "symbols": list(trim),
                "count": len(trim),
                "summary": f"{len(trim)} symbol(s) on trim priority watchlist.",
            }
        )
    for cw in risk.get("concentration_warnings") or []:
        kind = cw.get("type", "concentration")
        name = cw.get("name", "?")
        weight = _to_float_or(cw.get("weight_pct"), 0.0)
        severity = cw.get("severity", "WARN")
        out.append(
            {
                "kind": "CONCENTRATION",
                "symbols": [name],
                "count": 1,
                "summary": (
                    f"{severity}: {kind.title()} concentration in {name} " f"at {weight:.1f}%."
                ),
            }
        )
    contradictions = risk.get("lifecycle_contradictions") or []
    if contradictions:
        out.append(
            {
                "kind": "LIFECYCLE_CONTRADICTION",
                "symbols": [
                    c.get("ticker", "?") if isinstance(c, dict) else str(c) for c in contradictions
                ],
                "count": len(contradictions),
                "summary": (
                    f"{len(contradictions)} lifecycle/signal contradiction(s) " "to reconcile."
                ),
            }
        )
    return out


def _build_cio_recommendation(
    *,
    decision: str,
    decision_reason: str,
    confidence: float,
    deployment_pressure: float,
    defensive_pressure: float,
    breakdown: Dict[str, Any],
    meta: Dict[str, Any],
    policy: Dict[str, Any],
    risk: Dict[str, Any],
    scores: Dict[str, float],
    turnover: Dict[str, Any],
) -> str:
    posture = {
        DEC_HOLD: "stand pat and preserve optionality",
        DEC_DEPLOY_SELECTIVELY: "deploy selectively into the highest-conviction setups",
        DEC_DEPLOY_AGGRESSIVELY: "deploy aggressively across the full opportunity slate",
        DEC_DELEVER: "delever and free capital ahead of better setups",
        DEC_DEFENSIVE_ROTATION: "rotate defensively, taking the authorised sells now",
        DEC_CAPITAL_PRESERVATION: "preserve capital and stand down on new risk",
    }.get(decision, "stand pat")

    bits: List[str] = []
    bits.append(f"Triton recommends {decision} (confidence {confidence:.2f}) -- " f"{posture}.")
    bits.append(
        f"The regime is {policy['regime']}; meta self-trust is "
        f"{meta['trust_level']} (self_confidence {meta['self_confidence_score']:.2f}); "
        f"portfolio health is {scores['portfolio_health_score']:.2f} and "
        f"governance is {scores['governance_score']:.2f}."
    )
    bits.append(
        f"Deployment pressure scored {deployment_pressure:.2f} against "
        f"defensive pressure {defensive_pressure:.2f}; "
        f"{breakdown['n_execute_buys']} buy(s) and "
        f"{breakdown['n_execute_sells']} sell(s) are EXECUTE_NOW, "
        f"{breakdown['n_delay']} delayed, {breakdown['n_block']} blocked."
    )
    if risk["n_force_exit"] or risk["n_block_new_buy"] or risk["n_concentration_warnings"]:
        bits.append(
            f"Risk overlay pressure is {risk['risk_pressure_score']:.2f} "
            f"({risk['n_force_exit']} force-exit, "
            f"{risk['n_block_new_buy']} block-new-buy, "
            f"{risk['n_concentration_warnings']} concentration warning(s))."
        )
    if policy["meta_overlay_applied"]:
        bits.append(
            f"Active runtime policy is the meta-overlaid one "
            f"(confidence>={policy['confidence_threshold']:.2f}, "
            f"target_cash={policy['target_cash_pct']:.1f}%, "
            f"max_position={policy['max_position_pct']:.1f}%)."
        )
    else:
        bits.append(
            f"Active runtime policy is the regime baseline "
            f"(confidence>={policy['confidence_threshold']:.2f}, "
            f"target_cash={policy['target_cash_pct']:.1f}%, "
            f"max_position={policy['max_position_pct']:.1f}%)."
        )
    if turnover["portfolio_turnover_pct"] > 0.0:
        bits.append(
            f"Planned turnover is {turnover['portfolio_turnover_pct']:.1f}% "
            f"({turnover['estimated_capital_deployed']:.0f} USD deployed, "
            f"{turnover['estimated_capital_freed']:.0f} USD freed)."
        )
    bits.append(f"Decision rationale: {decision_reason}")
    return " ".join(bits)


def _render_markdown(
    *,
    decision: str,
    decision_reason: str,
    confidence: float,
    why_bullets: List[str],
    recommended: List[Dict[str, Any]],
    delayed: List[Dict[str, Any]],
    blocked: List[Dict[str, Any]],
    risk_notes: List[Dict[str, Any]],
    cio_recommendation: str,
    generated_at: str,
    regime: str,
    trust_level: str,
    self_confidence: float,
) -> str:
    lines: List[str] = []
    lines.append("# Triton Autonomous Committee Decision")
    lines.append("")
    lines.append(
        f"*Generated {generated_at} | Regime: {regime} | "
        f"Self-trust: {trust_level} ({self_confidence:.2f})*"
    )
    lines.append("")
    lines.append("## Final Decision")
    lines.append("")
    lines.append(f"**{decision}** (confidence {confidence:.2f})")
    lines.append("")
    lines.append(f"> {decision_reason}")
    lines.append("")
    lines.append("## Why")
    lines.append("")
    for b in why_bullets:
        lines.append(f"- {b}")
    if not why_bullets:
        lines.append("- No additional drivers beyond the headline pressures.")
    lines.append("")
    lines.append("## Recommended Actions")
    lines.append("")
    if recommended:
        lines.append("| Ticker | Action | Notional (USD) | Priority | Intent | Reason |")
        lines.append("|---|---|---:|---:|---:|---|")
        for r in recommended:
            lines.append(
                f"| {r['ticker']} | {r['rebalance_action']} | "
                f"{r['rebalance_amount_usd']:.2f} | {r['priority']:.2f} | "
                f"{r['intent_score']:.2f} | {r['reason'] or '-'} |"
            )
    else:
        lines.append("_None today._")
    lines.append("")
    lines.append("## Delayed")
    lines.append("")
    if delayed:
        for r in delayed:
            lines.append(
                f"- **{r['ticker']}** {r['rebalance_action']} "
                f"(intent {r['intent_score']:.2f}): {r['reason'] or 'awaiting confirmation'}"
            )
    else:
        lines.append("_None today._")
    lines.append("")
    lines.append("## Blocked")
    lines.append("")
    if blocked:
        for r in blocked:
            lines.append(
                f"- **{r['ticker']}** {r['rebalance_action']} "
                f"(intent {r['intent_score']:.2f}): {r['reason'] or 'blocked by guardrail'}"
            )
    else:
        lines.append("_None today._")
    lines.append("")
    lines.append("## Risk Notes")
    lines.append("")
    if risk_notes:
        for n in risk_notes:
            lines.append(f"- **{n['kind']}**: {n['summary']}")
    else:
        lines.append("_No active risk-overlay flags._")
    lines.append("")
    lines.append("## CIO Recommendation")
    lines.append("")
    lines.append(cio_recommendation)
    lines.append("")
    return "\n".join(lines)


def _build_why_bullets(
    *,
    decision: str,
    deployment_pressure: float,
    defensive_pressure: float,
    components_deploy: Dict[str, float],
    components_defensive: Dict[str, float],
    meta: Dict[str, Any],
    scores: Dict[str, float],
    risk: Dict[str, Any],
    breakdown: Dict[str, Any],
    policy: Dict[str, Any],
) -> List[str]:
    """Human-readable bullets explaining the dominant drivers."""
    out: List[str] = []
    out.append(
        f"Deployment pressure scored {deployment_pressure:.2f} vs "
        f"defensive pressure {defensive_pressure:.2f}."
    )
    out.append(
        f"Self-trust is {meta['trust_level']} "
        f"(self_confidence {meta['self_confidence_score']:.2f})."
    )
    out.append(
        f"Portfolio health {scores['portfolio_health_score']:.2f}, "
        f"deployment readiness {scores['deployment_readiness_score']:.2f}, "
        f"governance {scores['governance_score']:.2f}."
    )
    out.append(
        f"Executable book: {breakdown['n_execute_buys']} buy(s), "
        f"{breakdown['n_execute_sells']} sell(s), "
        f"{breakdown['n_delay']} delayed, {breakdown['n_block']} blocked."
    )
    if risk["risk_pressure_score"] > 0.0:
        out.append(
            f"Risk overlay pressure {risk['risk_pressure_score']:.2f} "
            f"(force-exit {risk['n_force_exit']}, "
            f"block-new-buy {risk['n_block_new_buy']}, "
            f"trim {risk['n_trim_priority']}, "
            f"concentration {risk['n_concentration_warnings']})."
        )
    out.append(
        f"Active regime {policy['regime']}, "
        f"target_cash {policy['target_cash_pct']:.1f}%, "
        f"max_position {policy['max_position_pct']:.1f}%, "
        f"confidence floor {policy['confidence_threshold']:.2f}."
    )
    # Add the strongest pressure component for the chosen direction.
    if decision in (DEC_DEPLOY_SELECTIVELY, DEC_DEPLOY_AGGRESSIVELY):
        top = max(components_deploy.items(), key=lambda kv: kv[1])
        out.append(f"Dominant deployment driver: {top[0]} = {top[1]:.2f}.")
    if decision in (DEC_DEFENSIVE_ROTATION, DEC_CAPITAL_PRESERVATION, DEC_DELEVER):
        top = max(components_defensive.items(), key=lambda kv: kv[1])
        out.append(f"Dominant defensive driver: {top[0]} = {top[1]:.2f}.")
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_committee_decision(
    *,
    committee: Dict[str, Any],
    meta_intel: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    intents_df: pd.DataFrame,
    rebalance_summary: Dict[str, Any],
    regime_json: Dict[str, Any],
    meta_policy_summary: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    Returns (decision_json, markdown, summary_json).
    """
    now_iso = _now_iso_utc()

    breakdown = _extract_executable_breakdown(intents_df)
    risk = _extract_risk_pressure(committee)
    scores = _extract_committee_scores(committee)
    meta = _extract_meta(meta_intel)
    policy = _extract_policy(runtime_policy)
    regime = _extract_regime(regime_json)
    turnover = _extract_turnover(rebalance_summary)

    # If the committee disagrees with the regime engine, prefer the
    # adaptive_regime (most recent operational regime).
    if regime["regime"] and regime["regime"] != "UNKNOWN":
        policy["regime"] = regime["regime"]

    dep_score, dep_components = _deployment_pressure(
        breakdown=breakdown,
        committee_scores=scores,
        meta=meta,
        risk=risk,
        policy=policy,
    )
    def_score, def_components = _defensive_pressure(
        breakdown=breakdown,
        committee_scores=scores,
        meta=meta,
        risk=risk,
        policy=policy,
    )
    decision, tags, decision_reason = _select_decision(
        deployment_pressure=dep_score,
        defensive_pressure=def_score,
        breakdown=breakdown,
        meta=meta,
        risk=risk,
    )
    confidence = _recommendation_confidence(
        decision=decision,
        deployment_pressure=dep_score,
        defensive_pressure=def_score,
        meta=meta,
        breakdown=breakdown,
    )
    recommended, delayed, blocked = _build_action_groups(intents_df)
    risk_notes = _build_risk_notes_list(committee)
    why_bullets = _build_why_bullets(
        decision=decision,
        deployment_pressure=dep_score,
        defensive_pressure=def_score,
        components_deploy=dep_components,
        components_defensive=def_components,
        meta=meta,
        scores=scores,
        risk=risk,
        breakdown=breakdown,
        policy=policy,
    )
    cio_text = _build_cio_recommendation(
        decision=decision,
        decision_reason=decision_reason,
        confidence=confidence,
        deployment_pressure=dep_score,
        defensive_pressure=def_score,
        breakdown=breakdown,
        meta=meta,
        policy=policy,
        risk=risk,
        scores=scores,
        turnover=turnover,
    )

    decision_json: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_committee_engine",
        "engine_version": 1,
        "decision": decision,
        "recommendation_confidence": confidence,
        "rationale_short": decision_reason,
        "rationale_tags": tags,
        "regime": policy["regime"],
        "meta_trust_level": meta["trust_level"],
        "self_confidence_score": round(meta["self_confidence_score"], 4),
        "deployment_pressure": dep_score,
        "defensive_pressure": def_score,
        "pressure_components": {
            "deployment": dep_components,
            "defensive": def_components,
        },
        "committee_scores": scores,
        "risk_pressure": risk,
        "policy_in_force": policy,
        "executable_breakdown": breakdown,
        "rebalance_turnover": turnover,
        "recommended_actions": recommended,
        "delayed_actions": delayed,
        "blocked_actions": blocked,
        "risk_notes": risk_notes,
        "why_bullets": why_bullets,
        "cio_recommendation": cio_text,
        "thresholds": {
            "deployment_aggressive_floor": DEPLOYMENT_AGGRESSIVE_FLOOR,
            "deployment_selective_floor": DEPLOYMENT_SELECTIVE_FLOOR,
            "defensive_capital_preservation_floor": DEFENSIVE_CAPITAL_PRESERVATION_FLOOR,
            "defensive_rotation_floor": DEFENSIVE_ROTATION_FLOOR,
            "risk_pressure_severe": RISK_PRESSURE_SEVERE,
            "risk_pressure_elevated": RISK_PRESSURE_ELEVATED,
        },
        "inputs_seen": {
            "investment_committee_report": bool(committee),
            "meta_decision_intelligence": bool(meta_intel),
            "runtime_policy": bool(runtime_policy),
            "portfolio_execution_intents": (intents_df is not None and not intents_df.empty),
            "portfolio_rebalance_summary": bool(rebalance_summary),
            "adaptive_regime": bool(regime_json),
            "meta_policy_summary": bool(meta_policy_summary),
        },
        "meta_policy_modifier_summary": ((meta_policy_summary or {}).get("modifier_summary") or {}),
    }

    markdown = _render_markdown(
        decision=decision,
        decision_reason=decision_reason,
        confidence=confidence,
        why_bullets=why_bullets,
        recommended=recommended,
        delayed=delayed,
        blocked=blocked,
        risk_notes=risk_notes,
        cio_recommendation=cio_text,
        generated_at=now_iso,
        regime=policy["regime"],
        trust_level=meta["trust_level"],
        self_confidence=meta["self_confidence_score"],
    )

    summary: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_committee_engine",
        "engine_version": 1,
        "decision": decision,
        "recommendation_confidence": confidence,
        "deployment_pressure": dep_score,
        "defensive_pressure": def_score,
        "regime": policy["regime"],
        "meta_trust_level": meta["trust_level"],
        "self_confidence_score": round(meta["self_confidence_score"], 4),
        "n_execute_now": breakdown["n_execute_now"],
        "n_delay": breakdown["n_delay"],
        "n_block": breakdown["n_block"],
        "n_skip": breakdown["n_skip"],
        "n_execute_buys": breakdown["n_execute_buys"],
        "n_execute_sells": breakdown["n_execute_sells"],
        "total_execute_notional_usd": round(breakdown["total_notional_usd"], 2),
        "rationale_short": decision_reason,
        "rationale_tags": tags,
        "top_recommended_tickers": [r["ticker"] for r in recommended[:5]],
        "top_blocked_tickers": [r["ticker"] for r in blocked[:5]],
    }
    return decision_json, markdown, summary


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only autonomous investment committee engine (Step 15). "
            "Synthesises every prior intelligence layer into a single "
            "categorical decision plus a CIO memo."
        ),
    )
    p.add_argument("--committee", default=str(DEFAULT_COMMITTEE_JSON))
    p.add_argument("--meta-intel", default=str(DEFAULT_META_INTEL))
    p.add_argument("--runtime-policy", default=str(DEFAULT_RUNTIME_POLICY))
    p.add_argument("--intents", default=str(DEFAULT_INTENTS_CSV))
    p.add_argument("--rebalance-summary", default=str(DEFAULT_REBALANCE_SUMMARY))
    p.add_argument("--regime", default=str(DEFAULT_REGIME_JSON))
    p.add_argument("--meta-policy-summary", default=str(DEFAULT_META_POLICY_SUMMARY))
    p.add_argument("--out-decision", default=str(DEFAULT_OUT_DECISION))
    p.add_argument("--out-report", default=str(DEFAULT_OUT_REPORT))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[AUTONOMOUS_COMMITTEE] starting (read-only final decision layer)", flush=True)

    committee = _safe_read_json(Path(args.committee), label="investment_committee_report.json")
    meta_intel = _safe_read_json(Path(args.meta_intel), label="meta_decision_intelligence.json")
    runtime_policy = _safe_read_json(Path(args.runtime_policy), label="runtime_policy.json")
    rebalance_summary = _safe_read_json(
        Path(args.rebalance_summary), label="portfolio_rebalance_summary.json"
    )
    regime_json = _safe_read_json(Path(args.regime), label="adaptive_regime.json")
    meta_policy_summary = _safe_read_json(
        Path(args.meta_policy_summary), label="meta_policy_summary.json"
    )
    intents_df = _safe_read_csv(Path(args.intents), label="portfolio_execution_intents.csv")

    decision_json, markdown, summary = build_committee_decision(
        committee=committee,
        meta_intel=meta_intel,
        runtime_policy=runtime_policy,
        intents_df=intents_df,
        rebalance_summary=rebalance_summary,
        regime_json=regime_json,
        meta_policy_summary=meta_policy_summary,
    )

    try:
        _atomic_write_json(decision_json, Path(args.out_decision))
    except Exception as e:
        _warn(f"failed to write {args.out_decision}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_text(markdown, Path(args.out_report))
    except Exception as e:
        _warn(f"failed to write {args.out_report}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(summary, Path(args.out_summary))
    except Exception as e:
        _warn(f"failed to write {args.out_summary}: {type(e).__name__}: {e}")
        return 2

    print(
        "[AUTONOMOUS_COMMITTEE] "
        f"decision={decision_json['decision']} "
        f"confidence={decision_json['recommendation_confidence']:.3f} "
        f"execute={summary['n_execute_now']} "
        f"delay={summary['n_delay']} "
        f"block={summary['n_block']}",
        flush=True,
    )
    print(
        f"[AUTONOMOUS_COMMITTEE_OUT] decision={Path(args.out_decision).as_posix()} "
        f"report={Path(args.out_report).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
