"""
Investment Committee Report Engine — Step 9 (final synthesis layer).

Reads:
    data/results/trade_rationale.csv
    data/results/portfolio_execution_summary.json
    data/results/portfolio_construction_summary.json
    data/results/portfolio_rebalance_summary.json
    data/results/opportunity_persistence_summary.json
    data/results/capital_deployment_summary.json
    data/results/performance_risk_overlay.csv

Writes:
    data/results/investment_committee_report.json
    data/results/investment_committee_report.md
    data/results/investment_committee_summary.json

Purpose
-------
Steps 4-8 each produced focused operational artefacts. This engine
synthesises them into a single CIO-grade memo:

    "What is Triton recommending overall and why?"

The report has six sections (Executive Summary, Recommended Actions,
Portfolio Diagnostics, Opportunity Pipeline, Risk Notes, CIO Narrative)
and five composite scores (portfolio_health, deployment_readiness,
conviction, diversification, governance).

Safety
------
* Read-only. No broker calls, no execution-state mutation, no writes
  outside ``data/results``.
* Missing inputs warn and continue — the report degrades gracefully and
  records which sections were synthesised from partial data.
* Atomic writes (``.tmp`` + ``os.replace``).
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

DEFAULT_RATIONALE_CSV = RESULTS_DIR / "trade_rationale.csv"
DEFAULT_EXECUTION_SUMMARY = RESULTS_DIR / "portfolio_execution_summary.json"
DEFAULT_CONSTRUCTION_SUMMARY = RESULTS_DIR / "portfolio_construction_summary.json"
DEFAULT_REBALANCE_SUMMARY = RESULTS_DIR / "portfolio_rebalance_summary.json"
DEFAULT_PERSISTENCE_SUMMARY = RESULTS_DIR / "opportunity_persistence_summary.json"
DEFAULT_DEPLOYMENT_SUMMARY = RESULTS_DIR / "capital_deployment_summary.json"
DEFAULT_RISK_OVERLAY_CSV = RESULTS_DIR / "performance_risk_overlay.csv"

DEFAULT_OUT_JSON = RESULTS_DIR / "investment_committee_report.json"
DEFAULT_OUT_MD = RESULTS_DIR / "investment_committee_report.md"
DEFAULT_OUT_SUMMARY_JSON = RESULTS_DIR / "investment_committee_summary.json"

# -----------------------------------------------------------
# Tunables (committee thresholds)
# -----------------------------------------------------------
# Market posture inferred from risk overlay severity vs deploy readiness.
POSTURE_DEFENSIVE_FORCE_EXIT = 2  # >=2 FORCE_EXIT symbols → DEFENSIVE
POSTURE_DEFENSIVE_BLOCK_NEW = 5  # >=5 BLOCK_NEW_BUY → DEFENSIVE
POSTURE_AGGRESSIVE_DEPLOY = 4  # >=4 EXECUTE_NOW buys & no force-exits

# Deployment stance.
STANCE_DEPLOY_MIN_EXECUTE = 1
STANCE_AGGRESSIVE_MIN_EXECUTE = 3

# Concentration risk thresholds (% of total book value in single sector).
CONCENTRATION_WARN_PCT = 25.0
CONCENTRATION_CRITICAL_PCT = 40.0

# Cash reserve discipline.
CASH_RESERVE_LOW_PCT = 5.0
CASH_RESERVE_OK_PCT = 10.0
CASH_RESERVE_TARGET_PCT = 15.0

# Conviction label → numeric (for averaging in conviction_score).
CONVICTION_LABEL_VALUE: Dict[str, float] = {
    "INSTITUTIONAL": 1.00,
    "STRONG": 0.75,
    "MODERATE": 0.50,
    "WEAK": 0.25,
}

# Risk flag tokens.
RISK_FORCE_EXIT = "FORCE_EXIT"
RISK_TRIM_PRIORITY = "TRIM_PRIORITY"
RISK_BLOCK_NEW_BUY = "BLOCK_NEW_BUY"
RISK_OK = "OK"

# Intent labels.
INTENT_EXECUTE = "EXECUTE_NOW"
INTENT_DELAY = "DELAY"
INTENT_SKIP = "SKIP"
INTENT_BLOCK = "BLOCK"

BUY_DIRECTION_ACTIONS: frozenset = frozenset({"BUY_NEW", "ADD"})
SELL_DIRECTION_ACTIONS: frozenset = frozenset({"FULL_EXIT", "TRIM", "SELL"})


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[COMMITTEE_WARN] {msg}", flush=True)


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


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


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
# Coercion helpers
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


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _norm_symbol(x: Any) -> str:
    return str(x or "").strip().upper()


def _norm_upper(x: Any) -> str:
    return str(x or "").strip().upper()


def _risk_components(risk_flag: str) -> List[str]:
    if not risk_flag:
        return []
    parts = [p.strip().upper() for p in str(risk_flag).split("|")]
    return [p for p in parts if p and p != RISK_OK]


def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------------------------------------
# Posture / stance / conviction labels
# -----------------------------------------------------------
def _classify_market_posture(
    *,
    force_exit_count: int,
    block_new_buy_count: int,
    execute_buy_count: int,
    blocked_count: int,
    total_actions: int,
) -> str:
    """DEFENSIVE → NEUTRAL → OPPORTUNISTIC → AGGRESSIVE."""
    if (
        force_exit_count >= POSTURE_DEFENSIVE_FORCE_EXIT
        or block_new_buy_count >= POSTURE_DEFENSIVE_BLOCK_NEW
    ):
        return "DEFENSIVE"
    if force_exit_count >= 1 or block_new_buy_count >= 2:
        return "NEUTRAL"
    if execute_buy_count >= POSTURE_AGGRESSIVE_DEPLOY and blocked_count == 0:
        return "AGGRESSIVE"
    if execute_buy_count >= 1:
        return "OPPORTUNISTIC"
    return "NEUTRAL"


def _classify_deployment_stance(
    *,
    execute_buy_count: int,
    delayed_count: int,
    blocked_count: int,
    deployable_capital: float,
) -> str:
    """HOLD → CAUTIOUS_DEPLOY → DEPLOY → AGGRESSIVE_DEPLOY."""
    if deployable_capital <= 0 and execute_buy_count == 0:
        return "HOLD"
    if execute_buy_count >= STANCE_AGGRESSIVE_MIN_EXECUTE and blocked_count == 0:
        return "AGGRESSIVE_DEPLOY"
    if execute_buy_count >= STANCE_DEPLOY_MIN_EXECUTE:
        if delayed_count > execute_buy_count or blocked_count > 0:
            return "CAUTIOUS_DEPLOY"
        return "DEPLOY"
    if delayed_count >= 1:
        return "CAUTIOUS_DEPLOY"
    return "HOLD"


def _classify_portfolio_conviction(conviction_counts: Counter) -> str:
    """Top-tier conviction label that has the most weight in the portfolio."""
    if not conviction_counts:
        return "WEAK"
    weighted = sum(
        CONVICTION_LABEL_VALUE.get(label, 0.0) * n for label, n in conviction_counts.items()
    )
    total = sum(conviction_counts.values())
    if total <= 0:
        return "WEAK"
    avg = weighted / total
    if avg >= 0.85:
        return "INSTITUTIONAL"
    if avg >= 0.65:
        return "STRONG"
    if avg >= 0.40:
        return "MODERATE"
    return "WEAK"


# -----------------------------------------------------------
# Composite scores
# -----------------------------------------------------------
def _portfolio_health_score(
    *,
    construction_score: Optional[float],
    risk_clean_ratio: float,
    cash_reserve_pct: Optional[float],
) -> float:
    """
    Health = construction quality (50%) + risk cleanliness (30%) +
    cash discipline (20%). Cash discipline peaks at the target reserve
    and decays both above and below.
    """
    cons = _clamp(_to_float_or_zero(construction_score))
    risk = _clamp(risk_clean_ratio)
    if cash_reserve_pct is None:
        cash = 0.5  # neutral if unknown
    else:
        # Triangular: 0 at <=0%, 1.0 at 15%, decays back to 0.5 at 30%+
        c = cash_reserve_pct
        if c <= 0:
            cash = 0.0
        elif c < CASH_RESERVE_TARGET_PCT:
            cash = c / CASH_RESERVE_TARGET_PCT
        elif c <= 2 * CASH_RESERVE_TARGET_PCT:
            cash = 1.0 - 0.5 * (c - CASH_RESERVE_TARGET_PCT) / CASH_RESERVE_TARGET_PCT
        else:
            cash = 0.5
        cash = _clamp(cash)
    return round(_clamp(0.5 * cons + 0.3 * risk + 0.2 * cash), 6)


def _deployment_readiness_score(
    *,
    execute_buy_count: int,
    total_buy_candidates: int,
    avg_intent_score: Optional[float],
    deployable_capital: float,
    total_portfolio_value: Optional[float],
) -> float:
    """
    Readiness = (execute-ready fraction of buys) blended with avg intent
    score and capacity-to-deploy. If there is no capital to deploy, the
    score collapses to zero regardless of intent.
    """
    if deployable_capital <= 0 or (
        total_portfolio_value is not None and total_portfolio_value <= 0
    ):
        return 0.0
    ready_frac = (execute_buy_count / total_buy_candidates) if total_buy_candidates > 0 else 0.0
    intent = _clamp(_to_float_or_zero(avg_intent_score))
    capacity = 1.0
    if total_portfolio_value is not None and total_portfolio_value > 0:
        capacity = _clamp(deployable_capital / max(1.0, 0.20 * total_portfolio_value))
    return round(_clamp(0.50 * ready_frac + 0.30 * intent + 0.20 * capacity), 6)


def _conviction_score(conviction_counts: Counter) -> float:
    if not conviction_counts:
        return 0.0
    weighted = sum(
        CONVICTION_LABEL_VALUE.get(label, 0.0) * n for label, n in conviction_counts.items()
    )
    total = sum(conviction_counts.values())
    if total <= 0:
        return 0.0
    return round(_clamp(weighted / total), 6)


def _governance_score(
    *,
    blocked_count: int,
    total_actions: int,
    risk_clean_ratio: float,
    cash_reserve_pct: Optional[float],
    construction_score: Optional[float],
    has_construction_summary: bool,
) -> float:
    """
    Governance = does Triton respect its own rules?
      * Blocked rows are *good* governance (the engine refused a bad trade).
      * Risk-clean ratio indicates discipline.
      * Cash reserve discipline (peaks at the OK floor).
      * Construction summary present = the constraint pipeline ran.
    """
    if total_actions <= 0:
        block_score = 0.5
    else:
        # Some blocks = good. But >50% blocks = something upstream is broken.
        ratio = blocked_count / total_actions
        if ratio == 0:
            block_score = 0.7  # no blocks needed → clean book
        elif ratio < 0.30:
            block_score = 1.0  # healthy guard-rail activity
        elif ratio < 0.60:
            block_score = 0.6
        else:
            block_score = 0.3
    risk = _clamp(risk_clean_ratio)
    if cash_reserve_pct is None:
        cash_disc = 0.5
    elif cash_reserve_pct >= CASH_RESERVE_OK_PCT:
        cash_disc = 1.0
    elif cash_reserve_pct >= CASH_RESERVE_LOW_PCT:
        cash_disc = 0.6
    else:
        cash_disc = 0.2
    pipeline_ok = 1.0 if has_construction_summary else 0.5
    cons = _clamp(_to_float_or_zero(construction_score))
    return round(
        _clamp(
            0.30 * block_score + 0.25 * risk + 0.20 * cash_disc + 0.10 * pipeline_ok + 0.15 * cons
        ),
        6,
    )


# -----------------------------------------------------------
# Aggregation
# -----------------------------------------------------------
def _rationale_dataframe_to_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        sym = _norm_symbol(r.get("ticker") or r.get("symbol"))
        if not sym:
            continue
        rows.append(
            {
                "ticker": sym,
                "execution_intent": _norm_upper(r.get("execution_intent")),
                "rebalance_action": _norm_upper(r.get("rebalance_action")),
                "confidence": _to_float(r.get("confidence")),
                "persistence_score": _to_float(r.get("persistence_score")),
                "delta_pct": _to_float(r.get("delta_pct")),
                "lifecycle_action": _norm_upper(r.get("lifecycle_action")),
                "signal": _norm_upper(r.get("signal")),
                "risk_flag": _norm_upper(r.get("risk_flag")) or RISK_OK,
                "rationale_short": str(r.get("rationale_short") or ""),
                "rationale_long": str(r.get("rationale_long") or ""),
                "rationale_tags": str(r.get("rationale_tags") or ""),
                "confidence_label": _norm_upper(r.get("confidence_label")),
                "conviction_label": _norm_upper(r.get("conviction_label")),
                "explanation_score": _to_float_or_zero(r.get("explanation_score")),
            }
        )
    return rows


def _risk_overlay_lookup(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if df is None or df.empty:
        return out
    sym_col = _pick_first_present(df, ("ticker", "symbol"))
    if not sym_col:
        return out
    for _, r in df.iterrows():
        sym = _norm_symbol(r.get(sym_col))
        if sym:
            out[sym] = _norm_upper(r.get("risk_flag")) or RISK_OK
    return out


# -----------------------------------------------------------
# Report builder
# -----------------------------------------------------------
def build_report(
    *,
    rationale_rows: List[Dict[str, Any]],
    execution_summary: Dict[str, Any],
    construction_summary: Dict[str, Any],
    rebalance_summary: Dict[str, Any],
    persistence_summary: Dict[str, Any],
    deployment_summary: Dict[str, Any],
    risk_overlay_map: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Pure planner — no IO. Returns (full_report_obj, compact_summary).
    """
    # ── Tally intents ────────────────────────────────────────────
    intent_counter: Counter = Counter(r["execution_intent"] for r in rationale_rows)
    execute_n = int(intent_counter.get(INTENT_EXECUTE, 0))
    delay_n = int(intent_counter.get(INTENT_DELAY, 0))
    skip_n = int(intent_counter.get(INTENT_SKIP, 0))
    block_n = int(intent_counter.get(INTENT_BLOCK, 0))
    total_actions = int(len(rationale_rows))

    # Buy vs sell breakdown inside EXECUTE_NOW.
    execute_buy_rows = [
        r
        for r in rationale_rows
        if r["execution_intent"] == INTENT_EXECUTE
        and r["rebalance_action"] in BUY_DIRECTION_ACTIONS
    ]
    execute_sell_rows = [
        r
        for r in rationale_rows
        if r["execution_intent"] == INTENT_EXECUTE
        and r["rebalance_action"] in SELL_DIRECTION_ACTIONS
    ]
    execute_buy_count = len(execute_buy_rows)
    execute_sell_count = len(execute_sell_rows)
    total_buy_candidates = sum(
        1 for r in rationale_rows if r["rebalance_action"] in BUY_DIRECTION_ACTIONS
    )

    # ── Risk overlay diagnostics ────────────────────────────────
    force_exit_symbols: List[str] = []
    trim_priority_symbols: List[str] = []
    block_new_buy_symbols: List[str] = []
    for sym, flag in risk_overlay_map.items():
        comps = _risk_components(flag)
        if RISK_FORCE_EXIT in comps:
            force_exit_symbols.append(sym)
        if RISK_TRIM_PRIORITY in comps:
            trim_priority_symbols.append(sym)
        if RISK_BLOCK_NEW_BUY in comps:
            block_new_buy_symbols.append(sym)
    force_exit_symbols.sort()
    trim_priority_symbols.sort()
    block_new_buy_symbols.sort()
    n_risk_total = max(1, len(risk_overlay_map))
    n_flagged = sum(1 for f in risk_overlay_map.values() if _risk_components(f))
    risk_clean_ratio = 1.0 - (n_flagged / n_risk_total)

    # ── Lifecycle contradictions (from rationale tags) ──────────
    lifecycle_contradictions: List[Dict[str, Any]] = [
        {
            "ticker": r["ticker"],
            "lifecycle_action": r["lifecycle_action"],
            "rebalance_action": r["rebalance_action"],
            "rationale_short": r["rationale_short"],
        }
        for r in rationale_rows
        if "lifecycle_contradiction" in r["rationale_tags"]
    ]

    # ── Concentration warnings (from construction summary) ──────
    sector_dist: Dict[str, float] = dict(construction_summary.get("sector_distribution") or {})
    cluster_dist: Dict[str, float] = dict(
        construction_summary.get("correlation_cluster_distribution") or {}
    )
    concentration_warnings: List[Dict[str, Any]] = []
    for label, dist in (("sector", sector_dist), ("cluster", cluster_dist)):
        for name, pct in dist.items():
            p = _to_float_or_zero(pct)
            if p >= CONCENTRATION_CRITICAL_PCT:
                concentration_warnings.append(
                    {
                        "type": label,
                        "name": name,
                        "weight_pct": p,
                        "severity": "CRITICAL",
                    }
                )
            elif p >= CONCENTRATION_WARN_PCT:
                concentration_warnings.append(
                    {
                        "type": label,
                        "name": name,
                        "weight_pct": p,
                        "severity": "WARN",
                    }
                )
    concentration_warnings.sort(key=lambda d: (-d["weight_pct"], d["name"]))

    # ── Conviction / confidence distributions ───────────────────
    conviction_counts: Counter = Counter(
        r["conviction_label"] for r in rationale_rows if r["conviction_label"]
    )
    confidence_counts: Counter = Counter(
        r["confidence_label"] for r in rationale_rows if r["confidence_label"]
    )

    # ── Section A: Executive Summary ────────────────────────────
    posture = _classify_market_posture(
        force_exit_count=len(force_exit_symbols),
        block_new_buy_count=len(block_new_buy_symbols),
        execute_buy_count=execute_buy_count,
        blocked_count=block_n,
        total_actions=total_actions,
    )
    deployable_capital = _to_float_or_zero(
        construction_summary.get("deployable_capital")
        or rebalance_summary.get("deployable_capital_estimate")
        or deployment_summary.get("deployable_capital_estimate")
    )
    total_portfolio_value = _to_float(
        construction_summary.get("total_portfolio_value")
        or rebalance_summary.get("total_portfolio_value")
        or deployment_summary.get("total_portfolio_value")
    )
    stance = _classify_deployment_stance(
        execute_buy_count=execute_buy_count,
        delayed_count=delay_n,
        blocked_count=block_n,
        deployable_capital=deployable_capital,
    )
    portfolio_conviction = _classify_portfolio_conviction(conviction_counts)

    if execute_buy_count > 0 and execute_sell_count > 0:
        overall_reco = (
            f"Rotate: {execute_sell_count} risk-protective sell(s) and "
            f"{execute_buy_count} conviction buy(s) recommended now."
        )
    elif execute_buy_count > 0:
        overall_reco = f"Deploy: {execute_buy_count} high-conviction buy(s) recommended now."
    elif execute_sell_count > 0:
        overall_reco = f"De-risk: {execute_sell_count} risk-protective sell(s) recommended now."
    elif delay_n > 0 or block_n > 0:
        overall_reco = (
            "Hold: no executes; "
            f"{delay_n} candidate(s) awaiting confirmation, "
            f"{block_n} blocked by upstream guards."
        )
    else:
        overall_reco = "Hold: no actionable trades today."

    # ── Section B: Recommended Actions (grouped) ────────────────
    def _action_card(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ticker": r["ticker"],
            "rebalance_action": r["rebalance_action"],
            "confidence_label": r["confidence_label"],
            "conviction_label": r["conviction_label"],
            "explanation_score": r["explanation_score"],
            "rationale_short": r["rationale_short"],
        }

    def _group(intent: str) -> List[Dict[str, Any]]:
        rows = [r for r in rationale_rows if r["execution_intent"] == intent]
        rows.sort(key=lambda r: (-r["explanation_score"], r["ticker"]))
        return [_action_card(r) for r in rows]

    recommended_actions: Dict[str, Any] = {
        "execute_now": _group(INTENT_EXECUTE),
        "delay": _group(INTENT_DELAY),
        "block": _group(INTENT_BLOCK),
        "skip": _group(INTENT_SKIP),
        "totals": {
            "execute_now": execute_n,
            "delay": delay_n,
            "block": block_n,
            "skip": skip_n,
            "execute_buys": execute_buy_count,
            "execute_sells": execute_sell_count,
        },
    }

    # ── Section C: Portfolio Diagnostics ────────────────────────
    construction_score = _to_float(construction_summary.get("portfolio_construction_score"))
    diversification_score = _to_float(construction_summary.get("diversification_score"))
    concentration_risk = _to_float(construction_summary.get("concentration_risk"))
    cash_reserve_pct = _to_float(construction_summary.get("cash_reserve_pct"))
    target_cash_pct = _to_float(construction_summary.get("target_cash_pct"))
    portfolio_turnover = _to_float(rebalance_summary.get("portfolio_turnover_pct"))
    current_positions = int(
        _to_float_or_zero(construction_summary.get("current_positions"))
        or _to_float_or_zero(deployment_summary.get("current_positions"))
    )
    max_positions = int(
        _to_float_or_zero(construction_summary.get("max_positions"))
        or _to_float_or_zero(deployment_summary.get("max_positions"))
    )
    available_slots = int(
        _to_float_or_zero(construction_summary.get("available_slots"))
        or _to_float_or_zero(deployment_summary.get("available_slots"))
        or _to_float_or_zero(rebalance_summary.get("available_slots"))
    )

    portfolio_diagnostics: Dict[str, Any] = {
        "total_portfolio_value": total_portfolio_value,
        "current_positions": current_positions,
        "max_positions": max_positions,
        "available_slots": available_slots,
        "construction_score": construction_score,
        "diversification_score": diversification_score,
        "concentration_risk_pct": concentration_risk,
        "cash_reserve_pct": cash_reserve_pct,
        "target_cash_pct": target_cash_pct,
        "deployable_capital_estimate": deployable_capital,
        "portfolio_turnover_pct": portfolio_turnover,
        "sector_distribution": sector_dist,
        "correlation_cluster_distribution": cluster_dist,
    }

    # ── Section D: Opportunity Pipeline ─────────────────────────
    strongest_watch = list(persistence_summary.get("strongest_candidates") or [])
    weakest_watch = list(persistence_summary.get("weakest_candidates") or [])
    top_deployments = list(deployment_summary.get("top_deployments") or [])
    promoted_candidates = [
        {
            "ticker": r["ticker"],
            "rebalance_action": r["rebalance_action"],
            "confidence_label": r["confidence_label"],
            "conviction_label": r["conviction_label"],
            "explanation_score": r["explanation_score"],
        }
        for r in rationale_rows
        if r["execution_intent"] == INTENT_EXECUTE
        and r["rebalance_action"] in BUY_DIRECTION_ACTIONS
    ]
    delayed_candidates = [
        {
            "ticker": r["ticker"],
            "rebalance_action": r["rebalance_action"],
            "confidence_label": r["confidence_label"],
            "conviction_label": r["conviction_label"],
            "rationale_short": r["rationale_short"],
        }
        for r in rationale_rows
        if r["execution_intent"] == INTENT_DELAY
    ]
    blocked_candidates = [
        {
            "ticker": r["ticker"],
            "rebalance_action": r["rebalance_action"],
            "rationale_short": r["rationale_short"],
        }
        for r in rationale_rows
        if r["execution_intent"] == INTENT_BLOCK
    ]
    opportunity_pipeline: Dict[str, Any] = {
        "watch_summary": {
            "watch_candidates": int(_to_float_or_zero(persistence_summary.get("watch_candidates"))),
            "promoted_confirmed": int(
                _to_float_or_zero(persistence_summary.get("promoted_confirmed"))
            ),
            "kept_watch": int(_to_float_or_zero(persistence_summary.get("kept_watch"))),
            "demoted": int(_to_float_or_zero(persistence_summary.get("demoted"))),
            "rejected": int(_to_float_or_zero(persistence_summary.get("rejected"))),
            "avg_persistence_score": _to_float(persistence_summary.get("avg_persistence_score")),
        },
        "strongest_watch_candidates": strongest_watch,
        "weakest_watch_candidates": weakest_watch,
        "top_capital_deployments": top_deployments,
        "promoted_to_execute": promoted_candidates,
        "delayed_candidates": delayed_candidates,
        "blocked_candidates": blocked_candidates,
    }

    # ── Section E: Risk Notes ───────────────────────────────────
    risk_notes: Dict[str, Any] = {
        "force_exit_symbols": force_exit_symbols,
        "trim_priority_symbols": trim_priority_symbols,
        "block_new_buy_symbols": block_new_buy_symbols,
        "lifecycle_contradictions": lifecycle_contradictions,
        "concentration_warnings": concentration_warnings,
        "risk_clean_ratio": round(risk_clean_ratio, 6),
        "n_symbols_with_risk_flag": n_flagged,
        "n_symbols_evaluated": int(len(risk_overlay_map)),
    }

    # ── Composite scores ────────────────────────────────────────
    portfolio_health_score = _portfolio_health_score(
        construction_score=construction_score,
        risk_clean_ratio=risk_clean_ratio,
        cash_reserve_pct=cash_reserve_pct,
    )
    deployment_readiness_score = _deployment_readiness_score(
        execute_buy_count=execute_buy_count,
        total_buy_candidates=total_buy_candidates,
        avg_intent_score=_to_float(execution_summary.get("avg_intent_score")),
        deployable_capital=deployable_capital,
        total_portfolio_value=total_portfolio_value,
    )
    conviction_score = _conviction_score(conviction_counts)
    diversification_score_norm = _clamp(_to_float_or_zero(diversification_score))
    governance_score = _governance_score(
        blocked_count=block_n,
        total_actions=total_actions,
        risk_clean_ratio=risk_clean_ratio,
        cash_reserve_pct=cash_reserve_pct,
        construction_score=construction_score,
        has_construction_summary=bool(construction_summary),
    )

    scores: Dict[str, Any] = {
        "portfolio_health_score": portfolio_health_score,
        "deployment_readiness_score": deployment_readiness_score,
        "conviction_score": conviction_score,
        "diversification_score": diversification_score_norm,
        "governance_score": governance_score,
    }

    executive_summary: Dict[str, Any] = {
        "market_posture": posture,
        "deployment_stance": stance,
        "portfolio_conviction": portfolio_conviction,
        "overall_recommendation": overall_reco,
        "scores": scores,
    }

    # ── Section F: CIO Narrative ────────────────────────────────
    cio_narrative = _build_cio_narrative(
        posture=posture,
        stance=stance,
        portfolio_conviction=portfolio_conviction,
        execute_buy_count=execute_buy_count,
        execute_sell_count=execute_sell_count,
        delay_n=delay_n,
        block_n=block_n,
        skip_n=skip_n,
        force_exit_symbols=force_exit_symbols,
        block_new_buy_symbols=block_new_buy_symbols,
        promoted_candidates=promoted_candidates,
        delayed_candidates=delayed_candidates,
        construction_score=construction_score,
        diversification_score=diversification_score,
        concentration_risk=concentration_risk,
        cash_reserve_pct=cash_reserve_pct,
        target_cash_pct=target_cash_pct,
        scores=scores,
        persistence_summary=persistence_summary,
        deployment_summary=deployment_summary,
    )

    # ── Assemble full report ────────────────────────────────────
    inputs_seen = {
        "execution_summary": bool(execution_summary),
        "construction_summary": bool(construction_summary),
        "rebalance_summary": bool(rebalance_summary),
        "persistence_summary": bool(persistence_summary),
        "deployment_summary": bool(deployment_summary),
        "rationale_rows": int(len(rationale_rows)),
        "risk_overlay_symbols": int(len(risk_overlay_map)),
    }

    full_report: Dict[str, Any] = {
        "generated_at_utc": _now_iso_utc(),
        "engine": "investment_committee_report_engine",
        "engine_version": 1,
        "inputs_seen": inputs_seen,
        "executive_summary": executive_summary,
        "recommended_actions": recommended_actions,
        "portfolio_diagnostics": portfolio_diagnostics,
        "opportunity_pipeline": opportunity_pipeline,
        "risk_notes": risk_notes,
        "cio_narrative": cio_narrative,
        "thresholds": {
            "posture_defensive_force_exit": POSTURE_DEFENSIVE_FORCE_EXIT,
            "posture_defensive_block_new": POSTURE_DEFENSIVE_BLOCK_NEW,
            "posture_aggressive_deploy": POSTURE_AGGRESSIVE_DEPLOY,
            "stance_deploy_min_execute": STANCE_DEPLOY_MIN_EXECUTE,
            "stance_aggressive_min_execute": STANCE_AGGRESSIVE_MIN_EXECUTE,
            "concentration_warn_pct": CONCENTRATION_WARN_PCT,
            "concentration_critical_pct": CONCENTRATION_CRITICAL_PCT,
            "cash_reserve_low_pct": CASH_RESERVE_LOW_PCT,
            "cash_reserve_ok_pct": CASH_RESERVE_OK_PCT,
            "cash_reserve_target_pct": CASH_RESERVE_TARGET_PCT,
        },
    }

    compact_summary: Dict[str, Any] = {
        "generated_at_utc": full_report["generated_at_utc"],
        "market_posture": posture,
        "deployment_stance": stance,
        "portfolio_conviction": portfolio_conviction,
        "overall_recommendation": overall_reco,
        "scores": scores,
        "intent_counts": {
            "execute_now": execute_n,
            "delay": delay_n,
            "block": block_n,
            "skip": skip_n,
        },
        "execute_buys": execute_buy_count,
        "execute_sells": execute_sell_count,
        "force_exit_symbols": force_exit_symbols,
        "block_new_buy_symbols": block_new_buy_symbols,
        "concentration_warnings": concentration_warnings,
        "cash_reserve_pct": cash_reserve_pct,
        "deployable_capital_estimate": deployable_capital,
        "construction_score": construction_score,
        "portfolio_turnover_pct": portfolio_turnover,
        "confidence_distribution": dict(confidence_counts),
        "conviction_distribution": dict(conviction_counts),
        "inputs_seen": inputs_seen,
    }
    return full_report, compact_summary


# -----------------------------------------------------------
# CIO Narrative
# -----------------------------------------------------------
def _build_cio_narrative(
    *,
    posture: str,
    stance: str,
    portfolio_conviction: str,
    execute_buy_count: int,
    execute_sell_count: int,
    delay_n: int,
    block_n: int,
    skip_n: int,
    force_exit_symbols: List[str],
    block_new_buy_symbols: List[str],
    promoted_candidates: List[Dict[str, Any]],
    delayed_candidates: List[Dict[str, Any]],
    construction_score: Optional[float],
    diversification_score: Optional[float],
    concentration_risk: Optional[float],
    cash_reserve_pct: Optional[float],
    target_cash_pct: Optional[float],
    scores: Dict[str, Any],
    persistence_summary: Dict[str, Any],
    deployment_summary: Dict[str, Any],
) -> str:
    """CIO-style multi-sentence paragraph synthesising the day."""
    parts: List[str] = []

    parts.append(
        f"Triton's read of the tape is {posture}. "
        f"Recommended deployment stance is {stance.replace('_', ' ').lower()}; "
        f"the aggregate book is rated {portfolio_conviction} on conviction."
    )

    # What we see
    see_bits: List[str] = []
    if force_exit_symbols:
        see_bits.append(
            f"the performance risk overlay is forcing exit on "
            f"{len(force_exit_symbols)} symbol(s) ({', '.join(force_exit_symbols[:5])}"
            + (", ..." if len(force_exit_symbols) > 5 else "")
            + ")"
        )
    if block_new_buy_symbols:
        see_bits.append(f"new-buy is blocked on {len(block_new_buy_symbols)} symbol(s)")
    if persistence_summary:
        avg_pers = _to_float(persistence_summary.get("avg_persistence_score"))
        watch_n = int(_to_float_or_zero(persistence_summary.get("watch_candidates")))
        if watch_n > 0:
            see_bits.append(
                f"{watch_n} WATCH candidate(s) tracked across the persistence engine"
                + (f" (avg persistence {avg_pers:.2f})" if avg_pers is not None else "")
            )
    if construction_score is not None:
        see_bits.append(f"portfolio construction score is {construction_score:.2f}")
    if concentration_risk is not None:
        see_bits.append(f"top sector concentration is {concentration_risk:.1f}%")
    if see_bits:
        parts.append("What Triton sees: " + "; ".join(see_bits) + ".")

    # What we recommend
    rec_bits: List[str] = []
    if execute_buy_count > 0:
        names = [c["ticker"] for c in promoted_candidates[:5]]
        rec_bits.append(
            f"{execute_buy_count} conviction buy(s)" + (f" ({', '.join(names)})" if names else "")
        )
    if execute_sell_count > 0:
        rec_bits.append(f"{execute_sell_count} risk-protective sell(s)")
    if delay_n > 0:
        names = [c["ticker"] for c in delayed_candidates[:5]]
        rec_bits.append(
            f"{delay_n} delayed candidate(s)"
            + (f" awaiting confirmation ({', '.join(names)})" if names else "")
        )
    if block_n > 0:
        rec_bits.append(f"{block_n} blocked candidate(s)")
    if skip_n > 0:
        rec_bits.append(f"{skip_n} skipped (non-actionable / below trade minimum)")
    if rec_bits:
        parts.append("What Triton recommends: " + "; ".join(rec_bits) + ".")
    else:
        parts.append("What Triton recommends: hold the line — no actionable trades today.")

    # Why patience or aggression
    if posture == "AGGRESSIVE":
        parts.append(
            "Why aggression: deploy gates are wide open (no severe risk flags, "
            "multiple confirmed setups), and capacity supports a coordinated push."
        )
    elif posture == "OPPORTUNISTIC":
        parts.append(
            "Why opportunism: at least one high-conviction setup cleared every gate "
            "(persistence, confidence, lifecycle, risk), so selective deployment is warranted."
        )
    elif posture == "DEFENSIVE":
        parts.append(
            "Why patience: the risk overlay is actively protecting the book — "
            "capital preservation dominates over new deployment until the risk-flag "
            "footprint shrinks."
        )
    elif stance == "CAUTIOUS_DEPLOY":
        parts.append(
            "Why caution: directional positivity is present but at least one gate "
            "(confidence borderline, persistence not yet confirmed, partial blocks) "
            "argues for measured sizing rather than a full deploy."
        )
    else:
        parts.append(
            "Why patience: no candidate fully cleared the deploy gates today; "
            "discipline over impulse — Triton waits for persistence to confirm "
            "before committing capital."
        )

    # Cash discipline closer
    if cash_reserve_pct is not None:
        if target_cash_pct is not None:
            if cash_reserve_pct < CASH_RESERVE_LOW_PCT:
                parts.append(
                    f"Cash discipline: reserve is thin ({cash_reserve_pct:.1f}% "
                    f"vs target {target_cash_pct:.1f}%); freeing capital via TRIM/EXIT "
                    "actions should precede any new deploys."
                )
            elif cash_reserve_pct >= CASH_RESERVE_OK_PCT:
                parts.append(
                    f"Cash discipline: reserve {cash_reserve_pct:.1f}% "
                    f"(target {target_cash_pct:.1f}%) — dry powder is healthy."
                )
            else:
                parts.append(
                    f"Cash discipline: reserve {cash_reserve_pct:.1f}% "
                    f"(target {target_cash_pct:.1f}%) — within tolerance."
                )

    parts.append(
        f"Composite scores -- health {scores['portfolio_health_score']:.2f}, "
        f"readiness {scores['deployment_readiness_score']:.2f}, "
        f"conviction {scores['conviction_score']:.2f}, "
        f"diversification {scores['diversification_score']:.2f}, "
        f"governance {scores['governance_score']:.2f}."
    )

    return " ".join(parts)


# -----------------------------------------------------------
# Markdown rendering
# -----------------------------------------------------------
def _fmt_pct(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}%"


def _fmt_score(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _fmt_money(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"${v:,.2f}"


def render_markdown(full_report: Dict[str, Any]) -> str:
    """Render the full report as a CIO-style markdown memo."""
    es = full_report["executive_summary"]
    ra = full_report["recommended_actions"]
    pd_ = full_report["portfolio_diagnostics"]
    op = full_report["opportunity_pipeline"]
    rn = full_report["risk_notes"]
    scores = es["scores"]

    lines: List[str] = []
    lines.append("# Triton Daily Investment Committee Memo")
    lines.append("")
    lines.append(f"*Generated {full_report['generated_at_utc']}*")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Market posture:** {es['market_posture']}")
    lines.append(f"- **Deployment stance:** {es['deployment_stance'].replace('_', ' ')}")
    lines.append(f"- **Portfolio conviction:** {es['portfolio_conviction']}")
    lines.append(f"- **Overall recommendation:** {es['overall_recommendation']}")
    lines.append("")
    lines.append("### Composite Scores")
    lines.append("")
    lines.append("| Score | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Portfolio health | {_fmt_score(scores['portfolio_health_score'])} |")
    lines.append(f"| Deployment readiness | {_fmt_score(scores['deployment_readiness_score'])} |")
    lines.append(f"| Conviction | {_fmt_score(scores['conviction_score'])} |")
    lines.append(f"| Diversification | {_fmt_score(scores['diversification_score'])} |")
    lines.append(f"| Governance | {_fmt_score(scores['governance_score'])} |")
    lines.append("")

    # Recommended Actions
    lines.append("## Recommended Actions")
    lines.append("")
    totals = ra["totals"]
    lines.append(
        f"_Totals — execute_now: {totals['execute_now']} "
        f"(buys {totals['execute_buys']}, sells {totals['execute_sells']}), "
        f"delay: {totals['delay']}, block: {totals['block']}, skip: {totals['skip']}._"
    )
    lines.append("")

    def _render_action_group(title: str, items: List[Dict[str, Any]]) -> None:
        lines.append(f"### {title}")
        lines.append("")
        if not items:
            lines.append("_(none)_")
            lines.append("")
            return
        lines.append("| Ticker | Action | Confidence | Conviction | Score | Rationale |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in items:
            lines.append(
                f"| {r['ticker']} | {r['rebalance_action']} | "
                f"{r['confidence_label'] or '-'} | {r['conviction_label'] or '-'} | "
                f"{_fmt_score(r['explanation_score'])} | {r['rationale_short']} |"
            )
        lines.append("")

    _render_action_group("EXECUTE_NOW", ra["execute_now"])
    _render_action_group("DELAY", ra["delay"])
    _render_action_group("BLOCK", ra["block"])
    _render_action_group("SKIP", ra["skip"])

    # Portfolio Diagnostics
    lines.append("## Portfolio Diagnostics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Total portfolio value | {_fmt_money(pd_['total_portfolio_value'])} |")
    lines.append(
        f"| Positions | {pd_['current_positions']} / {pd_['max_positions']} "
        f"(available slots {pd_['available_slots']}) |"
    )
    lines.append(f"| Construction score | {_fmt_score(pd_['construction_score'])} |")
    lines.append(f"| Diversification score | {_fmt_score(pd_['diversification_score'])} |")
    lines.append(
        f"| Concentration risk | {_fmt_pct(pd_['concentration_risk_pct'])} " f"(top sector) |"
    )
    lines.append(
        f"| Cash reserve | {_fmt_pct(pd_['cash_reserve_pct'])} "
        f"(target {_fmt_pct(pd_['target_cash_pct'])}) |"
    )
    lines.append(
        f"| Deployable capital estimate | {_fmt_money(pd_['deployable_capital_estimate'])} |"
    )
    lines.append(f"| Portfolio turnover | {_fmt_pct(pd_['portfolio_turnover_pct'])} |")
    lines.append("")
    if pd_["sector_distribution"]:
        lines.append(
            "**Sector distribution:** "
            + ", ".join(
                f"{k} {_fmt_pct(v)}"
                for k, v in sorted(
                    pd_["sector_distribution"].items(),
                    key=lambda kv: -_to_float_or_zero(kv[1]),
                )
            )
        )
        lines.append("")

    # Opportunity Pipeline
    lines.append("## Opportunity Pipeline")
    lines.append("")
    ws = op["watch_summary"]
    lines.append(
        f"_WATCH funnel — candidates: {ws['watch_candidates']}, "
        f"kept: {ws['kept_watch']}, promoted-confirmed: {ws['promoted_confirmed']}, "
        f"demoted: {ws['demoted']}, rejected: {ws['rejected']}, "
        f"avg persistence: {_fmt_score(ws['avg_persistence_score'])}._"
    )
    lines.append("")

    if op["promoted_to_execute"]:
        lines.append("### Promoted to Execute")
        lines.append("")
        lines.append("| Ticker | Action | Confidence | Conviction | Score |")
        lines.append("| --- | --- | --- | --- | --- |")
        for c in op["promoted_to_execute"]:
            lines.append(
                f"| {c['ticker']} | {c['rebalance_action']} | "
                f"{c['confidence_label'] or '-'} | {c['conviction_label'] or '-'} | "
                f"{_fmt_score(c['explanation_score'])} |"
            )
        lines.append("")

    if op["strongest_watch_candidates"]:
        lines.append("### Strongest Watch Candidates")
        lines.append("")
        lines.append("| Ticker | Decision | Persistence | Confidence | Reason |")
        lines.append("| --- | --- | --- | --- | --- |")
        for c in op["strongest_watch_candidates"][:10]:
            lines.append(
                f"| {c.get('ticker', '-')} | {c.get('promotion_decision', '-')} | "
                f"{_fmt_score(c.get('persistence_score'))} | "
                f"{_fmt_score(c.get('latest_confidence'))} | "
                f"{c.get('reason', '')} |"
            )
        lines.append("")

    if op["delayed_candidates"]:
        lines.append("### Delayed Candidates")
        lines.append("")
        for c in op["delayed_candidates"]:
            lines.append(
                f"- **{c['ticker']}** ({c['rebalance_action']}, "
                f"confidence {c['confidence_label']}, conviction {c['conviction_label']}): "
                f"{c['rationale_short']}"
            )
        lines.append("")

    if op["blocked_candidates"]:
        lines.append("### Blocked Candidates")
        lines.append("")
        for c in op["blocked_candidates"]:
            lines.append(f"- **{c['ticker']}** ({c['rebalance_action']}): {c['rationale_short']}")
        lines.append("")

    # Risk Notes
    lines.append("## Risk Notes")
    lines.append("")
    lines.append(
        f"_Risk-clean ratio: {_fmt_score(rn['risk_clean_ratio'])} "
        f"({rn['n_symbols_with_risk_flag']} flagged / "
        f"{rn['n_symbols_evaluated']} evaluated)._"
    )
    lines.append("")
    lines.append(
        f"- **FORCE_EXIT:** "
        + (", ".join(rn["force_exit_symbols"]) if rn["force_exit_symbols"] else "_none_")
    )
    lines.append(
        f"- **TRIM_PRIORITY:** "
        + (", ".join(rn["trim_priority_symbols"]) if rn["trim_priority_symbols"] else "_none_")
    )
    lines.append(
        f"- **BLOCK_NEW_BUY:** "
        + (", ".join(rn["block_new_buy_symbols"]) if rn["block_new_buy_symbols"] else "_none_")
    )
    lines.append("")

    if rn["lifecycle_contradictions"]:
        lines.append("### Lifecycle Contradictions")
        lines.append("")
        for c in rn["lifecycle_contradictions"]:
            lines.append(
                f"- **{c['ticker']}**: trade {c['rebalance_action']} contradicts "
                f"lifecycle {c['lifecycle_action']} -- {c['rationale_short']}"
            )
        lines.append("")

    if rn["concentration_warnings"]:
        lines.append("### Concentration Warnings")
        lines.append("")
        for w in rn["concentration_warnings"]:
            lines.append(
                f"- **{w['severity']}** -- {w['type']} `{w['name']}` at "
                f"{_fmt_pct(w['weight_pct'])}"
            )
        lines.append("")

    # CIO Narrative
    lines.append("## CIO Narrative")
    lines.append("")
    lines.append(full_report["cio_narrative"])
    lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only investment committee report engine (Step 9 of WATCH funnel)."
            " Synthesises all upstream artefacts into a CIO-grade memo."
        ),
    )
    p.add_argument("--rationale", default=str(DEFAULT_RATIONALE_CSV))
    p.add_argument("--execution-summary", default=str(DEFAULT_EXECUTION_SUMMARY))
    p.add_argument("--construction-summary", default=str(DEFAULT_CONSTRUCTION_SUMMARY))
    p.add_argument("--rebalance-summary", default=str(DEFAULT_REBALANCE_SUMMARY))
    p.add_argument("--persistence-summary", default=str(DEFAULT_PERSISTENCE_SUMMARY))
    p.add_argument("--deployment-summary", default=str(DEFAULT_DEPLOYMENT_SUMMARY))
    p.add_argument("--risk-overlay", default=str(DEFAULT_RISK_OVERLAY_CSV))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary-json", default=str(DEFAULT_OUT_SUMMARY_JSON))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[INVESTMENT_COMMITTEE] starting (read-only synthesis engine)", flush=True)

    rationale_df = _safe_read_csv(Path(args.rationale), label="trade_rationale.csv")
    execution_summary = _safe_read_json(
        Path(args.execution_summary), label="portfolio_execution_summary.json"
    )
    construction_summary = _safe_read_json(
        Path(args.construction_summary), label="portfolio_construction_summary.json"
    )
    rebalance_summary = _safe_read_json(
        Path(args.rebalance_summary), label="portfolio_rebalance_summary.json"
    )
    persistence_summary = _safe_read_json(
        Path(args.persistence_summary), label="opportunity_persistence_summary.json"
    )
    deployment_summary = _safe_read_json(
        Path(args.deployment_summary), label="capital_deployment_summary.json"
    )
    risk_df = _safe_read_csv(Path(args.risk_overlay), label="performance_risk_overlay.csv")

    rationale_rows = _rationale_dataframe_to_rows(rationale_df)
    risk_overlay_map = _risk_overlay_lookup(risk_df)

    full_report, compact_summary = build_report(
        rationale_rows=rationale_rows,
        execution_summary=execution_summary,
        construction_summary=construction_summary,
        rebalance_summary=rebalance_summary,
        persistence_summary=persistence_summary,
        deployment_summary=deployment_summary,
        risk_overlay_map=risk_overlay_map,
    )

    md_text = render_markdown(full_report)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_summary_json = Path(args.out_summary_json)

    try:
        _atomic_write_json(full_report, out_json)
    except Exception as e:
        _warn(f"failed to write {out_json}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_text(md_text, out_md)
    except Exception as e:
        _warn(f"failed to write {out_md}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(compact_summary, out_summary_json)
    except Exception as e:
        _warn(f"failed to write {out_summary_json}: {type(e).__name__}: {e}")
        return 2

    scores = compact_summary["scores"]
    counts = compact_summary["intent_counts"]
    print(
        "[INVESTMENT_COMMITTEE] "
        f"execute={counts['execute_now']} "
        f"delay={counts['delay']} "
        f"block={counts['block']} "
        f"skip={counts['skip']} "
        f"health_score={scores['portfolio_health_score']:.3f} "
        f"readiness={scores['deployment_readiness_score']:.3f}",
        flush=True,
    )
    print(
        f"[INVESTMENT_POSTURE] posture={compact_summary['market_posture']} "
        f"stance={compact_summary['deployment_stance']} "
        f"conviction={compact_summary['portfolio_conviction']}",
        flush=True,
    )
    print(
        f"[INVESTMENT_COMMITTEE_OUT] json={out_json.as_posix()} "
        f"md={out_md.as_posix()} summary={out_summary_json.as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
