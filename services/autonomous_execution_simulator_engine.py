"""
Autonomous Execution Simulator Engine -- Step 25.

Reads:
    data/results/autonomous_execution_plan.json           (Step 24)
    data/results/autonomous_execution_plan_summary.json   (Step 24)
    data/results/runtime_policy_governed.json             (Step 18)
    data/results/positions_snapshot.csv                   (existing)
    data/results/portfolio_history.csv                    (existing)
    data/results/autonomous_committee_summary.json        (Step 15)
    data/results/adaptive_regime.json                     (Step 10)

Writes:
    data/results/autonomous_execution_simulation.json
    data/results/autonomous_execution_simulation.md
    data/results/autonomous_execution_simulation_summary.json

Purpose
-------
Step 24 produced an authorized execution plan. Step 25 is the
"what-if" engine that projects the effect of executing that plan
on the portfolio:

    "What would happen if Triton executed this plan?"

It is *strictly* a simulator -- no broker calls, no portfolio
mutation, no orders. The output is a verdict (SAFE / SAFE_LIMITED /
WARNING / UNSAFE / BLOCKED), a set of projected portfolio metrics,
a list of policy violations, and operator-actionable recommendations.

Conservation assumption
-----------------------
Total NAV is treated as conserved across the simulated execution
(execution costs ignored). Buys move USD from cash into a position;
sells/trims move USD back. This is a deliberate first-order
approximation -- it lets a downstream operator validate plan shape
and policy compliance without modelling fills, slippage, or
commission.

Risk checks (spec section 2)
----------------------------
Each check returns a ``CheckResult`` tagged severity ``critical``
or ``warning``:

    concentration_risk          critical -- any projected position weight
                                              above max_position_pct
    insufficient_cash_buffer    critical -- projected cash below 50% of
                                              target_cash_pct
    excessive_deployment        warning  -- deployment_pct > 10%
    elevated_turnover           warning  -- turnover_pct > 20%
    approaching_cash_floor      warning  -- projected cash below 100% of
                                              target_cash_pct but not yet
                                              insufficient
    defensive_policy_violation  critical -- regime DEFENSIVE but
                                              deployment_pct > 5%
    low_diversification         warning  -- projected_position_count < 5

Verdict cascade (spec section 3)
--------------------------------
    BLOCKED       no execution plan or no authorized actions
    UNSAFE        >=1 critical violation
    WARNING       >=2 warning-only violations
    SAFE_LIMITED  exactly 1 warning-only violation
    SAFE          zero violations

Safety
------
* READ ONLY. Absolutely no broker calls, no portfolio mutation.
  The words "place_order" and "submit" do not appear.
* Atomic writes (.tmp + os.replace).
* Missing inputs warn-and-continue. With no plan or no positions
  the verdict is BLOCKED.
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_PLAN_JSON = RESULTS_DIR / "autonomous_execution_plan.json"
DEFAULT_PLAN_SUMMARY = RESULTS_DIR / "autonomous_execution_plan_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_POSITIONS_CSV = RESULTS_DIR / "positions_snapshot.csv"
DEFAULT_HISTORY_CSV = RESULTS_DIR / "portfolio_history.csv"
DEFAULT_COMMITTEE_SUMMARY = RESULTS_DIR / "autonomous_committee_summary.json"
DEFAULT_REGIME = RESULTS_DIR / "adaptive_regime.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "autonomous_execution_simulation.json"
DEFAULT_OUT_MD = RESULTS_DIR / "autonomous_execution_simulation.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "autonomous_execution_simulation_summary.json"


# -----------------------------------------------------------
# Tunables
# -----------------------------------------------------------
VERDICT_SAFE = "SAFE"
VERDICT_SAFE_LIMITED = "SAFE_LIMITED"
VERDICT_WARNING = "WARNING"
VERDICT_UNSAFE = "UNSAFE"
VERDICT_BLOCKED = "BLOCKED"

ALL_VERDICTS: Tuple[str, ...] = (
    VERDICT_BLOCKED,
    VERDICT_UNSAFE,
    VERDICT_WARNING,
    VERDICT_SAFE_LIMITED,
    VERDICT_SAFE,
)

SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"

# Risk-check thresholds (independent of runtime policy where the
# runtime policy is silent; otherwise we defer to it)
DEPLOYMENT_PCT_WARNING_FLOOR = 10.0  # >10% NAV deployed in a cycle = warning
TURNOVER_PCT_WARNING_FLOOR = 20.0  # >20% turnover in a cycle = warning
DEFENSIVE_DEPLOYMENT_CAP_PCT = 5.0  # defensive regime cap = 5% NAV
INSUFFICIENT_CASH_RATIO = 0.50  # projected cash < 50% of target
LOW_DIVERSIFICATION_MIN = 5  # projected positions < 5 = warning

# Defensive regimes
DEFENSIVE_REGIMES = {"DEFENSIVE", "RISK_OFF", "HIGH_VOLATILITY", "CAPITAL_PRESERVATION"}

# Default thresholds when runtime policy is silent
DEFAULT_MAX_POSITION_PCT = 8.0
DEFAULT_TARGET_CASH_PCT = 20.0

# Simulation confidence weights (spec section 6)
CONFIDENCE_WEIGHTS = {
    "plan_confidence": 0.40,
    "runtime_freshness": 0.20,
    "policy_compliance": 0.25,
    "portfolio_health": 0.15,
}


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[EXECUTION_SIM_WARN] {msg}", flush=True)


def _safe_read_json(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        if not path.is_file():
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


def _safe_read_csv_rows(path: Path, *, label: str) -> List[Dict[str, str]]:
    try:
        if not path.is_file():
            return []
    except OSError as e:
        _warn(f"stat failed for {label} ({path}): {type(e).__name__}: {e}")
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception as e:
        _warn(f"failed to parse {label} ({path}): {type(e).__name__}: {e}")
        return []


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
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _norm_symbol(s: Any) -> str:
    return str(s or "").strip().upper()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _first(row: Dict[str, Any], keys: Iterable[str], *, default: Any = None) -> Any:
    for k in keys:
        if k in row:
            v = row[k]
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() not in ("nan", "none", "null"):
                return v
    return default


# -----------------------------------------------------------
# Portfolio state extraction
# -----------------------------------------------------------
def _parse_positions(
    rows: List[Dict[str, str]],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Returns (ticker -> market_value, ticker -> sector)."""
    positions: Dict[str, float] = {}
    sectors: Dict[str, str] = {}
    for r in rows:
        t = _norm_symbol(_first(r, ["symbol", "ticker"]))
        if not t:
            continue
        mv = _to_float(
            _first(
                r,
                [
                    "market_value",
                    "equity",
                    "current_value",
                    "position_value",
                    "value_usd",
                ],
            )
        )
        if mv is None or mv <= 0:
            continue
        positions[t] = mv
        sec = _first(r, ["sector", "sector_bucket"], default="UNKNOWN")
        sectors[t] = str(sec or "UNKNOWN").strip().upper()
    return positions, sectors


def _extract_nav_cash(
    positions: Dict[str, float],
    history_rows: List[Dict[str, str]],
    policy: Dict[str, Any],
) -> Tuple[float, float, str]:
    """
    Returns (nav, cash, source_tag). Prefers explicit nav/cash from
    the most recent portfolio_history row; else estimates cash via
    the runtime policy's target_cash_pct.
    """
    total_positions = sum(positions.values())
    nav = 0.0
    cash = 0.0
    source = "estimated_from_policy"

    if history_rows:
        last = history_rows[-1]
        nav_v = _to_float(
            _first(
                last,
                [
                    "nav",
                    "total_portfolio_value",
                    "portfolio_value",
                    "equity",
                ],
            )
        )
        cash_v = _to_float(
            _first(
                last,
                [
                    "cash",
                    "available_cash",
                    "cash_usd",
                    "cash_balance",
                ],
            )
        )
        if nav_v is not None and nav_v > 0:
            nav = nav_v
            source = "portfolio_history"
            if cash_v is not None:
                cash = max(0.0, cash_v)
            else:
                cash = max(0.0, nav - total_positions)

    if nav <= 0:
        target_cash_pct = _to_float(policy.get("target_cash_pct")) or DEFAULT_TARGET_CASH_PCT
        # Assume positions are (100 - target_cash_pct)% of NAV
        invested_pct = max(1.0, 100.0 - target_cash_pct)
        if total_positions > 0:
            nav = total_positions / (invested_pct / 100.0)
            cash = max(0.0, nav - total_positions)
            source = "estimated_from_policy"
        else:
            # No positions at all -- nav unknown
            nav = 0.0
            cash = 0.0
            source = "unknown"

    return nav, cash, source


# -----------------------------------------------------------
# Plan -> projected state
# -----------------------------------------------------------
def _project_state(
    *,
    plan: Dict[str, Any],
    current_positions: Dict[str, float],
    current_cash: float,
    current_sectors: Dict[str, str],
) -> Dict[str, Any]:
    """Apply each *authorized* action to a copy of the portfolio."""
    projected = dict(current_positions)
    projected_cash = current_cash
    deployment_total = 0.0
    exit_total = 0.0
    turnover_total = 0.0
    buy_count = 0
    sell_count = 0
    new_positions = 0

    actions = plan.get("actions") or []
    for action in actions:
        if not action.get("allowed"):
            continue
        ticker = _norm_symbol(action.get("ticker"))
        if not ticker:
            continue
        notional = _to_float(action.get("estimated_notional_usd")) or 0.0
        amt = abs(notional)
        kind = str(action.get("action") or "").strip().lower()
        if kind in ("buy_new", "add"):
            prev = projected.get(ticker, 0.0)
            projected[ticker] = prev + amt
            projected_cash -= amt
            deployment_total += amt
            turnover_total += amt
            buy_count += 1
            if prev <= 0.0:
                new_positions += 1
        elif kind in ("sell", "trim"):
            prev = projected.get(ticker, 0.0)
            reduce = min(prev, amt)
            projected[ticker] = max(0.0, prev - reduce)
            projected_cash += reduce
            exit_total += reduce
            turnover_total += reduce
            sell_count += 1
        # rotation is treated as zero-net-cash (sell+buy combo); for
        # our first-order simulator we let the underlying buy/sell
        # action classes already in the plan carry the effect.

    # Drop positions that round to zero
    projected = {t: v for t, v in projected.items() if v > 1.0}
    projected_nav = sum(projected.values()) + max(0.0, projected_cash)
    if projected_nav <= 0:
        projected_nav = sum(current_positions.values()) + max(0.0, current_cash)

    return {
        "projected_positions": projected,
        "projected_cash": round(projected_cash, 2),
        "projected_nav": round(projected_nav, 2),
        "deployment_total": round(deployment_total, 2),
        "exit_total": round(exit_total, 2),
        "turnover_total": round(turnover_total, 2),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "new_position_count": new_positions,
        "current_sectors": current_sectors,
    }


# -----------------------------------------------------------
# Projected metrics
# -----------------------------------------------------------
def _projected_metrics(
    projection: Dict[str, Any],
    *,
    current_nav: float,
    current_position_count: int,
) -> Dict[str, Any]:
    projected_positions: Dict[str, float] = projection["projected_positions"]
    projected_cash = float(projection["projected_cash"] or 0.0)

    # NAV conservation: the simulator treats total NAV as conserved
    # across the simulated execution (no execution-cost modelling).
    # We anchor the denominator to the starting NAV from history (or
    # the policy-estimated NAV) rather than re-summing
    # positions+cash -- the snapshot may not cover the full portfolio
    # (phantom assets), and re-summing would silently shrink NAV.
    if current_nav > 0:
        projected_nav = current_nav
    else:
        projected_nav = sum(projected_positions.values()) + max(0.0, projected_cash)

    pos_count = sum(1 for v in projected_positions.values() if v > 0.0)
    if projected_nav > 0:
        cash_pct = (projected_cash / projected_nav) * 100.0
        max_pos = max(projected_positions.values(), default=0.0)
        concentration_pct = (max_pos / projected_nav) * 100.0
        deployment_pct = (projection["deployment_total"] / projected_nav) * 100.0
        turnover_pct = (projection["turnover_total"] / projected_nav) * 100.0
    else:
        cash_pct = 0.0
        concentration_pct = 0.0
        deployment_pct = 0.0
        turnover_pct = 0.0

    # Sector exposure projection
    sectors = projection.get("current_sectors") or {}
    sector_exposure: Dict[str, float] = {}
    if projected_nav > 0:
        for ticker, value in projected_positions.items():
            sector = sectors.get(ticker, "UNKNOWN")
            sector_exposure[sector] = (
                sector_exposure.get(sector, 0.0) + (value / projected_nav) * 100.0
            )
    sector_exposure = {k: round(v, 2) for k, v in sector_exposure.items()}

    return {
        "projected_cash_pct": round(cash_pct, 4),
        "projected_position_count": pos_count,
        "projected_concentration_pct": round(concentration_pct, 4),
        "projected_sector_exposure_pct": sector_exposure,
        "projected_turnover_pct": round(turnover_pct, 4),
        "projected_deployment_pct": round(deployment_pct, 4),
        "projected_nav": round(projected_nav, 2),
        "projected_cash": round(projected_cash, 2),
        "current_nav": round(current_nav, 2),
        "current_position_count": current_position_count,
    }


# -----------------------------------------------------------
# Risk checks
# -----------------------------------------------------------
def _run_risk_checks(
    *,
    metrics: Dict[str, Any],
    policy: Dict[str, Any],
    regime: str,
    plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    max_pos_pct = _to_float(policy.get("max_position_pct")) or DEFAULT_MAX_POSITION_PCT
    target_cash_pct = _to_float(policy.get("target_cash_pct")) or DEFAULT_TARGET_CASH_PCT

    concentration = float(metrics["projected_concentration_pct"])
    cash_pct = float(metrics["projected_cash_pct"])
    deployment_pct = float(metrics["projected_deployment_pct"])
    turnover_pct = float(metrics["projected_turnover_pct"])
    pos_count = int(metrics["projected_position_count"])

    # 1. Concentration
    if concentration > max_pos_pct:
        checks.append(
            {
                "name": "concentration_risk",
                "severity": SEVERITY_CRITICAL,
                "detail": f"projected_concentration {concentration:.2f}% > max_position_pct {max_pos_pct:.2f}%",
                "violated": True,
            }
        )

    # 2. Insufficient cash buffer
    cash_floor_critical = INSUFFICIENT_CASH_RATIO * target_cash_pct
    if cash_pct < cash_floor_critical:
        checks.append(
            {
                "name": "insufficient_cash_buffer",
                "severity": SEVERITY_CRITICAL,
                "detail": (
                    f"projected_cash {cash_pct:.2f}% below 50% of target "
                    f"{target_cash_pct:.2f}% (floor {cash_floor_critical:.2f}%)"
                ),
                "violated": True,
            }
        )
    elif cash_pct < target_cash_pct:
        checks.append(
            {
                "name": "approaching_cash_floor",
                "severity": SEVERITY_WARNING,
                "detail": (
                    f"projected_cash {cash_pct:.2f}% below target " f"{target_cash_pct:.2f}%"
                ),
                "violated": True,
            }
        )

    # 3. Excessive deployment (warning)
    if deployment_pct > DEPLOYMENT_PCT_WARNING_FLOOR:
        checks.append(
            {
                "name": "excessive_deployment",
                "severity": SEVERITY_WARNING,
                "detail": (
                    f"deployment {deployment_pct:.2f}% > warning floor "
                    f"{DEPLOYMENT_PCT_WARNING_FLOOR:.2f}%"
                ),
                "violated": True,
            }
        )

    # 4. Elevated turnover (warning)
    if turnover_pct > TURNOVER_PCT_WARNING_FLOOR:
        checks.append(
            {
                "name": "elevated_turnover",
                "severity": SEVERITY_WARNING,
                "detail": (
                    f"turnover {turnover_pct:.2f}% > warning floor "
                    f"{TURNOVER_PCT_WARNING_FLOOR:.2f}%"
                ),
                "violated": True,
            }
        )

    # 5. Defensive policy violation (critical when regime defensive)
    if regime in DEFENSIVE_REGIMES and deployment_pct > DEFENSIVE_DEPLOYMENT_CAP_PCT:
        checks.append(
            {
                "name": "defensive_policy_violation",
                "severity": SEVERITY_CRITICAL,
                "detail": (
                    f"regime={regime} but deployment {deployment_pct:.2f}% > "
                    f"defensive cap {DEFENSIVE_DEPLOYMENT_CAP_PCT:.2f}%"
                ),
                "violated": True,
            }
        )

    # 6. Low diversification (warning)
    if pos_count > 0 and pos_count < LOW_DIVERSIFICATION_MIN:
        checks.append(
            {
                "name": "low_diversification",
                "severity": SEVERITY_WARNING,
                "detail": (
                    f"projected_position_count {pos_count} < " f"min {LOW_DIVERSIFICATION_MIN}"
                ),
                "violated": True,
            }
        )

    return checks


# -----------------------------------------------------------
# Verdict cascade
# -----------------------------------------------------------
def _classify_verdict(
    *,
    plan: Dict[str, Any],
    violations: List[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    mode = str(plan.get("execution_mode") or "").strip().upper()
    actions = plan.get("actions") or []
    n_authorized = sum(1 for a in actions if a.get("allowed"))

    # BLOCKED: no plan to simulate
    if mode == "NO_EXECUTION":
        reasons.append("execution_mode=NO_EXECUTION; nothing to simulate")
        return VERDICT_BLOCKED, reasons
    if n_authorized == 0:
        reasons.append("zero authorized actions in plan")
        return VERDICT_BLOCKED, reasons

    critical = [v for v in violations if v["severity"] == SEVERITY_CRITICAL]
    warnings = [v for v in violations if v["severity"] == SEVERITY_WARNING]

    if critical:
        reasons.append(
            f"{len(critical)} critical violation(s): " + ", ".join(v["name"] for v in critical)
        )
        return VERDICT_UNSAFE, reasons

    if len(warnings) >= 2:
        reasons.append(
            f"{len(warnings)} warning-level violations: " + ", ".join(v["name"] for v in warnings)
        )
        return VERDICT_WARNING, reasons
    if len(warnings) == 1:
        reasons.append(f"single warning: {warnings[0]['name']}")
        return VERDICT_SAFE_LIMITED, reasons

    reasons.append("all risk checks passed")
    return VERDICT_SAFE, reasons


# -----------------------------------------------------------
# Simulation confidence
# -----------------------------------------------------------
def _simulation_confidence(
    *,
    plan: Dict[str, Any],
    n_violations: int,
    n_checks_run: int,
    committee_health: float,
    runtime_freshness: float,
) -> Tuple[float, Dict[str, float]]:
    plan_conf = _to_float(plan.get("plan_confidence")) or 0.0
    if n_checks_run > 0:
        policy_compliance = 1.0 - (n_violations / float(n_checks_run))
    else:
        policy_compliance = 1.0
    contributors = {
        "plan_confidence": _clamp(plan_conf, 0.0, 1.0),
        "runtime_freshness": _clamp(runtime_freshness, 0.0, 1.0),
        "policy_compliance": _clamp(policy_compliance, 0.0, 1.0),
        "portfolio_health": _clamp(committee_health, 0.0, 1.0),
    }
    total_w = sum(CONFIDENCE_WEIGHTS.values()) or 1.0
    blended = sum(CONFIDENCE_WEIGHTS[k] * contributors[k] for k in CONFIDENCE_WEIGHTS) / total_w
    return _clamp(blended, 0.0, 1.0), {k: round(v, 6) for k, v in contributors.items()}


def _runtime_freshness_from_policy(policy: Dict[str, Any]) -> float:
    # Heuristic: the governed runtime policy carries a generated_at_utc
    # field via Step 18; if absent we cannot judge freshness and default
    # to neutral (0.5).
    if not policy.get("generated_at_utc"):
        return 0.5
    # We don't have "now" without resolving stat-time on the policy
    # file; emit neutral-positive 0.85 to indicate "policy looks present"
    # (Step 20's freshness monitor is authoritative; this contributor
    # is only a secondary signal here).
    return 0.85


def _committee_health(
    committee_summary: Dict[str, Any],
    plan: Dict[str, Any],
) -> float:
    health = _to_float((committee_summary or {}).get("portfolio_health_score"))
    if health is not None:
        return _clamp(health, 0.0, 1.0)
    # Fall back to plan_confidence as a weak proxy
    pc = _to_float(plan.get("plan_confidence"))
    if pc is not None:
        return _clamp(pc, 0.0, 1.0)
    return 0.50


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _build_recommendations(
    *,
    verdict: str,
    violations: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    plan: Dict[str, Any],
    regime: str,
) -> List[str]:
    recs: List[str] = []
    names = {v["name"] for v in violations}

    if verdict == VERDICT_BLOCKED:
        recs.append(
            "No execution to simulate -- treat plan as advisory only "
            "and re-run after the next authorized cycle."
        )
        return recs

    if verdict == VERDICT_SAFE:
        recs.append("Execution simulation safe -- plan respects all policy constraints.")
        recs.append("Proceed only after a future execution engine is authorised and connected.")
        return recs

    if verdict == VERDICT_SAFE_LIMITED:
        only = next(iter(violations)) if violations else None
        recs.append("Execution safe for limited deployment -- single warning recorded.")
        if only:
            recs.append(f"Address {only['name']} before scaling deployment further.")

    if verdict == VERDICT_WARNING:
        recs.append(
            "Policy pressure elevated -- reduce deployment scale or wait for "
            "next cycle before honouring this plan."
        )
    if verdict == VERDICT_UNSAFE:
        recs.append(
            "Multiple policy violations -- do not honour this plan. "
            "Operator review required before any future execution."
        )

    # Per-violation targeted hints
    if "concentration_risk" in names:
        recs.append(
            "Trim concentration before any new buys -- projected max position "
            f"is {metrics['projected_concentration_pct']:.2f}% of NAV."
        )
    if "insufficient_cash_buffer" in names:
        recs.append("Reduce deployment by 25% (or more) to restore the cash buffer.")
    if "approaching_cash_floor" in names:
        recs.append(
            f"Maintain defensive cash target -- projected cash "
            f"{metrics['projected_cash_pct']:.2f}% is below the target."
        )
    if "defensive_policy_violation" in names:
        recs.append(
            f"Defensive regime {regime} caps single-cycle deployment at "
            f"{DEFENSIVE_DEPLOYMENT_CAP_PCT:.0f}% NAV; reduce buys until "
            "regime softens."
        )
    if "excessive_deployment" in names:
        recs.append("Stage deployment across multiple cycles instead of one large burst.")
    if "elevated_turnover" in names:
        recs.append(
            "Reduce turnover -- avoid simultaneously trimming and adding " "overlapping exposures."
        )
    if "low_diversification" in names:
        recs.append(
            f"Projected position count is below {LOW_DIVERSIFICATION_MIN}; "
            "broaden the candidate set before deploying."
        )
    return recs


# -----------------------------------------------------------
# Markdown report
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    verdict: str,
    verdict_reasons: List[str],
    plan: Dict[str, Any],
    metrics: Dict[str, Any],
    violations: List[Dict[str, Any]],
    risk_checks: List[Dict[str, Any]],
    recommendations: List[str],
    confidence: float,
    contributors: Dict[str, float],
    regime: str,
    nav_source: str,
) -> str:
    def fmt_money(v: Optional[float]) -> str:
        return f"${v:,.0f}" if v is not None else "-"

    lines: List[str] = []
    lines.append("# Triton Execution Simulation")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_")
    lines.append("")

    lines.append("## Simulation Verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append(f"- plan execution_mode: {plan.get('execution_mode', 'UNKNOWN')}")
    lines.append(
        f"- plan authorized_actions: {(plan.get('aggregates') or {}).get('authorized_actions', 0)}"
    )
    lines.append(f"- regime: {regime or 'UNKNOWN'}")
    lines.append(f"- simulation_confidence: **{confidence:.3f}**")
    lines.append(f"- nav source: {nav_source}")
    for r in verdict_reasons:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Projected Portfolio Effects")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| projected_nav | {fmt_money(metrics['projected_nav'])} |")
    lines.append(f"| projected_cash | {fmt_money(metrics['projected_cash'])} |")
    lines.append(f"| projected_cash_pct | {metrics['projected_cash_pct']:.2f}% |")
    lines.append(f"| projected_position_count | {metrics['projected_position_count']} |")
    lines.append(f"| projected_concentration_pct | {metrics['projected_concentration_pct']:.2f}% |")
    lines.append(f"| projected_deployment_pct | {metrics['projected_deployment_pct']:.2f}% |")
    lines.append(f"| projected_turnover_pct | {metrics['projected_turnover_pct']:.2f}% |")
    lines.append("")
    sectors = metrics.get("projected_sector_exposure_pct") or {}
    if sectors:
        lines.append("**Projected sector exposure (%):**")
        lines.append("")
        lines.append("| sector | weight |")
        lines.append("|---|---|")
        for s, w in sorted(sectors.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {s} | {w:.2f}% |")
        lines.append("")

    lines.append("## Risk Checks")
    lines.append("")
    if risk_checks:
        lines.append("| check | severity | result | detail |")
        lines.append("|---|---|---|---|")
        for c in risk_checks:
            lines.append(
                f"| {c['name']} | {c['severity']} | "
                f"{'VIOLATED' if c.get('violated') else 'PASS'} | "
                f"{c.get('detail', '-')} |"
            )
    else:
        lines.append("_(no risk checks triggered)_")
    lines.append("")

    lines.append("## Violations")
    lines.append("")
    if violations:
        for v in violations:
            lines.append(f"- `{v['name']}` ({v['severity']}): {v.get('detail', '-')}")
    else:
        lines.append("_(none)_")
    lines.append("")

    lines.append("**Simulation confidence contributors:**")
    lines.append("")
    lines.append("| contributor | score | weight |")
    lines.append("|---|---|---|")
    for k, w in CONFIDENCE_WEIGHTS.items():
        lines.append(f"| {k} | {contributors[k]:.3f} | {w:.2f} |")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Narrative")
    lines.append("")
    if verdict == VERDICT_BLOCKED:
        narrative = (
            f"Simulation BLOCKED -- no authorised execution plan to project. "
            f"plan_mode={plan.get('execution_mode')}, "
            f"authorized={(plan.get('aggregates') or {}).get('authorized_actions', 0)}."
        )
    elif verdict == VERDICT_UNSAFE:
        narrative = (
            f"Simulation UNSAFE -- {len(violations)} violation(s) detected, "
            f"including {sum(1 for v in violations if v['severity'] == SEVERITY_CRITICAL)} critical. "
            f"Honour the recommendations before any future execution is considered."
        )
    else:
        narrative = (
            f"Simulation verdict {verdict}: projected deployment "
            f"{metrics['projected_deployment_pct']:.2f}% of NAV, "
            f"turnover {metrics['projected_turnover_pct']:.2f}% of NAV, "
            f"projected cash {metrics['projected_cash_pct']:.2f}% (target band intact). "
            f"Confidence {confidence:.2f}."
        )
    lines.append(narrative)
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_simulation(
    *,
    plan: Dict[str, Any],
    plan_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    positions_rows: List[Dict[str, str]],
    history_rows: List[Dict[str, str]],
    committee_summary: Dict[str, Any],
    regime_json: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    regime = (
        str((regime_json or {}).get("regime") or (runtime_policy or {}).get("regime") or "")
        .strip()
        .upper()
        or "UNKNOWN"
    )

    current_positions, current_sectors = _parse_positions(positions_rows)
    current_nav, current_cash, nav_source = _extract_nav_cash(
        current_positions, history_rows, runtime_policy
    )

    projection = _project_state(
        plan=plan,
        current_positions=current_positions,
        current_cash=current_cash,
        current_sectors=current_sectors,
    )
    metrics = _projected_metrics(
        projection,
        current_nav=current_nav,
        current_position_count=len(current_positions),
    )

    # First-pass classify: detect structural BLOCKED (no plan / no
    # authorized actions) before running risk checks. With no plan to
    # simulate, any policy violations would be attributable to the
    # *current* portfolio, not to the proposed execution -- so we
    # suppress them.
    pre_verdict, pre_reasons = _classify_verdict(plan=plan, violations=[])
    if pre_verdict == VERDICT_BLOCKED:
        risk_checks: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []
        verdict, verdict_reasons = pre_verdict, pre_reasons
    else:
        risk_checks = _run_risk_checks(
            metrics=metrics,
            policy=runtime_policy,
            regime=regime,
            plan=plan,
        )
        violations = [c for c in risk_checks if c.get("violated")]
        verdict, verdict_reasons = _classify_verdict(plan=plan, violations=violations)

    committee_health = _committee_health(committee_summary, plan)
    runtime_freshness = _runtime_freshness_from_policy(runtime_policy)
    confidence, contributors = _simulation_confidence(
        plan=plan,
        n_violations=len(violations),
        n_checks_run=max(1, len(risk_checks)),
        committee_health=committee_health,
        runtime_freshness=runtime_freshness,
    )

    recommendations = _build_recommendations(
        verdict=verdict,
        violations=violations,
        metrics=metrics,
        plan=plan,
        regime=regime,
    )

    now_iso = _now_iso_utc()
    simulation: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_execution_simulator_engine",
        "engine_version": 1,
        "simulation_verdict": verdict,
        "simulation_confidence": round(confidence, 6),
        "simulation_confidence_contributors": contributors,
        "simulation_confidence_weights": CONFIDENCE_WEIGHTS,
        "verdict_reasons": verdict_reasons,
        "plan_execution_mode": plan.get("execution_mode"),
        "plan_confidence": plan.get("plan_confidence"),
        "regime": regime,
        "nav_source": nav_source,
        "projected_metrics": metrics,
        "risk_checks": risk_checks,
        "violations": violations,
        "n_violations": len(violations),
        "n_critical_violations": sum(1 for v in violations if v["severity"] == SEVERITY_CRITICAL),
        "n_warning_violations": sum(1 for v in violations if v["severity"] == SEVERITY_WARNING),
        "recommendations": recommendations,
        "thresholds": {
            "max_position_pct": _to_float((runtime_policy or {}).get("max_position_pct"))
            or DEFAULT_MAX_POSITION_PCT,
            "target_cash_pct": _to_float((runtime_policy or {}).get("target_cash_pct"))
            or DEFAULT_TARGET_CASH_PCT,
            "deployment_warning_floor_pct": DEPLOYMENT_PCT_WARNING_FLOOR,
            "turnover_warning_floor_pct": TURNOVER_PCT_WARNING_FLOOR,
            "defensive_deployment_cap_pct": DEFENSIVE_DEPLOYMENT_CAP_PCT,
            "insufficient_cash_ratio": INSUFFICIENT_CASH_RATIO,
            "low_diversification_min_count": LOW_DIVERSIFICATION_MIN,
        },
        "inputs_seen": {
            "autonomous_execution_plan": bool(plan),
            "autonomous_execution_plan_summary": bool(plan_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "positions_snapshot_rows": len(positions_rows),
            "portfolio_history_rows": len(history_rows),
            "autonomous_committee_summary": bool(committee_summary),
            "adaptive_regime": bool(regime_json),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_execution_simulator_engine",
        "simulation_verdict": verdict,
        "simulation_confidence": round(confidence, 6),
        "n_violations": len(violations),
        "n_critical_violations": sum(1 for v in violations if v["severity"] == SEVERITY_CRITICAL),
        "n_warning_violations": sum(1 for v in violations if v["severity"] == SEVERITY_WARNING),
        "projected_cash_pct": metrics["projected_cash_pct"],
        "projected_position_count": metrics["projected_position_count"],
        "projected_concentration_pct": metrics["projected_concentration_pct"],
        "projected_deployment_pct": metrics["projected_deployment_pct"],
        "projected_turnover_pct": metrics["projected_turnover_pct"],
        "plan_execution_mode": plan.get("execution_mode"),
        "regime": regime,
        "violation_names": [v["name"] for v in violations],
        "n_recommendations": len(recommendations),
    }

    md = _render_markdown(
        generated_at=now_iso,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
        plan=plan,
        metrics=metrics,
        violations=violations,
        risk_checks=risk_checks,
        recommendations=recommendations,
        confidence=confidence,
        contributors=contributors,
        regime=regime,
        nav_source=nav_source,
    )
    return simulation, summary, md


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only autonomous execution simulator (Step 25). Projects "
            "the effect of Step 24's authorized execution plan on the "
            "current portfolio, runs risk checks, emits a verdict from "
            "SAFE..BLOCKED, and produces operator-actionable recommendations. "
            "Places no orders and mutates no portfolio state."
        ),
    )
    p.add_argument("--plan", default=str(DEFAULT_PLAN_JSON))
    p.add_argument("--plan-summary", default=str(DEFAULT_PLAN_SUMMARY))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--positions", default=str(DEFAULT_POSITIONS_CSV))
    p.add_argument("--history", default=str(DEFAULT_HISTORY_CSV))
    p.add_argument("--committee-summary", default=str(DEFAULT_COMMITTEE_SUMMARY))
    p.add_argument("--regime", default=str(DEFAULT_REGIME))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[EXECUTION_SIM] starting (read-only what-if simulation)", flush=True)

    plan = _safe_read_json(Path(args.plan), label="autonomous_execution_plan.json")
    plan_summary = _safe_read_json(
        Path(args.plan_summary), label="autonomous_execution_plan_summary.json"
    )
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    positions_rows = _safe_read_csv_rows(Path(args.positions), label="positions_snapshot.csv")
    history_rows = _safe_read_csv_rows(Path(args.history), label="portfolio_history.csv")
    committee_summary = _safe_read_json(
        Path(args.committee_summary), label="autonomous_committee_summary.json"
    )
    regime_json = _safe_read_json(Path(args.regime), label="adaptive_regime.json")

    simulation, summary, md = build_simulation(
        plan=plan,
        plan_summary=plan_summary,
        runtime_policy=runtime_policy,
        positions_rows=positions_rows,
        history_rows=history_rows,
        committee_summary=committee_summary,
        regime_json=regime_json,
    )

    try:
        _atomic_write_json(simulation, Path(args.out_json))
    except Exception as e:
        _warn(f"failed to write {args.out_json}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_text(md, Path(args.out_md))
    except Exception as e:
        _warn(f"failed to write {args.out_md}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(summary, Path(args.out_summary))
    except Exception as e:
        _warn(f"failed to write {args.out_summary}: {type(e).__name__}: {e}")
        return 2

    print(
        "[EXECUTION_SIM] "
        f"verdict={simulation['simulation_verdict']} "
        f"violations={simulation['n_violations']} "
        f"confidence={simulation['simulation_confidence']:.3f} "
        f"deployment={simulation['projected_metrics']['projected_deployment_pct']:.2f}%",
        flush=True,
    )
    if simulation["violations"]:
        names = ",".join(v["name"] for v in simulation["violations"])
        print(f"[EXECUTION_SIM_VIOLATIONS] {names}", flush=True)
    print(
        f"[EXECUTION_SIM_OUT] json={Path(args.out_json).as_posix()} "
        f"md={Path(args.out_md).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
