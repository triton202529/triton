"""
Autonomous Readiness Engine -- Step 21.

Reads:
    data/results/autonomous_system_health_summary.json   (Step 20)
    data/results/autonomous_committee_summary.json       (Step 15)
    data/results/runtime_policy_governed.json            (Step 18)
    data/results/meta_decision_intelligence.json         (Step 13)
    data/results/governance_trust_feedback.json          (Step 17)
    data/results/autonomous_governance_summary.json      (Step 19)
    data/results/adaptive_regime.json                    (Step 10)
    data/results/portfolio_execution_summary.json        (Step 7)
    data/results/portfolio_execution_intents.csv         (Step 7)
    data/results/investment_committee_summary.json       (Step 9)

Writes:
    data/results/autonomous_readiness.json
    data/results/autonomous_readiness_summary.json
    data/results/autonomous_readiness.md

Purpose
-------
Step 20 (system health monitor) answers "is the intelligence stack
operational?". Step 21 answers the strategic question that sits
one level above it:

    "Should Triton actually deploy capital today?"

A HEALTHY system can be NOT_READY (no opportunities, weak conviction).
A STALE system gets READ_ONLY (allow analysis, no execution).
A CRITICAL system gets BLOCKED (no action at all).

Two orthogonal axes drive the five-state cascade:

    operational     strategic
    -----------     ---------
    BLOCKED         no action regardless of opportunities
    READ_ONLY       analysis only; refresh required first
    NOT_READY       healthy but nothing worth doing today
    READY_LIMITED   selective deployment under defensive constraint
    READY           full clear -- new buys + rebalance allowed

The engine also computes a 0-1 readiness_score (weighted blend of
seven contributors) and produces five spec-required gating booleans
(spec section 3) that downstream automation can consult directly
without parsing the state cascade.

Safety
------
* READ ONLY. No broker calls, no engine state mutation. The gating
  booleans are advisory; enforcement is each downstream engine's
  responsibility.
* Atomic writes (.tmp + os.replace) for all three outputs.
* Missing inputs warn-and-continue. With zero inputs the engine
  reports state=BLOCKED with rationale "no system_health summary
  available" -- safe default.
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
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"

DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_COMMITTEE_SUMMARY = RESULTS_DIR / "autonomous_committee_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_META_INTEL = RESULTS_DIR / "meta_decision_intelligence.json"
DEFAULT_GOV_FEEDBACK = RESULTS_DIR / "governance_trust_feedback.json"
DEFAULT_GOV_SUMMARY = RESULTS_DIR / "autonomous_governance_summary.json"
DEFAULT_REGIME = RESULTS_DIR / "adaptive_regime.json"
DEFAULT_EXEC_SUMMARY = RESULTS_DIR / "portfolio_execution_summary.json"
DEFAULT_EXEC_INTENTS_CSV = RESULTS_DIR / "portfolio_execution_intents.csv"
DEFAULT_IC_SUMMARY = RESULTS_DIR / "investment_committee_summary.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "autonomous_readiness.json"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_OUT_MD = RESULTS_DIR / "autonomous_readiness.md"


# -----------------------------------------------------------
# State constants & thresholds
# -----------------------------------------------------------
STATE_READY = "READY"
STATE_READY_LIMITED = "READY_LIMITED"
STATE_NOT_READY = "NOT_READY"
STATE_READ_ONLY = "READ_ONLY"
STATE_BLOCKED = "BLOCKED"

OPPORTUNITY_SATURATION = 10  # >=10 executable -> opportunity score = 1.0
MIN_RECOMMENDATION_CONF = 0.40  # below this, NOT_READY regardless of opportunities

# Health-status -> numeric mapping (Step 20 overall_status)
HEALTH_STATUS_NUMERIC: Dict[str, float] = {
    "HEALTHY": 1.00,
    "DEGRADED": 0.70,
    "STALE": 0.40,
    "CRITICAL": 0.10,
    "OFFLINE": 0.00,
}

# Trust-level -> numeric (matches Step 19's mapping)
TRUST_LEVEL_NUMERIC: Dict[str, float] = {
    "VERY_LOW": 0.10,
    "LOW": 0.30,
    "MODERATE": 0.55,
    "HIGH": 0.78,
    "VERY_HIGH": 0.92,
}

# Governance-trust-level -> numeric (Step 17 levels)
GOV_TRUST_LEVEL_NUMERIC: Dict[str, float] = {
    "COLLAPSED": 0.05,
    "WEAK": 0.30,
    "STABLE": 0.55,
    "STRONG": 0.78,
    "VERY_STRONG": 0.92,
}

# Regimes that force READY_LIMITED if otherwise READY
DEFENSIVE_REGIMES = {"DEFENSIVE", "RISK_OFF", "HIGH_VOLATILITY", "CAPITAL_PRESERVATION"}

# Committee decisions
DEPLOY_DECISIONS = {"DEPLOY_AGGRESSIVELY", "DEPLOY_SELECTIVELY"}
DEFENSIVE_COMMITTEE_DECISIONS = {
    "CAPITAL_PRESERVATION",
    "DEFENSIVE_ROTATION",
    "DELEVER",
}
HOLD_DECISIONS = {"HOLD"}

# Readiness-score contributor weights
SCORE_WEIGHTS: Dict[str, float] = {
    "system_health": 0.25,
    "committee_confidence": 0.15,
    "trust_level": 0.10,
    "governance_health": 0.10,
    "deployment_readiness": 0.15,
    "executable_opportunities": 0.15,
    "runtime_freshness": 0.10,
}


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[AUTONOMOUS_READINESS_WARN] {msg}", flush=True)


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
            reader = csv.DictReader(f)
            return [dict(r) for r in reader]
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


def _to_int(x: Any) -> int:
    v = _to_float(x)
    if v is None:
        return 0
    try:
        return int(v)
    except Exception:
        return 0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm01(x: Any, default: float = 0.50) -> float:
    v = _to_float(x)
    if v is None:
        return default
    return _clamp(v, 0.0, 1.0)


# -----------------------------------------------------------
# Contributor extractors
# -----------------------------------------------------------
def _count_execute_now(
    *,
    exec_summary: Dict[str, Any],
    exec_intents_rows: List[Dict[str, str]],
) -> int:
    n = _to_int(
        exec_summary.get("execute_now")
        or exec_summary.get("n_execute_now")
        or exec_summary.get("execute_now_count")
    )
    if n > 0:
        return n
    if not exec_intents_rows:
        return 0
    count = 0
    for row in exec_intents_rows:
        intent = str(row.get("execution_intent") or row.get("intent") or "").strip().upper()
        if intent == "EXECUTE_NOW":
            count += 1
    return count


def _runtime_freshness_score(
    *,
    health_summary: Dict[str, Any],
) -> float:
    stale = set(map(str, health_summary.get("stale_artifacts") or []))
    missing = set(map(str, health_summary.get("missing_artifacts") or []))
    if "runtime_policy_governed" in missing:
        return 0.0
    if "runtime_policy_governed" in stale:
        return 0.30
    return 1.0


def _build_contributors(
    *,
    health_summary: Dict[str, Any],
    committee_summary: Dict[str, Any],
    meta_intel: Dict[str, Any],
    feedback: Dict[str, Any],
    gov_summary: Dict[str, Any],
    ic_summary: Dict[str, Any],
    exec_summary: Dict[str, Any],
    exec_intents_rows: List[Dict[str, str]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Returns (contributors, raw_inputs) where contributors is the 0-1
    score per contributor and raw_inputs is the underlying values for
    provenance / debugging.
    """
    health_status = str((health_summary or {}).get("overall_status") or "OFFLINE").strip().upper()
    health_score = HEALTH_STATUS_NUMERIC.get(health_status, 0.0)

    committee_conf = _norm01(
        (committee_summary or {}).get("recommendation_confidence"),
        default=0.50,
    )

    trust_level = str((meta_intel or {}).get("trust_level") or "MODERATE").strip().upper()
    trust_score = TRUST_LEVEL_NUMERIC.get(trust_level, 0.55)

    # Governance health: prefer Step 17 governance_health_score when active,
    # else fall back to Step 19 governance_quality headline.
    gov_active = bool((feedback or {}).get("active", False))
    if gov_active:
        gov_health = _norm01((feedback or {}).get("governance_health_score"))
    else:
        gov_health = _norm01(
            ((gov_summary or {}).get("headline_scores") or {}).get("governance_quality"),
            default=0.50,
        )

    deployment_readiness = _norm01(
        (ic_summary or {}).get("deployment_readiness_score"),
        default=0.50,
    )

    execute_now = _count_execute_now(exec_summary=exec_summary, exec_intents_rows=exec_intents_rows)
    opp_score = _clamp(execute_now / float(OPPORTUNITY_SATURATION), 0.0, 1.0)

    runtime_fresh = _runtime_freshness_score(health_summary=health_summary)

    contributors: Dict[str, float] = {
        "system_health": round(health_score, 6),
        "committee_confidence": round(committee_conf, 6),
        "trust_level": round(trust_score, 6),
        "governance_health": round(gov_health, 6),
        "deployment_readiness": round(deployment_readiness, 6),
        "executable_opportunities": round(opp_score, 6),
        "runtime_freshness": round(runtime_fresh, 6),
    }
    raw_inputs: Dict[str, Any] = {
        "health_status": health_status,
        "trust_level": trust_level,
        "recommendation_confidence": committee_conf,
        "executable_opportunities_count": execute_now,
        "governance_feedback_active": gov_active,
    }
    return contributors, raw_inputs


def _readiness_score(contributors: Dict[str, float]) -> float:
    total_w = 0.0
    weighted = 0.0
    for k, w in SCORE_WEIGHTS.items():
        v = contributors.get(k, 0.0)
        weighted += w * _clamp(v, 0.0, 1.0)
        total_w += w
    if total_w <= 0.0:
        return 0.0
    return _clamp(weighted / total_w, 0.0, 1.0)


# -----------------------------------------------------------
# State cascade
# -----------------------------------------------------------
def _classify_state(
    *,
    contributors: Dict[str, float],
    raw: Dict[str, Any],
    health_summary: Dict[str, Any],
    committee_summary: Dict[str, Any],
    feedback: Dict[str, Any],
    regime_json: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """
    Strict precedence cascade. The first matching rule wins so that
    severe operational failures cannot be overridden by strong
    strategic signals (a BLOCKED state must never silently degrade
    to READY just because there are opportunities).
    """
    reasons: List[str] = []
    health_status = raw.get("health_status") or "OFFLINE"
    health_blocking = list(map(str, (health_summary or {}).get("blocking_flags") or []))
    deployment_allowed_by_health = bool(
        (health_summary or {}).get("autonomous_deployment_allowed", False)
    )

    committee_decision = (
        str(
            (committee_summary or {}).get("decision")
            or (committee_summary or {}).get("committee_decision")
            or ""
        )
        .strip()
        .upper()
    )
    governance_trust = (
        str((feedback or {}).get("governance_trust_level") or "STABLE").strip().upper()
    )
    regime = str((regime_json or {}).get("regime") or "").strip().upper()

    execute_now = int(raw.get("executable_opportunities_count") or 0)
    recommendation_conf = float(raw.get("recommendation_confidence") or 0.0)

    # ------------------------------------------------------
    # 1. BLOCKED -- severe operational or governance failure
    # ------------------------------------------------------
    if health_status in ("CRITICAL", "OFFLINE"):
        reasons.append(f"system health {health_status}")
        return STATE_BLOCKED, reasons
    if governance_trust == "COLLAPSED":
        reasons.append("governance trust COLLAPSED")
        return STATE_BLOCKED, reasons
    if "BLOCK_NEW_BUYS" in health_blocking and "BLOCK_AUTONOMOUS_DEPLOYMENT" in health_blocking:
        # Both flags imply the operational gate is fully closed.
        reasons.append("Step 20 set BLOCK_NEW_BUYS + BLOCK_AUTONOMOUS_DEPLOYMENT")
        return STATE_BLOCKED, reasons

    # ------------------------------------------------------
    # 2. READ_ONLY -- stale or refresh required
    # ------------------------------------------------------
    if health_status == "STALE":
        reasons.append("system health STALE -- pipeline refresh required")
        return STATE_READ_ONLY, reasons
    if not deployment_allowed_by_health:
        reasons.append(
            "Step 20 reports autonomous_deployment_allowed=False; " "downgraded to READ_ONLY"
        )
        return STATE_READ_ONLY, reasons
    if "REQUIRE_OPERATOR_REVIEW" in health_blocking:
        reasons.append("Step 20 set REQUIRE_OPERATOR_REVIEW")
        return STATE_READ_ONLY, reasons

    # At this point: operationally cleared. Strategic axis next.

    # ------------------------------------------------------
    # 3. NOT_READY -- system OK but no deployable case
    # ------------------------------------------------------
    has_opportunities = execute_now > 0
    conviction_adequate = recommendation_conf >= MIN_RECOMMENDATION_CONF

    if not has_opportunities:
        if committee_decision in HOLD_DECISIONS or committee_decision == "":
            reasons.append(
                f"committee={committee_decision or 'UNKNOWN'} with "
                f"zero executable opportunities"
            )
            return STATE_NOT_READY, reasons
        if committee_decision in DEFENSIVE_COMMITTEE_DECISIONS:
            reasons.append(
                f"committee={committee_decision} but no executable rebalance "
                f"actions to enforce it"
            )
            return STATE_NOT_READY, reasons
    if not conviction_adequate:
        reasons.append(
            f"recommendation_confidence {recommendation_conf:.2f} below floor "
            f"{MIN_RECOMMENDATION_CONF:.2f}"
        )
        return STATE_NOT_READY, reasons

    # ------------------------------------------------------
    # 4. READY_LIMITED -- selective deployment only
    # ------------------------------------------------------
    if regime in DEFENSIVE_REGIMES:
        reasons.append(f"regime={regime} forces selective deployment")
        return STATE_READY_LIMITED, reasons
    if committee_decision == "DEPLOY_SELECTIVELY":
        reasons.append("committee=DEPLOY_SELECTIVELY constrains breadth")
        return STATE_READY_LIMITED, reasons
    if committee_decision in DEFENSIVE_COMMITTEE_DECISIONS:
        reasons.append(f"committee={committee_decision} -- defensive rotation only")
        return STATE_READY_LIMITED, reasons

    # ------------------------------------------------------
    # 5. READY -- full clear
    # ------------------------------------------------------
    reasons.append(
        f"health={health_status}, committee={committee_decision}, "
        f"{execute_now} executable opportunities, "
        f"confidence {recommendation_conf:.2f}"
    )
    return STATE_READY, reasons


# -----------------------------------------------------------
# Gating booleans
# -----------------------------------------------------------
def _gating_booleans(state: str) -> Dict[str, bool]:
    """
    Spec section 3 booleans. Read-only is always allowed (matches
    Step 20's invariant). Rebalance is allowed in NOT_READY too,
    because trimming existing positions is a defensive action that
    doesn't depend on a "deployable case".
    """
    if state == STATE_READY:
        return {
            "autonomous_deployment_allowed": True,
            "new_buy_allowed": True,
            "rebalance_allowed": True,
            "read_only_allowed": True,
            "operator_review_required": False,
        }
    if state == STATE_READY_LIMITED:
        return {
            "autonomous_deployment_allowed": True,
            "new_buy_allowed": True,  # selective; Step 7 floors enforce
            "rebalance_allowed": True,
            "read_only_allowed": True,
            "operator_review_required": False,
        }
    if state == STATE_NOT_READY:
        return {
            "autonomous_deployment_allowed": False,
            "new_buy_allowed": False,
            "rebalance_allowed": True,  # trims/exits still allowed
            "read_only_allowed": True,
            "operator_review_required": False,
        }
    if state == STATE_READ_ONLY:
        return {
            "autonomous_deployment_allowed": False,
            "new_buy_allowed": False,
            "rebalance_allowed": False,
            "read_only_allowed": True,
            "operator_review_required": True,
        }
    # STATE_BLOCKED (and any unknown)
    return {
        "autonomous_deployment_allowed": False,
        "new_buy_allowed": False,
        "rebalance_allowed": False,
        "read_only_allowed": True,
        "operator_review_required": True,
    }


# -----------------------------------------------------------
# Rationale + recommendations
# -----------------------------------------------------------
def _rationale(
    *,
    state: str,
    reasons: List[str],
    score: float,
    contributors: Dict[str, float],
    raw: Dict[str, Any],
) -> Tuple[str, str]:
    bullet = "; ".join(reasons) if reasons else "no specific trigger recorded"
    rationale_short = (
        f"Triton is {state} (score {score:.2f}) -- {reasons[0] if reasons else 'see contributors'}."
    )
    weak = sorted(
        (k for k, v in contributors.items() if v < 0.40),
        key=lambda k: contributors[k],
    )
    strong = sorted(
        (k for k, v in contributors.items() if v >= 0.65),
        key=lambda k: -contributors[k],
    )
    rationale_long = (
        f"State {state} assigned because: {bullet}. "
        f"Composite readiness score is {score:.3f} "
        f"(committee_confidence={contributors['committee_confidence']:.2f}, "
        f"system_health={contributors['system_health']:.2f}, "
        f"executable_opportunities={contributors['executable_opportunities']:.2f}, "
        f"trust_level={contributors['trust_level']:.2f}, "
        f"governance_health={contributors['governance_health']:.2f}, "
        f"deployment_readiness={contributors['deployment_readiness']:.2f}, "
        f"runtime_freshness={contributors['runtime_freshness']:.2f}). "
        f"Weakest contributors: {weak or 'none'}. "
        f"Strongest contributors: {strong or 'none'}."
    )
    return rationale_short, rationale_long


def _recommendations(
    *,
    state: str,
    contributors: Dict[str, float],
    raw: Dict[str, Any],
    health_summary: Dict[str, Any],
) -> List[str]:
    recs: List[str] = []
    health_status = raw.get("health_status") or "OFFLINE"
    execute_now = int(raw.get("executable_opportunities_count") or 0)

    if state == STATE_BLOCKED:
        recs.append("Operator review required before any execution -- system is BLOCKED.")
        if health_status in ("CRITICAL", "OFFLINE"):
            recs.append("Refresh full pipeline (Steps 1..20) and re-run system health monitor.")
    elif state == STATE_READ_ONLY:
        recs.append("Refresh stale pipeline artifacts before resuming any deployment.")
        stale = sorted(map(str, (health_summary or {}).get("stale_artifacts") or []))
        if stale:
            recs.append(f"Regenerate stale artifacts: {', '.join(stale)}.")
        recs.append("Continue read-only monitoring while diagnostics refresh.")
    elif state == STATE_NOT_READY:
        recs.append(
            "Wait for stronger opportunities -- no executable rebalance "
            "actions clear the conviction floor today."
        )
        if execute_now == 0:
            recs.append("Continue monitoring; let watch candidates persist or strengthen.")
        if contributors["committee_confidence"] < 0.40:
            recs.append(
                "Committee confidence weak -- defer until next cycle re-evaluates "
                "deployment pressure."
            )
    elif state == STATE_READY_LIMITED:
        recs.append(
            "Restrict deployment to defensive assets -- regime/committee " "constrain breadth."
        )
        recs.append(
            "Honour position-size and cash-reserve floors from the governed runtime policy."
        )
    elif state == STATE_READY:
        recs.append("Proceed with full deployment plan -- system cleared all gates.")
        recs.append(
            "Continue monitoring -- re-evaluate readiness next cycle to catch any "
            "regime or governance shift."
        )

    # Per-contributor targeted hints (apply across multiple states)
    if contributors["runtime_freshness"] < 0.50:
        recs.append("Rebuild runtime policy (Steps 11 + 14 + 18) -- it is stale or missing.")
    if contributors["governance_health"] < 0.30:
        recs.append("Investigate governance health -- below the WEAK threshold.")
    if contributors["trust_level"] < 0.30:
        recs.append(
            "Meta trust LOW or VERY_LOW -- raise confidence floors and reduce position size."
        )

    # De-duplicate stable
    seen: List[str] = []
    for r in recs:
        if r not in seen:
            seen.append(r)
    return seen


# -----------------------------------------------------------
# Markdown report
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    score: float,
    rationale_short: str,
    rationale_long: str,
    contributors: Dict[str, float],
    raw: Dict[str, Any],
    gates: Dict[str, bool],
    recommendations: List[str],
) -> str:
    def _yes_no(b: bool) -> str:
        return "yes" if b else "no"

    lines: List[str] = []
    lines.append("# Triton Autonomous Readiness")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_")
    lines.append("")

    lines.append("## Readiness State")
    lines.append("")
    lines.append(f"**{state}**")
    lines.append("")
    lines.append(f"- system health: {raw.get('health_status')}")
    lines.append(f"- committee decision: {raw.get('committee_decision') or 'UNKNOWN'}")
    lines.append(f"- governance trust: {raw.get('governance_trust_level') or 'UNKNOWN'}")
    lines.append(f"- regime: {raw.get('regime') or 'UNKNOWN'}")
    lines.append(f"- executable opportunities: {raw.get('executable_opportunities_count')}")
    lines.append("")

    lines.append("## Readiness Score")
    lines.append("")
    lines.append(f"**{score:.3f}** (0.000 = unsafe, 1.000 = full clear)")
    lines.append("")
    lines.append("| contributor | score | weight |")
    lines.append("|---|---|---|")
    for k, w in SCORE_WEIGHTS.items():
        lines.append(f"| {k} | {contributors[k]:.3f} | {w:.2f} |")
    lines.append("")

    lines.append("## Why")
    lines.append("")
    lines.append(rationale_long)
    lines.append("")

    lines.append("## Deployment Permissions")
    lines.append("")
    lines.append("| permission | granted |")
    lines.append("|---|---|")
    for k in (
        "autonomous_deployment_allowed",
        "new_buy_allowed",
        "rebalance_allowed",
        "read_only_allowed",
        "operator_review_required",
    ):
        lines.append(f"| {k} | {_yes_no(gates[k])} |")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Narrative")
    lines.append("")
    lines.append(rationale_short)
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_readiness(
    *,
    health_summary: Dict[str, Any],
    committee_summary: Dict[str, Any],
    governed_policy: Dict[str, Any],
    meta_intel: Dict[str, Any],
    feedback: Dict[str, Any],
    gov_summary: Dict[str, Any],
    regime_json: Dict[str, Any],
    exec_summary: Dict[str, Any],
    exec_intents_rows: List[Dict[str, str]],
    ic_summary: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    contributors, raw = _build_contributors(
        health_summary=health_summary,
        committee_summary=committee_summary,
        meta_intel=meta_intel,
        feedback=feedback,
        gov_summary=gov_summary,
        ic_summary=ic_summary,
        exec_summary=exec_summary,
        exec_intents_rows=exec_intents_rows,
    )
    # Enrich raw with strategic context for the markdown renderer
    raw["committee_decision"] = (
        str(
            (committee_summary or {}).get("decision")
            or (committee_summary or {}).get("committee_decision")
            or ""
        )
        .strip()
        .upper()
    )
    raw["governance_trust_level"] = (
        str((feedback or {}).get("governance_trust_level") or "STABLE").strip().upper()
    )
    raw["regime"] = str((regime_json or {}).get("regime") or "").strip().upper()

    state, reasons = _classify_state(
        contributors=contributors,
        raw=raw,
        health_summary=health_summary,
        committee_summary=committee_summary,
        feedback=feedback,
        regime_json=regime_json,
    )
    score = _readiness_score(contributors)
    gates = _gating_booleans(state)
    rationale_short, rationale_long = _rationale(
        state=state,
        reasons=reasons,
        score=score,
        contributors=contributors,
        raw=raw,
    )
    recommendations = _recommendations(
        state=state,
        contributors=contributors,
        raw=raw,
        health_summary=health_summary,
    )

    now_iso = _now_iso_utc()
    readiness: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_readiness_engine",
        "engine_version": 1,
        "readiness_state": state,
        "readiness_score": round(score, 6),
        "rationale_short": rationale_short,
        "rationale_long": rationale_long,
        "reasons": reasons,
        "contributors": contributors,
        "contributor_weights": SCORE_WEIGHTS,
        "raw_inputs": raw,
        "gating": gates,
        "recommendations": recommendations,
        "thresholds": {
            "opportunity_saturation": OPPORTUNITY_SATURATION,
            "min_recommendation_confidence": MIN_RECOMMENDATION_CONF,
        },
        "inputs_seen": {
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_committee_summary": bool(committee_summary),
            "runtime_policy_governed": bool(governed_policy),
            "meta_decision_intelligence": bool(meta_intel),
            "governance_trust_feedback": bool(feedback),
            "autonomous_governance_summary": bool(gov_summary),
            "adaptive_regime": bool(regime_json),
            "portfolio_execution_summary": bool(exec_summary),
            "portfolio_execution_intents_rows": len(exec_intents_rows),
            "investment_committee_summary": bool(ic_summary),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": now_iso,
        "engine": "autonomous_readiness_engine",
        "readiness_state": state,
        "readiness_score": round(score, 6),
        "gating": gates,
        "rationale_short": rationale_short,
        "headline_contributors": contributors,
        "executable_opportunities_count": raw["executable_opportunities_count"],
        "committee_decision": raw["committee_decision"] or None,
        "regime": raw["regime"] or None,
        "n_recommendations": len(recommendations),
    }

    md = _render_markdown(
        generated_at=now_iso,
        state=state,
        score=score,
        rationale_short=rationale_short,
        rationale_long=rationale_long,
        contributors=contributors,
        raw=raw,
        gates=gates,
        recommendations=recommendations,
    )
    return readiness, summary, md


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only autonomous readiness engine (Step 21). "
            "Decides whether Triton should actually deploy capital today, "
            "given the operational health (Step 20) AND the strategic case "
            "(committee + trust + opportunities). Outputs a five-state "
            "label, 0-1 score, five gating booleans, and operator-actionable "
            "recommendations."
        ),
    )
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--committee-summary", default=str(DEFAULT_COMMITTEE_SUMMARY))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--meta-intel", default=str(DEFAULT_META_INTEL))
    p.add_argument("--gov-feedback", default=str(DEFAULT_GOV_FEEDBACK))
    p.add_argument("--gov-summary", default=str(DEFAULT_GOV_SUMMARY))
    p.add_argument("--regime", default=str(DEFAULT_REGIME))
    p.add_argument("--exec-summary", default=str(DEFAULT_EXEC_SUMMARY))
    p.add_argument("--exec-intents", default=str(DEFAULT_EXEC_INTENTS_CSV))
    p.add_argument("--ic-summary", default=str(DEFAULT_IC_SUMMARY))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print("[AUTONOMOUS_READINESS] starting (health + strategy -> readiness gate)", flush=True)

    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    committee_summary = _safe_read_json(
        Path(args.committee_summary), label="autonomous_committee_summary.json"
    )
    governed_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    meta_intel = _safe_read_json(Path(args.meta_intel), label="meta_decision_intelligence.json")
    feedback = _safe_read_json(Path(args.gov_feedback), label="governance_trust_feedback.json")
    gov_summary = _safe_read_json(
        Path(args.gov_summary), label="autonomous_governance_summary.json"
    )
    regime_json = _safe_read_json(Path(args.regime), label="adaptive_regime.json")
    exec_summary = _safe_read_json(
        Path(args.exec_summary), label="portfolio_execution_summary.json"
    )
    exec_intents = _safe_read_csv_rows(
        Path(args.exec_intents), label="portfolio_execution_intents.csv"
    )
    ic_summary = _safe_read_json(Path(args.ic_summary), label="investment_committee_summary.json")

    readiness, summary, md = build_readiness(
        health_summary=health_summary,
        committee_summary=committee_summary,
        governed_policy=governed_policy,
        meta_intel=meta_intel,
        feedback=feedback,
        gov_summary=gov_summary,
        regime_json=regime_json,
        exec_summary=exec_summary,
        exec_intents_rows=exec_intents,
        ic_summary=ic_summary,
    )

    try:
        _atomic_write_json(readiness, Path(args.out_json))
    except Exception as e:
        _warn(f"failed to write {args.out_json}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_json(summary, Path(args.out_summary))
    except Exception as e:
        _warn(f"failed to write {args.out_summary}: {type(e).__name__}: {e}")
        return 2
    try:
        _atomic_write_text(md, Path(args.out_md))
    except Exception as e:
        _warn(f"failed to write {args.out_md}: {type(e).__name__}: {e}")
        return 2

    g = readiness["gating"]
    print(
        "[AUTONOMOUS_READINESS] "
        f"state={readiness['readiness_state']} "
        f"score={readiness['readiness_score']:.3f} "
        f"deployment_allowed={g['autonomous_deployment_allowed']} "
        f"new_buy_allowed={g['new_buy_allowed']} "
        f"review_required={g['operator_review_required']}",
        flush=True,
    )
    print(
        f"[AUTONOMOUS_READINESS_OUT] json={Path(args.out_json).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()} "
        f"md={Path(args.out_md).as_posix()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
