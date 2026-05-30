"""
ARM Governance Doctrine Recommendation Engine -- Step 45.

Reads:
    data/results/arm_governance_doctrine_impact_assessment_summary.json   (Step 44)
    data/results/arm_governance_doctrine_impact_assessment.json           (Step 44)
    data/results/arm_governance_doctrine_impact_assessment_memory.csv   (Step 44)
    data/results/arm_governance_doctrine_simulation_summary.json        (Step 43)
    data/results/arm_governance_doctrine_activation_board_summary.json  (Step 42)
    data/results/arm_constitutional_court_summary.json                  (Step 33)
    data/results/arm_supreme_governance_council_summary.json            (Step 34)
    data/results/autonomous_governance_scorecard.json                   (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/runtime_policy_governed.json                           (Step 18)

Writes:
    data/results/arm_governance_doctrine_recommendation.json
    data/results/arm_governance_doctrine_recommendation.md
    data/results/arm_governance_doctrine_recommendation_summary.json
    data/results/arm_governance_doctrine_recommendation_memory.csv
    data/results/arm_governance_doctrine_recommendation_memory.parquet

Purpose
-------
This engine answers:

    "Which doctrine deserves future activation consideration?"

It converts governance doctrine impact assessments into institutional doctrine
recommendations. Beneficial != Recommended. Recommended != Activated.
Recommendations NEVER activate runtime policy. Constitutional law remains supreme.

Recommendation state cascade
----------------------------
    1. DOCTRINE_RECOMMENDATION_INSTITUTIONAL  mature recommendation process
    2. DOCTRINE_RECOMMENDATION_RESTRICTED     elevated pressure; defensive only
    3. DOCTRINE_RECOMMENDATION_ACTIVE          recommendations functioning normally
    4. DOCTRINE_RECOMMENDATION_FORMING         weak evidence; immature recommendations
    5. DOCTRINE_RECOMMENDATION_DORMANT         no impact evidence

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens are never written literally; an import-time
self-check raises if they ever appear.

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* runtime_mutation_allowed is ALWAYS false.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only recommendation memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to DOCTRINE_RECOMMENDATION_DORMANT.
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

DEFAULT_IMPACT_SUM = RESULTS_DIR / "arm_governance_doctrine_impact_assessment_summary.json"
DEFAULT_IMPACT_REC = RESULTS_DIR / "arm_governance_doctrine_impact_assessment.json"
DEFAULT_IMPACT_MEM = RESULTS_DIR / "arm_governance_doctrine_impact_assessment_memory.csv"
DEFAULT_SIMULATION_SUM = RESULTS_DIR / "arm_governance_doctrine_simulation_summary.json"
DEFAULT_ACTIVATION_SUM = RESULTS_DIR / "arm_governance_doctrine_activation_board_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_SUM = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_doctrine_recommendation.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_doctrine_recommendation.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_doctrine_recommendation_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_doctrine_recommendation_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_doctrine_recommendation_memory.parquet"


# -----------------------------------------------------------
# Recommendation state constants
# -----------------------------------------------------------
RECOMMENDATION_DORMANT = "DOCTRINE_RECOMMENDATION_DORMANT"
RECOMMENDATION_FORMING = "DOCTRINE_RECOMMENDATION_FORMING"
RECOMMENDATION_RESTRICTED = "DOCTRINE_RECOMMENDATION_RESTRICTED"
RECOMMENDATION_ACTIVE = "DOCTRINE_RECOMMENDATION_ACTIVE"
RECOMMENDATION_INSTITUTIONAL = "DOCTRINE_RECOMMENDATION_INSTITUTIONAL"

DECISION_RECOMMEND = "RECOMMEND_FOR_FUTURE_CONSIDERATION"
DECISION_MONITOR = "MONITOR_ONLY"
DECISION_DEFER = "DEFER"
DECISION_AVOID = "AVOID"
DECISION_OPERATOR = "OPERATOR_ESCALATION_REQUIRED"

IMPACT_BENEFICIAL = "BENEFICIAL"
IMPACT_NEUTRAL = "NEUTRAL"
IMPACT_HARMFUL = "HARMFUL"
IMPACT_UNCERTAIN = "UNCERTAIN"

HIGH_IMPACT_POLICIES = frozenset(
    {
        "confidence_threshold",
        "deployment_threshold",
        "autonomy_readiness_threshold",
        "min_observations_before_graduation",
        "auto_lock_manual_after_overruling",
        "skepticism_threshold",
        "persistence_threshold",
    }
)

DEFENSIVE_CAPITAL = frozenset({"target_cash_pct", "max_position_pct"})

RECOMMENDATION_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "recommendation_state",
    "recommended_count",
    "monitor_count",
    "deferred_count",
    "avoided_count",
    "operator_escalation_count",
    "recommendation_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_DOCTRINE_RECOMMENDATION_WARN] {msg}", flush=True)


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


def _atomic_write_csv(rows: List[Dict[str, Any]], path: Path, *, columns: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})
    os.replace(tmp, path)


def _atomic_write_parquet(rows: List[Dict[str, Any]], path: Path) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        _warn(f"pandas unavailable for parquet write: {type(e).__name__}: {e}")
        return False
    try:
        df = pd.DataFrame(rows, columns=list(RECOMMENDATION_MEMORY_COLUMNS))
        for col in ("recommendation_confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in (
            "recommended_count",
            "monitor_count",
            "deferred_count",
            "avoided_count",
            "operator_escalation_count",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return True
    except Exception as e:
        _warn(f"parquet write failed for {path}: {type(e).__name__}: {e}")
        return False


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


def _norm_upper(x: Any, default: str = "UNKNOWN") -> str:
    s = str(x or "").strip().upper()
    return s or default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _system_health_score(health: Dict[str, Any]) -> float:
    status = _norm_upper(health.get("overall_status"))
    n_total = _to_float(health.get("n_artifacts_total")) or 1.0
    n_fresh = _to_float(health.get("n_artifacts_fresh")) or 0.0
    n_sub = _to_float(health.get("n_subsystems_total")) or 1.0
    n_healthy = _to_float(health.get("n_subsystems_healthy")) or 0.0
    base = (n_fresh / max(n_total, 1.0)) * 0.55 + (n_healthy / max(n_sub, 1.0)) * 0.45
    if status == "STALE":
        base *= 0.55
    elif status == "DEGRADED":
        base *= 0.75
    elif status == "HEALTHY":
        base = max(base, 0.75)
    return _clamp(base, 0.0, 1.0)


def _court_stability_score(ctx: Dict[str, Any]) -> float:
    score = 1.0
    if ctx["constitution_violated"]:
        score -= 0.35
    if ctx["court_ruling"] == "COURT_OVERRULED":
        score -= 0.25
    if ctx["council_ruling"] in ("GOVERNANCE_REVOKE_AUTONOMY", "GOVERNANCE_LOCKDOWN"):
        score -= 0.15
    if ctx["operator_pressure"]:
        score -= 0.05
    return _clamp(score, 0.0, 1.0)


# -----------------------------------------------------------
# Context extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    impact_summary: Dict[str, Any],
    impact_record: Dict[str, Any],
    impact_mem: List[Dict[str, str]],
    simulation_summary: Dict[str, Any],
    activation_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    ctx: Dict[str, Any] = {
        "impact_state": _norm_upper(
            impact_summary.get("impact_state") or impact_record.get("impact_state")
        ),
        "impact_confidence": _clamp(
            _to_float(impact_summary.get("impact_confidence"))
            or _to_float(impact_record.get("impact_confidence"))
            or 0.0,
            0.0,
            1.0,
        ),
        "simulation_confidence": _clamp(
            _to_float(simulation_summary.get("simulation_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "activation_state": _norm_upper(activation_summary.get("activation_state")),
        "doctrine_impacts": impact_record.get("doctrine_impacts") or [],
        "impact_available": bool(impact_summary.get("beneficial_doctrine_available"))
        or len(impact_record.get("doctrine_impacts") or []) > 0,
        "impact_memory_depth": len(impact_mem),
        "governance_quality": _clamp(
            _to_float(scores.get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "constitutional_pressure": _clamp(
            0.75 if constitution_state == "CONSTITUTION_VIOLATED" else 0.30,
            0.0,
            1.0,
        ),
        "constitution_violated": constitution_state == "CONSTITUTION_VIOLATED",
        "court_ruling": _norm_upper(court_summary.get("judicial_ruling")),
        "council_ruling": _norm_upper(council_summary.get("governance_ruling")),
        "operator_pressure": bool(
            court_summary.get("operator_review_required")
            or council_summary.get("operator_supervision_required")
            or impact_summary.get("operator_review_required")
            or activation_summary.get("operator_activation_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "readiness_score": _clamp(
            _to_float(readiness_summary.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
    }
    ctx["court_stability"] = _court_stability_score(ctx)
    return ctx


# -----------------------------------------------------------
# Doctrine recommendations
# -----------------------------------------------------------
def _recommend_doctrine(
    impact: Dict[str, Any],
    *,
    ctx: Dict[str, Any],
    restricted: bool,
) -> Dict[str, Any]:
    name = str(impact.get("policy_name", ""))
    impact_dec = _norm_upper(impact.get("impact_decision"))
    conf = _to_float(impact.get("confidence")) or 0.0
    const_safe = bool(impact.get("constitutional_safe", True))
    gov_improved = _norm_upper(impact.get("governance_quality_impact")) == "IMPROVED"
    pressure_lower = _norm_upper(impact.get("constitutional_pressure_impact")) == "LOWER"
    risk_defensive = _norm_upper(impact.get("risk_posture_impact")) == "MORE_DEFENSIVE"

    decision = DECISION_DEFER
    reason = "insufficient evidence for doctrine recommendation"
    op_required = False
    future_candidate = False

    if not const_safe or impact_dec == IMPACT_HARMFUL:
        decision = DECISION_AVOID
        reason = "avoid: harmful or constitutionally unsafe doctrine under current governance"
    elif (
        impact_dec == IMPACT_UNCERTAIN
        and name in HIGH_IMPACT_POLICIES
        and (ctx["constitutional_pressure"] >= 0.55 or ctx["constitution_violated"])
    ):
        decision = DECISION_OPERATOR
        reason = (
            "operator escalation required for high-impact doctrine under constitutional sensitivity"
        )
        op_required = True
    elif impact_dec == IMPACT_UNCERTAIN:
        decision = DECISION_DEFER
        reason = "defer: insufficient evidence and unclear governance stability"
    elif impact_dec == IMPACT_NEUTRAL:
        decision = DECISION_MONITOR if conf >= 0.50 else DECISION_DEFER
        reason = "monitor only: neutral impact with limited governance benefit signal"
    elif impact_dec == IMPACT_BENEFICIAL and const_safe and gov_improved:
        if restricted and name not in DEFENSIVE_CAPITAL and not risk_defensive:
            decision = DECISION_MONITOR
            reason = (
                "monitor only: restricted recommendation permits defensive doctrine observation"
            )
        elif name in HIGH_IMPACT_POLICIES and ctx["constitution_violated"]:
            decision = DECISION_OPERATOR
            reason = "operator escalation required despite beneficial impact due to constitutional sensitivity"
            op_required = True
        elif conf >= 0.65 and (pressure_lower or risk_defensive):
            if restricted and conf < 0.70:
                decision = DECISION_MONITOR
                reason = "monitor only: beneficial defensive doctrine; observation recommended under pressure"
            else:
                decision = DECISION_RECOMMEND
                reason = f"recommend for future consideration: {name} improves governance under current profile"
                future_candidate = True
        elif conf >= 0.55:
            decision = DECISION_MONITOR
            reason = "monitor only: beneficial but evidence weak; defensive observation recommended"
        else:
            decision = DECISION_DEFER
            reason = "defer: beneficial signal too weak for institutional recommendation"
    else:
        decision = DECISION_DEFER
        reason = "defer pending stronger impact assessment evidence"

    if decision == DECISION_RECOMMEND and name == "target_cash_pct":
        reason = (
            "recommend for future consideration: elevated cash doctrine improves "
            "governance stability under constitutional pressure"
        )

    return {
        "policy_name": name,
        "recommendation_decision": decision,
        "impact_decision": impact_dec,
        "recommendation_reason": reason,
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "future_activation_candidate": future_candidate,
        "operator_review_required": op_required or decision == DECISION_OPERATOR,
        "runtime_mutation_allowed": False,
    }


def _recommend_all(ctx: Dict[str, Any], restricted: bool) -> List[Dict[str, Any]]:
    return [
        _recommend_doctrine(imp, ctx=ctx, restricted=restricted) for imp in ctx["doctrine_impacts"]
    ]


# -----------------------------------------------------------
# Recommendation confidence and state
# -----------------------------------------------------------
def _recommendation_confidence(ctx: Dict[str, Any], recommended: List[Dict[str, Any]]) -> float:
    active = [r for r in recommended if r["recommendation_decision"] != DECISION_DEFER]
    avg_conf = sum(r["confidence"] for r in active) / max(len(active), 1) if active else 0.0
    const_safety = (
        sum(1 for r in active if r["constitutional_safe"]) / max(len(active), 1) if active else 0.0
    )

    raw = (
        ctx["impact_confidence"] * 0.22
        + ctx["simulation_confidence"] * 0.10
        + ctx["governance_quality"] * 0.18
        + ctx["system_health_score"] * 0.16
        + ctx["readiness_score"] * 0.14
        + ctx["court_stability"] * 0.12
        + const_safety * 0.05
        + avg_conf * 0.03
    )

    penalty = ctx["constitutional_pressure"] * 0.18
    if ctx["constitution_violated"]:
        penalty += 0.10
    if ctx["court_ruling"] == "COURT_OVERRULED":
        penalty += 0.08
    if ctx["system_health_stale"]:
        penalty += 0.07
    if ctx["impact_confidence"] < 0.10:
        penalty += 0.05

    return round(_clamp(raw - penalty, 0.0, 1.0), 6)


def _count_decisions(recommended: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "recommended": sum(
            1 for r in recommended if r["recommendation_decision"] == DECISION_RECOMMEND
        ),
        "monitor": sum(1 for r in recommended if r["recommendation_decision"] == DECISION_MONITOR),
        "deferred": sum(1 for r in recommended if r["recommendation_decision"] == DECISION_DEFER),
        "avoided": sum(1 for r in recommended if r["recommendation_decision"] == DECISION_AVOID),
        "operator": sum(
            1 for r in recommended if r["recommendation_decision"] == DECISION_OPERATOR
        ),
    }


def _classify_recommendation_state(
    *,
    ctx: Dict[str, Any],
    recommendation_confidence: float,
    recommended: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> Tuple[str, List[str], bool]:
    reasons: List[str] = []

    if not ctx["impact_available"] or not ctx["doctrine_impacts"]:
        reasons.append("no impact evidence available for doctrine recommendation")
        return RECOMMENDATION_DORMANT, reasons, False

    restricted = (
        ctx["constitutional_pressure"] >= 0.55
        or ctx["constitution_violated"]
        or ctx["court_ruling"] == "COURT_OVERRULED"
        or ctx["impact_state"] == "DOCTRINE_IMPACT_RESTRICTED"
    )

    if (
        ctx["impact_state"] == "DOCTRINE_IMPACT_INSTITUTIONAL"
        and recommendation_confidence >= 0.58
        and ctx["impact_memory_depth"] >= 3
        and counts["recommended"] >= 2
    ):
        reasons.append("mature doctrine recommendation process with repeatable advisory quality")
        return RECOMMENDATION_INSTITUTIONAL, reasons, restricted

    if restricted:
        reasons.append("constitutional pressure elevated; defensive doctrine recommendations only")
        return RECOMMENDATION_RESTRICTED, reasons, True

    if recommendation_confidence < 0.35 or counts["recommended"] == 0:
        reasons.append("weak evidence; doctrine recommendations immature")
        return RECOMMENDATION_FORMING, reasons, False

    if recommendation_confidence >= 0.40 and len(recommended) >= 1:
        reasons.append("doctrine recommendations functioning under normal advisory process")
        return RECOMMENDATION_ACTIVE, reasons, False

    reasons.append("recommendation process forming institutional advisory posture")
    return RECOMMENDATION_FORMING, reasons, restricted


# -----------------------------------------------------------
# Booleans, recommendations list, rationale
# -----------------------------------------------------------
def _recommendation_booleans(
    state: str,
    recommended: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    return {
        "doctrine_recommendation_available": len(recommended) > 0,
        "future_activation_candidates_available": counts["recommended"] > 0,
        "operator_review_required": (counts["operator"] > 0 or ctx["operator_pressure"]),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "recommendation_memory_reliable": state == RECOMMENDATION_INSTITUTIONAL,
    }


def _recommendations_list(state: str, counts: Dict[str, int]) -> List[str]:
    recs: List[str] = []
    if counts["monitor"] > 0 or state == RECOMMENDATION_RESTRICTED:
        recs.append("Continue defensive doctrine monitoring")
    if counts["avoided"] > 0:
        recs.append("Avoid harmful doctrine consideration")
    if counts["deferred"] > 0 or counts["operator"] > 0:
        recs.append("Escalate uncertain doctrine to operator review")
    recs.append("Maintain runtime mutation lock")
    if state in (RECOMMENDATION_FORMING, RECOMMENDATION_DORMANT):
        recs.append("Continue governance evidence collection")
    if state == RECOMMENDATION_RESTRICTED:
        recs.append("Require constitutional review before activation consideration")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(recommended: List[Dict[str, Any]], state: str) -> str:
    cash = [
        r
        for r in recommended
        if r["policy_name"] == "target_cash_pct"
        and r["recommendation_decision"] == DECISION_RECOMMEND
    ]
    if cash:
        return (
            "Triton recommends elevated cash doctrine for future consideration because "
            "governance stability improved under constitutional pressure."
        )
    recs = [r for r in recommended if r["recommendation_decision"] == DECISION_RECOMMEND]
    if recs:
        names = ", ".join(r["policy_name"] for r in recs[:3])
        return f"Triton recommends {names} for future activation consideration based on beneficial impact."
    if state == RECOMMENDATION_RESTRICTED:
        return (
            "Recommendation advisory operates in restricted mode due to elevated "
            "constitutional pressure; only defensive doctrine may be recommended."
        )
    return "Governance doctrine recommendation advisory completed without runtime mutation."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    recommendation_confidence: float,
    recommended: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Doctrine Recommendation",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Recommendation State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| recommendation_confidence | {recommendation_confidence:.3f} |",
        f"| recommended | {counts['recommended']} |",
        f"| monitor | {counts['monitor']} |",
        f"| deferred | {counts['deferred']} |",
        f"| avoided | {counts['avoided']} |",
        f"| operator_escalation | {counts['operator']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Doctrine Recommendations",
        "",
    ]
    if recommended:
        lines.append("| policy | decision | impact | confidence | future_candidate |")
        lines.append("|---|---|---|---|---|")
        for r in recommended:
            lines.append(
                f"| {r['policy_name']} | {r['recommendation_decision']} | "
                f"{r['impact_decision']} | {r['confidence']:.2f} | {r['future_activation_candidate']} |"
            )
        lines.append("")
        for r in recommended:
            lines.append(
                f"- **{r['policy_name']}** ({r['recommendation_decision']}): {r['recommendation_reason']}"
            )
    else:
        lines.append("_No doctrine recommendations this cycle._")

    candidates = [r for r in recommended if r["future_activation_candidate"]]
    lines.extend(["", "## Future Activation Candidates", ""])
    if candidates:
        for r in candidates:
            lines.append(f"- {r['policy_name']}: {r['recommendation_reason']}")
    else:
        lines.append("_No future activation candidates this cycle._")

    deferred_avoided = [
        r
        for r in recommended
        if r["recommendation_decision"] in (DECISION_DEFER, DECISION_AVOID, DECISION_OPERATOR)
    ]
    lines.extend(["", "## Deferred or Avoided Doctrines", ""])
    if deferred_avoided:
        for r in deferred_avoided:
            lines.append(
                f"- {r['policy_name']} ({r['recommendation_decision']}): {r['recommendation_reason']}"
            )
    else:
        lines.append("_None deferred or avoided._")

    lines.extend(["", "## Recommendations", ""])
    for rec in recommendations:
        lines.append(f"- {rec}")
    lines.extend(["", "## Why", ""])
    for r in reasons:
        lines.append(f"- {r}")
    lines.extend(
        [
            "",
            "## Narrative",
            "",
            rationale,
            "",
            "Recommendations are governance advisory only. Beneficial ≠ Recommended. "
            "Recommended ≠ Activated. Activation consideration ≠ runtime mutation. "
            "No runtime policy is changed. Constitutional law, court rulings, "
            "capital preservation doctrine, and operator supremacy remain supreme.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Recommendation memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    recommendation_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "recommendation_state": state,
        "recommended_count": counts["recommended"],
        "monitor_count": counts["monitor"],
        "deferred_count": counts["deferred"],
        "avoided_count": counts["avoided"],
        "operator_escalation_count": counts["operator"],
        "recommendation_confidence": round(recommendation_confidence, 6),
        "rationale": rationale,
    }


def _merge_memory(
    existing: List[Dict[str, Any]],
    new_row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    keyed: Dict[str, Dict[str, Any]] = {}
    for r in existing:
        ts = str(r.get("timestamp", ""))
        if ts:
            keyed[ts] = r
    keyed[str(new_row.get("timestamp", ""))] = new_row
    out = list(keyed.values())
    for r in out:
        for c in RECOMMENDATION_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_doctrine_recommendation(
    *,
    impact_summary: Dict[str, Any],
    impact_record: Dict[str, Any],
    impact_mem: List[Dict[str, str]],
    simulation_summary: Dict[str, Any],
    activation_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    existing_recommendation_mem: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        impact_summary=impact_summary,
        impact_record=impact_record,
        impact_mem=impact_mem,
        simulation_summary=simulation_summary,
        activation_summary=activation_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        runtime_policy=runtime_policy,
    )

    pre_restricted = (
        ctx["constitutional_pressure"] >= 0.55
        or ctx["constitution_violated"]
        or ctx["court_ruling"] == "COURT_OVERRULED"
    )
    recommended = _recommend_all(ctx, restricted=pre_restricted)
    recommendation_confidence = _recommendation_confidence(ctx, recommended)
    counts = _count_decisions(recommended)

    state, reasons, restricted = _classify_recommendation_state(
        ctx=ctx,
        recommendation_confidence=recommendation_confidence,
        recommended=recommended,
        counts=counts,
    )

    if restricted != pre_restricted:
        recommended = _recommend_all(ctx, restricted=restricted)
        recommendation_confidence = _recommendation_confidence(ctx, recommended)
        counts = _count_decisions(recommended)

    booleans = _recommendation_booleans(state, recommended, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(recommended, state)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        recommendation_confidence=recommendation_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(existing_recommendation_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        recommendation_confidence=recommendation_confidence,
        recommended=recommended,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_recommendation_engine",
        "engine_version": 1,
        "recommendation_state": state,
        "recommendation_confidence": recommendation_confidence,
        "recommendation_reasons": reasons,
        "doctrine_recommendations": recommended,
        "decision_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "beneficial_vs_recommended_note": (
            "Beneficial ≠ Recommended. Recommended ≠ Activated. "
            "This engine provides governance advisory only; it never mutates runtime policy."
        ),
        "constitutional_supremacy_note": (
            "Recommendations cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "recommendation_memory_size_after_append": len(merged_memory),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "recommendation_advisory_only": True,
        },
        "inputs_seen": {
            "arm_governance_doctrine_impact_assessment_summary": bool(impact_summary),
            "arm_governance_doctrine_impact_assessment_record": bool(impact_record),
            "arm_governance_doctrine_impact_assessment_memory_rows": len(impact_mem),
            "arm_governance_doctrine_simulation_summary": bool(simulation_summary),
            "arm_governance_doctrine_activation_board_summary": bool(activation_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_recommendation_memory_rows": len(existing_recommendation_mem),
            "n_doctrine_impacts": len(ctx["doctrine_impacts"]),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_recommendation_engine",
        "recommendation_state": state,
        "recommendation_confidence": recommendation_confidence,
        "doctrine_recommendation_available": booleans["doctrine_recommendation_available"],
        "future_activation_candidates_available": booleans[
            "future_activation_candidates_available"
        ],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "recommendation_memory_reliable": booleans["recommendation_memory_reliable"],
        "recommended_count": counts["recommended"],
        "monitor_count": counts["monitor"],
        "deferred_count": counts["deferred"],
        "avoided_count": counts["avoided"],
        "operator_escalation_count": counts["operator"],
        "n_doctrines_recommended": len(recommended),
        "n_recommendations": len(recommendations),
        "recommendation_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance doctrine recommendation engine (Step 45). "
            "Converts impact assessments into institutional doctrine recommendations. "
            "Never mutates runtime policy. No broker calls."
        ),
    )
    p.add_argument("--impact-summary", default=str(DEFAULT_IMPACT_SUM))
    p.add_argument("--impact-record", default=str(DEFAULT_IMPACT_REC))
    p.add_argument("--impact-mem", default=str(DEFAULT_IMPACT_MEM))
    p.add_argument("--simulation-summary", default=str(DEFAULT_SIMULATION_SUM))
    p.add_argument("--activation-summary", default=str(DEFAULT_ACTIVATION_SUM))
    p.add_argument("--court-summary", default=str(DEFAULT_COURT_SUMMARY))
    p.add_argument("--council-summary", default=str(DEFAULT_COUNCIL_SUMMARY))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUM))
    p.add_argument("--runtime-policy", default=str(DEFAULT_RUNTIME_POLICY))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-mem-csv", default=str(DEFAULT_OUT_MEM_CSV))
    p.add_argument("--out-mem-parquet", default=str(DEFAULT_OUT_MEM_PQ))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[ARM_DOCTRINE_RECOMMENDATION] starting "
        "(read-only governance advisory; no runtime mutation; no broker calls)",
        flush=True,
    )

    impact_summary = _safe_read_json(
        Path(args.impact_summary), label="arm_governance_doctrine_impact_assessment_summary.json"
    )
    impact_record = _safe_read_json(
        Path(args.impact_record), label="arm_governance_doctrine_impact_assessment.json"
    )
    impact_mem = _safe_read_csv_rows(
        Path(args.impact_mem), label="arm_governance_doctrine_impact_assessment_memory.csv"
    )
    simulation_summary = _safe_read_json(
        Path(args.simulation_summary), label="arm_governance_doctrine_simulation_summary.json"
    )
    activation_summary = _safe_read_json(
        Path(args.activation_summary), label="arm_governance_doctrine_activation_board_summary.json"
    )
    court_summary = _safe_read_json(
        Path(args.court_summary), label="arm_constitutional_court_summary.json"
    )
    council_summary = _safe_read_json(
        Path(args.council_summary), label="arm_supreme_governance_council_summary.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="autonomous_readiness_summary.json"
    )
    runtime_policy = _safe_read_json(
        Path(args.runtime_policy), label="runtime_policy_governed.json"
    )
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_doctrine_recommendation_memory.csv"
    )

    record, summary, md, merged_memory = build_doctrine_recommendation(
        impact_summary=impact_summary,
        impact_record=impact_record,
        impact_mem=impact_mem,
        simulation_summary=simulation_summary,
        activation_summary=activation_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        runtime_policy=runtime_policy,
        existing_recommendation_mem=existing_mem,
    )

    try:
        _atomic_write_json(record, Path(args.out_json))
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
    try:
        _atomic_write_csv(
            merged_memory, Path(args.out_mem_csv), columns=RECOMMENDATION_MEMORY_COLUMNS
        )
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["decision_counts"]
    print(
        "[ARM_DOCTRINE_RECOMMENDATION] "
        f"state={record['recommendation_state']} "
        f"recommended={counts['recommended']} "
        f"monitor={counts['monitor']} "
        f"avoid={counts['avoided']} "
        f"confidence={record['recommendation_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_DOCTRINE_RECOMMENDATION_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[ARM_DOCTRINE_RECOMMENDATION_OUT] json={Path(args.out_json).as_posix()} "
        f"md={Path(args.out_md).as_posix()} "
        f"summary={Path(args.out_summary).as_posix()} "
        f"memory_csv={Path(args.out_mem_csv).as_posix()} "
        f"memory_parquet={Path(args.out_mem_parquet).as_posix() if parquet_ok else 'SKIPPED'}",
        flush=True,
    )
    return 0


# -----------------------------------------------------------
# Self-inspection: enforce the no-broker safety rule at import time
# -----------------------------------------------------------
_FORBIDDEN_TOKENS: Tuple[str, ...] = (
    "Alpaca" + "Broker",
    "place" + "_order",
    "submit" + "_order",
    "execute" + "_trades",
    "place" + "_live_orders",
    "broker" + "_client",
)


def _self_check_no_broker_tokens() -> None:
    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return
    for tok in _FORBIDDEN_TOKENS:
        if tok in src:
            raise RuntimeError(
                f"[ARM_DOCTRINE_RECOMMENDATION_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
