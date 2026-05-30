"""
ARM Governance Doctrine Impact Assessment Engine -- Step 44.

Reads:
    data/results/arm_governance_doctrine_simulation_summary.json       (Step 43)
    data/results/arm_governance_doctrine_simulation.json               (Step 43)
    data/results/arm_governance_doctrine_simulation_memory.csv         (Step 43)
    data/results/autonomous_governance_scorecard.json                  (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/arm_governance_recovery_effectiveness_summary.json    (Step 37)
    data/results/arm_governance_drift_detection_summary.json           (Step 35)
    data/results/arm_constitutional_court_summary.json                 (Step 33)
    data/results/arm_supreme_governance_council_summary.json           (Step 34)
    data/results/runtime_policy_governed.json                          (Step 18)

Writes:
    data/results/arm_governance_doctrine_impact_assessment.json
    data/results/arm_governance_doctrine_impact_assessment.md
    data/results/arm_governance_doctrine_impact_assessment_summary.json
    data/results/arm_governance_doctrine_impact_assessment_memory.csv
    data/results/arm_governance_doctrine_impact_assessment_memory.parquet

Purpose
-------
This engine answers:

    "If activated, would this doctrine improve governance?"

It evaluates whether simulated governance doctrine would improve, worsen, or
destabilize governance quality. Simulation != activation. Beneficial != runtime
mutation. Impact assessment NEVER mutates runtime policy.

Impact assessment state cascade
-------------------------------
    1. DOCTRINE_IMPACT_INSTITUTIONAL  mature doctrine assessment process
    2. DOCTRINE_IMPACT_RESTRICTED      elevated pressure; defensive doctrine only
    3. DOCTRINE_IMPACT_ACTIVE           doctrine impact evaluated normally
    4. DOCTRINE_IMPACT_FORMING          limited simulation evidence; weak confidence
    5. DOCTRINE_IMPACT_DORMANT          no simulation available

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
* Append-only impact memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to DOCTRINE_IMPACT_DORMANT.
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

DEFAULT_SIMULATION_SUM = RESULTS_DIR / "arm_governance_doctrine_simulation_summary.json"
DEFAULT_SIMULATION_REC = RESULTS_DIR / "arm_governance_doctrine_simulation.json"
DEFAULT_SIMULATION_MEM = RESULTS_DIR / "arm_governance_doctrine_simulation_memory.csv"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_SUM = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_RECOVERY_EFF_SUM = RESULTS_DIR / "arm_governance_recovery_effectiveness_summary.json"
DEFAULT_DRIFT_SUM = RESULTS_DIR / "arm_governance_drift_detection_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_doctrine_impact_assessment.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_doctrine_impact_assessment.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_doctrine_impact_assessment_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_doctrine_impact_assessment_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_doctrine_impact_assessment_memory.parquet"


# -----------------------------------------------------------
# Impact state constants
# -----------------------------------------------------------
IMPACT_DORMANT = "DOCTRINE_IMPACT_DORMANT"
IMPACT_FORMING = "DOCTRINE_IMPACT_FORMING"
IMPACT_RESTRICTED = "DOCTRINE_IMPACT_RESTRICTED"
IMPACT_ACTIVE = "DOCTRINE_IMPACT_ACTIVE"
IMPACT_INSTITUTIONAL = "DOCTRINE_IMPACT_INSTITUTIONAL"

DECISION_BENEFICIAL = "BENEFICIAL"
DECISION_NEUTRAL = "NEUTRAL"
DECISION_HARMFUL = "HARMFUL"
DECISION_UNCERTAIN = "UNCERTAIN"

SIM_ASSESSED = frozenset({"SIMULATED", "SIMULATED_DEFENSIVE"})
SIM_UNCERTAIN = frozenset({"SIMULATION_DEFERRED", "SIMULATION_OPERATOR_OBSERVATION"})
SIM_HARMFUL = frozenset({"SIMULATION_REJECTED"})

LOOSENING_POLICIES = frozenset(
    {
        "autonomy_loosen_threshold",
        "reduce_confidence_threshold",
        "reduce_target_cash_pct",
    }
)

DRIFT_NEGATIVE = frozenset(
    {
        "GOVERNANCE_DRIFTING",
        "GOVERNANCE_UNSTABLE",
        "GOVERNANCE_FAILURE_RISK",
    }
)
RECOVERY_WEAK = frozenset(
    {
        "RECOVERY_INEFFECTIVE",
        "RECOVERY_REGRESSING",
        "RECOVERY_STALLED",
        "RECOVERY_LOCKDOWN",
    }
)

IMPACT_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "impact_state",
    "beneficial_count",
    "neutral_count",
    "harmful_count",
    "uncertain_count",
    "impact_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_DOCTRINE_IMPACT_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(IMPACT_MEMORY_COLUMNS))
        for col in ("impact_confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("beneficial_count", "neutral_count", "harmful_count", "uncertain_count"):
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


def _direction_label(score: float) -> str:
    if score >= 0.35:
        return "IMPROVED"
    if score <= -0.35:
        return "WORSENED"
    return "FLAT"


def _pressure_label(score: float) -> str:
    if score >= 0.35:
        return "LOWER"
    if score <= -0.35:
        return "HIGHER"
    return "FLAT"


def _risk_posture_label(score: float) -> str:
    if score >= 0.35:
        return "MORE_DEFENSIVE"
    if score <= -0.35:
        return "MORE_AGGRESSIVE"
    return "FLAT"


def _is_defensive_change(name: str, current: Any, simulated: Any) -> bool:
    if name in LOOSENING_POLICIES:
        return False
    if name == "max_position_pct":
        c, s = _to_float(current), _to_float(simulated)
        return c is not None and s is not None and s < c
    if name == "auto_lock_manual_after_overruling":
        return simulated is True or str(simulated).lower() == "true"
    c, s = _to_float(current), _to_float(simulated)
    if c is not None and s is not None:
        return s >= c
    return True


def _is_aggressive_change(name: str, current: Any, simulated: Any) -> bool:
    if name in LOOSENING_POLICIES:
        return True
    if name == "max_position_pct":
        c, s = _to_float(current), _to_float(simulated)
        return c is not None and s is not None and s > c
    if name in (
        "confidence_threshold",
        "deployment_threshold",
        "target_cash_pct",
        "persistence_threshold",
        "skepticism_threshold",
        "autonomy_readiness_threshold",
    ):
        c, s = _to_float(current), _to_float(simulated)
        return c is not None and s is not None and s < c
    return False


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
    simulation_summary: Dict[str, Any],
    simulation_record: Dict[str, Any],
    simulation_mem: List[Dict[str, str]],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    recovery_eff_summary: Dict[str, Any],
    drift_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    ctx: Dict[str, Any] = {
        "simulation_state": _norm_upper(
            simulation_summary.get("simulation_state") or simulation_record.get("simulation_state")
        ),
        "simulation_confidence": _clamp(
            _to_float(simulation_summary.get("simulation_confidence"))
            or _to_float(simulation_record.get("simulation_confidence"))
            or 0.0,
            0.0,
            1.0,
        ),
        "simulated_doctrines": simulation_record.get("simulated_doctrines") or [],
        "simulation_available": bool(simulation_summary.get("doctrine_simulation_available")),
        "simulation_memory_depth": len(simulation_mem),
        "governance_quality": _clamp(
            _to_float(scores.get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "recovery_effectiveness": _clamp(
            _to_float(recovery_eff_summary.get("effectiveness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "recovery_working": bool(recovery_eff_summary.get("recovery_working")),
        "recovery_state": _norm_upper(recovery_eff_summary.get("effectiveness_state")),
        "drift_state": _norm_upper(drift_summary.get("drift_state")),
        "drift_score": _clamp(_to_float(drift_summary.get("drift_score")) or 0.0, 0.0, 1.0),
        "constitutional_pressure": _clamp(
            _to_float(drift_summary.get("constitutional_pressure"))
            or (0.75 if constitution_state == "CONSTITUTION_VIOLATED" else 0.30),
            0.0,
            1.0,
        ),
        "constitution_violated": constitution_state == "CONSTITUTION_VIOLATED",
        "court_ruling": _norm_upper(court_summary.get("judicial_ruling")),
        "council_ruling": _norm_upper(council_summary.get("governance_ruling")),
        "operator_pressure": bool(
            court_summary.get("operator_review_required")
            or council_summary.get("operator_supervision_required")
            or simulation_summary.get("operator_review_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "readiness_score": _clamp(
            _to_float(readiness_summary.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "governance_stressed": (
            _norm_upper(drift_summary.get("drift_state")) in DRIFT_NEGATIVE
            or _norm_upper(recovery_eff_summary.get("effectiveness_state")) in RECOVERY_WEAK
            or constitution_state == "CONSTITUTION_VIOLATED"
        ),
    }
    ctx["court_stability"] = _court_stability_score(ctx)
    return ctx


def _assessable_doctrines(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        d
        for d in ctx["simulated_doctrines"]
        if _norm_upper(d.get("simulation_decision")) in SIM_ASSESSED
    ]


# -----------------------------------------------------------
# Doctrine impact assessment
# -----------------------------------------------------------
def _dimension_scores(
    name: str,
    current: Any,
    simulated: Any,
    *,
    ctx: Dict[str, Any],
    defensive: bool,
    aggressive: bool,
) -> Dict[str, float]:
    """Signed scores in [-1, 1] for each impact dimension."""
    stress_boost = 0.15 if ctx["governance_stressed"] else 0.0
    scores = {
        "governance_quality": 0.0,
        "constitutional_pressure": 0.0,
        "recovery": 0.0,
        "readiness": 0.0,
        "autonomy_maturity": 0.0,
        "risk_posture": 0.0,
    }

    if aggressive:
        scores = {k: -0.6 for k in scores}
        return scores

    if not defensive:
        return scores

    base = 0.45 + stress_boost
    if name in ("target_cash_pct", "max_position_pct"):
        scores["governance_quality"] = base + 0.15
        scores["constitutional_pressure"] = base
        scores["recovery"] = base * 0.8
        scores["readiness"] = base * 0.5
        scores["autonomy_maturity"] = 0.1
        scores["risk_posture"] = base + 0.2
    elif name in ("confidence_threshold", "deployment_threshold", "skepticism_threshold"):
        scores["governance_quality"] = base + 0.1
        scores["constitutional_pressure"] = base + 0.1
        scores["recovery"] = base * 0.6
        scores["readiness"] = base * 0.4
        scores["autonomy_maturity"] = base * 0.3
        scores["risk_posture"] = base + 0.15
    elif name in (
        "persistence_threshold",
        "min_observations_before_graduation",
        "autonomy_readiness_threshold",
    ):
        scores["governance_quality"] = base * 0.7
        scores["constitutional_pressure"] = base * 0.8
        scores["recovery"] = base * 0.5
        scores["readiness"] = base * 0.3
        scores["autonomy_maturity"] = base + 0.2
        scores["risk_posture"] = base + 0.1
    elif name == "auto_lock_manual_after_overruling":
        scores["governance_quality"] = base + 0.2
        scores["constitutional_pressure"] = base + 0.25
        scores["recovery"] = base * 0.4
        scores["readiness"] = 0.0
        scores["autonomy_maturity"] = base * 0.5
        scores["risk_posture"] = base + 0.15
    elif name == "governance_monitoring_frequency_multiplier":
        scores["governance_quality"] = base + 0.05
        scores["constitutional_pressure"] = base * 0.6
        scores["recovery"] = base * 0.7
        scores["readiness"] = base * 0.3
        scores["autonomy_maturity"] = 0.05
        scores["risk_posture"] = base * 0.5
    else:
        scores["governance_quality"] = base * 0.5
        scores["risk_posture"] = base * 0.3

    return scores


def _assess_doctrine(
    doctrine: Dict[str, Any], *, ctx: Dict[str, Any], restricted: bool
) -> Dict[str, Any]:
    name = str(doctrine.get("policy_name", ""))
    sim_dec = _norm_upper(doctrine.get("simulation_decision"))
    conf = _to_float(doctrine.get("confidence")) or 0.0
    const_safe = bool(doctrine.get("constitutional_safe", True))
    current = doctrine.get("current_value")
    simulated = doctrine.get("simulated_value")
    defensive = _is_defensive_change(name, current, simulated)
    aggressive = _is_aggressive_change(name, current, simulated)

    impact_decision = DECISION_UNCERTAIN
    impact_rationale = "insufficient evidence to assess governance impact"

    if sim_dec in SIM_HARMFUL or not const_safe:
        impact_decision = DECISION_HARMFUL
        impact_rationale = "harmful or unsafe doctrine under current governance conditions"
    elif sim_dec in SIM_UNCERTAIN:
        impact_decision = DECISION_UNCERTAIN
        impact_rationale = "insufficient simulation evidence; conflicting or incomplete signals"
    elif aggressive:
        impact_decision = DECISION_HARMFUL
        impact_rationale = "doctrine loosens governance posture and may destabilize recovery"
    elif restricted and not defensive:
        impact_decision = DECISION_UNCERTAIN
        impact_rationale = "restricted assessment permits only defensive doctrine evaluation"
    elif ctx["system_health_stale"] and conf < 0.65:
        impact_decision = DECISION_UNCERTAIN
        impact_rationale = "stale system health weakens impact assessment confidence"
    else:
        dims = _dimension_scores(
            name,
            current,
            simulated,
            ctx=ctx,
            defensive=defensive,
            aggressive=aggressive,
        )
        net = sum(dims.values()) / len(dims)
        if net >= 0.35 and conf >= 0.55 and const_safe:
            impact_decision = DECISION_BENEFICIAL
            impact_rationale = (
                f"beneficial: {name} improves governance stability under current stress profile"
            )
        elif net <= -0.25:
            impact_decision = DECISION_HARMFUL
            impact_rationale = "harmful: simulated doctrine worsens governance stability"
        elif abs(net) < 0.15:
            impact_decision = DECISION_NEUTRAL
            impact_rationale = "neutral: little measurable governance impact expected"
        else:
            impact_decision = DECISION_UNCERTAIN
            impact_rationale = "uncertain: mixed governance signals require more evidence"

        if impact_decision == DECISION_BENEFICIAL and name == "target_cash_pct":
            impact_rationale = (
                "beneficial: elevated cash posture improves capital preservation "
                "and reduces constitutional pressure under stress"
            )

        return _build_impact_row(
            name=name,
            impact_decision=impact_decision,
            dims=dims,
            conf=conf,
            const_safe=const_safe,
            impact_rationale=impact_rationale,
        )

    dims = _dimension_scores(
        name, current, simulated, ctx=ctx, defensive=defensive, aggressive=aggressive
    )
    return _build_impact_row(
        name=name,
        impact_decision=impact_decision,
        dims=dims,
        conf=conf,
        const_safe=const_safe,
        impact_rationale=impact_rationale,
    )


def _build_impact_row(
    *,
    name: str,
    impact_decision: str,
    dims: Dict[str, float],
    conf: float,
    const_safe: bool,
    impact_rationale: str,
) -> Dict[str, Any]:
    return {
        "policy_name": name,
        "impact_decision": impact_decision,
        "governance_quality_impact": _direction_label(dims["governance_quality"]),
        "constitutional_pressure_impact": _pressure_label(dims["constitutional_pressure"]),
        "recovery_impact": _direction_label(dims["recovery"]),
        "readiness_impact": _direction_label(dims["readiness"]),
        "autonomy_maturity_impact": _direction_label(dims["autonomy_maturity"]),
        "risk_posture_impact": _risk_posture_label(dims["risk_posture"]),
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "runtime_mutation_allowed": False,
        "impact_rationale": impact_rationale,
    }


def _assess_all(ctx: Dict[str, Any], restricted: bool) -> List[Dict[str, Any]]:
    assessed: List[Dict[str, Any]] = []
    for doctrine in ctx["simulated_doctrines"]:
        sim_dec = _norm_upper(doctrine.get("simulation_decision"))
        if sim_dec in SIM_ASSESSED or sim_dec in SIM_UNCERTAIN or sim_dec in SIM_HARMFUL:
            assessed.append(_assess_doctrine(doctrine, ctx=ctx, restricted=restricted))
    return assessed


# -----------------------------------------------------------
# Impact confidence and state
# -----------------------------------------------------------
def _impact_confidence(ctx: Dict[str, Any], assessed: List[Dict[str, Any]]) -> float:
    active = [r for r in assessed if r["impact_decision"] != DECISION_UNCERTAIN]
    avg_conf = sum(r["confidence"] for r in active) / max(len(active), 1) if active else 0.0

    raw = (
        ctx["simulation_confidence"] * 0.20
        + ctx["governance_quality"] * 0.18
        + ctx["recovery_effectiveness"] * 0.16
        + ctx["readiness_score"] * 0.14
        + ctx["system_health_score"] * 0.14
        + ctx["court_stability"] * 0.10
        + avg_conf * 0.08
    )

    penalty = ctx["constitutional_pressure"] * 0.18
    if ctx["constitution_violated"]:
        penalty += 0.10
    if ctx["court_ruling"] == "COURT_OVERRULED":
        penalty += 0.08
    if ctx["system_health_stale"]:
        penalty += 0.07
    if ctx["drift_state"] in DRIFT_NEGATIVE:
        penalty += 0.05
    if ctx["simulation_confidence"] < 0.10:
        penalty += 0.05

    return round(_clamp(raw - penalty, 0.0, 1.0), 6)


def _count_decisions(assessed: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "beneficial": sum(1 for r in assessed if r["impact_decision"] == DECISION_BENEFICIAL),
        "neutral": sum(1 for r in assessed if r["impact_decision"] == DECISION_NEUTRAL),
        "harmful": sum(1 for r in assessed if r["impact_decision"] == DECISION_HARMFUL),
        "uncertain": sum(1 for r in assessed if r["impact_decision"] == DECISION_UNCERTAIN),
    }


def _classify_impact_state(
    *,
    ctx: Dict[str, Any],
    impact_confidence: float,
    assessed: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> Tuple[str, List[str], bool]:
    reasons: List[str] = []
    assessable = _assessable_doctrines(ctx)

    if not ctx["simulation_available"] or (not assessable and counts["beneficial"] == 0):
        if not ctx["simulated_doctrines"]:
            reasons.append("no simulation available for impact assessment")
            return IMPACT_DORMANT, reasons, False

    restricted = (
        ctx["constitutional_pressure"] >= 0.55
        or ctx["constitution_violated"]
        or ctx["court_ruling"] == "COURT_OVERRULED"
        or ctx["simulation_state"] == "DOCTRINE_SIMULATION_RESTRICTED"
    )

    if not assessable and counts["uncertain"] > 0 and counts["beneficial"] == 0:
        if not ctx["simulation_available"]:
            reasons.append("no simulation available for impact assessment")
            return IMPACT_DORMANT, reasons, False
        reasons.append("limited simulation evidence; impact confidence weak")
        return IMPACT_FORMING, reasons, restricted

    if (
        ctx["simulation_state"] == "DOCTRINE_SIMULATION_INSTITUTIONAL"
        and impact_confidence >= 0.58
        and ctx["simulation_memory_depth"] >= 3
        and counts["beneficial"] >= 2
    ):
        reasons.append("mature doctrine assessment process with repeatable impact quality")
        return IMPACT_INSTITUTIONAL, reasons, restricted

    if restricted:
        reasons.append("constitutional pressure elevated; defensive doctrine impact only")
        return IMPACT_RESTRICTED, reasons, True

    if impact_confidence < 0.35 or len(assessable) < 1:
        reasons.append("limited simulation evidence; impact confidence weak")
        return IMPACT_FORMING, reasons, False

    if impact_confidence >= 0.40 and len(assessed) >= 1:
        reasons.append("doctrine impact evaluated under normal assessment process")
        return IMPACT_ACTIVE, reasons, False

    reasons.append("impact assessment forming institutional evaluation posture")
    return IMPACT_FORMING, reasons, restricted


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _impact_booleans(
    state: str,
    assessed: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    return {
        "beneficial_doctrine_available": counts["beneficial"] > 0,
        "harmful_doctrine_detected": counts["harmful"] > 0,
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "operator_review_required": (counts["uncertain"] > 0 or ctx["operator_pressure"]),
        "runtime_mutation_allowed": False,
        "impact_memory_reliable": state == IMPACT_INSTITUTIONAL,
    }


def _recommendations(state: str, counts: Dict[str, int]) -> List[str]:
    recs: List[str] = []
    if counts["beneficial"] > 0 or state == IMPACT_RESTRICTED:
        recs.append("Continue defensive doctrine observation")
    if counts["harmful"] > 0:
        recs.append("Avoid harmful doctrine consideration")
    if counts["uncertain"] > 0:
        recs.append("Escalate uncertain doctrine to operator review")
    recs.append("Maintain runtime mutation lock")
    if state in (IMPACT_FORMING, IMPACT_DORMANT):
        recs.append("Increase doctrine evidence collection")
    if state == IMPACT_RESTRICTED:
        recs.append("Require constitutional review before activation consideration")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(assessed: List[Dict[str, Any]], state: str) -> str:
    cash = [
        r
        for r in assessed
        if r["policy_name"] == "target_cash_pct" and r["impact_decision"] == DECISION_BENEFICIAL
    ]
    if cash:
        return (
            "Triton assessed elevated cash doctrine as beneficial because governance "
            "stability improved and constitutional pressure weakened under simulation."
        )
    beneficial = [r for r in assessed if r["impact_decision"] == DECISION_BENEFICIAL]
    if beneficial:
        names = ", ".join(r["policy_name"] for r in beneficial[:3])
        return f"Triton assessed beneficial governance impact for simulated doctrine: {names}."
    if state == IMPACT_RESTRICTED:
        return (
            "Impact assessment operates in restricted mode due to elevated constitutional "
            "pressure; only defensive doctrine outcomes are evaluated as beneficial."
        )
    uncertain = sum(1 for r in assessed if r["impact_decision"] == DECISION_UNCERTAIN)
    if uncertain:
        return f"Impact assessment flagged {uncertain} doctrine(s) as uncertain pending stronger evidence."
    return "Governance doctrine impact assessment completed without runtime mutation."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    impact_confidence: float,
    assessed: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Doctrine Impact Assessment",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Impact Assessment State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| impact_confidence | {impact_confidence:.3f} |",
        f"| beneficial | {counts['beneficial']} |",
        f"| neutral | {counts['neutral']} |",
        f"| harmful | {counts['harmful']} |",
        f"| uncertain | {counts['uncertain']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Doctrine Impact Decisions",
        "",
    ]
    if assessed:
        lines.append("| policy | decision | gov_quality | pressure | recovery | readiness | risk |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in assessed:
            lines.append(
                f"| {r['policy_name']} | {r['impact_decision']} | "
                f"{r['governance_quality_impact']} | {r['constitutional_pressure_impact']} | "
                f"{r['recovery_impact']} | {r['readiness_impact']} | {r['risk_posture_impact']} |"
            )
        lines.append("")
        for r in assessed:
            lines.append(
                f"- **{r['policy_name']}** ({r['impact_decision']}): {r['impact_rationale']}"
            )
    else:
        lines.append("_No doctrines assessed this cycle._")

    beneficial = [r for r in assessed if r["impact_decision"] == DECISION_BENEFICIAL]
    lines.extend(["", "## Beneficial Doctrines", ""])
    if beneficial:
        for r in beneficial:
            lines.append(
                f"- {r['policy_name']}: governance={r['governance_quality_impact']}, "
                f"pressure={r['constitutional_pressure_impact']}, risk={r['risk_posture_impact']}"
            )
    else:
        lines.append("_None assessed as beneficial this cycle._")

    bad = [r for r in assessed if r["impact_decision"] in (DECISION_HARMFUL, DECISION_UNCERTAIN)]
    lines.extend(["", "## Harmful or Uncertain Doctrines", ""])
    if bad:
        for r in bad:
            lines.append(f"- {r['policy_name']} ({r['impact_decision']}): {r['impact_rationale']}")
    else:
        lines.append("_None harmful or uncertain._")

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
            "Impact assessment evaluates governance consequences only. Simulated ≠ approved "
            "for activation. Beneficial ≠ runtime mutation. No runtime policy is changed. "
            "Constitutional law, court rulings, capital preservation doctrine, and operator "
            "supremacy remain supreme.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Impact memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    impact_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "impact_state": state,
        "beneficial_count": counts["beneficial"],
        "neutral_count": counts["neutral"],
        "harmful_count": counts["harmful"],
        "uncertain_count": counts["uncertain"],
        "impact_confidence": round(impact_confidence, 6),
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
        for c in IMPACT_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_doctrine_impact_assessment(
    *,
    simulation_summary: Dict[str, Any],
    simulation_record: Dict[str, Any],
    simulation_mem: List[Dict[str, str]],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    recovery_eff_summary: Dict[str, Any],
    drift_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    existing_impact_mem: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        simulation_summary=simulation_summary,
        simulation_record=simulation_record,
        simulation_mem=simulation_mem,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        recovery_eff_summary=recovery_eff_summary,
        drift_summary=drift_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
    )

    pre_restricted = (
        ctx["constitutional_pressure"] >= 0.55
        or ctx["constitution_violated"]
        or ctx["court_ruling"] == "COURT_OVERRULED"
    )
    assessed = _assess_all(ctx, restricted=pre_restricted)
    impact_confidence = _impact_confidence(ctx, assessed)
    counts = _count_decisions(assessed)

    state, reasons, restricted = _classify_impact_state(
        ctx=ctx,
        impact_confidence=impact_confidence,
        assessed=assessed,
        counts=counts,
    )

    if restricted != pre_restricted:
        assessed = _assess_all(ctx, restricted=restricted)
        impact_confidence = _impact_confidence(ctx, assessed)
        counts = _count_decisions(assessed)

    booleans = _impact_booleans(state, assessed, ctx, counts)
    recommendations = _recommendations(state, counts)
    rationale = _build_rationale(assessed, state)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        impact_confidence=impact_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(existing_impact_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        impact_confidence=impact_confidence,
        assessed=assessed,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_impact_assessment_engine",
        "engine_version": 1,
        "impact_state": state,
        "impact_confidence": impact_confidence,
        "impact_reasons": reasons,
        "doctrine_impacts": assessed,
        "decision_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "simulated_vs_beneficial_note": (
            "Simulated ≠ approved for activation. Beneficial ≠ runtime mutation. "
            "This engine assesses governance consequences only."
        ),
        "constitutional_supremacy_note": (
            "Impact assessment cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "impact_memory_size_after_append": len(merged_memory),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "impact_assessment_only": True,
        },
        "inputs_seen": {
            "arm_governance_doctrine_simulation_summary": bool(simulation_summary),
            "arm_governance_doctrine_simulation_record": bool(simulation_record),
            "arm_governance_doctrine_simulation_memory_rows": len(simulation_mem),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "arm_governance_recovery_effectiveness_summary": bool(recovery_eff_summary),
            "arm_governance_drift_detection_summary": bool(drift_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_impact_memory_rows": len(existing_impact_mem),
            "n_assessable_simulated_doctrines": len(_assessable_doctrines(ctx)),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_impact_assessment_engine",
        "impact_state": state,
        "impact_confidence": impact_confidence,
        "beneficial_doctrine_available": booleans["beneficial_doctrine_available"],
        "harmful_doctrine_detected": booleans["harmful_doctrine_detected"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "operator_review_required": booleans["operator_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "impact_memory_reliable": booleans["impact_memory_reliable"],
        "beneficial_count": counts["beneficial"],
        "neutral_count": counts["neutral"],
        "harmful_count": counts["harmful"],
        "uncertain_count": counts["uncertain"],
        "n_doctrines_assessed": len(assessed),
        "n_recommendations": len(recommendations),
        "impact_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance doctrine impact assessment engine (Step 44). "
            "Evaluates whether simulated doctrine would improve governance. "
            "Never mutates runtime policy. No broker calls."
        ),
    )
    p.add_argument("--simulation-summary", default=str(DEFAULT_SIMULATION_SUM))
    p.add_argument("--simulation-record", default=str(DEFAULT_SIMULATION_REC))
    p.add_argument("--simulation-mem", default=str(DEFAULT_SIMULATION_MEM))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUM))
    p.add_argument("--recovery-eff-summary", default=str(DEFAULT_RECOVERY_EFF_SUM))
    p.add_argument("--drift-summary", default=str(DEFAULT_DRIFT_SUM))
    p.add_argument("--court-summary", default=str(DEFAULT_COURT_SUMMARY))
    p.add_argument("--council-summary", default=str(DEFAULT_COUNCIL_SUMMARY))
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
        "[ARM_DOCTRINE_IMPACT] starting "
        "(read-only governance consequence analysis; no runtime mutation; no broker calls)",
        flush=True,
    )

    simulation_summary = _safe_read_json(
        Path(args.simulation_summary), label="arm_governance_doctrine_simulation_summary.json"
    )
    simulation_record = _safe_read_json(
        Path(args.simulation_record), label="arm_governance_doctrine_simulation.json"
    )
    simulation_mem = _safe_read_csv_rows(
        Path(args.simulation_mem), label="arm_governance_doctrine_simulation_memory.csv"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="autonomous_readiness_summary.json"
    )
    recovery_eff_summary = _safe_read_json(
        Path(args.recovery_eff_summary), label="arm_governance_recovery_effectiveness_summary.json"
    )
    drift_summary = _safe_read_json(
        Path(args.drift_summary), label="arm_governance_drift_detection_summary.json"
    )
    court_summary = _safe_read_json(
        Path(args.court_summary), label="arm_constitutional_court_summary.json"
    )
    council_summary = _safe_read_json(
        Path(args.council_summary), label="arm_supreme_governance_council_summary.json"
    )
    runtime_policy = _safe_read_json(
        Path(args.runtime_policy), label="runtime_policy_governed.json"
    )
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_doctrine_impact_assessment_memory.csv"
    )

    record, summary, md, merged_memory = build_doctrine_impact_assessment(
        simulation_summary=simulation_summary,
        simulation_record=simulation_record,
        simulation_mem=simulation_mem,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        recovery_eff_summary=recovery_eff_summary,
        drift_summary=drift_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        existing_impact_mem=existing_mem,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=IMPACT_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["decision_counts"]
    print(
        "[ARM_DOCTRINE_IMPACT] "
        f"state={record['impact_state']} "
        f"beneficial={counts['beneficial']} "
        f"harmful={counts['harmful']} "
        f"uncertain={counts['uncertain']} "
        f"confidence={record['impact_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_DOCTRINE_IMPACT_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[ARM_DOCTRINE_IMPACT_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_DOCTRINE_IMPACT_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
