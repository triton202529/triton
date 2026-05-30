"""
ARM Governance Doctrine Evidence Accumulation Engine -- Step 46.

Reads:
    data/results/arm_governance_doctrine_recommendation_summary.json      (Step 45)
    data/results/arm_governance_doctrine_recommendation.json          (Step 45)
    data/results/arm_governance_doctrine_recommendation_memory.csv      (Step 45)
    data/results/arm_governance_doctrine_impact_assessment_summary.json (Step 44)
    data/results/arm_governance_doctrine_simulation_summary.json        (Step 43)
    data/results/arm_governance_doctrine_activation_board_summary.json  (Step 42)
    data/results/arm_constitutional_court_summary.json                  (Step 33)
    data/results/arm_supreme_governance_council_summary.json           (Step 34)
    data/results/autonomous_governance_scorecard.json                   (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/runtime_policy_governed.json                           (Step 18)

Writes:
    data/results/arm_governance_doctrine_evidence_accumulation.json
    data/results/arm_governance_doctrine_evidence_accumulation.md
    data/results/arm_governance_doctrine_evidence_accumulation_summary.json
    data/results/arm_governance_doctrine_evidence_accumulation_memory.csv
    data/results/arm_governance_doctrine_evidence_accumulation_memory.parquet

Purpose
-------
This engine answers:

    "When has enough evidence accumulated to justify serious future activation consideration?"

It accumulates institutional evidence over time for governance doctrine recommendations.
Recommended != institutionally proven. Evidence accumulation != activation approval.
Evidence NEVER activates doctrine or mutates runtime policy.

Evidence state cascade
----------------------
    1. DOCTRINE_EVIDENCE_INSTITUTIONAL  long-term stable evidence; institutional reliability
    2. DOCTRINE_EVIDENCE_STRONG           repeatable benefit; constitutional safety persistent
    3. DOCTRINE_EVIDENCE_EMERGING         recommendation consistency; beneficial recurrence
    4. DOCTRINE_EVIDENCE_FORMING          repeated signals beginning; weak repeatability
    5. DOCTRINE_EVIDENCE_DORMANT          insufficient observations

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
* Append-only evidence memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to DOCTRINE_EVIDENCE_DORMANT.
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

DEFAULT_RECOMMENDATION_SUM = RESULTS_DIR / "arm_governance_doctrine_recommendation_summary.json"
DEFAULT_RECOMMENDATION_REC = RESULTS_DIR / "arm_governance_doctrine_recommendation.json"
DEFAULT_RECOMMENDATION_MEM = RESULTS_DIR / "arm_governance_doctrine_recommendation_memory.csv"
DEFAULT_IMPACT_SUM = RESULTS_DIR / "arm_governance_doctrine_impact_assessment_summary.json"
DEFAULT_SIMULATION_SUM = RESULTS_DIR / "arm_governance_doctrine_simulation_summary.json"
DEFAULT_ACTIVATION_SUM = RESULTS_DIR / "arm_governance_doctrine_activation_board_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_SUM = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_doctrine_evidence_accumulation.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_doctrine_evidence_accumulation.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_doctrine_evidence_accumulation_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_doctrine_evidence_accumulation_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_doctrine_evidence_accumulation_memory.parquet"


# -----------------------------------------------------------
# Evidence state constants
# -----------------------------------------------------------
EVIDENCE_DORMANT = "DOCTRINE_EVIDENCE_DORMANT"
EVIDENCE_FORMING = "DOCTRINE_EVIDENCE_FORMING"
EVIDENCE_EMERGING = "DOCTRINE_EVIDENCE_EMERGING"
EVIDENCE_STRONG = "DOCTRINE_EVIDENCE_STRONG"
EVIDENCE_INSTITUTIONAL = "DOCTRINE_EVIDENCE_INSTITUTIONAL"

CLASS_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
CLASS_EMERGING = "EMERGING_EVIDENCE"
CLASS_STRONG = "STRONG_EVIDENCE"
CLASS_INSTITUTIONAL = "INSTITUTIONAL_EVIDENCE"

RECOMMEND_DECISION = "RECOMMEND_FOR_FUTURE_CONSIDERATION"
MONITOR_DECISION = "MONITOR_ONLY"
IMPACT_BENEFICIAL = "BENEFICIAL"

EVIDENCE_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "evidence_state",
    "emerging_count",
    "strong_count",
    "institutional_count",
    "evidence_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_DOCTRINE_EVIDENCE_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(EVIDENCE_MEMORY_COLUMNS))
        for col in ("evidence_confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("emerging_count", "strong_count", "institutional_count"):
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
    recommendation_summary: Dict[str, Any],
    recommendation_record: Dict[str, Any],
    recommendation_mem: List[Dict[str, str]],
    impact_summary: Dict[str, Any],
    simulation_summary: Dict[str, Any],
    activation_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    evidence_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    ctx: Dict[str, Any] = {
        "recommendation_state": _norm_upper(
            recommendation_summary.get("recommendation_state")
            or recommendation_record.get("recommendation_state")
        ),
        "recommendation_confidence": _clamp(
            _to_float(recommendation_summary.get("recommendation_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "impact_confidence": _clamp(
            _to_float(impact_summary.get("impact_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "simulation_confidence": _clamp(
            _to_float(simulation_summary.get("simulation_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "doctrine_recommendations": recommendation_record.get("doctrine_recommendations") or [],
        "recommendation_available": bool(
            recommendation_summary.get("doctrine_recommendation_available")
        ),
        "recommendation_memory_depth": len(recommendation_mem),
        "evidence_memory_depth": len(evidence_mem),
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
            or recommendation_summary.get("operator_review_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "readiness_score": _clamp(
            _to_float(readiness_summary.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "observation_cycles": max(len(recommendation_mem), len(evidence_mem), 1),
    }
    ctx["court_stability"] = _court_stability_score(ctx)
    return ctx


def _prior_doctrine_map(prior_record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in prior_record.get("doctrine_evidence") or []:
        name = str(row.get("policy_name", ""))
        if name:
            out[name] = row
    return out


# -----------------------------------------------------------
# Evidence accumulation per doctrine
# -----------------------------------------------------------
def _is_beneficial_observation(rec: Dict[str, Any]) -> bool:
    impact = _norm_upper(rec.get("impact_decision"))
    decision = _norm_upper(rec.get("recommendation_decision"))
    return (
        impact == IMPACT_BENEFICIAL
        or decision in (RECOMMEND_DECISION, MONITOR_DECISION)
        and impact == IMPACT_BENEFICIAL
    )


def _is_recommend_observation(rec: Dict[str, Any]) -> bool:
    return _norm_upper(rec.get("recommendation_decision")) == RECOMMEND_DECISION


def _classify_doctrine(
    *,
    observation_count: int,
    beneficial_frequency: float,
    recommendation_consistency: float,
    constitutional_safety_stability: float,
    governance_improvement_persistence: float,
    ctx: Dict[str, Any],
) -> str:
    if observation_count < 2 or beneficial_frequency < 0.40:
        return CLASS_INSUFFICIENT
    if (
        observation_count >= 5
        and beneficial_frequency >= 0.75
        and recommendation_consistency >= 0.75
        and constitutional_safety_stability >= 0.85
        and ctx["evidence_memory_depth"] >= 3
    ):
        return CLASS_INSTITUTIONAL
    if (
        observation_count >= 3
        and beneficial_frequency >= 0.65
        and recommendation_consistency >= 0.65
        and constitutional_safety_stability >= 0.80
        and governance_improvement_persistence >= 0.55
    ):
        return CLASS_STRONG
    if (
        observation_count >= 2
        and beneficial_frequency >= 0.50
        and recommendation_consistency >= 0.50
    ):
        return CLASS_EMERGING
    return CLASS_INSUFFICIENT


def _accumulate_doctrine(
    rec: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
    *,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(rec.get("policy_name", ""))
    conf = _to_float(rec.get("confidence")) or 0.0
    const_safe = bool(rec.get("constitutional_safe", True))

    prior_obs = int(_to_float(prior.get("observation_count")) or 0) if prior else 0
    prior_beneficial = _to_float(prior.get("beneficial_frequency")) or 0.0 if prior else 0.0
    prior_recommend = _to_float(prior.get("recommendation_consistency")) or 0.0 if prior else 0.0
    prior_const = _to_float(prior.get("constitutional_safety_stability")) or 0.0 if prior else 0.0
    prior_gov = _to_float(prior.get("governance_improvement_persistence")) or 0.0 if prior else 0.0

    observation_count = prior_obs + 1
    beneficial_now = 1.0 if _is_beneficial_observation(rec) else 0.0
    recommend_now = 1.0 if _is_recommend_observation(rec) else 0.0
    const_now = 1.0 if const_safe else 0.0
    gov_now = 1.0 if _norm_upper(rec.get("impact_decision")) == IMPACT_BENEFICIAL else 0.0

    beneficial_frequency = (
        (prior_beneficial * prior_obs + beneficial_now) / observation_count
        if observation_count
        else 0.0
    )
    recommendation_consistency = (
        (prior_recommend * prior_obs + recommend_now) / observation_count
        if observation_count
        else 0.0
    )
    constitutional_safety_stability = (
        (prior_const * prior_obs + const_now) / observation_count if observation_count else 0.0
    )
    governance_improvement_persistence = (
        (prior_gov * prior_obs + gov_now) / observation_count if observation_count else 0.0
    )

    classification = _classify_doctrine(
        observation_count=observation_count,
        beneficial_frequency=beneficial_frequency,
        recommendation_consistency=recommendation_consistency,
        constitutional_safety_stability=constitutional_safety_stability,
        governance_improvement_persistence=governance_improvement_persistence,
        ctx=ctx,
    )

    future_candidate = (
        classification in (CLASS_STRONG, CLASS_INSTITUTIONAL)
        and _is_recommend_observation(rec)
        and const_safe
    )

    rationale_map = {
        CLASS_INSTITUTIONAL: (
            f"institutional evidence: {name} shows persistent repeatable governance benefit"
        ),
        CLASS_STRONG: (f"strong evidence: {name} demonstrates stable beneficial recurrence"),
        CLASS_EMERGING: (f"emerging evidence: {name} shows increasing recommendation consistency"),
        CLASS_INSUFFICIENT: (
            f"insufficient evidence: {name} requires more observations for institutional proof"
        ),
    }

    return {
        "policy_name": name,
        "evidence_classification": classification,
        "observation_count": observation_count,
        "beneficial_frequency": round(beneficial_frequency, 4),
        "recommendation_consistency": round(recommendation_consistency, 4),
        "constitutional_safety_stability": round(constitutional_safety_stability, 4),
        "governance_improvement_persistence": round(governance_improvement_persistence, 4),
        "future_activation_candidate": future_candidate,
        "confidence": round(conf, 4),
        "runtime_mutation_allowed": False,
        "evidence_rationale": rationale_map.get(classification, rationale_map[CLASS_INSUFFICIENT]),
        "current_recommendation_decision": rec.get("recommendation_decision"),
        "current_impact_decision": rec.get("impact_decision"),
    }


def _accumulate_all(
    ctx: Dict[str, Any],
    prior_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    accumulated: List[Dict[str, Any]] = []
    seen: set = set()
    for rec in ctx["doctrine_recommendations"]:
        name = str(rec.get("policy_name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        accumulated.append(_accumulate_doctrine(rec, prior_map.get(name), ctx=ctx))

    # Carry forward prior doctrines not in current recommendations (no new observation)
    for name, prior in prior_map.items():
        if name not in seen:
            accumulated.append(
                {
                    "policy_name": name,
                    "evidence_classification": prior.get(
                        "evidence_classification", CLASS_INSUFFICIENT
                    ),
                    "observation_count": int(_to_float(prior.get("observation_count")) or 0),
                    "beneficial_frequency": _to_float(prior.get("beneficial_frequency")) or 0.0,
                    "recommendation_consistency": _to_float(prior.get("recommendation_consistency"))
                    or 0.0,
                    "constitutional_safety_stability": _to_float(
                        prior.get("constitutional_safety_stability")
                    )
                    or 0.0,
                    "governance_improvement_persistence": _to_float(
                        prior.get("governance_improvement_persistence")
                    )
                    or 0.0,
                    "future_activation_candidate": False,
                    "confidence": _to_float(prior.get("confidence")) or 0.0,
                    "runtime_mutation_allowed": False,
                    "evidence_rationale": "prior evidence retained; no new observation this cycle",
                    "current_recommendation_decision": None,
                    "current_impact_decision": None,
                }
            )
    return accumulated


# -----------------------------------------------------------
# Evidence confidence and state
# -----------------------------------------------------------
def _evidence_confidence(ctx: Dict[str, Any], accumulated: List[Dict[str, Any]]) -> float:
    if not accumulated:
        return 0.0

    avg_beneficial = sum(r["beneficial_frequency"] for r in accumulated) / len(accumulated)
    avg_recommend = sum(r["recommendation_consistency"] for r in accumulated) / len(accumulated)
    avg_gov = sum(r["governance_improvement_persistence"] for r in accumulated) / len(accumulated)
    avg_const = sum(r["constitutional_safety_stability"] for r in accumulated) / len(accumulated)

    raw = (
        avg_recommend * 0.25
        + avg_beneficial * 0.22
        + avg_gov * 0.18
        + avg_const * 0.15
        + ctx["system_health_score"] * 0.10
        + ctx["readiness_score"] * 0.10
    )

    penalty = ctx["constitutional_pressure"] * 0.18
    if ctx["constitution_violated"]:
        penalty += 0.10
    if ctx["court_ruling"] == "COURT_OVERRULED":
        penalty += 0.08
    if ctx["system_health_stale"]:
        penalty += 0.07
    if ctx["observation_cycles"] < 2:
        penalty += 0.08

    return round(_clamp(raw - penalty, 0.0, 1.0), 6)


def _count_classifications(accumulated: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "emerging": sum(1 for r in accumulated if r["evidence_classification"] == CLASS_EMERGING),
        "strong": sum(1 for r in accumulated if r["evidence_classification"] == CLASS_STRONG),
        "institutional": sum(
            1 for r in accumulated if r["evidence_classification"] == CLASS_INSTITUTIONAL
        ),
        "insufficient": sum(
            1 for r in accumulated if r["evidence_classification"] == CLASS_INSUFFICIENT
        ),
    }


def _classify_evidence_state(
    *,
    ctx: Dict[str, Any],
    evidence_confidence: float,
    accumulated: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not accumulated or not ctx["recommendation_available"]:
        reasons.append("insufficient observations for doctrine evidence accumulation")
        return EVIDENCE_DORMANT, reasons

    if counts["institutional"] >= 1 and ctx["evidence_memory_depth"] >= 3:
        reasons.append("long-term stable evidence with institutional reliability achieved")
        return EVIDENCE_INSTITUTIONAL, reasons

    if counts["strong"] >= 1 or (counts["emerging"] >= 2 and evidence_confidence >= 0.45):
        reasons.append("repeatable benefit observed with persistent constitutional safety")
        return EVIDENCE_STRONG, reasons

    if counts["emerging"] >= 1 or (counts["insufficient"] >= 1 and ctx["observation_cycles"] >= 2):
        reasons.append("recommendation consistency visible; beneficial effects recurring")
        return EVIDENCE_EMERGING, reasons

    if ctx["observation_cycles"] >= 1 or counts["insufficient"] >= 1:
        reasons.append("repeated signals beginning; evidence repeatability weak")
        return EVIDENCE_FORMING, reasons

    reasons.append("insufficient observations for doctrine evidence accumulation")
    return EVIDENCE_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _evidence_booleans(
    state: str,
    accumulated: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    return {
        "doctrine_evidence_available": len(accumulated) > 0,
        "strong_evidence_available": counts["strong"] > 0 or counts["institutional"] > 0,
        "institutional_evidence_available": counts["institutional"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"]
            or any(r.get("future_activation_candidate") for r in accumulated)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "evidence_memory_reliable": state == EVIDENCE_INSTITUTIONAL,
    }


def _recommendations_list(state: str, counts: Dict[str, int]) -> List[str]:
    recs = [
        "Continue evidence collection",
        "Maintain defensive doctrine observation",
        "Avoid premature activation consideration",
        "Maintain runtime mutation lock",
    ]
    if counts["strong"] > 0 or counts["institutional"] > 0:
        recs.append("Escalate strong evidence doctrine to operator review")
    if state == EVIDENCE_DORMANT:
        recs.append("Accumulate more recommendation cycles before institutional proof")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(accumulated: List[Dict[str, Any]], state: str) -> str:
    cash = [
        r
        for r in accumulated
        if r["policy_name"] == "target_cash_pct"
        and r["evidence_classification"] in (CLASS_EMERGING, CLASS_STRONG, CLASS_INSTITUTIONAL)
    ]
    if cash:
        return (
            "Triton accumulated repeated evidence that elevated cash doctrine consistently "
            "improves governance stability during constitutional stress."
        )
    strong = [
        r
        for r in accumulated
        if r["evidence_classification"] in (CLASS_STRONG, CLASS_INSTITUTIONAL)
    ]
    if strong:
        names = ", ".join(r["policy_name"] for r in strong[:3])
        return f"Triton accumulated strong institutional evidence for: {names}."
    if state == EVIDENCE_FORMING:
        return "Evidence accumulation is forming; repeated observations are beginning."
    return "Governance doctrine evidence accumulation completed without runtime mutation."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    evidence_confidence: float,
    accumulated: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Doctrine Evidence Accumulation",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Evidence State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| evidence_confidence | {evidence_confidence:.3f} |",
        f"| emerging | {counts['emerging']} |",
        f"| strong | {counts['strong']} |",
        f"| institutional | {counts['institutional']} |",
        f"| insufficient | {counts['insufficient']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Doctrine Evidence",
        "",
    ]
    if accumulated:
        lines.append(
            "| policy | classification | observations | beneficial | consistency | confidence |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in accumulated:
            lines.append(
                f"| {r['policy_name']} | {r['evidence_classification']} | {r['observation_count']} | "
                f"{r['beneficial_frequency']:.2f} | {r['recommendation_consistency']:.2f} | {r['confidence']:.2f} |"
            )
        lines.append("")
        for r in accumulated:
            lines.append(
                f"- **{r['policy_name']}** ({r['evidence_classification']}): {r['evidence_rationale']}"
            )
    else:
        lines.append("_No doctrine evidence accumulated this cycle._")

    strong_inst = [
        r
        for r in accumulated
        if r["evidence_classification"] in (CLASS_STRONG, CLASS_INSTITUTIONAL)
    ]
    lines.extend(["", "## Strong or Institutional Evidence", ""])
    if strong_inst:
        for r in strong_inst:
            lines.append(
                f"- {r['policy_name']}: {r['evidence_classification']} "
                f"(observations={r['observation_count']}, beneficial={r['beneficial_frequency']:.2f})"
            )
    else:
        lines.append("_No strong or institutional evidence yet._")

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
            "Evidence accumulation is governance observation only. Recommended != institutionally proven. "
            "Evidence accumulation != activation approval. Strong evidence != runtime mutation. "
            "No runtime policy is changed.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Evidence memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    evidence_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "evidence_state": state,
        "emerging_count": counts["emerging"],
        "strong_count": counts["strong"],
        "institutional_count": counts["institutional"],
        "evidence_confidence": round(evidence_confidence, 6),
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
        for c in EVIDENCE_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_doctrine_evidence_accumulation(
    *,
    recommendation_summary: Dict[str, Any],
    recommendation_record: Dict[str, Any],
    recommendation_mem: List[Dict[str, str]],
    impact_summary: Dict[str, Any],
    simulation_summary: Dict[str, Any],
    activation_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    evidence_mem: List[Dict[str, str]],
    prior_evidence_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        recommendation_summary=recommendation_summary,
        recommendation_record=recommendation_record,
        recommendation_mem=recommendation_mem,
        impact_summary=impact_summary,
        simulation_summary=simulation_summary,
        activation_summary=activation_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        runtime_policy=runtime_policy,
        evidence_mem=evidence_mem,
    )

    prior_map = _prior_doctrine_map(prior_evidence_record)
    accumulated = _accumulate_all(ctx, prior_map)
    evidence_confidence = _evidence_confidence(ctx, accumulated)
    counts = _count_classifications(accumulated)

    state, reasons = _classify_evidence_state(
        ctx=ctx,
        evidence_confidence=evidence_confidence,
        accumulated=accumulated,
        counts=counts,
    )

    booleans = _evidence_booleans(state, accumulated, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(accumulated, state)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        evidence_confidence=evidence_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(evidence_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        evidence_confidence=evidence_confidence,
        accumulated=accumulated,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_evidence_accumulation_engine",
        "engine_version": 1,
        "evidence_state": state,
        "evidence_confidence": evidence_confidence,
        "evidence_reasons": reasons,
        "doctrine_evidence": accumulated,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "recommended_vs_proven_note": (
            "Recommended != institutionally proven. Evidence accumulation != activation approval. "
            "Strong evidence != runtime mutation. Evidence never activates doctrine."
        ),
        "constitutional_supremacy_note": (
            "Evidence accumulation cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "evidence_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "evidence_accumulation_only": True,
        },
        "inputs_seen": {
            "arm_governance_doctrine_recommendation_summary": bool(recommendation_summary),
            "arm_governance_doctrine_recommendation_record": bool(recommendation_record),
            "arm_governance_doctrine_recommendation_memory_rows": len(recommendation_mem),
            "arm_governance_doctrine_impact_assessment_summary": bool(impact_summary),
            "arm_governance_doctrine_simulation_summary": bool(simulation_summary),
            "arm_governance_doctrine_activation_board_summary": bool(activation_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_evidence_memory_rows": len(evidence_mem),
            "prior_doctrine_evidence_entries": len(prior_map),
            "n_doctrines_accumulated": len(accumulated),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_evidence_accumulation_engine",
        "evidence_state": state,
        "evidence_confidence": evidence_confidence,
        "doctrine_evidence_available": booleans["doctrine_evidence_available"],
        "strong_evidence_available": booleans["strong_evidence_available"],
        "institutional_evidence_available": booleans["institutional_evidence_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "evidence_memory_reliable": booleans["evidence_memory_reliable"],
        "emerging_count": counts["emerging"],
        "strong_count": counts["strong"],
        "institutional_count": counts["institutional"],
        "insufficient_count": counts["insufficient"],
        "observation_cycles": ctx["observation_cycles"],
        "n_doctrines_tracked": len(accumulated),
        "n_recommendations": len(recommendations),
        "evidence_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance doctrine evidence accumulation engine (Step 46). "
            "Accumulates institutional evidence for doctrine recommendations. "
            "Never mutates runtime policy. No broker calls."
        ),
    )
    p.add_argument("--recommendation-summary", default=str(DEFAULT_RECOMMENDATION_SUM))
    p.add_argument("--recommendation-record", default=str(DEFAULT_RECOMMENDATION_REC))
    p.add_argument("--recommendation-mem", default=str(DEFAULT_RECOMMENDATION_MEM))
    p.add_argument("--impact-summary", default=str(DEFAULT_IMPACT_SUM))
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
        "[ARM_DOCTRINE_EVIDENCE] starting "
        "(read-only evidence accumulation; no runtime mutation; no broker calls)",
        flush=True,
    )

    recommendation_summary = _safe_read_json(
        Path(args.recommendation_summary),
        label="arm_governance_doctrine_recommendation_summary.json",
    )
    recommendation_record = _safe_read_json(
        Path(args.recommendation_record), label="arm_governance_doctrine_recommendation.json"
    )
    recommendation_mem = _safe_read_csv_rows(
        Path(args.recommendation_mem), label="arm_governance_doctrine_recommendation_memory.csv"
    )
    impact_summary = _safe_read_json(
        Path(args.impact_summary), label="arm_governance_doctrine_impact_assessment_summary.json"
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
    evidence_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_doctrine_evidence_accumulation_memory.csv"
    )
    prior_evidence_record = _safe_read_json(
        Path(args.out_json), label="prior_arm_governance_doctrine_evidence_accumulation.json"
    )

    record, summary, md, merged_memory = build_doctrine_evidence_accumulation(
        recommendation_summary=recommendation_summary,
        recommendation_record=recommendation_record,
        recommendation_mem=recommendation_mem,
        impact_summary=impact_summary,
        simulation_summary=simulation_summary,
        activation_summary=activation_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        runtime_policy=runtime_policy,
        evidence_mem=evidence_mem,
        prior_evidence_record=prior_evidence_record,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=EVIDENCE_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["classification_counts"]
    print(
        "[ARM_DOCTRINE_EVIDENCE] "
        f"state={record['evidence_state']} "
        f"strong={counts['strong']} "
        f"institutional={counts['institutional']} "
        f"confidence={record['evidence_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_DOCTRINE_EVIDENCE_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[ARM_DOCTRINE_EVIDENCE_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_DOCTRINE_EVIDENCE_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
