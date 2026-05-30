"""
ARM Runtime Governance Enablement Recommendation Engine -- Step 57.

Reads:
    data/results/arm_runtime_governance_constitutional_eligibility_board_summary.json (Step 56)
    data/results/arm_runtime_governance_constitutional_eligibility_board.json         (Step 56)
    data/results/arm_runtime_governance_constitutional_eligibility_board_memory.csv   (Step 56)
    data/results/arm_runtime_governance_admission_board_summary.json                  (Step 55)
    data/results/arm_runtime_governance_readiness_gate_summary.json                   (Step 54)
    data/results/arm_governance_runtime_shadow_activation_summary.json                  (Step 53)
    data/results/autonomous_governance_scorecard.json                                 (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/arm_constitutional_court_summary.json                                (Step 33)
    data/results/arm_supreme_governance_council_summary.json                            (Step 34)
    data/results/runtime_policy_governed.json                                           (Step 18)

Writes:
    data/results/arm_runtime_governance_enablement_recommendation_engine.json
    data/results/arm_runtime_governance_enablement_recommendation_engine.md
    data/results/arm_runtime_governance_enablement_recommendation_engine_summary.json
    data/results/arm_runtime_governance_enablement_recommendation_engine_memory.csv
    data/results/arm_runtime_governance_enablement_recommendation_engine_memory.parquet

Purpose
-------
This engine answers:

    "Even if runtime governance is constitutionally eligible, should it actually be
    recommended for future enablement?"

It converts runtime constitutional eligibility into institutional runtime enablement
recommendation. This is the FINAL institutional recommendation layer before any
hypothetical runtime enablement discussion.
Constitutionally eligible != recommended. Recommended != enabled.
Enablement recommendation != runtime mutation. Recommendation NEVER enables runtime.

Recommendation state cascade
----------------------------
    1. RUNTIME_ENABLEMENT_RECOMMENDATION_INSTITUTIONAL  stable institutional recommendation
    2. RUNTIME_ENABLEMENT_RECOMMENDATION_READY            recommendation plausible
    3. RUNTIME_ENABLEMENT_RECOMMENDATION_LIMITED          limited recommendation only
    4. RUNTIME_ENABLEMENT_RECOMMENDATION_OBSERVE          observation only; no recommendation
    5. RUNTIME_ENABLEMENT_RECOMMENDATION_DORMANT          not constitutionally eligible

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
* Missing inputs warn-and-continue; defaults to RUNTIME_ENABLEMENT_RECOMMENDATION_DORMANT.
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

DEFAULT_ELIGIBILITY_SUM = (
    RESULTS_DIR / "arm_runtime_governance_constitutional_eligibility_board_summary.json"
)
DEFAULT_ELIGIBILITY_REC = (
    RESULTS_DIR / "arm_runtime_governance_constitutional_eligibility_board.json"
)
DEFAULT_ELIGIBILITY_MEM = (
    RESULTS_DIR / "arm_runtime_governance_constitutional_eligibility_board_memory.csv"
)
DEFAULT_ADMISSION_SUM = RESULTS_DIR / "arm_runtime_governance_admission_board_summary.json"
DEFAULT_READINESS_SUM = RESULTS_DIR / "arm_runtime_governance_readiness_gate_summary.json"
DEFAULT_SHADOW_SUM = RESULTS_DIR / "arm_governance_runtime_shadow_activation_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_AUTO = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_runtime_governance_enablement_recommendation_engine.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_runtime_governance_enablement_recommendation_engine.md"
DEFAULT_OUT_SUMMARY = (
    RESULTS_DIR / "arm_runtime_governance_enablement_recommendation_engine_summary.json"
)
DEFAULT_OUT_MEM_CSV = (
    RESULTS_DIR / "arm_runtime_governance_enablement_recommendation_engine_memory.csv"
)
DEFAULT_OUT_MEM_PQ = (
    RESULTS_DIR / "arm_runtime_governance_enablement_recommendation_engine_memory.parquet"
)


# -----------------------------------------------------------
# Recommendation state constants
# -----------------------------------------------------------
REC_DORMANT = "RUNTIME_ENABLEMENT_RECOMMENDATION_DORMANT"
REC_OBSERVE = "RUNTIME_ENABLEMENT_RECOMMENDATION_OBSERVE"
REC_LIMITED = "RUNTIME_ENABLEMENT_RECOMMENDATION_LIMITED"
REC_READY = "RUNTIME_ENABLEMENT_RECOMMENDATION_READY"
REC_INSTITUTIONAL = "RUNTIME_ENABLEMENT_RECOMMENDATION_INSTITUTIONAL"

CLASS_NOT_REC = "NOT_RECOMMENDED"
CLASS_OBSERVE = "OBSERVE_ENABLEMENT"
CLASS_LIMITED = "LIMITED_ENABLEMENT_RECOMMENDATION"
CLASS_FULL = "FULL_ENABLEMENT_RECOMMENDATION"

CE_NOT = "NOT_CONSTITUTIONALLY_ELIGIBLE"
CE_OBSERVE = "OBSERVE_CONSTITUTIONAL_ELIGIBILITY"
CE_LIMITED = "LIMITED_CONSTITUTIONAL_ELIGIBILITY"
CE_FULL = "FULL_CONSTITUTIONAL_ELIGIBILITY"

MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "recommendation_state",
    "observe_count",
    "limited_count",
    "full_count",
    "recommendation_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[RUNTIME_ENABLEMENT_RECOMMENDATION_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(MEMORY_COLUMNS))
        for col in ("recommendation_confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("observe_count", "limited_count", "full_count"):
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


def _scale_recommendation(classification: str) -> float:
    if classification == CLASS_FULL:
        return 1.0
    if classification == CLASS_LIMITED:
        return 0.55
    if classification == CLASS_OBSERVE:
        return 0.30
    return 0.0


def _court_constitutional_safety(
    court_summary: Dict[str, Any], council_summary: Dict[str, Any]
) -> float:
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    ruling = _norm_upper(court_summary.get("judicial_ruling"))
    if constitution_state == "CONSTITUTION_VIOLATED":
        return 0.0
    if ruling == "COURT_OVERRULED":
        return 0.25
    if constitution_state == "CONSTITUTION_COMPLIANT":
        return 1.0
    return 0.50


# -----------------------------------------------------------
# Context extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    eligibility_summary: Dict[str, Any],
    eligibility_record: Dict[str, Any],
    eligibility_mem: List[Dict[str, str]],
    admission_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    shadow_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    recommendation_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    ce = eligibility_record.get("constitutional_eligibility") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    court_safety = _court_constitutional_safety(court_summary, council_summary)

    ctx: Dict[str, Any] = {
        "eligibility_state": _norm_upper(
            eligibility_summary.get("constitutional_eligibility_state")
            or eligibility_record.get("constitutional_eligibility_state")
        ),
        "eligibility_confidence": _clamp(
            _to_float(
                eligibility_summary.get("constitutional_eligibility_confidence")
                or eligibility_record.get("constitutional_eligibility_confidence")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "eligibility_available": bool(
            eligibility_summary.get("runtime_constitutional_eligibility_available")
        ),
        "constitutional_eligibility": ce,
        "eligibility_classification": _norm_upper(
            ce.get("runtime_constitutional_eligibility_classification")
            or eligibility_summary.get("runtime_constitutional_eligibility_classification")
        ),
        "eligibility_score": _clamp(
            _to_float(
                ce.get("runtime_constitutional_eligibility_score")
                or eligibility_summary.get("runtime_constitutional_eligibility_score")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "admission_confidence": _clamp(
            _to_float(admission_summary.get("admission_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "readiness_confidence": _clamp(
            _to_float(readiness_summary.get("readiness_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "shadow_confidence": _clamp(
            _to_float(shadow_summary.get("shadow_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "governance_quality": _clamp(
            _to_float(scores.get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "constitutional_safe_eligibility": bool(ce.get("constitutional_safe")),
        "constitutional_safety": _clamp(
            court_safety * 0.60 + (1.0 if bool(ce.get("constitutional_safe")) else 0.0) * 0.40,
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
            or eligibility_summary.get("operator_review_required")
            or admission_summary.get("operator_review_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "autonomous_readiness_score": _clamp(
            _to_float(autonomous_readiness.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "eligibility_memory_depth": len(eligibility_mem),
        "recommendation_memory_depth": len(recommendation_mem),
        "observation_cycles": max(
            _to_float(eligibility_summary.get("observation_cycles")) or 0,
            len(eligibility_mem),
            1,
        ),
    }
    return ctx


# -----------------------------------------------------------
# Enablement recommendation assessment
# -----------------------------------------------------------
def _base_recommendation_score(ctx: Dict[str, Any]) -> float:
    raw = (
        ctx["eligibility_confidence"] * 0.24
        + ctx["eligibility_score"] * 0.20
        + ctx["admission_confidence"] * 0.14
        + ctx["readiness_confidence"] * 0.12
        + ctx["shadow_confidence"] * 0.10
        + ctx["governance_quality"] * 0.10
        + ctx["system_health_score"] * 0.06
        + ctx["constitutional_safety"] * 0.04
    )
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.22
    return round(_clamp(raw, 0.0, 1.0), 4)


def _compute_recommendation_score(classification: str, ctx: Dict[str, Any]) -> float:
    raw = _base_recommendation_score(ctx)
    raw *= _scale_recommendation(classification)
    return round(_clamp(raw, 0.0, 1.0), 4)


def _classify_enablement_recommendation(*, gate_score: float, ctx: Dict[str, Any]) -> str:
    ec = ctx["eligibility_classification"]
    const_safe = (
        ctx["constitutional_safe_eligibility"]
        and ctx["constitutional_safety"] >= 0.50
        and not ctx["constitution_violated"]
    )

    if not const_safe or ec == CE_NOT or gate_score < 0.15:
        return CLASS_NOT_REC

    if ec == CE_FULL and gate_score >= 0.44:
        if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
            return CLASS_LIMITED
        if ctx["system_health_stale"] or ctx["governance_quality"] < 0.45:
            return CLASS_LIMITED
        if ctx["constitutional_safety"] < 0.72:
            return CLASS_LIMITED
        return CLASS_FULL

    if ec == CE_LIMITED and gate_score >= 0.30:
        if const_safe:
            return CLASS_LIMITED

    if ec == CE_OBSERVE and gate_score >= 0.17:
        return CLASS_OBSERVE

    if gate_score >= 0.20 and ec != CE_NOT and const_safe:
        return CLASS_OBSERVE

    return CLASS_NOT_REC


def _recommendation_rationale(
    classification: str, recommendation_score: float, ctx: Dict[str, Any]
) -> str:
    templates = {
        CLASS_FULL: (
            "full enablement recommendation: runtime governance enablement is institutionally "
            "recommended for future consideration without runtime activation"
        ),
        CLASS_LIMITED: (
            "limited enablement recommendation: constitutionally safe defensive posture "
            "supports limited recommendation only"
        ),
        CLASS_OBSERVE: (
            "observe enablement: observation maturity only; no enablement recommendation"
        ),
        CLASS_NOT_REC: (
            "not recommended: constitutional pressure or eligibility immaturity blocks recommendation"
        ),
    }
    base = templates.get(classification, templates[CLASS_NOT_REC])
    if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
        base += " under elevated constitutional pressure"
    return f"{base} (recommendation_score={recommendation_score:.2f})"


def _build_enablement_recommendation(
    *,
    ctx: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    gate_score = _base_recommendation_score(ctx)
    if prior:
        prior_score = _to_float(prior.get("runtime_enablement_recommendation_score")) or gate_score
        gate_score = round(_clamp(gate_score * 0.65 + prior_score * 0.35, 0.0, 1.0), 4)

    classification = _classify_enablement_recommendation(gate_score=gate_score, ctx=ctx)
    recommendation_score = _compute_recommendation_score(classification, ctx)

    ce = ctx["constitutional_eligibility"]
    eligibility_class = ctx["eligibility_classification"]
    eligibility_score = ctx["eligibility_score"]

    const_safe = (
        ctx["constitutional_safe_eligibility"]
        and ctx["constitutional_safety"] >= 0.50
        and not ctx["constitution_violated"]
        and classification != CLASS_NOT_REC
    )
    institutionally_recommended = classification == CLASS_FULL
    future_candidate = classification in (CLASS_LIMITED, CLASS_FULL) or bool(
        ce.get("future_runtime_candidate")
    )

    conf = round(
        _clamp(
            gate_score * 0.38
            + ctx["eligibility_confidence"] * 0.32
            + ctx["constitutional_safety"] * 0.30,
            0.0,
            1.0,
        ),
        4,
    )

    return {
        "runtime_enablement_recommendation_classification": classification,
        "runtime_enablement_recommendation_score": recommendation_score,
        "runtime_constitutional_eligibility_classification": eligibility_class,
        "runtime_constitutional_eligibility_score": round(eligibility_score, 4),
        "institutionally_runtime_recommended": institutionally_recommended,
        "future_runtime_candidate": future_candidate,
        "constitutional_safe": const_safe,
        "confidence": conf,
        "runtime_mutation_allowed": False,
        "runtime_enablement_rationale": _recommendation_rationale(
            classification,
            recommendation_score,
            ctx,
        ),
    }


# -----------------------------------------------------------
# Recommendation confidence and state
# -----------------------------------------------------------
def _recommendation_confidence(
    ctx: Dict[str, Any],
    enablement_recommendation: Dict[str, Any],
) -> float:
    raw = (
        ctx["eligibility_confidence"] * 0.22
        + ctx["admission_confidence"] * 0.16
        + ctx["readiness_confidence"] * 0.14
        + ctx["shadow_confidence"] * 0.12
        + ctx["governance_quality"] * 0.14
        + ctx["system_health_score"] * 0.12
        + ctx["constitutional_safety"] * 0.10
    )
    raw += (
        _to_float(enablement_recommendation.get("runtime_enablement_recommendation_score")) or 0.0
    ) * 0.05

    penalty = ctx["constitutional_pressure"] * 0.32
    if ctx["constitution_violated"]:
        penalty += 0.12
    if ctx["court_ruling"] == "COURT_OVERRULED":
        penalty += 0.10
    if ctx["system_health_stale"]:
        penalty += 0.07
    if ctx["governance_quality"] < 0.45:
        penalty += 0.06
    if ctx["observation_cycles"] < 2:
        penalty += 0.06

    return round(_clamp(raw - penalty, 0.0, 1.0), 6)


def _classification_counts(classification: str) -> Dict[str, int]:
    return {
        "not_recommended": 1 if classification == CLASS_NOT_REC else 0,
        "observe": 1 if classification == CLASS_OBSERVE else 0,
        "limited": 1 if classification == CLASS_LIMITED else 0,
        "full": 1 if classification == CLASS_FULL else 0,
    }


def _classify_recommendation_state(
    *,
    ctx: Dict[str, Any],
    classification: str,
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if classification == CLASS_NOT_REC or not ctx["eligibility_available"]:
        reasons.append("runtime not constitutionally eligible; governance maturity insufficient")
        return REC_DORMANT, reasons

    if counts["full"] >= 1 and ctx["recommendation_memory_depth"] >= 2:
        reasons.append("mature governance process with stable institutional recommendation quality")
        return REC_INSTITUTIONAL, reasons

    if counts["full"] >= 1:
        reasons.append("governance maturity stable; enablement recommendation plausible")
        return REC_READY, reasons

    if counts["limited"] >= 1:
        reasons.append("defensive governance maturity; limited recommendation only")
        return REC_LIMITED, reasons

    if counts["observe"] >= 1:
        reasons.append("observation maturity only; no recommendation")
        return REC_OBSERVE, reasons

    reasons.append("runtime not constitutionally eligible; governance maturity insufficient")
    return REC_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations list, rationale
# -----------------------------------------------------------
def _recommendation_booleans(
    state: str,
    enablement_recommendation: Dict[str, Any],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    classification = enablement_recommendation.get(
        "runtime_enablement_recommendation_classification",
        CLASS_NOT_REC,
    )
    return {
        "runtime_enablement_recommendation_available": bool(enablement_recommendation),
        "limited_runtime_enablement_available": counts["limited"] > 0 or counts["full"] > 0,
        "full_runtime_enablement_available": counts["full"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"]
            or enablement_recommendation.get("future_runtime_candidate", False)
            or classification in (CLASS_LIMITED, CLASS_FULL)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "recommendation_memory_reliable": state == REC_INSTITUTIONAL,
    }


def _recommendations_list(state: str, counts: Dict[str, int]) -> List[str]:
    recs = [
        "Continue governance maturity accumulation",
        "Continue shadow activation rehearsal",
        "Maintain defensive constitutional posture",
        "Avoid premature runtime assumptions",
        "Maintain runtime mutation lock",
    ]
    if counts["full"] > 0 or counts["limited"] > 0:
        recs.append("Escalate enablement recommendation to operator review")
    if state == REC_DORMANT:
        recs.append("Resolve constitutional eligibility before enablement recommendation")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(
    enablement_recommendation: Dict[str, Any],
    state: str,
    ctx: Dict[str, Any],
) -> str:
    classification = enablement_recommendation.get(
        "runtime_enablement_recommendation_classification",
        CLASS_NOT_REC,
    )
    if classification == CLASS_NOT_REC or ctx["constitution_violated"]:
        return (
            "Triton does not recommend runtime governance enablement because constitutional "
            "pressure remains elevated despite improving governance maturity."
        )
    if classification == CLASS_FULL:
        return (
            "Triton institutionally recommends full runtime governance enablement for future "
            "consideration without activating or mutating live runtime."
        )
    if classification == CLASS_LIMITED:
        return (
            "Triton recommends limited runtime governance enablement under defensive conditions; "
            "full recommendation remains constrained."
        )
    if classification == CLASS_OBSERVE:
        return "Enablement recommendation remains observe-only; governance maturity is immature."
    return (
        "Runtime governance enablement recommendation completed without runtime mutation. "
        "Constitutionally eligible != recommended."
    )


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    recommendation_confidence: float,
    enablement_recommendation: Dict[str, Any],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
    ctx: Dict[str, Any],
) -> str:
    er = enablement_recommendation
    lines = [
        "# Triton Runtime Governance Enablement Recommendation",
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
        f"| classification | {er.get('runtime_enablement_recommendation_classification', CLASS_NOT_REC)} |",
        f"| recommendation_score | {_to_float(er.get('runtime_enablement_recommendation_score')) or 0.0:.3f} |",
        f"| eligibility_classification | {er.get('runtime_constitutional_eligibility_classification', CE_NOT)} |",
        f"| eligibility_score | {_to_float(er.get('runtime_constitutional_eligibility_score')) or 0.0:.3f} |",
        f"| institutionally_runtime_recommended | {er.get('institutionally_runtime_recommended', False)} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Runtime Enablement Recommendation",
        "",
        f"- **Classification:** {er.get('runtime_enablement_recommendation_classification', CLASS_NOT_REC)}",
        f"- **Constitutional safe:** {er.get('constitutional_safe', False)}",
        f"- **Future runtime candidate:** {er.get('future_runtime_candidate', False)}",
        f"- **Eligibility state:** {ctx.get('eligibility_state', 'UNKNOWN')}",
        "",
        f"_{er.get('runtime_enablement_rationale', '')}_",
        "",
        "## Runtime Governance Recommendation",
        "",
        f"| tier | count |",
        f"|---|---|",
        f"| observe | {counts['observe']} |",
        f"| limited | {counts['limited']} |",
        f"| full | {counts['full']} |",
        "",
        f"Eligibility confidence: {ctx['eligibility_confidence']:.3f} | "
        f"Admission confidence: {ctx['admission_confidence']:.3f} | "
        f"Constitutional safety: {ctx['constitutional_safety']:.3f} | "
        f"Regime: {ctx.get('regime', 'UNKNOWN')}",
        "",
        "## Recommendations",
        "",
    ]
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
            "This is the final institutional recommendation layer before any hypothetical runtime "
            "enablement discussion. Constitutionally eligible != recommended. Recommended != enabled. "
            "No live runtime policy is changed.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Memory
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
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
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
        for c in MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_enablement_recommendation(
    *,
    eligibility_summary: Dict[str, Any],
    eligibility_record: Dict[str, Any],
    eligibility_mem: List[Dict[str, str]],
    admission_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    shadow_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    recommendation_mem: List[Dict[str, str]],
    prior_recommendation_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        eligibility_summary=eligibility_summary,
        eligibility_record=eligibility_record,
        eligibility_mem=eligibility_mem,
        admission_summary=admission_summary,
        readiness_summary=readiness_summary,
        shadow_summary=shadow_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        autonomous_readiness=autonomous_readiness,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        recommendation_mem=recommendation_mem,
    )

    prior_er = prior_recommendation_record.get("enablement_recommendation") or {}
    enablement_recommendation = _build_enablement_recommendation(ctx=ctx, prior=prior_er or None)
    recommendation_confidence = _recommendation_confidence(ctx, enablement_recommendation)
    counts = _classification_counts(
        enablement_recommendation.get(
            "runtime_enablement_recommendation_classification",
            CLASS_NOT_REC,
        )
    )

    state, reasons = _classify_recommendation_state(
        ctx=ctx,
        classification=enablement_recommendation.get(
            "runtime_enablement_recommendation_classification",
            CLASS_NOT_REC,
        ),
        counts=counts,
    )

    booleans = _recommendation_booleans(state, enablement_recommendation, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(enablement_recommendation, state, ctx)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        recommendation_confidence=recommendation_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(recommendation_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        recommendation_confidence=recommendation_confidence,
        enablement_recommendation=enablement_recommendation,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
        ctx=ctx,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_enablement_recommendation_engine",
        "engine_version": 1,
        "recommendation_state": state,
        "recommendation_confidence": recommendation_confidence,
        "recommendation_reasons": reasons,
        "enablement_recommendation": enablement_recommendation,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "eligible_vs_recommended_note": (
            "Constitutionally eligible != recommended. Recommended != enabled. "
            "Enablement recommendation != runtime mutation. Live runtime is never changed."
        ),
        "constitutional_supremacy_note": (
            "Runtime recommendation cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "recommendation_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "live_runtime_regime": ctx["regime"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutated": False,
            "runtime_mutation_allowed": False,
            "recommendation_evaluation_only": True,
        },
        "inputs_seen": {
            "arm_runtime_governance_constitutional_eligibility_board_summary": bool(
                eligibility_summary
            ),
            "arm_runtime_governance_constitutional_eligibility_board_record": bool(
                eligibility_record
            ),
            "arm_runtime_governance_constitutional_eligibility_board_memory_rows": len(
                eligibility_mem
            ),
            "arm_runtime_governance_admission_board_summary": bool(admission_summary),
            "arm_runtime_governance_readiness_gate_summary": bool(readiness_summary),
            "arm_governance_runtime_shadow_activation_summary": bool(shadow_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(autonomous_readiness),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_recommendation_memory_rows": len(recommendation_mem),
            "prior_enablement_recommendation": bool(prior_er),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_enablement_recommendation_engine",
        "recommendation_state": state,
        "recommendation_confidence": recommendation_confidence,
        "runtime_enablement_recommendation_available": booleans[
            "runtime_enablement_recommendation_available"
        ],
        "limited_runtime_enablement_available": booleans["limited_runtime_enablement_available"],
        "full_runtime_enablement_available": booleans["full_runtime_enablement_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "recommendation_memory_reliable": booleans["recommendation_memory_reliable"],
        "not_recommended_count": counts["not_recommended"],
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "observation_cycles": ctx["observation_cycles"],
        "runtime_enablement_recommendation_classification": enablement_recommendation.get(
            "runtime_enablement_recommendation_classification",
        ),
        "runtime_enablement_recommendation_score": enablement_recommendation.get(
            "runtime_enablement_recommendation_score",
        ),
        "runtime_constitutional_eligibility_classification": enablement_recommendation.get(
            "runtime_constitutional_eligibility_classification",
        ),
        "institutionally_runtime_recommended": enablement_recommendation.get(
            "institutionally_runtime_recommended",
        ),
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
            "Read-only ARM runtime governance enablement recommendation engine (Step 57). "
            "Final recommendation layer without mutating runtime. No broker calls."
        ),
    )
    p.add_argument("--eligibility-summary", default=str(DEFAULT_ELIGIBILITY_SUM))
    p.add_argument("--eligibility-record", default=str(DEFAULT_ELIGIBILITY_REC))
    p.add_argument("--eligibility-mem", default=str(DEFAULT_ELIGIBILITY_MEM))
    p.add_argument("--admission-summary", default=str(DEFAULT_ADMISSION_SUM))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUM))
    p.add_argument("--shadow-summary", default=str(DEFAULT_SHADOW_SUM))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--autonomous-readiness", default=str(DEFAULT_READINESS_AUTO))
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
        "[RUNTIME_ENABLEMENT_RECOMMENDATION] starting "
        "(read-only recommendation evaluation; no runtime mutation; no broker calls)",
        flush=True,
    )

    eligibility_summary = _safe_read_json(
        Path(args.eligibility_summary),
        label="arm_runtime_governance_constitutional_eligibility_board_summary.json",
    )
    eligibility_record = _safe_read_json(
        Path(args.eligibility_record),
        label="arm_runtime_governance_constitutional_eligibility_board.json",
    )
    eligibility_mem = _safe_read_csv_rows(
        Path(args.eligibility_mem),
        label="arm_runtime_governance_constitutional_eligibility_board_memory.csv",
    )
    admission_summary = _safe_read_json(
        Path(args.admission_summary), label="arm_runtime_governance_admission_board_summary.json"
    )
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="arm_runtime_governance_readiness_gate_summary.json"
    )
    shadow_summary = _safe_read_json(
        Path(args.shadow_summary), label="arm_governance_runtime_shadow_activation_summary.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    autonomous_readiness = _safe_read_json(
        Path(args.autonomous_readiness), label="autonomous_readiness_summary.json"
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
    recommendation_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv),
        label="arm_runtime_governance_enablement_recommendation_engine_memory.csv",
    )
    prior_recommendation_record = _safe_read_json(
        Path(args.out_json),
        label="prior_arm_runtime_governance_enablement_recommendation_engine.json",
    )

    record, summary, md, merged_memory = build_enablement_recommendation(
        eligibility_summary=eligibility_summary,
        eligibility_record=eligibility_record,
        eligibility_mem=eligibility_mem,
        admission_summary=admission_summary,
        readiness_summary=readiness_summary,
        shadow_summary=shadow_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        autonomous_readiness=autonomous_readiness,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        recommendation_mem=recommendation_mem,
        prior_recommendation_record=prior_recommendation_record,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["classification_counts"]
    print(
        "[RUNTIME_ENABLEMENT_RECOMMENDATION] "
        f"state={record['recommendation_state']} "
        f"limited={counts['limited']} "
        f"full={counts['full']} "
        f"confidence={record['recommendation_confidence']:.3f}",
        flush=True,
    )
    print(
        "[RUNTIME_ENABLEMENT_RECOMMENDATION_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[RUNTIME_ENABLEMENT_RECOMMENDATION_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[RUNTIME_ENABLEMENT_RECOMMENDATION_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
