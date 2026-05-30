"""
ARM Runtime Governance Human Escalation Dossier Engine -- Step 60.

Reads:
    data/results/arm_runtime_governance_institutional_verdict_engine_summary.json (Step 59)
    data/results/arm_runtime_governance_institutional_verdict_engine.json         (Step 59)
    data/results/arm_runtime_governance_institutional_verdict_engine_memory.csv   (Step 59)
    data/results/arm_runtime_governance_enablement_review_board_summary.json    (Step 58)
    data/results/arm_runtime_governance_enablement_recommendation_engine_summary.json (Step 57)
    data/results/arm_runtime_governance_constitutional_eligibility_board_summary.json (Step 56)
    data/results/arm_runtime_governance_admission_board_summary.json              (Step 55)
    data/results/arm_runtime_governance_readiness_gate_summary.json               (Step 54)
    data/results/arm_governance_runtime_shadow_activation_summary.json            (Step 53)
    data/results/autonomous_governance_scorecard.json                             (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/arm_constitutional_court_summary.json                            (Step 33)
    data/results/arm_supreme_governance_council_summary.json                      (Step 34)
    data/results/runtime_policy_governed.json                                     (Step 18)

Writes:
    data/results/arm_runtime_governance_human_escalation_dossier.json
    data/results/arm_runtime_governance_human_escalation_dossier.md
    data/results/arm_runtime_governance_human_escalation_dossier_summary.json
    data/results/arm_runtime_governance_human_escalation_dossier_memory.csv
    data/results/arm_runtime_governance_human_escalation_dossier_memory.parquet

Purpose
-------
This engine answers:

    "If a human operator must decide, what complete institutional case should Triton present?"

It converts Triton's institutional runtime governance verdict into a complete operator-facing
governance dossier. This is the FINAL governance communication layer before any hypothetical
human/operator judgment. Verdict != human decision. Dossier != decision.
Human escalation != runtime mutation. Escalation NEVER enables runtime.

Dossier state cascade
---------------------
    1. HUMAN_ESCALATION_INSTITUTIONAL  strong operator briefing justified
    2. HUMAN_ESCALATION_READY            operator review recommended
    3. HUMAN_ESCALATION_LIMITED          limited operator review justified
    4. HUMAN_ESCALATION_OBSERVE          operator awareness only
    5. HUMAN_ESCALATION_DORMANT          escalation unnecessary

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
* Append-only dossier memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to HUMAN_ESCALATION_DORMANT.
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

DEFAULT_VERDICT_SUM = (
    RESULTS_DIR / "arm_runtime_governance_institutional_verdict_engine_summary.json"
)
DEFAULT_VERDICT_REC = RESULTS_DIR / "arm_runtime_governance_institutional_verdict_engine.json"
DEFAULT_VERDICT_MEM = RESULTS_DIR / "arm_runtime_governance_institutional_verdict_engine_memory.csv"
DEFAULT_REVIEW_SUM = RESULTS_DIR / "arm_runtime_governance_enablement_review_board_summary.json"
DEFAULT_RECOMMENDATION_SUM = (
    RESULTS_DIR / "arm_runtime_governance_enablement_recommendation_engine_summary.json"
)
DEFAULT_ELIGIBILITY_SUM = (
    RESULTS_DIR / "arm_runtime_governance_constitutional_eligibility_board_summary.json"
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

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_runtime_governance_human_escalation_dossier.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_runtime_governance_human_escalation_dossier.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_runtime_governance_human_escalation_dossier_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_runtime_governance_human_escalation_dossier_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_runtime_governance_human_escalation_dossier_memory.parquet"


# -----------------------------------------------------------
# Dossier state constants
# -----------------------------------------------------------
DOSSIER_DORMANT = "HUMAN_ESCALATION_DORMANT"
DOSSIER_OBSERVE = "HUMAN_ESCALATION_OBSERVE"
DOSSIER_LIMITED = "HUMAN_ESCALATION_LIMITED"
DOSSIER_READY = "HUMAN_ESCALATION_READY"
DOSSIER_INSTITUTIONAL = "HUMAN_ESCALATION_INSTITUTIONAL"

CLASS_NO_ESCALATION = "NO_ESCALATION"
CLASS_OBSERVE = "OBSERVE_ONLY_ESCALATION"
CLASS_LIMITED = "LIMITED_ESCALATION"
CLASS_FULL = "FULL_OPERATOR_ESCALATION"

VERDICT_DO_NOT = "DO_NOT_ENABLE_RUNTIME"
VERDICT_OBSERVE = "OBSERVE_RUNTIME_ONLY"
VERDICT_LIMITED = "LIMITED_RUNTIME_SUPPORT"
VERDICT_FAVOR = "FAVORABLE_RUNTIME_VERDICT"

POSTURE_NO_ESCALATION = "OPERATOR_POSTURE_NO_ESCALATION_REQUIRED"
POSTURE_OBSERVE = "OPERATOR_POSTURE_OBSERVE_ONLY"
POSTURE_LIMITED = "OPERATOR_POSTURE_LIMITED_CAUTIOUS_REVIEW"
POSTURE_FULL = "OPERATOR_POSTURE_FORMAL_REVIEW_RECOMMENDED"

MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "dossier_state",
    "observe_count",
    "limited_count",
    "full_count",
    "escalation_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[HUMAN_ESCALATION_DOSSIER_WARN] {msg}", flush=True)


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
        for col in ("escalation_confidence",):
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


def _scale_escalation(classification: str) -> float:
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


def _operator_posture(classification: str) -> str:
    return {
        CLASS_FULL: POSTURE_FULL,
        CLASS_LIMITED: POSTURE_LIMITED,
        CLASS_OBSERVE: POSTURE_OBSERVE,
        CLASS_NO_ESCALATION: POSTURE_NO_ESCALATION,
    }.get(classification, POSTURE_NO_ESCALATION)


# -----------------------------------------------------------
# Context extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    verdict_summary: Dict[str, Any],
    verdict_record: Dict[str, Any],
    verdict_mem: List[Dict[str, str]],
    review_summary: Dict[str, Any],
    recommendation_summary: Dict[str, Any],
    eligibility_summary: Dict[str, Any],
    admission_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    shadow_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    dossier_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    iv = verdict_record.get("institutional_verdict") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    court_safety = _court_constitutional_safety(court_summary, council_summary)

    ctx: Dict[str, Any] = {
        "verdict_state": _norm_upper(
            verdict_summary.get("verdict_state") or verdict_record.get("verdict_state")
        ),
        "verdict_confidence": _clamp(
            _to_float(
                verdict_summary.get("verdict_confidence")
                or verdict_record.get("verdict_confidence")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "verdict_available": bool(verdict_summary.get("runtime_verdict_available")),
        "institutional_verdict": iv,
        "verdict_classification": _norm_upper(
            iv.get("runtime_verdict_classification")
            or verdict_summary.get("runtime_verdict_classification")
        ),
        "verdict_score": _clamp(
            _to_float(
                iv.get("runtime_verdict_score") or verdict_summary.get("runtime_verdict_score")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "institutional_position": _norm_upper(
            iv.get("institutional_runtime_position")
            or verdict_summary.get("institutional_runtime_position")
        ),
        "review_confidence": _clamp(
            _to_float(review_summary.get("review_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "recommendation_confidence": _clamp(
            _to_float(recommendation_summary.get("recommendation_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "eligibility_confidence": _clamp(
            _to_float(eligibility_summary.get("constitutional_eligibility_confidence")) or 0.0,
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
        "shadow_state": _norm_upper(shadow_summary.get("shadow_state")),
        "governance_quality": _clamp(
            _to_float(scores.get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "constitutional_safe_verdict": bool(iv.get("constitutional_safe")),
        "constitutional_safety": _clamp(
            court_safety * 0.60 + (1.0 if bool(iv.get("constitutional_safe")) else 0.0) * 0.40,
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
            or verdict_summary.get("operator_review_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "autonomous_readiness_score": _clamp(
            _to_float(autonomous_readiness.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "verdict_memory_depth": len(verdict_mem),
        "dossier_memory_depth": len(dossier_mem),
        "observation_cycles": max(
            _to_float(verdict_summary.get("observation_cycles")) or 0,
            len(verdict_mem),
            1,
        ),
        "chain_summary": {
            "shadow_state": _norm_upper(shadow_summary.get("shadow_state")),
            "readiness_state": _norm_upper(readiness_summary.get("readiness_state")),
            "admission_state": _norm_upper(admission_summary.get("admission_state")),
            "eligibility_state": _norm_upper(
                eligibility_summary.get("constitutional_eligibility_state")
            ),
            "recommendation_state": _norm_upper(recommendation_summary.get("recommendation_state")),
            "review_state": _norm_upper(review_summary.get("review_state")),
            "verdict_state": _norm_upper(verdict_summary.get("verdict_state")),
        },
    }
    return ctx


# -----------------------------------------------------------
# Human escalation dossier assessment
# -----------------------------------------------------------
def _base_escalation_score(ctx: Dict[str, Any]) -> float:
    raw = (
        ctx["verdict_confidence"] * 0.18
        + ctx["verdict_score"] * 0.14
        + ctx["review_confidence"] * 0.12
        + ctx["recommendation_confidence"] * 0.10
        + ctx["eligibility_confidence"] * 0.08
        + ctx["admission_confidence"] * 0.06
        + ctx["readiness_confidence"] * 0.06
        + ctx["shadow_confidence"] * 0.06
        + ctx["governance_quality"] * 0.10
        + ctx["system_health_score"] * 0.06
        + ctx["constitutional_safety"] * 0.04
    )
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.16
    return round(_clamp(raw, 0.0, 1.0), 4)


def _compute_escalation_score(classification: str, ctx: Dict[str, Any]) -> float:
    raw = _base_escalation_score(ctx)
    raw *= _scale_escalation(classification)
    return round(_clamp(raw, 0.0, 1.0), 4)


def _classify_human_escalation(*, gate_score: float, ctx: Dict[str, Any]) -> str:
    vc = ctx["verdict_classification"]
    const_safe = (
        ctx["constitutional_safe_verdict"]
        and ctx["constitutional_safety"] >= 0.50
        and not ctx["constitution_violated"]
    )

    if not const_safe or vc == VERDICT_DO_NOT or gate_score < 0.10:
        return CLASS_NO_ESCALATION

    if vc == VERDICT_FAVOR and gate_score >= 0.38:
        if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
            return CLASS_LIMITED
        if ctx["system_health_stale"] or ctx["governance_quality"] < 0.45:
            return CLASS_LIMITED
        if ctx["constitutional_safety"] < 0.78:
            return CLASS_LIMITED
        return CLASS_FULL

    if vc == VERDICT_LIMITED and gate_score >= 0.24:
        if const_safe:
            return CLASS_LIMITED

    if vc == VERDICT_OBSERVE and gate_score >= 0.12:
        return CLASS_OBSERVE

    if gate_score >= 0.14 and vc != VERDICT_DO_NOT and const_safe:
        return CLASS_OBSERVE

    return CLASS_NO_ESCALATION


def _case_for_runtime(ctx: Dict[str, Any], classification: str) -> List[str]:
    points: List[str] = []
    if ctx["shadow_confidence"] >= 0.40:
        points.append("Shadow activation rehearsal has accumulated observable governance maturity.")
    if ctx["readiness_confidence"] >= 0.40:
        points.append(
            "Runtime readiness gate indicates partial institutional admissibility maturity."
        )
    if ctx["governance_quality"] >= 0.55:
        points.append(
            f"Governance quality score ({ctx['governance_quality']:.2f}) supports institutional discipline."
        )
    if ctx["system_health_score"] >= 0.70:
        points.append("System health artifacts remain sufficiently fresh for governance review.")
    if classification in (CLASS_LIMITED, CLASS_FULL):
        points.append(
            "Institutional verdict supports at least limited runtime governance consideration."
        )
    if not points:
        points.append(
            "Governance chain is maturing but insufficient for a strong case for runtime enablement."
        )
    return points


def _case_against_runtime(ctx: Dict[str, Any], classification: str) -> List[str]:
    points: List[str] = []
    if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
        points.append("Constitutional court pressure or overruling remains elevated.")
    if ctx["shadow_confidence"] < 0.30:
        points.append("Shadow governance reliability remains immature.")
    if ctx["readiness_confidence"] < 0.30:
        points.append("Runtime readiness confidence remains low.")
    if ctx["governance_quality"] < 0.50:
        points.append("Governance quality has not reached stable institutional thresholds.")
    if classification == CLASS_NO_ESCALATION:
        points.append("Institutional verdict does not support runtime governance enablement.")
    points.append(
        "Runtime enablement would mutate live posture; current dossier is communication only."
    )
    return points


def _key_risks(ctx: Dict[str, Any]) -> List[str]:
    risks: List[str] = []
    if ctx["constitution_violated"]:
        risks.append("Constitutional violation state active")
    if ctx["court_ruling"] == "COURT_OVERRULED":
        risks.append("Constitutional court overruling in effect")
    if ctx["system_health_stale"]:
        risks.append("System health artifacts stale")
    if ctx["governance_quality"] < 0.45:
        risks.append("Governance quality below stability floor")
    if ctx["shadow_confidence"] < 0.25:
        risks.append("Shadow activation maturity insufficient")
    if not risks:
        risks.append("Residual governance immaturity under defensive regime")
    return risks


def _key_safeguards(ctx: Dict[str, Any]) -> List[str]:
    return [
        "Runtime mutation lock enforced across all governance engines",
        "Constitutional court and supreme governance council supremacy preserved",
        "Capital preservation doctrine remains binding",
        "Operator supremacy retained; dossier does not enable runtime",
        f"Live runtime regime: {ctx['regime']} (unchanged by this dossier)",
        "Append-only governance memory preserves institutional audit trail",
    ]


def _executive_summary(ctx: Dict[str, Any], classification: str) -> str:
    if classification == CLASS_NO_ESCALATION or ctx["constitution_violated"]:
        return (
            "Triton does not currently support runtime governance enablement. "
            "Governance maturity remains insufficient, constitutional pressure remains elevated, "
            "and shadow governance reliability remains immature."
        )
    if classification == CLASS_FULL:
        return (
            "Triton presents a favorable institutional case for operator review of future "
            "runtime governance. Governance maturity is stable; constitutional safety is adequate. "
            "This dossier does not enable runtime."
        )
    if classification == CLASS_LIMITED:
        return (
            "Triton presents a limited operator escalation case under defensive governance conditions. "
            "Cautious review is justified; full enablement remains constrained."
        )
    return (
        "Triton recommends operator awareness only. Governance maturity remains observe-level; "
        "formal escalation is not yet justified."
    )


def _escalation_rationale(classification: str, score: float, ctx: Dict[str, Any]) -> str:
    templates = {
        CLASS_FULL: "full operator escalation: complete institutional case prepared for human review",
        CLASS_LIMITED: "limited escalation: cautious operator briefing under defensive posture",
        CLASS_OBSERVE: "observe-only escalation: operator awareness without formal review",
        CLASS_NO_ESCALATION: "no escalation: institutional verdict does not justify operator briefing",
    }
    base = templates.get(classification, templates[CLASS_NO_ESCALATION])
    return f"{base} (escalation_score={score:.2f})"


def _build_human_escalation_dossier(
    *,
    ctx: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    gate_score = _base_escalation_score(ctx)
    if prior:
        prior_score = _to_float(prior.get("human_escalation_score")) or gate_score
        gate_score = round(_clamp(gate_score * 0.65 + prior_score * 0.35, 0.0, 1.0), 4)

    classification = _classify_human_escalation(gate_score=gate_score, ctx=ctx)
    escalation_score = _compute_escalation_score(classification, ctx)

    iv = ctx["institutional_verdict"]
    verdict_class = ctx["verdict_classification"]
    verdict_score = ctx["verdict_score"]
    position = ctx["institutional_position"]

    const_safe = (
        ctx["constitutional_safe_verdict"]
        and ctx["constitutional_safety"] >= 0.50
        and not ctx["constitution_violated"]
        and classification != CLASS_NO_ESCALATION
    )
    future_candidate = classification in (CLASS_LIMITED, CLASS_FULL) or bool(
        iv.get("future_runtime_candidate")
    )

    conf = round(
        _clamp(
            gate_score * 0.32
            + ctx["verdict_confidence"] * 0.34
            + ctx["constitutional_safety"] * 0.34,
            0.0,
            1.0,
        ),
        4,
    )

    return {
        "human_escalation_classification": classification,
        "human_escalation_score": escalation_score,
        "runtime_verdict_classification": verdict_class,
        "runtime_verdict_score": round(verdict_score, 4),
        "institutional_runtime_position": position,
        "executive_summary": _executive_summary(ctx, classification),
        "case_for_runtime": _case_for_runtime(ctx, classification),
        "case_against_runtime": _case_against_runtime(ctx, classification),
        "key_risks": _key_risks(ctx),
        "key_safeguards": _key_safeguards(ctx),
        "recommended_operator_posture": _operator_posture(classification),
        "future_runtime_candidate": future_candidate,
        "constitutional_safe": const_safe,
        "confidence": conf,
        "runtime_mutation_allowed": False,
        "human_escalation_rationale": _escalation_rationale(classification, escalation_score, ctx),
        "governance_chain_summary": ctx["chain_summary"],
    }


# -----------------------------------------------------------
# Escalation confidence and state
# -----------------------------------------------------------
def _escalation_confidence(ctx: Dict[str, Any], dossier: Dict[str, Any]) -> float:
    raw = (
        ctx["verdict_confidence"] * 0.16
        + ctx["review_confidence"] * 0.12
        + ctx["recommendation_confidence"] * 0.10
        + ctx["eligibility_confidence"] * 0.08
        + ctx["admission_confidence"] * 0.06
        + ctx["readiness_confidence"] * 0.06
        + ctx["shadow_confidence"] * 0.08
        + ctx["governance_quality"] * 0.14
        + ctx["system_health_score"] * 0.10
        + ctx["constitutional_safety"] * 0.10
    )
    raw += (_to_float(dossier.get("human_escalation_score")) or 0.0) * 0.05

    penalty = ctx["constitutional_pressure"] * 0.34
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
        "no_escalation": 1 if classification == CLASS_NO_ESCALATION else 0,
        "observe": 1 if classification == CLASS_OBSERVE else 0,
        "limited": 1 if classification == CLASS_LIMITED else 0,
        "full": 1 if classification == CLASS_FULL else 0,
    }


def _classify_dossier_state(
    *,
    ctx: Dict[str, Any],
    classification: str,
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if classification == CLASS_NO_ESCALATION or not ctx["verdict_available"]:
        reasons.append("runtime governance not supported; escalation unnecessary")
        return DOSSIER_DORMANT, reasons

    if counts["full"] >= 1 and ctx["dossier_memory_depth"] >= 2:
        reasons.append("mature governance process; strong operator briefing justified")
        return DOSSIER_INSTITUTIONAL, reasons

    if counts["full"] >= 1:
        reasons.append("governance maturity stable; operator review recommended")
        return DOSSIER_READY, reasons

    if counts["limited"] >= 1:
        reasons.append("cautious governance support; limited operator review justified")
        return DOSSIER_LIMITED, reasons

    if counts["observe"] >= 1:
        reasons.append("observation maturity only; operator awareness only")
        return DOSSIER_OBSERVE, reasons

    reasons.append("runtime governance not supported; escalation unnecessary")
    return DOSSIER_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _dossier_booleans(
    state: str,
    dossier: Dict[str, Any],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    classification = dossier.get("human_escalation_classification", CLASS_NO_ESCALATION)
    return {
        "human_escalation_available": bool(dossier),
        "limited_escalation_available": counts["limited"] > 0 or counts["full"] > 0,
        "full_operator_escalation_available": counts["full"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"]
            or dossier.get("future_runtime_candidate", False)
            or classification in (CLASS_LIMITED, CLASS_FULL)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "dossier_memory_reliable": state == DOSSIER_INSTITUTIONAL,
    }


def _recommendations_list(state: str, counts: Dict[str, int]) -> List[str]:
    recs = [
        "Continue governance maturity accumulation",
        "Continue shadow activation rehearsal",
        "Maintain defensive constitutional posture",
        "Avoid premature runtime assumptions",
        "Maintain runtime mutation lock",
        "Revisit after governance maturity improves",
    ]
    if counts["full"] > 0 or counts["limited"] > 0:
        recs.append("Schedule formal operator review using this dossier")
    if state == DOSSIER_DORMANT:
        recs.append("No operator escalation required at this time")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(dossier: Dict[str, Any], state: str, ctx: Dict[str, Any]) -> str:
    classification = dossier.get("human_escalation_classification", CLASS_NO_ESCALATION)
    if classification == CLASS_NO_ESCALATION or ctx["constitution_violated"]:
        return (
            "Triton does not place runtime governance under human escalation because "
            "constitutional pressure remains elevated despite improving governance maturity."
        )
    if classification == CLASS_FULL:
        return (
            "Triton prepares a full operator-facing dossier for formal runtime governance review. "
            "This is communication only; no runtime is enabled."
        )
    if classification == CLASS_LIMITED:
        return "Triton prepares a limited operator dossier under defensive governance conditions."
    if classification == CLASS_OBSERVE:
        return "Operator awareness dossier only; formal escalation is not yet justified."
    return (
        "Human escalation dossier completed without runtime mutation. "
        "Verdict != human decision. Dossier != decision."
    )


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    escalation_confidence: float,
    dossier: Dict[str, Any],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
    ctx: Dict[str, Any],
) -> str:
    d = dossier
    lines = [
        "# Triton Runtime Governance Human Escalation Dossier",
        "",
        f"_Generated at {generated_at}_",
        "",
        f"**Dossier state:** {state} | **Escalation confidence:** {escalation_confidence:.3f}",
        "",
        "## Executive Summary",
        "",
        d.get("executive_summary", ""),
        "",
        "## Institutional Runtime Position",
        "",
        f"- **Verdict:** {d.get('runtime_verdict_classification', VERDICT_DO_NOT)}",
        f"- **Position:** {d.get('institutional_runtime_position', 'UNKNOWN')}",
        f"- **Escalation classification:** {d.get('human_escalation_classification', CLASS_NO_ESCALATION)}",
        f"- **Recommended operator posture:** {d.get('recommended_operator_posture', POSTURE_NO_ESCALATION)}",
        f"- **Runtime mutation allowed:** {booleans['runtime_mutation_allowed']}",
        "",
        "## Case For Runtime Governance",
        "",
    ]
    for item in d.get("case_for_runtime") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Case Against Runtime Governance", ""])
    for item in d.get("case_against_runtime") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Risks", ""])
    for item in d.get("key_risks") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Safeguards", ""])
    for item in d.get("key_safeguards") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "### Governance Chain",
            "",
            "| stage | state |",
            "|---|---|",
        ]
    )
    chain = d.get("governance_chain_summary") or ctx.get("chain_summary") or {}
    for stage, st in chain.items():
        lines.append(f"| {stage} | {st} |")
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
            "This is the final governance communication layer before any hypothetical human/operator "
            "judgment. Verdict != human decision. Dossier != decision. No live runtime policy is changed.",
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
    escalation_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "dossier_state": state,
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "escalation_confidence": round(escalation_confidence, 6),
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
def build_human_escalation_dossier(
    *,
    verdict_summary: Dict[str, Any],
    verdict_record: Dict[str, Any],
    verdict_mem: List[Dict[str, str]],
    review_summary: Dict[str, Any],
    recommendation_summary: Dict[str, Any],
    eligibility_summary: Dict[str, Any],
    admission_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    shadow_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    dossier_mem: List[Dict[str, str]],
    prior_dossier_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        verdict_summary=verdict_summary,
        verdict_record=verdict_record,
        verdict_mem=verdict_mem,
        review_summary=review_summary,
        recommendation_summary=recommendation_summary,
        eligibility_summary=eligibility_summary,
        admission_summary=admission_summary,
        readiness_summary=readiness_summary,
        shadow_summary=shadow_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        autonomous_readiness=autonomous_readiness,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        dossier_mem=dossier_mem,
    )

    prior_d = prior_dossier_record.get("human_escalation_dossier") or {}
    human_dossier = _build_human_escalation_dossier(ctx=ctx, prior=prior_d or None)
    escalation_confidence = _escalation_confidence(ctx, human_dossier)
    counts = _classification_counts(
        human_dossier.get("human_escalation_classification", CLASS_NO_ESCALATION)
    )

    state, reasons = _classify_dossier_state(
        ctx=ctx,
        classification=human_dossier.get("human_escalation_classification", CLASS_NO_ESCALATION),
        counts=counts,
    )

    booleans = _dossier_booleans(state, human_dossier, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(human_dossier, state, ctx)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        escalation_confidence=escalation_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(dossier_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        escalation_confidence=escalation_confidence,
        dossier=human_dossier,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
        ctx=ctx,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_human_escalation_dossier_engine",
        "engine_version": 1,
        "dossier_state": state,
        "escalation_confidence": escalation_confidence,
        "dossier_reasons": reasons,
        "human_escalation_dossier": human_dossier,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "verdict_vs_dossier_note": (
            "Verdict != human decision. Dossier != decision. "
            "Human escalation != runtime mutation. Live runtime is never changed."
        ),
        "constitutional_supremacy_note": (
            "Human escalation cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "dossier_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "live_runtime_regime": ctx["regime"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutated": False,
            "runtime_mutation_allowed": False,
            "communication_only": True,
        },
        "inputs_seen": {
            "arm_runtime_governance_institutional_verdict_engine_summary": bool(verdict_summary),
            "arm_runtime_governance_institutional_verdict_engine_record": bool(verdict_record),
            "arm_runtime_governance_institutional_verdict_engine_memory_rows": len(verdict_mem),
            "arm_runtime_governance_enablement_review_board_summary": bool(review_summary),
            "arm_runtime_governance_enablement_recommendation_engine_summary": bool(
                recommendation_summary
            ),
            "arm_runtime_governance_constitutional_eligibility_board_summary": bool(
                eligibility_summary
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
            "existing_dossier_memory_rows": len(dossier_mem),
            "prior_human_escalation_dossier": bool(prior_d),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_human_escalation_dossier_engine",
        "dossier_state": state,
        "escalation_confidence": escalation_confidence,
        "human_escalation_available": booleans["human_escalation_available"],
        "limited_escalation_available": booleans["limited_escalation_available"],
        "full_operator_escalation_available": booleans["full_operator_escalation_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "dossier_memory_reliable": booleans["dossier_memory_reliable"],
        "no_escalation_count": counts["no_escalation"],
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "observation_cycles": ctx["observation_cycles"],
        "human_escalation_classification": human_dossier.get("human_escalation_classification"),
        "human_escalation_score": human_dossier.get("human_escalation_score"),
        "runtime_verdict_classification": human_dossier.get("runtime_verdict_classification"),
        "institutional_runtime_position": human_dossier.get("institutional_runtime_position"),
        "recommended_operator_posture": human_dossier.get("recommended_operator_posture"),
        "future_runtime_candidate": human_dossier.get("future_runtime_candidate"),
        "n_recommendations": len(recommendations),
        "dossier_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM runtime governance human escalation dossier engine (Step 60). "
            "Operator-facing dossier without mutating runtime. No broker calls."
        ),
    )
    p.add_argument("--verdict-summary", default=str(DEFAULT_VERDICT_SUM))
    p.add_argument("--verdict-record", default=str(DEFAULT_VERDICT_REC))
    p.add_argument("--verdict-mem", default=str(DEFAULT_VERDICT_MEM))
    p.add_argument("--review-summary", default=str(DEFAULT_REVIEW_SUM))
    p.add_argument("--recommendation-summary", default=str(DEFAULT_RECOMMENDATION_SUM))
    p.add_argument("--eligibility-summary", default=str(DEFAULT_ELIGIBILITY_SUM))
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
        "[HUMAN_ESCALATION_DOSSIER] starting "
        "(read-only communication; no runtime mutation; no broker calls)",
        flush=True,
    )

    verdict_summary = _safe_read_json(
        Path(args.verdict_summary),
        label="arm_runtime_governance_institutional_verdict_engine_summary.json",
    )
    verdict_record = _safe_read_json(
        Path(args.verdict_record), label="arm_runtime_governance_institutional_verdict_engine.json"
    )
    verdict_mem = _safe_read_csv_rows(
        Path(args.verdict_mem),
        label="arm_runtime_governance_institutional_verdict_engine_memory.csv",
    )
    review_summary = _safe_read_json(
        Path(args.review_summary),
        label="arm_runtime_governance_enablement_review_board_summary.json",
    )
    recommendation_summary = _safe_read_json(
        Path(args.recommendation_summary),
        label="arm_runtime_governance_enablement_recommendation_engine_summary.json",
    )
    eligibility_summary = _safe_read_json(
        Path(args.eligibility_summary),
        label="arm_runtime_governance_constitutional_eligibility_board_summary.json",
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
    dossier_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_runtime_governance_human_escalation_dossier_memory.csv"
    )
    prior_dossier_record = _safe_read_json(
        Path(args.out_json), label="prior_arm_runtime_governance_human_escalation_dossier.json"
    )

    record, summary, md, merged_memory = build_human_escalation_dossier(
        verdict_summary=verdict_summary,
        verdict_record=verdict_record,
        verdict_mem=verdict_mem,
        review_summary=review_summary,
        recommendation_summary=recommendation_summary,
        eligibility_summary=eligibility_summary,
        admission_summary=admission_summary,
        readiness_summary=readiness_summary,
        shadow_summary=shadow_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        autonomous_readiness=autonomous_readiness,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        dossier_mem=dossier_mem,
        prior_dossier_record=prior_dossier_record,
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
        "[HUMAN_ESCALATION_DOSSIER] "
        f"state={record['dossier_state']} "
        f"limited={counts['limited']} "
        f"full={counts['full']} "
        f"confidence={record['escalation_confidence']:.3f}",
        flush=True,
    )
    print(
        "[HUMAN_ESCALATION_DOSSIER_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[HUMAN_ESCALATION_DOSSIER_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[HUMAN_ESCALATION_DOSSIER_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
