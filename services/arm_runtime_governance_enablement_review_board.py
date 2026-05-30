"""
ARM Runtime Governance Enablement Review Board -- Step 58.

Reads:
    data/results/arm_runtime_governance_enablement_recommendation_engine_summary.json (Step 57)
    data/results/arm_runtime_governance_enablement_recommendation_engine.json         (Step 57)
    data/results/arm_runtime_governance_enablement_recommendation_engine_memory.csv   (Step 57)
    data/results/arm_runtime_governance_constitutional_eligibility_board_summary.json (Step 56)
    data/results/arm_runtime_governance_admission_board_summary.json                  (Step 55)
    data/results/arm_runtime_governance_readiness_gate_summary.json                   (Step 54)
    data/results/arm_governance_runtime_shadow_activation_summary.json                (Step 53)
    data/results/autonomous_governance_scorecard.json                                 (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/arm_constitutional_court_summary.json                                (Step 33)
    data/results/arm_supreme_governance_council_summary.json                            (Step 34)
    data/results/runtime_policy_governed.json                                         (Step 18)

Writes:
    data/results/arm_runtime_governance_enablement_review_board.json
    data/results/arm_runtime_governance_enablement_review_board.md
    data/results/arm_runtime_governance_enablement_review_board_summary.json
    data/results/arm_runtime_governance_enablement_review_board_memory.csv
    data/results/arm_runtime_governance_enablement_review_board_memory.parquet

Purpose
-------
This engine answers:

    "Even if runtime governance is recommended, should it enter formal institutional review?"

It converts runtime enablement recommendation into formal institutional runtime enablement review.
This is the FINAL institutional review layer before any hypothetical runtime enablement conversation.
Recommended != under review. Under review != enabled.
Enablement review != runtime mutation. Review NEVER enables runtime.

Review state cascade
--------------------
    1. RUNTIME_ENABLEMENT_REVIEW_INSTITUTIONAL  stable institutional review quality
    2. RUNTIME_ENABLEMENT_REVIEW_READY            review plausible
    3. RUNTIME_ENABLEMENT_REVIEW_LIMITED          limited review only
    4. RUNTIME_ENABLEMENT_REVIEW_OBSERVE          observation only; no formal review
    5. RUNTIME_ENABLEMENT_REVIEW_DORMANT          not recommended; governance immature

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
* Append-only review memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to RUNTIME_ENABLEMENT_REVIEW_DORMANT.
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

DEFAULT_RECOMMENDATION_SUM = (
    RESULTS_DIR / "arm_runtime_governance_enablement_recommendation_engine_summary.json"
)
DEFAULT_RECOMMENDATION_REC = (
    RESULTS_DIR / "arm_runtime_governance_enablement_recommendation_engine.json"
)
DEFAULT_RECOMMENDATION_MEM = (
    RESULTS_DIR / "arm_runtime_governance_enablement_recommendation_engine_memory.csv"
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

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_runtime_governance_enablement_review_board.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_runtime_governance_enablement_review_board.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_runtime_governance_enablement_review_board_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_runtime_governance_enablement_review_board_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_runtime_governance_enablement_review_board_memory.parquet"


# -----------------------------------------------------------
# Review state constants
# -----------------------------------------------------------
REVIEW_DORMANT = "RUNTIME_ENABLEMENT_REVIEW_DORMANT"
REVIEW_OBSERVE = "RUNTIME_ENABLEMENT_REVIEW_OBSERVE"
REVIEW_LIMITED = "RUNTIME_ENABLEMENT_REVIEW_LIMITED"
REVIEW_READY = "RUNTIME_ENABLEMENT_REVIEW_READY"
REVIEW_INSTITUTIONAL = "RUNTIME_ENABLEMENT_REVIEW_INSTITUTIONAL"

CLASS_NOT_REVIEW = "NOT_UNDER_REVIEW"
CLASS_OBSERVE = "OBSERVE_REVIEW"
CLASS_LIMITED = "LIMITED_REVIEW"
CLASS_FULL = "FULL_REVIEW"

REC_NOT = "NOT_RECOMMENDED"
REC_OBSERVE = "OBSERVE_ENABLEMENT"
REC_LIMITED = "LIMITED_ENABLEMENT_RECOMMENDATION"
REC_FULL = "FULL_ENABLEMENT_RECOMMENDATION"

MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "review_state",
    "observe_count",
    "limited_count",
    "full_count",
    "review_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[RUNTIME_ENABLEMENT_REVIEW_WARN] {msg}", flush=True)


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
        for col in ("review_confidence",):
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


def _scale_review(classification: str) -> float:
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
    recommendation_summary: Dict[str, Any],
    recommendation_record: Dict[str, Any],
    recommendation_mem: List[Dict[str, str]],
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
    review_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    er = recommendation_record.get("enablement_recommendation") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    court_safety = _court_constitutional_safety(court_summary, council_summary)

    ctx: Dict[str, Any] = {
        "recommendation_state": _norm_upper(
            recommendation_summary.get("recommendation_state")
            or recommendation_record.get("recommendation_state")
        ),
        "recommendation_confidence": _clamp(
            _to_float(
                recommendation_summary.get("recommendation_confidence")
                or recommendation_record.get("recommendation_confidence")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "recommendation_available": bool(
            recommendation_summary.get("runtime_enablement_recommendation_available")
        ),
        "enablement_recommendation": er,
        "recommendation_classification": _norm_upper(
            er.get("runtime_enablement_recommendation_classification")
            or recommendation_summary.get("runtime_enablement_recommendation_classification")
        ),
        "recommendation_score": _clamp(
            _to_float(
                er.get("runtime_enablement_recommendation_score")
                or recommendation_summary.get("runtime_enablement_recommendation_score")
            )
            or 0.0,
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
        "governance_quality": _clamp(
            _to_float(scores.get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "constitutional_safe_recommendation": bool(er.get("constitutional_safe")),
        "constitutional_safety": _clamp(
            court_safety * 0.60 + (1.0 if bool(er.get("constitutional_safe")) else 0.0) * 0.40,
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
            or eligibility_summary.get("operator_review_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "autonomous_readiness_score": _clamp(
            _to_float(autonomous_readiness.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "recommendation_memory_depth": len(recommendation_mem),
        "review_memory_depth": len(review_mem),
        "observation_cycles": max(
            _to_float(recommendation_summary.get("observation_cycles")) or 0,
            len(recommendation_mem),
            1,
        ),
    }
    return ctx


# -----------------------------------------------------------
# Enablement review assessment
# -----------------------------------------------------------
def _base_review_score(ctx: Dict[str, Any]) -> float:
    raw = (
        ctx["recommendation_confidence"] * 0.22
        + ctx["recommendation_score"] * 0.18
        + ctx["eligibility_confidence"] * 0.14
        + ctx["admission_confidence"] * 0.10
        + ctx["readiness_confidence"] * 0.08
        + ctx["shadow_confidence"] * 0.08
        + ctx["governance_quality"] * 0.10
        + ctx["system_health_score"] * 0.06
        + ctx["constitutional_safety"] * 0.04
    )
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.20
    return round(_clamp(raw, 0.0, 1.0), 4)


def _compute_review_score(classification: str, ctx: Dict[str, Any]) -> float:
    raw = _base_review_score(ctx)
    raw *= _scale_review(classification)
    return round(_clamp(raw, 0.0, 1.0), 4)


def _classify_enablement_review(*, gate_score: float, ctx: Dict[str, Any]) -> str:
    rc = ctx["recommendation_classification"]
    const_safe = (
        ctx["constitutional_safe_recommendation"]
        and ctx["constitutional_safety"] >= 0.50
        and not ctx["constitution_violated"]
    )

    if not const_safe or rc == REC_NOT or gate_score < 0.14:
        return CLASS_NOT_REVIEW

    if rc == REC_FULL and gate_score >= 0.42:
        if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
            return CLASS_LIMITED
        if ctx["system_health_stale"] or ctx["governance_quality"] < 0.45:
            return CLASS_LIMITED
        if ctx["constitutional_safety"] < 0.74:
            return CLASS_LIMITED
        return CLASS_FULL

    if rc == REC_LIMITED and gate_score >= 0.28:
        if const_safe:
            return CLASS_LIMITED

    if rc == REC_OBSERVE and gate_score >= 0.16:
        return CLASS_OBSERVE

    if gate_score >= 0.18 and rc != REC_NOT and const_safe:
        return CLASS_OBSERVE

    return CLASS_NOT_REVIEW


def _review_rationale(classification: str, review_score: float, ctx: Dict[str, Any]) -> str:
    templates = {
        CLASS_FULL: (
            "full enablement review: runtime governance enters formal institutional review "
            "without runtime activation"
        ),
        CLASS_LIMITED: (
            "limited enablement review: constitutionally safe defensive posture supports "
            "limited formal review only"
        ),
        CLASS_OBSERVE: ("observe review: observation maturity only; no formal enablement review"),
        CLASS_NOT_REVIEW: (
            "not under review: constitutional pressure or recommendation immaturity blocks review"
        ),
    }
    base = templates.get(classification, templates[CLASS_NOT_REVIEW])
    if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
        base += " under elevated constitutional pressure"
    return f"{base} (review_score={review_score:.2f})"


def _build_enablement_review(
    *,
    ctx: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    gate_score = _base_review_score(ctx)
    if prior:
        prior_score = _to_float(prior.get("runtime_enablement_review_score")) or gate_score
        gate_score = round(_clamp(gate_score * 0.65 + prior_score * 0.35, 0.0, 1.0), 4)

    classification = _classify_enablement_review(gate_score=gate_score, ctx=ctx)
    review_score = _compute_review_score(classification, ctx)

    er = ctx["enablement_recommendation"]
    recommendation_class = ctx["recommendation_classification"]
    recommendation_score = ctx["recommendation_score"]

    const_safe = (
        ctx["constitutional_safe_recommendation"]
        and ctx["constitutional_safety"] >= 0.50
        and not ctx["constitution_violated"]
        and classification != CLASS_NOT_REVIEW
    )
    under_review = classification == CLASS_FULL
    future_candidate = classification in (CLASS_LIMITED, CLASS_FULL) or bool(
        er.get("future_runtime_candidate")
    )

    conf = round(
        _clamp(
            gate_score * 0.36
            + ctx["recommendation_confidence"] * 0.34
            + ctx["constitutional_safety"] * 0.30,
            0.0,
            1.0,
        ),
        4,
    )

    return {
        "runtime_enablement_review_classification": classification,
        "runtime_enablement_review_score": review_score,
        "runtime_enablement_recommendation_classification": recommendation_class,
        "runtime_enablement_recommendation_score": round(recommendation_score, 4),
        "institutionally_under_runtime_review": under_review,
        "future_runtime_candidate": future_candidate,
        "constitutional_safe": const_safe,
        "confidence": conf,
        "runtime_mutation_allowed": False,
        "runtime_enablement_review_rationale": _review_rationale(
            classification,
            review_score,
            ctx,
        ),
    }


# -----------------------------------------------------------
# Review confidence and state
# -----------------------------------------------------------
def _review_confidence(ctx: Dict[str, Any], enablement_review: Dict[str, Any]) -> float:
    raw = (
        ctx["recommendation_confidence"] * 0.20
        + ctx["eligibility_confidence"] * 0.16
        + ctx["admission_confidence"] * 0.12
        + ctx["readiness_confidence"] * 0.10
        + ctx["shadow_confidence"] * 0.10
        + ctx["governance_quality"] * 0.14
        + ctx["system_health_score"] * 0.10
        + ctx["constitutional_safety"] * 0.08
    )
    raw += (_to_float(enablement_review.get("runtime_enablement_review_score")) or 0.0) * 0.05

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
        "not_under_review": 1 if classification == CLASS_NOT_REVIEW else 0,
        "observe": 1 if classification == CLASS_OBSERVE else 0,
        "limited": 1 if classification == CLASS_LIMITED else 0,
        "full": 1 if classification == CLASS_FULL else 0,
    }


def _classify_review_state(
    *,
    ctx: Dict[str, Any],
    classification: str,
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if classification == CLASS_NOT_REVIEW or not ctx["recommendation_available"]:
        reasons.append("runtime not recommended; governance maturity insufficient")
        return REVIEW_DORMANT, reasons

    if counts["full"] >= 1 and ctx["review_memory_depth"] >= 2:
        reasons.append("mature governance process with stable institutional review quality")
        return REVIEW_INSTITUTIONAL, reasons

    if counts["full"] >= 1:
        reasons.append("governance maturity stable; formal review plausible")
        return REVIEW_READY, reasons

    if counts["limited"] >= 1:
        reasons.append("defensive governance maturity; limited formal review only")
        return REVIEW_LIMITED, reasons

    if counts["observe"] >= 1:
        reasons.append("observation maturity only; no formal review")
        return REVIEW_OBSERVE, reasons

    reasons.append("runtime not recommended; governance maturity insufficient")
    return REVIEW_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations list, rationale
# -----------------------------------------------------------
def _review_booleans(
    state: str,
    enablement_review: Dict[str, Any],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    classification = enablement_review.get(
        "runtime_enablement_review_classification",
        CLASS_NOT_REVIEW,
    )
    return {
        "runtime_enablement_review_available": bool(enablement_review),
        "limited_runtime_review_available": counts["limited"] > 0 or counts["full"] > 0,
        "full_runtime_review_available": counts["full"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"]
            or enablement_review.get("future_runtime_candidate", False)
            or classification in (CLASS_LIMITED, CLASS_FULL)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "review_memory_reliable": state == REVIEW_INSTITUTIONAL,
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
        recs.append("Escalate enablement review to operator oversight")
    if state == REVIEW_DORMANT:
        recs.append("Resolve enablement recommendation before formal review")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(
    enablement_review: Dict[str, Any],
    state: str,
    ctx: Dict[str, Any],
) -> str:
    classification = enablement_review.get(
        "runtime_enablement_review_classification",
        CLASS_NOT_REVIEW,
    )
    if classification == CLASS_NOT_REVIEW or ctx["constitution_violated"]:
        return (
            "Triton does not place runtime governance under review because constitutional "
            "pressure remains elevated despite improving governance maturity."
        )
    if classification == CLASS_FULL:
        return (
            "Triton places runtime governance under full formal institutional review for future "
            "enablement consideration without activating or mutating live runtime."
        )
    if classification == CLASS_LIMITED:
        return (
            "Triton places runtime governance under limited formal review under defensive "
            "conditions; full review remains constrained."
        )
    if classification == CLASS_OBSERVE:
        return "Enablement review remains observe-only; governance maturity is immature."
    return (
        "Runtime governance enablement review board completed without runtime mutation. "
        "Recommended != under review."
    )


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    review_confidence: float,
    enablement_review: Dict[str, Any],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
    ctx: Dict[str, Any],
) -> str:
    rv = enablement_review
    lines = [
        "# Triton Runtime Governance Enablement Review Board",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Review State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| review_confidence | {review_confidence:.3f} |",
        f"| classification | {rv.get('runtime_enablement_review_classification', CLASS_NOT_REVIEW)} |",
        f"| review_score | {_to_float(rv.get('runtime_enablement_review_score')) or 0.0:.3f} |",
        f"| recommendation_classification | {rv.get('runtime_enablement_recommendation_classification', REC_NOT)} |",
        f"| recommendation_score | {_to_float(rv.get('runtime_enablement_recommendation_score')) or 0.0:.3f} |",
        f"| institutionally_under_runtime_review | {rv.get('institutionally_under_runtime_review', False)} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Runtime Enablement Review",
        "",
        f"- **Classification:** {rv.get('runtime_enablement_review_classification', CLASS_NOT_REVIEW)}",
        f"- **Constitutional safe:** {rv.get('constitutional_safe', False)}",
        f"- **Future runtime candidate:** {rv.get('future_runtime_candidate', False)}",
        f"- **Recommendation state:** {ctx.get('recommendation_state', 'UNKNOWN')}",
        "",
        f"_{rv.get('runtime_enablement_review_rationale', '')}_",
        "",
        "## Runtime Governance Review Status",
        "",
        f"| tier | count |",
        f"|---|---|",
        f"| observe | {counts['observe']} |",
        f"| limited | {counts['limited']} |",
        f"| full | {counts['full']} |",
        "",
        f"Recommendation confidence: {ctx['recommendation_confidence']:.3f} | "
        f"Eligibility confidence: {ctx['eligibility_confidence']:.3f} | "
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
            "This is the final institutional review layer before any hypothetical runtime "
            "enablement conversation. Recommended != under review. Under review != enabled. "
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
    review_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "review_state": state,
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "review_confidence": round(review_confidence, 6),
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
def build_enablement_review_board(
    *,
    recommendation_summary: Dict[str, Any],
    recommendation_record: Dict[str, Any],
    recommendation_mem: List[Dict[str, str]],
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
    review_mem: List[Dict[str, str]],
    prior_review_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        recommendation_summary=recommendation_summary,
        recommendation_record=recommendation_record,
        recommendation_mem=recommendation_mem,
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
        review_mem=review_mem,
    )

    prior_rv = prior_review_record.get("enablement_review") or {}
    enablement_review = _build_enablement_review(ctx=ctx, prior=prior_rv or None)
    review_confidence = _review_confidence(ctx, enablement_review)
    counts = _classification_counts(
        enablement_review.get("runtime_enablement_review_classification", CLASS_NOT_REVIEW)
    )

    state, reasons = _classify_review_state(
        ctx=ctx,
        classification=enablement_review.get(
            "runtime_enablement_review_classification",
            CLASS_NOT_REVIEW,
        ),
        counts=counts,
    )

    booleans = _review_booleans(state, enablement_review, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(enablement_review, state, ctx)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        review_confidence=review_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(review_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        review_confidence=review_confidence,
        enablement_review=enablement_review,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
        ctx=ctx,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_enablement_review_board",
        "engine_version": 1,
        "review_state": state,
        "review_confidence": review_confidence,
        "review_reasons": reasons,
        "enablement_review": enablement_review,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "recommended_vs_review_note": (
            "Recommended != under review. Under review != enabled. "
            "Enablement review != runtime mutation. Live runtime is never changed."
        ),
        "constitutional_supremacy_note": (
            "Runtime review cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "review_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "live_runtime_regime": ctx["regime"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutated": False,
            "runtime_mutation_allowed": False,
            "review_evaluation_only": True,
        },
        "inputs_seen": {
            "arm_runtime_governance_enablement_recommendation_engine_summary": bool(
                recommendation_summary
            ),
            "arm_runtime_governance_enablement_recommendation_engine_record": bool(
                recommendation_record
            ),
            "arm_runtime_governance_enablement_recommendation_engine_memory_rows": len(
                recommendation_mem
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
            "existing_review_memory_rows": len(review_mem),
            "prior_enablement_review": bool(prior_rv),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_enablement_review_board",
        "review_state": state,
        "review_confidence": review_confidence,
        "runtime_enablement_review_available": booleans["runtime_enablement_review_available"],
        "limited_runtime_review_available": booleans["limited_runtime_review_available"],
        "full_runtime_review_available": booleans["full_runtime_review_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "review_memory_reliable": booleans["review_memory_reliable"],
        "not_under_review_count": counts["not_under_review"],
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "observation_cycles": ctx["observation_cycles"],
        "runtime_enablement_review_classification": enablement_review.get(
            "runtime_enablement_review_classification",
        ),
        "runtime_enablement_review_score": enablement_review.get("runtime_enablement_review_score"),
        "runtime_enablement_recommendation_classification": enablement_review.get(
            "runtime_enablement_recommendation_classification",
        ),
        "institutionally_under_runtime_review": enablement_review.get(
            "institutionally_under_runtime_review",
        ),
        "n_recommendations": len(recommendations),
        "review_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM runtime governance enablement review board (Step 58). "
            "Formal review evaluation without mutating runtime. No broker calls."
        ),
    )
    p.add_argument("--recommendation-summary", default=str(DEFAULT_RECOMMENDATION_SUM))
    p.add_argument("--recommendation-record", default=str(DEFAULT_RECOMMENDATION_REC))
    p.add_argument("--recommendation-mem", default=str(DEFAULT_RECOMMENDATION_MEM))
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
        "[RUNTIME_ENABLEMENT_REVIEW] starting "
        "(read-only review evaluation; no runtime mutation; no broker calls)",
        flush=True,
    )

    recommendation_summary = _safe_read_json(
        Path(args.recommendation_summary),
        label="arm_runtime_governance_enablement_recommendation_engine_summary.json",
    )
    recommendation_record = _safe_read_json(
        Path(args.recommendation_record),
        label="arm_runtime_governance_enablement_recommendation_engine.json",
    )
    recommendation_mem = _safe_read_csv_rows(
        Path(args.recommendation_mem),
        label="arm_runtime_governance_enablement_recommendation_engine_memory.csv",
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
    review_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_runtime_governance_enablement_review_board_memory.csv"
    )
    prior_review_record = _safe_read_json(
        Path(args.out_json), label="prior_arm_runtime_governance_enablement_review_board.json"
    )

    record, summary, md, merged_memory = build_enablement_review_board(
        recommendation_summary=recommendation_summary,
        recommendation_record=recommendation_record,
        recommendation_mem=recommendation_mem,
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
        review_mem=review_mem,
        prior_review_record=prior_review_record,
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
        "[RUNTIME_ENABLEMENT_REVIEW] "
        f"state={record['review_state']} "
        f"limited={counts['limited']} "
        f"full={counts['full']} "
        f"confidence={record['review_confidence']:.3f}",
        flush=True,
    )
    print(
        "[RUNTIME_ENABLEMENT_REVIEW_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[RUNTIME_ENABLEMENT_REVIEW_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[RUNTIME_ENABLEMENT_REVIEW_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
