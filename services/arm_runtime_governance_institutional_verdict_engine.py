"""
ARM Runtime Governance Institutional Verdict Engine -- Step 59.

Reads:
    data/results/arm_runtime_governance_enablement_review_board_summary.json       (Step 58)
    data/results/arm_runtime_governance_enablement_review_board.json               (Step 58)
    data/results/arm_runtime_governance_enablement_review_board_memory.csv         (Step 58)
    data/results/arm_runtime_governance_enablement_recommendation_engine_summary.json (Step 57)
    data/results/arm_runtime_governance_constitutional_eligibility_board_summary.json (Step 56)
    data/results/arm_runtime_governance_admission_board_summary.json               (Step 55)
    data/results/arm_runtime_governance_readiness_gate_summary.json                (Step 54)
    data/results/arm_governance_runtime_shadow_activation_summary.json             (Step 53)
    data/results/autonomous_governance_scorecard.json                              (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/arm_constitutional_court_summary.json                           (Step 33)
    data/results/arm_supreme_governance_council_summary.json                       (Step 34)
    data/results/runtime_policy_governed.json                                      (Step 18)

Writes:
    data/results/arm_runtime_governance_institutional_verdict_engine.json
    data/results/arm_runtime_governance_institutional_verdict_engine.md
    data/results/arm_runtime_governance_institutional_verdict_engine_summary.json
    data/results/arm_runtime_governance_institutional_verdict_engine_memory.csv
    data/results/arm_runtime_governance_institutional_verdict_engine_memory.parquet

Purpose
-------
This engine answers:

    "After all governance layers, what is Triton's institutional verdict on future
    runtime governance?"

It converts runtime enablement review into a final institutional governance verdict.
This is the FINAL governance-only conclusion before any hypothetical human/operator decision.
Under review != verdict. Verdict != enabled. Institutional verdict != runtime mutation.
Verdict NEVER enables runtime.

Verdict state cascade
---------------------
    1. RUNTIME_VERDICT_INSTITUTIONAL  stable institutional verdict quality
    2. RUNTIME_VERDICT_READY            favorable verdict plausible
    3. RUNTIME_VERDICT_LIMITED          cautious verdict only
    4. RUNTIME_VERDICT_OBSERVE          verdict = observe
    5. RUNTIME_VERDICT_DORMANT          not under review; governance immature

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
* Append-only verdict memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to RUNTIME_VERDICT_DORMANT.
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

DEFAULT_REVIEW_SUM = RESULTS_DIR / "arm_runtime_governance_enablement_review_board_summary.json"
DEFAULT_REVIEW_REC = RESULTS_DIR / "arm_runtime_governance_enablement_review_board.json"
DEFAULT_REVIEW_MEM = RESULTS_DIR / "arm_runtime_governance_enablement_review_board_memory.csv"
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

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_runtime_governance_institutional_verdict_engine.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_runtime_governance_institutional_verdict_engine.md"
DEFAULT_OUT_SUMMARY = (
    RESULTS_DIR / "arm_runtime_governance_institutional_verdict_engine_summary.json"
)
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_runtime_governance_institutional_verdict_engine_memory.csv"
DEFAULT_OUT_MEM_PQ = (
    RESULTS_DIR / "arm_runtime_governance_institutional_verdict_engine_memory.parquet"
)


# -----------------------------------------------------------
# Verdict state constants
# -----------------------------------------------------------
VERDICT_DORMANT = "RUNTIME_VERDICT_DORMANT"
VERDICT_OBSERVE = "RUNTIME_VERDICT_OBSERVE"
VERDICT_LIMITED = "RUNTIME_VERDICT_LIMITED"
VERDICT_READY = "RUNTIME_VERDICT_READY"
VERDICT_INSTITUTIONAL = "RUNTIME_VERDICT_INSTITUTIONAL"

CLASS_DO_NOT_ENABLE = "DO_NOT_ENABLE_RUNTIME"
CLASS_OBSERVE_ONLY = "OBSERVE_RUNTIME_ONLY"
CLASS_LIMITED = "LIMITED_RUNTIME_SUPPORT"
CLASS_FAVORABLE = "FAVORABLE_RUNTIME_VERDICT"

REVIEW_NOT = "NOT_UNDER_REVIEW"
REVIEW_OBSERVE = "OBSERVE_REVIEW"
REVIEW_LIMITED = "LIMITED_REVIEW"
REVIEW_FULL = "FULL_REVIEW"

POSITION_DO_NOT_ENABLE = "INSTITUTIONAL_POSITION_DO_NOT_ENABLE"
POSITION_OBSERVE = "INSTITUTIONAL_POSITION_OBSERVE_ONLY"
POSITION_LIMITED = "INSTITUTIONAL_POSITION_LIMITED_SUPPORT"
POSITION_FAVORABLE = "INSTITUTIONAL_POSITION_FAVORABLE"

MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "verdict_state",
    "observe_count",
    "limited_count",
    "favorable_count",
    "verdict_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[RUNTIME_INSTITUTIONAL_VERDICT_WARN] {msg}", flush=True)


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
        for col in ("verdict_confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("observe_count", "limited_count", "favorable_count"):
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


def _scale_verdict(classification: str) -> float:
    if classification == CLASS_FAVORABLE:
        return 1.0
    if classification == CLASS_LIMITED:
        return 0.55
    if classification == CLASS_OBSERVE_ONLY:
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


def _institutional_position(classification: str) -> str:
    return {
        CLASS_FAVORABLE: POSITION_FAVORABLE,
        CLASS_LIMITED: POSITION_LIMITED,
        CLASS_OBSERVE_ONLY: POSITION_OBSERVE,
        CLASS_DO_NOT_ENABLE: POSITION_DO_NOT_ENABLE,
    }.get(classification, POSITION_DO_NOT_ENABLE)


# -----------------------------------------------------------
# Context extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    review_summary: Dict[str, Any],
    review_record: Dict[str, Any],
    review_mem: List[Dict[str, str]],
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
    verdict_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    rv = review_record.get("enablement_review") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    court_safety = _court_constitutional_safety(court_summary, council_summary)

    ctx: Dict[str, Any] = {
        "review_state": _norm_upper(
            review_summary.get("review_state") or review_record.get("review_state")
        ),
        "review_confidence": _clamp(
            _to_float(
                review_summary.get("review_confidence") or review_record.get("review_confidence")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "review_available": bool(review_summary.get("runtime_enablement_review_available")),
        "enablement_review": rv,
        "review_classification": _norm_upper(
            rv.get("runtime_enablement_review_classification")
            or review_summary.get("runtime_enablement_review_classification")
        ),
        "review_score": _clamp(
            _to_float(
                rv.get("runtime_enablement_review_score")
                or review_summary.get("runtime_enablement_review_score")
            )
            or 0.0,
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
        "governance_quality": _clamp(
            _to_float(scores.get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "constitutional_safe_review": bool(rv.get("constitutional_safe")),
        "constitutional_safety": _clamp(
            court_safety * 0.60 + (1.0 if bool(rv.get("constitutional_safe")) else 0.0) * 0.40,
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
            or review_summary.get("operator_review_required")
            or recommendation_summary.get("operator_review_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "autonomous_readiness_score": _clamp(
            _to_float(autonomous_readiness.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "review_memory_depth": len(review_mem),
        "verdict_memory_depth": len(verdict_mem),
        "observation_cycles": max(
            _to_float(review_summary.get("observation_cycles")) or 0,
            len(review_mem),
            1,
        ),
    }
    return ctx


# -----------------------------------------------------------
# Institutional verdict assessment
# -----------------------------------------------------------
def _base_verdict_score(ctx: Dict[str, Any]) -> float:
    raw = (
        ctx["review_confidence"] * 0.20
        + ctx["review_score"] * 0.16
        + ctx["recommendation_confidence"] * 0.14
        + ctx["eligibility_confidence"] * 0.12
        + ctx["admission_confidence"] * 0.08
        + ctx["readiness_confidence"] * 0.08
        + ctx["shadow_confidence"] * 0.08
        + ctx["governance_quality"] * 0.08
        + ctx["system_health_score"] * 0.04
        + ctx["constitutional_safety"] * 0.02
    )
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.18
    return round(_clamp(raw, 0.0, 1.0), 4)


def _compute_verdict_score(classification: str, ctx: Dict[str, Any]) -> float:
    raw = _base_verdict_score(ctx)
    raw *= _scale_verdict(classification)
    return round(_clamp(raw, 0.0, 1.0), 4)


def _classify_institutional_verdict(*, gate_score: float, ctx: Dict[str, Any]) -> str:
    rc = ctx["review_classification"]
    const_safe = (
        ctx["constitutional_safe_review"]
        and ctx["constitutional_safety"] >= 0.50
        and not ctx["constitution_violated"]
    )

    if not const_safe or rc == REVIEW_NOT or gate_score < 0.12:
        return CLASS_DO_NOT_ENABLE

    if rc == REVIEW_FULL and gate_score >= 0.40:
        if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
            return CLASS_LIMITED
        if ctx["system_health_stale"] or ctx["governance_quality"] < 0.45:
            return CLASS_LIMITED
        if ctx["constitutional_safety"] < 0.76:
            return CLASS_LIMITED
        return CLASS_FAVORABLE

    if rc == REVIEW_LIMITED and gate_score >= 0.26:
        if const_safe:
            return CLASS_LIMITED

    if rc == REVIEW_OBSERVE and gate_score >= 0.14:
        return CLASS_OBSERVE_ONLY

    if gate_score >= 0.16 and rc != REVIEW_NOT and const_safe:
        return CLASS_OBSERVE_ONLY

    return CLASS_DO_NOT_ENABLE


def _verdict_rationale(classification: str, verdict_score: float, ctx: Dict[str, Any]) -> str:
    templates = {
        CLASS_FAVORABLE: (
            "favorable runtime verdict: institutional governance supports future runtime "
            "consideration without runtime activation"
        ),
        CLASS_LIMITED: (
            "limited runtime support: constitutionally safe defensive posture yields cautious "
            "institutional verdict only"
        ),
        CLASS_OBSERVE_ONLY: (
            "observe runtime only: observation maturity; no enablement support in verdict"
        ),
        CLASS_DO_NOT_ENABLE: (
            "do not enable runtime: constitutional pressure or review immaturity blocks verdict"
        ),
    }
    base = templates.get(classification, templates[CLASS_DO_NOT_ENABLE])
    if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
        base += " under elevated constitutional pressure"
    return f"{base} (verdict_score={verdict_score:.2f})"


def _build_institutional_verdict(
    *,
    ctx: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    gate_score = _base_verdict_score(ctx)
    if prior:
        prior_score = _to_float(prior.get("runtime_verdict_score")) or gate_score
        gate_score = round(_clamp(gate_score * 0.65 + prior_score * 0.35, 0.0, 1.0), 4)

    classification = _classify_institutional_verdict(gate_score=gate_score, ctx=ctx)
    verdict_score = _compute_verdict_score(classification, ctx)

    rv = ctx["enablement_review"]
    review_class = ctx["review_classification"]
    review_score = ctx["review_score"]

    const_safe = (
        ctx["constitutional_safe_review"]
        and ctx["constitutional_safety"] >= 0.50
        and not ctx["constitution_violated"]
        and classification != CLASS_DO_NOT_ENABLE
    )
    future_candidate = classification in (CLASS_LIMITED, CLASS_FAVORABLE) or bool(
        rv.get("future_runtime_candidate")
    )

    conf = round(
        _clamp(
            gate_score * 0.34
            + ctx["review_confidence"] * 0.33
            + ctx["constitutional_safety"] * 0.33,
            0.0,
            1.0,
        ),
        4,
    )

    return {
        "runtime_verdict_classification": classification,
        "runtime_verdict_score": verdict_score,
        "runtime_enablement_review_classification": review_class,
        "runtime_enablement_review_score": round(review_score, 4),
        "institutional_runtime_position": _institutional_position(classification),
        "future_runtime_candidate": future_candidate,
        "constitutional_safe": const_safe,
        "confidence": conf,
        "runtime_mutation_allowed": False,
        "runtime_verdict_rationale": _verdict_rationale(
            classification,
            verdict_score,
            ctx,
        ),
    }


# -----------------------------------------------------------
# Verdict confidence and state
# -----------------------------------------------------------
def _verdict_confidence(ctx: Dict[str, Any], institutional_verdict: Dict[str, Any]) -> float:
    raw = (
        ctx["review_confidence"] * 0.18
        + ctx["recommendation_confidence"] * 0.14
        + ctx["eligibility_confidence"] * 0.12
        + ctx["admission_confidence"] * 0.10
        + ctx["readiness_confidence"] * 0.08
        + ctx["shadow_confidence"] * 0.08
        + ctx["governance_quality"] * 0.14
        + ctx["system_health_score"] * 0.10
        + ctx["constitutional_safety"] * 0.06
    )
    raw += (_to_float(institutional_verdict.get("runtime_verdict_score")) or 0.0) * 0.05

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
        "do_not_enable": 1 if classification == CLASS_DO_NOT_ENABLE else 0,
        "observe": 1 if classification == CLASS_OBSERVE_ONLY else 0,
        "limited": 1 if classification == CLASS_LIMITED else 0,
        "favorable": 1 if classification == CLASS_FAVORABLE else 0,
    }


def _classify_verdict_state(
    *,
    ctx: Dict[str, Any],
    classification: str,
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if classification == CLASS_DO_NOT_ENABLE or not ctx["review_available"]:
        reasons.append("runtime not under review; governance maturity insufficient")
        return VERDICT_DORMANT, reasons

    if counts["favorable"] >= 1 and ctx["verdict_memory_depth"] >= 2:
        reasons.append("mature governance process with stable institutional verdict quality")
        return VERDICT_INSTITUTIONAL, reasons

    if counts["favorable"] >= 1:
        reasons.append("governance maturity stable; favorable verdict plausible")
        return VERDICT_READY, reasons

    if counts["limited"] >= 1:
        reasons.append("defensive governance maturity; cautious verdict only")
        return VERDICT_LIMITED, reasons

    if counts["observe"] >= 1:
        reasons.append("observation maturity only; verdict = observe")
        return VERDICT_OBSERVE, reasons

    reasons.append("runtime not under review; governance maturity insufficient")
    return VERDICT_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations list, rationale
# -----------------------------------------------------------
def _verdict_booleans(
    state: str,
    institutional_verdict: Dict[str, Any],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    classification = institutional_verdict.get(
        "runtime_verdict_classification", CLASS_DO_NOT_ENABLE
    )
    return {
        "runtime_verdict_available": bool(institutional_verdict),
        "limited_runtime_support_available": counts["limited"] > 0 or counts["favorable"] > 0,
        "favorable_runtime_verdict_available": counts["favorable"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"]
            or institutional_verdict.get("future_runtime_candidate", False)
            or classification in (CLASS_LIMITED, CLASS_FAVORABLE)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "verdict_memory_reliable": state == VERDICT_INSTITUTIONAL,
    }


def _recommendations_list(state: str, counts: Dict[str, int]) -> List[str]:
    recs = [
        "Continue governance maturity accumulation",
        "Continue shadow activation rehearsal",
        "Maintain defensive constitutional posture",
        "Avoid premature runtime assumptions",
        "Maintain runtime mutation lock",
    ]
    if counts["favorable"] > 0 or counts["limited"] > 0:
        recs.append("Escalate institutional verdict to operator decision")
    if state == VERDICT_DORMANT:
        recs.append("Resolve enablement review before institutional verdict")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(
    institutional_verdict: Dict[str, Any],
    state: str,
    ctx: Dict[str, Any],
) -> str:
    classification = institutional_verdict.get(
        "runtime_verdict_classification", CLASS_DO_NOT_ENABLE
    )
    if classification == CLASS_DO_NOT_ENABLE or ctx["constitution_violated"]:
        return (
            "Triton's institutional verdict does not support runtime governance enablement "
            "because constitutional pressure remains elevated despite improving governance maturity."
        )
    if classification == CLASS_FAVORABLE:
        return (
            "Triton's institutional verdict is favorable toward future runtime governance "
            "consideration without activating or mutating live runtime."
        )
    if classification == CLASS_LIMITED:
        return (
            "Triton's institutional verdict offers limited runtime support under defensive "
            "conditions; favorable verdict remains constrained."
        )
    if classification == CLASS_OBSERVE_ONLY:
        return "Institutional verdict remains observe-only; governance maturity is immature."
    return (
        "Runtime governance institutional verdict completed without runtime mutation. "
        "Under review != verdict. Verdict != enabled."
    )


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    verdict_confidence: float,
    institutional_verdict: Dict[str, Any],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
    ctx: Dict[str, Any],
) -> str:
    iv = institutional_verdict
    lines = [
        "# Triton Runtime Governance Institutional Verdict",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Verdict State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| verdict_confidence | {verdict_confidence:.3f} |",
        f"| classification | {iv.get('runtime_verdict_classification', CLASS_DO_NOT_ENABLE)} |",
        f"| verdict_score | {_to_float(iv.get('runtime_verdict_score')) or 0.0:.3f} |",
        f"| review_classification | {iv.get('runtime_enablement_review_classification', REVIEW_NOT)} |",
        f"| review_score | {_to_float(iv.get('runtime_enablement_review_score')) or 0.0:.3f} |",
        f"| institutional_runtime_position | {iv.get('institutional_runtime_position', POSITION_DO_NOT_ENABLE)} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Institutional Runtime Verdict",
        "",
        f"- **Classification:** {iv.get('runtime_verdict_classification', CLASS_DO_NOT_ENABLE)}",
        f"- **Constitutional safe:** {iv.get('constitutional_safe', False)}",
        f"- **Future runtime candidate:** {iv.get('future_runtime_candidate', False)}",
        f"- **Review state:** {ctx.get('review_state', 'UNKNOWN')}",
        "",
        f"_{iv.get('runtime_verdict_rationale', '')}_",
        "",
        "## Runtime Governance Position",
        "",
        f"| tier | count |",
        f"|---|---|",
        f"| observe | {counts['observe']} |",
        f"| limited | {counts['limited']} |",
        f"| favorable | {counts['favorable']} |",
        "",
        f"Review confidence: {ctx['review_confidence']:.3f} | "
        f"Recommendation confidence: {ctx['recommendation_confidence']:.3f} | "
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
            "This is the final governance-only conclusion before any hypothetical human/operator "
            "decision. Under review != verdict. Verdict != enabled. No live runtime policy is changed.",
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
    verdict_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "verdict_state": state,
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "favorable_count": counts["favorable"],
        "verdict_confidence": round(verdict_confidence, 6),
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
def build_institutional_verdict(
    *,
    review_summary: Dict[str, Any],
    review_record: Dict[str, Any],
    review_mem: List[Dict[str, str]],
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
    verdict_mem: List[Dict[str, str]],
    prior_verdict_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        review_summary=review_summary,
        review_record=review_record,
        review_mem=review_mem,
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
        verdict_mem=verdict_mem,
    )

    prior_iv = prior_verdict_record.get("institutional_verdict") or {}
    institutional_verdict = _build_institutional_verdict(ctx=ctx, prior=prior_iv or None)
    verdict_confidence = _verdict_confidence(ctx, institutional_verdict)
    counts = _classification_counts(
        institutional_verdict.get("runtime_verdict_classification", CLASS_DO_NOT_ENABLE)
    )

    state, reasons = _classify_verdict_state(
        ctx=ctx,
        classification=institutional_verdict.get(
            "runtime_verdict_classification",
            CLASS_DO_NOT_ENABLE,
        ),
        counts=counts,
    )

    booleans = _verdict_booleans(state, institutional_verdict, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(institutional_verdict, state, ctx)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        verdict_confidence=verdict_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(verdict_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        verdict_confidence=verdict_confidence,
        institutional_verdict=institutional_verdict,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
        ctx=ctx,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_institutional_verdict_engine",
        "engine_version": 1,
        "verdict_state": state,
        "verdict_confidence": verdict_confidence,
        "verdict_reasons": reasons,
        "institutional_verdict": institutional_verdict,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "review_vs_verdict_note": (
            "Under review != verdict. Verdict != enabled. "
            "Institutional verdict != runtime mutation. Live runtime is never changed."
        ),
        "constitutional_supremacy_note": (
            "Institutional verdict cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "verdict_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "live_runtime_regime": ctx["regime"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutated": False,
            "runtime_mutation_allowed": False,
            "verdict_evaluation_only": True,
        },
        "inputs_seen": {
            "arm_runtime_governance_enablement_review_board_summary": bool(review_summary),
            "arm_runtime_governance_enablement_review_board_record": bool(review_record),
            "arm_runtime_governance_enablement_review_board_memory_rows": len(review_mem),
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
            "existing_verdict_memory_rows": len(verdict_mem),
            "prior_institutional_verdict": bool(prior_iv),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_institutional_verdict_engine",
        "verdict_state": state,
        "verdict_confidence": verdict_confidence,
        "runtime_verdict_available": booleans["runtime_verdict_available"],
        "limited_runtime_support_available": booleans["limited_runtime_support_available"],
        "favorable_runtime_verdict_available": booleans["favorable_runtime_verdict_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "verdict_memory_reliable": booleans["verdict_memory_reliable"],
        "do_not_enable_count": counts["do_not_enable"],
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "favorable_count": counts["favorable"],
        "observation_cycles": ctx["observation_cycles"],
        "runtime_verdict_classification": institutional_verdict.get(
            "runtime_verdict_classification"
        ),
        "runtime_verdict_score": institutional_verdict.get("runtime_verdict_score"),
        "runtime_enablement_review_classification": institutional_verdict.get(
            "runtime_enablement_review_classification",
        ),
        "institutional_runtime_position": institutional_verdict.get(
            "institutional_runtime_position"
        ),
        "future_runtime_candidate": institutional_verdict.get("future_runtime_candidate"),
        "n_recommendations": len(recommendations),
        "verdict_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM runtime governance institutional verdict engine (Step 59). "
            "Final governance verdict without mutating runtime. No broker calls."
        ),
    )
    p.add_argument("--review-summary", default=str(DEFAULT_REVIEW_SUM))
    p.add_argument("--review-record", default=str(DEFAULT_REVIEW_REC))
    p.add_argument("--review-mem", default=str(DEFAULT_REVIEW_MEM))
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
        "[RUNTIME_INSTITUTIONAL_VERDICT] starting "
        "(read-only verdict evaluation; no runtime mutation; no broker calls)",
        flush=True,
    )

    review_summary = _safe_read_json(
        Path(args.review_summary),
        label="arm_runtime_governance_enablement_review_board_summary.json",
    )
    review_record = _safe_read_json(
        Path(args.review_record), label="arm_runtime_governance_enablement_review_board.json"
    )
    review_mem = _safe_read_csv_rows(
        Path(args.review_mem), label="arm_runtime_governance_enablement_review_board_memory.csv"
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
    verdict_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv),
        label="arm_runtime_governance_institutional_verdict_engine_memory.csv",
    )
    prior_verdict_record = _safe_read_json(
        Path(args.out_json), label="prior_arm_runtime_governance_institutional_verdict_engine.json"
    )

    record, summary, md, merged_memory = build_institutional_verdict(
        review_summary=review_summary,
        review_record=review_record,
        review_mem=review_mem,
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
        verdict_mem=verdict_mem,
        prior_verdict_record=prior_verdict_record,
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
        "[RUNTIME_INSTITUTIONAL_VERDICT] "
        f"state={record['verdict_state']} "
        f"limited={counts['limited']} "
        f"favorable={counts['favorable']} "
        f"confidence={record['verdict_confidence']:.3f}",
        flush=True,
    )
    print(
        "[RUNTIME_INSTITUTIONAL_VERDICT_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[RUNTIME_INSTITUTIONAL_VERDICT_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[RUNTIME_INSTITUTIONAL_VERDICT_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
