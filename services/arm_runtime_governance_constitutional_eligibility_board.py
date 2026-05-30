"""
ARM Runtime Governance Constitutional Eligibility Board -- Step 56.

Reads:
    data/results/arm_runtime_governance_admission_board_summary.json       (Step 55)
    data/results/arm_runtime_governance_admission_board.json               (Step 55)
    data/results/arm_runtime_governance_admission_board_memory.csv         (Step 55)
    data/results/arm_runtime_governance_readiness_gate_summary.json        (Step 54)
    data/results/arm_governance_runtime_shadow_activation_summary.json     (Step 53)
    data/results/autonomous_governance_scorecard.json                      (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/arm_constitutional_court_summary.json                     (Step 33)
    data/results/arm_supreme_governance_council_summary.json               (Step 34)
    data/results/runtime_policy_governed.json                                (Step 18)

Writes:
    data/results/arm_runtime_governance_constitutional_eligibility_board.json
    data/results/arm_runtime_governance_constitutional_eligibility_board.md
    data/results/arm_runtime_governance_constitutional_eligibility_board_summary.json
    data/results/arm_runtime_governance_constitutional_eligibility_board_memory.csv
    data/results/arm_runtime_governance_constitutional_eligibility_board_memory.parquet

Purpose
-------
This engine answers:

    "Even if runtime governance is admitted, is it constitutionally eligible?"

It converts runtime governance admission into constitutional runtime governance eligibility.
This is the FINAL constitutional gate before any hypothetical future runtime enablement.
Runtime admitted != constitutionally eligible. Constitutionally eligible != runtime active.
Constitutional eligibility != runtime mutation. Constitutional eligibility NEVER mutates runtime.

Constitutional eligibility state cascade
----------------------------------------
    1. RUNTIME_CONSTITUTIONAL_ELIGIBILITY_INSTITUTIONAL  long-run constitutional stability
    2. RUNTIME_CONSTITUTIONAL_ELIGIBILITY_READY            constitutional stability sufficient
    3. RUNTIME_CONSTITUTIONAL_ELIGIBILITY_LIMITED          defensive posture; limited eligibility
    4. RUNTIME_CONSTITUTIONAL_ELIGIBILITY_OBSERVE          observation only; constitutionally inadmissible
    5. RUNTIME_CONSTITUTIONAL_ELIGIBILITY_DORMANT          not admitted; constitutional concern

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
* Append-only constitutional eligibility memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to RUNTIME_CONSTITUTIONAL_ELIGIBILITY_DORMANT.
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

DEFAULT_ADMISSION_SUM = RESULTS_DIR / "arm_runtime_governance_admission_board_summary.json"
DEFAULT_ADMISSION_REC = RESULTS_DIR / "arm_runtime_governance_admission_board.json"
DEFAULT_ADMISSION_MEM = RESULTS_DIR / "arm_runtime_governance_admission_board_memory.csv"
DEFAULT_READINESS_SUM = RESULTS_DIR / "arm_runtime_governance_readiness_gate_summary.json"
DEFAULT_SHADOW_SUM = RESULTS_DIR / "arm_governance_runtime_shadow_activation_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_AUTO = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_runtime_governance_constitutional_eligibility_board.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_runtime_governance_constitutional_eligibility_board.md"
DEFAULT_OUT_SUMMARY = (
    RESULTS_DIR / "arm_runtime_governance_constitutional_eligibility_board_summary.json"
)
DEFAULT_OUT_MEM_CSV = (
    RESULTS_DIR / "arm_runtime_governance_constitutional_eligibility_board_memory.csv"
)
DEFAULT_OUT_MEM_PQ = (
    RESULTS_DIR / "arm_runtime_governance_constitutional_eligibility_board_memory.parquet"
)


# -----------------------------------------------------------
# Constitutional eligibility state constants
# -----------------------------------------------------------
ELIG_DORMANT = "RUNTIME_CONSTITUTIONAL_ELIGIBILITY_DORMANT"
ELIG_OBSERVE = "RUNTIME_CONSTITUTIONAL_ELIGIBILITY_OBSERVE"
ELIG_LIMITED = "RUNTIME_CONSTITUTIONAL_ELIGIBILITY_LIMITED"
ELIG_READY = "RUNTIME_CONSTITUTIONAL_ELIGIBILITY_READY"
ELIG_INSTITUTIONAL = "RUNTIME_CONSTITUTIONAL_ELIGIBILITY_INSTITUTIONAL"

CLASS_NOT_ELIGIBLE = "NOT_CONSTITUTIONALLY_ELIGIBLE"
CLASS_OBSERVE = "OBSERVE_CONSTITUTIONAL_ELIGIBILITY"
CLASS_LIMITED = "LIMITED_CONSTITUTIONAL_ELIGIBILITY"
CLASS_FULL = "FULL_CONSTITUTIONAL_ELIGIBILITY"

ADMISSION_NOT = "NOT_RUNTIME_ADMITTED"
ADMISSION_OBSERVE = "OBSERVE_RUNTIME_ADMISSION"
ADMISSION_LIMITED = "LIMITED_RUNTIME_ADMISSION"
ADMISSION_FULL = "FULL_RUNTIME_ADMISSION"

MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "constitutional_eligibility_state",
    "observe_count",
    "limited_count",
    "full_count",
    "constitutional_eligibility_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[RUNTIME_CONSTITUTIONAL_ELIGIBILITY_WARN] {msg}", flush=True)


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
        for col in ("constitutional_eligibility_confidence",):
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


def _scale_eligibility(classification: str) -> float:
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
    admission_summary: Dict[str, Any],
    admission_record: Dict[str, Any],
    admission_mem: List[Dict[str, str]],
    readiness_summary: Dict[str, Any],
    shadow_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    eligibility_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    ra = admission_record.get("runtime_admission") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    court_safety = _court_constitutional_safety(court_summary, council_summary)

    ctx: Dict[str, Any] = {
        "admission_state": _norm_upper(
            admission_summary.get("admission_state") or admission_record.get("admission_state")
        ),
        "admission_confidence": _clamp(
            _to_float(
                admission_summary.get("admission_confidence")
                or admission_record.get("admission_confidence")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "admission_available": bool(admission_summary.get("runtime_admission_available")),
        "runtime_admission": ra,
        "admission_classification": _norm_upper(
            ra.get("runtime_admission_classification")
            or admission_summary.get("runtime_admission_classification")
        ),
        "admission_score": _clamp(
            _to_float(
                ra.get("runtime_admission_score")
                or admission_summary.get("runtime_admission_score")
            )
            or 0.0,
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
        "constitutional_safe_admission": bool(ra.get("constitutional_safe")),
        "constitutional_safety": _clamp(
            court_safety * 0.60 + (1.0 if bool(ra.get("constitutional_safe")) else 0.0) * 0.40,
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
            or admission_summary.get("operator_review_required")
            or shadow_summary.get("operator_review_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "autonomous_readiness_score": _clamp(
            _to_float(autonomous_readiness.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "admission_memory_depth": len(admission_mem),
        "eligibility_memory_depth": len(eligibility_mem),
        "observation_cycles": max(
            _to_float(admission_summary.get("observation_cycles")) or 0,
            len(admission_mem),
            1,
        ),
    }
    return ctx


# -----------------------------------------------------------
# Constitutional eligibility assessment
# -----------------------------------------------------------
def _base_eligibility_score(ctx: Dict[str, Any]) -> float:
    raw = (
        ctx["admission_confidence"] * 0.26
        + ctx["admission_score"] * 0.20
        + ctx["readiness_confidence"] * 0.16
        + ctx["shadow_confidence"] * 0.12
        + ctx["governance_quality"] * 0.12
        + ctx["system_health_score"] * 0.08
        + ctx["constitutional_safety"] * 0.06
    )
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.25
    return round(_clamp(raw, 0.0, 1.0), 4)


def _compute_eligibility_score(classification: str, ctx: Dict[str, Any]) -> float:
    raw = _base_eligibility_score(ctx)
    raw *= _scale_eligibility(classification)
    return round(_clamp(raw, 0.0, 1.0), 4)


def _classify_constitutional_eligibility(*, gate_score: float, ctx: Dict[str, Any]) -> str:
    ac = ctx["admission_classification"]
    const_safe = (
        ctx["constitutional_safe_admission"]
        and ctx["constitutional_safety"] >= 0.50
        and not ctx["constitution_violated"]
    )

    if not const_safe or ac == ADMISSION_NOT or gate_score < 0.16:
        return CLASS_NOT_ELIGIBLE

    if ac == ADMISSION_FULL and gate_score >= 0.46:
        if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
            return CLASS_LIMITED
        if ctx["system_health_stale"] or ctx["governance_quality"] < 0.45:
            return CLASS_LIMITED
        if ctx["constitutional_safety"] < 0.70:
            return CLASS_LIMITED
        return CLASS_FULL

    if ac == ADMISSION_LIMITED and gate_score >= 0.32:
        if const_safe:
            return CLASS_LIMITED

    if ac == ADMISSION_OBSERVE and gate_score >= 0.18:
        return CLASS_OBSERVE

    if gate_score >= 0.22 and ac != ADMISSION_NOT and const_safe:
        return CLASS_OBSERVE

    return CLASS_NOT_ELIGIBLE


def _eligibility_rationale(
    classification: str, eligibility_score: float, ctx: Dict[str, Any]
) -> str:
    templates = {
        CLASS_FULL: (
            "full constitutional eligibility: runtime governance is constitutionally eligible "
            "for future consideration without runtime activation"
        ),
        CLASS_LIMITED: (
            "limited constitutional eligibility: constitutionally safe defensive posture "
            "supports limited eligibility only"
        ),
        CLASS_OBSERVE: (
            "observe constitutional eligibility: observation maturity only; "
            "runtime constitutionally inadmissible"
        ),
        CLASS_NOT_ELIGIBLE: (
            "not constitutionally eligible: constitutional pressure or admission immaturity "
            "blocks eligibility"
        ),
    }
    base = templates.get(classification, templates[CLASS_NOT_ELIGIBLE])
    if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
        base += " under elevated constitutional pressure"
    return f"{base} (eligibility_score={eligibility_score:.2f})"


def _build_constitutional_eligibility(
    *,
    ctx: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    gate_score = _base_eligibility_score(ctx)
    if prior:
        prior_score = _to_float(prior.get("runtime_constitutional_eligibility_score")) or gate_score
        gate_score = round(_clamp(gate_score * 0.65 + prior_score * 0.35, 0.0, 1.0), 4)

    classification = _classify_constitutional_eligibility(gate_score=gate_score, ctx=ctx)
    eligibility_score = _compute_eligibility_score(classification, ctx)

    ra = ctx["runtime_admission"]
    admission_class = ctx["admission_classification"]
    admission_score = ctx["admission_score"]

    const_safe = (
        ctx["constitutional_safe_admission"]
        and ctx["constitutional_safety"] >= 0.50
        and not ctx["constitution_violated"]
        and classification != CLASS_NOT_ELIGIBLE
    )
    constitutionally_eligible = classification == CLASS_FULL
    future_candidate = classification in (CLASS_LIMITED, CLASS_FULL) or bool(
        ra.get("future_runtime_candidate")
    )

    conf = round(
        _clamp(
            gate_score * 0.40
            + ctx["admission_confidence"] * 0.30
            + ctx["constitutional_safety"] * 0.30,
            0.0,
            1.0,
        ),
        4,
    )

    return {
        "runtime_constitutional_eligibility_classification": classification,
        "runtime_constitutional_eligibility_score": eligibility_score,
        "runtime_admission_classification": admission_class,
        "runtime_admission_score": round(admission_score, 4),
        "constitutionally_runtime_eligible": constitutionally_eligible,
        "future_runtime_candidate": future_candidate,
        "constitutional_safe": const_safe,
        "confidence": conf,
        "runtime_mutation_allowed": False,
        "constitutional_eligibility_rationale": _eligibility_rationale(
            classification,
            eligibility_score,
            ctx,
        ),
    }


# -----------------------------------------------------------
# Confidence and state
# -----------------------------------------------------------
def _constitutional_eligibility_confidence(
    ctx: Dict[str, Any],
    constitutional_eligibility: Dict[str, Any],
) -> float:
    raw = (
        ctx["admission_confidence"] * 0.24
        + ctx["readiness_confidence"] * 0.18
        + ctx["shadow_confidence"] * 0.14
        + ctx["governance_quality"] * 0.16
        + ctx["system_health_score"] * 0.14
        + ctx["constitutional_safety"] * 0.14
    )
    raw += (
        _to_float(constitutional_eligibility.get("runtime_constitutional_eligibility_score")) or 0.0
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
        "not_eligible": 1 if classification == CLASS_NOT_ELIGIBLE else 0,
        "observe": 1 if classification == CLASS_OBSERVE else 0,
        "limited": 1 if classification == CLASS_LIMITED else 0,
        "full": 1 if classification == CLASS_FULL else 0,
    }


def _classify_eligibility_state(
    *,
    ctx: Dict[str, Any],
    classification: str,
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if classification == CLASS_NOT_ELIGIBLE or not ctx["admission_available"]:
        reasons.append("runtime not admitted; constitutional concern exists")
        return ELIG_DORMANT, reasons

    if counts["full"] >= 1 and ctx["eligibility_memory_depth"] >= 2:
        reasons.append("long-run constitutional governance stability established")
        return ELIG_INSTITUTIONAL, reasons

    if counts["full"] >= 1:
        reasons.append("constitutional stability sufficient; governance maturity persistent")
        return ELIG_READY, reasons

    if counts["limited"] >= 1:
        reasons.append("constitutionally safe defensive posture; limited eligibility only")
        return ELIG_LIMITED, reasons

    if counts["observe"] >= 1:
        reasons.append("observation maturity only; runtime constitutionally inadmissible")
        return ELIG_OBSERVE, reasons

    reasons.append("runtime not admitted; constitutional concern exists")
    return ELIG_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _eligibility_booleans(
    state: str,
    constitutional_eligibility: Dict[str, Any],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    classification = constitutional_eligibility.get(
        "runtime_constitutional_eligibility_classification",
        CLASS_NOT_ELIGIBLE,
    )
    return {
        "runtime_constitutional_eligibility_available": bool(constitutional_eligibility),
        "limited_runtime_constitutional_eligibility_available": (
            counts["limited"] > 0 or counts["full"] > 0
        ),
        "full_runtime_constitutional_eligibility_available": counts["full"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"]
            or constitutional_eligibility.get("future_runtime_candidate", False)
            or classification in (CLASS_LIMITED, CLASS_FULL)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "constitutional_eligibility_memory_reliable": state == ELIG_INSTITUTIONAL,
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
        recs.append("Escalate constitutional eligibility to operator review")
    if state == ELIG_DORMANT:
        recs.append("Resolve constitutional pressure before eligibility consideration")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(
    constitutional_eligibility: Dict[str, Any],
    state: str,
    ctx: Dict[str, Any],
) -> str:
    classification = constitutional_eligibility.get(
        "runtime_constitutional_eligibility_classification",
        CLASS_NOT_ELIGIBLE,
    )
    if classification == CLASS_NOT_ELIGIBLE or ctx["constitution_violated"]:
        return (
            "Triton remains not constitutionally eligible for runtime governance because "
            "constitutional pressure remains elevated despite improving governance maturity."
        )
    if classification == CLASS_FULL:
        return (
            "Triton grants full constitutional eligibility for runtime governance "
            "without activating or mutating live runtime."
        )
    if classification == CLASS_LIMITED:
        return (
            "Triton grants limited constitutional eligibility under defensive governance "
            "conditions; full eligibility remains constrained."
        )
    if classification == CLASS_OBSERVE:
        return "Constitutional eligibility remains observe-only; governance maturity is immature."
    return (
        "Runtime governance constitutional eligibility board completed without runtime mutation. "
        "Runtime admitted != constitutionally eligible."
    )


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    eligibility_confidence: float,
    constitutional_eligibility: Dict[str, Any],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
    ctx: Dict[str, Any],
) -> str:
    ce = constitutional_eligibility
    lines = [
        "# Triton Runtime Governance Constitutional Eligibility Board",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Constitutional Eligibility State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| constitutional_eligibility_confidence | {eligibility_confidence:.3f} |",
        f"| classification | {ce.get('runtime_constitutional_eligibility_classification', CLASS_NOT_ELIGIBLE)} |",
        f"| eligibility_score | {_to_float(ce.get('runtime_constitutional_eligibility_score')) or 0.0:.3f} |",
        f"| admission_classification | {ce.get('runtime_admission_classification', ADMISSION_NOT)} |",
        f"| admission_score | {_to_float(ce.get('runtime_admission_score')) or 0.0:.3f} |",
        f"| constitutionally_runtime_eligible | {ce.get('constitutionally_runtime_eligible', False)} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Runtime Constitutional Eligibility",
        "",
        f"- **Classification:** {ce.get('runtime_constitutional_eligibility_classification', CLASS_NOT_ELIGIBLE)}",
        f"- **Constitutional safe:** {ce.get('constitutional_safe', False)}",
        f"- **Future runtime candidate:** {ce.get('future_runtime_candidate', False)}",
        f"- **Admission state:** {ctx.get('admission_state', 'UNKNOWN')}",
        "",
        f"_{ce.get('constitutional_eligibility_rationale', '')}_",
        "",
        "## Governance Constitutional Safety",
        "",
        f"| tier | count |",
        f"|---|---|",
        f"| observe | {counts['observe']} |",
        f"| limited | {counts['limited']} |",
        f"| full | {counts['full']} |",
        "",
        f"Admission confidence: {ctx['admission_confidence']:.3f} | "
        f"Constitutional safety: {ctx['constitutional_safety']:.3f} | "
        f"Court ruling: {ctx.get('court_ruling', 'UNKNOWN')} | "
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
            "This is the final constitutional gate before any hypothetical future runtime enablement. "
            "Runtime admitted != constitutionally eligible. Constitutionally eligible != runtime active. "
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
    eligibility_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "constitutional_eligibility_state": state,
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "constitutional_eligibility_confidence": round(eligibility_confidence, 6),
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
def build_constitutional_eligibility_board(
    *,
    admission_summary: Dict[str, Any],
    admission_record: Dict[str, Any],
    admission_mem: List[Dict[str, str]],
    readiness_summary: Dict[str, Any],
    shadow_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    eligibility_mem: List[Dict[str, str]],
    prior_eligibility_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        admission_summary=admission_summary,
        admission_record=admission_record,
        admission_mem=admission_mem,
        readiness_summary=readiness_summary,
        shadow_summary=shadow_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        autonomous_readiness=autonomous_readiness,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        eligibility_mem=eligibility_mem,
    )

    prior_ce = prior_eligibility_record.get("constitutional_eligibility") or {}
    constitutional_eligibility = _build_constitutional_eligibility(
        ctx=ctx,
        prior=prior_ce or None,
    )
    eligibility_confidence = _constitutional_eligibility_confidence(ctx, constitutional_eligibility)
    counts = _classification_counts(
        constitutional_eligibility.get(
            "runtime_constitutional_eligibility_classification",
            CLASS_NOT_ELIGIBLE,
        )
    )

    state, reasons = _classify_eligibility_state(
        ctx=ctx,
        classification=constitutional_eligibility.get(
            "runtime_constitutional_eligibility_classification",
            CLASS_NOT_ELIGIBLE,
        ),
        counts=counts,
    )

    booleans = _eligibility_booleans(state, constitutional_eligibility, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(constitutional_eligibility, state, ctx)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        eligibility_confidence=eligibility_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(eligibility_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        eligibility_confidence=eligibility_confidence,
        constitutional_eligibility=constitutional_eligibility,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
        ctx=ctx,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_constitutional_eligibility_board",
        "engine_version": 1,
        "constitutional_eligibility_state": state,
        "constitutional_eligibility_confidence": eligibility_confidence,
        "constitutional_eligibility_reasons": reasons,
        "constitutional_eligibility": constitutional_eligibility,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "admitted_vs_eligible_note": (
            "Runtime admitted != constitutionally eligible. "
            "Constitutionally eligible != runtime active. "
            "Constitutional eligibility != runtime mutation. Live runtime is never changed."
        ),
        "constitutional_supremacy_note": (
            "Runtime constitutional eligibility cannot override the constitution, "
            "constitutional court, capital preservation doctrine, or operator supremacy."
        ),
        "eligibility_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "live_runtime_regime": ctx["regime"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutated": False,
            "runtime_mutation_allowed": False,
            "constitutional_evaluation_only": True,
        },
        "inputs_seen": {
            "arm_runtime_governance_admission_board_summary": bool(admission_summary),
            "arm_runtime_governance_admission_board_record": bool(admission_record),
            "arm_runtime_governance_admission_board_memory_rows": len(admission_mem),
            "arm_runtime_governance_readiness_gate_summary": bool(readiness_summary),
            "arm_governance_runtime_shadow_activation_summary": bool(shadow_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(autonomous_readiness),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_constitutional_eligibility_memory_rows": len(eligibility_mem),
            "prior_constitutional_eligibility": bool(prior_ce),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_constitutional_eligibility_board",
        "constitutional_eligibility_state": state,
        "constitutional_eligibility_confidence": eligibility_confidence,
        "runtime_constitutional_eligibility_available": booleans[
            "runtime_constitutional_eligibility_available"
        ],
        "limited_runtime_constitutional_eligibility_available": booleans[
            "limited_runtime_constitutional_eligibility_available"
        ],
        "full_runtime_constitutional_eligibility_available": booleans[
            "full_runtime_constitutional_eligibility_available"
        ],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "constitutional_eligibility_memory_reliable": booleans[
            "constitutional_eligibility_memory_reliable"
        ],
        "not_eligible_count": counts["not_eligible"],
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "observation_cycles": ctx["observation_cycles"],
        "runtime_constitutional_eligibility_classification": constitutional_eligibility.get(
            "runtime_constitutional_eligibility_classification",
        ),
        "runtime_constitutional_eligibility_score": constitutional_eligibility.get(
            "runtime_constitutional_eligibility_score",
        ),
        "runtime_admission_classification": constitutional_eligibility.get(
            "runtime_admission_classification"
        ),
        "constitutionally_runtime_eligible": constitutional_eligibility.get(
            "constitutionally_runtime_eligible"
        ),
        "n_recommendations": len(recommendations),
        "constitutional_eligibility_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM runtime governance constitutional eligibility board (Step 56). "
            "Final constitutional gate without mutating runtime. No broker calls."
        ),
    )
    p.add_argument("--admission-summary", default=str(DEFAULT_ADMISSION_SUM))
    p.add_argument("--admission-record", default=str(DEFAULT_ADMISSION_REC))
    p.add_argument("--admission-mem", default=str(DEFAULT_ADMISSION_MEM))
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
        "[RUNTIME_CONSTITUTIONAL_ELIGIBILITY] starting "
        "(read-only constitutional evaluation; no runtime mutation; no broker calls)",
        flush=True,
    )

    admission_summary = _safe_read_json(
        Path(args.admission_summary), label="arm_runtime_governance_admission_board_summary.json"
    )
    admission_record = _safe_read_json(
        Path(args.admission_record), label="arm_runtime_governance_admission_board.json"
    )
    admission_mem = _safe_read_csv_rows(
        Path(args.admission_mem), label="arm_runtime_governance_admission_board_memory.csv"
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
    eligibility_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv),
        label="arm_runtime_governance_constitutional_eligibility_board_memory.csv",
    )
    prior_eligibility_record = _safe_read_json(
        Path(args.out_json),
        label="prior_arm_runtime_governance_constitutional_eligibility_board.json",
    )

    record, summary, md, merged_memory = build_constitutional_eligibility_board(
        admission_summary=admission_summary,
        admission_record=admission_record,
        admission_mem=admission_mem,
        readiness_summary=readiness_summary,
        shadow_summary=shadow_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        autonomous_readiness=autonomous_readiness,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        eligibility_mem=eligibility_mem,
        prior_eligibility_record=prior_eligibility_record,
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
        "[RUNTIME_CONSTITUTIONAL_ELIGIBILITY] "
        f"state={record['constitutional_eligibility_state']} "
        f"limited={counts['limited']} "
        f"full={counts['full']} "
        f"confidence={record['constitutional_eligibility_confidence']:.3f}",
        flush=True,
    )
    print(
        "[RUNTIME_CONSTITUTIONAL_ELIGIBILITY_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[RUNTIME_CONSTITUTIONAL_ELIGIBILITY_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[RUNTIME_CONSTITUTIONAL_ELIGIBILITY_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
