"""
ARM Runtime Governance Admission Board -- Step 55.

Reads:
    data/results/arm_runtime_governance_readiness_gate_summary.json  (Step 54)
    data/results/arm_runtime_governance_readiness_gate.json          (Step 54)
    data/results/arm_runtime_governance_readiness_gate_memory.csv    (Step 54)
    data/results/arm_governance_runtime_shadow_activation_summary.json (Step 53)
    data/results/arm_governance_doctrine_activation_eligibility_summary.json (Step 52)
    data/results/arm_governance_doctrine_activation_authorization_summary.json (Step 51)
    data/results/autonomous_governance_scorecard.json                (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/arm_constitutional_court_summary.json               (Step 33)
    data/results/arm_supreme_governance_council_summary.json         (Step 34)
    data/results/runtime_policy_governed.json                        (Step 18)

Writes:
    data/results/arm_runtime_governance_admission_board.json
    data/results/arm_runtime_governance_admission_board.md
    data/results/arm_runtime_governance_admission_board_summary.json
    data/results/arm_runtime_governance_admission_board_memory.csv
    data/results/arm_runtime_governance_admission_board_memory.parquet

Purpose
-------
This engine answers:

    "Should Triton be institutionally admitted to future runtime governance?"

It converts runtime governance readiness into institutional runtime governance admission.
This is the institutional runtime admission committee.
Runtime ready != runtime admitted. Runtime admitted != runtime active.
Runtime admission != runtime mutation. Runtime admission NEVER mutates runtime policy.

Admission state cascade
-----------------------
    1. RUNTIME_ADMISSION_INSTITUTIONAL  mature governance process; stable institutional admission
    2. RUNTIME_ADMISSION_READY            governance maturity stable; admission plausible
    3. RUNTIME_ADMISSION_LIMITED          defensive maturity; limited admission consideration
    4. RUNTIME_ADMISSION_OBSERVE          observation maturity only; runtime not admitted
    5. RUNTIME_ADMISSION_DORMANT          runtime inadmissible; governance immature

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
* Append-only admission memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to RUNTIME_ADMISSION_DORMANT.
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

DEFAULT_READINESS_SUM = RESULTS_DIR / "arm_runtime_governance_readiness_gate_summary.json"
DEFAULT_READINESS_REC = RESULTS_DIR / "arm_runtime_governance_readiness_gate.json"
DEFAULT_READINESS_MEM = RESULTS_DIR / "arm_runtime_governance_readiness_gate_memory.csv"
DEFAULT_SHADOW_SUM = RESULTS_DIR / "arm_governance_runtime_shadow_activation_summary.json"
DEFAULT_ELIGIBILITY_SUM = (
    RESULTS_DIR / "arm_governance_doctrine_activation_eligibility_summary.json"
)
DEFAULT_AUTHORIZATION_SUM = (
    RESULTS_DIR / "arm_governance_doctrine_activation_authorization_summary.json"
)
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_AUTO = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_runtime_governance_admission_board.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_runtime_governance_admission_board.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_runtime_governance_admission_board_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_runtime_governance_admission_board_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_runtime_governance_admission_board_memory.parquet"


# -----------------------------------------------------------
# Admission state constants
# -----------------------------------------------------------
ADMISSION_DORMANT = "RUNTIME_ADMISSION_DORMANT"
ADMISSION_OBSERVE = "RUNTIME_ADMISSION_OBSERVE"
ADMISSION_LIMITED = "RUNTIME_ADMISSION_LIMITED"
ADMISSION_READY = "RUNTIME_ADMISSION_READY"
ADMISSION_INSTITUTIONAL = "RUNTIME_ADMISSION_INSTITUTIONAL"

CLASS_NOT_ADMITTED = "NOT_RUNTIME_ADMITTED"
CLASS_OBSERVE = "OBSERVE_RUNTIME_ADMISSION"
CLASS_LIMITED = "LIMITED_RUNTIME_ADMISSION"
CLASS_FULL = "FULL_RUNTIME_ADMISSION"

READINESS_NOT = "NOT_RUNTIME_READY"
READINESS_OBSERVE = "OBSERVE_RUNTIME_READINESS"
READINESS_LIMITED = "LIMITED_RUNTIME_READINESS"
READINESS_FULL = "FULL_RUNTIME_READINESS"

ADMISSION_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "admission_state",
    "observe_count",
    "limited_count",
    "full_count",
    "admission_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[RUNTIME_GOVERNANCE_ADMISSION_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(ADMISSION_MEMORY_COLUMNS))
        for col in ("admission_confidence",):
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


def _scale_admission(classification: str) -> float:
    if classification == CLASS_FULL:
        return 1.0
    if classification == CLASS_LIMITED:
        return 0.55
    if classification == CLASS_OBSERVE:
        return 0.30
    return 0.0


# -----------------------------------------------------------
# Context extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    readiness_summary: Dict[str, Any],
    readiness_record: Dict[str, Any],
    readiness_mem: List[Dict[str, str]],
    shadow_summary: Dict[str, Any],
    eligibility_summary: Dict[str, Any],
    authorization_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    admission_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    rr = readiness_record.get("runtime_readiness") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )

    ctx: Dict[str, Any] = {
        "readiness_state": _norm_upper(
            readiness_summary.get("readiness_state") or readiness_record.get("readiness_state")
        ),
        "readiness_confidence": _clamp(
            _to_float(
                readiness_summary.get("readiness_confidence")
                or readiness_record.get("readiness_confidence")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "readiness_available": bool(readiness_summary.get("runtime_readiness_available")),
        "runtime_readiness": rr,
        "readiness_classification": _norm_upper(
            rr.get("runtime_readiness_classification")
            or readiness_summary.get("runtime_readiness_classification")
        ),
        "readiness_score": _clamp(
            _to_float(
                rr.get("runtime_readiness_score")
                or readiness_summary.get("runtime_readiness_score")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "shadow_confidence": _clamp(
            _to_float(shadow_summary.get("shadow_confidence") or rr.get("shadow_confidence"))
            or 0.0,
            0.0,
            1.0,
        ),
        "eligibility_confidence": _clamp(
            _to_float(eligibility_summary.get("eligibility_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "authorization_confidence": _clamp(
            _to_float(authorization_summary.get("authorization_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "governance_quality": _clamp(
            _to_float(scores.get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "constitutional_safe_readiness": bool(rr.get("constitutional_safe")),
        "constitutional_safety": _clamp(
            1.0 if bool(rr.get("constitutional_safe")) else 0.0,
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
            or readiness_summary.get("operator_review_required")
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
        "readiness_memory_depth": len(readiness_mem),
        "admission_memory_depth": len(admission_mem),
        "observation_cycles": max(
            _to_float(readiness_summary.get("observation_cycles")) or 0,
            len(readiness_mem),
            1,
        ),
    }
    return ctx


# -----------------------------------------------------------
# Runtime admission assessment
# -----------------------------------------------------------
def _base_admission_score(ctx: Dict[str, Any]) -> float:
    raw = (
        ctx["readiness_confidence"] * 0.28
        + ctx["readiness_score"] * 0.22
        + ctx["shadow_confidence"] * 0.16
        + ctx["governance_quality"] * 0.14
        + ctx["system_health_score"] * 0.10
        + ctx["constitutional_safety"] * 0.10
    )
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.22
    return round(_clamp(raw, 0.0, 1.0), 4)


def _compute_admission_score(classification: str, ctx: Dict[str, Any]) -> float:
    raw = _base_admission_score(ctx)
    raw *= _scale_admission(classification)
    return round(_clamp(raw, 0.0, 1.0), 4)


def _classify_admission(*, gate_score: float, ctx: Dict[str, Any]) -> str:
    rc = ctx["readiness_classification"]
    const_safe = (
        ctx["constitutional_safe_readiness"]
        and not ctx["constitution_violated"]
        and ctx["constitutional_safety"] >= 0.50
    )

    if not const_safe or rc == READINESS_NOT or gate_score < 0.18:
        return CLASS_NOT_ADMITTED

    if rc == READINESS_FULL and gate_score >= 0.48:
        if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
            return CLASS_LIMITED
        if ctx["system_health_stale"] or ctx["governance_quality"] < 0.45:
            return CLASS_LIMITED
        return CLASS_FULL

    if rc == READINESS_LIMITED and gate_score >= 0.33:
        if const_safe:
            return CLASS_LIMITED

    if rc == READINESS_OBSERVE and gate_score >= 0.20:
        return CLASS_OBSERVE

    if gate_score >= 0.24 and rc != READINESS_NOT and const_safe:
        return CLASS_OBSERVE

    return CLASS_NOT_ADMITTED


def _admission_rationale(classification: str, admission_score: float, ctx: Dict[str, Any]) -> str:
    templates = {
        CLASS_FULL: (
            "full runtime admission: institutional admission to future runtime governance "
            "consideration granted without runtime activation"
        ),
        CLASS_LIMITED: (
            "limited runtime admission: defensive governance posture supports limited "
            "institutional admission consideration only"
        ),
        CLASS_OBSERVE: (
            "observe runtime admission: observation maturity only; runtime not admitted"
        ),
        CLASS_NOT_ADMITTED: (
            "not runtime admitted: governance maturity remains insufficient for admission"
        ),
    }
    base = templates.get(classification, templates[CLASS_NOT_ADMITTED])
    if ctx["constitution_violated"]:
        base += " under constitutional pressure"
    return f"{base} (admission_score={admission_score:.2f})"


def _build_runtime_admission(
    *,
    ctx: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    gate_score = _base_admission_score(ctx)
    if prior:
        prior_score = _to_float(prior.get("runtime_admission_score")) or gate_score
        gate_score = round(_clamp(gate_score * 0.65 + prior_score * 0.35, 0.0, 1.0), 4)

    classification = _classify_admission(gate_score=gate_score, ctx=ctx)
    admission_score = _compute_admission_score(classification, ctx)

    rr = ctx["runtime_readiness"]
    readiness_class = ctx["readiness_classification"]
    readiness_score = ctx["readiness_score"]

    const_safe = (
        ctx["constitutional_safe_readiness"]
        and not ctx["constitution_violated"]
        and classification != CLASS_NOT_ADMITTED
    )
    institutionally_admitted = classification == CLASS_FULL
    future_candidate = classification in (CLASS_LIMITED, CLASS_FULL) or bool(
        rr.get("future_runtime_candidate")
    )

    conf = round(
        _clamp(
            gate_score * 0.45
            + ctx["readiness_confidence"] * 0.30
            + ctx["shadow_confidence"] * 0.25,
            0.0,
            1.0,
        ),
        4,
    )

    return {
        "runtime_admission_classification": classification,
        "runtime_admission_score": admission_score,
        "runtime_readiness_classification": readiness_class,
        "runtime_readiness_score": round(readiness_score, 4),
        "institutionally_runtime_admitted": institutionally_admitted,
        "future_runtime_candidate": future_candidate,
        "constitutional_safe": const_safe,
        "confidence": conf,
        "runtime_mutation_allowed": False,
        "runtime_admission_rationale": _admission_rationale(classification, admission_score, ctx),
    }


# -----------------------------------------------------------
# Admission confidence and state
# -----------------------------------------------------------
def _admission_confidence(ctx: Dict[str, Any], runtime_admission: Dict[str, Any]) -> float:
    raw = (
        ctx["readiness_confidence"] * 0.28
        + ctx["shadow_confidence"] * 0.20
        + ctx["governance_quality"] * 0.18
        + ctx["system_health_score"] * 0.16
        + ctx["constitutional_safety"] * 0.18
    )
    raw += (_to_float(runtime_admission.get("runtime_admission_score")) or 0.0) * 0.05

    penalty = ctx["constitutional_pressure"] * 0.30
    if ctx["constitution_violated"]:
        penalty += 0.10
    if ctx["court_ruling"] == "COURT_OVERRULED":
        penalty += 0.08
    if ctx["system_health_stale"]:
        penalty += 0.07
    if ctx["governance_quality"] < 0.45:
        penalty += 0.06
    if ctx["observation_cycles"] < 2:
        penalty += 0.06

    return round(_clamp(raw - penalty, 0.0, 1.0), 6)


def _classification_counts(classification: str) -> Dict[str, int]:
    return {
        "not_admitted": 1 if classification == CLASS_NOT_ADMITTED else 0,
        "observe": 1 if classification == CLASS_OBSERVE else 0,
        "limited": 1 if classification == CLASS_LIMITED else 0,
        "full": 1 if classification == CLASS_FULL else 0,
    }


def _classify_admission_state(
    *,
    ctx: Dict[str, Any],
    classification: str,
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if classification == CLASS_NOT_ADMITTED or not ctx["readiness_available"]:
        reasons.append("runtime inadmissible; governance immature")
        return ADMISSION_DORMANT, reasons

    if counts["full"] >= 1 and ctx["admission_memory_depth"] >= 2:
        reasons.append("mature governance process with stable institutional runtime admission")
        return ADMISSION_INSTITUTIONAL, reasons

    if counts["full"] >= 1:
        reasons.append("governance maturity stable; runtime admission plausible")
        return ADMISSION_READY, reasons

    if counts["limited"] >= 1:
        reasons.append("defensive governance maturity; limited admission consideration only")
        return ADMISSION_LIMITED, reasons

    if counts["observe"] >= 1:
        reasons.append("observation maturity only; runtime not admitted")
        return ADMISSION_OBSERVE, reasons

    reasons.append("runtime inadmissible; governance immature")
    return ADMISSION_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _admission_booleans(
    state: str,
    runtime_admission: Dict[str, Any],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    classification = runtime_admission.get("runtime_admission_classification", CLASS_NOT_ADMITTED)
    return {
        "runtime_admission_available": bool(runtime_admission),
        "limited_runtime_admission_available": counts["limited"] > 0 or counts["full"] > 0,
        "full_runtime_admission_available": counts["full"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"]
            or runtime_admission.get("future_runtime_candidate", False)
            or classification in (CLASS_LIMITED, CLASS_FULL)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "admission_memory_reliable": state == ADMISSION_INSTITUTIONAL,
    }


def _recommendations_list(state: str, counts: Dict[str, int]) -> List[str]:
    recs = [
        "Continue governance maturity accumulation",
        "Continue shadow activation rehearsal",
        "Maintain defensive governance posture",
        "Avoid premature runtime assumptions",
        "Maintain runtime mutation lock",
    ]
    if counts["full"] > 0 or counts["limited"] > 0:
        recs.append("Escalate mature runtime admission to operator review")
    if state == ADMISSION_DORMANT:
        recs.append("Accumulate more readiness maturity before admission consideration")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(runtime_admission: Dict[str, Any], state: str, ctx: Dict[str, Any]) -> str:
    classification = runtime_admission.get("runtime_admission_classification", CLASS_NOT_ADMITTED)
    if classification == CLASS_NOT_ADMITTED or ctx["constitution_violated"]:
        return (
            "Triton remains not runtime admitted because governance maturity remains "
            "insufficient under constitutional pressure."
        )
    if classification == CLASS_FULL:
        return (
            "Triton grants full institutional runtime admission for future governance "
            "consideration without activating or mutating live runtime."
        )
    if classification == CLASS_LIMITED:
        return (
            "Triton grants limited runtime admission under defensive governance conditions; "
            "full institutional admission remains constrained."
        )
    if classification == CLASS_OBSERVE:
        return "Runtime admission remains observe-only; governance maturity is immature."
    return (
        "Runtime governance admission board assessment completed without runtime mutation. "
        "Runtime ready != runtime admitted."
    )


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    admission_confidence: float,
    runtime_admission: Dict[str, Any],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
    ctx: Dict[str, Any],
) -> str:
    ra = runtime_admission
    lines = [
        "# Triton Runtime Governance Admission Board",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Runtime Admission State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| admission_confidence | {admission_confidence:.3f} |",
        f"| classification | {ra.get('runtime_admission_classification', CLASS_NOT_ADMITTED)} |",
        f"| admission_score | {_to_float(ra.get('runtime_admission_score')) or 0.0:.3f} |",
        f"| readiness_classification | {ra.get('runtime_readiness_classification', READINESS_NOT)} |",
        f"| readiness_score | {_to_float(ra.get('runtime_readiness_score')) or 0.0:.3f} |",
        f"| institutionally_runtime_admitted | {ra.get('institutionally_runtime_admitted', False)} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Runtime Admission Decision",
        "",
        f"- **Classification:** {ra.get('runtime_admission_classification', CLASS_NOT_ADMITTED)}",
        f"- **Constitutional safe:** {ra.get('constitutional_safe', False)}",
        f"- **Future runtime candidate:** {ra.get('future_runtime_candidate', False)}",
        f"- **Readiness state:** {ctx.get('readiness_state', 'UNKNOWN')}",
        "",
        f"_{ra.get('runtime_admission_rationale', '')}_",
        "",
        "## Runtime Governance Admissibility",
        "",
        f"| tier | count |",
        f"|---|---|",
        f"| observe | {counts['observe']} |",
        f"| limited | {counts['limited']} |",
        f"| full | {counts['full']} |",
        "",
        f"Readiness confidence: {ctx['readiness_confidence']:.3f} | "
        f"Shadow confidence: {ctx['shadow_confidence']:.3f} | "
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
            "This is the institutional runtime admission committee. "
            "Runtime ready != runtime admitted. Runtime admitted != runtime active. "
            "No live runtime policy is changed.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Admission memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    admission_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "admission_state": state,
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "admission_confidence": round(admission_confidence, 6),
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
        for c in ADMISSION_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_runtime_admission_board(
    *,
    readiness_summary: Dict[str, Any],
    readiness_record: Dict[str, Any],
    readiness_mem: List[Dict[str, str]],
    shadow_summary: Dict[str, Any],
    eligibility_summary: Dict[str, Any],
    authorization_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    admission_mem: List[Dict[str, str]],
    prior_admission_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        readiness_summary=readiness_summary,
        readiness_record=readiness_record,
        readiness_mem=readiness_mem,
        shadow_summary=shadow_summary,
        eligibility_summary=eligibility_summary,
        authorization_summary=authorization_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        autonomous_readiness=autonomous_readiness,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        admission_mem=admission_mem,
    )

    prior_ra = prior_admission_record.get("runtime_admission") or {}
    runtime_admission = _build_runtime_admission(ctx=ctx, prior=prior_ra or None)
    admission_confidence = _admission_confidence(ctx, runtime_admission)
    counts = _classification_counts(
        runtime_admission.get("runtime_admission_classification", CLASS_NOT_ADMITTED)
    )

    state, reasons = _classify_admission_state(
        ctx=ctx,
        classification=runtime_admission.get(
            "runtime_admission_classification", CLASS_NOT_ADMITTED
        ),
        counts=counts,
    )

    booleans = _admission_booleans(state, runtime_admission, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(runtime_admission, state, ctx)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        admission_confidence=admission_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(admission_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        admission_confidence=admission_confidence,
        runtime_admission=runtime_admission,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
        ctx=ctx,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_admission_board",
        "engine_version": 1,
        "admission_state": state,
        "admission_confidence": admission_confidence,
        "admission_reasons": reasons,
        "runtime_admission": runtime_admission,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "ready_vs_admitted_note": (
            "Runtime ready != runtime admitted. Runtime admitted != runtime active. "
            "Runtime admission != runtime mutation. Live runtime is never changed."
        ),
        "constitutional_supremacy_note": (
            "Runtime admission cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "admission_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "live_runtime_regime": ctx["regime"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutated": False,
            "runtime_mutation_allowed": False,
            "admission_evaluation_only": True,
        },
        "inputs_seen": {
            "arm_runtime_governance_readiness_gate_summary": bool(readiness_summary),
            "arm_runtime_governance_readiness_gate_record": bool(readiness_record),
            "arm_runtime_governance_readiness_gate_memory_rows": len(readiness_mem),
            "arm_governance_runtime_shadow_activation_summary": bool(shadow_summary),
            "arm_governance_doctrine_activation_eligibility_summary": bool(eligibility_summary),
            "arm_governance_doctrine_activation_authorization_summary": bool(authorization_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(autonomous_readiness),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_admission_memory_rows": len(admission_mem),
            "prior_runtime_admission": bool(prior_ra),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_admission_board",
        "admission_state": state,
        "admission_confidence": admission_confidence,
        "runtime_admission_available": booleans["runtime_admission_available"],
        "limited_runtime_admission_available": booleans["limited_runtime_admission_available"],
        "full_runtime_admission_available": booleans["full_runtime_admission_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "admission_memory_reliable": booleans["admission_memory_reliable"],
        "not_admitted_count": counts["not_admitted"],
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "observation_cycles": ctx["observation_cycles"],
        "runtime_admission_classification": runtime_admission.get(
            "runtime_admission_classification"
        ),
        "runtime_admission_score": runtime_admission.get("runtime_admission_score"),
        "runtime_readiness_classification": runtime_admission.get(
            "runtime_readiness_classification"
        ),
        "institutionally_runtime_admitted": runtime_admission.get(
            "institutionally_runtime_admitted"
        ),
        "n_recommendations": len(recommendations),
        "admission_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM runtime governance admission board (Step 55). "
            "Institutional admission evaluation without mutating runtime. No broker calls."
        ),
    )
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUM))
    p.add_argument("--readiness-record", default=str(DEFAULT_READINESS_REC))
    p.add_argument("--readiness-mem", default=str(DEFAULT_READINESS_MEM))
    p.add_argument("--shadow-summary", default=str(DEFAULT_SHADOW_SUM))
    p.add_argument("--eligibility-summary", default=str(DEFAULT_ELIGIBILITY_SUM))
    p.add_argument("--authorization-summary", default=str(DEFAULT_AUTHORIZATION_SUM))
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
        "[RUNTIME_GOVERNANCE_ADMISSION] starting "
        "(read-only admission evaluation; no runtime mutation; no broker calls)",
        flush=True,
    )

    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="arm_runtime_governance_readiness_gate_summary.json"
    )
    readiness_record = _safe_read_json(
        Path(args.readiness_record), label="arm_runtime_governance_readiness_gate.json"
    )
    readiness_mem = _safe_read_csv_rows(
        Path(args.readiness_mem), label="arm_runtime_governance_readiness_gate_memory.csv"
    )
    shadow_summary = _safe_read_json(
        Path(args.shadow_summary), label="arm_governance_runtime_shadow_activation_summary.json"
    )
    eligibility_summary = _safe_read_json(
        Path(args.eligibility_summary),
        label="arm_governance_doctrine_activation_eligibility_summary.json",
    )
    authorization_summary = _safe_read_json(
        Path(args.authorization_summary),
        label="arm_governance_doctrine_activation_authorization_summary.json",
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
    admission_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_runtime_governance_admission_board_memory.csv"
    )
    prior_admission_record = _safe_read_json(
        Path(args.out_json), label="prior_arm_runtime_governance_admission_board.json"
    )

    record, summary, md, merged_memory = build_runtime_admission_board(
        readiness_summary=readiness_summary,
        readiness_record=readiness_record,
        readiness_mem=readiness_mem,
        shadow_summary=shadow_summary,
        eligibility_summary=eligibility_summary,
        authorization_summary=authorization_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        autonomous_readiness=autonomous_readiness,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        admission_mem=admission_mem,
        prior_admission_record=prior_admission_record,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=ADMISSION_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["classification_counts"]
    print(
        "[RUNTIME_GOVERNANCE_ADMISSION] "
        f"state={record['admission_state']} "
        f"limited={counts['limited']} "
        f"full={counts['full']} "
        f"confidence={record['admission_confidence']:.3f}",
        flush=True,
    )
    print(
        "[RUNTIME_GOVERNANCE_ADMISSION_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[RUNTIME_GOVERNANCE_ADMISSION_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[RUNTIME_GOVERNANCE_ADMISSION_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
