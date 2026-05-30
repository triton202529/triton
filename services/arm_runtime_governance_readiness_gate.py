"""
ARM Runtime Governance Readiness Gate -- Step 54.

Reads:
    data/results/arm_governance_runtime_shadow_activation_summary.json       (Step 53)
    data/results/arm_governance_runtime_shadow_activation.json               (Step 53)
    data/results/arm_governance_runtime_shadow_activation_memory.csv         (Step 53)
    data/results/arm_governance_doctrine_activation_eligibility_summary.json (Step 52)
    data/results/arm_governance_doctrine_activation_authorization_summary.json (Step 51)
    data/results/arm_governance_doctrine_approval_board_summary.json         (Step 50)
    data/results/arm_governance_doctrine_readiness_summary.json              (Step 48)
    data/results/arm_governance_doctrine_institutional_trust_summary.json    (Step 47)
    data/results/autonomous_governance_scorecard.json                        (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/arm_constitutional_court_summary.json                       (Step 33)
    data/results/arm_supreme_governance_council_summary.json                 (Step 34)
    data/results/runtime_policy_governed.json                                (Step 18)

Writes:
    data/results/arm_runtime_governance_readiness_gate.json
    data/results/arm_runtime_governance_readiness_gate.md
    data/results/arm_runtime_governance_readiness_gate_summary.json
    data/results/arm_runtime_governance_readiness_gate_memory.csv
    data/results/arm_runtime_governance_readiness_gate_memory.parquet

Purpose
-------
This engine answers:

    "Is Triton institutionally ready for runtime governance?"

It determines whether governance maturity is sufficient to consider a future
runtime governance layer. This is the FIRST institutional runtime admissibility gate.
Shadow activated != runtime ready. Runtime ready != runtime active.
Runtime readiness != runtime mutation. Runtime readiness NEVER mutates runtime policy.

Readiness state cascade
-----------------------
    1. RUNTIME_READINESS_INSTITUTIONAL  long-run governance stability
    2. RUNTIME_READINESS_READY            governance maturity sufficient
    3. RUNTIME_READINESS_LIMITED          limited maturity; defensive admissibility only
    4. RUNTIME_READINESS_OBSERVE          observation-only maturity
    5. RUNTIME_READINESS_DORMANT          governance immature; doctrine not shadow mature

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
* Append-only readiness memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to RUNTIME_READINESS_DORMANT.
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

DEFAULT_SHADOW_SUM = RESULTS_DIR / "arm_governance_runtime_shadow_activation_summary.json"
DEFAULT_SHADOW_REC = RESULTS_DIR / "arm_governance_runtime_shadow_activation.json"
DEFAULT_SHADOW_MEM = RESULTS_DIR / "arm_governance_runtime_shadow_activation_memory.csv"
DEFAULT_ELIGIBILITY_SUM = (
    RESULTS_DIR / "arm_governance_doctrine_activation_eligibility_summary.json"
)
DEFAULT_AUTHORIZATION_SUM = (
    RESULTS_DIR / "arm_governance_doctrine_activation_authorization_summary.json"
)
DEFAULT_APPROVAL_SUM = RESULTS_DIR / "arm_governance_doctrine_approval_board_summary.json"
DEFAULT_DOCTRINE_READINESS_SUM = RESULTS_DIR / "arm_governance_doctrine_readiness_summary.json"
DEFAULT_TRUST_SUM = RESULTS_DIR / "arm_governance_doctrine_institutional_trust_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_SUM = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_runtime_governance_readiness_gate.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_runtime_governance_readiness_gate.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_runtime_governance_readiness_gate_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_runtime_governance_readiness_gate_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_runtime_governance_readiness_gate_memory.parquet"


# -----------------------------------------------------------
# Readiness state constants
# -----------------------------------------------------------
READINESS_DORMANT = "RUNTIME_READINESS_DORMANT"
READINESS_OBSERVE = "RUNTIME_READINESS_OBSERVE"
READINESS_LIMITED = "RUNTIME_READINESS_LIMITED"
READINESS_READY = "RUNTIME_READINESS_READY"
READINESS_INSTITUTIONAL = "RUNTIME_READINESS_INSTITUTIONAL"

CLASS_NOT_READY = "NOT_RUNTIME_READY"
CLASS_OBSERVE = "OBSERVE_RUNTIME_READINESS"
CLASS_LIMITED = "LIMITED_RUNTIME_READINESS"
CLASS_FULL = "FULL_RUNTIME_READINESS"

SHADOW_CLASS_FULL = "FULL_SHADOW_ACTIVATION"
SHADOW_CLASS_LIMITED = "LIMITED_SHADOW_ACTIVATION"
SHADOW_CLASS_OBSERVE = "OBSERVE_ONLY_SHADOW"

READINESS_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "readiness_state",
    "observe_count",
    "limited_count",
    "full_count",
    "readiness_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[RUNTIME_GOVERNANCE_READINESS_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(READINESS_MEMORY_COLUMNS))
        for col in ("readiness_confidence",):
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


def _scale_readiness(classification: str) -> float:
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
def _shadow_counts(shadow_record: Dict[str, Any]) -> Dict[str, int]:
    rows = shadow_record.get("doctrine_shadow") or []
    return {
        "no_shadow": sum(
            1 for r in rows if r.get("shadow_classification") == "NO_SHADOW_ACTIVATION"
        ),
        "observe": sum(1 for r in rows if r.get("shadow_classification") == SHADOW_CLASS_OBSERVE),
        "limited": sum(1 for r in rows if r.get("shadow_classification") == SHADOW_CLASS_LIMITED),
        "full": sum(1 for r in rows if r.get("shadow_classification") == SHADOW_CLASS_FULL),
        "activated": sum(1 for r in rows if r.get("shadow_activated")),
    }


def _constitutional_safety_from_shadow(shadow_record: Dict[str, Any]) -> float:
    rows = shadow_record.get("doctrine_shadow") or []
    if not rows:
        return 0.0
    vals = [1.0 if bool(r.get("constitutional_safe")) else 0.0 for r in rows]
    return sum(vals) / len(vals)


def _extract_context(
    *,
    shadow_summary: Dict[str, Any],
    shadow_record: Dict[str, Any],
    shadow_mem: List[Dict[str, str]],
    eligibility_summary: Dict[str, Any],
    authorization_summary: Dict[str, Any],
    approval_summary: Dict[str, Any],
    doctrine_readiness_summary: Dict[str, Any],
    trust_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    readiness_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    shadow_counts = _shadow_counts(shadow_record)

    ctx: Dict[str, Any] = {
        "shadow_state": _norm_upper(
            shadow_summary.get("shadow_state") or shadow_record.get("shadow_state")
        ),
        "shadow_confidence": _clamp(
            _to_float(
                shadow_summary.get("shadow_confidence") or shadow_record.get("shadow_confidence")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "shadow_available": bool(shadow_summary.get("shadow_activation_available")),
        "shadow_counts": shadow_counts,
        "shadow_memory_depth": len(shadow_mem),
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
        "approval_confidence": _clamp(
            _to_float(approval_summary.get("approval_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "doctrine_readiness_confidence": _clamp(
            _to_float(doctrine_readiness_summary.get("readiness_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "trust_confidence": _clamp(
            _to_float(trust_summary.get("trust_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "governance_quality": _clamp(
            _to_float(scores.get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "autonomous_readiness_score": _clamp(
            _to_float(readiness_summary.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "constitutional_safety": _constitutional_safety_from_shadow(shadow_record),
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
            or shadow_summary.get("operator_review_required")
            or eligibility_summary.get("operator_review_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "observation_cycles": max(
            _to_float(shadow_summary.get("observation_cycles")) or 0,
            _to_float(eligibility_summary.get("observation_cycles")) or 0,
            len(shadow_mem),
            1,
        ),
        "readiness_memory_depth": len(readiness_mem),
    }
    return ctx


# -----------------------------------------------------------
# Runtime readiness assessment
# -----------------------------------------------------------
def _base_readiness_score(ctx: Dict[str, Any]) -> float:
    sc = ctx["shadow_counts"]
    shadow_maturity = 0.0
    if sc["full"] > 0:
        shadow_maturity = 0.85
    elif sc["limited"] > 0:
        shadow_maturity = 0.60
    elif sc["observe"] > 0:
        shadow_maturity = 0.35
    elif ctx["shadow_state"] in (
        "SHADOW_ACTIVATION_READY",
        "SHADOW_ACTIVATION_INSTITUTIONAL",
    ):
        shadow_maturity = 0.75
    elif ctx["shadow_state"] == "SHADOW_ACTIVATION_LIMITED":
        shadow_maturity = 0.55
    elif ctx["shadow_state"] == "SHADOW_ACTIVATION_OBSERVE":
        shadow_maturity = 0.30

    raw = (
        ctx["shadow_confidence"] * 0.22
        + ctx["eligibility_confidence"] * 0.14
        + ctx["authorization_confidence"] * 0.12
        + ctx["approval_confidence"] * 0.10
        + ctx["doctrine_readiness_confidence"] * 0.10
        + ctx["trust_confidence"] * 0.08
        + ctx["governance_quality"] * 0.10
        + ctx["system_health_score"] * 0.08
        + ctx["constitutional_safety"] * 0.06
    )
    raw += shadow_maturity * 0.10
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.20
    return round(_clamp(raw, 0.0, 1.0), 4)


def _compute_readiness_score(classification: str, ctx: Dict[str, Any]) -> float:
    raw = _base_readiness_score(ctx)
    raw *= _scale_readiness(classification)
    return round(_clamp(raw, 0.0, 1.0), 4)


def _classify_runtime_readiness(*, gate_score: float, ctx: Dict[str, Any]) -> str:
    sc = ctx["shadow_counts"]
    const_safe = ctx["constitutional_safety"] >= 0.50 and not ctx["constitution_violated"]

    if not const_safe or gate_score < 0.20:
        return CLASS_NOT_READY

    if (
        sc["full"] >= 1
        or ctx["shadow_state"] in ("SHADOW_ACTIVATION_READY", "SHADOW_ACTIVATION_INSTITUTIONAL")
    ) and gate_score >= 0.50:
        if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
            return CLASS_LIMITED
        if ctx["system_health_stale"] or ctx["governance_quality"] < 0.45:
            return CLASS_LIMITED
        return CLASS_FULL

    if sc["limited"] >= 1 or ctx["shadow_state"] == "SHADOW_ACTIVATION_LIMITED":
        if gate_score >= 0.35 and const_safe:
            return CLASS_LIMITED

    if sc["observe"] >= 1 or ctx["shadow_state"] == "SHADOW_ACTIVATION_OBSERVE":
        if gate_score >= 0.22:
            return CLASS_OBSERVE

    if gate_score >= 0.28 and const_safe:
        return CLASS_OBSERVE

    return CLASS_NOT_READY


def _readiness_rationale(classification: str, gate_score: float, ctx: Dict[str, Any]) -> str:
    templates = {
        CLASS_FULL: (
            "full runtime readiness: governance maturity sufficient for institutional "
            "runtime admissibility consideration"
        ),
        CLASS_LIMITED: (
            "limited runtime readiness: defensive governance posture supports limited "
            "runtime admissibility only"
        ),
        CLASS_OBSERVE: (
            "observe runtime readiness: observation maturity only; runtime inadmissible"
        ),
        CLASS_NOT_READY: (
            "not runtime ready: governance maturity and shadow activation stability "
            "remain insufficient"
        ),
    }
    base = templates.get(classification, templates[CLASS_NOT_READY])
    if ctx["constitution_violated"]:
        base += " under constitutional pressure"
    return f"{base} (readiness_score={gate_score:.2f})"


def _build_runtime_readiness(
    *,
    ctx: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    gate_score = _base_readiness_score(ctx)
    if prior:
        prior_score = _to_float(prior.get("runtime_readiness_score")) or gate_score
        gate_score = round(_clamp(gate_score * 0.65 + prior_score * 0.35, 0.0, 1.0), 4)

    classification = _classify_runtime_readiness(gate_score=gate_score, ctx=ctx)
    readiness_score = _compute_readiness_score(classification, ctx)

    const_safe = (
        ctx["constitutional_safety"] >= 0.50
        and not ctx["constitution_violated"]
        and classification != CLASS_NOT_READY
    )
    institutional_ready = classification == CLASS_FULL
    future_candidate = (
        classification in (CLASS_LIMITED, CLASS_FULL) or ctx["shadow_counts"]["activated"] > 0
    )

    conf = round(
        _clamp(
            gate_score * 0.50
            + ctx["shadow_confidence"] * 0.25
            + ctx["constitutional_safety"] * 0.25,
            0.0,
            1.0,
        ),
        4,
    )

    return {
        "runtime_readiness_classification": classification,
        "runtime_readiness_score": readiness_score,
        "shadow_state": ctx["shadow_state"],
        "shadow_confidence": ctx["shadow_confidence"],
        "institutional_governance_ready": institutional_ready,
        "future_runtime_candidate": future_candidate,
        "constitutional_safe": const_safe,
        "confidence": conf,
        "runtime_mutation_allowed": False,
        "runtime_readiness_rationale": _readiness_rationale(classification, readiness_score, ctx),
    }


# -----------------------------------------------------------
# Readiness confidence and state
# -----------------------------------------------------------
def _readiness_confidence(ctx: Dict[str, Any], runtime_readiness: Dict[str, Any]) -> float:
    raw = (
        ctx["shadow_confidence"] * 0.18
        + ctx["eligibility_confidence"] * 0.14
        + ctx["authorization_confidence"] * 0.12
        + ctx["approval_confidence"] * 0.10
        + ctx["doctrine_readiness_confidence"] * 0.10
        + ctx["governance_quality"] * 0.12
        + ctx["system_health_score"] * 0.10
        + ctx["constitutional_safety"] * 0.14
    )
    raw += (_to_float(runtime_readiness.get("runtime_readiness_score")) or 0.0) * 0.05

    penalty = ctx["constitutional_pressure"] * 0.28
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
        "not_ready": 1 if classification == CLASS_NOT_READY else 0,
        "observe": 1 if classification == CLASS_OBSERVE else 0,
        "limited": 1 if classification == CLASS_LIMITED else 0,
        "full": 1 if classification == CLASS_FULL else 0,
    }


def _classify_readiness_state(
    *,
    ctx: Dict[str, Any],
    classification: str,
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if classification == CLASS_NOT_READY or not ctx["shadow_available"]:
        reasons.append("governance immature; doctrine not shadow mature")
        return READINESS_DORMANT, reasons

    if counts["full"] >= 1 and ctx["readiness_memory_depth"] >= 2:
        reasons.append("long-run governance stability supports institutional runtime admissibility")
        return READINESS_INSTITUTIONAL, reasons

    if counts["full"] >= 1:
        reasons.append("governance maturity sufficient; runtime admissibility plausible")
        return READINESS_READY, reasons

    if counts["limited"] >= 1:
        reasons.append("limited maturity; defensive runtime admissibility only")
        return READINESS_LIMITED, reasons

    if counts["observe"] >= 1:
        reasons.append("observation-only maturity; runtime inadmissible")
        return READINESS_OBSERVE, reasons

    reasons.append("governance immature; doctrine not shadow mature")
    return READINESS_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _readiness_booleans(
    state: str,
    runtime_readiness: Dict[str, Any],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    classification = runtime_readiness.get("runtime_readiness_classification", CLASS_NOT_READY)
    return {
        "runtime_readiness_available": bool(runtime_readiness),
        "limited_runtime_readiness_available": counts["limited"] > 0 or counts["full"] > 0,
        "full_runtime_readiness_available": counts["full"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"]
            or runtime_readiness.get("future_runtime_candidate", False)
            or classification in (CLASS_LIMITED, CLASS_FULL)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "readiness_memory_reliable": state == READINESS_INSTITUTIONAL,
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
        recs.append("Escalate mature runtime readiness to operator review")
    if state == READINESS_DORMANT:
        recs.append("Accumulate more shadow maturity before runtime admissibility review")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(runtime_readiness: Dict[str, Any], state: str, ctx: Dict[str, Any]) -> str:
    classification = runtime_readiness.get("runtime_readiness_classification", CLASS_NOT_READY)
    if classification == CLASS_NOT_READY or ctx["constitution_violated"]:
        return (
            "Triton remains not runtime ready because governance maturity and shadow "
            "activation stability remain insufficient under constitutional pressure."
        )
    if classification == CLASS_FULL:
        return (
            "Triton demonstrates sufficient governance maturity for full runtime readiness "
            "consideration without mutating live runtime."
        )
    if classification == CLASS_LIMITED:
        return (
            "Triton shows limited runtime readiness under defensive governance conditions; "
            "runtime admissibility remains constrained."
        )
    if classification == CLASS_OBSERVE:
        return "Runtime readiness remains observe-only; governance maturity is immature."
    return (
        "Runtime governance readiness gate completed without runtime mutation. "
        "Shadow activated != runtime ready."
    )


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    readiness_confidence: float,
    runtime_readiness: Dict[str, Any],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
    ctx: Dict[str, Any],
) -> str:
    rr = runtime_readiness
    lines = [
        "# Triton Runtime Governance Readiness Gate",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Runtime Readiness State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| readiness_confidence | {readiness_confidence:.3f} |",
        f"| classification | {rr.get('runtime_readiness_classification', CLASS_NOT_READY)} |",
        f"| readiness_score | {_to_float(rr.get('runtime_readiness_score')) or 0.0:.3f} |",
        f"| shadow_state | {rr.get('shadow_state', 'UNKNOWN')} |",
        f"| shadow_confidence | {_to_float(rr.get('shadow_confidence')) or 0.0:.3f} |",
        f"| institutional_governance_ready | {rr.get('institutional_governance_ready', False)} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Runtime Admissibility",
        "",
        f"- **Classification:** {rr.get('runtime_readiness_classification', CLASS_NOT_READY)}",
        f"- **Constitutional safe:** {rr.get('constitutional_safe', False)}",
        f"- **Future runtime candidate:** {rr.get('future_runtime_candidate', False)}",
        f"- **Regime:** {ctx.get('regime', 'UNKNOWN')}",
        "",
        "## Runtime Governance Maturity",
        "",
        f"| tier | count |",
        f"|---|---|",
        f"| observe | {counts['observe']} |",
        f"| limited | {counts['limited']} |",
        f"| full | {counts['full']} |",
        "",
        f"Shadow doctrine maturity: limited={ctx['shadow_counts']['limited']} "
        f"full={ctx['shadow_counts']['full']} activated={ctx['shadow_counts']['activated']}",
        "",
        f"_{rr.get('runtime_readiness_rationale', '')}_",
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
            "This is the first institutional runtime admissibility gate. "
            "Shadow activated != runtime ready. Runtime ready != runtime active. "
            "No live runtime policy is changed.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Readiness memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    readiness_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "readiness_state": state,
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "readiness_confidence": round(readiness_confidence, 6),
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
        for c in READINESS_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_runtime_readiness_gate(
    *,
    shadow_summary: Dict[str, Any],
    shadow_record: Dict[str, Any],
    shadow_mem: List[Dict[str, str]],
    eligibility_summary: Dict[str, Any],
    authorization_summary: Dict[str, Any],
    approval_summary: Dict[str, Any],
    doctrine_readiness_summary: Dict[str, Any],
    trust_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    readiness_mem: List[Dict[str, str]],
    prior_readiness_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        shadow_summary=shadow_summary,
        shadow_record=shadow_record,
        shadow_mem=shadow_mem,
        eligibility_summary=eligibility_summary,
        authorization_summary=authorization_summary,
        approval_summary=approval_summary,
        doctrine_readiness_summary=doctrine_readiness_summary,
        trust_summary=trust_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        readiness_mem=readiness_mem,
    )

    prior_rr = prior_readiness_record.get("runtime_readiness") or {}
    runtime_readiness = _build_runtime_readiness(ctx=ctx, prior=prior_rr or None)
    readiness_confidence = _readiness_confidence(ctx, runtime_readiness)
    counts = _classification_counts(
        runtime_readiness.get("runtime_readiness_classification", CLASS_NOT_READY)
    )

    state, reasons = _classify_readiness_state(
        ctx=ctx,
        classification=runtime_readiness.get("runtime_readiness_classification", CLASS_NOT_READY),
        counts=counts,
    )

    booleans = _readiness_booleans(state, runtime_readiness, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(runtime_readiness, state, ctx)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        readiness_confidence=readiness_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(readiness_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        readiness_confidence=readiness_confidence,
        runtime_readiness=runtime_readiness,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
        ctx=ctx,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_readiness_gate",
        "engine_version": 1,
        "readiness_state": state,
        "readiness_confidence": readiness_confidence,
        "readiness_reasons": reasons,
        "runtime_readiness": runtime_readiness,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "shadow_vs_runtime_note": (
            "Shadow activated != runtime ready. Runtime ready != runtime active. "
            "Runtime readiness != runtime mutation. Live runtime is never changed."
        ),
        "constitutional_supremacy_note": (
            "Runtime readiness cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "readiness_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "live_runtime_regime": ctx["regime"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutated": False,
            "runtime_mutation_allowed": False,
            "readiness_evaluation_only": True,
        },
        "inputs_seen": {
            "arm_governance_runtime_shadow_activation_summary": bool(shadow_summary),
            "arm_governance_runtime_shadow_activation_record": bool(shadow_record),
            "arm_governance_runtime_shadow_activation_memory_rows": len(shadow_mem),
            "arm_governance_doctrine_activation_eligibility_summary": bool(eligibility_summary),
            "arm_governance_doctrine_activation_authorization_summary": bool(authorization_summary),
            "arm_governance_doctrine_approval_board_summary": bool(approval_summary),
            "arm_governance_doctrine_readiness_summary": bool(doctrine_readiness_summary),
            "arm_governance_doctrine_institutional_trust_summary": bool(trust_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_readiness_memory_rows": len(readiness_mem),
            "prior_runtime_readiness": bool(prior_rr),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_runtime_governance_readiness_gate",
        "readiness_state": state,
        "readiness_confidence": readiness_confidence,
        "runtime_readiness_available": booleans["runtime_readiness_available"],
        "limited_runtime_readiness_available": booleans["limited_runtime_readiness_available"],
        "full_runtime_readiness_available": booleans["full_runtime_readiness_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "readiness_memory_reliable": booleans["readiness_memory_reliable"],
        "not_ready_count": counts["not_ready"],
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "observation_cycles": ctx["observation_cycles"],
        "runtime_readiness_classification": runtime_readiness.get(
            "runtime_readiness_classification"
        ),
        "runtime_readiness_score": runtime_readiness.get("runtime_readiness_score"),
        "shadow_state": runtime_readiness.get("shadow_state"),
        "shadow_confidence": runtime_readiness.get("shadow_confidence"),
        "n_recommendations": len(recommendations),
        "readiness_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM runtime governance readiness gate (Step 54). "
            "Evaluates institutional runtime readiness without mutating runtime. No broker calls."
        ),
    )
    p.add_argument("--shadow-summary", default=str(DEFAULT_SHADOW_SUM))
    p.add_argument("--shadow-record", default=str(DEFAULT_SHADOW_REC))
    p.add_argument("--shadow-mem", default=str(DEFAULT_SHADOW_MEM))
    p.add_argument("--eligibility-summary", default=str(DEFAULT_ELIGIBILITY_SUM))
    p.add_argument("--authorization-summary", default=str(DEFAULT_AUTHORIZATION_SUM))
    p.add_argument("--approval-summary", default=str(DEFAULT_APPROVAL_SUM))
    p.add_argument("--doctrine-readiness-summary", default=str(DEFAULT_DOCTRINE_READINESS_SUM))
    p.add_argument("--trust-summary", default=str(DEFAULT_TRUST_SUM))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUM))
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
        "[RUNTIME_GOVERNANCE_READINESS] starting "
        "(read-only readiness evaluation; no runtime mutation; no broker calls)",
        flush=True,
    )

    shadow_summary = _safe_read_json(
        Path(args.shadow_summary), label="arm_governance_runtime_shadow_activation_summary.json"
    )
    shadow_record = _safe_read_json(
        Path(args.shadow_record), label="arm_governance_runtime_shadow_activation.json"
    )
    shadow_mem = _safe_read_csv_rows(
        Path(args.shadow_mem), label="arm_governance_runtime_shadow_activation_memory.csv"
    )
    eligibility_summary = _safe_read_json(
        Path(args.eligibility_summary),
        label="arm_governance_doctrine_activation_eligibility_summary.json",
    )
    authorization_summary = _safe_read_json(
        Path(args.authorization_summary),
        label="arm_governance_doctrine_activation_authorization_summary.json",
    )
    approval_summary = _safe_read_json(
        Path(args.approval_summary), label="arm_governance_doctrine_approval_board_summary.json"
    )
    doctrine_readiness_summary = _safe_read_json(
        Path(args.doctrine_readiness_summary),
        label="arm_governance_doctrine_readiness_summary.json",
    )
    trust_summary = _safe_read_json(
        Path(args.trust_summary), label="arm_governance_doctrine_institutional_trust_summary.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="autonomous_readiness_summary.json"
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
    readiness_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_runtime_governance_readiness_gate_memory.csv"
    )
    prior_readiness_record = _safe_read_json(
        Path(args.out_json), label="prior_arm_runtime_governance_readiness_gate.json"
    )

    record, summary, md, merged_memory = build_runtime_readiness_gate(
        shadow_summary=shadow_summary,
        shadow_record=shadow_record,
        shadow_mem=shadow_mem,
        eligibility_summary=eligibility_summary,
        authorization_summary=authorization_summary,
        approval_summary=approval_summary,
        doctrine_readiness_summary=doctrine_readiness_summary,
        trust_summary=trust_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        runtime_policy=runtime_policy,
        readiness_mem=readiness_mem,
        prior_readiness_record=prior_readiness_record,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=READINESS_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["classification_counts"]
    print(
        "[RUNTIME_GOVERNANCE_READINESS] "
        f"state={record['readiness_state']} "
        f"limited={counts['limited']} "
        f"full={counts['full']} "
        f"confidence={record['readiness_confidence']:.3f}",
        flush=True,
    )
    print(
        "[RUNTIME_GOVERNANCE_READINESS_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[RUNTIME_GOVERNANCE_READINESS_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[RUNTIME_GOVERNANCE_READINESS_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
