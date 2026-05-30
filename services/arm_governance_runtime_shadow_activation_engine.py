"""
ARM Governance Runtime Shadow Activation Engine -- Step 53.

Reads:
    data/results/arm_governance_doctrine_activation_eligibility_summary.json  (Step 52)
    data/results/arm_governance_doctrine_activation_eligibility.json          (Step 52)
    data/results/arm_governance_doctrine_activation_eligibility_memory.csv  (Step 52)
    data/results/runtime_policy_governed.json                                 (Step 18)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/autonomous_governance_scorecard.json                       (Step 19)
    data/results/arm_constitutional_court_summary.json                      (Step 33)
    data/results/arm_supreme_governance_council_summary.json               (Step 34)
    data/results/arm_governance_doctrine_simulation_summary.json            (Step 43)
    data/results/arm_governance_doctrine_impact_assessment_summary.json     (Step 44)

Writes:
    data/results/arm_governance_runtime_shadow_activation.json
    data/results/arm_governance_runtime_shadow_activation.md
    data/results/arm_governance_runtime_shadow_activation_summary.json
    data/results/arm_governance_runtime_shadow_activation_memory.csv
    data/results/arm_governance_runtime_shadow_activation_memory.parquet

Purpose
-------
This engine answers:

    "What would happen if eligible doctrine behaved as active without changing runtime?"

It shadow-activates eligible governance doctrine without mutating runtime behavior.
This is the FINAL rehearsal layer before future ARM runtime.
Eligible != shadow activated. Shadow activated != runtime activated.
Shadow activation != runtime mutation. Shadow activation NEVER mutates runtime policy.

Shadow state cascade
--------------------
    1. SHADOW_ACTIVATION_INSTITUTIONAL  stable long-term governance shadow process
    2. SHADOW_ACTIVATION_READY            full eligibility; mature shadow activation
    3. SHADOW_ACTIVATION_LIMITED          limited eligibility; defensive shadow only
    4. SHADOW_ACTIVATION_OBSERVE          observation-only eligibility; passive shadowing
    5. SHADOW_ACTIVATION_DORMANT          doctrine not eligible; insufficient maturity

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens are never written literally; an import-time
self-check raises if they ever appear.

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* runtime_mutation_allowed is ALWAYS false.
* shadow_activated != runtime_activated.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only shadow memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to SHADOW_ACTIVATION_DORMANT.
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
    RESULTS_DIR / "arm_governance_doctrine_activation_eligibility_summary.json"
)
DEFAULT_ELIGIBILITY_REC = RESULTS_DIR / "arm_governance_doctrine_activation_eligibility.json"
DEFAULT_ELIGIBILITY_MEM = RESULTS_DIR / "arm_governance_doctrine_activation_eligibility_memory.csv"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_SUM = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_SIMULATION_SUM = RESULTS_DIR / "arm_governance_doctrine_simulation_summary.json"
DEFAULT_IMPACT_SUM = RESULTS_DIR / "arm_governance_doctrine_impact_assessment_summary.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_runtime_shadow_activation.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_runtime_shadow_activation.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_runtime_shadow_activation_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_runtime_shadow_activation_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_runtime_shadow_activation_memory.parquet"


# -----------------------------------------------------------
# Shadow state constants
# -----------------------------------------------------------
SHADOW_DORMANT = "SHADOW_ACTIVATION_DORMANT"
SHADOW_OBSERVE = "SHADOW_ACTIVATION_OBSERVE"
SHADOW_LIMITED = "SHADOW_ACTIVATION_LIMITED"
SHADOW_READY = "SHADOW_ACTIVATION_READY"
SHADOW_INSTITUTIONAL = "SHADOW_ACTIVATION_INSTITUTIONAL"

CLASS_NO_SHADOW = "NO_SHADOW_ACTIVATION"
CLASS_OBSERVE = "OBSERVE_ONLY_SHADOW"
CLASS_LIMITED = "LIMITED_SHADOW_ACTIVATION"
CLASS_FULL = "FULL_SHADOW_ACTIVATION"

ELIG_NOT = "NOT_ELIGIBLE"
ELIG_OBSERVE = "OBSERVE_ONLY_ELIGIBILITY"
ELIG_LIMITED = "LIMITED_ELIGIBILITY"
ELIG_FULL = "FULL_ELIGIBILITY"

SHADOW_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "shadow_state",
    "observe_count",
    "limited_count",
    "full_count",
    "shadow_confidence",
    "rationale",
)

_DELTA_KEYS: Tuple[str, ...] = (
    "confidence_threshold_delta",
    "deployment_threshold_delta",
    "target_cash_pct_delta",
    "max_position_pct_delta",
    "skepticism_threshold_delta",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_RUNTIME_SHADOW_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(SHADOW_MEMORY_COLUMNS))
        for col in ("shadow_confidence",):
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


def _zero_deltas() -> Dict[str, float]:
    return {k: 0.0 for k in _DELTA_KEYS}


# -----------------------------------------------------------
# Context extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    eligibility_summary: Dict[str, Any],
    eligibility_record: Dict[str, Any],
    eligibility_mem: List[Dict[str, str]],
    runtime_policy: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    simulation_summary: Dict[str, Any],
    impact_summary: Dict[str, Any],
    shadow_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    ctx: Dict[str, Any] = {
        "eligibility_state": _norm_upper(
            eligibility_summary.get("eligibility_state")
            or eligibility_record.get("eligibility_state")
        ),
        "eligibility_confidence": _clamp(
            _to_float(
                eligibility_summary.get("eligibility_confidence")
                or eligibility_record.get("eligibility_confidence")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "doctrine_eligibility": eligibility_record.get("doctrine_eligibility") or [],
        "eligibility_available": bool(eligibility_summary.get("doctrine_eligibility_available")),
        "eligibility_memory_depth": len(eligibility_mem),
        "shadow_memory_depth": len(shadow_mem),
        "simulation_confidence": _clamp(
            _to_float(simulation_summary.get("simulation_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "impact_confidence": _clamp(
            _to_float(impact_summary.get("impact_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "readiness_confidence": _clamp(
            _to_float(readiness_summary.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
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
            or eligibility_summary.get("operator_review_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "runtime_policy": runtime_policy,
        "observation_cycles": max(
            _to_float(eligibility_summary.get("observation_cycles")) or 0,
            len(eligibility_mem),
            1,
        ),
    }
    return ctx


def _prior_shadow_map(prior_record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in prior_record.get("doctrine_shadow") or []:
        name = str(row.get("policy_name", ""))
        if name:
            out[name] = row
    return out


# -----------------------------------------------------------
# Shadow simulation (read-only deltas; never applied)
# -----------------------------------------------------------
def _scale(shadow_class: str) -> float:
    if shadow_class == CLASS_FULL:
        return 1.0
    if shadow_class == CLASS_LIMITED:
        return 0.55
    return 0.0


def _simulated_runtime_changes(
    policy_name: str,
    shadow_class: str,
    *,
    ctx: Dict[str, Any],
) -> Dict[str, float]:
    deltas = _zero_deltas()
    if shadow_class not in (CLASS_LIMITED, CLASS_FULL):
        return deltas

    scale = _scale(shadow_class)
    regime = ctx["regime"]
    defensive = regime in ("DEFENSIVE", "RESTRICTED", "HOLD") or ctx["constitution_violated"]

    policy_deltas: Dict[str, Dict[str, float]] = {
        "target_cash_pct": {
            "target_cash_pct_delta": (4.0 if defensive else 2.0) * scale,
        },
        "max_position_pct": {
            "max_position_pct_delta": (-0.8 if defensive else -0.4) * scale,
        },
        "confidence_threshold": {
            "confidence_threshold_delta": (0.03 if defensive else 0.015) * scale,
        },
        "deployment_threshold": {
            "deployment_threshold_delta": (0.04 if defensive else 0.02) * scale,
        },
        "persistence_threshold": {
            "confidence_threshold_delta": (0.01 if defensive else 0.005) * scale,
        },
        "skepticism_threshold": {
            "skepticism_threshold_delta": (0.025 if defensive else 0.012) * scale,
        },
        "governance_monitoring_frequency_multiplier": {
            "confidence_threshold_delta": (0.005 if defensive else 0.002) * scale,
        },
    }

    mapped = policy_deltas.get(policy_name, {})
    for k, v in mapped.items():
        deltas[k] = round(v, 4)
    return deltas


def _shadow_runtime_behavior(
    policy_name: str,
    shadow_class: str,
    deltas: Dict[str, float],
    *,
    ctx: Dict[str, Any],
) -> str:
    if shadow_class == CLASS_NO_SHADOW:
        return "no shadow runtime behavior; doctrine not eligible"
    if shadow_class == CLASS_OBSERVE:
        return "passive observation only; no simulated runtime posture shift"
    active = [k for k, v in deltas.items() if abs(v) > 1e-9]
    if not active:
        return f"shadow rehearsal for {policy_name} with neutral simulated posture"
    parts = [f"{k}={v:+.4f}" for k, v in deltas.items() if abs(v) > 1e-9]
    mode = "defensive" if ctx["regime"] in ("DEFENSIVE", "RESTRICTED") else "neutral"
    return (
        f"shadow {mode} rehearsal: live runtime unchanged; simulated deltas "
        f"({'; '.join(parts)})"
    )


def _base_shadow_score(de: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    """Unscaled score used for classification gates (before shadow class is known)."""
    elig_score = _to_float(de.get("eligibility_score")) or 0.0
    const_safe = 1.0 if bool(de.get("constitutional_safe")) else 0.0
    sim = ctx["simulation_confidence"]
    raw = elig_score * 0.55 + const_safe * 0.25 + sim * 0.20
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.25
    return round(_clamp(raw, 0.0, 1.0), 4)


def _compute_shadow_score(de: Dict[str, Any], shadow_class: str, ctx: Dict[str, Any]) -> float:
    raw = _base_shadow_score(de, ctx)
    raw *= _scale(shadow_class) if shadow_class in (CLASS_LIMITED, CLASS_FULL) else 0.35
    return round(_clamp(raw, 0.0, 1.0), 4)


def _classify_shadow(
    *,
    de: Dict[str, Any],
    shadow_score: float,
    ctx: Dict[str, Any],
) -> str:
    elig_class = _norm_upper(de.get("eligibility_classification"))
    const_safe = bool(de.get("constitutional_safe"))

    if not const_safe or elig_class == ELIG_NOT:
        return CLASS_NO_SHADOW

    if elig_class == ELIG_FULL and shadow_score >= 0.50:
        if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
            return CLASS_LIMITED
        if ctx["system_health_stale"]:
            return CLASS_LIMITED
        return CLASS_FULL

    if elig_class == ELIG_LIMITED and shadow_score >= 0.35:
        return CLASS_LIMITED

    if elig_class == ELIG_OBSERVE:
        return CLASS_OBSERVE

    return CLASS_NO_SHADOW


def _shadow_rationale(name: str, classification: str, shadow_score: float) -> str:
    templates = {
        CLASS_FULL: (
            f"full shadow activation: {name} rehearsed as active without runtime mutation"
        ),
        CLASS_LIMITED: (
            f"limited shadow activation: {name} rehearsed under defensive governance conditions"
        ),
        CLASS_OBSERVE: (
            f"observe only shadow: {name} passively shadowed without posture simulation"
        ),
        CLASS_NO_SHADOW: (f"no shadow activation: {name} not eligible for shadow rehearsal"),
    }
    base = templates.get(classification, templates[CLASS_NO_SHADOW])
    return f"{base} (shadow_score={shadow_score:.2f})"


def _build_doctrine_shadow(
    de: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
    *,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(de.get("policy_name", ""))
    elig_class = _norm_upper(de.get("eligibility_classification"))
    elig_score = _to_float(de.get("eligibility_score")) or 0.0
    conf = _to_float(de.get("confidence")) or 0.0
    const_safe = bool(de.get("constitutional_safe"))

    gate_score = _base_shadow_score(de, ctx)
    if prior:
        prior_score = _to_float(prior.get("shadow_score")) or gate_score
        gate_score = round(_clamp(gate_score * 0.65 + prior_score * 0.35, 0.0, 1.0), 4)

    classification = _classify_shadow(de=de, shadow_score=gate_score, ctx=ctx)
    shadow_score = _compute_shadow_score(de, classification, ctx)
    deltas = _simulated_runtime_changes(name, classification, ctx=ctx)
    behavior = _shadow_runtime_behavior(name, classification, deltas, ctx=ctx)

    future_candidate = bool(de.get("future_activation_candidate"))
    shadow_activated = classification in (CLASS_LIMITED, CLASS_FULL)

    return {
        "policy_name": name,
        "shadow_classification": classification,
        "shadow_score": shadow_score,
        "eligibility_classification": elig_class,
        "eligibility_score": round(elig_score, 4),
        "simulated_runtime_changes": deltas,
        "shadow_runtime_behavior": behavior,
        "future_activation_candidate": future_candidate,
        "shadow_activated": shadow_activated,
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "runtime_mutation_allowed": False,
        "shadow_activation_rationale": _shadow_rationale(name, classification, shadow_score),
    }


def _build_all_shadow(
    ctx: Dict[str, Any],
    prior_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for de in ctx["doctrine_eligibility"]:
        name = str(de.get("policy_name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        rows.append(_build_doctrine_shadow(de, prior_map.get(name), ctx=ctx))

    for name, prior in prior_map.items():
        if name not in seen:
            rows.append(
                {
                    "policy_name": name,
                    "shadow_classification": prior.get("shadow_classification", CLASS_NO_SHADOW),
                    "shadow_score": _to_float(prior.get("shadow_score")) or 0.0,
                    "eligibility_classification": prior.get("eligibility_classification", ELIG_NOT),
                    "eligibility_score": _to_float(prior.get("eligibility_score")) or 0.0,
                    "simulated_runtime_changes": prior.get("simulated_runtime_changes")
                    or _zero_deltas(),
                    "shadow_runtime_behavior": prior.get(
                        "shadow_runtime_behavior", "prior shadow retained"
                    ),
                    "future_activation_candidate": False,
                    "shadow_activated": prior.get("shadow_classification")
                    in (CLASS_LIMITED, CLASS_FULL),
                    "confidence": _to_float(prior.get("confidence")) or 0.0,
                    "constitutional_safe": bool(prior.get("constitutional_safe")),
                    "runtime_mutation_allowed": False,
                    "shadow_activation_rationale": "prior shadow retained; no new eligibility this cycle",
                }
            )
    return rows


# -----------------------------------------------------------
# Shadow confidence and state
# -----------------------------------------------------------
def _constitutional_safety_aggregate(shadow_rows: List[Dict[str, Any]]) -> float:
    if not shadow_rows:
        return 0.0
    vals = [1.0 if bool(r.get("constitutional_safe")) else 0.0 for r in shadow_rows]
    return sum(vals) / len(vals)


def _shadow_confidence(ctx: Dict[str, Any], shadow_rows: List[Dict[str, Any]]) -> float:
    if not shadow_rows:
        return 0.0

    avg_shadow = sum(r["shadow_score"] for r in shadow_rows) / len(shadow_rows)

    raw = (
        ctx["eligibility_confidence"] * 0.22
        + ctx["impact_confidence"] * 0.14
        + ctx["simulation_confidence"] * 0.14
        + ctx["readiness_confidence"] * 0.12
        + _constitutional_safety_aggregate(shadow_rows) * 0.14
        + ctx["governance_quality"] * 0.12
        + ctx["system_health_score"] * 0.12
    )
    raw += avg_shadow * 0.05

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


def _count_shadow(shadow_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "no_shadow": sum(1 for r in shadow_rows if r["shadow_classification"] == CLASS_NO_SHADOW),
        "observe": sum(1 for r in shadow_rows if r["shadow_classification"] == CLASS_OBSERVE),
        "limited": sum(1 for r in shadow_rows if r["shadow_classification"] == CLASS_LIMITED),
        "full": sum(1 for r in shadow_rows if r["shadow_classification"] == CLASS_FULL),
    }


def _classify_shadow_state(
    *,
    ctx: Dict[str, Any],
    shadow_confidence: float,
    shadow_rows: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not shadow_rows or not ctx["eligibility_available"]:
        reasons.append("doctrine not eligible; insufficient maturity for shadow activation")
        return SHADOW_DORMANT, reasons

    if counts["full"] >= 1 and ctx["shadow_memory_depth"] >= 2:
        reasons.append("stable long-term governance shadow process established")
        return SHADOW_INSTITUTIONAL, reasons

    if counts["full"] >= 1:
        reasons.append("full eligibility enables mature shadow activation rehearsal")
        return SHADOW_READY, reasons

    if counts["limited"] >= 1:
        reasons.append("limited eligibility enables defensive shadow activation only")
        return SHADOW_LIMITED, reasons

    if counts["observe"] >= 1:
        reasons.append("observation-only eligibility; passive shadowing only")
        return SHADOW_OBSERVE, reasons

    if counts["no_shadow"] >= 1 or ctx["eligibility_state"] == "DOCTRINE_ELIGIBILITY_DORMANT":
        reasons.append("doctrine not eligible; insufficient maturity")
        return SHADOW_DORMANT, reasons

    reasons.append("doctrine not eligible; insufficient maturity for shadow activation")
    return SHADOW_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _shadow_booleans(
    state: str,
    shadow_rows: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    return {
        "shadow_activation_available": len(shadow_rows) > 0,
        "limited_shadow_available": counts["limited"] > 0 or counts["full"] > 0,
        "full_shadow_available": counts["full"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"] or any(r.get("shadow_activated") for r in shadow_rows)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "shadow_memory_reliable": state == SHADOW_INSTITUTIONAL,
    }


def _recommendations_list(state: str, counts: Dict[str, int]) -> List[str]:
    recs = [
        "Continue shadow activation observation",
        "Maintain defensive doctrine rehearsal",
        "Avoid premature runtime assumptions",
        "Maintain runtime mutation lock",
    ]
    if counts["full"] > 0 or counts["limited"] > 0:
        recs.append("Escalate mature shadow doctrine to operator review")
    if state == SHADOW_DORMANT:
        recs.append("Accumulate more eligibility before shadow activation rehearsal")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(shadow_rows: List[Dict[str, Any]], state: str, ctx: Dict[str, Any]) -> str:
    cash = [
        r
        for r in shadow_rows
        if r["policy_name"] == "target_cash_pct"
        and r["shadow_classification"] in (CLASS_LIMITED, CLASS_FULL)
    ]
    if cash:
        return (
            "Triton shadow-activated elevated cash doctrine under defensive governance conditions "
            "to observe how runtime posture would change without mutating live runtime."
        )
    full = [r for r in shadow_rows if r["shadow_classification"] == CLASS_FULL]
    if full:
        names = ", ".join(r["policy_name"] for r in full[:3])
        return f"Triton shadow-activated full rehearsal for: {names}."
    lim = [r for r in shadow_rows if r["shadow_classification"] == CLASS_LIMITED]
    if lim:
        names = ", ".join(r["policy_name"] for r in lim[:3])
        return f"Triton shadow-activated limited defensive rehearsal for: {names}."
    if state == SHADOW_OBSERVE:
        return "Shadow activation remains observe-only; eligibility is immature."
    return (
        "Governance runtime shadow activation completed without runtime mutation. "
        "Shadow activated != runtime activated."
    )


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    shadow_confidence: float,
    shadow_rows: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Runtime Shadow Activation",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Shadow Activation State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| shadow_confidence | {shadow_confidence:.3f} |",
        f"| no_shadow | {counts['no_shadow']} |",
        f"| observe | {counts['observe']} |",
        f"| limited | {counts['limited']} |",
        f"| full | {counts['full']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Shadow Doctrine Behavior",
        "",
    ]
    if shadow_rows:
        lines.append("| policy | classification | shadow | eligibility | shadow_activated |")
        lines.append("|---|---|---|---|---|")
        for r in shadow_rows:
            lines.append(
                f"| {r['policy_name']} | {r['shadow_classification']} | {r['shadow_score']:.2f} | "
                f"{r['eligibility_classification']} | {r['shadow_activated']} |"
            )
        lines.append("")
        for r in shadow_rows:
            lines.append(
                f"- **{r['policy_name']}** ({r['shadow_classification']}): "
                f"{r['shadow_runtime_behavior']}"
            )
    else:
        lines.append("_No doctrine shadow assessments this cycle._")

    lines.extend(["", "## Runtime Shadow Changes", ""])
    active = [r for r in shadow_rows if r["shadow_activated"]]
    if active:
        for r in active:
            deltas = r["simulated_runtime_changes"]
            delta_str = (
                ", ".join(
                    f"{k}={v:+.4f}" for k, v in deltas.items() if abs(_to_float(v) or 0.0) > 1e-9
                )
                or "none"
            )
            lines.append(
                f"- **{r['policy_name']}**: {delta_str} _(simulated only; live runtime unchanged)_"
            )
    else:
        lines.append("_No simulated runtime changes this cycle._")

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
            "Shadow activation is the final rehearsal layer before future ARM runtime. "
            "Eligible != shadow activated. Shadow activated != runtime activated. "
            "No live runtime policy is changed.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Shadow memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    shadow_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "shadow_state": state,
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "shadow_confidence": round(shadow_confidence, 6),
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
        for c in SHADOW_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_runtime_shadow_activation(
    *,
    eligibility_summary: Dict[str, Any],
    eligibility_record: Dict[str, Any],
    eligibility_mem: List[Dict[str, str]],
    runtime_policy: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    simulation_summary: Dict[str, Any],
    impact_summary: Dict[str, Any],
    shadow_mem: List[Dict[str, str]],
    prior_shadow_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        eligibility_summary=eligibility_summary,
        eligibility_record=eligibility_record,
        eligibility_mem=eligibility_mem,
        runtime_policy=runtime_policy,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        scorecard=scorecard,
        court_summary=court_summary,
        council_summary=council_summary,
        simulation_summary=simulation_summary,
        impact_summary=impact_summary,
        shadow_mem=shadow_mem,
    )

    prior_map = _prior_shadow_map(prior_shadow_record)
    shadow_rows = _build_all_shadow(ctx, prior_map)
    shadow_confidence = _shadow_confidence(ctx, shadow_rows)
    counts = _count_shadow(shadow_rows)

    state, reasons = _classify_shadow_state(
        ctx=ctx,
        shadow_confidence=shadow_confidence,
        shadow_rows=shadow_rows,
        counts=counts,
    )

    booleans = _shadow_booleans(state, shadow_rows, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(shadow_rows, state, ctx)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        shadow_confidence=shadow_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(shadow_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        shadow_confidence=shadow_confidence,
        shadow_rows=shadow_rows,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_runtime_shadow_activation_engine",
        "engine_version": 1,
        "shadow_state": state,
        "shadow_confidence": shadow_confidence,
        "shadow_reasons": reasons,
        "doctrine_shadow": shadow_rows,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "eligible_vs_shadow_note": (
            "Eligible != shadow activated. Shadow activated != runtime activated. "
            "Shadow activation != runtime mutation. Live runtime is never changed."
        ),
        "constitutional_supremacy_note": (
            "Shadow activation cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "shadow_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "live_runtime_regime": ctx["regime"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutated": False,
            "runtime_mutation_allowed": False,
            "shadow_rehearsal_only": True,
        },
        "inputs_seen": {
            "arm_governance_doctrine_activation_eligibility_summary": bool(eligibility_summary),
            "arm_governance_doctrine_activation_eligibility_record": bool(eligibility_record),
            "arm_governance_doctrine_activation_eligibility_memory_rows": len(eligibility_mem),
            "runtime_policy_governed": bool(runtime_policy),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "arm_governance_doctrine_simulation_summary": bool(simulation_summary),
            "arm_governance_doctrine_impact_assessment_summary": bool(impact_summary),
            "existing_shadow_memory_rows": len(shadow_mem),
            "prior_doctrine_shadow_entries": len(prior_map),
            "n_doctrines_assessed": len(shadow_rows),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_runtime_shadow_activation_engine",
        "shadow_state": state,
        "shadow_confidence": shadow_confidence,
        "shadow_activation_available": booleans["shadow_activation_available"],
        "limited_shadow_available": booleans["limited_shadow_available"],
        "full_shadow_available": booleans["full_shadow_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "shadow_memory_reliable": booleans["shadow_memory_reliable"],
        "no_shadow_count": counts["no_shadow"],
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "full_count": counts["full"],
        "observation_cycles": ctx["observation_cycles"],
        "n_doctrines_tracked": len(shadow_rows),
        "n_recommendations": len(recommendations),
        "shadow_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance runtime shadow activation engine (Step 53). "
            "Rehearses eligible doctrine without mutating runtime. No broker calls."
        ),
    )
    p.add_argument("--eligibility-summary", default=str(DEFAULT_ELIGIBILITY_SUM))
    p.add_argument("--eligibility-record", default=str(DEFAULT_ELIGIBILITY_REC))
    p.add_argument("--eligibility-mem", default=str(DEFAULT_ELIGIBILITY_MEM))
    p.add_argument("--runtime-policy", default=str(DEFAULT_RUNTIME_POLICY))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUM))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--court-summary", default=str(DEFAULT_COURT_SUMMARY))
    p.add_argument("--council-summary", default=str(DEFAULT_COUNCIL_SUMMARY))
    p.add_argument("--simulation-summary", default=str(DEFAULT_SIMULATION_SUM))
    p.add_argument("--impact-summary", default=str(DEFAULT_IMPACT_SUM))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-mem-csv", default=str(DEFAULT_OUT_MEM_CSV))
    p.add_argument("--out-mem-parquet", default=str(DEFAULT_OUT_MEM_PQ))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[ARM_RUNTIME_SHADOW] starting "
        "(read-only shadow rehearsal; no runtime mutation; no broker calls)",
        flush=True,
    )

    eligibility_summary = _safe_read_json(
        Path(args.eligibility_summary),
        label="arm_governance_doctrine_activation_eligibility_summary.json",
    )
    eligibility_record = _safe_read_json(
        Path(args.eligibility_record), label="arm_governance_doctrine_activation_eligibility.json"
    )
    eligibility_mem = _safe_read_csv_rows(
        Path(args.eligibility_mem),
        label="arm_governance_doctrine_activation_eligibility_memory.csv",
    )
    runtime_policy = _safe_read_json(
        Path(args.runtime_policy), label="runtime_policy_governed.json"
    )
    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="autonomous_readiness_summary.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    court_summary = _safe_read_json(
        Path(args.court_summary), label="arm_constitutional_court_summary.json"
    )
    council_summary = _safe_read_json(
        Path(args.council_summary), label="arm_supreme_governance_council_summary.json"
    )
    simulation_summary = _safe_read_json(
        Path(args.simulation_summary), label="arm_governance_doctrine_simulation_summary.json"
    )
    impact_summary = _safe_read_json(
        Path(args.impact_summary), label="arm_governance_doctrine_impact_assessment_summary.json"
    )
    shadow_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_runtime_shadow_activation_memory.csv"
    )
    prior_shadow_record = _safe_read_json(
        Path(args.out_json), label="prior_arm_governance_runtime_shadow_activation.json"
    )

    record, summary, md, merged_memory = build_runtime_shadow_activation(
        eligibility_summary=eligibility_summary,
        eligibility_record=eligibility_record,
        eligibility_mem=eligibility_mem,
        runtime_policy=runtime_policy,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        scorecard=scorecard,
        court_summary=court_summary,
        council_summary=council_summary,
        simulation_summary=simulation_summary,
        impact_summary=impact_summary,
        shadow_mem=shadow_mem,
        prior_shadow_record=prior_shadow_record,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=SHADOW_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["classification_counts"]
    print(
        "[ARM_RUNTIME_SHADOW] "
        f"state={record['shadow_state']} "
        f"limited={counts['limited']} "
        f"full={counts['full']} "
        f"confidence={record['shadow_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_RUNTIME_SHADOW_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[ARM_RUNTIME_SHADOW_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_RUNTIME_SHADOW_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
