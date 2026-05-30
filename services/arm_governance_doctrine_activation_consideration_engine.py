"""
ARM Governance Doctrine Activation Consideration Engine -- Step 49.

Reads:
    data/results/arm_governance_doctrine_readiness_summary.json              (Step 48)
    data/results/arm_governance_doctrine_readiness.json                      (Step 48)
    data/results/arm_governance_doctrine_readiness_memory.csv              (Step 48)
    data/results/arm_governance_doctrine_institutional_trust_summary.json  (Step 47)
    data/results/arm_governance_doctrine_evidence_accumulation_summary.json (Step 46)
    data/results/arm_governance_doctrine_recommendation_summary.json       (Step 45)
    data/results/arm_governance_doctrine_impact_assessment_summary.json    (Step 44)
    data/results/arm_governance_doctrine_simulation_summary.json           (Step 43)
    data/results/arm_constitutional_court_summary.json                     (Step 33)
    data/results/arm_supreme_governance_council_summary.json              (Step 34)
    data/results/autonomous_governance_scorecard.json                      (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/runtime_policy_governed.json                              (Step 18)

Writes:
    data/results/arm_governance_doctrine_activation_consideration.json
    data/results/arm_governance_doctrine_activation_consideration.md
    data/results/arm_governance_doctrine_activation_consideration_summary.json
    data/results/arm_governance_doctrine_activation_consideration_memory.csv
    data/results/arm_governance_doctrine_activation_consideration_memory.parquet

Purpose
-------
This engine answers:

    "Should serious activation consideration begin?"

It converts governance doctrine readiness into formal governance activation consideration.
Ready != considered. Considered != approved. Approved != activated.
Activation consideration != runtime mutation.
Activation consideration NEVER activates doctrine or mutates runtime policy.

Consideration state cascade
---------------------------
    1. DOCTRINE_CONSIDERATION_INSTITUTIONAL  mature doctrine governance; stable consideration
    2. DOCTRINE_CONSIDERATION_SERIOUS        institutional readiness; serious deliberation
    3. DOCTRINE_CONSIDERATION_LIMITED        limited readiness; limited deliberation allowed
    4. DOCTRINE_CONSIDERATION_OBSERVE        early readiness; observe only
    5. DOCTRINE_CONSIDERATION_DORMANT        doctrine not ready; insufficient maturity

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
* Append-only consideration memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to DOCTRINE_CONSIDERATION_DORMANT.
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

DEFAULT_READINESS_SUM = RESULTS_DIR / "arm_governance_doctrine_readiness_summary.json"
DEFAULT_READINESS_REC = RESULTS_DIR / "arm_governance_doctrine_readiness.json"
DEFAULT_READINESS_MEM = RESULTS_DIR / "arm_governance_doctrine_readiness_memory.csv"
DEFAULT_TRUST_SUM = RESULTS_DIR / "arm_governance_doctrine_institutional_trust_summary.json"
DEFAULT_EVIDENCE_SUM = RESULTS_DIR / "arm_governance_doctrine_evidence_accumulation_summary.json"
DEFAULT_RECOMMENDATION_SUM = RESULTS_DIR / "arm_governance_doctrine_recommendation_summary.json"
DEFAULT_IMPACT_SUM = RESULTS_DIR / "arm_governance_doctrine_impact_assessment_summary.json"
DEFAULT_SIMULATION_SUM = RESULTS_DIR / "arm_governance_doctrine_simulation_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_AUTONOMOUS_READINESS_SUM = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_doctrine_activation_consideration.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_doctrine_activation_consideration.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_doctrine_activation_consideration_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_doctrine_activation_consideration_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_doctrine_activation_consideration_memory.parquet"


# -----------------------------------------------------------
# Consideration state constants
# -----------------------------------------------------------
CONSIDERATION_DORMANT = "DOCTRINE_CONSIDERATION_DORMANT"
CONSIDERATION_OBSERVE = "DOCTRINE_CONSIDERATION_OBSERVE"
CONSIDERATION_LIMITED = "DOCTRINE_CONSIDERATION_LIMITED"
CONSIDERATION_SERIOUS = "DOCTRINE_CONSIDERATION_SERIOUS"
CONSIDERATION_INSTITUTIONAL = "DOCTRINE_CONSIDERATION_INSTITUTIONAL"

CLASS_NONE = "NO_CONSIDERATION"
CLASS_OBSERVE = "OBSERVE_ONLY"
CLASS_LIMITED = "LIMITED_CONSIDERATION"
CLASS_SERIOUS = "SERIOUS_CONSIDERATION"

READINESS_NOT_READY = "NOT_READY"
READINESS_EARLY = "EARLY_READINESS"
READINESS_LIMITED = "READY_FOR_LIMITED_CONSIDERATION"
READINESS_INSTITUTIONAL = "INSTITUTIONAL_READINESS"

CONSIDERATION_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "consideration_state",
    "observe_count",
    "limited_count",
    "serious_count",
    "consideration_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_DOCTRINE_CONSIDERATION_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(CONSIDERATION_MEMORY_COLUMNS))
        for col in ("consideration_confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("observe_count", "limited_count", "serious_count"):
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


# -----------------------------------------------------------
# Context extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    readiness_summary: Dict[str, Any],
    readiness_record: Dict[str, Any],
    readiness_mem: List[Dict[str, str]],
    trust_summary: Dict[str, Any],
    evidence_summary: Dict[str, Any],
    recommendation_summary: Dict[str, Any],
    impact_summary: Dict[str, Any],
    simulation_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    consideration_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
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
        "doctrine_readiness": readiness_record.get("doctrine_readiness") or [],
        "readiness_available": bool(readiness_summary.get("doctrine_readiness_available")),
        "readiness_memory_depth": len(readiness_mem),
        "consideration_memory_depth": len(consideration_mem),
        "trust_confidence": _clamp(
            _to_float(trust_summary.get("trust_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "evidence_confidence": _clamp(
            _to_float(evidence_summary.get("evidence_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "observation_cycles": max(
            _to_float(readiness_summary.get("observation_cycles")) or 0,
            _to_float(evidence_summary.get("observation_cycles")) or 0,
            len(readiness_mem),
            1,
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
            or readiness_summary.get("operator_review_required")
            or trust_summary.get("operator_review_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "autonomous_readiness_score": _clamp(
            _to_float(autonomous_readiness_summary.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
    }
    return ctx


def _prior_consideration_map(prior_record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in prior_record.get("doctrine_consideration") or []:
        name = str(row.get("policy_name", ""))
        if name:
            out[name] = row
    return out


# -----------------------------------------------------------
# Per-doctrine consideration
# -----------------------------------------------------------
def _compute_consideration_score(rd: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    readiness_score = _to_float(rd.get("readiness_score")) or 0.0
    obs = int(_to_float(rd.get("observation_count")) or 0)
    obs_factor = _clamp(obs / 5.0, 0.0, 1.0)
    const_safe = 1.0 if bool(rd.get("constitutional_safe")) else 0.0

    raw = readiness_score * 0.70 + obs_factor * 0.15 + const_safe * 0.15
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.22
    return round(_clamp(raw, 0.0, 1.0), 4)


def _classify_consideration(
    *,
    rd: Dict[str, Any],
    consideration_score: float,
    ctx: Dict[str, Any],
) -> str:
    readiness_class = _norm_upper(rd.get("readiness_classification"))
    const_safe = bool(rd.get("constitutional_safe"))
    obs = int(_to_float(rd.get("observation_count")) or 0)

    if const_safe and (
        readiness_class == READINESS_INSTITUTIONAL
        or (obs >= 5 and consideration_score >= 0.70 and ctx["readiness_memory_depth"] >= 2)
    ):
        return CLASS_SERIOUS

    if const_safe and (
        readiness_class == READINESS_LIMITED or (obs >= 3 and consideration_score >= 0.55)
    ):
        return CLASS_LIMITED

    if readiness_class == READINESS_EARLY or (obs >= 2 and consideration_score >= 0.30):
        return CLASS_OBSERVE

    return CLASS_NONE


def _consideration_rationale(name: str, classification: str, consideration_score: float) -> str:
    templates = {
        CLASS_SERIOUS: (
            f"serious consideration: {name} warrants institutional activation deliberation"
        ),
        CLASS_LIMITED: (f"limited consideration: {name} may begin limited activation deliberation"),
        CLASS_OBSERVE: (f"observe only: {name} requires continued observation before deliberation"),
        CLASS_NONE: (f"no consideration: {name} is not mature enough for activation deliberation"),
    }
    base = templates.get(classification, templates[CLASS_NONE])
    return f"{base} (consideration_score={consideration_score:.2f})"


def _build_doctrine_consideration(
    rd: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
    *,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(rd.get("policy_name", ""))
    readiness_class = _norm_upper(rd.get("readiness_classification"))
    readiness_score = _to_float(rd.get("readiness_score")) or 0.0
    conf = _to_float(rd.get("confidence")) or 0.0
    const_safe = bool(rd.get("constitutional_safe"))

    consideration_score = _compute_consideration_score(rd, ctx)
    if prior:
        prior_score = _to_float(prior.get("consideration_score")) or consideration_score
        consideration_score = round(
            _clamp(consideration_score * 0.65 + prior_score * 0.35, 0.0, 1.0),
            4,
        )

    classification = _classify_consideration(
        rd=rd,
        consideration_score=consideration_score,
        ctx=ctx,
    )

    future_candidate = bool(rd.get("future_activation_candidate"))
    serious_candidate = (
        classification in (CLASS_LIMITED, CLASS_SERIOUS)
        and const_safe
        and (bool(rd.get("serious_activation_candidate")) or classification == CLASS_SERIOUS)
    )
    institutionally_considered = classification == CLASS_SERIOUS

    return {
        "policy_name": name,
        "consideration_classification": classification,
        "consideration_score": consideration_score,
        "readiness_classification": readiness_class,
        "readiness_score": round(readiness_score, 4),
        "future_activation_candidate": future_candidate,
        "serious_activation_candidate": serious_candidate,
        "institutionally_considered": institutionally_considered,
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "runtime_mutation_allowed": False,
        "consideration_rationale": _consideration_rationale(
            name, classification, consideration_score
        ),
    }


def _build_all_consideration(
    ctx: Dict[str, Any],
    prior_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for rd in ctx["doctrine_readiness"]:
        name = str(rd.get("policy_name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        rows.append(_build_doctrine_consideration(rd, prior_map.get(name), ctx=ctx))

    for name, prior in prior_map.items():
        if name not in seen:
            rows.append(
                {
                    "policy_name": name,
                    "consideration_classification": prior.get(
                        "consideration_classification", CLASS_NONE
                    ),
                    "consideration_score": _to_float(prior.get("consideration_score")) or 0.0,
                    "readiness_classification": prior.get(
                        "readiness_classification", READINESS_NOT_READY
                    ),
                    "readiness_score": _to_float(prior.get("readiness_score")) or 0.0,
                    "future_activation_candidate": False,
                    "serious_activation_candidate": False,
                    "institutionally_considered": prior.get("consideration_classification")
                    == CLASS_SERIOUS,
                    "confidence": _to_float(prior.get("confidence")) or 0.0,
                    "constitutional_safe": bool(prior.get("constitutional_safe")),
                    "runtime_mutation_allowed": False,
                    "consideration_rationale": "prior consideration retained; no new readiness this cycle",
                }
            )
    return rows


# -----------------------------------------------------------
# Consideration confidence and state
# -----------------------------------------------------------
def _evidence_repeatability(ctx: Dict[str, Any]) -> float:
    cycles = ctx["observation_cycles"]
    evidence_conf = ctx["evidence_confidence"]
    return round(_clamp(cycles / 5.0 * 0.5 + evidence_conf * 0.5, 0.0, 1.0), 4)


def _governance_persistence(ctx: Dict[str, Any]) -> float:
    rows = ctx["doctrine_readiness"]
    if not rows:
        return 0.0
    # Proxy via readiness scores for doctrines with beneficial readiness
    vals = [
        _to_float(r.get("readiness_score")) or 0.0
        for r in rows
        if _norm_upper(r.get("readiness_classification")) != READINESS_NOT_READY
    ]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _constitutional_safety_aggregate(ctx: Dict[str, Any]) -> float:
    rows = ctx["doctrine_readiness"]
    if not rows:
        return 0.0
    vals = [1.0 if bool(r.get("constitutional_safe")) else 0.0 for r in rows]
    return sum(vals) / len(vals)


def _consideration_confidence(
    ctx: Dict[str, Any], consideration_rows: List[Dict[str, Any]]
) -> float:
    if not consideration_rows:
        return 0.0

    avg_consideration = sum(r["consideration_score"] for r in consideration_rows) / len(
        consideration_rows
    )

    raw = (
        ctx["readiness_confidence"] * 0.22
        + ctx["trust_confidence"] * 0.18
        + _evidence_repeatability(ctx) * 0.16
        + _governance_persistence(ctx) * 0.14
        + _constitutional_safety_aggregate(ctx) * 0.12
        + ctx["system_health_score"] * 0.10
        + ctx["autonomous_readiness_score"] * 0.08
    )
    raw += avg_consideration * 0.05

    penalty = ctx["constitutional_pressure"] * 0.24
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


def _count_consideration(consideration_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "none": sum(
            1 for r in consideration_rows if r["consideration_classification"] == CLASS_NONE
        ),
        "observe": sum(
            1 for r in consideration_rows if r["consideration_classification"] == CLASS_OBSERVE
        ),
        "limited": sum(
            1 for r in consideration_rows if r["consideration_classification"] == CLASS_LIMITED
        ),
        "serious": sum(
            1 for r in consideration_rows if r["consideration_classification"] == CLASS_SERIOUS
        ),
    }


def _classify_consideration_state(
    *,
    ctx: Dict[str, Any],
    consideration_confidence: float,
    consideration_rows: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not consideration_rows or not ctx["readiness_available"]:
        reasons.append("doctrine not ready; insufficient maturity for activation consideration")
        return CONSIDERATION_DORMANT, reasons

    if counts["serious"] >= 1 and ctx["consideration_memory_depth"] >= 2:
        reasons.append("mature doctrine governance with stable activation consideration quality")
        return CONSIDERATION_INSTITUTIONAL, reasons

    if counts["serious"] >= 1:
        reasons.append("institutional readiness permits serious activation deliberation")
        return CONSIDERATION_SERIOUS, reasons

    if counts["limited"] >= 1:
        reasons.append("limited readiness permits limited activation deliberation")
        return CONSIDERATION_LIMITED, reasons

    if counts["observe"] >= 1:
        reasons.append("early readiness; observe only")
        return CONSIDERATION_OBSERVE, reasons

    if counts["none"] >= 1 or ctx["readiness_state"] in (
        "DOCTRINE_READINESS_FORMING",
        "DOCTRINE_READINESS_DORMANT",
    ):
        reasons.append("doctrine not ready; insufficient maturity")
        return CONSIDERATION_DORMANT, reasons

    reasons.append("doctrine not ready; insufficient maturity for activation consideration")
    return CONSIDERATION_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _consideration_booleans(
    state: str,
    consideration_rows: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    return {
        "doctrine_consideration_available": len(consideration_rows) > 0,
        "limited_consideration_available": counts["limited"] > 0 or counts["serious"] > 0,
        "serious_consideration_available": counts["serious"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"]
            or any(r.get("serious_activation_candidate") for r in consideration_rows)
            or any(r.get("institutionally_considered") for r in consideration_rows)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "consideration_memory_reliable": state == CONSIDERATION_INSTITUTIONAL,
    }


def _recommendations_list(state: str, counts: Dict[str, int]) -> List[str]:
    recs = [
        "Continue doctrine observation",
        "Maintain defensive doctrine monitoring",
        "Avoid premature activation assumptions",
        "Maintain runtime mutation lock",
    ]
    if counts["serious"] > 0 or counts["limited"] > 0:
        recs.append("Escalate serious consideration doctrine to operator review")
    if state == CONSIDERATION_DORMANT:
        recs.append("Accumulate more readiness before formal activation consideration")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(
    consideration_rows: List[Dict[str, Any]], state: str, ctx: Dict[str, Any]
) -> str:
    limited = [
        r
        for r in consideration_rows
        if r["policy_name"] == "target_cash_pct"
        and r["consideration_classification"] in (CLASS_LIMITED, CLASS_SERIOUS)
    ]
    if limited and ctx["constitution_violated"]:
        return (
            "Triton begins limited activation consideration for elevated cash doctrine "
            "because repeated governance stabilization persisted under constitutional stress."
        )
    serious = [r for r in consideration_rows if r["consideration_classification"] == CLASS_SERIOUS]
    if serious:
        names = ", ".join(r["policy_name"] for r in serious[:3])
        return f"Triton permits serious activation consideration for: {names}."
    lim = [r for r in consideration_rows if r["consideration_classification"] == CLASS_LIMITED]
    if lim:
        names = ", ".join(r["policy_name"] for r in lim[:3])
        return f"Triton begins limited activation consideration for: {names}."
    if state == CONSIDERATION_OBSERVE:
        return "Activation consideration remains observe-only; readiness is immature."
    return "Governance doctrine activation consideration completed without runtime mutation."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    consideration_confidence: float,
    consideration_rows: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Doctrine Activation Consideration",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Consideration State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| consideration_confidence | {consideration_confidence:.3f} |",
        f"| no_consideration | {counts['none']} |",
        f"| observe | {counts['observe']} |",
        f"| limited | {counts['limited']} |",
        f"| serious | {counts['serious']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Doctrine Consideration",
        "",
    ]
    if consideration_rows:
        lines.append("| policy | classification | consideration | readiness | serious_candidate |")
        lines.append("|---|---|---|---|---|")
        for r in consideration_rows:
            lines.append(
                f"| {r['policy_name']} | {r['consideration_classification']} | "
                f"{r['consideration_score']:.2f} | {r['readiness_classification']} | "
                f"{r['serious_activation_candidate']} |"
            )
        lines.append("")
        for r in consideration_rows:
            lines.append(
                f"- **{r['policy_name']}** ({r['consideration_classification']}): "
                f"{r['consideration_rationale']}"
            )
    else:
        lines.append("_No doctrine consideration assessments this cycle._")

    limited_serious = [
        r
        for r in consideration_rows
        if r["consideration_classification"] in (CLASS_LIMITED, CLASS_SERIOUS)
    ]
    lines.extend(["", "## Limited or Serious Consideration", ""])
    if limited_serious:
        for r in limited_serious:
            lines.append(
                f"- {r['policy_name']}: {r['consideration_classification']} "
                f"(consideration_score={r['consideration_score']:.2f}, "
                f"institutionally_considered={r['institutionally_considered']})"
            )
    else:
        lines.append("_No limited or serious consideration yet._")

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
            "Activation consideration is governance deliberation only. Ready != considered. "
            "Considered != approved. Approved != activated. Activation consideration != runtime mutation. "
            "No runtime policy is changed.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Consideration memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    consideration_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "consideration_state": state,
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "serious_count": counts["serious"],
        "consideration_confidence": round(consideration_confidence, 6),
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
        for c in CONSIDERATION_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_doctrine_activation_consideration(
    *,
    readiness_summary: Dict[str, Any],
    readiness_record: Dict[str, Any],
    readiness_mem: List[Dict[str, str]],
    trust_summary: Dict[str, Any],
    evidence_summary: Dict[str, Any],
    recommendation_summary: Dict[str, Any],
    impact_summary: Dict[str, Any],
    simulation_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    consideration_mem: List[Dict[str, str]],
    prior_consideration_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        readiness_summary=readiness_summary,
        readiness_record=readiness_record,
        readiness_mem=readiness_mem,
        trust_summary=trust_summary,
        evidence_summary=evidence_summary,
        recommendation_summary=recommendation_summary,
        impact_summary=impact_summary,
        simulation_summary=simulation_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        autonomous_readiness_summary=autonomous_readiness_summary,
        runtime_policy=runtime_policy,
        consideration_mem=consideration_mem,
    )

    prior_map = _prior_consideration_map(prior_consideration_record)
    consideration_rows = _build_all_consideration(ctx, prior_map)
    consideration_confidence = _consideration_confidence(ctx, consideration_rows)
    counts = _count_consideration(consideration_rows)

    state, reasons = _classify_consideration_state(
        ctx=ctx,
        consideration_confidence=consideration_confidence,
        consideration_rows=consideration_rows,
        counts=counts,
    )

    booleans = _consideration_booleans(state, consideration_rows, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(consideration_rows, state, ctx)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        consideration_confidence=consideration_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(consideration_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        consideration_confidence=consideration_confidence,
        consideration_rows=consideration_rows,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_activation_consideration_engine",
        "engine_version": 1,
        "consideration_state": state,
        "consideration_confidence": consideration_confidence,
        "consideration_reasons": reasons,
        "doctrine_consideration": consideration_rows,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "ready_vs_considered_note": (
            "Ready != considered. Considered != approved. Approved != activated. "
            "Activation consideration != runtime mutation. Consideration never activates doctrine."
        ),
        "constitutional_supremacy_note": (
            "Doctrine activation consideration cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "consideration_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "consideration_deliberation_only": True,
        },
        "inputs_seen": {
            "arm_governance_doctrine_readiness_summary": bool(readiness_summary),
            "arm_governance_doctrine_readiness_record": bool(readiness_record),
            "arm_governance_doctrine_readiness_memory_rows": len(readiness_mem),
            "arm_governance_doctrine_institutional_trust_summary": bool(trust_summary),
            "arm_governance_doctrine_evidence_accumulation_summary": bool(evidence_summary),
            "arm_governance_doctrine_recommendation_summary": bool(recommendation_summary),
            "arm_governance_doctrine_impact_assessment_summary": bool(impact_summary),
            "arm_governance_doctrine_simulation_summary": bool(simulation_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(autonomous_readiness_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_consideration_memory_rows": len(consideration_mem),
            "prior_doctrine_consideration_entries": len(prior_map),
            "n_doctrines_assessed": len(consideration_rows),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_activation_consideration_engine",
        "consideration_state": state,
        "consideration_confidence": consideration_confidence,
        "doctrine_consideration_available": booleans["doctrine_consideration_available"],
        "limited_consideration_available": booleans["limited_consideration_available"],
        "serious_consideration_available": booleans["serious_consideration_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "consideration_memory_reliable": booleans["consideration_memory_reliable"],
        "none_count": counts["none"],
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "serious_count": counts["serious"],
        "observation_cycles": ctx["observation_cycles"],
        "n_doctrines_tracked": len(consideration_rows),
        "n_recommendations": len(recommendations),
        "consideration_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance doctrine activation consideration engine (Step 49). "
            "Converts readiness into formal activation consideration. "
            "Never mutates runtime policy. No broker calls."
        ),
    )
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUM))
    p.add_argument("--readiness-record", default=str(DEFAULT_READINESS_REC))
    p.add_argument("--readiness-mem", default=str(DEFAULT_READINESS_MEM))
    p.add_argument("--trust-summary", default=str(DEFAULT_TRUST_SUM))
    p.add_argument("--evidence-summary", default=str(DEFAULT_EVIDENCE_SUM))
    p.add_argument("--recommendation-summary", default=str(DEFAULT_RECOMMENDATION_SUM))
    p.add_argument("--impact-summary", default=str(DEFAULT_IMPACT_SUM))
    p.add_argument("--simulation-summary", default=str(DEFAULT_SIMULATION_SUM))
    p.add_argument("--court-summary", default=str(DEFAULT_COURT_SUMMARY))
    p.add_argument("--council-summary", default=str(DEFAULT_COUNCIL_SUMMARY))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--autonomous-readiness-summary", default=str(DEFAULT_AUTONOMOUS_READINESS_SUM))
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
        "[ARM_DOCTRINE_CONSIDERATION] starting "
        "(read-only activation deliberation; no runtime mutation; no broker calls)",
        flush=True,
    )

    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="arm_governance_doctrine_readiness_summary.json"
    )
    readiness_record = _safe_read_json(
        Path(args.readiness_record), label="arm_governance_doctrine_readiness.json"
    )
    readiness_mem = _safe_read_csv_rows(
        Path(args.readiness_mem), label="arm_governance_doctrine_readiness_memory.csv"
    )
    trust_summary = _safe_read_json(
        Path(args.trust_summary), label="arm_governance_doctrine_institutional_trust_summary.json"
    )
    evidence_summary = _safe_read_json(
        Path(args.evidence_summary),
        label="arm_governance_doctrine_evidence_accumulation_summary.json",
    )
    recommendation_summary = _safe_read_json(
        Path(args.recommendation_summary),
        label="arm_governance_doctrine_recommendation_summary.json",
    )
    impact_summary = _safe_read_json(
        Path(args.impact_summary), label="arm_governance_doctrine_impact_assessment_summary.json"
    )
    simulation_summary = _safe_read_json(
        Path(args.simulation_summary), label="arm_governance_doctrine_simulation_summary.json"
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
    autonomous_readiness_summary = _safe_read_json(
        Path(args.autonomous_readiness_summary), label="autonomous_readiness_summary.json"
    )
    runtime_policy = _safe_read_json(
        Path(args.runtime_policy), label="runtime_policy_governed.json"
    )
    consideration_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_doctrine_activation_consideration_memory.csv"
    )
    prior_consideration_record = _safe_read_json(
        Path(args.out_json), label="prior_arm_governance_doctrine_activation_consideration.json"
    )

    record, summary, md, merged_memory = build_doctrine_activation_consideration(
        readiness_summary=readiness_summary,
        readiness_record=readiness_record,
        readiness_mem=readiness_mem,
        trust_summary=trust_summary,
        evidence_summary=evidence_summary,
        recommendation_summary=recommendation_summary,
        impact_summary=impact_summary,
        simulation_summary=simulation_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        autonomous_readiness_summary=autonomous_readiness_summary,
        runtime_policy=runtime_policy,
        consideration_mem=consideration_mem,
        prior_consideration_record=prior_consideration_record,
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
        _atomic_write_csv(
            merged_memory, Path(args.out_mem_csv), columns=CONSIDERATION_MEMORY_COLUMNS
        )
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["classification_counts"]
    print(
        "[ARM_DOCTRINE_CONSIDERATION] "
        f"state={record['consideration_state']} "
        f"limited={counts['limited']} "
        f"serious={counts['serious']} "
        f"confidence={record['consideration_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_DOCTRINE_CONSIDERATION_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[ARM_DOCTRINE_CONSIDERATION_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_DOCTRINE_CONSIDERATION_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
