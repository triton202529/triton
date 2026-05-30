"""
ARM Governance Doctrine Readiness Engine -- Step 48.

Reads:
    data/results/arm_governance_doctrine_institutional_trust_summary.json   (Step 47)
    data/results/arm_governance_doctrine_institutional_trust.json           (Step 47)
    data/results/arm_governance_doctrine_institutional_trust_memory.csv   (Step 47)
    data/results/arm_governance_doctrine_evidence_accumulation_summary.json (Step 46)
    data/results/arm_governance_doctrine_recommendation_summary.json      (Step 45)
    data/results/arm_governance_doctrine_impact_assessment_summary.json   (Step 44)
    data/results/arm_governance_doctrine_simulation_summary.json          (Step 43)
    data/results/arm_constitutional_court_summary.json                    (Step 33)
    data/results/arm_supreme_governance_council_summary.json             (Step 34)
    data/results/autonomous_governance_scorecard.json                     (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/runtime_policy_governed.json                             (Step 18)

Writes:
    data/results/arm_governance_doctrine_readiness.json
    data/results/arm_governance_doctrine_readiness.md
    data/results/arm_governance_doctrine_readiness_summary.json
    data/results/arm_governance_doctrine_readiness_memory.csv
    data/results/arm_governance_doctrine_readiness_memory.parquet

Purpose
-------
This engine answers:

    "Is this doctrine mature enough for serious activation consideration?"

It converts institutional governance trust into governance doctrine activation readiness.
Trusted != ready. Ready != activated. Readiness != runtime mutation.
Readiness NEVER activates doctrine or mutates runtime policy.

Readiness state cascade
-----------------------
    1. DOCTRINE_READINESS_INSTITUTIONAL  institutional trust stable; mature readiness
    2. DOCTRINE_READINESS_LIMITED        high trust; safe for limited future consideration
    3. DOCTRINE_READINESS_EARLY          cautious trust; activation maturity beginning
    4. DOCTRINE_READINESS_FORMING        trust immature; readiness weak
    5. DOCTRINE_READINESS_DORMANT        insufficient trust and repeatability

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
* Missing inputs warn-and-continue; defaults to DOCTRINE_READINESS_DORMANT.
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

DEFAULT_TRUST_SUM = RESULTS_DIR / "arm_governance_doctrine_institutional_trust_summary.json"
DEFAULT_TRUST_REC = RESULTS_DIR / "arm_governance_doctrine_institutional_trust.json"
DEFAULT_TRUST_MEM = RESULTS_DIR / "arm_governance_doctrine_institutional_trust_memory.csv"
DEFAULT_EVIDENCE_SUM = RESULTS_DIR / "arm_governance_doctrine_evidence_accumulation_summary.json"
DEFAULT_RECOMMENDATION_SUM = RESULTS_DIR / "arm_governance_doctrine_recommendation_summary.json"
DEFAULT_IMPACT_SUM = RESULTS_DIR / "arm_governance_doctrine_impact_assessment_summary.json"
DEFAULT_SIMULATION_SUM = RESULTS_DIR / "arm_governance_doctrine_simulation_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_SUM = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_doctrine_readiness.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_doctrine_readiness.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_doctrine_readiness_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_doctrine_readiness_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_doctrine_readiness_memory.parquet"


# -----------------------------------------------------------
# Readiness state constants
# -----------------------------------------------------------
READINESS_DORMANT = "DOCTRINE_READINESS_DORMANT"
READINESS_FORMING = "DOCTRINE_READINESS_FORMING"
READINESS_EARLY = "DOCTRINE_READINESS_EARLY"
READINESS_LIMITED = "DOCTRINE_READINESS_LIMITED"
READINESS_INSTITUTIONAL = "DOCTRINE_READINESS_INSTITUTIONAL"

CLASS_NOT_READY = "NOT_READY"
CLASS_EARLY = "EARLY_READINESS"
CLASS_LIMITED = "READY_FOR_LIMITED_CONSIDERATION"
CLASS_INSTITUTIONAL = "INSTITUTIONAL_READINESS"

TRUST_LOW = "LOW_TRUST"
TRUST_CAUTIOUS = "CAUTIOUS_TRUST"
TRUST_HIGH = "HIGH_TRUST"
TRUST_INSTITUTIONAL = "INSTITUTIONAL_TRUST"

READINESS_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "readiness_state",
    "early_count",
    "limited_count",
    "institutional_count",
    "readiness_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_DOCTRINE_READINESS_WARN] {msg}", flush=True)


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
        for col in ("early_count", "limited_count", "institutional_count"):
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
    trust_summary: Dict[str, Any],
    trust_record: Dict[str, Any],
    trust_mem: List[Dict[str, str]],
    evidence_summary: Dict[str, Any],
    recommendation_summary: Dict[str, Any],
    impact_summary: Dict[str, Any],
    simulation_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    readiness_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    ctx: Dict[str, Any] = {
        "trust_state": _norm_upper(
            trust_summary.get("trust_state") or trust_record.get("trust_state")
        ),
        "trust_confidence": _clamp(
            _to_float(trust_summary.get("trust_confidence") or trust_record.get("trust_confidence"))
            or 0.0,
            0.0,
            1.0,
        ),
        "doctrine_trust": trust_record.get("doctrine_trust") or [],
        "trust_available": bool(trust_summary.get("doctrine_trust_available")),
        "trust_memory_depth": len(trust_mem),
        "readiness_memory_depth": len(readiness_mem),
        "evidence_confidence": _clamp(
            _to_float(evidence_summary.get("evidence_confidence")) or 0.0,
            0.0,
            1.0,
        ),
        "observation_cycles": max(
            _to_float(evidence_summary.get("observation_cycles")) or 0,
            _to_float(trust_summary.get("observation_cycles")) or 0,
            len(trust_mem),
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
        "recommendation_confidence": _clamp(
            _to_float(recommendation_summary.get("recommendation_confidence")) or 0.0,
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
            or trust_summary.get("operator_review_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "readiness_score": _clamp(
            _to_float(readiness_summary.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
    }
    return ctx


def _prior_readiness_map(prior_record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in prior_record.get("doctrine_readiness") or []:
        name = str(row.get("policy_name", ""))
        if name:
            out[name] = row
    return out


# -----------------------------------------------------------
# Per-doctrine readiness
# -----------------------------------------------------------
def _evidence_repeatability(tr: Dict[str, Any]) -> float:
    obs = int(_to_float(tr.get("observation_count")) or 0)
    beneficial = _to_float(tr.get("beneficial_frequency")) or 0.0
    obs_factor = _clamp(obs / 5.0, 0.0, 1.0)
    return round(_clamp(beneficial * 0.6 + obs_factor * 0.4, 0.0, 1.0), 4)


def _compute_readiness_score(tr: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    trust_score = _to_float(tr.get("trust_score")) or 0.0
    repeatability = _evidence_repeatability(tr)
    const_safe = _to_float(tr.get("constitutional_safety_stability")) or 0.0
    gov_persist = _to_float(tr.get("governance_improvement_persistence")) or 0.0

    raw = trust_score * 0.35 + repeatability * 0.25 + const_safe * 0.20 + gov_persist * 0.20
    # Readiness is more conservative than trust under constitutional pressure
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.20
    return round(_clamp(raw, 0.0, 1.0), 4)


def _is_constitutional_safe(tr: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    const_stability = _to_float(tr.get("constitutional_safety_stability")) or 0.0
    return const_stability >= 0.80 and not ctx["constitution_violated"]


def _classify_readiness(
    *,
    tr: Dict[str, Any],
    readiness_score: float,
    ctx: Dict[str, Any],
) -> str:
    trust_class = _norm_upper(tr.get("trust_classification"))
    obs = int(_to_float(tr.get("observation_count")) or 0)
    beneficial = _to_float(tr.get("beneficial_frequency")) or 0.0
    const_safe = _is_constitutional_safe(tr, ctx)
    gov_persist = _to_float(tr.get("governance_improvement_persistence")) or 0.0

    if trust_class == TRUST_INSTITUTIONAL or (
        obs >= 5
        and beneficial >= 0.75
        and const_safe
        and gov_persist >= 0.65
        and readiness_score >= 0.70
        and ctx["trust_memory_depth"] >= 2
    ):
        return CLASS_INSTITUTIONAL

    if trust_class == TRUST_HIGH or (
        obs >= 3
        and beneficial >= 0.65
        and const_safe
        and gov_persist >= 0.55
        and readiness_score >= 0.55
    ):
        return CLASS_LIMITED

    if trust_class == TRUST_CAUTIOUS or (
        obs >= 2 and beneficial >= 0.50 and readiness_score >= 0.35
    ):
        return CLASS_EARLY

    return CLASS_NOT_READY


def _readiness_rationale(name: str, classification: str, readiness_score: float) -> str:
    templates = {
        CLASS_INSTITUTIONAL: (
            f"institutional readiness: {name} has mature doctrine confidence and persistent reliability"
        ),
        CLASS_LIMITED: (f"limited readiness: {name} is ready for limited future consideration"),
        CLASS_EARLY: (
            f"early readiness: {name} shows emerging activation maturity with cautious trust"
        ),
        CLASS_NOT_READY: (
            f"not ready: {name} lacks sufficient trust and repeatability for activation consideration"
        ),
    }
    base = templates.get(classification, templates[CLASS_NOT_READY])
    return f"{base} (readiness_score={readiness_score:.2f})"


def _build_doctrine_readiness(
    tr: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
    *,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(tr.get("policy_name", ""))
    trust_class = _norm_upper(tr.get("trust_classification"))
    trust_score = _to_float(tr.get("trust_score")) or 0.0
    conf = _to_float(tr.get("confidence")) or 0.0
    obs = int(_to_float(tr.get("observation_count")) or 0)

    readiness_score = _compute_readiness_score(tr, ctx)
    if prior:
        prior_score = _to_float(prior.get("readiness_score")) or readiness_score
        readiness_score = round(_clamp(readiness_score * 0.65 + prior_score * 0.35, 0.0, 1.0), 4)

    classification = _classify_readiness(tr=tr, readiness_score=readiness_score, ctx=ctx)
    const_safe = _is_constitutional_safe(tr, ctx)

    future_candidate = bool(tr.get("future_activation_candidate"))
    serious_candidate = (
        classification in (CLASS_LIMITED, CLASS_INSTITUTIONAL)
        and const_safe
        and (future_candidate or classification == CLASS_INSTITUTIONAL)
    )
    institutionally_ready = classification == CLASS_INSTITUTIONAL

    return {
        "policy_name": name,
        "readiness_classification": classification,
        "readiness_score": readiness_score,
        "trust_classification": trust_class,
        "trust_score": round(trust_score, 4),
        "observation_count": obs,
        "future_activation_candidate": future_candidate,
        "serious_activation_candidate": serious_candidate,
        "institutionally_ready": institutionally_ready,
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "runtime_mutation_allowed": False,
        "readiness_rationale": _readiness_rationale(name, classification, readiness_score),
    }


def _build_all_readiness(
    ctx: Dict[str, Any],
    prior_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for tr in ctx["doctrine_trust"]:
        name = str(tr.get("policy_name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        rows.append(_build_doctrine_readiness(tr, prior_map.get(name), ctx=ctx))

    for name, prior in prior_map.items():
        if name not in seen:
            rows.append(
                {
                    "policy_name": name,
                    "readiness_classification": prior.get(
                        "readiness_classification", CLASS_NOT_READY
                    ),
                    "readiness_score": _to_float(prior.get("readiness_score")) or 0.0,
                    "trust_classification": prior.get("trust_classification", TRUST_LOW),
                    "trust_score": _to_float(prior.get("trust_score")) or 0.0,
                    "observation_count": int(_to_float(prior.get("observation_count")) or 0),
                    "future_activation_candidate": False,
                    "serious_activation_candidate": False,
                    "institutionally_ready": prior.get("readiness_classification")
                    == CLASS_INSTITUTIONAL,
                    "confidence": _to_float(prior.get("confidence")) or 0.0,
                    "constitutional_safe": bool(prior.get("constitutional_safe")),
                    "runtime_mutation_allowed": False,
                    "readiness_rationale": "prior readiness retained; no new trust this cycle",
                }
            )
    return rows


# -----------------------------------------------------------
# Readiness confidence and state
# -----------------------------------------------------------
def _avg_metric(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    vals = [_to_float(r.get(key)) or 0.0 for r in rows]
    return sum(vals) / len(vals)


def _evidence_repeatability_aggregate(ctx: Dict[str, Any]) -> float:
    rows = ctx["doctrine_trust"]
    if not rows:
        return 0.0
    vals = [_evidence_repeatability(r) for r in rows]
    return sum(vals) / len(vals)


def _readiness_confidence(ctx: Dict[str, Any], readiness_rows: List[Dict[str, Any]]) -> float:
    if not readiness_rows:
        return 0.0

    avg_readiness = sum(r["readiness_score"] for r in readiness_rows) / len(readiness_rows)

    raw = (
        ctx["trust_confidence"] * 0.22
        + _evidence_repeatability_aggregate(ctx) * 0.20
        + _avg_metric(ctx["doctrine_trust"], "governance_improvement_persistence") * 0.18
        + _avg_metric(ctx["doctrine_trust"], "constitutional_safety_stability") * 0.15
        + ctx["system_health_score"] * 0.12
        + ctx["readiness_score"] * 0.08
        + avg_readiness * 0.05
    )

    penalty = ctx["constitutional_pressure"] * 0.22
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


def _count_readiness(readiness_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "not_ready": sum(
            1 for r in readiness_rows if r["readiness_classification"] == CLASS_NOT_READY
        ),
        "early": sum(1 for r in readiness_rows if r["readiness_classification"] == CLASS_EARLY),
        "limited": sum(1 for r in readiness_rows if r["readiness_classification"] == CLASS_LIMITED),
        "institutional": sum(
            1 for r in readiness_rows if r["readiness_classification"] == CLASS_INSTITUTIONAL
        ),
    }


def _classify_readiness_state(
    *,
    ctx: Dict[str, Any],
    readiness_confidence: float,
    readiness_rows: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not readiness_rows or not ctx["trust_available"]:
        reasons.append("insufficient trust for doctrine activation readiness")
        return READINESS_DORMANT, reasons

    if counts["institutional"] >= 1 and ctx["readiness_memory_depth"] >= 2:
        reasons.append("institutional trust stable; mature readiness achieved")
        return READINESS_INSTITUTIONAL, reasons

    if counts["limited"] >= 1:
        reasons.append("high trust doctrines ready for limited future consideration")
        return READINESS_LIMITED, reasons

    if counts["early"] >= 1:
        reasons.append("cautious trust; activation maturity beginning")
        return READINESS_EARLY, reasons

    if (
        ctx["observation_cycles"] >= 1
        or counts["not_ready"] >= 1
        or ctx["trust_state"] in ("DOCTRINE_TRUST_FORMING", "DOCTRINE_TRUST_CAUTIOUS")
    ):
        reasons.append("trust immature; readiness weak")
        return READINESS_FORMING, reasons

    reasons.append("insufficient trust for doctrine activation readiness")
    return READINESS_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _readiness_booleans(
    state: str,
    readiness_rows: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    return {
        "doctrine_readiness_available": len(readiness_rows) > 0,
        "limited_readiness_available": counts["limited"] > 0 or counts["institutional"] > 0,
        "institutional_readiness_available": counts["institutional"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"]
            or any(r.get("serious_activation_candidate") for r in readiness_rows)
            or any(r.get("institutionally_ready") for r in readiness_rows)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "readiness_memory_reliable": state == READINESS_INSTITUTIONAL,
    }


def _recommendations_list(state: str, counts: Dict[str, int]) -> List[str]:
    recs = [
        "Continue readiness accumulation",
        "Maintain defensive doctrine observation",
        "Avoid premature activation assumptions",
        "Maintain runtime mutation lock",
    ]
    if counts["institutional"] > 0 or counts["limited"] > 0:
        recs.append("Escalate institutional readiness doctrine to operator review")
    if state == READINESS_DORMANT:
        recs.append("Accumulate more trust before forming activation readiness")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(readiness_rows: List[Dict[str, Any]], state: str, ctx: Dict[str, Any]) -> str:
    limited = [
        r
        for r in readiness_rows
        if r["policy_name"] == "target_cash_pct"
        and r["readiness_classification"] in (CLASS_EARLY, CLASS_LIMITED, CLASS_INSTITUTIONAL)
    ]
    if limited and ctx["constitution_violated"]:
        return (
            "Triton considers elevated cash doctrine ready for limited future consideration "
            "because governance stabilization repeatedly persisted under constitutional stress."
        )
    inst = [r for r in readiness_rows if r["readiness_classification"] == CLASS_INSTITUTIONAL]
    if inst:
        names = ", ".join(r["policy_name"] for r in inst[:3])
        return f"Triton holds institutional readiness for: {names}."
    lim = [r for r in readiness_rows if r["readiness_classification"] == CLASS_LIMITED]
    if lim:
        names = ", ".join(r["policy_name"] for r in lim[:3])
        return f"Triton considers ready for limited consideration: {names}."
    if state == READINESS_FORMING:
        return "Doctrine activation readiness is forming; trust remains immature."
    return "Governance doctrine readiness assessment completed without runtime mutation."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    readiness_confidence: float,
    readiness_rows: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Doctrine Readiness",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Readiness State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| readiness_confidence | {readiness_confidence:.3f} |",
        f"| not_ready | {counts['not_ready']} |",
        f"| early | {counts['early']} |",
        f"| limited | {counts['limited']} |",
        f"| institutional | {counts['institutional']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Doctrine Readiness",
        "",
    ]
    if readiness_rows:
        lines.append(
            "| policy | classification | readiness | trust | observations | serious_candidate |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in readiness_rows:
            lines.append(
                f"| {r['policy_name']} | {r['readiness_classification']} | {r['readiness_score']:.2f} | "
                f"{r['trust_classification']} | {r['observation_count']} | {r['serious_activation_candidate']} |"
            )
        lines.append("")
        for r in readiness_rows:
            lines.append(
                f"- **{r['policy_name']}** ({r['readiness_classification']}): {r['readiness_rationale']}"
            )
    else:
        lines.append("_No doctrine readiness assessments this cycle._")

    limited_inst = [
        r
        for r in readiness_rows
        if r["readiness_classification"] in (CLASS_LIMITED, CLASS_INSTITUTIONAL)
    ]
    lines.extend(["", "## Limited or Institutional Readiness", ""])
    if limited_inst:
        for r in limited_inst:
            lines.append(
                f"- {r['policy_name']}: {r['readiness_classification']} "
                f"(readiness_score={r['readiness_score']:.2f}, institutionally_ready={r['institutionally_ready']})"
            )
    else:
        lines.append("_No limited or institutional readiness yet._")

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
            "Doctrine readiness is governance observation only. Trusted != ready. "
            "Ready != activated. Readiness != runtime mutation. "
            "No runtime policy is changed.",
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
        "early_count": counts["early"],
        "limited_count": counts["limited"],
        "institutional_count": counts["institutional"],
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
def build_doctrine_readiness(
    *,
    trust_summary: Dict[str, Any],
    trust_record: Dict[str, Any],
    trust_mem: List[Dict[str, str]],
    evidence_summary: Dict[str, Any],
    recommendation_summary: Dict[str, Any],
    impact_summary: Dict[str, Any],
    simulation_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    readiness_mem: List[Dict[str, str]],
    prior_readiness_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        trust_summary=trust_summary,
        trust_record=trust_record,
        trust_mem=trust_mem,
        evidence_summary=evidence_summary,
        recommendation_summary=recommendation_summary,
        impact_summary=impact_summary,
        simulation_summary=simulation_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        runtime_policy=runtime_policy,
        readiness_mem=readiness_mem,
    )

    prior_map = _prior_readiness_map(prior_readiness_record)
    readiness_rows = _build_all_readiness(ctx, prior_map)
    readiness_confidence = _readiness_confidence(ctx, readiness_rows)
    counts = _count_readiness(readiness_rows)

    state, reasons = _classify_readiness_state(
        ctx=ctx,
        readiness_confidence=readiness_confidence,
        readiness_rows=readiness_rows,
        counts=counts,
    )

    booleans = _readiness_booleans(state, readiness_rows, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(readiness_rows, state, ctx)

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
        readiness_rows=readiness_rows,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_readiness_engine",
        "engine_version": 1,
        "readiness_state": state,
        "readiness_confidence": readiness_confidence,
        "readiness_reasons": reasons,
        "doctrine_readiness": readiness_rows,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "trusted_vs_ready_note": (
            "Trusted != ready. Ready != activated. Readiness != runtime mutation. "
            "Readiness never activates doctrine."
        ),
        "constitutional_supremacy_note": (
            "Doctrine readiness cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "readiness_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "readiness_evaluation_only": True,
        },
        "inputs_seen": {
            "arm_governance_doctrine_institutional_trust_summary": bool(trust_summary),
            "arm_governance_doctrine_institutional_trust_record": bool(trust_record),
            "arm_governance_doctrine_institutional_trust_memory_rows": len(trust_mem),
            "arm_governance_doctrine_evidence_accumulation_summary": bool(evidence_summary),
            "arm_governance_doctrine_recommendation_summary": bool(recommendation_summary),
            "arm_governance_doctrine_impact_assessment_summary": bool(impact_summary),
            "arm_governance_doctrine_simulation_summary": bool(simulation_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_readiness_memory_rows": len(readiness_mem),
            "prior_doctrine_readiness_entries": len(prior_map),
            "n_doctrines_assessed": len(readiness_rows),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_readiness_engine",
        "readiness_state": state,
        "readiness_confidence": readiness_confidence,
        "doctrine_readiness_available": booleans["doctrine_readiness_available"],
        "limited_readiness_available": booleans["limited_readiness_available"],
        "institutional_readiness_available": booleans["institutional_readiness_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "readiness_memory_reliable": booleans["readiness_memory_reliable"],
        "not_ready_count": counts["not_ready"],
        "early_count": counts["early"],
        "limited_count": counts["limited"],
        "institutional_count": counts["institutional"],
        "observation_cycles": ctx["observation_cycles"],
        "n_doctrines_tracked": len(readiness_rows),
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
            "Read-only ARM governance doctrine readiness engine (Step 48). "
            "Converts institutional trust into activation readiness. "
            "Never mutates runtime policy. No broker calls."
        ),
    )
    p.add_argument("--trust-summary", default=str(DEFAULT_TRUST_SUM))
    p.add_argument("--trust-record", default=str(DEFAULT_TRUST_REC))
    p.add_argument("--trust-mem", default=str(DEFAULT_TRUST_MEM))
    p.add_argument("--evidence-summary", default=str(DEFAULT_EVIDENCE_SUM))
    p.add_argument("--recommendation-summary", default=str(DEFAULT_RECOMMENDATION_SUM))
    p.add_argument("--impact-summary", default=str(DEFAULT_IMPACT_SUM))
    p.add_argument("--simulation-summary", default=str(DEFAULT_SIMULATION_SUM))
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
        "[ARM_DOCTRINE_READINESS] starting "
        "(read-only activation readiness; no runtime mutation; no broker calls)",
        flush=True,
    )

    trust_summary = _safe_read_json(
        Path(args.trust_summary), label="arm_governance_doctrine_institutional_trust_summary.json"
    )
    trust_record = _safe_read_json(
        Path(args.trust_record), label="arm_governance_doctrine_institutional_trust.json"
    )
    trust_mem = _safe_read_csv_rows(
        Path(args.trust_mem), label="arm_governance_doctrine_institutional_trust_memory.csv"
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
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="autonomous_readiness_summary.json"
    )
    runtime_policy = _safe_read_json(
        Path(args.runtime_policy), label="runtime_policy_governed.json"
    )
    readiness_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_doctrine_readiness_memory.csv"
    )
    prior_readiness_record = _safe_read_json(
        Path(args.out_json), label="prior_arm_governance_doctrine_readiness.json"
    )

    record, summary, md, merged_memory = build_doctrine_readiness(
        trust_summary=trust_summary,
        trust_record=trust_record,
        trust_mem=trust_mem,
        evidence_summary=evidence_summary,
        recommendation_summary=recommendation_summary,
        impact_summary=impact_summary,
        simulation_summary=simulation_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
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
        "[ARM_DOCTRINE_READINESS] "
        f"state={record['readiness_state']} "
        f"limited={counts['limited']} "
        f"institutional={counts['institutional']} "
        f"confidence={record['readiness_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_DOCTRINE_READINESS_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[ARM_DOCTRINE_READINESS_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_DOCTRINE_READINESS_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
