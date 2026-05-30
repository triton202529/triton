"""
ARM Governance Doctrine Institutional Trust Engine -- Step 47.

Reads:
    data/results/arm_governance_doctrine_evidence_accumulation_summary.json  (Step 46)
    data/results/arm_governance_doctrine_evidence_accumulation.json          (Step 46)
    data/results/arm_governance_doctrine_evidence_accumulation_memory.csv  (Step 46)
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
    data/results/arm_governance_doctrine_institutional_trust.json
    data/results/arm_governance_doctrine_institutional_trust.md
    data/results/arm_governance_doctrine_institutional_trust_summary.json
    data/results/arm_governance_doctrine_institutional_trust_memory.csv
    data/results/arm_governance_doctrine_institutional_trust_memory.parquet

Purpose
-------
This engine answers:

    "Do we trust this doctrine institutionally?"

It converts governance doctrine evidence into institutional governance trust.
Evidence != institutional trust. Trusted != activated. Institutional trust != runtime mutation.
Trust NEVER activates doctrine or mutates runtime policy.

Trust state cascade
-------------------
    1. DOCTRINE_TRUST_INSTITUTIONAL  institutional-grade reliability; stable long-term trust
    2. DOCTRINE_TRUST_HIGH            strong evidence; governance improvement repeatable
    3. DOCTRINE_TRUST_CAUTIOUS        emerging evidence; trust limited
    4. DOCTRINE_TRUST_FORMING         early repeatability; trust immature
    5. DOCTRINE_TRUST_DORMANT         insufficient evidence

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
* Append-only trust memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to DOCTRINE_TRUST_DORMANT.
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

DEFAULT_EVIDENCE_SUM = RESULTS_DIR / "arm_governance_doctrine_evidence_accumulation_summary.json"
DEFAULT_EVIDENCE_REC = RESULTS_DIR / "arm_governance_doctrine_evidence_accumulation.json"
DEFAULT_EVIDENCE_MEM = RESULTS_DIR / "arm_governance_doctrine_evidence_accumulation_memory.csv"
DEFAULT_RECOMMENDATION_SUM = RESULTS_DIR / "arm_governance_doctrine_recommendation_summary.json"
DEFAULT_IMPACT_SUM = RESULTS_DIR / "arm_governance_doctrine_impact_assessment_summary.json"
DEFAULT_SIMULATION_SUM = RESULTS_DIR / "arm_governance_doctrine_simulation_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_SUM = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_doctrine_institutional_trust.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_doctrine_institutional_trust.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_doctrine_institutional_trust_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_doctrine_institutional_trust_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_doctrine_institutional_trust_memory.parquet"


# -----------------------------------------------------------
# Trust state constants
# -----------------------------------------------------------
TRUST_DORMANT = "DOCTRINE_TRUST_DORMANT"
TRUST_FORMING = "DOCTRINE_TRUST_FORMING"
TRUST_CAUTIOUS = "DOCTRINE_TRUST_CAUTIOUS"
TRUST_HIGH = "DOCTRINE_TRUST_HIGH"
TRUST_INSTITUTIONAL = "DOCTRINE_TRUST_INSTITUTIONAL"

CLASS_LOW = "LOW_TRUST"
CLASS_CAUTIOUS = "CAUTIOUS_TRUST"
CLASS_HIGH = "HIGH_TRUST"
CLASS_INSTITUTIONAL = "INSTITUTIONAL_TRUST"

EVIDENCE_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
EVIDENCE_EMERGING = "EMERGING_EVIDENCE"
EVIDENCE_STRONG = "STRONG_EVIDENCE"
EVIDENCE_INSTITUTIONAL = "INSTITUTIONAL_EVIDENCE"

TRUST_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "trust_state",
    "low_count",
    "cautious_count",
    "high_count",
    "institutional_count",
    "trust_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_DOCTRINE_TRUST_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(TRUST_MEMORY_COLUMNS))
        for col in ("trust_confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("low_count", "cautious_count", "high_count", "institutional_count"):
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
    evidence_summary: Dict[str, Any],
    evidence_record: Dict[str, Any],
    evidence_mem: List[Dict[str, str]],
    recommendation_summary: Dict[str, Any],
    impact_summary: Dict[str, Any],
    simulation_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    trust_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    ctx: Dict[str, Any] = {
        "evidence_state": _norm_upper(
            evidence_summary.get("evidence_state") or evidence_record.get("evidence_state")
        ),
        "evidence_confidence": _clamp(
            _to_float(
                evidence_summary.get("evidence_confidence")
                or evidence_record.get("evidence_confidence")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "doctrine_evidence": evidence_record.get("doctrine_evidence") or [],
        "evidence_available": bool(evidence_summary.get("doctrine_evidence_available")),
        "evidence_memory_depth": len(evidence_mem),
        "trust_memory_depth": len(trust_mem),
        "observation_cycles": max(
            _to_float(evidence_summary.get("observation_cycles")) or 0,
            len(evidence_mem),
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
            or evidence_summary.get("operator_review_required")
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


def _prior_trust_map(prior_record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in prior_record.get("doctrine_trust") or []:
        name = str(row.get("policy_name", ""))
        if name:
            out[name] = row
    return out


# -----------------------------------------------------------
# Per-doctrine trust
# -----------------------------------------------------------
def _compute_trust_score(ev: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    obs = int(_to_float(ev.get("observation_count")) or 0)
    obs_factor = _clamp(obs / 5.0, 0.0, 1.0)
    beneficial = _to_float(ev.get("beneficial_frequency")) or 0.0
    const_safe = _to_float(ev.get("constitutional_safety_stability")) or 0.0
    gov_persist = _to_float(ev.get("governance_improvement_persistence")) or 0.0
    recommend = _to_float(ev.get("recommendation_consistency")) or 0.0

    raw = (
        beneficial * 0.28
        + const_safe * 0.22
        + gov_persist * 0.22
        + recommend * 0.18
        + obs_factor * 0.10
    )
    # Trust is slightly more conservative than raw evidence under pressure
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.15
    return round(_clamp(raw, 0.0, 1.0), 4)


def _classify_trust(
    *,
    ev: Dict[str, Any],
    trust_score: float,
    ctx: Dict[str, Any],
) -> str:
    evidence_class = _norm_upper(ev.get("evidence_classification"))
    obs = int(_to_float(ev.get("observation_count")) or 0)
    beneficial = _to_float(ev.get("beneficial_frequency")) or 0.0
    const_safe = _to_float(ev.get("constitutional_safety_stability")) or 0.0
    gov_persist = _to_float(ev.get("governance_improvement_persistence")) or 0.0

    if evidence_class == EVIDENCE_INSTITUTIONAL or (
        obs >= 5
        and beneficial >= 0.75
        and const_safe >= 0.85
        and gov_persist >= 0.65
        and ctx["evidence_memory_depth"] >= 3
        and trust_score >= 0.70
    ):
        return CLASS_INSTITUTIONAL

    if evidence_class == EVIDENCE_STRONG or (
        obs >= 3
        and beneficial >= 0.65
        and const_safe >= 0.80
        and gov_persist >= 0.55
        and trust_score >= 0.55
    ):
        return CLASS_HIGH

    if evidence_class == EVIDENCE_EMERGING or (
        obs >= 2 and beneficial >= 0.50 and trust_score >= 0.35
    ):
        return CLASS_CAUTIOUS

    return CLASS_LOW


def _trust_rationale(name: str, classification: str, trust_score: float) -> str:
    templates = {
        CLASS_INSTITUTIONAL: (
            f"institutional trust: {name} demonstrates persistent long-term governance reliability"
        ),
        CLASS_HIGH: (f"high trust: {name} shows strong repeatable governance benefit"),
        CLASS_CAUTIOUS: (
            f"cautious trust: {name} has emerging evidence with limited institutional proof"
        ),
        CLASS_LOW: (
            f"low trust: {name} lacks sufficient repeatable evidence for institutional trust"
        ),
    }
    base = templates.get(classification, templates[CLASS_LOW])
    return f"{base} (trust_score={trust_score:.2f})"


def _build_doctrine_trust(
    ev: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
    *,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(ev.get("policy_name", ""))
    evidence_class = _norm_upper(ev.get("evidence_classification"))
    conf = _to_float(ev.get("confidence")) or 0.0

    trust_score = _compute_trust_score(ev, ctx)
    # Smooth with prior trust score if available
    if prior:
        prior_score = _to_float(prior.get("trust_score")) or trust_score
        trust_score = round(_clamp(trust_score * 0.65 + prior_score * 0.35, 0.0, 1.0), 4)

    classification = _classify_trust(ev=ev, trust_score=trust_score, ctx=ctx)

    future_candidate = classification in (CLASS_HIGH, CLASS_INSTITUTIONAL) and bool(
        ev.get("future_activation_candidate")
    )
    institutionally_trusted = classification == CLASS_INSTITUTIONAL

    return {
        "policy_name": name,
        "trust_classification": classification,
        "trust_score": trust_score,
        "observation_count": int(_to_float(ev.get("observation_count")) or 0),
        "evidence_classification": evidence_class,
        "beneficial_frequency": _to_float(ev.get("beneficial_frequency")) or 0.0,
        "constitutional_safety_stability": _to_float(ev.get("constitutional_safety_stability"))
        or 0.0,
        "governance_improvement_persistence": _to_float(
            ev.get("governance_improvement_persistence")
        )
        or 0.0,
        "future_activation_candidate": future_candidate,
        "institutionally_trusted": institutionally_trusted,
        "confidence": round(conf, 4),
        "runtime_mutation_allowed": False,
        "trust_rationale": _trust_rationale(name, classification, trust_score),
    }


def _build_all_trust(
    ctx: Dict[str, Any],
    prior_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    trust_rows: List[Dict[str, Any]] = []
    seen: set = set()
    for ev in ctx["doctrine_evidence"]:
        name = str(ev.get("policy_name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        trust_rows.append(_build_doctrine_trust(ev, prior_map.get(name), ctx=ctx))

    for name, prior in prior_map.items():
        if name not in seen:
            trust_rows.append(
                {
                    "policy_name": name,
                    "trust_classification": prior.get("trust_classification", CLASS_LOW),
                    "trust_score": _to_float(prior.get("trust_score")) or 0.0,
                    "observation_count": int(_to_float(prior.get("observation_count")) or 0),
                    "evidence_classification": prior.get(
                        "evidence_classification", EVIDENCE_INSUFFICIENT
                    ),
                    "beneficial_frequency": _to_float(prior.get("beneficial_frequency")) or 0.0,
                    "constitutional_safety_stability": _to_float(
                        prior.get("constitutional_safety_stability")
                    )
                    or 0.0,
                    "governance_improvement_persistence": _to_float(
                        prior.get("governance_improvement_persistence")
                    )
                    or 0.0,
                    "future_activation_candidate": False,
                    "institutionally_trusted": prior.get("trust_classification")
                    == CLASS_INSTITUTIONAL,
                    "confidence": _to_float(prior.get("confidence")) or 0.0,
                    "runtime_mutation_allowed": False,
                    "trust_rationale": "prior trust retained; no new evidence this cycle",
                }
            )
    return trust_rows


# -----------------------------------------------------------
# Trust confidence and state
# -----------------------------------------------------------
def _avg_recommendation_consistency(ctx: Dict[str, Any]) -> float:
    rows = ctx["doctrine_evidence"]
    if not rows:
        return 0.0
    vals = [_to_float(r.get("recommendation_consistency")) or 0.0 for r in rows]
    return sum(vals) / len(vals)


def _avg_governance_persistence(ctx: Dict[str, Any]) -> float:
    rows = ctx["doctrine_evidence"]
    if not rows:
        return 0.0
    vals = [_to_float(r.get("governance_improvement_persistence")) or 0.0 for r in rows]
    return sum(vals) / len(vals)


def _avg_constitutional_safety(ctx: Dict[str, Any]) -> float:
    rows = ctx["doctrine_evidence"]
    if not rows:
        return 0.0
    vals = [_to_float(r.get("constitutional_safety_stability")) or 0.0 for r in rows]
    return sum(vals) / len(vals)


def _trust_confidence(ctx: Dict[str, Any], trust_rows: List[Dict[str, Any]]) -> float:
    if not trust_rows:
        return 0.0

    avg_trust = sum(r["trust_score"] for r in trust_rows) / len(trust_rows)

    raw = (
        ctx["evidence_confidence"] * 0.22
        + _avg_recommendation_consistency(ctx) * 0.18
        + _avg_governance_persistence(ctx) * 0.18
        + _avg_constitutional_safety(ctx) * 0.15
        + ctx["system_health_score"] * 0.12
        + ctx["readiness_score"] * 0.10
        + avg_trust * 0.05
    )

    penalty = ctx["constitutional_pressure"] * 0.20
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


def _count_trust(trust_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "low": sum(1 for r in trust_rows if r["trust_classification"] == CLASS_LOW),
        "cautious": sum(1 for r in trust_rows if r["trust_classification"] == CLASS_CAUTIOUS),
        "high": sum(1 for r in trust_rows if r["trust_classification"] == CLASS_HIGH),
        "institutional": sum(
            1 for r in trust_rows if r["trust_classification"] == CLASS_INSTITUTIONAL
        ),
    }


def _classify_trust_state(
    *,
    ctx: Dict[str, Any],
    trust_confidence: float,
    trust_rows: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not trust_rows or not ctx["evidence_available"]:
        reasons.append("insufficient evidence for institutional trust formation")
        return TRUST_DORMANT, reasons

    if counts["institutional"] >= 1 and ctx["trust_memory_depth"] >= 2:
        reasons.append("institutional-grade reliability with stable long-term trust")
        return TRUST_INSTITUTIONAL, reasons

    if counts["high"] >= 1 or (counts["cautious"] >= 2 and trust_confidence >= 0.40):
        reasons.append("strong evidence with repeatable governance improvement")
        return TRUST_HIGH, reasons

    if counts["cautious"] >= 1:
        reasons.append("emerging evidence; trust remains limited")
        return TRUST_CAUTIOUS, reasons

    if (
        ctx["observation_cycles"] >= 1
        or counts["low"] >= 1
        or ctx["evidence_state"] in ("DOCTRINE_EVIDENCE_FORMING", "DOCTRINE_EVIDENCE_EMERGING")
    ):
        reasons.append("early repeatability observed; trust immature")
        return TRUST_FORMING, reasons

    reasons.append("insufficient evidence for institutional trust formation")
    return TRUST_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _trust_booleans(
    state: str,
    trust_rows: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    return {
        "doctrine_trust_available": len(trust_rows) > 0,
        "high_trust_available": counts["high"] > 0 or counts["institutional"] > 0,
        "institutional_trust_available": counts["institutional"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"]
            or any(r.get("future_activation_candidate") for r in trust_rows)
            or any(r.get("institutionally_trusted") for r in trust_rows)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "trust_memory_reliable": state == TRUST_INSTITUTIONAL,
    }


def _recommendations_list(state: str, counts: Dict[str, int]) -> List[str]:
    recs = [
        "Continue trust accumulation",
        "Maintain defensive doctrine observation",
        "Avoid premature activation assumptions",
        "Maintain runtime mutation lock",
    ]
    if counts["high"] > 0 or counts["institutional"] > 0:
        recs.append("Escalate institutional trust doctrine to operator review")
    if state == TRUST_DORMANT:
        recs.append("Accumulate more evidence before forming institutional trust")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(trust_rows: List[Dict[str, Any]], state: str, ctx: Dict[str, Any]) -> str:
    cash = [
        r
        for r in trust_rows
        if r["policy_name"] == "target_cash_pct"
        and r["trust_classification"] in (CLASS_CAUTIOUS, CLASS_HIGH, CLASS_INSTITUTIONAL)
    ]
    if cash and ctx["constitution_violated"]:
        return (
            "Triton institutionally trusts elevated cash doctrine because repeated "
            "governance stabilization occurred during constitutional stress."
        )
    inst = [r for r in trust_rows if r["trust_classification"] == CLASS_INSTITUTIONAL]
    if inst:
        names = ", ".join(r["policy_name"] for r in inst[:3])
        return f"Triton holds institutional trust for: {names}."
    high = [r for r in trust_rows if r["trust_classification"] == CLASS_HIGH]
    if high:
        names = ", ".join(r["policy_name"] for r in high[:3])
        return f"Triton holds high institutional trust for: {names}."
    if state == TRUST_FORMING:
        return "Institutional trust is forming; evidence repeatability remains immature."
    return "Governance doctrine institutional trust assessment completed without runtime mutation."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    trust_confidence: float,
    trust_rows: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Doctrine Institutional Trust",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Trust State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| trust_confidence | {trust_confidence:.3f} |",
        f"| low | {counts['low']} |",
        f"| cautious | {counts['cautious']} |",
        f"| high | {counts['high']} |",
        f"| institutional | {counts['institutional']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Doctrine Trust",
        "",
    ]
    if trust_rows:
        lines.append(
            "| policy | classification | trust_score | observations | beneficial | institutionally_trusted |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in trust_rows:
            lines.append(
                f"| {r['policy_name']} | {r['trust_classification']} | {r['trust_score']:.2f} | "
                f"{r['observation_count']} | {r['beneficial_frequency']:.2f} | {r['institutionally_trusted']} |"
            )
        lines.append("")
        for r in trust_rows:
            lines.append(
                f"- **{r['policy_name']}** ({r['trust_classification']}): {r['trust_rationale']}"
            )
    else:
        lines.append("_No doctrine trust assessments this cycle._")

    high_inst = [
        r for r in trust_rows if r["trust_classification"] in (CLASS_HIGH, CLASS_INSTITUTIONAL)
    ]
    lines.extend(["", "## High or Institutional Trust", ""])
    if high_inst:
        for r in high_inst:
            lines.append(
                f"- {r['policy_name']}: {r['trust_classification']} "
                f"(trust_score={r['trust_score']:.2f}, institutionally_trusted={r['institutionally_trusted']})"
            )
    else:
        lines.append("_No high or institutional trust yet._")

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
            "Institutional trust is governance observation only. Evidence != institutional trust. "
            "Trusted != activated. Institutional trust != runtime mutation. "
            "No runtime policy is changed.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Trust memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    trust_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "trust_state": state,
        "low_count": counts["low"],
        "cautious_count": counts["cautious"],
        "high_count": counts["high"],
        "institutional_count": counts["institutional"],
        "trust_confidence": round(trust_confidence, 6),
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
        for c in TRUST_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_doctrine_institutional_trust(
    *,
    evidence_summary: Dict[str, Any],
    evidence_record: Dict[str, Any],
    evidence_mem: List[Dict[str, str]],
    recommendation_summary: Dict[str, Any],
    impact_summary: Dict[str, Any],
    simulation_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    trust_mem: List[Dict[str, str]],
    prior_trust_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        evidence_summary=evidence_summary,
        evidence_record=evidence_record,
        evidence_mem=evidence_mem,
        recommendation_summary=recommendation_summary,
        impact_summary=impact_summary,
        simulation_summary=simulation_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        runtime_policy=runtime_policy,
        trust_mem=trust_mem,
    )

    prior_map = _prior_trust_map(prior_trust_record)
    trust_rows = _build_all_trust(ctx, prior_map)
    trust_confidence = _trust_confidence(ctx, trust_rows)
    counts = _count_trust(trust_rows)

    state, reasons = _classify_trust_state(
        ctx=ctx,
        trust_confidence=trust_confidence,
        trust_rows=trust_rows,
        counts=counts,
    )

    booleans = _trust_booleans(state, trust_rows, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(trust_rows, state, ctx)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        trust_confidence=trust_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(trust_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        trust_confidence=trust_confidence,
        trust_rows=trust_rows,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_institutional_trust_engine",
        "engine_version": 1,
        "trust_state": state,
        "trust_confidence": trust_confidence,
        "trust_reasons": reasons,
        "doctrine_trust": trust_rows,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "evidence_vs_trust_note": (
            "Evidence != institutional trust. Trusted != activated. "
            "Institutional trust != runtime mutation. Trust never activates doctrine."
        ),
        "constitutional_supremacy_note": (
            "Institutional trust cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "trust_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "institutional_trust_only": True,
        },
        "inputs_seen": {
            "arm_governance_doctrine_evidence_accumulation_summary": bool(evidence_summary),
            "arm_governance_doctrine_evidence_accumulation_record": bool(evidence_record),
            "arm_governance_doctrine_evidence_accumulation_memory_rows": len(evidence_mem),
            "arm_governance_doctrine_recommendation_summary": bool(recommendation_summary),
            "arm_governance_doctrine_impact_assessment_summary": bool(impact_summary),
            "arm_governance_doctrine_simulation_summary": bool(simulation_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_trust_memory_rows": len(trust_mem),
            "prior_doctrine_trust_entries": len(prior_map),
            "n_doctrines_assessed": len(trust_rows),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_institutional_trust_engine",
        "trust_state": state,
        "trust_confidence": trust_confidence,
        "doctrine_trust_available": booleans["doctrine_trust_available"],
        "high_trust_available": booleans["high_trust_available"],
        "institutional_trust_available": booleans["institutional_trust_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "trust_memory_reliable": booleans["trust_memory_reliable"],
        "low_count": counts["low"],
        "cautious_count": counts["cautious"],
        "high_count": counts["high"],
        "institutional_count": counts["institutional"],
        "observation_cycles": ctx["observation_cycles"],
        "n_doctrines_tracked": len(trust_rows),
        "n_recommendations": len(recommendations),
        "trust_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance doctrine institutional trust engine (Step 47). "
            "Converts evidence into institutional governance trust. "
            "Never mutates runtime policy. No broker calls."
        ),
    )
    p.add_argument("--evidence-summary", default=str(DEFAULT_EVIDENCE_SUM))
    p.add_argument("--evidence-record", default=str(DEFAULT_EVIDENCE_REC))
    p.add_argument("--evidence-mem", default=str(DEFAULT_EVIDENCE_MEM))
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
        "[ARM_DOCTRINE_TRUST] starting "
        "(read-only institutional trust; no runtime mutation; no broker calls)",
        flush=True,
    )

    evidence_summary = _safe_read_json(
        Path(args.evidence_summary),
        label="arm_governance_doctrine_evidence_accumulation_summary.json",
    )
    evidence_record = _safe_read_json(
        Path(args.evidence_record), label="arm_governance_doctrine_evidence_accumulation.json"
    )
    evidence_mem = _safe_read_csv_rows(
        Path(args.evidence_mem), label="arm_governance_doctrine_evidence_accumulation_memory.csv"
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
    trust_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_doctrine_institutional_trust_memory.csv"
    )
    prior_trust_record = _safe_read_json(
        Path(args.out_json), label="prior_arm_governance_doctrine_institutional_trust.json"
    )

    record, summary, md, merged_memory = build_doctrine_institutional_trust(
        evidence_summary=evidence_summary,
        evidence_record=evidence_record,
        evidence_mem=evidence_mem,
        recommendation_summary=recommendation_summary,
        impact_summary=impact_summary,
        simulation_summary=simulation_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        runtime_policy=runtime_policy,
        trust_mem=trust_mem,
        prior_trust_record=prior_trust_record,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=TRUST_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["classification_counts"]
    print(
        "[ARM_DOCTRINE_TRUST] "
        f"state={record['trust_state']} "
        f"high={counts['high']} "
        f"institutional={counts['institutional']} "
        f"confidence={record['trust_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_DOCTRINE_TRUST_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[ARM_DOCTRINE_TRUST_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_DOCTRINE_TRUST_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
