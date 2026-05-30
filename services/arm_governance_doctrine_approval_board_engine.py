"""
ARM Governance Doctrine Approval Board Engine -- Step 50.

Reads:
    data/results/arm_governance_doctrine_activation_consideration_summary.json  (Step 49)
    data/results/arm_governance_doctrine_activation_consideration.json          (Step 49)
    data/results/arm_governance_doctrine_activation_consideration_memory.csv  (Step 49)
    data/results/arm_governance_doctrine_readiness_summary.json               (Step 48)
    data/results/arm_governance_doctrine_institutional_trust_summary.json     (Step 47)
    data/results/arm_governance_doctrine_evidence_accumulation_summary.json   (Step 46)
    data/results/arm_governance_doctrine_recommendation_summary.json        (Step 45)
    data/results/arm_governance_doctrine_impact_assessment_summary.json     (Step 44)
    data/results/arm_constitutional_court_summary.json                      (Step 33)
    data/results/arm_supreme_governance_council_summary.json                 (Step 34)
    data/results/autonomous_governance_scorecard.json                       (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/runtime_policy_governed.json                               (Step 18)

Writes:
    data/results/arm_governance_doctrine_approval_board.json
    data/results/arm_governance_doctrine_approval_board.md
    data/results/arm_governance_doctrine_approval_board_summary.json
    data/results/arm_governance_doctrine_approval_board_memory.csv
    data/results/arm_governance_doctrine_approval_board_memory.parquet

Purpose
-------
This engine answers:

    "Should this doctrine be institutionally approved for future activation?"

It converts governance doctrine activation consideration into institutional governance approval.
Considered != approved. Approved != activated. Institutional approval != runtime mutation.
Approval NEVER activates doctrine or mutates runtime policy.

Approval state cascade
----------------------
    1. DOCTRINE_APPROVAL_INSTITUTIONAL  stable doctrine governance; persistent approval quality
    2. DOCTRINE_APPROVAL_SERIOUS        mature governance approval; institutional confidence high
    3. DOCTRINE_APPROVAL_LIMITED       limited doctrine approval allowed; constitutionally safe
    4. DOCTRINE_APPROVAL_OBSERVE        observation continues; approval immature
    5. DOCTRINE_APPROVAL_DORMANT        doctrine not mature enough; insufficient deliberation

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
* Append-only approval memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to DOCTRINE_APPROVAL_DORMANT.
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

DEFAULT_CONSIDERATION_SUM = (
    RESULTS_DIR / "arm_governance_doctrine_activation_consideration_summary.json"
)
DEFAULT_CONSIDERATION_REC = RESULTS_DIR / "arm_governance_doctrine_activation_consideration.json"
DEFAULT_CONSIDERATION_MEM = (
    RESULTS_DIR / "arm_governance_doctrine_activation_consideration_memory.csv"
)
DEFAULT_READINESS_SUM = RESULTS_DIR / "arm_governance_doctrine_readiness_summary.json"
DEFAULT_TRUST_SUM = RESULTS_DIR / "arm_governance_doctrine_institutional_trust_summary.json"
DEFAULT_EVIDENCE_SUM = RESULTS_DIR / "arm_governance_doctrine_evidence_accumulation_summary.json"
DEFAULT_RECOMMENDATION_SUM = RESULTS_DIR / "arm_governance_doctrine_recommendation_summary.json"
DEFAULT_IMPACT_SUM = RESULTS_DIR / "arm_governance_doctrine_impact_assessment_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_AUTONOMOUS_READINESS_SUM = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_doctrine_approval_board.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_doctrine_approval_board.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_doctrine_approval_board_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_doctrine_approval_board_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_doctrine_approval_board_memory.parquet"


# -----------------------------------------------------------
# Approval state constants
# -----------------------------------------------------------
APPROVAL_DORMANT = "DOCTRINE_APPROVAL_DORMANT"
APPROVAL_OBSERVE = "DOCTRINE_APPROVAL_OBSERVE"
APPROVAL_LIMITED = "DOCTRINE_APPROVAL_LIMITED"
APPROVAL_SERIOUS = "DOCTRINE_APPROVAL_SERIOUS"
APPROVAL_INSTITUTIONAL = "DOCTRINE_APPROVAL_INSTITUTIONAL"

CLASS_NOT_APPROVED = "NOT_APPROVED"
CLASS_OBSERVE = "OBSERVE_CONTINUED"
CLASS_LIMITED = "LIMITED_APPROVAL"
CLASS_INSTITUTIONAL = "INSTITUTIONAL_APPROVAL"

CONSIDERATION_NONE = "NO_CONSIDERATION"
CONSIDERATION_OBSERVE = "OBSERVE_ONLY"
CONSIDERATION_LIMITED = "LIMITED_CONSIDERATION"
CONSIDERATION_SERIOUS = "SERIOUS_CONSIDERATION"

APPROVAL_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "approval_state",
    "observe_count",
    "limited_count",
    "institutional_count",
    "approval_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_DOCTRINE_APPROVAL_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(APPROVAL_MEMORY_COLUMNS))
        for col in ("approval_confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("observe_count", "limited_count", "institutional_count"):
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
    consideration_summary: Dict[str, Any],
    consideration_record: Dict[str, Any],
    consideration_mem: List[Dict[str, str]],
    readiness_summary: Dict[str, Any],
    trust_summary: Dict[str, Any],
    evidence_summary: Dict[str, Any],
    recommendation_summary: Dict[str, Any],
    impact_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    approval_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    ctx: Dict[str, Any] = {
        "consideration_state": _norm_upper(
            consideration_summary.get("consideration_state")
            or consideration_record.get("consideration_state")
        ),
        "consideration_confidence": _clamp(
            _to_float(
                consideration_summary.get("consideration_confidence")
                or consideration_record.get("consideration_confidence")
            )
            or 0.0,
            0.0,
            1.0,
        ),
        "doctrine_consideration": consideration_record.get("doctrine_consideration") or [],
        "consideration_available": bool(
            consideration_summary.get("doctrine_consideration_available")
        ),
        "consideration_memory_depth": len(consideration_mem),
        "approval_memory_depth": len(approval_mem),
        "readiness_confidence": _clamp(
            _to_float(readiness_summary.get("readiness_confidence")) or 0.0,
            0.0,
            1.0,
        ),
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
            _to_float(consideration_summary.get("observation_cycles")) or 0,
            _to_float(readiness_summary.get("observation_cycles")) or 0,
            _to_float(evidence_summary.get("observation_cycles")) or 0,
            len(consideration_mem),
            1,
        ),
        "impact_confidence": _clamp(
            _to_float(impact_summary.get("impact_confidence")) or 0.0,
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
            or consideration_summary.get("operator_review_required")
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


def _prior_approval_map(prior_record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in prior_record.get("doctrine_approval") or []:
        name = str(row.get("policy_name", ""))
        if name:
            out[name] = row
    return out


# -----------------------------------------------------------
# Per-doctrine approval
# -----------------------------------------------------------
def _compute_approval_score(dc: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    consideration_score = _to_float(dc.get("consideration_score")) or 0.0
    const_safe = 1.0 if bool(dc.get("constitutional_safe")) else 0.0

    raw = consideration_score * 0.75 + const_safe * 0.25
    raw *= 1.0 - ctx["constitutional_pressure"] * 0.25
    return round(_clamp(raw, 0.0, 1.0), 4)


def _classify_approval(
    *,
    dc: Dict[str, Any],
    approval_score: float,
    ctx: Dict[str, Any],
) -> str:
    consideration_class = _norm_upper(dc.get("consideration_classification"))
    const_safe = bool(dc.get("constitutional_safe"))

    if not const_safe:
        return CLASS_NOT_APPROVED

    if consideration_class == CONSIDERATION_SERIOUS and approval_score >= 0.55:
        if ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED":
            return CLASS_LIMITED
        return CLASS_INSTITUTIONAL

    if consideration_class == CONSIDERATION_LIMITED and approval_score >= 0.40:
        return CLASS_LIMITED

    if consideration_class == CONSIDERATION_OBSERVE:
        return CLASS_OBSERVE

    return CLASS_NOT_APPROVED


def _approval_rationale(name: str, classification: str, approval_score: float) -> str:
    templates = {
        CLASS_INSTITUTIONAL: (
            f"institutional approval: {name} is institutionally approved for future activation"
        ),
        CLASS_LIMITED: (f"limited approval: {name} receives limited institutional approval"),
        CLASS_OBSERVE: (
            f"observe continued: {name} requires continued observation before approval"
        ),
        CLASS_NOT_APPROVED: (
            f"not approved: {name} lacks sufficient deliberation for institutional approval"
        ),
    }
    base = templates.get(classification, templates[CLASS_NOT_APPROVED])
    return f"{base} (approval_score={approval_score:.2f})"


def _build_doctrine_approval(
    dc: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
    *,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(dc.get("policy_name", ""))
    consideration_class = _norm_upper(dc.get("consideration_classification"))
    consideration_score = _to_float(dc.get("consideration_score")) or 0.0
    conf = _to_float(dc.get("confidence")) or 0.0
    const_safe = bool(dc.get("constitutional_safe"))

    approval_score = _compute_approval_score(dc, ctx)
    if prior:
        prior_score = _to_float(prior.get("approval_score")) or approval_score
        approval_score = round(_clamp(approval_score * 0.65 + prior_score * 0.35, 0.0, 1.0), 4)

    classification = _classify_approval(dc=dc, approval_score=approval_score, ctx=ctx)

    future_candidate = bool(dc.get("future_activation_candidate"))
    institutionally_approved = classification == CLASS_INSTITUTIONAL

    return {
        "policy_name": name,
        "approval_classification": classification,
        "approval_score": approval_score,
        "consideration_classification": consideration_class,
        "consideration_score": round(consideration_score, 4),
        "future_activation_candidate": future_candidate,
        "institutionally_approved": institutionally_approved,
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "runtime_mutation_allowed": False,
        "approval_rationale": _approval_rationale(name, classification, approval_score),
    }


def _build_all_approval(
    ctx: Dict[str, Any],
    prior_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for dc in ctx["doctrine_consideration"]:
        name = str(dc.get("policy_name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        rows.append(_build_doctrine_approval(dc, prior_map.get(name), ctx=ctx))

    for name, prior in prior_map.items():
        if name not in seen:
            rows.append(
                {
                    "policy_name": name,
                    "approval_classification": prior.get(
                        "approval_classification", CLASS_NOT_APPROVED
                    ),
                    "approval_score": _to_float(prior.get("approval_score")) or 0.0,
                    "consideration_classification": prior.get(
                        "consideration_classification", CONSIDERATION_NONE
                    ),
                    "consideration_score": _to_float(prior.get("consideration_score")) or 0.0,
                    "future_activation_candidate": False,
                    "institutionally_approved": prior.get("approval_classification")
                    == CLASS_INSTITUTIONAL,
                    "confidence": _to_float(prior.get("confidence")) or 0.0,
                    "constitutional_safe": bool(prior.get("constitutional_safe")),
                    "runtime_mutation_allowed": False,
                    "approval_rationale": "prior approval retained; no new consideration this cycle",
                }
            )
    return rows


# -----------------------------------------------------------
# Approval confidence and state
# -----------------------------------------------------------
def _evidence_repeatability(ctx: Dict[str, Any]) -> float:
    cycles = ctx["observation_cycles"]
    evidence_conf = ctx["evidence_confidence"]
    return round(_clamp(cycles / 5.0 * 0.5 + evidence_conf * 0.5, 0.0, 1.0), 4)


def _governance_persistence(ctx: Dict[str, Any]) -> float:
    rows = ctx["doctrine_consideration"]
    if not rows:
        return 0.0
    vals = [_to_float(r.get("consideration_score")) or 0.0 for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def _constitutional_safety_aggregate(approval_rows: List[Dict[str, Any]]) -> float:
    if not approval_rows:
        return 0.0
    vals = [1.0 if bool(r.get("constitutional_safe")) else 0.0 for r in approval_rows]
    return sum(vals) / len(vals)


def _approval_confidence(ctx: Dict[str, Any], approval_rows: List[Dict[str, Any]]) -> float:
    if not approval_rows:
        return 0.0

    avg_approval = sum(r["approval_score"] for r in approval_rows) / len(approval_rows)

    raw = (
        ctx["consideration_confidence"] * 0.20
        + ctx["readiness_confidence"] * 0.16
        + ctx["trust_confidence"] * 0.14
        + _evidence_repeatability(ctx) * 0.14
        + _governance_persistence(ctx) * 0.12
        + _constitutional_safety_aggregate(approval_rows) * 0.12
        + ctx["system_health_score"] * 0.12
    )
    raw += avg_approval * 0.05

    penalty = ctx["constitutional_pressure"] * 0.26
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


def _count_approval(approval_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "not_approved": sum(
            1 for r in approval_rows if r["approval_classification"] == CLASS_NOT_APPROVED
        ),
        "observe": sum(1 for r in approval_rows if r["approval_classification"] == CLASS_OBSERVE),
        "limited": sum(1 for r in approval_rows if r["approval_classification"] == CLASS_LIMITED),
        "institutional": sum(
            1 for r in approval_rows if r["approval_classification"] == CLASS_INSTITUTIONAL
        ),
    }


def _classify_approval_state(
    *,
    ctx: Dict[str, Any],
    approval_confidence: float,
    approval_rows: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not approval_rows or not ctx["consideration_available"]:
        reasons.append("doctrine not mature enough; insufficient deliberation for approval")
        return APPROVAL_DORMANT, reasons

    if counts["institutional"] >= 1 and ctx["approval_memory_depth"] >= 2:
        reasons.append("stable doctrine governance with persistent institutional approval quality")
        return APPROVAL_INSTITUTIONAL, reasons

    if counts["institutional"] >= 1:
        reasons.append("mature governance approval with high institutional confidence")
        return APPROVAL_SERIOUS, reasons

    if counts["limited"] >= 1:
        reasons.append("limited doctrine approval allowed; constitutionally safe")
        return APPROVAL_LIMITED, reasons

    if counts["observe"] >= 1:
        reasons.append("observation continues; approval immature")
        return APPROVAL_OBSERVE, reasons

    if counts["not_approved"] >= 1 or ctx["consideration_state"] in (
        "DOCTRINE_CONSIDERATION_DORMANT",
        "DOCTRINE_CONSIDERATION_OBSERVE",
    ):
        reasons.append("doctrine not mature enough; insufficient deliberation")
        return APPROVAL_DORMANT, reasons

    reasons.append("doctrine not mature enough; insufficient deliberation for approval")
    return APPROVAL_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _approval_booleans(
    state: str,
    approval_rows: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    return {
        "doctrine_approval_available": len(approval_rows) > 0,
        "limited_approval_available": counts["limited"] > 0 or counts["institutional"] > 0,
        "institutional_approval_available": counts["institutional"] > 0,
        "operator_review_required": (
            ctx["operator_pressure"]
            or any(r.get("institutionally_approved") for r in approval_rows)
            or any(r.get("future_activation_candidate") for r in approval_rows)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "approval_memory_reliable": state == APPROVAL_INSTITUTIONAL,
    }


def _recommendations_list(state: str, counts: Dict[str, int]) -> List[str]:
    recs = [
        "Continue governance observation",
        "Maintain defensive doctrine monitoring",
        "Avoid premature activation assumptions",
        "Maintain runtime mutation lock",
    ]
    if counts["institutional"] > 0 or counts["limited"] > 0:
        recs.append("Escalate institutional approval doctrine to operator review")
    if state == APPROVAL_DORMANT:
        recs.append("Accumulate more deliberation before institutional approval")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(approval_rows: List[Dict[str, Any]], state: str, ctx: Dict[str, Any]) -> str:
    limited = [
        r
        for r in approval_rows
        if r["policy_name"] == "target_cash_pct"
        and r["approval_classification"] in (CLASS_LIMITED, CLASS_INSTITUTIONAL)
    ]
    if limited and ctx["constitution_violated"]:
        return (
            "Triton grants limited approval to elevated cash doctrine because repeated "
            "governance stabilization persisted under constitutional stress."
        )
    inst = [r for r in approval_rows if r["approval_classification"] == CLASS_INSTITUTIONAL]
    if inst:
        names = ", ".join(r["policy_name"] for r in inst[:3])
        return f"Triton grants institutional approval for: {names}."
    lim = [r for r in approval_rows if r["approval_classification"] == CLASS_LIMITED]
    if lim:
        names = ", ".join(r["policy_name"] for r in lim[:3])
        return f"Triton grants limited approval to: {names}."
    if state == APPROVAL_OBSERVE:
        return "Institutional approval remains observe-only; deliberation is immature."
    return "Governance doctrine approval board assessment completed without runtime mutation."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    approval_confidence: float,
    approval_rows: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Doctrine Approval Board",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Approval State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| approval_confidence | {approval_confidence:.3f} |",
        f"| not_approved | {counts['not_approved']} |",
        f"| observe | {counts['observe']} |",
        f"| limited | {counts['limited']} |",
        f"| institutional | {counts['institutional']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Doctrine Approval",
        "",
    ]
    if approval_rows:
        lines.append(
            "| policy | classification | approval | consideration | institutionally_approved |"
        )
        lines.append("|---|---|---|---|---|")
        for r in approval_rows:
            lines.append(
                f"| {r['policy_name']} | {r['approval_classification']} | {r['approval_score']:.2f} | "
                f"{r['consideration_classification']} | {r['institutionally_approved']} |"
            )
        lines.append("")
        for r in approval_rows:
            lines.append(
                f"- **{r['policy_name']}** ({r['approval_classification']}): {r['approval_rationale']}"
            )
    else:
        lines.append("_No doctrine approval assessments this cycle._")

    limited_inst = [
        r
        for r in approval_rows
        if r["approval_classification"] in (CLASS_LIMITED, CLASS_INSTITUTIONAL)
    ]
    lines.extend(["", "## Limited or Institutional Approval", ""])
    if limited_inst:
        for r in limited_inst:
            lines.append(
                f"- {r['policy_name']}: {r['approval_classification']} "
                f"(approval_score={r['approval_score']:.2f}, institutionally_approved={r['institutionally_approved']})"
            )
    else:
        lines.append("_No limited or institutional approval yet._")

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
            "Institutional approval is governance deliberation only. Considered != approved. "
            "Approved != activated. Institutional approval != runtime mutation. "
            "No runtime policy is changed.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Approval memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    approval_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "approval_state": state,
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "institutional_count": counts["institutional"],
        "approval_confidence": round(approval_confidence, 6),
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
        for c in APPROVAL_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_doctrine_approval_board(
    *,
    consideration_summary: Dict[str, Any],
    consideration_record: Dict[str, Any],
    consideration_mem: List[Dict[str, str]],
    readiness_summary: Dict[str, Any],
    trust_summary: Dict[str, Any],
    evidence_summary: Dict[str, Any],
    recommendation_summary: Dict[str, Any],
    impact_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    autonomous_readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    approval_mem: List[Dict[str, str]],
    prior_approval_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        consideration_summary=consideration_summary,
        consideration_record=consideration_record,
        consideration_mem=consideration_mem,
        readiness_summary=readiness_summary,
        trust_summary=trust_summary,
        evidence_summary=evidence_summary,
        recommendation_summary=recommendation_summary,
        impact_summary=impact_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        autonomous_readiness_summary=autonomous_readiness_summary,
        runtime_policy=runtime_policy,
        approval_mem=approval_mem,
    )

    prior_map = _prior_approval_map(prior_approval_record)
    approval_rows = _build_all_approval(ctx, prior_map)
    approval_confidence = _approval_confidence(ctx, approval_rows)
    counts = _count_approval(approval_rows)

    state, reasons = _classify_approval_state(
        ctx=ctx,
        approval_confidence=approval_confidence,
        approval_rows=approval_rows,
        counts=counts,
    )

    booleans = _approval_booleans(state, approval_rows, ctx, counts)
    recommendations = _recommendations_list(state, counts)
    rationale = _build_rationale(approval_rows, state, ctx)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        approval_confidence=approval_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(approval_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        approval_confidence=approval_confidence,
        approval_rows=approval_rows,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_approval_board_engine",
        "engine_version": 1,
        "approval_state": state,
        "approval_confidence": approval_confidence,
        "approval_reasons": reasons,
        "doctrine_approval": approval_rows,
        "classification_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "considered_vs_approved_note": (
            "Considered != approved. Approved != activated. "
            "Institutional approval != runtime mutation. Approval never activates doctrine."
        ),
        "constitutional_supremacy_note": (
            "Doctrine approval cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "approval_memory_size_after_append": len(merged_memory),
        "observation_cycles": ctx["observation_cycles"],
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "approval_only": True,
        },
        "inputs_seen": {
            "arm_governance_doctrine_activation_consideration_summary": bool(consideration_summary),
            "arm_governance_doctrine_activation_consideration_record": bool(consideration_record),
            "arm_governance_doctrine_activation_consideration_memory_rows": len(consideration_mem),
            "arm_governance_doctrine_readiness_summary": bool(readiness_summary),
            "arm_governance_doctrine_institutional_trust_summary": bool(trust_summary),
            "arm_governance_doctrine_evidence_accumulation_summary": bool(evidence_summary),
            "arm_governance_doctrine_recommendation_summary": bool(recommendation_summary),
            "arm_governance_doctrine_impact_assessment_summary": bool(impact_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(autonomous_readiness_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_approval_memory_rows": len(approval_mem),
            "prior_doctrine_approval_entries": len(prior_map),
            "n_doctrines_assessed": len(approval_rows),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_approval_board_engine",
        "approval_state": state,
        "approval_confidence": approval_confidence,
        "doctrine_approval_available": booleans["doctrine_approval_available"],
        "limited_approval_available": booleans["limited_approval_available"],
        "institutional_approval_available": booleans["institutional_approval_available"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "approval_memory_reliable": booleans["approval_memory_reliable"],
        "not_approved_count": counts["not_approved"],
        "observe_count": counts["observe"],
        "limited_count": counts["limited"],
        "institutional_count": counts["institutional"],
        "observation_cycles": ctx["observation_cycles"],
        "n_doctrines_tracked": len(approval_rows),
        "n_recommendations": len(recommendations),
        "approval_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance doctrine approval board engine (Step 50). "
            "Converts activation consideration into institutional approval. "
            "Never mutates runtime policy. No broker calls."
        ),
    )
    p.add_argument("--consideration-summary", default=str(DEFAULT_CONSIDERATION_SUM))
    p.add_argument("--consideration-record", default=str(DEFAULT_CONSIDERATION_REC))
    p.add_argument("--consideration-mem", default=str(DEFAULT_CONSIDERATION_MEM))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUM))
    p.add_argument("--trust-summary", default=str(DEFAULT_TRUST_SUM))
    p.add_argument("--evidence-summary", default=str(DEFAULT_EVIDENCE_SUM))
    p.add_argument("--recommendation-summary", default=str(DEFAULT_RECOMMENDATION_SUM))
    p.add_argument("--impact-summary", default=str(DEFAULT_IMPACT_SUM))
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
        "[ARM_DOCTRINE_APPROVAL] starting "
        "(read-only institutional approval; no runtime mutation; no broker calls)",
        flush=True,
    )

    consideration_summary = _safe_read_json(
        Path(args.consideration_summary),
        label="arm_governance_doctrine_activation_consideration_summary.json",
    )
    consideration_record = _safe_read_json(
        Path(args.consideration_record),
        label="arm_governance_doctrine_activation_consideration.json",
    )
    consideration_mem = _safe_read_csv_rows(
        Path(args.consideration_mem),
        label="arm_governance_doctrine_activation_consideration_memory.csv",
    )
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="arm_governance_doctrine_readiness_summary.json"
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
    approval_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_doctrine_approval_board_memory.csv"
    )
    prior_approval_record = _safe_read_json(
        Path(args.out_json), label="prior_arm_governance_doctrine_approval_board.json"
    )

    record, summary, md, merged_memory = build_doctrine_approval_board(
        consideration_summary=consideration_summary,
        consideration_record=consideration_record,
        consideration_mem=consideration_mem,
        readiness_summary=readiness_summary,
        trust_summary=trust_summary,
        evidence_summary=evidence_summary,
        recommendation_summary=recommendation_summary,
        impact_summary=impact_summary,
        court_summary=court_summary,
        council_summary=council_summary,
        scorecard=scorecard,
        health_summary=health_summary,
        autonomous_readiness_summary=autonomous_readiness_summary,
        runtime_policy=runtime_policy,
        approval_mem=approval_mem,
        prior_approval_record=prior_approval_record,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=APPROVAL_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["classification_counts"]
    print(
        "[ARM_DOCTRINE_APPROVAL] "
        f"state={record['approval_state']} "
        f"limited={counts['limited']} "
        f"institutional={counts['institutional']} "
        f"confidence={record['approval_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_DOCTRINE_APPROVAL_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[ARM_DOCTRINE_APPROVAL_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_DOCTRINE_APPROVAL_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
