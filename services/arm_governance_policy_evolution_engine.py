"""
ARM Governance Policy Evolution Engine -- Step 39.

Reads:
    data/results/arm_governance_learning_summary.json              (Step 38)
    data/results/arm_governance_learning.json                      (Step 38)
    data/results/arm_governance_learning_memory.csv                (Step 38)
    data/results/arm_governance_recovery_effectiveness_summary.json (Step 37)
    data/results/arm_governance_drift_detection_summary.json       (Step 35)
    data/results/arm_supreme_governance_council_summary.json       (Step 34)
    data/results/arm_constitutional_court_summary.json             (Step 33)
    data/results/runtime_policy_governed.json                      (Step 18)
    data/results/autonomous_governance_scorecard.json              (Step 19)

Writes:
    data/results/arm_governance_policy_evolution.json
    data/results/arm_governance_policy_evolution.md
    data/results/arm_governance_policy_evolution_summary.json
    data/results/arm_governance_policy_evolution_memory.csv
    data/results/arm_governance_policy_evolution_memory.parquet

Purpose
-------
This engine answers:

    "How should Triton's governance policies evolve based on institutional learning?"

It transforms governance lessons into persistent policy evolution *proposals*.
It NEVER mutates runtime policy directly. Constitutional law remains supreme.

Evolution state cascade
-----------------------
    1. POLICY_EVOLUTION_INSTITUTIONAL  long-term stable institutional memory
    2. POLICY_EVOLUTION_MATURE         repeatable governance improvements
    3. POLICY_EVOLUTION_ADAPTIVE       stable lessons, adaptation recommended
    4. POLICY_EVOLUTION_FORMING        recurring lessons, candidate shifts
    5. POLICY_EVOLUTION_DORMANT        insufficient governance lessons

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
* Append-only policy evolution memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to POLICY_EVOLUTION_DORMANT.
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

DEFAULT_LEARNING_SUMMARY = RESULTS_DIR / "arm_governance_learning_summary.json"
DEFAULT_LEARNING_RECORD = RESULTS_DIR / "arm_governance_learning.json"
DEFAULT_LEARNING_MEM = RESULTS_DIR / "arm_governance_learning_memory.csv"
DEFAULT_EFF_SUMMARY = RESULTS_DIR / "arm_governance_recovery_effectiveness_summary.json"
DEFAULT_DRIFT_SUMMARY = RESULTS_DIR / "arm_governance_drift_detection_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_policy_evolution.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_policy_evolution.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_policy_evolution_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_policy_evolution_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_policy_evolution_memory.parquet"


# -----------------------------------------------------------
# Evolution state constants
# -----------------------------------------------------------
EVOLUTION_DORMANT = "POLICY_EVOLUTION_DORMANT"
EVOLUTION_FORMING = "POLICY_EVOLUTION_FORMING"
EVOLUTION_ADAPTIVE = "POLICY_EVOLUTION_ADAPTIVE"
EVOLUTION_MATURE = "POLICY_EVOLUTION_MATURE"
EVOLUTION_INSTITUTIONAL = "POLICY_EVOLUTION_INSTITUTIONAL"

LEARNING_TO_EVOLUTION: Dict[str, str] = {
    "GOVERNANCE_LEARNING_MINIMAL": EVOLUTION_DORMANT,
    "GOVERNANCE_LEARNING_FORMING": EVOLUTION_FORMING,
    "GOVERNANCE_LEARNING_ADAPTIVE": EVOLUTION_ADAPTIVE,
    "GOVERNANCE_LEARNING_MATURE": EVOLUTION_MATURE,
    "GOVERNANCE_LEARNING_INSTITUTIONAL": EVOLUTION_INSTITUTIONAL,
}

RECOVERY_FAILURE = frozenset({"RECOVERY_INEFFECTIVE", "RECOVERY_REGRESSING", "RECOVERY_STALLED"})
DRIFT_NEGATIVE = frozenset(
    {
        "GOVERNANCE_DRIFTING",
        "GOVERNANCE_UNSTABLE",
        "GOVERNANCE_FAILURE_RISK",
    }
)

EVOLUTION_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "evolution_state",
    "evolution_score",
    "proposals_generated",
    "governance_confidence",
    "constitutional_pressure",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_POLICY_EVOLUTION_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(EVOLUTION_MEMORY_COLUMNS))
        for col in ("evolution_score", "governance_confidence", "constitutional_pressure"):
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


def _round_policy(x: float, decimals: int = 4) -> float:
    return round(x, decimals)


# -----------------------------------------------------------
# Evidence extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    learning_summary: Dict[str, Any],
    learning_record: Dict[str, Any],
    eff_summary: Dict[str, Any],
    drift_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
) -> Dict[str, Any]:
    signals = learning_record.get("current_signals") or {}
    lessons = learning_record.get("lessons_learned") or []
    lesson_ids = {str(l.get("lesson_id", "")) for l in lessons}

    return {
        "learning_state": _norm_upper(
            learning_summary.get("learning_state") or learning_record.get("learning_state")
        ),
        "learning_score": _to_float(
            learning_summary.get("learning_score") or learning_record.get("learning_score")
        )
        or 0.0,
        "n_lessons": len(lessons),
        "lesson_ids": lesson_ids,
        "lessons": lessons,
        "learned_policies": learning_record.get("learned_policies") or [],
        "governance_confidence": _clamp(
            _to_float(learning_summary.get("governance_confidence"))
            or _to_float(signals.get("governance_confidence"))
            or _to_float(council_summary.get("council_confidence"))
            or 0.0,
            0.0,
            1.0,
        ),
        "constitutional_pressure": _clamp(
            _to_float(learning_summary.get("constitutional_pressure"))
            or _to_float(signals.get("constitutional_pressure"))
            or 0.0,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(
            runtime_policy.get("regime") or scorecard.get("regime") or signals.get("regime")
        ),
        "effectiveness_state": _norm_upper(
            eff_summary.get("effectiveness_state") or signals.get("effectiveness_state")
        ),
        "drift_state": _norm_upper(drift_summary.get("drift_state") or signals.get("drift_state")),
        "drift_score": _clamp(
            _to_float(drift_summary.get("drift_score")) or 0.0,
            0.0,
            1.0,
        ),
        "council_ruling": _norm_upper(council_summary.get("governance_ruling")),
        "court_ruling": _norm_upper(court_summary.get("judicial_ruling")),
        "constitution_state": _norm_upper(
            court_summary.get("constitution_state") or council_summary.get("constitution_state")
        ),
        "constitution_violated": _norm_upper(
            court_summary.get("constitution_state") or council_summary.get("constitution_state")
        )
        == "CONSTITUTION_VIOLATED",
        "court_override": bool(court_summary.get("constitutional_override_triggered")),
        "operator_review_required": bool(
            court_summary.get("operator_review_required")
            or council_summary.get("operator_supervision_required")
        ),
        "confidence_threshold": _to_float(runtime_policy.get("confidence_threshold")) or 0.55,
        "deployment_threshold": _to_float(runtime_policy.get("deployment_threshold")) or 0.65,
        "skepticism_threshold": _to_float(runtime_policy.get("skepticism_threshold")) or 0.50,
        "target_cash_pct": _to_float(runtime_policy.get("target_cash_pct")) or 15.0,
        "max_position_pct": _to_float(runtime_policy.get("max_position_pct")) or 6.0,
        "persistence_threshold": _to_float(runtime_policy.get("persistence_threshold")) or 0.70,
        "max_cash_reserve_pct": _to_float(runtime_policy.get("max_cash_reserve_pct")) or 35.0,
        "min_cash_reserve_pct": _to_float(runtime_policy.get("min_cash_reserve_pct")) or 10.0,
    }


def _lesson_conf(ctx: Dict[str, Any], lesson_id: str, default: float = 0.60) -> float:
    for l in ctx.get("lessons") or []:
        if l.get("lesson_id") == lesson_id:
            return _to_float(l.get("confidence")) or default
    return default


def _make_proposal(
    *,
    policy_name: str,
    current: float,
    proposed: float,
    reason: str,
    confidence: float,
    constitutional_safe: bool,
    operator_review: bool,
) -> Dict[str, Any]:
    return {
        "policy_name": policy_name,
        "current_value": _round_policy(current),
        "proposed_value": _round_policy(proposed),
        "delta": _round_policy(proposed - current),
        "reason": reason,
        "confidence": round(_clamp(confidence, 0.0, 1.0), 4),
        "constitutional_safe": constitutional_safe,
        "operator_review_required": operator_review,
    }


# -----------------------------------------------------------
# Policy evolution proposals
# -----------------------------------------------------------
def _generate_proposals(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate additive evolution proposals; never apply to runtime."""
    proposals: List[Dict[str, Any]] = []
    ids = ctx["lesson_ids"]
    op_review = ctx["operator_review_required"] or ctx["constitution_violated"]
    low_conf = ctx["governance_confidence"] < 0.35
    high_pressure = ctx["constitutional_pressure"] >= 0.55
    defensive = ctx["regime"] == "DEFENSIVE"
    drift_bad = ctx["drift_state"] in DRIFT_NEGATIVE

    # ---- Confidence policy ----
    if (
        low_conf
        or high_pressure
        or drift_bad
        or "LESSON_LOW_CONF_AUTONOMY" in ids
        or "LESSON_STALE_INTELLIGENCE" in ids
    ):
        cur = ctx["confidence_threshold"]
        bump = 0.05 if low_conf else 0.03
        if high_pressure:
            bump = max(bump, 0.04)
        proposed = _clamp(cur + bump, 0.45, 0.90)
        if proposed > cur + 0.001:
            proposals.append(
                _make_proposal(
                    policy_name="confidence_threshold",
                    current=cur,
                    proposed=proposed,
                    reason=(
                        "raise confidence_threshold because repeated governance instability "
                        "emerged during low-confidence periods"
                    ),
                    confidence=_lesson_conf(ctx, "LESSON_STALE_INTELLIGENCE", 0.70),
                    constitutional_safe=True,
                    operator_review=op_review or high_pressure,
                )
            )

    if drift_bad or high_pressure:
        cur = ctx["deployment_threshold"]
        proposed = _clamp(cur + 0.03, 0.40, 0.95)
        if proposed > cur + 0.001:
            proposals.append(
                _make_proposal(
                    policy_name="deployment_threshold",
                    current=cur,
                    proposed=proposed,
                    reason="tighten deployment_threshold during governance drift or constitutional pressure",
                    confidence=0.65 if drift_bad else 0.55,
                    constitutional_safe=True,
                    operator_review=op_review,
                )
            )

    # ---- Capital preservation ----
    if (
        "LESSON_ELEVATED_CASH" in ids
        or defensive
        or high_pressure
        or ctx["council_ruling"] in ("GOVERNANCE_TIGHTEN", "GOVERNANCE_REVOKE_AUTONOMY")
    ):
        cur = ctx["target_cash_pct"]
        bump = 2.0 if defensive else 1.5
        if high_pressure:
            bump = max(bump, 2.5)
        proposed = _clamp(cur + bump, ctx["min_cash_reserve_pct"], ctx["max_cash_reserve_pct"])
        if proposed > cur + 0.01:
            proposals.append(
                _make_proposal(
                    policy_name="target_cash_pct",
                    current=cur,
                    proposed=proposed,
                    reason="preserve elevated cash posture during instability and defensive regimes",
                    confidence=_lesson_conf(ctx, "LESSON_ELEVATED_CASH", 0.68),
                    constitutional_safe=True,
                    operator_review=False,
                )
            )

    if high_pressure or drift_bad:
        cur = ctx["max_position_pct"]
        proposed = _clamp(cur - 0.5, 2.0, cur)
        if proposed < cur - 0.01:
            proposals.append(
                _make_proposal(
                    policy_name="max_position_pct",
                    current=cur,
                    proposed=proposed,
                    reason="reduce max_position_pct to improve capital preservation under governance stress",
                    confidence=0.62,
                    constitutional_safe=True,
                    operator_review=False,
                )
            )

    # ---- Autonomy policy ----
    if "LESSON_APPRENTICESHIP" in ids or "LESSON_AUTONOMY_REGRESSION" in ids or drift_bad:
        cur = ctx["persistence_threshold"]
        proposed = _clamp(cur + 0.05, 0.40, 0.90)
        if proposed > cur + 0.001:
            proposals.append(
                _make_proposal(
                    policy_name="persistence_threshold",
                    current=cur,
                    proposed=proposed,
                    reason="extend apprenticeship duration by raising persistence threshold before graduation",
                    confidence=_lesson_conf(ctx, "LESSON_APPRENTICESHIP", 0.62),
                    constitutional_safe=True,
                    operator_review=op_review,
                )
            )

        proposals.append(
            {
                "policy_name": "min_observations_before_graduation",
                "current_value": 25,
                "proposed_value": 35,
                "delta": 10,
                "reason": "require more observations before autonomy graduation after governance drift",
                "confidence": _lesson_conf(ctx, "LESSON_APPRENTICESHIP", 0.60),
                "constitutional_safe": True,
                "operator_review_required": op_review,
            }
        )

        cur_readiness = 0.60
        proposed_readiness = 0.70
        proposals.append(
            {
                "policy_name": "autonomy_readiness_threshold",
                "current_value": cur_readiness,
                "proposed_value": proposed_readiness,
                "delta": proposed_readiness - cur_readiness,
                "reason": "strengthen autonomy readiness threshold after constitutional crises",
                "confidence": 0.65,
                "constitutional_safe": True,
                "operator_review_required": True,
            }
        )

    # ---- Governance policy ----
    if high_pressure or drift_bad or "LESSON_COURT_OVERRULING" in ids:
        cur = ctx["skepticism_threshold"]
        proposed = _clamp(cur + 0.08, 0.0, 1.0)
        if proposed > cur + 0.001:
            proposals.append(
                _make_proposal(
                    policy_name="skepticism_threshold",
                    current=cur,
                    proposed=proposed,
                    reason="increase skepticism_threshold under elevated constitutional pressure",
                    confidence=_lesson_conf(ctx, "LESSON_COURT_OVERRULING", 0.68),
                    constitutional_safe=True,
                    operator_review=op_review,
                )
            )

    if (
        "LESSON_COURT_OVERRULING" in ids
        or "LESSON_MANUAL_RECOVERY" in ids
        or ctx["court_ruling"] == "COURT_OVERRULED"
    ):
        proposals.append(
            {
                "policy_name": "auto_lock_manual_after_overruling",
                "current_value": False,
                "proposed_value": True,
                "delta": True,
                "reason": "auto-lock manual mode after repeated court overruling",
                "confidence": _lesson_conf(ctx, "LESSON_MANUAL_RECOVERY", 0.72),
                "constitutional_safe": True,
                "operator_review_required": True,
            }
        )

    if drift_bad or ctx["effectiveness_state"] in RECOVERY_FAILURE:
        proposals.append(
            {
                "policy_name": "governance_monitoring_frequency_multiplier",
                "current_value": 1.0,
                "proposed_value": 1.5,
                "delta": 0.5,
                "reason": "increase governance monitoring frequency during recovery stalls",
                "confidence": 0.58,
                "constitutional_safe": True,
                "operator_review_required": False,
            }
        )

    # dedupe by policy_name keeping highest confidence
    keyed: Dict[str, Dict[str, Any]] = {}
    for p in proposals:
        name = str(p["policy_name"])
        if name not in keyed or p["confidence"] > keyed[name]["confidence"]:
            keyed[name] = p
    return list(keyed.values())


# -----------------------------------------------------------
# Evolution score and classification
# -----------------------------------------------------------
def _evolution_score(ctx: Dict[str, Any], proposals: List[Dict[str, Any]]) -> float:
    repeatability = 0.0
    if ctx["lessons"]:
        repeated = sum(1 for l in ctx["lessons"] if (l.get("repeat_count") or 0) >= 2)
        repeatability = repeated / len(ctx["lessons"])

    eff_bonus = 0.0
    if ctx["effectiveness_state"] in ("RECOVERY_EFFECTIVE", "RECOVERY_IMPROVING"):
        eff_bonus = 0.15
    elif ctx["effectiveness_state"] in RECOVERY_FAILURE:
        eff_bonus = -0.08

    conf_factor = _clamp(ctx["governance_confidence"], 0.0, 1.0) * 0.10
    pressure_penalty = ctx["constitutional_pressure"] * 0.12
    drift_penalty = ctx["drift_score"] * 0.10

    proposal_strength = 0.0
    if proposals:
        proposal_strength = sum(p["confidence"] for p in proposals) / len(proposals) * 0.25

    raw = (
        ctx["learning_score"] * 0.30
        + repeatability * 0.20
        + proposal_strength
        + eff_bonus
        + conf_factor
        - pressure_penalty
        - drift_penalty
    )

    if ctx["effectiveness_state"] == "RECOVERY_REGRESSING":
        raw -= 0.10
    if ctx["constitution_violated"] and ctx["court_override"]:
        raw -= 0.05

    return round(_clamp(raw, 0.0, 1.0), 6)


def _classify_evolution_state(
    *,
    ctx: Dict[str, Any],
    evolution_score: float,
    n_proposals: int,
    have_evidence: bool,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not have_evidence or ctx["n_lessons"] < 2:
        reasons.append("insufficient governance lessons for policy evolution")
        return EVOLUTION_DORMANT, reasons

    mapped = LEARNING_TO_EVOLUTION.get(ctx["learning_state"], EVOLUTION_DORMANT)

    if evolution_score >= 0.80 and ctx["n_lessons"] >= 6 and n_proposals >= 4:
        reasons.append("long-term stable institutional policy memory")
        return EVOLUTION_INSTITUTIONAL, reasons

    if evolution_score >= 0.65 and ctx["n_lessons"] >= 5 and n_proposals >= 3:
        reasons.append("repeatable governance improvements visible in lesson memory")
        return EVOLUTION_MATURE, reasons

    if evolution_score >= 0.45 and n_proposals >= 2:
        reasons.append("stable lessons detected; governance adaptation recommended")
        return EVOLUTION_ADAPTIVE, reasons

    if mapped != EVOLUTION_DORMANT or n_proposals >= 1:
        reasons.append("recurring lessons emerging; candidate policy shifts forming")
        return EVOLUTION_FORMING, reasons

    reasons.append("insufficient governance lessons")
    return EVOLUTION_DORMANT, reasons


# -----------------------------------------------------------
# Booleans, rationale, recommendations
# -----------------------------------------------------------
def _evolution_booleans(
    state: str,
    proposals: List[Dict[str, Any]],
    ctx: Dict[str, Any],
) -> Dict[str, bool]:
    return {
        "policy_adjustments_available": len(proposals) > 0,
        "governance_adaptation_recommended": state
        in (
            EVOLUTION_ADAPTIVE,
            EVOLUTION_MATURE,
            EVOLUTION_INSTITUTIONAL,
        ),
        "operator_review_required": (
            ctx["operator_review_required"]
            or ctx["constitution_violated"]
            or any(p.get("operator_review_required") for p in proposals)
        ),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "policy_memory_reliable": state in (EVOLUTION_MATURE, EVOLUTION_INSTITUTIONAL),
    }


def _recommendations(state: str, proposals: List[Dict[str, Any]]) -> List[str]:
    recs: List[str] = []
    names = {p["policy_name"] for p in proposals}
    if "confidence_threshold" in names:
        recs.append("Raise confidence threshold gradually")
    if "target_cash_pct" in names:
        recs.append("Preserve elevated cash during instability")
    if "persistence_threshold" in names or "min_observations_before_graduation" in names:
        recs.append("Extend apprenticeship requirements")
    if "auto_lock_manual_after_overruling" in names:
        recs.append("Maintain manual lock after repeated overruling")
    recs.append("Continue governance learning accumulation")
    if state in (EVOLUTION_MATURE, EVOLUTION_INSTITUTIONAL):
        recs.append("Submit proposals for operator constitutional review")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(ctx: Dict[str, Any], proposals: List[Dict[str, Any]]) -> str:
    conf_prop = next((p for p in proposals if p["policy_name"] == "confidence_threshold"), None)
    if conf_prop and ctx["governance_confidence"] < 0.35:
        return (
            "Triton proposes stricter confidence thresholds because repeated governance "
            "instability emerged during low-confidence periods."
        )
    cash_prop = next((p for p in proposals if p["policy_name"] == "target_cash_pct"), None)
    if cash_prop and ctx["constitutional_pressure"] >= 0.55:
        return (
            "Triton proposes elevated cash preservation because constitutional pressure "
            "remains elevated and defensive regimes dominate governance memory."
        )
    manual = next(
        (p for p in proposals if p["policy_name"] == "auto_lock_manual_after_overruling"), None
    )
    if manual:
        return (
            "Triton proposes automatic manual-mode lock because court overruling "
            "patterns indicate governance inconsistency requiring operator supremacy."
        )
    if proposals:
        return (
            f"Triton proposes {len(proposals)} governance policy evolution adjustments "
            f"derived from {ctx['n_lessons']} institutional lessons."
        )
    return "Insufficient institutional lessons to propose policy evolution at this time."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    score: float,
    proposals: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    ctx: Dict[str, Any],
) -> str:
    lines = [
        "# Triton Governance Policy Evolution",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Evolution State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| evolution_score | {score:.3f} |",
        f"| learning_state | {ctx['learning_state']} |",
        f"| n_proposals | {len(proposals)} |",
        f"| policy_adjustments_available | {booleans['policy_adjustments_available']} |",
        f"| governance_adaptation_recommended | {booleans['governance_adaptation_recommended']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        f"| operator_review_required | {booleans['operator_review_required']} |",
        f"| constitutional_review_required | {booleans['constitutional_review_required']} |",
        "",
        "## Policy Evolution Proposals",
        "",
    ]
    if proposals:
        lines.append("| policy | current | proposed | delta | confidence | operator_review |")
        lines.append("|---|---|---|---|---|---|")
        for p in proposals:
            cur = p["current_value"]
            prop = p["proposed_value"]
            delta = p["delta"]
            lines.append(
                f"| {p['policy_name']} | {cur} | {prop} | {delta} | "
                f"{p['confidence']:.2f} | {p['operator_review_required']} |"
            )
        lines.append("")
        for p in proposals:
            lines.append(f"- **{p['policy_name']}**: {p['reason']}")
    else:
        lines.append("_No policy evolution proposals at this time._")
    lines.extend(
        [
            "",
            "## Why",
            "",
        ]
    )
    for r in reasons:
        lines.append(f"- {r}")
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )
    for rec in recommendations:
        lines.append(f"- {rec}")
    lines.extend(
        [
            "",
            "## Narrative",
            "",
            rationale,
            "",
            "These are evolution *proposals* only. Triton does not mutate runtime policy "
            "directly. All proposals remain subject to constitutional law, court rulings, "
            "capital preservation doctrine, and operator supremacy.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Evolution memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    score: float,
    proposals: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "evolution_state": state,
        "evolution_score": round(score, 6),
        "proposals_generated": "|".join(p["policy_name"] for p in proposals),
        "governance_confidence": ctx["governance_confidence"],
        "constitutional_pressure": ctx["constitutional_pressure"],
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
        for c in EVOLUTION_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_policy_evolution(
    *,
    learning_summary: Dict[str, Any],
    learning_record: Dict[str, Any],
    learning_mem: List[Dict[str, str]],
    eff_summary: Dict[str, Any],
    drift_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
    existing_evolution_mem: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        learning_summary=learning_summary,
        learning_record=learning_record,
        eff_summary=eff_summary,
        drift_summary=drift_summary,
        council_summary=council_summary,
        court_summary=court_summary,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
    )

    have_evidence = bool(learning_summary or learning_record)

    proposals = _generate_proposals(ctx)
    score = _evolution_score(ctx, proposals)
    state, reasons = _classify_evolution_state(
        ctx=ctx,
        evolution_score=score,
        n_proposals=len(proposals),
        have_evidence=have_evidence,
    )

    booleans = _evolution_booleans(state, proposals, ctx)
    recommendations = _recommendations(state, proposals)
    rationale = _build_rationale(ctx, proposals)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        score=score,
        proposals=proposals,
        ctx=ctx,
        rationale=rationale,
    )
    merged_memory = _merge_memory(existing_evolution_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        score=score,
        proposals=proposals,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        ctx=ctx,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_policy_evolution_engine",
        "engine_version": 1,
        "evolution_state": state,
        "evolution_score": score,
        "evolution_reasons": reasons,
        "policy_evolution_proposals": proposals,
        "current_policy_snapshot": {
            "confidence_threshold": ctx["confidence_threshold"],
            "deployment_threshold": ctx["deployment_threshold"],
            "skepticism_threshold": ctx["skepticism_threshold"],
            "target_cash_pct": ctx["target_cash_pct"],
            "max_position_pct": ctx["max_position_pct"],
            "persistence_threshold": ctx["persistence_threshold"],
            "regime": ctx["regime"],
        },
        "learning_context": {
            "learning_state": ctx["learning_state"],
            "learning_score": ctx["learning_score"],
            "n_lessons": ctx["n_lessons"],
            "learned_policies": ctx["learned_policies"],
        },
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "constitutional_supremacy_note": (
            "Policy evolution proposes changes only. It NEVER mutates runtime policy. "
            "Proposals cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "evolution_memory_size_after_append": len(merged_memory),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "proposals_only": True,
        },
        "inputs_seen": {
            "arm_governance_learning_summary": bool(learning_summary),
            "arm_governance_learning_record": bool(learning_record),
            "arm_governance_learning_memory_rows": len(learning_mem),
            "arm_governance_recovery_effectiveness_summary": bool(eff_summary),
            "arm_governance_drift_detection_summary": bool(drift_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "autonomous_governance_scorecard": bool(scorecard),
            "existing_evolution_memory_rows": len(existing_evolution_mem),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_policy_evolution_engine",
        "evolution_state": state,
        "evolution_score": score,
        "policy_adjustments_available": booleans["policy_adjustments_available"],
        "governance_adaptation_recommended": booleans["governance_adaptation_recommended"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "policy_memory_reliable": booleans["policy_memory_reliable"],
        "n_proposals": len(proposals),
        "n_recommendations": len(recommendations),
        "governance_confidence": ctx["governance_confidence"],
        "constitutional_pressure": ctx["constitutional_pressure"],
        "learning_state": ctx["learning_state"],
        "evolution_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance policy evolution engine (Step 39). "
            "Proposes policy changes from institutional learning. "
            "Never mutates runtime policy. No broker calls."
        ),
    )
    p.add_argument("--learning-summary", default=str(DEFAULT_LEARNING_SUMMARY))
    p.add_argument("--learning-record", default=str(DEFAULT_LEARNING_RECORD))
    p.add_argument("--learning-mem", default=str(DEFAULT_LEARNING_MEM))
    p.add_argument("--eff-summary", default=str(DEFAULT_EFF_SUMMARY))
    p.add_argument("--drift-summary", default=str(DEFAULT_DRIFT_SUMMARY))
    p.add_argument("--council-summary", default=str(DEFAULT_COUNCIL_SUMMARY))
    p.add_argument("--court-summary", default=str(DEFAULT_COURT_SUMMARY))
    p.add_argument("--runtime-policy", default=str(DEFAULT_RUNTIME_POLICY))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-mem-csv", default=str(DEFAULT_OUT_MEM_CSV))
    p.add_argument("--out-mem-parquet", default=str(DEFAULT_OUT_MEM_PQ))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[ARM_POLICY_EVOLUTION] starting "
        "(read-only policy evolution proposals; no runtime mutation; no broker calls)",
        flush=True,
    )

    learning_summary = _safe_read_json(
        Path(args.learning_summary), label="arm_governance_learning_summary.json"
    )
    learning_record = _safe_read_json(
        Path(args.learning_record), label="arm_governance_learning.json"
    )
    learning_mem = _safe_read_csv_rows(
        Path(args.learning_mem), label="arm_governance_learning_memory.csv"
    )
    eff_summary = _safe_read_json(
        Path(args.eff_summary), label="arm_governance_recovery_effectiveness_summary.json"
    )
    drift_summary = _safe_read_json(
        Path(args.drift_summary), label="arm_governance_drift_detection_summary.json"
    )
    council_summary = _safe_read_json(
        Path(args.council_summary), label="arm_supreme_governance_council_summary.json"
    )
    court_summary = _safe_read_json(
        Path(args.court_summary), label="arm_constitutional_court_summary.json"
    )
    runtime_policy = _safe_read_json(
        Path(args.runtime_policy), label="runtime_policy_governed.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_policy_evolution_memory.csv"
    )

    record, summary, md, merged_memory = build_policy_evolution(
        learning_summary=learning_summary,
        learning_record=learning_record,
        learning_mem=learning_mem,
        eff_summary=eff_summary,
        drift_summary=drift_summary,
        council_summary=council_summary,
        court_summary=court_summary,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
        existing_evolution_mem=existing_mem,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=EVOLUTION_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    booleans = record["governance_booleans"]
    print(
        "[ARM_POLICY_EVOLUTION] "
        f"state={record['evolution_state']} "
        f"score={record['evolution_score']:.3f} "
        f"proposals={len(record['policy_evolution_proposals'])} "
        f"adaptation={booleans['governance_adaptation_recommended']} "
        f"confidence={summary['governance_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_POLICY_EVOLUTION_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[ARM_POLICY_EVOLUTION_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_POLICY_EVOLUTION_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
