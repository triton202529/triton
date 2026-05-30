"""
ARM Governance Recovery Engine -- Step 36.

Reads:
    data/results/arm_governance_drift_detection_summary.json  (Step 35)
    data/results/arm_governance_drift_detection.json          (Step 35)
    data/results/arm_supreme_governance_council_summary.json  (Step 34)
    data/results/arm_constitutional_court_summary.json        (Step 33)
    data/results/arm_autonomy_constitution_summary.json       (Step 32)
    data/results/arm_shadow_performance_summary.json          (Step 30)
    data/results/arm_autonomy_graduation_summary.json         (Step 31)
    data/results/autonomous_governance_scorecard.json         (Step 19)
    data/results/autonomous_system_health_summary.json        (Step 20)
    data/results/autonomous_readiness_summary.json            (Step 21)
    data/results/runtime_policy_governed.json                   (Step 18)

Writes:
    data/results/arm_governance_recovery.json
    data/results/arm_governance_recovery.md
    data/results/arm_governance_recovery_summary.json
    data/results/arm_governance_recovery_memory.csv
    data/results/arm_governance_recovery_memory.parquet

Purpose
-------
This engine answers:

    "How should Triton recover from governance instability?"

It is Triton's governance recovery and stabilization layer -- read-only
recommendations for restoring institutional stability after deterioration
is detected. Recovery cannot override constitutional law, the constitutional
court, capital preservation, or operator supremacy.

Recovery state cascade
----------------------
    1. RECOVERY_LOCKDOWN       failure risk, constitutional crisis
    2. RECOVERY_INTENSIVE      unstable governance, elevated pressure
    3. RECOVERY_ACTIVE         drifting governance, corrective action
    4. RECOVERY_MONITORING     early warning, observation mode
    5. RECOVERY_NOT_REQUIRED   stable governance, no drift

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens are never written literally; an import-time
self-check raises if they ever appear.

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only recovery memory keyed by timestamp.
* Missing inputs warn-and-continue; absent evidence defaults to
  RECOVERY_INTENSIVE as the safe recovery posture.
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

DEFAULT_DRIFT_SUMMARY = RESULTS_DIR / "arm_governance_drift_detection_summary.json"
DEFAULT_DRIFT_RECORD = RESULTS_DIR / "arm_governance_drift_detection.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_CONST_SUMMARY = RESULTS_DIR / "arm_autonomy_constitution_summary.json"
DEFAULT_SHADOW_PERF = RESULTS_DIR / "arm_shadow_performance_summary.json"
DEFAULT_GRAD_SUMMARY = RESULTS_DIR / "arm_autonomy_graduation_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_recovery.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_recovery.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_recovery_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_recovery_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_recovery_memory.parquet"


# -----------------------------------------------------------
# Recovery state constants
# -----------------------------------------------------------
RECOVERY_NOT_REQUIRED = "RECOVERY_NOT_REQUIRED"
RECOVERY_MONITORING = "RECOVERY_MONITORING"
RECOVERY_ACTIVE = "RECOVERY_ACTIVE"
RECOVERY_INTENSIVE = "RECOVERY_INTENSIVE"
RECOVERY_LOCKDOWN = "RECOVERY_LOCKDOWN"

DRIFT_STABLE = "GOVERNANCE_STABLE"
DRIFT_EARLY_WARNING = "GOVERNANCE_EARLY_WARNING"
DRIFT_DRIFTING = "GOVERNANCE_DRIFTING"
DRIFT_UNSTABLE = "GOVERNANCE_UNSTABLE"
DRIFT_FAILURE_RISK = "GOVERNANCE_FAILURE_RISK"

COURT_LOCKDOWN = "COURT_LOCKDOWN"
COURT_OVERRULED = "COURT_OVERRULED"

CONSTITUTION_LOCKDOWN = "CONSTITUTION_LOCKDOWN"
CONSTITUTION_VIOLATED = "CONSTITUTION_VIOLATED"

COUNCIL_LOCKDOWN = "GOVERNANCE_LOCKDOWN"
COUNCIL_REVOKE = "GOVERNANCE_REVOKE_AUTONOMY"

PROHIBITED_HEALTH = frozenset({"STALE", "CRITICAL", "OFFLINE", "DEGRADED"})

RECOVERY_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "recovery_state",
    "recovery_confidence",
    "drift_state",
    "recovery_actions",
    "constitutional_pressure",
    "governance_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_RECOVERY_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(RECOVERY_MEMORY_COLUMNS))
        for col in ("recovery_confidence", "constitutional_pressure", "governance_confidence"):
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


# -----------------------------------------------------------
# Evidence extraction
# -----------------------------------------------------------
def _extract_evidence(
    *,
    drift_summary: Dict[str, Any],
    drift_record: Dict[str, Any],
    council_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    const_summary: Dict[str, Any],
    shadow_perf: Dict[str, Any],
    grad_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health: Dict[str, Any],
    readiness: Dict[str, Any],
    runtime_policy: Dict[str, Any],
) -> Dict[str, Any]:
    drift_state = _norm_upper(drift_summary.get("drift_state") or drift_record.get("drift_state"))
    drift_score = (
        _to_float(drift_summary.get("drift_score"))
        or _to_float(drift_record.get("drift_score"))
        or 0.0
    )

    constitutional_pressure = (
        _to_float(
            drift_summary.get("constitutional_pressure")
            or drift_record.get("drift_signals", {}).get("constitutional_pressure")
        )
        or 0.0
    )
    overruling_frequency = (
        _to_float(
            drift_summary.get("overruling_frequency")
            or drift_record.get("drift_signals", {}).get("overruling_frequency")
        )
        or 0.0
    )
    governance_confidence = (
        _to_float(
            drift_summary.get("governance_confidence") or council_summary.get("council_confidence")
        )
        or 0.0
    )

    court_ruling = _norm_upper(court_summary.get("judicial_ruling"))
    court_confidence = _to_float(court_summary.get("judicial_confidence")) or 0.0
    court_lockdown = bool(court_summary.get("lockdown_required"))
    court_override = bool(court_summary.get("constitutional_override_triggered"))

    constitution_state = _norm_upper(
        const_summary.get("constitution_state")
        or court_summary.get("constitution_state")
        or council_summary.get("constitution_state")
    )
    constitution_violated = bool(const_summary.get("constitution_violated"))
    constitution_lockdown = bool(const_summary.get("lockdown_required"))
    constitution_clear = bool(const_summary.get("constitution_clear"))

    council_ruling = _norm_upper(council_summary.get("governance_ruling"))
    graduation_state = _norm_upper(
        grad_summary.get("graduation_state") or court_summary.get("graduation_state")
    )

    scores = scorecard.get("scores") or {}
    governance_quality = _to_float(scores.get("governance_quality_score")) or 0.5
    health_score = _to_float(health.get("system_health_score")) or 0.5
    health_status = _norm_upper(health.get("overall_status"))
    readiness_score = _to_float(readiness.get("readiness_score")) or 0.5
    readiness_state = _norm_upper(readiness.get("readiness_state"))

    shadow_readiness = _to_float(shadow_perf.get("shadow_autonomy_readiness_score")) or 0.5
    shadow_discipline = _to_float(shadow_perf.get("shadow_discipline_score"))
    shadow_alpha = _to_float(shadow_perf.get("shadow_alpha_score")) or 0.5
    apprenticeship_verdict = _norm_upper(shadow_perf.get("apprenticeship_verdict"))
    trajectory_improving = bool(shadow_perf.get("trajectory_improving"))

    regime = _norm_upper(runtime_policy.get("regime") or scorecard.get("regime"))
    target_cash = _to_float(runtime_policy.get("target_cash_pct")) or 15.0
    confidence_threshold = _to_float(runtime_policy.get("confidence_threshold")) or 0.55
    max_position_pct = _to_float(runtime_policy.get("max_position_pct")) or 6.0
    skepticism_threshold = _to_float(runtime_policy.get("skepticism_threshold")) or 0.50

    drift_detected = bool(drift_summary.get("governance_drift_detected"))
    if not drift_summary and drift_state != DRIFT_STABLE:
        drift_detected = True

    return {
        "drift_state": drift_state,
        "drift_score": _clamp(drift_score, 0.0, 1.0),
        "governance_drift_detected": drift_detected,
        "constitutional_pressure": _clamp(constitutional_pressure, 0.0, 1.0),
        "overruling_frequency": _clamp(overruling_frequency, 0.0, 1.0),
        "governance_confidence": _clamp(governance_confidence, 0.0, 1.0),
        "court_ruling": court_ruling,
        "court_confidence": _clamp(court_confidence, 0.0, 1.0),
        "court_lockdown": court_lockdown,
        "court_override": court_override,
        "constitution_state": constitution_state,
        "constitution_violated": constitution_violated,
        "constitution_lockdown": constitution_lockdown,
        "constitution_clear": constitution_clear,
        "council_ruling": council_ruling,
        "graduation_state": graduation_state,
        "governance_quality": _clamp(governance_quality, 0.0, 1.0),
        "health_score": _clamp(health_score, 0.0, 1.0),
        "health_status": health_status,
        "readiness_score": _clamp(readiness_score, 0.0, 1.0),
        "readiness_state": readiness_state,
        "shadow_readiness": _clamp(shadow_readiness, 0.0, 1.0),
        "shadow_discipline": shadow_discipline,
        "shadow_alpha": _clamp(shadow_alpha, 0.0, 1.0),
        "trajectory_improving": trajectory_improving,
        "apprenticeship_verdict": apprenticeship_verdict,
        "regime": regime,
        "target_cash_pct": target_cash,
        "confidence_threshold": confidence_threshold,
        "max_position_pct": max_position_pct,
        "skepticism_threshold": skepticism_threshold,
        "autonomy_revoked": bool(grad_summary.get("autonomy_revoked")),
        "operator_review_required": bool(
            court_summary.get("operator_review_required")
            or const_summary.get("operator_override_required")
        ),
    }


# -----------------------------------------------------------
# Recovery classification
# -----------------------------------------------------------
def _classify_recovery_state(
    *,
    evidence: Dict[str, Any],
    have_evidence: bool,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not have_evidence:
        reasons.append("no upstream recovery evidence; defaulting to RECOVERY_INTENSIVE")
        return RECOVERY_INTENSIVE, reasons

    drift = evidence["drift_state"]
    ds = evidence["drift_score"]

    # ---- 1. RECOVERY_LOCKDOWN ----
    lockdown_triggers = (
        drift == DRIFT_FAILURE_RISK
        or evidence["court_lockdown"]
        or evidence["constitution_lockdown"]
        or evidence["council_ruling"] == COUNCIL_LOCKDOWN
        or evidence["court_ruling"] == COURT_LOCKDOWN
        or evidence["constitution_state"] == CONSTITUTION_LOCKDOWN
        or evidence["health_status"] in ("CRITICAL", "OFFLINE")
        or (drift == DRIFT_FAILURE_RISK and evidence["constitution_violated"])
    )
    if lockdown_triggers:
        if drift == DRIFT_FAILURE_RISK:
            reasons.append("governance drift escalated to failure risk")
        if evidence["constitution_violated"]:
            reasons.append("constitutional crisis active")
        if evidence["court_lockdown"] or evidence["constitution_lockdown"]:
            reasons.append("constitutional lockdown in effect")
        if not reasons:
            reasons.append("manual-only recovery required")
        return RECOVERY_LOCKDOWN, reasons

    # ---- 2. RECOVERY_INTENSIVE ----
    intensive_triggers = (
        drift == DRIFT_UNSTABLE
        or evidence["constitution_violated"]
        or evidence["constitutional_pressure"] >= 0.55
        or evidence["council_ruling"] == COUNCIL_REVOKE
        or evidence["graduation_state"] == "MANUAL_LOCKED"
        or (
            evidence["court_ruling"] == COURT_OVERRULED
            and drift in (DRIFT_UNSTABLE, DRIFT_FAILURE_RISK)
        )
    )
    if intensive_triggers:
        if evidence["constitutional_pressure"] >= 0.55:
            reasons.append(
                f"constitutional pressure elevated ({evidence['constitutional_pressure']:.2f})"
            )
        if evidence["constitution_violated"]:
            reasons.append(f"constitution_state={evidence['constitution_state']}")
        if evidence["graduation_state"] == "MANUAL_LOCKED":
            reasons.append("autonomy maturity weakened (MANUAL_LOCKED)")
        if not reasons:
            reasons.append("governance unstable; aggressive stabilization required")
        return RECOVERY_INTENSIVE, reasons

    # ---- 3. RECOVERY_ACTIVE ----
    active_triggers = (
        drift == DRIFT_DRIFTING
        or (ds >= 0.35 and drift not in (DRIFT_EARLY_WARNING, DRIFT_STABLE))
        or (evidence["overruling_frequency"] >= 0.5 and drift == DRIFT_DRIFTING)
    )
    if active_triggers:
        if drift == DRIFT_DRIFTING:
            reasons.append("governance drifting; corrective action required")
        if evidence["overruling_frequency"] >= 0.5:
            reasons.append("court overruling frequency elevated")
        if not reasons:
            reasons.append("autonomy must remain constrained during recovery")
        return RECOVERY_ACTIVE, reasons

    # ---- 4. RECOVERY_MONITORING ----
    monitoring_triggers = (
        drift == DRIFT_EARLY_WARNING
        or ds >= 0.15
        or evidence["health_status"] in PROHIBITED_HEALTH
        or evidence["readiness_score"] < 0.55
    )
    if monitoring_triggers:
        if drift == DRIFT_EARLY_WARNING:
            reasons.append("early warning active; observation mode")
        if evidence["health_status"] in PROHIBITED_HEALTH:
            reasons.append(f"system health {evidence['health_status']}")
        if not reasons:
            reasons.append("mild deterioration warrants recovery monitoring")
        return RECOVERY_MONITORING, reasons

    # ---- 5. RECOVERY_NOT_REQUIRED ----
    reasons.append("governance stable; no drift signals; healthy confidence")
    return RECOVERY_NOT_REQUIRED, reasons


# -----------------------------------------------------------
# Recovery actions
# -----------------------------------------------------------
def _recovery_actions(recovery_state: str, evidence: Dict[str, Any]) -> List[str]:
    """Additive, reversible, constitutional recovery recommendations."""
    actions: List[str] = []

    if recovery_state == RECOVERY_NOT_REQUIRED:
        actions.append("continue recovery monitoring")
        return actions

    # Observability (all non-stable states)
    if recovery_state in (
        RECOVERY_MONITORING,
        RECOVERY_ACTIVE,
        RECOVERY_INTENSIVE,
        RECOVERY_LOCKDOWN,
    ):
        actions.extend(
            [
                "increase monitoring frequency",
                "increase governance logging",
            ]
        )
    if recovery_state in (RECOVERY_ACTIVE, RECOVERY_INTENSIVE, RECOVERY_LOCKDOWN):
        actions.append("require additional diagnostics")

    # Governance tightening
    if recovery_state in (RECOVERY_ACTIVE, RECOVERY_INTENSIVE, RECOVERY_LOCKDOWN):
        actions.extend(
            [
                "raise confidence_threshold",
                "raise skepticism_threshold",
                "reduce deployment aggressiveness",
            ]
        )
    if recovery_state in (RECOVERY_INTENSIVE, RECOVERY_LOCKDOWN):
        actions.extend(
            [
                "reduce max_position_pct",
                "increase target_cash_pct",
            ]
        )

    # Autonomy stabilization
    if recovery_state in (RECOVERY_ACTIVE, RECOVERY_INTENSIVE, RECOVERY_LOCKDOWN):
        actions.extend(
            [
                "freeze autonomy promotion",
                "extend apprenticeship period",
                "disable autonomy escalation",
            ]
        )
    if recovery_state in (RECOVERY_INTENSIVE, RECOVERY_LOCKDOWN):
        actions.extend(
            [
                "force MANUAL mode",
                "require operator approval",
            ]
        )
    if recovery_state == RECOVERY_LOCKDOWN:
        actions.extend(
            [
                "halt all governance relaxation",
                "enforce manual-only operation",
                "refresh stale intelligence pipeline",
            ]
        )

    if recovery_state == RECOVERY_MONITORING:
        actions.append("maintain elevated observation posture")

    # dedupe preserving order
    seen: set = set()
    out: List[str] = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


# -----------------------------------------------------------
# Recovery goals
# -----------------------------------------------------------
def _recovery_goals(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    shadow_disc = evidence["shadow_discipline"]
    shadow_target = 0.60 if shadow_disc is None else min(0.75, max(0.55, shadow_disc + 0.10))

    return [
        {
            "objective": "reduce constitutional pressure",
            "current": evidence["constitutional_pressure"],
            "target": _clamp(evidence["constitutional_pressure"] * 0.5, 0.0, 0.35),
            "priority": "high" if evidence["constitutional_pressure"] >= 0.5 else "medium",
        },
        {
            "objective": "reduce overruling frequency",
            "current": evidence["overruling_frequency"],
            "target": _clamp(evidence["overruling_frequency"] * 0.25, 0.0, 0.25),
            "priority": "high" if evidence["overruling_frequency"] >= 0.5 else "low",
        },
        {
            "objective": "improve governance confidence",
            "current": evidence["governance_confidence"],
            "target": min(0.75, evidence["governance_confidence"] + 0.25),
            "priority": "high",
        },
        {
            "objective": "improve shadow discipline",
            "current": shadow_disc if shadow_disc is not None else evidence["shadow_readiness"],
            "target": shadow_target,
            "priority": "medium",
        },
        {
            "objective": "restore readiness stability",
            "current": evidence["readiness_score"],
            "target": min(0.80, evidence["readiness_score"] + 0.20),
            "priority": "high" if evidence["readiness_score"] < 0.55 else "medium",
        },
        {
            "objective": "rebuild autonomy maturity",
            "current": evidence["graduation_state"],
            "target": "ASSISTED_APPROVED",
            "priority": "medium" if evidence["graduation_state"] == "MANUAL_LOCKED" else "low",
        },
    ]


# -----------------------------------------------------------
# Booleans and confidence
# -----------------------------------------------------------
def _recovery_booleans(recovery_state: str, evidence: Dict[str, Any]) -> Dict[str, bool]:
    stabilization = recovery_state != RECOVERY_NOT_REQUIRED
    intensive_plus = recovery_state in (RECOVERY_INTENSIVE, RECOVERY_LOCKDOWN)
    active_plus = recovery_state in (RECOVERY_ACTIVE, RECOVERY_INTENSIVE, RECOVERY_LOCKDOWN)
    return {
        "recovery_required": stabilization,
        "governance_stabilization_active": stabilization,
        "autonomy_freeze_required": active_plus or evidence["autonomy_revoked"],
        "manual_supervision_required": intensive_plus
        or evidence["graduation_state"] == "MANUAL_LOCKED",
        "operator_escalation_required": intensive_plus or evidence["operator_review_required"],
        "lockdown_required": recovery_state == RECOVERY_LOCKDOWN,
    }


def _recovery_confidence(
    evidence: Dict[str, Any],
    recovery_state: str,
) -> float:
    """Higher = more confidence recovery will succeed."""
    base_parts: List[float] = [
        evidence["governance_confidence"] * 0.25,
        evidence["readiness_score"] * 0.20,
        evidence["health_score"] * 0.15,
        evidence["shadow_readiness"] * 0.15,
        evidence["governance_quality"] * 0.10,
    ]
    if evidence["shadow_discipline"] is not None:
        base_parts.append(_clamp(evidence["shadow_discipline"], 0.0, 1.0) * 0.15)
    else:
        base_parts.append(evidence["shadow_alpha"] * 0.10)

    base = sum(base_parts)

    penalty = (
        evidence["drift_score"] * 0.30
        + evidence["constitutional_pressure"] * 0.25
        + evidence["overruling_frequency"] * 0.15
    )
    if evidence["court_ruling"] == COURT_OVERRULED:
        penalty += 0.15
    if recovery_state == RECOVERY_LOCKDOWN:
        penalty += 0.20
    if evidence["constitution_violated"]:
        penalty += 0.10
    if not evidence["trajectory_improving"]:
        penalty += 0.05

    return round(_clamp(base - penalty, 0.0, 1.0), 6)


# -----------------------------------------------------------
# Recommendations and rationale
# -----------------------------------------------------------
def _recommendations(
    recovery_state: str,
    booleans: Dict[str, bool],
) -> List[str]:
    recs: List[str] = []
    if booleans["autonomy_freeze_required"]:
        recs.append("Freeze autonomy escalation")
    if recovery_state in (RECOVERY_INTENSIVE, RECOVERY_LOCKDOWN):
        recs.append("Increase capital preservation posture")
    if booleans["manual_supervision_required"]:
        recs.append("Maintain manual supervision")
    if recovery_state in (RECOVERY_ACTIVE, RECOVERY_INTENSIVE, RECOVERY_LOCKDOWN):
        recs.append("Extend apprenticeship period")
    if recovery_state == RECOVERY_MONITORING:
        recs.append("Continue recovery monitoring")
    if recovery_state != RECOVERY_NOT_REQUIRED:
        recs.append("Reassess after governance stabilization")
    if recovery_state == RECOVERY_NOT_REQUIRED:
        recs.append("Continue recovery monitoring")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(
    recovery_state: str,
    reasons: List[str],
    evidence: Dict[str, Any],
) -> str:
    if recovery_state == RECOVERY_NOT_REQUIRED:
        return (
            "Recovery not required because governance is stable, "
            "drift signals are absent, and confidence remains healthy."
        )
    if recovery_state == RECOVERY_LOCKDOWN and evidence["drift_state"] == DRIFT_FAILURE_RISK:
        return (
            "Recovery intensified because governance drift escalated to failure risk, "
            "constitutional pressure increased, and autonomy maturity weakened."
        )
    parts = reasons[:3] if reasons else ["governance instability detected"]
    return f"Recovery {'activated' if recovery_state == RECOVERY_ACTIVE else 'adjusted'} because {'; '.join(parts)}."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    recovery_state: str,
    recovery_confidence: float,
    evidence: Dict[str, Any],
    actions: List[str],
    goals: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
) -> str:
    lines = [
        "# Triton Governance Recovery",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Recovery State",
        "",
        f"**{recovery_state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| recovery_confidence | {recovery_confidence:.3f} |",
        f"| drift_state | {evidence['drift_state']} |",
        f"| drift_score | {evidence['drift_score']:.3f} |",
        f"| recovery_required | {booleans['recovery_required']} |",
        f"| governance_stabilization_active | {booleans['governance_stabilization_active']} |",
        f"| autonomy_freeze_required | {booleans['autonomy_freeze_required']} |",
        f"| manual_supervision_required | {booleans['manual_supervision_required']} |",
        f"| operator_escalation_required | {booleans['operator_escalation_required']} |",
        f"| lockdown_required | {booleans['lockdown_required']} |",
        "",
        "## Recovery Actions",
        "",
    ]
    for a in actions:
        lines.append(f"- {a}")
    lines.extend(
        [
            "",
            "## Recovery Goals",
            "",
            "| objective | current | target | priority |",
            "|---|---|---|---|",
        ]
    )
    for g in goals:
        cur = g["current"]
        tgt = g["target"]
        if isinstance(cur, float):
            cur_s = f"{cur:.3f}"
        else:
            cur_s = str(cur)
        if isinstance(tgt, float):
            tgt_s = f"{tgt:.3f}"
        else:
            tgt_s = str(tgt)
        lines.append(f"| {g['objective']} | {cur_s} | {tgt_s} | {g['priority']} |")
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
            f"Recovery confidence {recovery_confidence:.2f} reflects governance confidence, "
            f"drift score ({evidence['drift_score']:.2f}), constitutional pressure "
            f"({evidence['constitutional_pressure']:.2f}), readiness, health, and shadow "
            f"discipline. Recovery actions are additive, reversible, and cannot override "
            f"constitutional law or operator supremacy.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Recovery memory
# -----------------------------------------------------------
def _build_recovery_row(
    *,
    timestamp: str,
    recovery_state: str,
    recovery_confidence: float,
    evidence: Dict[str, Any],
    actions: List[str],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "recovery_state": recovery_state,
        "recovery_confidence": round(recovery_confidence, 6),
        "drift_state": evidence["drift_state"],
        "recovery_actions": "|".join(actions),
        "constitutional_pressure": evidence["constitutional_pressure"],
        "governance_confidence": evidence["governance_confidence"],
        "rationale": rationale,
    }


def _merge_recovery_memory(
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
        for c in RECOVERY_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_recovery_plan(
    *,
    drift_summary: Dict[str, Any],
    drift_record: Dict[str, Any],
    council_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    const_summary: Dict[str, Any],
    shadow_perf: Dict[str, Any],
    grad_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    health: Dict[str, Any],
    readiness: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    existing_memory_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    evidence = _extract_evidence(
        drift_summary=drift_summary,
        drift_record=drift_record,
        council_summary=council_summary,
        court_summary=court_summary,
        const_summary=const_summary,
        shadow_perf=shadow_perf,
        grad_summary=grad_summary,
        scorecard=scorecard,
        health=health,
        readiness=readiness,
        runtime_policy=runtime_policy,
    )

    have_evidence = any(
        bool(x)
        for x in (
            drift_summary,
            drift_record,
            council_summary,
            court_summary,
            const_summary,
            health,
            readiness,
        )
    )

    recovery_state, reasons = _classify_recovery_state(
        evidence=evidence,
        have_evidence=have_evidence,
    )

    actions = _recovery_actions(recovery_state, evidence)
    goals = _recovery_goals(evidence)
    booleans = _recovery_booleans(recovery_state, evidence)
    recovery_confidence = _recovery_confidence(evidence, recovery_state)
    recommendations = _recommendations(recovery_state, booleans)
    rationale = _build_rationale(recovery_state, reasons, evidence)

    recovery_row = _build_recovery_row(
        timestamp=timestamp,
        recovery_state=recovery_state,
        recovery_confidence=recovery_confidence,
        evidence=evidence,
        actions=actions,
        rationale=rationale,
    )
    merged_memory = _merge_recovery_memory(existing_memory_rows, recovery_row)

    md = _render_markdown(
        generated_at=timestamp,
        recovery_state=recovery_state,
        recovery_confidence=recovery_confidence,
        evidence=evidence,
        actions=actions,
        goals=goals,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_recovery_engine",
        "engine_version": 1,
        "recovery_state": recovery_state,
        "recovery_confidence": recovery_confidence,
        "recovery_reasons": reasons,
        "evidence": evidence,
        "recovery_actions": actions,
        "recovery_goals": goals,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "constitutional_supremacy_note": (
            "Recovery recommendations are additive and reversible. "
            "Recovery cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "recovery_memory_size_after_append": len(merged_memory),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "recovery_only": True,
        },
        "inputs_seen": {
            "arm_governance_drift_detection_summary": bool(drift_summary),
            "arm_governance_drift_detection_record": bool(drift_record),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_autonomy_constitution_summary": bool(const_summary),
            "arm_shadow_performance_summary": bool(shadow_perf),
            "arm_autonomy_graduation_summary": bool(grad_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health),
            "autonomous_readiness_summary": bool(readiness),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_recovery_memory_rows": len(existing_memory_rows),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_recovery_engine",
        "recovery_state": recovery_state,
        "recovery_confidence": recovery_confidence,
        "drift_state": evidence["drift_state"],
        "drift_score": evidence["drift_score"],
        "recovery_required": booleans["recovery_required"],
        "governance_stabilization_active": booleans["governance_stabilization_active"],
        "autonomy_freeze_required": booleans["autonomy_freeze_required"],
        "manual_supervision_required": booleans["manual_supervision_required"],
        "operator_escalation_required": booleans["operator_escalation_required"],
        "lockdown_required": booleans["lockdown_required"],
        "constitutional_pressure": evidence["constitutional_pressure"],
        "governance_confidence": evidence["governance_confidence"],
        "recovery_actions": actions,
        "n_recovery_goals": len(goals),
        "n_recommendations": len(recommendations),
        "recovery_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance recovery engine (Step 36). "
            "Plans stabilization after governance deterioration. "
            "No broker calls."
        ),
    )
    p.add_argument("--drift-summary", default=str(DEFAULT_DRIFT_SUMMARY))
    p.add_argument("--drift-record", default=str(DEFAULT_DRIFT_RECORD))
    p.add_argument("--council-summary", default=str(DEFAULT_COUNCIL_SUMMARY))
    p.add_argument("--court-summary", default=str(DEFAULT_COURT_SUMMARY))
    p.add_argument("--const-summary", default=str(DEFAULT_CONST_SUMMARY))
    p.add_argument("--shadow-perf", default=str(DEFAULT_SHADOW_PERF))
    p.add_argument("--grad-summary", default=str(DEFAULT_GRAD_SUMMARY))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--health", default=str(DEFAULT_HEALTH))
    p.add_argument("--readiness", default=str(DEFAULT_READINESS))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-mem-csv", default=str(DEFAULT_OUT_MEM_CSV))
    p.add_argument("--out-mem-parquet", default=str(DEFAULT_OUT_MEM_PQ))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[ARM_RECOVERY] starting (read-only governance recovery; no broker calls)",
        flush=True,
    )

    drift_summary = _safe_read_json(
        Path(args.drift_summary), label="arm_governance_drift_detection_summary.json"
    )
    drift_record = _safe_read_json(
        Path(args.drift_record), label="arm_governance_drift_detection.json"
    )
    council_summary = _safe_read_json(
        Path(args.council_summary), label="arm_supreme_governance_council_summary.json"
    )
    court_summary = _safe_read_json(
        Path(args.court_summary), label="arm_constitutional_court_summary.json"
    )
    const_summary = _safe_read_json(
        Path(args.const_summary), label="arm_autonomy_constitution_summary.json"
    )
    shadow_perf = _safe_read_json(
        Path(args.shadow_perf), label="arm_shadow_performance_summary.json"
    )
    grad_summary = _safe_read_json(
        Path(args.grad_summary), label="arm_autonomy_graduation_summary.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    health = _safe_read_json(Path(args.health), label="autonomous_system_health_summary.json")
    readiness = _safe_read_json(Path(args.readiness), label="autonomous_readiness_summary.json")
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_recovery_memory.csv"
    )

    record, summary, md, merged_memory = build_recovery_plan(
        drift_summary=drift_summary,
        drift_record=drift_record,
        council_summary=council_summary,
        court_summary=court_summary,
        const_summary=const_summary,
        shadow_perf=shadow_perf,
        grad_summary=grad_summary,
        scorecard=scorecard,
        health=health,
        readiness=readiness,
        runtime_policy=runtime_policy,
        existing_memory_rows=existing_mem,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=RECOVERY_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    booleans = record["governance_booleans"]
    print(
        "[ARM_RECOVERY] "
        f"state={record['recovery_state']} "
        f"required={booleans['recovery_required']} "
        f"freeze={booleans['autonomy_freeze_required']} "
        f"lockdown={booleans['lockdown_required']} "
        f"confidence={record['recovery_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_RECOVERY_SAFETY] broker_calls=0 orders_placed=0 portfolio_mutated=False",
        flush=True,
    )
    print(
        f"[ARM_RECOVERY_OUT] json={Path(args.out_json).as_posix()} "
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
            raise RuntimeError(f"[ARM_RECOVERY_SAFETY] forbidden broker token detected: {tok!r}")


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
