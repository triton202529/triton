"""
ARM Supreme Governance Council Engine -- Step 34.

Reads:
    data/results/arm_constitutional_court_summary.json      (Step 33)
    data/results/arm_constitutional_court.json            (Step 33)
    data/results/arm_autonomy_constitution_summary.json   (Step 32)
    data/results/arm_autonomy_graduation_summary.json     (Step 31)
    data/results/arm_shadow_performance_summary.json      (Step 30)
    data/results/autonomous_governance_scorecard.json     (Step 19)
    data/results/autonomous_system_health_summary.json    (Step 20)
    data/results/autonomous_readiness_summary.json        (Step 21)
    data/results/runtime_policy_governed.json             (Step 18)
    data/results/meta_decision_intelligence.json          (Step 13)

Writes:
    data/results/arm_supreme_governance_council.json
    data/results/arm_supreme_governance_council.md
    data/results/arm_supreme_governance_council_summary.json
    data/results/arm_governance_posture_memory.csv
    data/results/arm_governance_posture_memory.parquet

Purpose
-------
This engine answers:

    "Should Triton change its governance posture?"

It is Triton's meta-governance layer -- a read-only council that
evaluates whether governance posture should tighten, maintain, loosen,
or revoke autonomy. The council may *recommend* governance changes but
CANNOT override constitutional law or the constitutional court.

Council precedence (must respect, cannot violate)
-------------------------------------------------
    1. Constitutional Court
    2. Constitution
    3. Capital Preservation Doctrine
    4. Operator supremacy
    5. Risk Committee
    6. Execution Certificate
    7. ARM Governance

Posture ruling cascade
----------------------
    1. GOVERNANCE_LOCKDOWN           court/constitution lockdown, collapse
    2. GOVERNANCE_REVOKE_AUTONOMY    deterioration, constitutional pressure
    3. GOVERNANCE_TIGHTEN            uncertainty, weak shadow, defensive
    4. GOVERNANCE_LOOSEN             strong apprenticeship, stable governance
    5. GOVERNANCE_MAINTAIN           stable, no deterioration

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens are never written literally; an import-time
self-check raises if they ever appear.

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only posture memory keyed by timestamp_utc.
* Missing inputs warn-and-continue; absent evidence defaults to
  GOVERNANCE_LOCKDOWN as the safe council posture.
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

DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_COURT_RECORD = RESULTS_DIR / "arm_constitutional_court.json"
DEFAULT_CONST_SUMMARY = RESULTS_DIR / "arm_autonomy_constitution_summary.json"
DEFAULT_GRAD_SUMMARY = RESULTS_DIR / "arm_autonomy_graduation_summary.json"
DEFAULT_SHADOW_PERF = RESULTS_DIR / "arm_shadow_performance_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_META_DECISION = RESULTS_DIR / "meta_decision_intelligence.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_supreme_governance_council.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_supreme_governance_council.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_posture_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_posture_memory.parquet"


# -----------------------------------------------------------
# Ruling constants
# -----------------------------------------------------------
RULING_TIGHTEN = "GOVERNANCE_TIGHTEN"
RULING_MAINTAIN = "GOVERNANCE_MAINTAIN"
RULING_LOOSEN = "GOVERNANCE_LOOSEN"
RULING_REVOKE = "GOVERNANCE_REVOKE_AUTONOMY"
RULING_LOCKDOWN = "GOVERNANCE_LOCKDOWN"

COURT_LOCKDOWN = "COURT_LOCKDOWN"
COURT_OVERRULED = "COURT_OVERRULED"

CONSTITUTION_LOCKDOWN = "CONSTITUTION_LOCKDOWN"
CONSTITUTION_VIOLATED = "CONSTITUTION_VIOLATED"
CONSTITUTION_DEFENSIVE = "CONSTITUTION_DEFENSIVE"
CONSTITUTION_CLEAR = "CONSTITUTION_CLEAR"

DEFENSIVE_REGIMES = frozenset({"DEFENSIVE", "HIGH_VOLATILITY", "RISK_OFF"})
PROHIBITED_HEALTH = frozenset({"STALE", "CRITICAL", "OFFLINE"})

# Apprenticeship verdicts that permit loosening
LOOSEN_VERDICTS = frozenset({"TRUST_BUILDING", "AUTONOMY_CANDIDATE"})
WEAK_VERDICTS = frozenset({"LEARNING", "AUTONOMY_NOT_READY", "IMPROVING"})

COUNCIL_PRECEDENCE: Tuple[str, ...] = (
    "constitutional_court",
    "constitution",
    "capital_preservation",
    "operator_supremacy",
    "risk_committee",
    "execution_certificate",
    "arm_governance",
)

POSTURE_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp_utc",
    "ruling",
    "governance_state",
    "confidence",
    "recommended_actions",
    "autonomy_state",
    "constitutional_state",
    "court_ruling",
    "apprenticeship_verdict",
    "system_health_status",
    "regime",
    "rationale",
    "governance_change_recommended",
    "lockdown_required",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_COUNCIL_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(POSTURE_MEMORY_COLUMNS))
        for col in ("confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("governance_change_recommended", "lockdown_required"):
            if col in df.columns:
                df[col] = df[col].map(_to_bool_optional)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return True
    except Exception as e:
        _warn(f"parquet write failed for {path}: {type(e).__name__}: {e}")
        return False


def _to_bool_optional(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("", "nan", "none", "null"):
        return None
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


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


def _to_int(x: Any) -> Optional[int]:
    v = _to_float(x)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


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
    court_summary: Dict[str, Any],
    court_record: Dict[str, Any],
    const_summary: Dict[str, Any],
    grad_summary: Dict[str, Any],
    shadow_perf: Dict[str, Any],
    scorecard: Dict[str, Any],
    health: Dict[str, Any],
    readiness: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    meta_decision: Dict[str, Any],
) -> Dict[str, Any]:
    court_ruling = _norm_upper(
        court_summary.get("judicial_ruling") or court_record.get("judicial_ruling")
    )
    court_confidence = (
        _to_float(
            court_summary.get("judicial_confidence") or court_record.get("judicial_confidence")
        )
        or 0.0
    )
    court_lockdown = bool(court_summary.get("lockdown_required"))
    court_override = bool(court_summary.get("constitutional_override_triggered"))

    const_state = _norm_upper(
        const_summary.get("constitution_state") or court_summary.get("constitution_state")
    )
    const_confidence = _to_float(const_summary.get("confidence")) or 0.0
    const_severity = _norm_upper(const_summary.get("severity"))
    const_clear = bool(const_summary.get("constitution_clear"))
    const_violated = bool(const_summary.get("constitution_violated"))
    const_lockdown = bool(const_summary.get("lockdown_required"))
    const_defensive = bool(const_summary.get("defensive_constraints_required"))

    grad_state = _norm_upper(
        grad_summary.get("graduation_state") or court_summary.get("graduation_state")
    )
    autonomy_revoked = bool(grad_summary.get("autonomy_revoked"))
    autonomy_earned = bool(grad_summary.get("autonomy_promotion_earned"))
    auto_approved = bool(grad_summary.get("auto_mode_approved"))
    grad_confidence = _to_float(grad_summary.get("confidence")) or 0.0

    apprenticeship = _norm_upper(shadow_perf.get("apprenticeship_verdict"))
    shadow_readiness = _to_float(shadow_perf.get("shadow_autonomy_readiness_score")) or 0.0
    shadow_discipline = _to_float(shadow_perf.get("shadow_discipline_score"))
    shadow_alpha = _to_float(shadow_perf.get("shadow_alpha_score")) or 0.5
    shadow_observations = _to_int(shadow_perf.get("n_labelled")) or 0
    trajectory_improving = bool(shadow_perf.get("trajectory_improving"))

    governance_quality = (
        _to_float(
            scorecard.get("governance_quality_score")
            or scorecard.get("intelligence_health_score")
            or grad_summary.get("governance_quality")
        )
        or 0.5
    )
    governance_state = str(scorecard.get("system_state") or "")

    health_status = _norm_upper(
        health.get("overall_status") or const_summary.get("system_health_status")
    )
    health_score = _to_float(health.get("system_health_score"))
    if health_score is None:
        health_score = {
            "HEALTHY": 0.90,
            "DEGRADED": 0.55,
            "STALE": 0.35,
            "CRITICAL": 0.10,
            "OFFLINE": 0.00,
        }.get(health_status, 0.50)

    readiness_state = _norm_upper(readiness.get("readiness_state"))
    readiness_score = _to_float(readiness.get("readiness_score")) or 0.5

    regime = _norm_upper((runtime_policy or {}).get("regime"))
    target_cash = _to_float((runtime_policy or {}).get("target_cash_pct"))
    confidence_threshold = _to_float((runtime_policy or {}).get("confidence_threshold"))
    max_position = _to_float((runtime_policy or {}).get("max_position_pct"))

    trust_level = _norm_upper((meta_decision or {}).get("trust_level"))
    self_confidence = _to_float((meta_decision or {}).get("self_confidence_score")) or 0.5

    return {
        "court_ruling": court_ruling,
        "court_confidence": _clamp(court_confidence, 0.0, 1.0),
        "court_lockdown": court_lockdown,
        "court_override": court_override,
        "constitution_state": const_state,
        "constitution_confidence": _clamp(const_confidence, 0.0, 1.0),
        "constitution_severity": const_severity,
        "constitution_clear": const_clear,
        "constitution_violated": const_violated,
        "constitution_lockdown": const_lockdown,
        "defensive_constraints": const_defensive,
        "graduation_state": grad_state,
        "autonomy_revoked": autonomy_revoked,
        "autonomy_promotion_earned": autonomy_earned,
        "auto_mode_approved": auto_approved,
        "graduation_confidence": _clamp(grad_confidence, 0.0, 1.0),
        "apprenticeship_verdict": apprenticeship,
        "shadow_autonomy_readiness": _clamp(shadow_readiness, 0.0, 1.0),
        "shadow_discipline_score": shadow_discipline,
        "shadow_alpha_score": _clamp(shadow_alpha, 0.0, 1.0),
        "shadow_observations": shadow_observations,
        "trajectory_improving": trajectory_improving,
        "governance_quality": _clamp(governance_quality, 0.0, 1.0),
        "governance_state": governance_state,
        "system_health_status": health_status,
        "system_health_score": _clamp(health_score, 0.0, 1.0),
        "readiness_state": readiness_state,
        "readiness_score": _clamp(readiness_score, 0.0, 1.0),
        "regime": regime,
        "target_cash_pct": target_cash,
        "confidence_threshold": confidence_threshold,
        "max_position_pct": max_position,
        "trust_level": trust_level,
        "self_confidence_score": _clamp(self_confidence, 0.0, 1.0),
    }


# -----------------------------------------------------------
# Deterioration signals
# -----------------------------------------------------------
def _deterioration_signals(
    evidence: Dict[str, Any],
    memory_rows: List[Dict[str, str]],
) -> List[str]:
    signals: List[str] = []
    if evidence["autonomy_revoked"]:
        signals.append("autonomy_revoked=True")
    if evidence["court_override"]:
        signals.append("constitutional_court_override=True")
    if evidence["constitution_violated"]:
        signals.append(f"constitution_violated (state={evidence['constitution_state']})")
    if evidence["apprenticeship_verdict"] == "AUTONOMY_NOT_READY":
        signals.append("apprenticeship_verdict=AUTONOMY_NOT_READY")
    if (
        evidence["shadow_discipline_score"] is not None
        and evidence["shadow_discipline_score"] < 0.45
    ):
        signals.append(f"shadow_discipline={evidence['shadow_discipline_score']:.2f} collapsed")
    if evidence["governance_quality"] < 0.40:
        signals.append(f"governance_quality={evidence['governance_quality']:.2f} unstable")
    if evidence["court_confidence"] < 0.35:
        signals.append(f"court_confidence={evidence['court_confidence']:.2f} collapsed")

    # Compare to prior posture memory if available
    if memory_rows:
        sorted_rows = sorted(memory_rows, key=lambda r: str(r.get("timestamp_utc", "")))
        last = sorted_rows[-1]
        last_ruling = str(last.get("ruling", ""))
        if last_ruling in (RULING_LOOSEN, RULING_MAINTAIN) and (
            evidence["constitution_violated"] or evidence["court_override"]
        ):
            signals.append(f"prior posture was {last_ruling} but constitutional pressure elevated")
    return signals


# -----------------------------------------------------------
# Ruling classification
# -----------------------------------------------------------
def _classify_ruling(
    *,
    evidence: Dict[str, Any],
    deterioration: List[str],
    have_evidence: bool,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not have_evidence:
        reasons.append("no upstream council evidence; defaulting to LOCKDOWN")
        return RULING_LOCKDOWN, reasons

    # ---- 1. GOVERNANCE_LOCKDOWN ----
    if (
        evidence["court_lockdown"]
        or evidence["constitution_lockdown"]
        or evidence["court_ruling"] == COURT_LOCKDOWN
        or evidence["constitution_state"] == CONSTITUTION_LOCKDOWN
        or evidence["system_health_status"] in ("CRITICAL", "OFFLINE")
        or evidence["governance_quality"] < 0.25
    ):
        if evidence["court_lockdown"]:
            reasons.append("constitutional court lockdown_required=True")
        if evidence["constitution_lockdown"]:
            reasons.append("constitution lockdown_required=True")
        if evidence["system_health_status"] in ("CRITICAL", "OFFLINE"):
            reasons.append(f"system_health={evidence['system_health_status']}")
        if evidence["governance_quality"] < 0.25:
            reasons.append(f"governance collapse (quality={evidence['governance_quality']:.2f})")
        return RULING_LOCKDOWN, reasons

    # ---- 2. GOVERNANCE_REVOKE_AUTONOMY ----
    revoke_triggers = (
        len(deterioration) >= 2
        or evidence["autonomy_revoked"]
        or (evidence["court_ruling"] == COURT_OVERRULED and evidence["constitution_violated"])
        or evidence["apprenticeship_verdict"] == "AUTONOMY_NOT_READY"
        or (evidence["shadow_autonomy_readiness"] < 0.40 and evidence["shadow_observations"] >= 10)
    )
    if revoke_triggers:
        reasons.extend(deterioration[:3] if deterioration else [])
        if evidence["court_ruling"] == COURT_OVERRULED:
            reasons.append(f"court_ruling={COURT_OVERRULED}")
        if not reasons:
            reasons.append("governance deterioration detected")
        return RULING_REVOKE, reasons

    # ---- 3. GOVERNANCE_TIGHTEN ----
    tighten_triggers = (
        evidence["constitution_violated"]
        or evidence["defensive_constraints"]
        or evidence["constitution_state"] in (CONSTITUTION_VIOLATED, CONSTITUTION_DEFENSIVE)
        or evidence["regime"] in DEFENSIVE_REGIMES
        or evidence["system_health_status"] in PROHIBITED_HEALTH
        or evidence["apprenticeship_verdict"] in WEAK_VERDICTS
        or evidence["shadow_observations"] < 25
        or evidence["governance_quality"] < 0.55
        or evidence["trust_level"] in ("VERY_LOW", "LOW")
        or evidence["court_ruling"] in (COURT_OVERRULED, "COURT_ESCALATED")
    )
    if tighten_triggers:
        if evidence["constitution_violated"]:
            reasons.append(f"constitutional pressure (state={evidence['constitution_state']})")
        if evidence["regime"] in DEFENSIVE_REGIMES:
            reasons.append(f"defensive regime={evidence['regime']}")
        if evidence["apprenticeship_verdict"] in WEAK_VERDICTS:
            reasons.append(f"immature apprenticeship ({evidence['apprenticeship_verdict']})")
        if evidence["system_health_status"] in PROHIBITED_HEALTH:
            reasons.append(f"system_health={evidence['system_health_status']}")
        if not reasons:
            reasons.append("elevated uncertainty warrants tighter governance")
        return RULING_TIGHTEN, reasons

    # ---- 4. GOVERNANCE_LOOSEN ----
    loosen_ok = (
        evidence["constitution_clear"]
        and evidence["court_ruling"] in ("COURT_APPROVED", "COURT_APPROVED_LIMITED")
        and not evidence["court_override"]
        and evidence["apprenticeship_verdict"] in LOOSEN_VERDICTS
        and evidence["shadow_autonomy_readiness"] >= 0.60
        and evidence["governance_quality"] >= 0.60
        and evidence["system_health_status"] == "HEALTHY"
        and evidence["shadow_observations"] >= 30
        and (
            evidence["trajectory_improving"]
            or evidence["shadow_discipline_score"] is None
            or evidence["shadow_discipline_score"] >= 0.55
        )
    )
    if loosen_ok:
        reasons.append(
            f"strong apprenticeship ({evidence['apprenticeship_verdict']}), "
            f"constitutional stability, governance_quality="
            f"{evidence['governance_quality']:.2f}"
        )
        return RULING_LOOSEN, reasons

    # ---- 5. GOVERNANCE_MAINTAIN ----
    reasons.append(
        "stable governance posture; no material deterioration or " "improvement warranting change"
    )
    return RULING_MAINTAIN, reasons


# -----------------------------------------------------------
# Governance actions
# -----------------------------------------------------------
def _governance_actions(ruling: str, evidence: Dict[str, Any]) -> List[str]:
    actions: List[str] = []
    if ruling == RULING_LOCKDOWN:
        actions.extend(
            [
                "halt all governance relaxation",
                "enforce manual-only operation",
                "refresh stale intelligence pipeline",
            ]
        )
        return actions
    if ruling == RULING_REVOKE:
        actions.extend(
            [
                "freeze autonomy promotion",
                "revoke autonomy escalation",
                "extend shadow apprenticeship",
                "increase target_cash_pct",
            ]
        )
        return actions
    if ruling == RULING_TIGHTEN:
        actions.append("raise confidence_threshold")
        actions.append("increase target_cash_pct")
        actions.append("reduce deployment aggressiveness")
        actions.append("extend shadow apprenticeship")
        if evidence["autonomy_promotion_earned"]:
            actions.append("freeze autonomy promotion")
        if evidence["system_health_status"] in PROHIBITED_HEALTH:
            actions.append("require pipeline refresh before governance review")
        return actions
    if ruling == RULING_LOOSEN:
        actions.append("permit assisted autonomy trials")
        actions.append("maintain constitutional floor")
        if evidence["auto_mode_approved"]:
            actions.append("consider selective governance relaxation")
        else:
            actions.append("permit assisted-mode observation only")
        return actions
    actions.append("maintain current governance")
    return actions


# -----------------------------------------------------------
# Governance booleans
# -----------------------------------------------------------
def _governance_booleans(ruling: str, evidence: Dict[str, Any]) -> Dict[str, bool]:
    lockdown = ruling == RULING_LOCKDOWN
    revoke = ruling == RULING_REVOKE
    tighten = ruling == RULING_TIGHTEN
    loosen = ruling == RULING_LOOSEN
    return {
        "governance_change_recommended": ruling != RULING_MAINTAIN,
        "governance_tightening_required": tighten or revoke or lockdown,
        "governance_relaxation_allowed": loosen and evidence["constitution_clear"],
        "autonomy_revocation_required": revoke or lockdown,
        "operator_supervision_required": (
            lockdown
            or revoke
            or tighten
            or evidence["court_override"]
            or not evidence["auto_mode_approved"]
        ),
        "lockdown_required": lockdown,
    }


# -----------------------------------------------------------
# Council confidence
# -----------------------------------------------------------
def _council_confidence(
    evidence: Dict[str, Any],
    ruling: str,
    deterioration: List[str],
) -> float:
    shadow_perf = evidence["shadow_autonomy_readiness"]
    if evidence["shadow_observations"] < 10:
        shadow_perf = 0.5  # neutral when immature

    base = (
        0.20 * evidence["constitution_confidence"]
        + 0.20 * evidence["court_confidence"]
        + 0.15 * shadow_perf
        + 0.15 * evidence["governance_quality"]
        + 0.15 * evidence["readiness_score"]
        + 0.15 * evidence["system_health_score"]
    )
    penalty = 0.0
    if evidence["constitution_severity"] == "CRITICAL":
        penalty += 0.15
    if evidence["constitution_severity"] == "LOCKDOWN":
        penalty += 0.30
    if evidence["court_ruling"] == COURT_OVERRULED:
        penalty += 0.10
    if evidence["court_ruling"] == COURT_LOCKDOWN:
        penalty += 0.25
    penalty += min(0.20, len(deterioration) * 0.05)
    if ruling == RULING_LOCKDOWN:
        penalty += 0.10
    return _clamp(base - penalty, 0.0, 1.0)


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _recommendations(ruling: str, evidence: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    if ruling == RULING_LOCKDOWN:
        recs.append("Continue manual supervision; council enforces governance LOCKDOWN.")
        recs.append("Do not attempt any governance relaxation until constitutional recovery.")
        recs.append("Refresh stale intelligence and restore system health.")
        return recs
    if ruling == RULING_REVOKE:
        recs.append("Freeze autonomy escalation; revoke pending promotion.")
        recs.append("Extend apprenticeship period until shadow performance recovers.")
        recs.append("Increase cash preservation posture.")
        return recs
    if ruling == RULING_TIGHTEN:
        recs.append("Increase cash preservation and operator oversight.")
        recs.append("Extend apprenticeship period; do not promote autonomy.")
        if evidence["defensive_constraints"]:
            recs.append("Permit assisted-mode observation only under defensive constraints.")
        return recs
    if ruling == RULING_LOOSEN:
        recs.append("Governance may relax cautiously; constitutional floor remains active.")
        recs.append("Permit assisted-mode observation under continued operator review.")
        recs.append("Maintain shadow evaluation in parallel as regression guard.")
        return recs
    recs.append("Maintain governance posture; continue monitoring for deterioration.")
    recs.append("No governance change recommended at this time.")
    return recs


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    ruling: str,
    reasons: List[str],
    evidence: Dict[str, Any],
    actions: List[str],
    booleans: Dict[str, bool],
    confidence: float,
    recommendations: List[str],
    deterioration: List[str],
) -> str:
    def fmt(x: Optional[float], spec: str = ".3f") -> str:
        if x is None:
            return "-"
        return format(x, spec)

    lines: List[str] = []
    lines.append("# Triton Supreme Governance Council")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_")
    lines.append("")

    lines.append("## Governance Ruling")
    lines.append("")
    lines.append(f"**{ruling}**")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| council_confidence | {confidence:.3f} |")
    lines.append(f"| governance_change_recommended | {booleans['governance_change_recommended']} |")
    lines.append(
        f"| governance_tightening_required | {booleans['governance_tightening_required']} |"
    )
    lines.append(f"| governance_relaxation_allowed | {booleans['governance_relaxation_allowed']} |")
    lines.append(f"| autonomy_revocation_required | {booleans['autonomy_revocation_required']} |")
    lines.append(f"| operator_supervision_required | {booleans['operator_supervision_required']} |")
    lines.append(f"| lockdown_required | {booleans['lockdown_required']} |")
    lines.append("")

    lines.append("## Governance Posture")
    lines.append("")
    lines.append("| signal | value |")
    lines.append("|---|---|")
    lines.append(f"| court_ruling | {evidence['court_ruling']} |")
    lines.append(f"| constitution_state | {evidence['constitution_state']} |")
    lines.append(f"| graduation_state | {evidence['graduation_state']} |")
    lines.append(f"| apprenticeship_verdict | {evidence['apprenticeship_verdict']} |")
    lines.append(f"| governance_quality | {fmt(evidence['governance_quality'])} |")
    lines.append(f"| shadow_observations | {evidence['shadow_observations']} |")
    lines.append(f"| shadow_autonomy_readiness | {fmt(evidence['shadow_autonomy_readiness'])} |")
    lines.append(f"| system_health_status | {evidence['system_health_status']} |")
    lines.append(f"| readiness_state | {evidence['readiness_state']} |")
    lines.append(f"| regime | {evidence['regime']} |")
    lines.append(f"| target_cash_pct | {fmt(evidence.get('target_cash_pct'), '.1f')} |")
    lines.append("")

    lines.append("## Recommended Actions")
    lines.append("")
    for a in actions:
        lines.append(f"- {a}")
    lines.append("")

    lines.append("## Why")
    lines.append("")
    for r in reasons:
        lines.append(f"- {r}")
    if deterioration:
        lines.append("")
        lines.append("_Deterioration signals:_")
        for d in deterioration:
            lines.append(f"- {d}")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Narrative")
    lines.append("")
    lines.append(_narrative(ruling, evidence, confidence, booleans))
    lines.append("")
    return "\n".join(lines)


def _narrative(
    ruling: str,
    evidence: Dict[str, Any],
    confidence: float,
    booleans: Dict[str, bool],
) -> str:
    if ruling == RULING_LOCKDOWN:
        return (
            f"The Supreme Governance Council entered {RULING_LOCKDOWN}. "
            f"Constitutional court or governance collapse requires manual-only "
            f"operation. The council cannot recommend any relaxation. "
            f"Confidence {confidence:.2f}."
        )
    if ruling == RULING_REVOKE:
        return (
            f"The council {RULING_REVOKE} because deterioration was detected "
            f"across shadow performance, constitutional pressure, or court "
            f"overruling. Autonomy escalation is frozen. Confidence {confidence:.2f}."
        )
    if ruling == RULING_TIGHTEN:
        return (
            f"Governance tightened because constitutional pressure increased, "
            f"shadow performance remains immature "
            f"(verdict={evidence['apprenticeship_verdict']}, "
            f"obs={evidence['shadow_observations']}), and defensive capital "
            f"preservation takes precedence. Confidence {confidence:.2f}."
        )
    if ruling == RULING_LOOSEN:
        return (
            f"The council {RULING_LOOSEN} cautiously. Strong apprenticeship "
            f"({evidence['apprenticeship_verdict']}) and stable constitutional "
            f"governance permit selective relaxation under operator review. "
            f"Confidence {confidence:.2f}."
        )
    return (
        f"The council {RULING_MAINTAIN}. Governance posture is stable with "
        f"no material deterioration or improvement warranting change. "
        f"Confidence {confidence:.2f}."
    )


# -----------------------------------------------------------
# Posture memory
# -----------------------------------------------------------
def _build_posture_row(
    *,
    timestamp: str,
    ruling: str,
    evidence: Dict[str, Any],
    actions: List[str],
    reasons: List[str],
    confidence: float,
    booleans: Dict[str, bool],
) -> Dict[str, Any]:
    return {
        "timestamp_utc": timestamp,
        "ruling": ruling,
        "governance_state": evidence["governance_state"] or evidence["graduation_state"],
        "confidence": round(confidence, 6),
        "recommended_actions": "|".join(actions),
        "autonomy_state": evidence["graduation_state"],
        "constitutional_state": evidence["constitution_state"],
        "court_ruling": evidence["court_ruling"],
        "apprenticeship_verdict": evidence["apprenticeship_verdict"],
        "system_health_status": evidence["system_health_status"],
        "regime": evidence["regime"],
        "rationale": " | ".join(reasons),
        "governance_change_recommended": booleans["governance_change_recommended"],
        "lockdown_required": booleans["lockdown_required"],
    }


def _merge_posture_memory(
    existing: List[Dict[str, Any]],
    new_row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Append-only keyed by timestamp_utc."""
    keyed: Dict[str, Dict[str, Any]] = {}
    for r in existing:
        ts = str(r.get("timestamp_utc", ""))
        if ts:
            keyed[ts] = r
    keyed[str(new_row.get("timestamp_utc", ""))] = new_row
    out = list(keyed.values())
    for r in out:
        for c in POSTURE_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_council_ruling(
    *,
    court_summary: Dict[str, Any],
    court_record: Dict[str, Any],
    const_summary: Dict[str, Any],
    grad_summary: Dict[str, Any],
    shadow_perf: Dict[str, Any],
    scorecard: Dict[str, Any],
    health: Dict[str, Any],
    readiness: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    meta_decision: Dict[str, Any],
    existing_memory_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    evidence = _extract_evidence(
        court_summary=court_summary,
        court_record=court_record,
        const_summary=const_summary,
        grad_summary=grad_summary,
        shadow_perf=shadow_perf,
        scorecard=scorecard,
        health=health,
        readiness=readiness,
        runtime_policy=runtime_policy,
        meta_decision=meta_decision,
    )

    have_evidence = any(
        bool(x)
        for x in (
            court_summary,
            const_summary,
            grad_summary,
            health,
        )
    )

    deterioration = _deterioration_signals(evidence, existing_memory_rows) if have_evidence else []
    ruling, reasons = _classify_ruling(
        evidence=evidence,
        deterioration=deterioration,
        have_evidence=have_evidence,
    )

    # Council cannot violate constitutional court -- cap LOOSEN if court blocks
    if ruling == RULING_LOOSEN and (
        evidence["court_lockdown"]
        or evidence["constitution_violated"]
        or evidence["court_override"]
    ):
        ruling = RULING_MAINTAIN
        reasons = ["council LOOSEN blocked: constitutional court/constitution prohibits relaxation"]

    actions = _governance_actions(ruling, evidence)
    booleans = _governance_booleans(ruling, evidence)
    confidence = _council_confidence(evidence, ruling, deterioration)
    recommendations = _recommendations(ruling, evidence)

    posture_row = _build_posture_row(
        timestamp=timestamp,
        ruling=ruling,
        evidence=evidence,
        actions=actions,
        reasons=reasons,
        confidence=confidence,
        booleans=booleans,
    )
    merged_memory = _merge_posture_memory(existing_memory_rows, posture_row)

    md = _render_markdown(
        generated_at=timestamp,
        ruling=ruling,
        reasons=reasons,
        evidence=evidence,
        actions=actions,
        booleans=booleans,
        confidence=confidence,
        recommendations=recommendations,
        deterioration=deterioration,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_supreme_governance_council_engine",
        "engine_version": 1,
        "governance_ruling": ruling,
        "ruling_reasons": reasons,
        "council_confidence": round(confidence, 6),
        "evidence": evidence,
        "deterioration_signals": deterioration,
        "governance_booleans": booleans,
        "governance_actions": actions,
        "recommendations": recommendations,
        "council_precedence": list(COUNCIL_PRECEDENCE),
        "constitutional_supremacy_note": (
            "The council may recommend governance changes but cannot "
            "override constitutional law or the constitutional court."
        ),
        "posture_memory_size_after_append": len(merged_memory),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "council_only": True,
        },
        "inputs_seen": {
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_constitutional_court_record": bool(court_record),
            "arm_autonomy_constitution_summary": bool(const_summary),
            "arm_autonomy_graduation_summary": bool(grad_summary),
            "arm_shadow_performance_summary": bool(shadow_perf),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health),
            "autonomous_readiness_summary": bool(readiness),
            "runtime_policy_governed": bool(runtime_policy),
            "meta_decision_intelligence": bool(meta_decision),
            "existing_posture_memory_rows": len(existing_memory_rows),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_supreme_governance_council_engine",
        "governance_ruling": ruling,
        "council_confidence": record["council_confidence"],
        "governance_change_recommended": booleans["governance_change_recommended"],
        "governance_tightening_required": booleans["governance_tightening_required"],
        "governance_relaxation_allowed": booleans["governance_relaxation_allowed"],
        "autonomy_revocation_required": booleans["autonomy_revocation_required"],
        "operator_supervision_required": booleans["operator_supervision_required"],
        "lockdown_required": booleans["lockdown_required"],
        "court_ruling": evidence["court_ruling"],
        "constitution_state": evidence["constitution_state"],
        "graduation_state": evidence["graduation_state"],
        "apprenticeship_verdict": evidence["apprenticeship_verdict"],
        "governance_actions": actions,
        "n_recommendations": len(recommendations),
        "posture_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM supreme governance council engine (Step 34). "
            "Evaluates whether governance posture should change. "
            "Cannot override constitutional law. No broker calls."
        ),
    )
    p.add_argument("--court-summary", default=str(DEFAULT_COURT_SUMMARY))
    p.add_argument("--court-record", default=str(DEFAULT_COURT_RECORD))
    p.add_argument("--const-summary", default=str(DEFAULT_CONST_SUMMARY))
    p.add_argument("--grad-summary", default=str(DEFAULT_GRAD_SUMMARY))
    p.add_argument("--shadow-perf", default=str(DEFAULT_SHADOW_PERF))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--health", default=str(DEFAULT_HEALTH))
    p.add_argument("--readiness", default=str(DEFAULT_READINESS))
    p.add_argument("--gov-policy", default=str(DEFAULT_GOV_POLICY))
    p.add_argument("--meta-decision", default=str(DEFAULT_META_DECISION))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-mem-csv", default=str(DEFAULT_OUT_MEM_CSV))
    p.add_argument("--out-mem-parquet", default=str(DEFAULT_OUT_MEM_PQ))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[ARM_COUNCIL] starting (read-only supreme governance council; no broker calls)",
        flush=True,
    )

    court_summary = _safe_read_json(
        Path(args.court_summary), label="arm_constitutional_court_summary.json"
    )
    court_record = _safe_read_json(Path(args.court_record), label="arm_constitutional_court.json")
    const_summary = _safe_read_json(
        Path(args.const_summary), label="arm_autonomy_constitution_summary.json"
    )
    grad_summary = _safe_read_json(
        Path(args.grad_summary), label="arm_autonomy_graduation_summary.json"
    )
    shadow_perf = _safe_read_json(
        Path(args.shadow_perf), label="arm_shadow_performance_summary.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    health = _safe_read_json(Path(args.health), label="autonomous_system_health_summary.json")
    readiness = _safe_read_json(Path(args.readiness), label="autonomous_readiness_summary.json")
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    meta_decision = _safe_read_json(
        Path(args.meta_decision), label="meta_decision_intelligence.json"
    )
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_posture_memory.csv"
    )

    record, summary, md, merged_memory = build_council_ruling(
        court_summary=court_summary,
        court_record=court_record,
        const_summary=const_summary,
        grad_summary=grad_summary,
        shadow_perf=shadow_perf,
        scorecard=scorecard,
        health=health,
        readiness=readiness,
        runtime_policy=runtime_policy,
        meta_decision=meta_decision,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=POSTURE_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    booleans = record["governance_booleans"]
    print(
        "[ARM_COUNCIL] "
        f"ruling={record['governance_ruling']} "
        f"tighten={booleans['governance_tightening_required']} "
        f"revoke={booleans['autonomy_revocation_required']} "
        f"lockdown={booleans['lockdown_required']} "
        f"confidence={record['council_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_COUNCIL_SAFETY] broker_calls=0 orders_placed=0 portfolio_mutated=False",
        flush=True,
    )
    print(
        f"[ARM_COUNCIL_OUT] json={Path(args.out_json).as_posix()} "
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
    """Refuse to import if any forbidden broker token appears in source."""
    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return
    for tok in _FORBIDDEN_TOKENS:
        if tok in src:
            raise RuntimeError(f"[ARM_COUNCIL_SAFETY] forbidden broker token detected: {tok!r}")


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
