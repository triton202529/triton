"""
ARM Autonomy Graduation Engine -- Step 31.

Reads:
    data/results/arm_shadow_performance_summary.json     (Step 30)
    data/results/arm_shadow_performance.json             (Step 30)
    data/results/arm_shadow_outcomes_memory.csv          (Step 30 -- history)
    data/results/arm_mode_governance_summary.json        (Step 28)
    data/results/autonomous_execution_certificate_summary.json   (Step 27)
    data/results/autonomous_governance_scorecard.json    (Step 19)
    data/results/autonomous_readiness_summary.json       (Step 21)
    data/results/autonomous_system_health_summary.json   (Step 20)
    data/results/runtime_policy_governed.json            (Step 18)

Writes:
    data/results/arm_autonomy_graduation.json
    data/results/arm_autonomy_graduation.md
    data/results/arm_autonomy_graduation_summary.json
    data/results/arm_autonomy_progression_memory.csv
    data/results/arm_autonomy_progression_memory.parquet

Purpose
-------
This engine answers:

    "Has Triton earned more autonomy?"

It is the evidence-based promotion / demotion layer that turns Step
30's shadow performance evaluation into a formal autonomy
classification. The seven-tier ladder spans from MANUAL_LOCKED at
the floor through ASSISTED_CANDIDATE, ASSISTED_APPROVED,
AUTO_DISABLED_CANDIDATE, AUTO_ALLOWED_CANDIDATE,
AUTO_ALLOWED_APPROVED, with AUTONOMY_REVOKED as an orthogonal
demotion signal triggered by sustained deterioration relative to
the historical peak.

The engine is governance only -- it has no execution authority,
makes no broker calls, and never mutates portfolio state. The
progression memory it persists is the audit trail that proves any
future autonomy was earned through observed evidence rather than
operator opinion.

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens (defined in Step 29 spec) are never written
literally; an import-time self-check raises if they ever appear.

Graduation cascade (strict precedence, evaluated top-down)
----------------------------------------------------------
    0. Safety floor       -> MANUAL_LOCKED
    1. Deterioration      -> AUTONOMY_REVOKED
    2. AUTO_ALLOWED_APPROVED  (sustained strong, full evidence)
    3. AUTO_ALLOWED_CANDIDATE (strong sustained, op approval req.)
    4. AUTO_DISABLED_CANDIDATE (strong, more shadow time required)
    5. ASSISTED_APPROVED       (stable positive shadow discipline)
    6. ASSISTED_CANDIDATE      (improving, immature)
    7. MANUAL_LOCKED          (default floor)

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only progression memory keyed by timestamp_utc.
* Missing inputs warn-and-continue; an empty pipeline produces
  MANUAL_LOCKED with confidence reflecting the missing evidence.
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

DEFAULT_PERF_SUMMARY = RESULTS_DIR / "arm_shadow_performance_summary.json"
DEFAULT_PERF_RECORD = RESULTS_DIR / "arm_shadow_performance.json"
DEFAULT_OUTCOMES_CSV = RESULTS_DIR / "arm_shadow_outcomes_memory.csv"
DEFAULT_ARM_SUMMARY = RESULTS_DIR / "arm_mode_governance_summary.json"
DEFAULT_CERT_SUMMARY = RESULTS_DIR / "autonomous_execution_certificate_summary.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_READINESS = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_HEALTH = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_autonomy_graduation.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_autonomy_graduation.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_autonomy_graduation_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_autonomy_progression_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_autonomy_progression_memory.parquet"


# -----------------------------------------------------------
# Graduation states
# -----------------------------------------------------------
STATE_MANUAL_LOCKED = "MANUAL_LOCKED"
STATE_ASSISTED_CANDIDATE = "ASSISTED_CANDIDATE"
STATE_ASSISTED_APPROVED = "ASSISTED_APPROVED"
STATE_AUTO_DISABLED_CANDIDATE = "AUTO_DISABLED_CANDIDATE"
STATE_AUTO_ALLOWED_CANDIDATE = "AUTO_ALLOWED_CANDIDATE"
STATE_AUTO_ALLOWED_APPROVED = "AUTO_ALLOWED_APPROVED"
STATE_AUTONOMY_REVOKED = "AUTONOMY_REVOKED"

# Ladder rank (used by deterioration detection only; REVOKED is
# orthogonal and is not part of the rank ladder).
STATE_RANK: Dict[str, int] = {
    STATE_MANUAL_LOCKED: 0,
    STATE_ASSISTED_CANDIDATE: 1,
    STATE_ASSISTED_APPROVED: 2,
    STATE_AUTO_DISABLED_CANDIDATE: 3,
    STATE_AUTO_ALLOWED_CANDIDATE: 4,
    STATE_AUTO_ALLOWED_APPROVED: 5,
}

# Apprenticeship verdict ordinal (Step 30 verdict scale)
VERDICT_RANK: Dict[str, int] = {
    "AUTONOMY_NOT_READY": -1,
    "LEARNING": 0,
    "IMPROVING": 1,
    "TRUST_BUILDING": 2,
    "AUTONOMY_CANDIDATE": 3,
}

# Thresholds (spec section 2)
THR_ASSISTED_APPROVED_OBS = 25
THR_ASSISTED_APPROVED_WIN = 0.50
THR_ASSISTED_APPROVED_DISC = 0.55
THR_ASSISTED_APPROVED_GOV = 0.55

THR_AUTO_DISABLED_OBS = 60
THR_AUTO_DISABLED_READINESS = 0.65

THR_AUTO_ALLOWED_CANDIDATE_OBS = 75
THR_AUTO_ALLOWED_CANDIDATE_READINESS = 0.70
THR_AUTO_ALLOWED_CANDIDATE_GOV = 0.65
THR_AUTO_ALLOWED_CANDIDATE_DRAWDOWN = 0.65

THR_AUTO_ALLOWED_APPROVED_OBS = 150
THR_AUTO_ALLOWED_APPROVED_READINESS = 0.80
THR_AUTO_ALLOWED_APPROVED_GOV = 0.75
THR_AUTO_ALLOWED_APPROVED_CAP_PRES = 0.75

# Deterioration triggers (REVOKED)
DETERIORATION_WIN_FLOOR = 0.40
DETERIORATION_DISCIPLINE_FLOOR = 0.40
DETERIORATION_GOV_FLOOR = 0.40
DETERIORATION_READINESS_DROP = 0.15  # vs historical peak

PROGRESSION_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp_utc",
    "graduation_state",
    "previous_state",
    "observations",
    "apprenticeship_verdict",
    "autonomy_readiness",
    "shadow_win_rate",
    "shadow_discipline_score",
    "shadow_drawdown_score",
    "governance_quality",
    "system_health_score",
    "certificate_confidence",
    "confidence",
    "autonomy_promotion_earned",
    "autonomy_revoked",
    "reason",
    "governance_booleans_json",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_GRADUATION_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(PROGRESSION_MEMORY_COLUMNS))
        numeric_cols = (
            "observations",
            "autonomy_readiness",
            "shadow_win_rate",
            "shadow_discipline_score",
            "shadow_drawdown_score",
            "governance_quality",
            "system_health_score",
            "certificate_confidence",
            "confidence",
        )
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("autonomy_promotion_earned", "autonomy_revoked"):
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
    perf_summary: Dict[str, Any],
    perf_record: Dict[str, Any],
    arm_summary: Dict[str, Any],
    cert_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    readiness: Dict[str, Any],
    health: Dict[str, Any],
    runtime_policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Flatten upstream artefacts into a single evidence dict."""
    metrics = (perf_record or {}).get("metrics") or {}
    summary_stats = (perf_record or {}).get("summary_stats") or {}

    verdict = _norm_upper(
        perf_summary.get("apprenticeship_verdict")
        or perf_record.get("apprenticeship_verdict")
        or "LEARNING"
    )
    observations = (
        _to_int(perf_summary.get("n_labelled") or summary_stats.get("n_labelled") or 0) or 0
    )

    autonomy_readiness = _to_float(
        metrics.get("shadow_autonomy_readiness_score")
        or perf_summary.get("shadow_autonomy_readiness_score")
    )
    win_rate = _to_float(metrics.get("shadow_win_rate") or perf_summary.get("shadow_win_rate"))
    discipline = _to_float(
        metrics.get("shadow_discipline_score") or perf_summary.get("shadow_discipline_score")
    )
    drawdown = _to_float(metrics.get("shadow_drawdown_score"))
    governance = _to_float(
        metrics.get("shadow_governance_score")
        or perf_summary.get("shadow_governance_score")
        or (scorecard or {}).get("governance_quality_score")
        or (scorecard or {}).get("intelligence_health_score")
    )

    # capital preservation proxy: drawdown_score from Step 30
    capital_preservation = drawdown if drawdown is not None else 0.5

    # System / cert evidence (safety-floor signals)
    health_status = _norm_upper((health or {}).get("overall_status"))
    health_score = _to_float((health or {}).get("system_health_score"))
    if health_score is None:
        # Approximate from status when missing
        health_score = {
            "HEALTHY": 0.90,
            "DEGRADED": 0.55,
            "STALE": 0.35,
            "CRITICAL": 0.10,
            "OFFLINE": 0.00,
        }.get(health_status, 0.50)

    cert_state = _norm_upper((cert_summary or {}).get("certification_state"))
    cert_conf = _to_float((cert_summary or {}).get("certificate_confidence")) or 0.0
    cert_valid = bool(
        (cert_summary or {}).get(
            "certificate_valid",
            cert_state in ("EXECUTION_CERTIFIED", "EXECUTION_CERTIFIED_LIMITED"),
        )
    )

    readiness_state = _norm_upper((readiness or {}).get("readiness_state"))
    governance_state = _norm_upper(
        (scorecard or {}).get("system_state") or (arm_summary or {}).get("governance_state")
    )

    return {
        "apprenticeship_verdict": verdict,
        "verdict_rank": VERDICT_RANK.get(verdict, 0),
        "observations": observations,
        "autonomy_readiness": autonomy_readiness,
        "shadow_win_rate": win_rate,
        "shadow_discipline_score": discipline,
        "shadow_drawdown_score": drawdown,
        "capital_preservation_score": capital_preservation,
        "governance_quality": governance if governance is not None else 0.5,
        "system_health_status": health_status,
        "system_health_score": _clamp(health_score, 0.0, 1.0),
        "certification_state": cert_state,
        "certificate_confidence": _clamp(cert_conf, 0.0, 1.0),
        "certificate_valid": cert_valid,
        "readiness_state": readiness_state,
        "arm_mode": _norm_upper((arm_summary or {}).get("arm_mode")),
        "governance_state": governance_state,
        "runtime_policy_version": (runtime_policy or {}).get("policy_version"),
        "trajectory_improving": bool(
            (perf_record or {}).get("summary_stats", {}).get("trajectory_improving")
        ),
    }


# -----------------------------------------------------------
# Deterioration detection (AUTONOMY_REVOKED)
# -----------------------------------------------------------
def _peak_state_from_memory(
    memory_rows: List[Dict[str, str]],
) -> Tuple[Optional[str], Optional[float]]:
    """Return (peak_state, peak_autonomy_readiness) seen in history."""
    if not memory_rows:
        return None, None
    peak_state: Optional[str] = None
    peak_rank = -1
    peak_readiness: Optional[float] = None
    for r in memory_rows:
        st = str(r.get("graduation_state") or "")
        rk = STATE_RANK.get(st, -1)
        if rk > peak_rank:
            peak_rank = rk
            peak_state = st
        ar = _to_float(r.get("autonomy_readiness"))
        if ar is not None and (peak_readiness is None or ar > peak_readiness):
            peak_readiness = ar
    return peak_state, peak_readiness


def _detect_deterioration(
    *,
    evidence: Dict[str, Any],
    memory_rows: List[Dict[str, str]],
) -> Tuple[bool, List[str]]:
    """
    Trigger AUTONOMY_REVOKED only when we *previously* earned
    promotion AND current evidence shows material deterioration.
    """
    reasons: List[str] = []
    peak_state, peak_readiness = _peak_state_from_memory(memory_rows)
    if peak_state is None:
        return False, reasons
    peak_rank = STATE_RANK.get(peak_state, -1)
    # Only consider revocation if we ever crossed into ASSISTED_APPROVED+
    if peak_rank < STATE_RANK[STATE_ASSISTED_APPROVED]:
        return False, reasons

    triggers: List[str] = []
    verdict_rank = evidence["verdict_rank"]
    win = evidence["shadow_win_rate"]
    disc = evidence["shadow_discipline_score"]
    gov = evidence["governance_quality"]
    cur_readiness = evidence["autonomy_readiness"]

    if verdict_rank < 0:  # AUTONOMY_NOT_READY explicitly
        triggers.append("apprenticeship_verdict regressed to AUTONOMY_NOT_READY")
    if win is not None and win < DETERIORATION_WIN_FLOOR:
        triggers.append(f"shadow_win_rate={win:.2f} below {DETERIORATION_WIN_FLOOR}")
    if disc is not None and disc < DETERIORATION_DISCIPLINE_FLOOR:
        triggers.append(
            f"shadow_discipline_score={disc:.2f} below {DETERIORATION_DISCIPLINE_FLOOR}"
        )
    if gov is not None and gov < DETERIORATION_GOV_FLOOR:
        triggers.append(f"governance_quality={gov:.2f} below {DETERIORATION_GOV_FLOOR}")
    if (
        cur_readiness is not None
        and peak_readiness is not None
        and (peak_readiness - cur_readiness) > DETERIORATION_READINESS_DROP
    ):
        triggers.append(
            f"autonomy_readiness dropped {peak_readiness:.2f}->{cur_readiness:.2f} "
            f"(>{DETERIORATION_READINESS_DROP})"
        )

    if not triggers:
        return False, reasons
    reasons.append(f"prior peak state was {peak_state}; deterioration triggered:")
    reasons.extend(triggers)
    return True, reasons


# -----------------------------------------------------------
# Graduation classification
# -----------------------------------------------------------
def _safety_floor_locked(evidence: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Return (locked, reasons) if any safety floor forces MANUAL_LOCKED."""
    reasons: List[str] = []
    if evidence["system_health_status"] in ("CRITICAL", "OFFLINE"):
        reasons.append(f"system_health={evidence['system_health_status']}")
        return True, reasons
    if evidence["certification_state"] in ("EXECUTION_BLOCKED", "EXECUTION_DENIED"):
        reasons.append(f"certification_state={evidence['certification_state']}")
        return True, reasons
    if not evidence["certificate_valid"] and evidence["certification_state"] not in ("UNKNOWN", ""):
        reasons.append(f"certificate_valid=False (state={evidence['certification_state']})")
        return True, reasons
    gov_state = evidence["governance_state"]
    if gov_state in ("COLLAPSED", "GOVERNANCE_COLLAPSED"):
        reasons.append(f"governance_state={gov_state}")
        return True, reasons
    return False, reasons


def _classify(
    *,
    evidence: Dict[str, Any],
    memory_rows: List[Dict[str, str]],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    # ---- 0. Safety floor ----
    locked, lock_reasons = _safety_floor_locked(evidence)
    if locked:
        reasons.extend(lock_reasons)
        reasons.append("safety floor forces MANUAL_LOCKED")
        return STATE_MANUAL_LOCKED, reasons

    # ---- 1. AUTONOMY_REVOKED (orthogonal demotion) ----
    revoked, revoke_reasons = _detect_deterioration(
        evidence=evidence,
        memory_rows=memory_rows,
    )
    if revoked:
        reasons.extend(revoke_reasons)
        return STATE_AUTONOMY_REVOKED, reasons

    obs = evidence["observations"]
    vrank = evidence["verdict_rank"]
    readiness = evidence["autonomy_readiness"] or 0.0
    win = evidence["shadow_win_rate"] or 0.0
    disc = evidence["shadow_discipline_score"] or 0.0
    gov = evidence["governance_quality"] or 0.0
    cap_pres = evidence["capital_preservation_score"] or 0.0
    drawdown = evidence["shadow_drawdown_score"] or 0.0
    system_healthy = evidence["system_health_status"] == "HEALTHY"
    readiness_ready = evidence["readiness_state"] == "READY"

    # ---- 2. AUTO_ALLOWED_APPROVED ----
    if (
        vrank >= VERDICT_RANK["AUTONOMY_CANDIDATE"]
        and obs >= THR_AUTO_ALLOWED_APPROVED_OBS
        and readiness >= THR_AUTO_ALLOWED_APPROVED_READINESS
        and gov >= THR_AUTO_ALLOWED_APPROVED_GOV
        and cap_pres >= THR_AUTO_ALLOWED_APPROVED_CAP_PRES
        and system_healthy
        and readiness_ready
    ):
        reasons.append(
            f"verdict=AUTONOMY_CANDIDATE, obs={obs}>={THR_AUTO_ALLOWED_APPROVED_OBS}, "
            f"readiness={readiness:.2f}>={THR_AUTO_ALLOWED_APPROVED_READINESS}, "
            f"governance={gov:.2f}>={THR_AUTO_ALLOWED_APPROVED_GOV}, "
            f"capital_preservation={cap_pres:.2f}>={THR_AUTO_ALLOWED_APPROVED_CAP_PRES}, "
            f"system HEALTHY, readiness READY"
        )
        return STATE_AUTO_ALLOWED_APPROVED, reasons

    # ---- 3. AUTO_ALLOWED_CANDIDATE ----
    if (
        vrank >= VERDICT_RANK["TRUST_BUILDING"]
        and obs >= THR_AUTO_ALLOWED_CANDIDATE_OBS
        and readiness >= THR_AUTO_ALLOWED_CANDIDATE_READINESS
        and gov >= THR_AUTO_ALLOWED_CANDIDATE_GOV
        and drawdown >= THR_AUTO_ALLOWED_CANDIDATE_DRAWDOWN
    ):
        reasons.append(
            f"verdict>=TRUST_BUILDING, obs={obs}>={THR_AUTO_ALLOWED_CANDIDATE_OBS}, "
            f"readiness={readiness:.2f}>={THR_AUTO_ALLOWED_CANDIDATE_READINESS}, "
            f"governance={gov:.2f}>={THR_AUTO_ALLOWED_CANDIDATE_GOV}, "
            f"drawdown_score={drawdown:.2f}>={THR_AUTO_ALLOWED_CANDIDATE_DRAWDOWN}; "
            "operator approval still required"
        )
        return STATE_AUTO_ALLOWED_CANDIDATE, reasons

    # ---- 4. AUTO_DISABLED_CANDIDATE ----
    if (
        vrank >= VERDICT_RANK["TRUST_BUILDING"]
        and obs >= THR_AUTO_DISABLED_OBS
        and readiness >= THR_AUTO_DISABLED_READINESS
    ):
        reasons.append(
            f"verdict>=TRUST_BUILDING, obs={obs}>={THR_AUTO_DISABLED_OBS}, "
            f"readiness={readiness:.2f}>={THR_AUTO_DISABLED_READINESS}; "
            "shadow-mode continuation required"
        )
        return STATE_AUTO_DISABLED_CANDIDATE, reasons

    # ---- 5. ASSISTED_APPROVED ----
    if (
        vrank >= VERDICT_RANK["IMPROVING"]
        and obs >= THR_ASSISTED_APPROVED_OBS
        and win >= THR_ASSISTED_APPROVED_WIN
        and disc >= THR_ASSISTED_APPROVED_DISC
        and gov >= THR_ASSISTED_APPROVED_GOV
    ):
        reasons.append(
            f"verdict>=IMPROVING, obs={obs}>={THR_ASSISTED_APPROVED_OBS}, "
            f"win_rate={win:.2f}>={THR_ASSISTED_APPROVED_WIN}, "
            f"discipline={disc:.2f}>={THR_ASSISTED_APPROVED_DISC}, "
            f"governance={gov:.2f}>={THR_ASSISTED_APPROVED_GOV}"
        )
        return STATE_ASSISTED_APPROVED, reasons

    # ---- 6. ASSISTED_CANDIDATE ----
    if vrank >= VERDICT_RANK["IMPROVING"] and obs >= 10:
        reasons.append(f"verdict>=IMPROVING, obs={obs}>=10; promotion criteria not yet met")
        return STATE_ASSISTED_CANDIDATE, reasons

    # ---- 7. MANUAL_LOCKED (default) ----
    reasons.append(
        f"verdict={evidence['apprenticeship_verdict']}, obs={obs}; "
        "insufficient evidence to advance"
    )
    return STATE_MANUAL_LOCKED, reasons


# -----------------------------------------------------------
# Booleans and confidence
# -----------------------------------------------------------
def _booleans_for(state: str) -> Dict[str, bool]:
    earned_states = {
        STATE_ASSISTED_APPROVED,
        STATE_AUTO_DISABLED_CANDIDATE,
        STATE_AUTO_ALLOWED_CANDIDATE,
        STATE_AUTO_ALLOWED_APPROVED,
    }
    assisted_eligible = earned_states
    auto_candidate = {STATE_AUTO_DISABLED_CANDIDATE, STATE_AUTO_ALLOWED_CANDIDATE}
    return {
        "autonomy_promotion_earned": state in earned_states,
        "assisted_mode_eligible": state in assisted_eligible,
        "auto_mode_candidate": state in auto_candidate,
        "auto_mode_approved": state == STATE_AUTO_ALLOWED_APPROVED,
        "operator_signoff_required": state != STATE_AUTO_ALLOWED_APPROVED,
        "autonomy_revoked": state == STATE_AUTONOMY_REVOKED,
    }


def _confidence(evidence: Dict[str, Any]) -> float:
    """Weighted blend of: readiness 0.30, governance 0.20, system_health 0.15,
    certificate_confidence 0.15, apprenticeship_maturity 0.20."""
    readiness = (
        evidence["autonomy_readiness"] if evidence["autonomy_readiness"] is not None else 0.5
    )
    governance = evidence["governance_quality"]
    health = evidence["system_health_score"]
    cert = evidence["certificate_confidence"]
    maturity = _clamp(evidence["observations"] / float(THR_AUTO_ALLOWED_APPROVED_OBS), 0.0, 1.0)
    components = [
        (0.30, readiness),
        (0.20, governance),
        (0.15, health),
        (0.15, cert),
        (0.20, maturity),
    ]
    total_w = sum(w for w, _ in components)
    score = sum(w * v for w, v in components) / total_w
    return _clamp(score, 0.0, 1.0)


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
def _recommendations(state: str, evidence: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    obs = evidence["observations"]
    if state == STATE_MANUAL_LOCKED:
        recs.append("Maintain manual override; extend shadow evaluation period.")
        if obs < THR_ASSISTED_APPROVED_OBS:
            recs.append(
                f"Accumulate at least {THR_ASSISTED_APPROVED_OBS} labelled observations "
                f"(current: {obs})."
            )
        recs.append("Resolve any safety-floor blockers (system health, certificate, governance).")
        return recs
    if state == STATE_AUTONOMY_REVOKED:
        recs.append("Revoke autonomy escalation; return to shadow-only operation.")
        recs.append(
            "Investigate which evidence channel degraded (win rate, discipline, governance)."
        )
        recs.append("Require sustained recovery before permitting any new graduation attempt.")
        return recs
    if state == STATE_ASSISTED_CANDIDATE:
        recs.append("Continue apprenticeship; promotion criteria not yet met.")
        recs.append("Aim for sustained win_rate >= 0.50 and discipline >= 0.55 across more cycles.")
        return recs
    if state == STATE_ASSISTED_APPROVED:
        recs.append("Permit assisted-mode trials under continued operator supervision.")
        recs.append("Compare assisted-mode outcomes against shadow predictions to validate trust.")
        return recs
    if state == STATE_AUTO_DISABLED_CANDIDATE:
        recs.append("Continue shadow operation; autonomy nearly earned but not yet ready.")
        recs.append(
            f"Strengthen capital preservation (drawdown_score >= "
            f"{THR_AUTO_ALLOWED_CANDIDATE_DRAWDOWN}) and governance "
            f">= {THR_AUTO_ALLOWED_CANDIDATE_GOV} before advancing."
        )
        return recs
    if state == STATE_AUTO_ALLOWED_CANDIDATE:
        recs.append(
            "Strong sustained performance -- operator approval still required for auto-mode."
        )
        recs.append("Run a controlled assisted-mode pilot before lifting to full autonomy.")
        return recs
    # AUTO_ALLOWED_APPROVED
    recs.append("Full earned autonomy unlocked; continue shadow evaluation in parallel.")
    recs.append("Maintain operator review for any policy changes that affect risk posture.")
    return recs


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    previous_state: Optional[str],
    reasons: List[str],
    evidence: Dict[str, Any],
    booleans: Dict[str, bool],
    confidence: float,
    recommendations: List[str],
) -> str:
    def fmt(x: Optional[float], spec: str = ".3f") -> str:
        if x is None:
            return "-"
        return format(x, spec)

    lines: List[str] = []
    lines.append("# Triton ARM Autonomy Graduation")
    lines.append("")
    lines.append(f"_Generated at {generated_at}_")
    lines.append("")

    lines.append("## Graduation State")
    lines.append("")
    lines.append(f"**{state}**")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| previous_state | {previous_state or '-'} |")
    lines.append(f"| confidence | {confidence:.3f} |")
    lines.append(f"| autonomy_promotion_earned | {booleans['autonomy_promotion_earned']} |")
    lines.append(f"| assisted_mode_eligible | {booleans['assisted_mode_eligible']} |")
    lines.append(f"| auto_mode_candidate | {booleans['auto_mode_candidate']} |")
    lines.append(f"| auto_mode_approved | {booleans['auto_mode_approved']} |")
    lines.append(f"| operator_signoff_required | {booleans['operator_signoff_required']} |")
    lines.append(f"| autonomy_revoked | {booleans['autonomy_revoked']} |")
    lines.append("")

    lines.append("## Promotion Evidence")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| apprenticeship_verdict | {evidence['apprenticeship_verdict']} |")
    lines.append(f"| observations | {evidence['observations']} |")
    lines.append(f"| autonomy_readiness | {fmt(evidence.get('autonomy_readiness'))} |")
    lines.append(f"| shadow_win_rate | {fmt(evidence.get('shadow_win_rate'))} |")
    lines.append(f"| shadow_discipline_score | {fmt(evidence.get('shadow_discipline_score'))} |")
    lines.append(
        f"| shadow_drawdown_score (capital pres.) | {fmt(evidence.get('shadow_drawdown_score'))} |"
    )
    lines.append(f"| governance_quality | {fmt(evidence.get('governance_quality'))} |")
    lines.append(f"| system_health_status | {evidence.get('system_health_status')} |")
    lines.append(f"| certification_state | {evidence.get('certification_state')} |")
    lines.append(f"| certificate_confidence | {fmt(evidence.get('certificate_confidence'))} |")
    lines.append(f"| readiness_state | {evidence.get('readiness_state')} |")
    lines.append(f"| arm_mode | {evidence.get('arm_mode')} |")
    lines.append("")

    lines.append("## Why")
    lines.append("")
    for r in reasons:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Narrative")
    lines.append("")
    lines.append(_narrative(state, evidence, previous_state, confidence))
    lines.append("")
    return "\n".join(lines)


def _narrative(
    state: str,
    evidence: Dict[str, Any],
    previous_state: Optional[str],
    confidence: float,
) -> str:
    obs = evidence["observations"]
    verdict = evidence["apprenticeship_verdict"]
    readiness = evidence["autonomy_readiness"]
    rd_txt = f"{readiness:.2f}" if readiness is not None else "n/a"
    if state == STATE_MANUAL_LOCKED:
        return (
            f"Triton remains MANUAL_LOCKED. With {obs} labelled "
            f"observation(s) and apprenticeship verdict {verdict}, "
            f"there is not yet enough evidence -- or a safety floor "
            f"is asserted -- to promote autonomy. Confidence "
            f"{confidence:.2f}."
        )
    if state == STATE_AUTONOMY_REVOKED:
        return (
            f"Triton has been demoted to AUTONOMY_REVOKED from "
            f"{previous_state or 'a prior promoted state'}. Recent "
            f"shadow evidence shows material deterioration; autonomy "
            f"must be re-earned through sustained recovery. "
            f"Confidence {confidence:.2f}."
        )
    if state == STATE_ASSISTED_CANDIDATE:
        return (
            f"Triton is an ASSISTED_CANDIDATE: the trajectory is "
            f"positive (verdict {verdict}, readiness {rd_txt}) but "
            f"promotion criteria are not yet met. The apprenticeship "
            f"continues. Confidence {confidence:.2f}."
        )
    if state == STATE_ASSISTED_APPROVED:
        return (
            f"Triton is ASSISTED_APPROVED after {obs} labelled "
            f"observation(s) of stable positive shadow discipline. "
            f"Assisted-mode trials are permitted under continued "
            f"operator supervision. Confidence {confidence:.2f}."
        )
    if state == STATE_AUTO_DISABLED_CANDIDATE:
        return (
            f"Triton is an AUTO_DISABLED_CANDIDATE: shadow evidence "
            f"is strong (readiness {rd_txt} across {obs} obs), but "
            f"more shadow time is required before auto-mode can be "
            f"approved. Confidence {confidence:.2f}."
        )
    if state == STATE_AUTO_ALLOWED_CANDIDATE:
        return (
            f"Triton is an AUTO_ALLOWED_CANDIDATE: sustained strong "
            f"performance (readiness {rd_txt}, {obs} obs) makes "
            f"Triton eligible for auto-mode pending explicit operator "
            f"approval. Confidence {confidence:.2f}."
        )
    # AUTO_ALLOWED_APPROVED
    return (
        f"Triton has earned AUTO_ALLOWED_APPROVED autonomy with "
        f"sustained strong evidence ({obs} obs, readiness {rd_txt}). "
        f"Shadow evaluation continues in parallel as a regression "
        f"guard. Confidence {confidence:.2f}."
    )


# -----------------------------------------------------------
# Progression memory
# -----------------------------------------------------------
def _build_progression_row(
    *,
    timestamp: str,
    state: str,
    previous_state: Optional[str],
    evidence: Dict[str, Any],
    booleans: Dict[str, bool],
    confidence: float,
    reasons: List[str],
) -> Dict[str, Any]:
    return {
        "timestamp_utc": timestamp,
        "graduation_state": state,
        "previous_state": previous_state or "",
        "observations": evidence["observations"],
        "apprenticeship_verdict": evidence["apprenticeship_verdict"],
        "autonomy_readiness": evidence["autonomy_readiness"],
        "shadow_win_rate": evidence["shadow_win_rate"],
        "shadow_discipline_score": evidence["shadow_discipline_score"],
        "shadow_drawdown_score": evidence["shadow_drawdown_score"],
        "governance_quality": evidence["governance_quality"],
        "system_health_score": evidence["system_health_score"],
        "certificate_confidence": evidence["certificate_confidence"],
        "confidence": round(confidence, 6),
        "autonomy_promotion_earned": booleans["autonomy_promotion_earned"],
        "autonomy_revoked": booleans["autonomy_revoked"],
        "reason": " | ".join(reasons),
        "governance_booleans_json": json.dumps(booleans, default=_json_safe, sort_keys=True),
    }


def _merge_progression(
    existing: List[Dict[str, Any]],
    new_row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Append-only with dedup by timestamp_utc (idempotent re-runs)."""
    by_ts: Dict[str, Dict[str, Any]] = {}
    for r in existing:
        ts = str(r.get("timestamp_utc", ""))
        if not ts:
            continue
        by_ts[ts] = r
    by_ts[str(new_row.get("timestamp_utc", ""))] = new_row
    out = list(by_ts.values())
    for r in out:
        for c in PROGRESSION_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


def _previous_state_from_memory(memory_rows: List[Dict[str, str]]) -> Optional[str]:
    if not memory_rows:
        return None
    sorted_rows = sorted(memory_rows, key=lambda r: str(r.get("timestamp_utc", "")))
    return str(sorted_rows[-1].get("graduation_state") or "") or None


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_graduation(
    *,
    perf_summary: Dict[str, Any],
    perf_record: Dict[str, Any],
    arm_summary: Dict[str, Any],
    cert_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    readiness: Dict[str, Any],
    health: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    outcomes_rows: List[Dict[str, Any]],
    existing_memory_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    evidence = _extract_evidence(
        perf_summary=perf_summary,
        perf_record=perf_record,
        arm_summary=arm_summary,
        cert_summary=cert_summary,
        scorecard=scorecard,
        readiness=readiness,
        health=health,
        runtime_policy=runtime_policy,
    )

    previous_state = _previous_state_from_memory(existing_memory_rows)
    state, reasons = _classify(evidence=evidence, memory_rows=existing_memory_rows)
    booleans = _booleans_for(state)
    confidence = _confidence(evidence)
    recommendations = _recommendations(state, evidence)

    new_row = _build_progression_row(
        timestamp=timestamp,
        state=state,
        previous_state=previous_state,
        evidence=evidence,
        booleans=booleans,
        confidence=confidence,
        reasons=reasons,
    )
    merged_memory = _merge_progression(existing_memory_rows, new_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        previous_state=previous_state,
        reasons=reasons,
        evidence=evidence,
        booleans=booleans,
        confidence=confidence,
        recommendations=recommendations,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_autonomy_graduation_engine",
        "engine_version": 1,
        "graduation_state": state,
        "previous_state": previous_state,
        "state_changed": (previous_state is not None and previous_state != state),
        "reasons": reasons,
        "confidence": round(confidence, 6),
        "evidence": evidence,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "outcomes_history_rows": len(outcomes_rows),
        "progression_memory_size_after_append": len(merged_memory),
        "thresholds": {
            "assisted_approved": {
                "observations": THR_ASSISTED_APPROVED_OBS,
                "shadow_win_rate": THR_ASSISTED_APPROVED_WIN,
                "shadow_discipline_score": THR_ASSISTED_APPROVED_DISC,
                "governance_quality": THR_ASSISTED_APPROVED_GOV,
            },
            "auto_disabled_candidate": {
                "observations": THR_AUTO_DISABLED_OBS,
                "autonomy_readiness": THR_AUTO_DISABLED_READINESS,
            },
            "auto_allowed_candidate": {
                "observations": THR_AUTO_ALLOWED_CANDIDATE_OBS,
                "autonomy_readiness": THR_AUTO_ALLOWED_CANDIDATE_READINESS,
                "governance_quality": THR_AUTO_ALLOWED_CANDIDATE_GOV,
                "shadow_drawdown_score": THR_AUTO_ALLOWED_CANDIDATE_DRAWDOWN,
            },
            "auto_allowed_approved": {
                "observations": THR_AUTO_ALLOWED_APPROVED_OBS,
                "autonomy_readiness": THR_AUTO_ALLOWED_APPROVED_READINESS,
                "governance_quality": THR_AUTO_ALLOWED_APPROVED_GOV,
                "capital_preservation_score": THR_AUTO_ALLOWED_APPROVED_CAP_PRES,
            },
            "deterioration": {
                "win_rate_floor": DETERIORATION_WIN_FLOOR,
                "discipline_floor": DETERIORATION_DISCIPLINE_FLOOR,
                "governance_floor": DETERIORATION_GOV_FLOOR,
                "readiness_drop_max": DETERIORATION_READINESS_DROP,
            },
        },
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "governance_only": True,
        },
        "inputs_seen": {
            "shadow_performance_summary": bool(perf_summary),
            "shadow_performance_record": bool(perf_record),
            "shadow_outcomes_memory_rows": len(outcomes_rows),
            "arm_mode_governance_summary": bool(arm_summary),
            "autonomous_execution_certificate_summary": bool(cert_summary),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_readiness_summary": bool(readiness),
            "autonomous_system_health_summary": bool(health),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_progression_memory_rows": len(existing_memory_rows),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_autonomy_graduation_engine",
        "graduation_state": state,
        "previous_state": previous_state,
        "state_changed": record["state_changed"],
        "confidence": record["confidence"],
        "autonomy_promotion_earned": booleans["autonomy_promotion_earned"],
        "auto_mode_approved": booleans["auto_mode_approved"],
        "autonomy_revoked": booleans["autonomy_revoked"],
        "observations": evidence["observations"],
        "apprenticeship_verdict": evidence["apprenticeship_verdict"],
        "autonomy_readiness": evidence["autonomy_readiness"],
        "governance_quality": evidence["governance_quality"],
        "system_health_status": evidence["system_health_status"],
        "certification_state": evidence["certification_state"],
        "progression_memory_size": len(merged_memory),
        "n_recommendations": len(recommendations),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM autonomy graduation engine (Step 31). "
            "Converts shadow performance evaluation into an "
            "evidence-based autonomy classification. No broker calls; "
            "no portfolio mutation."
        ),
    )
    p.add_argument("--perf-summary", default=str(DEFAULT_PERF_SUMMARY))
    p.add_argument("--perf-record", default=str(DEFAULT_PERF_RECORD))
    p.add_argument("--outcomes-csv", default=str(DEFAULT_OUTCOMES_CSV))
    p.add_argument("--arm-summary", default=str(DEFAULT_ARM_SUMMARY))
    p.add_argument("--cert-summary", default=str(DEFAULT_CERT_SUMMARY))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--readiness", default=str(DEFAULT_READINESS))
    p.add_argument("--health", default=str(DEFAULT_HEALTH))
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
        "[ARM_GRADUATION] starting (read-only autonomy graduation; no broker calls)",
        flush=True,
    )

    perf_summary = _safe_read_json(
        Path(args.perf_summary), label="arm_shadow_performance_summary.json"
    )
    perf_record = _safe_read_json(Path(args.perf_record), label="arm_shadow_performance.json")
    arm_summary = _safe_read_json(Path(args.arm_summary), label="arm_mode_governance_summary.json")
    cert_summary = _safe_read_json(
        Path(args.cert_summary), label="autonomous_execution_certificate_summary.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    readiness = _safe_read_json(Path(args.readiness), label="autonomous_readiness_summary.json")
    health = _safe_read_json(Path(args.health), label="autonomous_system_health_summary.json")
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    outcomes_rows = _safe_read_csv_rows(
        Path(args.outcomes_csv), label="arm_shadow_outcomes_memory.csv"
    )
    existing_memory = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_autonomy_progression_memory.csv"
    )

    record, summary, md, merged_memory = build_graduation(
        perf_summary=perf_summary,
        perf_record=perf_record,
        arm_summary=arm_summary,
        cert_summary=cert_summary,
        scorecard=scorecard,
        readiness=readiness,
        health=health,
        runtime_policy=runtime_policy,
        outcomes_rows=outcomes_rows,
        existing_memory_rows=existing_memory,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=PROGRESSION_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    booleans = record["governance_booleans"]
    print(
        "[ARM_GRADUATION] "
        f"state={record['graduation_state']} "
        f"earned={booleans['autonomy_promotion_earned']} "
        f"approved={booleans['auto_mode_approved']} "
        f"revoked={booleans['autonomy_revoked']} "
        f"confidence={record['confidence']:.3f} "
        f"observations={record['evidence']['observations']}",
        flush=True,
    )
    print(
        "[ARM_GRADUATION_SAFETY] broker_calls=0 orders_placed=0 portfolio_mutated=False",
        flush=True,
    )
    print(
        f"[ARM_GRADUATION_OUT] json={Path(args.out_json).as_posix()} "
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
            raise RuntimeError(f"[ARM_GRADUATION_SAFETY] forbidden broker token detected: {tok!r}")


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
