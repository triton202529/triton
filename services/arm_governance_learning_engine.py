"""
ARM Governance Learning Engine -- Step 38.

Reads:
    data/results/arm_governance_recovery_effectiveness_summary.json  (Step 37)
    data/results/arm_governance_recovery_effectiveness.json          (Step 37)
    data/results/arm_governance_recovery_effectiveness_memory.csv  (Step 37)
    data/results/arm_governance_recovery_memory.csv                 (Step 36)
    data/results/arm_governance_drift_memory.csv                   (Step 35)
    data/results/arm_constitution_violation_memory.csv             (Step 32)
    data/results/arm_constitutional_precedent_memory.csv           (Step 33)
    data/results/arm_governance_posture_memory.csv                 (Step 34)
    data/results/arm_autonomy_progression_memory.csv               (Step 31)
    data/results/autonomous_governance_scorecard.json              (Step 19)
    data/results/runtime_policy_governed.json                        (Step 18)

Writes:
    data/results/arm_governance_learning.json
    data/results/arm_governance_learning.md
    data/results/arm_governance_learning_summary.json
    data/results/arm_governance_learning_memory.csv
    data/results/arm_governance_learning_memory.parquet

Purpose
-------
This engine answers:

    "What governance lessons should Triton permanently learn?"

It is Triton's institutional governance memory layer -- read-only synthesis
of failures, recoveries, and stabilizations into permanent learning signals.
Learning cannot override constitutional law, the constitutional court,
capital preservation, or operator supremacy.

Learning state cascade
----------------------
    1. GOVERNANCE_LEARNING_INSTITUTIONAL  persistent stable learning
    2. GOVERNANCE_LEARNING_MATURE         strong repeatability in recovery
    3. GOVERNANCE_LEARNING_ADAPTIVE       responding to prior failures
    4. GOVERNANCE_LEARNING_FORMING        repeated patterns emerging
    5. GOVERNANCE_LEARNING_MINIMAL        little historical evidence

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens are never written literally; an import-time
self-check raises if they ever appear.

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only learning memory keyed by timestamp.
* Missing inputs warn-and-continue; absent evidence defaults to
  GOVERNANCE_LEARNING_MINIMAL as the safe learning posture.
* main() returns 0 on success, 2 on output-write failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
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

DEFAULT_EFF_SUMMARY = RESULTS_DIR / "arm_governance_recovery_effectiveness_summary.json"
DEFAULT_EFF_RECORD = RESULTS_DIR / "arm_governance_recovery_effectiveness.json"
DEFAULT_EFF_MEM = RESULTS_DIR / "arm_governance_recovery_effectiveness_memory.csv"
DEFAULT_RECOVERY_MEM = RESULTS_DIR / "arm_governance_recovery_memory.csv"
DEFAULT_DRIFT_MEM = RESULTS_DIR / "arm_governance_drift_memory.csv"
DEFAULT_VIOLATION_MEM = RESULTS_DIR / "arm_constitution_violation_memory.csv"
DEFAULT_PRECEDENT_MEM = RESULTS_DIR / "arm_constitutional_precedent_memory.csv"
DEFAULT_POSTURE_MEM = RESULTS_DIR / "arm_governance_posture_memory.csv"
DEFAULT_PROGRESSION_MEM = RESULTS_DIR / "arm_autonomy_progression_memory.csv"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_learning.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_learning.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_learning_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_learning_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_learning_memory.parquet"


# -----------------------------------------------------------
# Learning state constants
# -----------------------------------------------------------
LEARNING_MINIMAL = "GOVERNANCE_LEARNING_MINIMAL"
LEARNING_FORMING = "GOVERNANCE_LEARNING_FORMING"
LEARNING_ADAPTIVE = "GOVERNANCE_LEARNING_ADAPTIVE"
LEARNING_MATURE = "GOVERNANCE_LEARNING_MATURE"
LEARNING_INSTITUTIONAL = "GOVERNANCE_LEARNING_INSTITUTIONAL"

STALE_LAW_MARKERS = frozenset(
    {
        "LAW_004_STALE_INTELLIGENCE_PROHIBITION",
        "STALE_INTELLIGENCE",
    }
)
CERT_LAW_MARKERS = frozenset({"LAW_002_CERTIFICATE_REQUIRED"})
DEFENSIVE_LAW_MARKERS = frozenset({"LAW_006_DEFENSIVE_POSTURE_REQUIREMENT"})

NEGATIVE_POSTURE = frozenset(
    {
        "GOVERNANCE_TIGHTEN",
        "GOVERNANCE_REVOKE_AUTONOMY",
        "GOVERNANCE_LOCKDOWN",
    }
)
RECOVERY_SUCCESS = frozenset({"RECOVERY_EFFECTIVE", "RECOVERY_IMPROVING"})
RECOVERY_FAILURE = frozenset({"RECOVERY_INEFFECTIVE", "RECOVERY_REGRESSING"})

LEARNING_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "learning_state",
    "learning_score",
    "lessons_detected",
    "learned_policies",
    "governance_confidence",
    "constitutional_pressure",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_GOVERNANCE_LEARNING_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(LEARNING_MEMORY_COLUMNS))
        for col in ("learning_score", "governance_confidence", "constitutional_pressure"):
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


def _sorted_rows(rows: List[Dict[str, str]], key: str) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda r: str(r.get(key, "")))


# -----------------------------------------------------------
# Evidence and pattern mining
# -----------------------------------------------------------
def _memory_depth(
    *,
    violation_mem: List[Dict[str, str]],
    precedent_mem: List[Dict[str, str]],
    posture_mem: List[Dict[str, str]],
    drift_mem: List[Dict[str, str]],
    recovery_mem: List[Dict[str, str]],
    eff_mem: List[Dict[str, str]],
    progression_mem: List[Dict[str, str]],
) -> int:
    return (
        len(violation_mem)
        + len(precedent_mem)
        + len(posture_mem)
        + len(drift_mem)
        + len(recovery_mem)
        + len(eff_mem)
        + len(progression_mem)
    )


def _current_signals(
    *,
    eff_summary: Dict[str, Any],
    eff_record: Dict[str, Any],
    scorecard: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    posture_mem: List[Dict[str, str]],
    drift_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    snap = eff_record.get("current_snapshot") or {}
    if posture_mem:
        last = _sorted_rows(posture_mem, "timestamp_utc")[-1]
        gov_conf = _to_float(last.get("confidence")) or 0.0
        const_pressure = (
            0.75
            if _norm_upper(last.get("constitutional_state")) == "CONSTITUTION_VIOLATED"
            else 0.3
        )
    else:
        gov_conf = _to_float(eff_summary.get("governance_confidence_delta")) or 0.0
        const_pressure = 0.5

    if drift_mem:
        last_d = _sorted_rows(drift_mem, "timestamp")[-1]
        const_pressure = _to_float(last_d.get("constitutional_pressure")) or const_pressure
        gov_conf = _to_float(last_d.get("governance_confidence")) or gov_conf

    regime = _norm_upper(runtime_policy.get("regime") or scorecard.get("regime"))
    return {
        "governance_confidence": _clamp(gov_conf, 0.0, 1.0),
        "constitutional_pressure": _clamp(const_pressure, 0.0, 1.0),
        "regime": regime,
        "effectiveness_state": _norm_upper(eff_summary.get("effectiveness_state")),
        "recovery_state": _norm_upper(eff_summary.get("recovery_state")),
        "drift_state": _norm_upper(eff_summary.get("drift_state")),
    }


def _detect_lessons(
    *,
    violation_mem: List[Dict[str, str]],
    precedent_mem: List[Dict[str, str]],
    posture_mem: List[Dict[str, str]],
    drift_mem: List[Dict[str, str]],
    recovery_mem: List[Dict[str, str]],
    eff_mem: List[Dict[str, str]],
    progression_mem: List[Dict[str, str]],
    signals: Dict[str, Any],
) -> List[Dict[str, Any]]:
    lessons: List[Dict[str, Any]] = []

    # ---- constitutional / stale intelligence ----
    law_counts = Counter(_norm_upper(r.get("law_id")) for r in violation_mem)
    stale_hits = sum(law_counts.get(m, 0) for m in STALE_LAW_MARKERS) + sum(
        1
        for r in violation_mem
        if "STALE" in _norm_upper(r.get("reason")) or _norm_upper(r.get("system_state")) == "STALE"
    )
    if stale_hits >= 1:
        repeat = stale_hits >= 2
        lessons.append(
            {
                "lesson_id": "LESSON_STALE_INTELLIGENCE",
                "text": (
                    "stale intelligence repeatedly causes constitutional violations"
                    if repeat
                    else "stale intelligence precedes constitutional violations"
                ),
                "confidence": 0.85 if repeat else 0.60,
                "repeat_count": stale_hits,
            }
        )

    overruled = sum(1 for r in precedent_mem if _norm_upper(r.get("ruling")) == "COURT_OVERRULED")
    if overruled >= 1:
        lessons.append(
            {
                "lesson_id": "LESSON_COURT_OVERRULING",
                "text": (
                    "repeated court overruling signals governance inconsistency"
                    if overruled >= 2
                    else "constitutional court overruling indicates governance pressure"
                ),
                "confidence": min(0.90, 0.55 + overruled * 0.15),
                "repeat_count": overruled,
            }
        )

    # ---- recovery / manual ----
    lockdown_recovery = sum(
        1 for r in recovery_mem if _norm_upper(r.get("recovery_state")) == "RECOVERY_LOCKDOWN"
    )
    manual_posture = sum(
        1 for r in posture_mem if _norm_upper(r.get("autonomy_state")) == "MANUAL_LOCKED"
    )
    if lockdown_recovery >= 1 or manual_posture >= 1:
        lessons.append(
            {
                "lesson_id": "LESSON_MANUAL_RECOVERY",
                "text": "manual-only recovery reduces failure risk during constitutional crisis",
                "confidence": 0.75 if lockdown_recovery >= 1 else 0.55,
                "repeat_count": lockdown_recovery + manual_posture,
            }
        )

    # ---- cash / defensive ----
    defensive_regimes = sum(1 for r in posture_mem if _norm_upper(r.get("regime")) == "DEFENSIVE")
    defensive_violations = sum(law_counts.get(m, 0) for m in DEFENSIVE_LAW_MARKERS)
    target_cash = _to_float(signals.get("target_cash_pct"))
    if defensive_regimes >= 1 or defensive_violations >= 1 or signals.get("regime") == "DEFENSIVE":
        lessons.append(
            {
                "lesson_id": "LESSON_ELEVATED_CASH",
                "text": "increased cash posture improves stability under defensive regimes",
                "confidence": 0.70,
                "repeat_count": defensive_regimes + defensive_violations,
            }
        )

    # ---- autonomy escalation ----
    low_conf_revoke = sum(
        1
        for r in posture_mem
        if _norm_upper(r.get("ruling")) in NEGATIVE_POSTURE
        and (_to_float(r.get("confidence")) or 1.0) < 0.35
    )
    if low_conf_revoke >= 1:
        lessons.append(
            {
                "lesson_id": "LESSON_LOW_CONF_AUTONOMY",
                "text": "autonomy escalation during low confidence worsens outcomes",
                "confidence": 0.80,
                "repeat_count": low_conf_revoke,
            }
        )

    # ---- apprenticeship ----
    learning_verdicts = sum(
        1 for r in posture_mem if _norm_upper(r.get("apprenticeship_verdict")) == "LEARNING"
    )
    if learning_verdicts >= 1:
        lessons.append(
            {
                "lesson_id": "LESSON_APPRENTICESHIP",
                "text": "shadow apprenticeship extension improves governance stability",
                "confidence": 0.65,
                "repeat_count": learning_verdicts,
            }
        )

    # ---- governance tightening ----
    tighten_postures = sum(
        1 for r in posture_mem if _norm_upper(r.get("ruling")) == "GOVERNANCE_TIGHTEN"
    )
    if tighten_postures >= 1 and overruled >= 1:
        lessons.append(
            {
                "lesson_id": "LESSON_TIGHTEN_OVERRULING",
                "text": "governance tightening reduces overruling frequency after drift",
                "confidence": 0.60,
                "repeat_count": tighten_postures,
            }
        )

    # ---- recovery effectiveness ----
    success_count = sum(
        1 for r in eff_mem if _norm_upper(r.get("effectiveness_state")) in RECOVERY_SUCCESS
    )
    failure_count = sum(
        1 for r in eff_mem if _norm_upper(r.get("effectiveness_state")) in RECOVERY_FAILURE
    )
    if success_count >= 1:
        lessons.append(
            {
                "lesson_id": "LESSON_RECOVERY_SUCCESS",
                "text": "structured recovery programs produce measurable stabilization",
                "confidence": 0.70,
                "repeat_count": success_count,
            }
        )
    if failure_count >= 1:
        lessons.append(
            {
                "lesson_id": "LESSON_RECOVERY_FAILURE",
                "text": "failed recoveries require governance strategy reassessment",
                "confidence": 0.65,
                "repeat_count": failure_count,
            }
        )

    # ---- autonomy regression ----
    if len(progression_mem) >= 2:
        sorted_p = _sorted_rows(progression_mem, "timestamp_utc")
        first = _norm_upper(sorted_p[0].get("graduation_state"))
        last = _norm_upper(sorted_p[-1].get("graduation_state"))
        rank = {"MANUAL_LOCKED": 0, "ASSISTED_APPROVED": 2, "AUTO_ALLOWED_APPROVED": 4}
        if rank.get(last, 1) < rank.get(first, 1):
            lessons.append(
                {
                    "lesson_id": "LESSON_AUTONOMY_REGRESSION",
                    "text": "premature autonomy promotion leads to regression and lockdown",
                    "confidence": 0.75,
                    "repeat_count": 1,
                }
            )

    # dedupe by lesson_id, keep highest confidence
    keyed: Dict[str, Dict[str, Any]] = {}
    for les in lessons:
        lid = les["lesson_id"]
        if lid not in keyed or les["confidence"] > keyed[lid]["confidence"]:
            keyed[lid] = les
    return list(keyed.values())


def _derive_policies(lessons: List[Dict[str, Any]], signals: Dict[str, Any]) -> List[str]:
    policies: List[str] = []
    ids = {l["lesson_id"] for l in lessons}

    if "LESSON_STALE_INTELLIGENCE" in ids:
        policies.append("require pipeline freshness before autonomy escalation")
    if "LESSON_ELEVATED_CASH" in ids or signals.get("regime") == "DEFENSIVE":
        policies.append("prefer elevated cash posture under uncertainty")
    if "LESSON_APPRENTICESHIP" in ids or "LESSON_LOW_CONF_AUTONOMY" in ids:
        policies.append("extend apprenticeship during governance drift")
    if "LESSON_ELEVATED_CASH" in ids or signals.get("regime") == "DEFENSIVE":
        policies.append("tighten thresholds during defensive regimes")
    if "LESSON_COURT_OVERRULING" in ids or "LESSON_MANUAL_RECOVERY" in ids:
        policies.append("force manual mode after repeated court overruling")
    if "LESSON_LOW_CONF_AUTONOMY" in ids:
        policies.append("require stronger confidence before autonomy escalation")
    if "LESSON_TIGHTEN_OVERRULING" in ids:
        policies.append("apply governance tightening when overruling frequency rises")
    if "LESSON_RECOVERY_SUCCESS" in ids:
        policies.append("maintain structured recovery protocols after drift detection")

    if not policies:
        policies.append("continue accumulating governance memory before policy derivation")

    seen: set = set()
    out: List[str] = []
    for p in policies:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# -----------------------------------------------------------
# Learning score and classification
# -----------------------------------------------------------
def _learning_score(
    *,
    lessons: List[Dict[str, Any]],
    memory_depth: int,
    eff_mem: List[Dict[str, str]],
    drift_mem: List[Dict[str, str]],
    violation_mem: List[Dict[str, str]],
) -> float:
    if not lessons and memory_depth == 0:
        return 0.0

    repeatability = 0.0
    if lessons:
        repeated = sum(1 for l in lessons if l.get("repeat_count", 0) >= 2)
        repeatability = repeated / len(lessons)

    success_n = sum(
        1 for r in eff_mem if _norm_upper(r.get("effectiveness_state")) in RECOVERY_SUCCESS
    )
    total_eff = len(eff_mem) or 1
    recovery_success_rate = success_n / total_eff

    stabilization = 0.0
    if len(drift_mem) >= 2:
        sorted_d = _sorted_rows(drift_mem, "timestamp")
        scores = [_to_float(r.get("drift_score")) or 0.0 for r in sorted_d]
        if scores[-1] < scores[0]:
            stabilization = _clamp((scores[0] - scores[-1]) / max(scores[0], 0.01), 0.0, 1.0)

    conf_improvement = 0.0
    if len(drift_mem) >= 2:
        sorted_d = _sorted_rows(drift_mem, "timestamp")
        confs = [_to_float(r.get("governance_confidence")) or 0.0 for r in sorted_d]
        if confs[-1] > confs[0]:
            conf_improvement = _clamp(confs[-1] - confs[0], 0.0, 1.0)

    pressure_reduction = 0.0
    if len(drift_mem) >= 2:
        sorted_d = _sorted_rows(drift_mem, "timestamp")
        pressures = [_to_float(r.get("constitutional_pressure")) or 0.0 for r in sorted_d]
        if pressures[-1] < pressures[0]:
            pressure_reduction = _clamp(pressures[0] - pressures[-1], 0.0, 1.0)

    avg_lesson_conf = sum(l["confidence"] for l in lessons) / len(lessons) if lessons else 0.0
    depth_factor = _clamp(memory_depth / 20.0, 0.0, 1.0)

    raw = (
        repeatability * 0.20
        + recovery_success_rate * 0.18
        + stabilization * 0.15
        + conf_improvement * 0.15
        + pressure_reduction * 0.12
        + avg_lesson_conf * 0.10
        + depth_factor * 0.10
    )

    failure_n = sum(
        1 for r in eff_mem if _norm_upper(r.get("effectiveness_state")) in RECOVERY_FAILURE
    )
    penalty = failure_n * 0.08
    recurring_crises = sum(1 for r in violation_mem if _norm_upper(r.get("severity")) == "CRITICAL")
    if recurring_crises >= 3:
        penalty += 0.10

    regressions = sum(
        1 for r in eff_mem if _norm_upper(r.get("effectiveness_state")) == "RECOVERY_REGRESSING"
    )
    penalty += regressions * 0.06

    return round(_clamp(raw - penalty, 0.0, 1.0), 6)


def _classify_learning_state(
    *,
    learning_score: float,
    lessons: List[Dict[str, Any]],
    memory_depth: int,
    have_evidence: bool,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    n_lessons = len(lessons)
    repeated = sum(1 for l in lessons if l.get("repeat_count", 0) >= 2)

    if not have_evidence:
        reasons.append("no upstream learning evidence; defaulting to MINIMAL")
        return LEARNING_MINIMAL, reasons

    if learning_score >= 0.80 and memory_depth >= 15 and n_lessons >= 6 and repeated >= 3:
        reasons.append("persistent stable learning across governance memory")
        reasons.append("institutional habits influencing recovery outcomes")
        return LEARNING_INSTITUTIONAL, reasons

    if learning_score >= 0.65 and memory_depth >= 8 and n_lessons >= 5 and repeated >= 2:
        reasons.append("strong repeatability in recovery and stabilization")
        return LEARNING_MATURE, reasons

    if learning_score >= 0.45 and n_lessons >= 3 and repeated >= 1:
        reasons.append("governance responding to prior failures")
        if any(l["lesson_id"] == "LESSON_RECOVERY_SUCCESS" for l in lessons):
            reasons.append("successful stabilization patterns detected")
        return LEARNING_ADAPTIVE, reasons

    if memory_depth >= 3 or n_lessons >= 2:
        reasons.append("repeated governance patterns emerging")
        if memory_depth >= 5:
            reasons.append("recovery observations increasing")
        return LEARNING_FORMING, reasons

    reasons.append("little historical evidence; no stable lessons yet")
    return LEARNING_MINIMAL, reasons


# -----------------------------------------------------------
# Booleans, rationale, recommendations
# -----------------------------------------------------------
def _learning_booleans(
    state: str,
    lessons: List[Dict[str, Any]],
    policies: List[str],
    signals: Dict[str, Any],
) -> Dict[str, bool]:
    constitutional = any(
        l["lesson_id"] in ("LESSON_STALE_INTELLIGENCE", "LESSON_COURT_OVERRULING") for l in lessons
    )
    recovery_lessons = any(
        l["lesson_id"]
        in ("LESSON_RECOVERY_SUCCESS", "LESSON_MANUAL_RECOVERY", "LESSON_RECOVERY_FAILURE")
        for l in lessons
    )
    return {
        "governance_learning_active": state != LEARNING_MINIMAL,
        "learned_policy_adjustments_available": len(policies) >= 2
        and state not in (LEARNING_MINIMAL,),
        "governance_memory_reliable": state in (LEARNING_MATURE, LEARNING_INSTITUTIONAL),
        "recovery_lessons_detected": recovery_lessons,
        "constitutional_patterns_detected": constitutional,
        "operator_review_recommended": (
            signals.get("constitutional_pressure", 0) >= 0.55
            or signals.get("effectiveness_state") in RECOVERY_FAILURE
        ),
    }


def _recommendations(state: str, booleans: Dict[str, bool], policies: List[str]) -> List[str]:
    recs: List[str] = []
    if booleans["learned_policy_adjustments_available"]:
        recs.append("Permanently tighten governance during low confidence")
    if booleans["constitutional_patterns_detected"]:
        recs.append("Require extended apprenticeship after constitutional crises")
    if booleans["recovery_lessons_detected"]:
        recs.append("Preserve elevated cash posture during instability")
    recs.append("Increase monitoring under defensive regimes")
    recs.append("Continue governance learning accumulation")
    if state in (LEARNING_MATURE, LEARNING_INSTITUTIONAL):
        recs.append("Apply learned policies to future recovery planning")
    if booleans["operator_review_recommended"]:
        recs.append("Escalate operator review for constitutional pattern validation")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(lessons: List[Dict[str, Any]], reasons: List[str]) -> str:
    stale = next((l for l in lessons if l["lesson_id"] == "LESSON_STALE_INTELLIGENCE"), None)
    if stale and stale.get("repeat_count", 0) >= 2:
        return (
            "Triton learned that stale intelligence consistently precedes "
            "constitutional overruling, making pipeline freshness a governance prerequisite."
        )
    over = next((l for l in lessons if l["lesson_id"] == "LESSON_COURT_OVERRULING"), None)
    if over:
        return (
            "Triton learned that court overruling patterns signal governance "
            "inconsistency requiring manual supervision and tightened thresholds."
        )
    parts = [l["text"] for l in lessons[:2]] or reasons[:2] or ["governance memory accumulating"]
    return f"Triton learned that {'; '.join(parts)}."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    score: float,
    lessons: List[Dict[str, Any]],
    policies: List[str],
    booleans: Dict[str, bool],
    signals: Dict[str, Any],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
) -> str:
    lines = [
        "# Triton Governance Learning",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Learning State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| learning_score | {score:.3f} |",
        f"| governance_learning_active | {booleans['governance_learning_active']} |",
        f"| governance_memory_reliable | {booleans['governance_memory_reliable']} |",
        f"| learned_policy_adjustments_available | {booleans['learned_policy_adjustments_available']} |",
        f"| governance_confidence | {signals['governance_confidence']:.3f} |",
        f"| constitutional_pressure | {signals['constitutional_pressure']:.3f} |",
        "",
        "## Lessons Learned",
        "",
    ]
    if lessons:
        lines.append("| lesson | confidence | repeats |")
        lines.append("|---|---|---|")
        for l in lessons:
            lines.append(f"| {l['text']} | {l['confidence']:.2f} | {l.get('repeat_count', 0)} |")
    else:
        lines.append("_No stable lessons detected yet._")
    lines.extend(
        [
            "",
            "## Learned Policies",
            "",
        ]
    )
    for p in policies:
        lines.append(f"- {p}")
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )
    for r in recommendations:
        lines.append(f"- {r}")
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
            "## Narrative",
            "",
            rationale,
            "",
            f"Learning score {score:.2f} reflects lesson repeatability, recovery success "
            f"frequency, stabilization persistence, and constitutional pressure trends. "
            f"Institutional learning is read-only and cannot override constitutional law "
            f"or operator supremacy.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Learning memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    score: float,
    lessons: List[Dict[str, Any]],
    policies: List[str],
    signals: Dict[str, Any],
    rationale: str,
) -> Dict[str, Any]:
    lesson_text = "|".join(l["lesson_id"] for l in lessons)
    return {
        "timestamp": timestamp,
        "learning_state": state,
        "learning_score": round(score, 6),
        "lessons_detected": lesson_text,
        "learned_policies": "|".join(policies),
        "governance_confidence": signals["governance_confidence"],
        "constitutional_pressure": signals["constitutional_pressure"],
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
        for c in LEARNING_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_governance_learning(
    *,
    eff_summary: Dict[str, Any],
    eff_record: Dict[str, Any],
    eff_mem: List[Dict[str, str]],
    recovery_mem: List[Dict[str, str]],
    drift_mem: List[Dict[str, str]],
    violation_mem: List[Dict[str, str]],
    precedent_mem: List[Dict[str, str]],
    posture_mem: List[Dict[str, str]],
    progression_mem: List[Dict[str, str]],
    scorecard: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    existing_learning_mem: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    signals = _current_signals(
        eff_summary=eff_summary,
        eff_record=eff_record,
        scorecard=scorecard,
        runtime_policy=runtime_policy,
        posture_mem=posture_mem,
        drift_mem=drift_mem,
    )
    signals["target_cash_pct"] = _to_float(runtime_policy.get("target_cash_pct"))

    memory_depth = _memory_depth(
        violation_mem=violation_mem,
        precedent_mem=precedent_mem,
        posture_mem=posture_mem,
        drift_mem=drift_mem,
        recovery_mem=recovery_mem,
        eff_mem=eff_mem,
        progression_mem=progression_mem,
    )

    have_evidence = memory_depth > 0 or bool(eff_summary)

    lessons = _detect_lessons(
        violation_mem=violation_mem,
        precedent_mem=precedent_mem,
        posture_mem=posture_mem,
        drift_mem=drift_mem,
        recovery_mem=recovery_mem,
        eff_mem=eff_mem,
        progression_mem=progression_mem,
        signals=signals,
    )

    policies = _derive_policies(lessons, signals)
    score = _learning_score(
        lessons=lessons,
        memory_depth=memory_depth,
        eff_mem=eff_mem,
        drift_mem=drift_mem,
        violation_mem=violation_mem,
    )
    state, reasons = _classify_learning_state(
        learning_score=score,
        lessons=lessons,
        memory_depth=memory_depth,
        have_evidence=have_evidence,
    )

    booleans = _learning_booleans(state, lessons, policies, signals)
    recommendations = _recommendations(state, booleans, policies)
    rationale = _build_rationale(lessons, reasons)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        score=score,
        lessons=lessons,
        policies=policies,
        signals=signals,
        rationale=rationale,
    )
    merged_memory = _merge_memory(existing_learning_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        score=score,
        lessons=lessons,
        policies=policies,
        booleans=booleans,
        signals=signals,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_learning_engine",
        "engine_version": 1,
        "learning_state": state,
        "learning_score": score,
        "learning_reasons": reasons,
        "lessons_learned": lessons,
        "learned_policies": policies,
        "current_signals": signals,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "memory_depth": memory_depth,
        "constitutional_supremacy_note": (
            "Institutional governance learning is read-only. "
            "Learned policies cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "learning_memory_size_after_append": len(merged_memory),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "learning_only": True,
        },
        "inputs_seen": {
            "arm_governance_recovery_effectiveness_summary": bool(eff_summary),
            "arm_governance_recovery_effectiveness_record": bool(eff_record),
            "arm_governance_recovery_effectiveness_memory_rows": len(eff_mem),
            "arm_governance_recovery_memory_rows": len(recovery_mem),
            "arm_governance_drift_memory_rows": len(drift_mem),
            "arm_constitution_violation_memory_rows": len(violation_mem),
            "arm_constitutional_precedent_memory_rows": len(precedent_mem),
            "arm_governance_posture_memory_rows": len(posture_mem),
            "arm_autonomy_progression_memory_rows": len(progression_mem),
            "autonomous_governance_scorecard": bool(scorecard),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_learning_memory_rows": len(existing_learning_mem),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_learning_engine",
        "learning_state": state,
        "learning_score": score,
        "governance_learning_active": booleans["governance_learning_active"],
        "learned_policy_adjustments_available": booleans["learned_policy_adjustments_available"],
        "governance_memory_reliable": booleans["governance_memory_reliable"],
        "recovery_lessons_detected": booleans["recovery_lessons_detected"],
        "constitutional_patterns_detected": booleans["constitutional_patterns_detected"],
        "operator_review_recommended": booleans["operator_review_recommended"],
        "n_lessons": len(lessons),
        "n_learned_policies": len(policies),
        "n_recommendations": len(recommendations),
        "governance_confidence": signals["governance_confidence"],
        "constitutional_pressure": signals["constitutional_pressure"],
        "memory_depth": memory_depth,
        "learning_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance learning engine (Step 38). "
            "Synthesizes institutional governance memory. No broker calls."
        ),
    )
    p.add_argument("--eff-summary", default=str(DEFAULT_EFF_SUMMARY))
    p.add_argument("--eff-record", default=str(DEFAULT_EFF_RECORD))
    p.add_argument("--eff-mem", default=str(DEFAULT_EFF_MEM))
    p.add_argument("--recovery-mem", default=str(DEFAULT_RECOVERY_MEM))
    p.add_argument("--drift-mem", default=str(DEFAULT_DRIFT_MEM))
    p.add_argument("--violation-mem", default=str(DEFAULT_VIOLATION_MEM))
    p.add_argument("--precedent-mem", default=str(DEFAULT_PRECEDENT_MEM))
    p.add_argument("--posture-mem", default=str(DEFAULT_POSTURE_MEM))
    p.add_argument("--progression-mem", default=str(DEFAULT_PROGRESSION_MEM))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
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
        "[ARM_GOVERNANCE_LEARNING] starting "
        "(read-only institutional governance learning; no broker calls)",
        flush=True,
    )

    eff_summary = _safe_read_json(
        Path(args.eff_summary), label="arm_governance_recovery_effectiveness_summary.json"
    )
    eff_record = _safe_read_json(
        Path(args.eff_record), label="arm_governance_recovery_effectiveness.json"
    )
    eff_mem = _safe_read_csv_rows(
        Path(args.eff_mem), label="arm_governance_recovery_effectiveness_memory.csv"
    )
    recovery_mem = _safe_read_csv_rows(
        Path(args.recovery_mem), label="arm_governance_recovery_memory.csv"
    )
    drift_mem = _safe_read_csv_rows(Path(args.drift_mem), label="arm_governance_drift_memory.csv")
    violation_mem = _safe_read_csv_rows(
        Path(args.violation_mem), label="arm_constitution_violation_memory.csv"
    )
    precedent_mem = _safe_read_csv_rows(
        Path(args.precedent_mem), label="arm_constitutional_precedent_memory.csv"
    )
    posture_mem = _safe_read_csv_rows(
        Path(args.posture_mem), label="arm_governance_posture_memory.csv"
    )
    progression_mem = _safe_read_csv_rows(
        Path(args.progression_mem), label="arm_autonomy_progression_memory.csv"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_learning_memory.csv"
    )

    record, summary, md, merged_memory = build_governance_learning(
        eff_summary=eff_summary,
        eff_record=eff_record,
        eff_mem=eff_mem,
        recovery_mem=recovery_mem,
        drift_mem=drift_mem,
        violation_mem=violation_mem,
        precedent_mem=precedent_mem,
        posture_mem=posture_mem,
        progression_mem=progression_mem,
        scorecard=scorecard,
        runtime_policy=runtime_policy,
        existing_learning_mem=existing_mem,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=LEARNING_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    booleans = record["governance_booleans"]
    print(
        "[ARM_GOVERNANCE_LEARNING] "
        f"state={record['learning_state']} "
        f"score={record['learning_score']:.3f} "
        f"lessons={len(record['lessons_learned'])} "
        f"memory_reliable={booleans['governance_memory_reliable']} "
        f"confidence={summary['governance_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_GOVERNANCE_LEARNING_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False",
        flush=True,
    )
    print(
        f"[ARM_GOVERNANCE_LEARNING_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_GOVERNANCE_LEARNING_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
