"""
ARM Governance Drift Detection Engine -- Step 35.

Reads:
    data/results/arm_governance_posture_memory.csv          (Step 34)
    data/results/arm_supreme_governance_council_summary.json (Step 34)
    data/results/arm_constitutional_precedent_memory.csv    (Step 33)
    data/results/arm_constitution_violation_memory.csv      (Step 32)
    data/results/arm_shadow_performance_summary.json        (Step 30)
    data/results/arm_autonomy_progression_memory.csv        (Step 31)
    data/results/autonomous_governance_scorecard.json       (Step 19)
    data/results/autonomous_system_health_summary.json      (Step 20)
    data/results/autonomous_readiness_summary.json          (Step 21)
    data/results/runtime_policy_governed.json               (Step 18)

Writes:
    data/results/arm_governance_drift_detection.json
    data/results/arm_governance_drift_detection.md
    data/results/arm_governance_drift_detection_summary.json
    data/results/arm_governance_drift_memory.csv
    data/results/arm_governance_drift_memory.parquet

Purpose
-------
This engine answers:

    "Is governance drifting in the wrong direction?"

It is Triton's early-warning governance layer -- read-only drift detection
that surfaces slow deterioration, instability, or unhealthy governance trends
before governance failure becomes obvious. Constitutional law remains supreme.

Drift state cascade
-------------------
    1. GOVERNANCE_FAILURE_RISK   severe degradation, collapse likely
    2. GOVERNANCE_UNSTABLE       constitutional pressure, inconsistency
    3. GOVERNANCE_DRIFTING       worsening trends, repeated overruling
    4. GOVERNANCE_EARLY_WARNING  mild deterioration, caution warranted
    5. GOVERNANCE_STABLE         healthy posture trend

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens are never written literally; an import-time
self-check raises if they ever appear.

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only drift memory keyed by timestamp_utc.
* Missing inputs warn-and-continue; absent evidence defaults to
  GOVERNANCE_UNSTABLE as the safe early-warning posture.
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

DEFAULT_POSTURE_MEM = RESULTS_DIR / "arm_governance_posture_memory.csv"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_PRECEDENT_MEM = RESULTS_DIR / "arm_constitutional_precedent_memory.csv"
DEFAULT_VIOLATION_MEM = RESULTS_DIR / "arm_constitution_violation_memory.csv"
DEFAULT_SHADOW_PERF = RESULTS_DIR / "arm_shadow_performance_summary.json"
DEFAULT_PROGRESSION_MEM = RESULTS_DIR / "arm_autonomy_progression_memory.csv"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_GOV_POLICY = RESULTS_DIR / "runtime_policy_governed.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_drift_detection.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_drift_detection.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_drift_detection_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_drift_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_drift_memory.parquet"


# -----------------------------------------------------------
# Drift state constants
# -----------------------------------------------------------
STATE_STABLE = "GOVERNANCE_STABLE"
STATE_EARLY_WARNING = "GOVERNANCE_EARLY_WARNING"
STATE_DRIFTING = "GOVERNANCE_DRIFTING"
STATE_UNSTABLE = "GOVERNANCE_UNSTABLE"
STATE_FAILURE_RISK = "GOVERNANCE_FAILURE_RISK"

COURT_OVERRULED = "COURT_OVERRULED"
CONSTITUTION_VIOLATED = "CONSTITUTION_VIOLATED"

NEGATIVE_RULINGS = frozenset(
    {
        "GOVERNANCE_TIGHTEN",
        "GOVERNANCE_REVOKE_AUTONOMY",
        "GOVERNANCE_LOCKDOWN",
    }
)

PROHIBITED_HEALTH = frozenset({"STALE", "CRITICAL", "OFFLINE", "DEGRADED"})

AUTONOMY_RANK: Dict[str, int] = {
    "MANUAL_LOCKED": 0,
    "MANUAL": 0,
    "ASSISTED_CANDIDATE": 1,
    "ASSISTED_APPROVED": 2,
    "AUTO_ALLOWED_CANDIDATE": 3,
    "AUTO_ALLOWED_APPROVED": 4,
}

DRIFT_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "drift_state",
    "drift_score",
    "constitutional_pressure",
    "overruling_frequency",
    "governance_confidence",
    "shadow_performance",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_DRIFT_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(DRIFT_MEMORY_COLUMNS))
        for col in (
            "drift_score",
            "constitutional_pressure",
            "overruling_frequency",
            "governance_confidence",
            "shadow_performance",
        ):
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
# Signal extraction
# -----------------------------------------------------------
def _extract_snapshot(
    *,
    council_summary: Dict[str, Any],
    shadow_perf: Dict[str, Any],
    scorecard: Dict[str, Any],
    health: Dict[str, Any],
    readiness: Dict[str, Any],
    runtime_policy: Dict[str, Any],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    governance_quality = _to_float(scores.get("governance_quality_score")) or 0.5
    health_score = _to_float(health.get("system_health_score")) or 0.5
    health_status = _norm_upper(health.get("overall_status"))
    readiness_score = _to_float(readiness.get("readiness_score")) or 0.5
    readiness_state = _norm_upper(readiness.get("readiness_state"))

    shadow_readiness = _to_float(shadow_perf.get("shadow_autonomy_readiness_score")) or 0.5
    shadow_discipline = _to_float(shadow_perf.get("shadow_discipline_score"))
    shadow_alpha = _to_float(shadow_perf.get("shadow_alpha_score")) or 0.5
    trajectory_improving = bool(shadow_perf.get("trajectory_improving"))
    apprenticeship_verdict = _norm_upper(shadow_perf.get("apprenticeship_verdict"))

    council_confidence = _to_float(council_summary.get("council_confidence")) or 0.0
    council_ruling = _norm_upper(council_summary.get("governance_ruling"))
    constitution_state = _norm_upper(council_summary.get("constitution_state"))
    court_ruling = _norm_upper(council_summary.get("court_ruling"))
    graduation_state = _norm_upper(council_summary.get("graduation_state"))
    regime = _norm_upper(runtime_policy.get("regime") or scorecard.get("regime"))
    target_cash = _to_float(runtime_policy.get("target_cash_pct")) or 0.0

    return {
        "council_confidence": _clamp(council_confidence, 0.0, 1.0),
        "council_ruling": council_ruling,
        "constitution_state": constitution_state,
        "court_ruling": court_ruling,
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
    }


def _confidence_trend(posture_rows: List[Dict[str, str]]) -> Tuple[float, str]:
    """Return (trend_delta, label). Negative delta = deteriorating."""
    if len(posture_rows) < 2:
        return 0.0, "insufficient_posture_history"
    sorted_rows = _sorted_rows(posture_rows, "timestamp_utc")
    confs: List[float] = []
    for r in sorted_rows:
        v = _to_float(r.get("confidence"))
        if v is not None:
            confs.append(v)
    if len(confs) < 2:
        return 0.0, "insufficient_confidence_history"
    delta = confs[-1] - confs[0]
    if delta <= -0.15:
        label = "declining"
    elif delta <= -0.05:
        label = "weakening"
    elif delta >= 0.05:
        label = "improving"
    else:
        label = "flat"
    return delta, label


def _negative_ruling_rate(posture_rows: List[Dict[str, str]]) -> float:
    if not posture_rows:
        return 0.0
    n_neg = sum(1 for r in posture_rows if _norm_upper(r.get("ruling")) in NEGATIVE_RULINGS)
    return n_neg / len(posture_rows)


def _violation_stats(violation_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    total = len(violation_rows)
    unresolved = sum(
        1
        for r in violation_rows
        if str(r.get("resolved", "")).strip().lower() in ("false", "0", "no")
    )
    critical = sum(1 for r in violation_rows if _norm_upper(r.get("severity")) == "CRITICAL")
    pressure = (
        _clamp(
            (total * 0.15 + unresolved * 0.25 + critical * 0.35) / max(total, 1),
            0.0,
            1.0,
        )
        if total
        else 0.0
    )
    if total >= 3 and unresolved >= 2:
        pressure = max(pressure, 0.65)
    if critical >= 2:
        pressure = max(pressure, 0.75)
    return {
        "total": total,
        "unresolved": unresolved,
        "critical": critical,
        "constitutional_pressure": round(pressure, 6),
    }


def _overruling_frequency(precedent_rows: List[Dict[str, str]]) -> float:
    if not precedent_rows:
        return 0.0
    n_over = sum(1 for r in precedent_rows if _norm_upper(r.get("ruling")) == COURT_OVERRULED)
    return round(n_over / len(precedent_rows), 6)


def _shadow_performance_score(snapshot: Dict[str, Any]) -> float:
    parts: List[float] = [snapshot["shadow_readiness"], snapshot["shadow_alpha"]]
    if snapshot["shadow_discipline"] is not None:
        parts.append(_clamp(snapshot["shadow_discipline"], 0.0, 1.0))
    base = sum(parts) / len(parts)
    if not snapshot["trajectory_improving"]:
        base *= 0.92
    if snapshot["apprenticeship_verdict"] in ("LEARNING", "AUTONOMY_NOT_READY"):
        base *= 0.90
    return round(_clamp(base, 0.0, 1.0), 6)


def _shadow_trend(
    progression_rows: List[Dict[str, str]],
    snapshot: Dict[str, Any],
) -> Tuple[float, str]:
    """Return (trend_delta, label) for shadow/autonomy readiness."""
    if len(progression_rows) < 2:
        return 0.0, "insufficient_progression_history"
    sorted_rows = _sorted_rows(progression_rows, "timestamp_utc")
    vals: List[float] = []
    for r in sorted_rows:
        v = _to_float(r.get("autonomy_readiness")) or _to_float(r.get("shadow_discipline_score"))
        if v is not None:
            vals.append(v)
    if len(vals) < 2:
        return 0.0, "insufficient_shadow_history"
    delta = vals[-1] - vals[0]
    if delta <= -0.10:
        label = "deteriorating"
    elif delta <= -0.03:
        label = "weakening"
    elif delta >= 0.03:
        label = "improving"
    else:
        label = "flat"
    return delta, label


def _autonomy_maturity_trend(progression_rows: List[Dict[str, str]]) -> Tuple[float, str]:
    if len(progression_rows) < 2:
        return 0.0, "insufficient_autonomy_history"
    sorted_rows = _sorted_rows(progression_rows, "timestamp_utc")
    ranks: List[int] = []
    for r in sorted_rows:
        state = _norm_upper(r.get("graduation_state"))
        ranks.append(AUTONOMY_RANK.get(state, 1))
    delta = ranks[-1] - ranks[0]
    if delta < 0:
        label = "regressed"
    elif delta > 0:
        label = "advanced"
    else:
        label = "flat"
    return float(delta), label


def _readiness_trend(
    progression_rows: List[Dict[str, str]],
    snapshot: Dict[str, Any],
) -> Tuple[float, str]:
    if len(progression_rows) >= 2:
        sorted_rows = _sorted_rows(progression_rows, "timestamp_utc")
        vals = [_to_float(r.get("confidence")) for r in sorted_rows]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            delta = vals[-1] - vals[0]
            label = "declining" if delta < -0.05 else ("improving" if delta > 0.05 else "flat")
            return delta, label
    score = snapshot["readiness_score"]
    if score < 0.40:
        return -0.10, "low_readiness"
    if score < 0.55:
        return -0.03, "moderate_readiness"
    return 0.0, "stable_readiness"


def _health_trend(
    posture_rows: List[Dict[str, str]],
    snapshot: Dict[str, Any],
) -> Tuple[float, str]:
    stale_count = sum(
        1 for r in posture_rows if _norm_upper(r.get("system_health_status")) in PROHIBITED_HEALTH
    )
    if snapshot["health_status"] in PROHIBITED_HEALTH:
        return -0.15, f"health_{snapshot['health_status'].lower()}"
    if posture_rows and stale_count / len(posture_rows) >= 0.5:
        return -0.08, "persistent_stale_health"
    if snapshot["health_score"] < 0.45:
        return -0.06, "low_health_score"
    return 0.0, "healthy"


def _build_drift_signals(
    *,
    snapshot: Dict[str, Any],
    posture_rows: List[Dict[str, str]],
    violation_rows: List[Dict[str, str]],
    precedent_rows: List[Dict[str, str]],
    progression_rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    conf_delta, conf_label = _confidence_trend(posture_rows)
    viol = _violation_stats(violation_rows)
    over_freq = _overruling_frequency(precedent_rows)
    shadow_delta, shadow_label = _shadow_trend(progression_rows, snapshot)
    autonomy_delta, autonomy_label = _autonomy_maturity_trend(progression_rows)
    readiness_delta, readiness_label = _readiness_trend(progression_rows, snapshot)
    health_delta, health_label = _health_trend(posture_rows, snapshot)
    shadow_perf = _shadow_performance_score(snapshot)

    governance_confidence = snapshot["council_confidence"]
    if posture_rows:
        last_conf = _to_float(_sorted_rows(posture_rows, "timestamp_utc")[-1].get("confidence"))
        if last_conf is not None:
            governance_confidence = last_conf

    return {
        "governance_confidence_trend": {
            "delta": round(conf_delta, 6),
            "label": conf_label,
        },
        "constitutional_violation_frequency": {
            "total": viol["total"],
            "unresolved": viol["unresolved"],
            "critical": viol["critical"],
            "pressure": viol["constitutional_pressure"],
        },
        "court_overruling_frequency": {
            "rate": over_freq,
            "persistent": over_freq >= 0.5 and len(precedent_rows) >= 2,
        },
        "shadow_performance_trend": {
            "delta": round(shadow_delta, 6),
            "label": shadow_label,
            "current_score": shadow_perf,
        },
        "autonomy_maturity_trend": {
            "delta": round(autonomy_delta, 6),
            "label": autonomy_label,
        },
        "readiness_trend": {
            "delta": round(readiness_delta, 6),
            "label": readiness_label,
            "current_score": snapshot["readiness_score"],
        },
        "health_trend": {
            "delta": round(health_delta, 6),
            "label": health_label,
            "current_score": snapshot["health_score"],
            "status": snapshot["health_status"],
        },
        "negative_ruling_rate": round(_negative_ruling_rate(posture_rows), 6),
        "governance_confidence": round(governance_confidence, 6),
        "shadow_performance": shadow_perf,
        "constitutional_pressure": viol["constitutional_pressure"],
        "overruling_frequency": over_freq,
    }


# -----------------------------------------------------------
# Drift score and classification
# -----------------------------------------------------------
def _compute_drift_score(signals: Dict[str, Any], snapshot: Dict[str, Any]) -> float:
    pressure = _to_float(signals.get("constitutional_pressure")) or 0.0
    over_freq = _to_float(signals.get("overruling_frequency")) or 0.0
    conf_trend = signals["governance_confidence_trend"]
    conf_penalty = _clamp(-(_to_float(conf_trend.get("delta")) or 0.0), 0.0, 1.0)
    if (_to_float(signals.get("governance_confidence")) or 1.0) < 0.25:
        conf_penalty = max(conf_penalty, 0.55)

    shadow_trend = signals["shadow_performance_trend"]
    shadow_penalty = 0.0
    if shadow_trend.get("label") in ("deteriorating", "weakening"):
        shadow_penalty = 0.45 if shadow_trend["label"] == "deteriorating" else 0.25
    shadow_score = _to_float(signals.get("shadow_performance")) or 0.5
    if shadow_score < 0.45:
        shadow_penalty = max(shadow_penalty, 0.35)

    readiness_trend = signals["readiness_trend"]
    readiness_penalty = _clamp(-(_to_float(readiness_trend.get("delta")) or 0.0) * 2.0, 0.0, 0.5)
    if (_to_float(readiness_trend.get("current_score")) or 1.0) < 0.45:
        readiness_penalty = max(readiness_penalty, 0.30)

    health_trend = signals["health_trend"]
    health_penalty = _clamp(-(_to_float(health_trend.get("delta")) or 0.0) * 2.5, 0.0, 0.55)
    if _norm_upper(health_trend.get("status")) in PROHIBITED_HEALTH:
        health_penalty = max(health_penalty, 0.40)

    autonomy_trend = signals["autonomy_maturity_trend"]
    autonomy_penalty = 0.35 if autonomy_trend.get("label") == "regressed" else 0.0

    negative_rate = _to_float(signals.get("negative_ruling_rate")) or 0.0

    raw = (
        pressure * 0.20
        + over_freq * 0.15
        + conf_penalty * 0.20
        + shadow_penalty * 0.15
        + readiness_penalty * 0.15
        + health_penalty * 0.10
        + autonomy_penalty * 0.05
        + negative_rate * 0.10
    )

    if snapshot["constitution_state"] == CONSTITUTION_VIOLATED:
        raw = max(raw, 0.45)
    if snapshot["court_ruling"] == COURT_OVERRULED:
        raw = max(raw, 0.40)
    if snapshot["council_ruling"] in ("GOVERNANCE_REVOKE_AUTONOMY", "GOVERNANCE_LOCKDOWN"):
        raw = max(raw, 0.55)

    return round(_clamp(raw, 0.0, 1.0), 6)


def _classify_drift_state(
    *,
    drift_score: float,
    signals: Dict[str, Any],
    snapshot: Dict[str, Any],
    have_evidence: bool,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not have_evidence:
        reasons.append("no upstream drift evidence; defaulting to UNSTABLE early warning")
        return STATE_UNSTABLE, reasons

    viol = signals["constitutional_violation_frequency"]
    over = signals["court_overruling_frequency"]
    conf_trend = signals["governance_confidence_trend"]
    shadow_trend = signals["shadow_performance_trend"]
    autonomy_trend = signals["autonomy_maturity_trend"]

    # ---- 1. GOVERNANCE_FAILURE_RISK ----
    failure_triggers = (
        drift_score >= 0.75
        or (viol["critical"] >= 2 and viol["unresolved"] >= 2)
        or (over["persistent"] and viol["total"] >= 2)
        or (
            snapshot["constitution_state"] == CONSTITUTION_VIOLATED
            and snapshot["court_ruling"] == COURT_OVERRULED
            and snapshot["health_status"] in PROHIBITED_HEALTH
            and drift_score >= 0.50
        )
        or snapshot["council_ruling"] == "GOVERNANCE_LOCKDOWN"
    )
    if failure_triggers:
        if viol["critical"] >= 2:
            reasons.append(f"repeated constitutional violations (critical={viol['critical']})")
        if over["persistent"]:
            reasons.append("court overruling persistent")
        if drift_score >= 0.75:
            reasons.append(f"severe drift score ({drift_score:.2f})")
        if not reasons:
            reasons.append("governance collapse likely")
        return STATE_FAILURE_RISK, reasons

    # ---- 2. GOVERNANCE_UNSTABLE ----
    unstable_triggers = (
        drift_score >= 0.55
        or snapshot["constitution_state"] == CONSTITUTION_VIOLATED
        or autonomy_trend["label"] == "regressed"
        or (snapshot["council_ruling"] == "GOVERNANCE_REVOKE_AUTONOMY" and viol["unresolved"] >= 1)
        or (viol["unresolved"] >= 2 and over["rate"] >= 0.5)
    )
    if unstable_triggers:
        if snapshot["constitution_state"] == CONSTITUTION_VIOLATED:
            reasons.append("constitutional pressure elevated")
        if autonomy_trend["label"] == "regressed":
            reasons.append("autonomy maturity regressed")
        if over["rate"] >= 0.5:
            reasons.append("governance inconsistency from court overruling")
        if not reasons:
            reasons.append("governance instability detected")
        return STATE_UNSTABLE, reasons

    # ---- 3. GOVERNANCE_DRIFTING ----
    drifting_triggers = (
        drift_score >= 0.35
        or conf_trend["label"] in ("declining", "weakening")
        or shadow_trend["label"] in ("deteriorating", "weakening")
        or over["rate"] >= 0.5
        or signals["negative_ruling_rate"] >= 0.5
    )
    if drifting_triggers:
        if conf_trend["label"] in ("declining", "weakening"):
            reasons.append(
                f"governance confidence {conf_trend['label']} " f"(delta={conf_trend['delta']:.2f})"
            )
        if shadow_trend["label"] in ("deteriorating", "weakening"):
            reasons.append(f"shadow performance {shadow_trend['label']}")
        if over["rate"] >= 0.5:
            reasons.append("repeated constitutional overruling")
        if not reasons:
            reasons.append("worsening governance trend")
        return STATE_DRIFTING, reasons

    # ---- 4. GOVERNANCE_EARLY_WARNING ----
    warning_triggers = (
        drift_score >= 0.20
        or conf_trend["label"] == "weakening"
        or viol["total"] >= 1
        or snapshot["health_status"] in PROHIBITED_HEALTH
        or snapshot["readiness_score"] < 0.55
    )
    if warning_triggers:
        if conf_trend["label"] == "weakening":
            reasons.append("governance confidence weakening")
        if viol["total"] >= 1:
            reasons.append("constitutional violations recorded")
        if snapshot["health_status"] in PROHIBITED_HEALTH:
            reasons.append(f"system health {snapshot['health_status']}")
        if not reasons:
            reasons.append("mild deterioration warrants elevated caution")
        return STATE_EARLY_WARNING, reasons

    # ---- 5. GOVERNANCE_STABLE ----
    reasons.append("healthy posture trend; low constitutional pressure; stable confidence")
    return STATE_STABLE, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _drift_booleans(
    drift_state: str, signals: Dict[str, Any], snapshot: Dict[str, Any]
) -> Dict[str, bool]:
    unstable_plus = drift_state in (STATE_DRIFTING, STATE_UNSTABLE, STATE_FAILURE_RISK)
    warning_plus = drift_state != STATE_STABLE
    return {
        "governance_drift_detected": warning_plus,
        "intervention_required": drift_state
        in (STATE_DRIFTING, STATE_UNSTABLE, STATE_FAILURE_RISK),
        "autonomy_freeze_recommended": unstable_plus
        or signals["autonomy_maturity_trend"]["label"] == "regressed",
        "governance_tightening_recommended": warning_plus,
        "operator_escalation_required": drift_state in (STATE_UNSTABLE, STATE_FAILURE_RISK),
        "constitutional_pressure_elevated": (
            (signals["constitutional_violation_frequency"]["total"] >= 1)
            or snapshot["constitution_state"] == CONSTITUTION_VIOLATED
            or signals["court_overruling_frequency"]["rate"] >= 0.5
        ),
    }


def _recommendations(drift_state: str, booleans: Dict[str, bool]) -> List[str]:
    recs: List[str] = []
    if booleans["autonomy_freeze_recommended"]:
        recs.append("Freeze autonomy promotion")
    if booleans["governance_tightening_recommended"]:
        recs.append("Tighten governance thresholds")
    if drift_state in (STATE_DRIFTING, STATE_UNSTABLE, STATE_FAILURE_RISK):
        recs.append("Increase target cash posture")
        recs.append("Extend apprenticeship period")
    if booleans["operator_escalation_required"]:
        recs.append("Escalate operator supervision")
    if drift_state == STATE_STABLE:
        recs.append("Continue monitoring")
    elif drift_state == STATE_EARLY_WARNING:
        recs.append("Continue monitoring with elevated caution")
    if not recs:
        recs.append("Continue monitoring")
    # dedupe preserving order
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(drift_state: str, reasons: List[str], signals: Dict[str, Any]) -> str:
    if drift_state == STATE_STABLE:
        return (
            "Governance remains stable with healthy posture trend, "
            "low constitutional pressure, and stable confidence."
        )
    parts = reasons[:3] if reasons else ["governance drift signals elevated"]
    over = signals["court_overruling_frequency"]["rate"]
    conf = signals["governance_confidence_trend"]
    if over >= 0.5 and conf["label"] in ("declining", "weakening"):
        return (
            "Governance drift detected because constitutional overruling frequency "
            "increased while governance confidence declined."
        )
    return f"Governance drift detected because {'; '.join(parts)}."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    drift_state: str,
    drift_score: float,
    signals: Dict[str, Any],
    snapshot: Dict[str, Any],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
) -> str:
    lines = [
        "# Triton Governance Drift Detection",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Drift State",
        "",
        f"**{drift_state}**",
        "",
        f"| field | value |",
        f"|---|---|",
        f"| drift_score | {drift_score:.3f} |",
        f"| governance_drift_detected | {booleans['governance_drift_detected']} |",
        f"| intervention_required | {booleans['intervention_required']} |",
        f"| autonomy_freeze_recommended | {booleans['autonomy_freeze_recommended']} |",
        f"| operator_escalation_required | {booleans['operator_escalation_required']} |",
        "",
        "## Drift Signals",
        "",
        "| signal | value |",
        "|---|---|",
        f"| governance_confidence_trend | {signals['governance_confidence_trend']['label']} "
        f"(delta={signals['governance_confidence_trend']['delta']:.3f}) |",
        f"| constitutional_violations | total={signals['constitutional_violation_frequency']['total']} "
        f"unresolved={signals['constitutional_violation_frequency']['unresolved']} |",
        f"| court_overruling_frequency | {signals['court_overruling_frequency']['rate']:.3f} |",
        f"| shadow_performance_trend | {signals['shadow_performance_trend']['label']} "
        f"(score={signals['shadow_performance']:.3f}) |",
        f"| autonomy_maturity_trend | {signals['autonomy_maturity_trend']['label']} |",
        f"| readiness_trend | {signals['readiness_trend']['label']} "
        f"(score={signals['readiness_trend']['current_score']:.3f}) |",
        f"| health_trend | {signals['health_trend']['label']} "
        f"(status={signals['health_trend']['status']}) |",
        "",
        "## Governance Pressure",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| constitutional_pressure | {signals['constitutional_pressure']:.3f} |",
        f"| overruling_frequency | {signals['overruling_frequency']:.3f} |",
        f"| governance_confidence | {signals['governance_confidence']:.3f} |",
        f"| shadow_performance | {signals['shadow_performance']:.3f} |",
        f"| constitution_state | {snapshot['constitution_state']} |",
        f"| court_ruling | {snapshot['court_ruling']} |",
        f"| council_ruling | {snapshot['council_ruling']} |",
        f"| regime | {snapshot['regime']} |",
        "",
        "## Recommendations",
        "",
    ]
    for rec in recommendations:
        lines.append(f"- {rec}")
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
            f"Drift score {drift_score:.2f} reflects constitutional pressure, "
            f"overruling frequency, confidence deterioration, shadow performance, "
            f"readiness decline, and health signals.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Drift memory
# -----------------------------------------------------------
def _build_drift_row(
    *,
    timestamp: str,
    drift_state: str,
    drift_score: float,
    signals: Dict[str, Any],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "drift_state": drift_state,
        "drift_score": round(drift_score, 6),
        "constitutional_pressure": signals["constitutional_pressure"],
        "overruling_frequency": signals["overruling_frequency"],
        "governance_confidence": signals["governance_confidence"],
        "shadow_performance": signals["shadow_performance"],
        "rationale": rationale,
    }


def _merge_drift_memory(
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
        for c in DRIFT_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_drift_detection(
    *,
    posture_rows: List[Dict[str, str]],
    council_summary: Dict[str, Any],
    precedent_rows: List[Dict[str, str]],
    violation_rows: List[Dict[str, str]],
    shadow_perf: Dict[str, Any],
    progression_rows: List[Dict[str, str]],
    scorecard: Dict[str, Any],
    health: Dict[str, Any],
    readiness: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    existing_drift_memory: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    snapshot = _extract_snapshot(
        council_summary=council_summary,
        shadow_perf=shadow_perf,
        scorecard=scorecard,
        health=health,
        readiness=readiness,
        runtime_policy=runtime_policy,
    )

    have_evidence = any(
        bool(x)
        for x in (
            council_summary,
            scorecard,
            health,
            readiness,
            posture_rows,
            violation_rows,
            precedent_rows,
        )
    )

    signals = _build_drift_signals(
        snapshot=snapshot,
        posture_rows=posture_rows,
        violation_rows=violation_rows,
        precedent_rows=precedent_rows,
        progression_rows=progression_rows,
    )

    drift_score = _compute_drift_score(signals, snapshot)
    drift_state, reasons = _classify_drift_state(
        drift_score=drift_score,
        signals=signals,
        snapshot=snapshot,
        have_evidence=have_evidence,
    )

    booleans = _drift_booleans(drift_state, signals, snapshot)
    recommendations = _recommendations(drift_state, booleans)
    rationale = _build_rationale(drift_state, reasons, signals)

    drift_row = _build_drift_row(
        timestamp=timestamp,
        drift_state=drift_state,
        drift_score=drift_score,
        signals=signals,
        rationale=rationale,
    )
    merged_memory = _merge_drift_memory(existing_drift_memory, drift_row)

    md = _render_markdown(
        generated_at=timestamp,
        drift_state=drift_state,
        drift_score=drift_score,
        signals=signals,
        snapshot=snapshot,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_drift_detection_engine",
        "engine_version": 1,
        "drift_state": drift_state,
        "drift_score": drift_score,
        "drift_reasons": reasons,
        "drift_signals": signals,
        "snapshot": snapshot,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "constitutional_supremacy_note": (
            "Drift detection is an early-warning layer only. "
            "It cannot override constitutional law or execution prohibitions."
        ),
        "drift_memory_size_after_append": len(merged_memory),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "drift_detection_only": True,
        },
        "inputs_seen": {
            "arm_governance_posture_memory": len(posture_rows),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "arm_constitutional_precedent_memory": len(precedent_rows),
            "arm_constitution_violation_memory": len(violation_rows),
            "arm_shadow_performance_summary": bool(shadow_perf),
            "arm_autonomy_progression_memory": len(progression_rows),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health),
            "autonomous_readiness_summary": bool(readiness),
            "runtime_policy_governed": bool(runtime_policy),
            "existing_drift_memory_rows": len(existing_drift_memory),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_drift_detection_engine",
        "drift_state": drift_state,
        "drift_score": drift_score,
        "governance_drift_detected": booleans["governance_drift_detected"],
        "intervention_required": booleans["intervention_required"],
        "autonomy_freeze_recommended": booleans["autonomy_freeze_recommended"],
        "governance_tightening_recommended": booleans["governance_tightening_recommended"],
        "operator_escalation_required": booleans["operator_escalation_required"],
        "constitutional_pressure_elevated": booleans["constitutional_pressure_elevated"],
        "governance_confidence": signals["governance_confidence"],
        "constitutional_pressure": signals["constitutional_pressure"],
        "overruling_frequency": signals["overruling_frequency"],
        "shadow_performance": signals["shadow_performance"],
        "n_recommendations": len(recommendations),
        "drift_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance drift detection engine (Step 35). "
            "Detects slow governance deterioration before failure. "
            "No broker calls."
        ),
    )
    p.add_argument("--posture-mem", default=str(DEFAULT_POSTURE_MEM))
    p.add_argument("--council-summary", default=str(DEFAULT_COUNCIL_SUMMARY))
    p.add_argument("--precedent-mem", default=str(DEFAULT_PRECEDENT_MEM))
    p.add_argument("--violation-mem", default=str(DEFAULT_VIOLATION_MEM))
    p.add_argument("--shadow-perf", default=str(DEFAULT_SHADOW_PERF))
    p.add_argument("--progression-mem", default=str(DEFAULT_PROGRESSION_MEM))
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
        "[ARM_DRIFT] starting (read-only governance drift detection; no broker calls)",
        flush=True,
    )

    posture_rows = _safe_read_csv_rows(
        Path(args.posture_mem), label="arm_governance_posture_memory.csv"
    )
    council_summary = _safe_read_json(
        Path(args.council_summary), label="arm_supreme_governance_council_summary.json"
    )
    precedent_rows = _safe_read_csv_rows(
        Path(args.precedent_mem), label="arm_constitutional_precedent_memory.csv"
    )
    violation_rows = _safe_read_csv_rows(
        Path(args.violation_mem), label="arm_constitution_violation_memory.csv"
    )
    shadow_perf = _safe_read_json(
        Path(args.shadow_perf), label="arm_shadow_performance_summary.json"
    )
    progression_rows = _safe_read_csv_rows(
        Path(args.progression_mem), label="arm_autonomy_progression_memory.csv"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    health = _safe_read_json(Path(args.health), label="autonomous_system_health_summary.json")
    readiness = _safe_read_json(Path(args.readiness), label="autonomous_readiness_summary.json")
    runtime_policy = _safe_read_json(Path(args.gov_policy), label="runtime_policy_governed.json")
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_drift_memory.csv"
    )

    record, summary, md, merged_memory = build_drift_detection(
        posture_rows=posture_rows,
        council_summary=council_summary,
        precedent_rows=precedent_rows,
        violation_rows=violation_rows,
        shadow_perf=shadow_perf,
        progression_rows=progression_rows,
        scorecard=scorecard,
        health=health,
        readiness=readiness,
        runtime_policy=runtime_policy,
        existing_drift_memory=existing_mem,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=DRIFT_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    booleans = record["governance_booleans"]
    print(
        "[ARM_DRIFT] "
        f"state={record['drift_state']} "
        f"score={record['drift_score']:.3f} "
        f"intervention={booleans['intervention_required']} "
        f"freeze={booleans['autonomy_freeze_recommended']} "
        f"confidence={summary['governance_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_DRIFT_SAFETY] broker_calls=0 orders_placed=0 portfolio_mutated=False",
        flush=True,
    )
    print(
        f"[ARM_DRIFT_OUT] json={Path(args.out_json).as_posix()} "
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
            raise RuntimeError(f"[ARM_DRIFT_SAFETY] forbidden broker token detected: {tok!r}")


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
