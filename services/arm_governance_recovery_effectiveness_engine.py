"""
ARM Governance Recovery Effectiveness Engine -- Step 37.

Reads:
    data/results/arm_governance_recovery_summary.json              (Step 36)
    data/results/arm_governance_recovery.json                      (Step 36)
    data/results/arm_governance_recovery_memory.csv                (Step 36)
    data/results/arm_governance_drift_detection_summary.json       (Step 35)
    data/results/arm_governance_drift_memory.csv                   (Step 35)
    data/results/arm_constitution_violation_memory.csv             (Step 32)
    data/results/arm_constitutional_precedent_memory.csv           (Step 33)
    data/results/arm_shadow_performance_summary.json               (Step 30)
    data/results/arm_autonomy_progression_memory.csv               (Step 31)
    data/results/autonomous_governance_scorecard.json              (Step 19)
    data/results/autonomous_system_health_summary.json             (Step 20)
    data/results/autonomous_readiness_summary.json                 (Step 21)

Writes:
    data/results/arm_governance_recovery_effectiveness.json
    data/results/arm_governance_recovery_effectiveness.md
    data/results/arm_governance_recovery_effectiveness_summary.json
    data/results/arm_governance_recovery_effectiveness_memory.csv
    data/results/arm_governance_recovery_effectiveness_memory.parquet

Purpose
-------
This engine answers:

    "Is governance recovery working?"

It is Triton's closed-loop healing auditor -- read-only evaluation of whether
recovery actions are restoring institutional stability. Effectiveness
analysis cannot override constitutional law, the constitutional court,
capital preservation, or operator supremacy.

Effectiveness state cascade
---------------------------
    1. RECOVERY_REGRESSING   deterioration worsening
    2. RECOVERY_INEFFECTIVE  recovery failing, drift unchanged/worsening
    3. RECOVERY_STALLED      little measurable change
    4. RECOVERY_IMPROVING    modest measurable improvement
    5. RECOVERY_EFFECTIVE    confidence up, pressure down, goals improving

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens are never written literally; an import-time
self-check raises if they ever appear.

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only effectiveness memory keyed by timestamp.
* Missing inputs warn-and-continue; absent evidence defaults to
  RECOVERY_STALLED as the safe auditor posture.
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

DEFAULT_RECOVERY_SUMMARY = RESULTS_DIR / "arm_governance_recovery_summary.json"
DEFAULT_RECOVERY_RECORD = RESULTS_DIR / "arm_governance_recovery.json"
DEFAULT_RECOVERY_MEM = RESULTS_DIR / "arm_governance_recovery_memory.csv"
DEFAULT_DRIFT_SUMMARY = RESULTS_DIR / "arm_governance_drift_detection_summary.json"
DEFAULT_DRIFT_MEM = RESULTS_DIR / "arm_governance_drift_memory.csv"
DEFAULT_VIOLATION_MEM = RESULTS_DIR / "arm_constitution_violation_memory.csv"
DEFAULT_PRECEDENT_MEM = RESULTS_DIR / "arm_constitutional_precedent_memory.csv"
DEFAULT_SHADOW_PERF = RESULTS_DIR / "arm_shadow_performance_summary.json"
DEFAULT_PROGRESSION_MEM = RESULTS_DIR / "arm_autonomy_progression_memory.csv"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS = RESULTS_DIR / "autonomous_readiness_summary.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_recovery_effectiveness.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_recovery_effectiveness.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_recovery_effectiveness_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_recovery_effectiveness_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_recovery_effectiveness_memory.parquet"


# -----------------------------------------------------------
# Effectiveness state constants
# -----------------------------------------------------------
STATE_EFFECTIVE = "RECOVERY_EFFECTIVE"
STATE_IMPROVING = "RECOVERY_IMPROVING"
STATE_STALLED = "RECOVERY_STALLED"
STATE_INEFFECTIVE = "RECOVERY_INEFFECTIVE"
STATE_REGRESSING = "RECOVERY_REGRESSING"

RECOVERY_NOT_REQUIRED = "RECOVERY_NOT_REQUIRED"

AUTONOMY_RANK: Dict[str, int] = {
    "MANUAL_LOCKED": 0,
    "MANUAL": 0,
    "ASSISTED_CANDIDATE": 1,
    "ASSISTED_APPROVED": 2,
    "AUTO_ALLOWED_CANDIDATE": 3,
    "AUTO_ALLOWED_APPROVED": 4,
}

EFFECTIVENESS_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "effectiveness_state",
    "effectiveness_score",
    "drift_delta",
    "governance_confidence_delta",
    "constitutional_pressure_delta",
    "overruling_delta",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_RECOVERY_EFFECTIVENESS_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(EFFECTIVENESS_MEMORY_COLUMNS))
        for col in (
            "effectiveness_score",
            "drift_delta",
            "governance_confidence_delta",
            "constitutional_pressure_delta",
            "overruling_delta",
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
# Baseline and current snapshot
# -----------------------------------------------------------
def _overruling_rate(precedent_rows: List[Dict[str, str]]) -> float:
    if not precedent_rows:
        return 0.0
    n_over = sum(1 for r in precedent_rows if _norm_upper(r.get("ruling")) == "COURT_OVERRULED")
    return round(n_over / len(precedent_rows), 6)


def _autonomy_rank_from_progression(progression_rows: List[Dict[str, str]]) -> float:
    if not progression_rows:
        return 1.0
    state = _norm_upper(_sorted_rows(progression_rows, "timestamp_utc")[-1].get("graduation_state"))
    return float(AUTONOMY_RANK.get(state, 1))


def _extract_current(
    *,
    recovery_summary: Dict[str, Any],
    recovery_record: Dict[str, Any],
    drift_summary: Dict[str, Any],
    shadow_perf: Dict[str, Any],
    scorecard: Dict[str, Any],
    health: Dict[str, Any],
    readiness: Dict[str, Any],
    precedent_rows: List[Dict[str, str]],
    progression_rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    evidence = recovery_record.get("evidence") or {}
    scores = scorecard.get("scores") or {}

    shadow_discipline = _to_float(shadow_perf.get("shadow_discipline_score"))
    if shadow_discipline is None:
        shadow_discipline = _to_float(shadow_perf.get("shadow_autonomy_readiness_score")) or 0.5

    return {
        "recovery_state": _norm_upper(
            recovery_summary.get("recovery_state") or recovery_record.get("recovery_state")
        ),
        "recovery_required": bool(recovery_summary.get("recovery_required")),
        "drift_state": _norm_upper(
            drift_summary.get("drift_state")
            or recovery_summary.get("drift_state")
            or evidence.get("drift_state")
        ),
        "drift_score": _clamp(
            _to_float(drift_summary.get("drift_score"))
            or _to_float(recovery_summary.get("drift_score"))
            or _to_float(evidence.get("drift_score"))
            or 0.0,
            0.0,
            1.0,
        ),
        "governance_confidence": _clamp(
            _to_float(recovery_summary.get("governance_confidence"))
            or _to_float(evidence.get("governance_confidence"))
            or 0.0,
            0.0,
            1.0,
        ),
        "constitutional_pressure": _clamp(
            _to_float(recovery_summary.get("constitutional_pressure"))
            or _to_float(evidence.get("constitutional_pressure"))
            or 0.0,
            0.0,
            1.0,
        ),
        "overruling_frequency": _clamp(
            _to_float(drift_summary.get("overruling_frequency"))
            or _overruling_rate(precedent_rows),
            0.0,
            1.0,
        ),
        "shadow_discipline": _clamp(shadow_discipline, 0.0, 1.0),
        "autonomy_maturity_rank": _autonomy_rank_from_progression(progression_rows),
        "readiness_score": _clamp(
            _to_float(readiness.get("readiness_score")) or 0.5,
            0.0,
            1.0,
        ),
        "health_score": _clamp(
            _to_float(health.get("system_health_score")) or 0.5,
            0.0,
            1.0,
        ),
        "governance_quality": _clamp(
            _to_float(scores.get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "shadow_performance": _clamp(
            _to_float(drift_summary.get("shadow_performance")) or shadow_discipline,
            0.0,
            1.0,
        ),
    }


def _baseline_from_memory(
    *,
    recovery_mem: List[Dict[str, str]],
    drift_mem: List[Dict[str, str]],
    progression_mem: List[Dict[str, str]],
    precedent_mem: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Earliest memory row as recovery baseline."""
    baseline: Dict[str, Any] = {}

    if recovery_mem:
        first = _sorted_rows(recovery_mem, "timestamp")[0]
        baseline["governance_confidence"] = _to_float(first.get("governance_confidence")) or 0.0
        baseline["constitutional_pressure"] = _to_float(first.get("constitutional_pressure")) or 0.0
        baseline["drift_score"] = None  # filled from drift mem if available

    if drift_mem:
        first_d = _sorted_rows(drift_mem, "timestamp")[0]
        baseline["drift_score"] = _to_float(first_d.get("drift_score")) or 0.0
        baseline.setdefault(
            "governance_confidence", _to_float(first_d.get("governance_confidence")) or 0.0
        )
        baseline.setdefault(
            "constitutional_pressure", _to_float(first_d.get("constitutional_pressure")) or 0.0
        )
        baseline["overruling_frequency"] = _to_float(first_d.get("overruling_frequency")) or 0.0
        baseline["shadow_performance"] = _to_float(first_d.get("shadow_performance")) or 0.5

    if len(recovery_mem) >= 2:
        first_r = _sorted_rows(recovery_mem, "timestamp")[0]
        last_r = _sorted_rows(recovery_mem, "timestamp")[-1]
        baseline["governance_confidence"] = _to_float(first_r.get("governance_confidence")) or 0.0
        baseline["constitutional_pressure"] = (
            _to_float(first_r.get("constitutional_pressure")) or 0.0
        )

    if len(drift_mem) >= 2:
        first_d = _sorted_rows(drift_mem, "timestamp")[0]
        baseline["drift_score"] = _to_float(first_d.get("drift_score")) or 0.0

    if progression_mem:
        first_p = _sorted_rows(progression_mem, "timestamp_utc")[0]
        state = _norm_upper(first_p.get("graduation_state"))
        baseline["autonomy_maturity_rank"] = float(AUTONOMY_RANK.get(state, 1))
        baseline["readiness_proxy"] = _to_float(first_p.get("confidence")) or 0.5
    else:
        baseline["autonomy_maturity_rank"] = 1.0
        baseline["readiness_proxy"] = 0.5

    baseline["overruling_frequency"] = baseline.get("overruling_frequency") or _overruling_rate(
        precedent_mem
    )
    baseline.setdefault("drift_score", 0.0)
    baseline.setdefault("governance_confidence", 0.0)
    baseline.setdefault("constitutional_pressure", 0.0)
    baseline.setdefault("shadow_performance", 0.5)
    baseline.setdefault("health_score", 0.5)
    baseline.setdefault("readiness_score", baseline.get("readiness_proxy", 0.5))

    return baseline


def _compute_metrics(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """Deltas: positive = improvement for pressure/drift/overruling; signed for confidence."""
    conf_delta = current["governance_confidence"] - (baseline.get("governance_confidence") or 0.0)
    pressure_delta = (baseline.get("constitutional_pressure") or 0.0) - current[
        "constitutional_pressure"
    ]
    over_delta = (baseline.get("overruling_frequency") or 0.0) - current["overruling_frequency"]
    drift_delta = (baseline.get("drift_score") or 0.0) - current["drift_score"]
    shadow_delta = current["shadow_discipline"] - (baseline.get("shadow_performance") or 0.5)
    autonomy_delta = current["autonomy_maturity_rank"] - (
        baseline.get("autonomy_maturity_rank") or 1.0
    )
    readiness_delta = current["readiness_score"] - (baseline.get("readiness_score") or 0.5)
    health_delta = current["health_score"] - (baseline.get("health_score") or 0.5)

    return {
        "governance_confidence_delta": round(conf_delta, 6),
        "constitutional_pressure_delta": round(pressure_delta, 6),
        "overruling_frequency_delta": round(over_delta, 6),
        "drift_score_delta": round(drift_delta, 6),
        "shadow_discipline_delta": round(shadow_delta, 6),
        "autonomy_maturity_delta": round(autonomy_delta, 6),
        "readiness_delta": round(readiness_delta, 6),
        "health_delta": round(health_delta, 6),
    }


# -----------------------------------------------------------
# Effectiveness score and classification
# -----------------------------------------------------------
def _effectiveness_score(metrics: Dict[str, Any]) -> float:
    """Higher = recovery more effective."""
    improvement = (
        _clamp(metrics["governance_confidence_delta"], -1.0, 1.0) * 0.25
        + _clamp(metrics["constitutional_pressure_delta"], -1.0, 1.0) * 0.20
        + _clamp(metrics["drift_score_delta"], -1.0, 1.0) * 0.20
        + _clamp(metrics["overruling_frequency_delta"], -1.0, 1.0) * 0.10
        + _clamp(metrics["readiness_delta"], -1.0, 1.0) * 0.10
        + _clamp(metrics["health_delta"], -1.0, 1.0) * 0.08
        + _clamp(metrics["shadow_discipline_delta"], -1.0, 1.0) * 0.07
    )
    # Map roughly [-0.5, 0.5] improvement range to [0, 1]
    raw = 0.5 + improvement
    return round(_clamp(raw, 0.0, 1.0), 6)


def _net_improvement(metrics: Dict[str, Any]) -> float:
    return (
        metrics["governance_confidence_delta"]
        + metrics["constitutional_pressure_delta"]
        + metrics["drift_score_delta"]
        + metrics["overruling_frequency_delta"] * 0.5
    )


def _classify_effectiveness(
    *,
    metrics: Dict[str, Any],
    current: Dict[str, Any],
    effectiveness_score: float,
    have_baseline: bool,
    have_evidence: bool,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if not have_evidence:
        reasons.append("no upstream effectiveness evidence; defaulting to STALLED")
        return STATE_STALLED, reasons

    if current["recovery_state"] == RECOVERY_NOT_REQUIRED and current["drift_score"] < 0.20:
        reasons.append("recovery not required; governance stable")
        return STATE_EFFECTIVE, reasons

    regressing = (
        metrics["governance_confidence_delta"] < -0.05
        or metrics["constitutional_pressure_delta"] < -0.05
        or metrics["drift_score_delta"] < -0.05
        or metrics["overruling_frequency_delta"] < -0.10
    )
    if regressing:
        if metrics["constitutional_pressure_delta"] < -0.05:
            reasons.append("constitutional pressure rising")
        if metrics["governance_confidence_delta"] < -0.05:
            reasons.append("governance confidence declining")
        if metrics["drift_score_delta"] < -0.05:
            reasons.append("drift score worsening")
        if not reasons:
            reasons.append("governance deterioration worsening")
        return STATE_REGRESSING, reasons

    effective = (
        metrics["governance_confidence_delta"] >= 0.08
        and metrics["constitutional_pressure_delta"] >= 0.05
        and metrics["drift_score_delta"] >= 0.03
        and effectiveness_score >= 0.65
    )
    if effective:
        reasons.append("governance confidence increased")
        reasons.append("constitutional pressure declined")
        reasons.append("drift score improved")
        return STATE_EFFECTIVE, reasons

    improving = (
        _net_improvement(metrics) >= 0.08
        or effectiveness_score >= 0.58
        or (metrics["governance_confidence_delta"] >= 0.03 and metrics["drift_score_delta"] >= 0.0)
    )
    if improving and not regressing:
        reasons.append("modest measurable improvement detected")
        if metrics["readiness_delta"] >= 0.03:
            reasons.append("readiness stabilizing")
        return STATE_IMPROVING, reasons

    ineffective = (
        current["recovery_required"]
        and _net_improvement(metrics) < -0.03
        and effectiveness_score < 0.45
    )
    if ineffective:
        reasons.append("recovery actions failing to reduce drift")
        if metrics["drift_score_delta"] <= 0:
            reasons.append("drift unchanged or worsening")
        return STATE_INEFFECTIVE, reasons

    stalled = not have_baseline or (
        abs(metrics["governance_confidence_delta"]) < 0.03
        and abs(metrics["constitutional_pressure_delta"]) < 0.03
        and abs(metrics["drift_score_delta"]) < 0.03
    )
    if stalled:
        if not have_baseline:
            reasons.append("insufficient recovery history for trend comparison")
        else:
            reasons.append("little measurable change since recovery began")
        return STATE_STALLED, reasons

    reasons.append("recovery progress indeterminate; monitoring continues")
    return STATE_STALLED, reasons


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _effectiveness_booleans(
    state: str,
    current: Dict[str, Any],
) -> Dict[str, bool]:
    working = state in (STATE_EFFECTIVE, STATE_IMPROVING)
    bad = state in (STATE_INEFFECTIVE, STATE_REGRESSING)
    return {
        "recovery_working": working,
        "recovery_adjustment_required": state
        in (STATE_STALLED, STATE_INEFFECTIVE, STATE_REGRESSING),
        "governance_tightening_required": bad,
        "autonomy_freeze_should_continue": current["recovery_required"] and not working,
        "operator_escalation_required": state == STATE_REGRESSING
        or (state == STATE_INEFFECTIVE and current["constitutional_pressure"] >= 0.55),
        "recovery_restart_required": state == STATE_REGRESSING,
    }


def _recommendations(state: str, booleans: Dict[str, bool]) -> List[str]:
    recs: List[str] = []
    if state in (STATE_EFFECTIVE, STATE_IMPROVING):
        recs.append("Continue recovery program")
    if state == STATE_IMPROVING and booleans["recovery_working"]:
        recs.append("Reduce recovery restrictions gradually")
    if booleans["governance_tightening_required"]:
        recs.append("Intensify governance tightening")
    if booleans["autonomy_freeze_should_continue"]:
        recs.append("Extend manual-only operation")
    if booleans["recovery_adjustment_required"]:
        recs.append("Increase governance monitoring")
        recs.append("Reassess recovery strategy")
    if state == STATE_STALLED:
        recs.append("Continue recovery monitoring")
    if state == STATE_EFFECTIVE:
        recs.append("Reassess after governance stabilization")
    if not recs:
        recs.append("Continue recovery monitoring")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(state: str, metrics: Dict[str, Any], reasons: List[str]) -> str:
    if state == STATE_EFFECTIVE:
        return (
            "Recovery effective because governance confidence increased, "
            "constitutional pressure declined, and drift score improved."
        )
    if state == STATE_IMPROVING:
        return (
            (
                "Recovery improving because governance confidence increased, "
                "constitutional pressure declined, and drift score improved."
            )
            if (
                metrics["governance_confidence_delta"] > 0
                and metrics["constitutional_pressure_delta"] > 0
            )
            else ("Recovery improving because modest measurable stabilization is occurring.")
        )
    if state == STATE_REGRESSING:
        return (
            "Recovery regressing because governance deterioration is worsening, "
            "constitutional pressure is rising, or overruling frequency increased."
        )
    parts = reasons[:2] if reasons else ["recovery progress under evaluation"]
    return f"Recovery assessed as {state.lower().replace('_', ' ')} because {'; '.join(parts)}."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    score: float,
    metrics: Dict[str, Any],
    current: Dict[str, Any],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
) -> str:
    lines = [
        "# Triton Governance Recovery Effectiveness",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Recovery Effectiveness State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| effectiveness_score | {score:.3f} |",
        f"| recovery_working | {booleans['recovery_working']} |",
        f"| recovery_adjustment_required | {booleans['recovery_adjustment_required']} |",
        f"| recovery_state | {current['recovery_state']} |",
        f"| drift_state | {current['drift_state']} |",
        f"| drift_score | {current['drift_score']:.3f} |",
        "",
        "## Recovery Metrics",
        "",
        "| metric | delta | direction |",
        "|---|---|---|",
    ]
    metric_labels = (
        ("governance_confidence_delta", "governance confidence"),
        ("constitutional_pressure_delta", "constitutional pressure"),
        ("overruling_frequency_delta", "overruling frequency"),
        ("drift_score_delta", "drift score"),
        ("shadow_discipline_delta", "shadow discipline"),
        ("autonomy_maturity_delta", "autonomy maturity"),
        ("readiness_delta", "readiness"),
        ("health_delta", "health"),
    )
    for key, label in metric_labels:
        delta = metrics[key]
        if delta > 0.01:
            direction = "improving"
        elif delta < -0.01:
            direction = "worsening"
        else:
            direction = "flat"
        lines.append(f"| {label} | {delta:+.3f} | {direction} |")

    lines.extend(
        [
            "",
            "## Governance Healing",
            "",
            "| signal | value |",
            "|---|---|",
            f"| governance_confidence | {current['governance_confidence']:.3f} |",
            f"| constitutional_pressure | {current['constitutional_pressure']:.3f} |",
            f"| overruling_frequency | {current['overruling_frequency']:.3f} |",
            f"| shadow_discipline | {current['shadow_discipline']:.3f} |",
            f"| readiness_score | {current['readiness_score']:.3f} |",
            f"| health_score | {current['health_score']:.3f} |",
            f"| autonomy_freeze_should_continue | {booleans['autonomy_freeze_should_continue']} |",
            f"| operator_escalation_required | {booleans['operator_escalation_required']} |",
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
            f"Effectiveness score {score:.2f} synthesizes confidence improvement, drift "
            f"reduction, constitutional pressure reduction, readiness, health, and shadow "
            f"discipline trends. This auditor is read-only and cannot override constitutional "
            f"law or operator supremacy.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Effectiveness memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    score: float,
    metrics: Dict[str, Any],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "effectiveness_state": state,
        "effectiveness_score": round(score, 6),
        "drift_delta": metrics["drift_score_delta"],
        "governance_confidence_delta": metrics["governance_confidence_delta"],
        "constitutional_pressure_delta": metrics["constitutional_pressure_delta"],
        "overruling_delta": metrics["overruling_frequency_delta"],
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
        for c in EFFECTIVENESS_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_effectiveness_audit(
    *,
    recovery_summary: Dict[str, Any],
    recovery_record: Dict[str, Any],
    recovery_mem: List[Dict[str, str]],
    drift_summary: Dict[str, Any],
    drift_mem: List[Dict[str, str]],
    violation_mem: List[Dict[str, str]],
    precedent_mem: List[Dict[str, str]],
    shadow_perf: Dict[str, Any],
    progression_mem: List[Dict[str, str]],
    scorecard: Dict[str, Any],
    health: Dict[str, Any],
    readiness: Dict[str, Any],
    existing_effectiveness_mem: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    current = _extract_current(
        recovery_summary=recovery_summary,
        recovery_record=recovery_record,
        drift_summary=drift_summary,
        shadow_perf=shadow_perf,
        scorecard=scorecard,
        health=health,
        readiness=readiness,
        precedent_rows=precedent_mem,
        progression_rows=progression_mem,
    )

    baseline = _baseline_from_memory(
        recovery_mem=recovery_mem,
        drift_mem=drift_mem,
        progression_mem=progression_mem,
        precedent_mem=precedent_mem,
    )

    have_baseline = bool(recovery_mem or drift_mem)
    have_evidence = any(
        bool(x)
        for x in (
            recovery_summary,
            recovery_record,
            drift_summary,
        )
    )

    metrics = _compute_metrics(current, baseline)
    eff_score = _effectiveness_score(metrics)
    state, reasons = _classify_effectiveness(
        metrics=metrics,
        current=current,
        effectiveness_score=eff_score,
        have_baseline=have_baseline,
        have_evidence=have_evidence,
    )

    booleans = _effectiveness_booleans(state, current)
    recommendations = _recommendations(state, booleans)
    rationale = _build_rationale(state, metrics, reasons)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        score=eff_score,
        metrics=metrics,
        rationale=rationale,
    )
    merged_memory = _merge_memory(existing_effectiveness_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        score=eff_score,
        metrics=metrics,
        current=current,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_recovery_effectiveness_engine",
        "engine_version": 1,
        "effectiveness_state": state,
        "effectiveness_score": eff_score,
        "effectiveness_reasons": reasons,
        "recovery_metrics": metrics,
        "current_snapshot": current,
        "baseline_snapshot": baseline,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "constitutional_supremacy_note": (
            "Recovery effectiveness auditing is read-only. "
            "It cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "effectiveness_memory_size_after_append": len(merged_memory),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "effectiveness_audit_only": True,
        },
        "inputs_seen": {
            "arm_governance_recovery_summary": bool(recovery_summary),
            "arm_governance_recovery_record": bool(recovery_record),
            "arm_governance_recovery_memory_rows": len(recovery_mem),
            "arm_governance_drift_detection_summary": bool(drift_summary),
            "arm_governance_drift_memory_rows": len(drift_mem),
            "arm_constitution_violation_memory_rows": len(violation_mem),
            "arm_constitutional_precedent_memory_rows": len(precedent_mem),
            "arm_shadow_performance_summary": bool(shadow_perf),
            "arm_autonomy_progression_memory_rows": len(progression_mem),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health),
            "autonomous_readiness_summary": bool(readiness),
            "existing_effectiveness_memory_rows": len(existing_effectiveness_mem),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_recovery_effectiveness_engine",
        "effectiveness_state": state,
        "effectiveness_score": eff_score,
        "recovery_working": booleans["recovery_working"],
        "recovery_adjustment_required": booleans["recovery_adjustment_required"],
        "governance_tightening_required": booleans["governance_tightening_required"],
        "autonomy_freeze_should_continue": booleans["autonomy_freeze_should_continue"],
        "operator_escalation_required": booleans["operator_escalation_required"],
        "recovery_restart_required": booleans["recovery_restart_required"],
        "governance_confidence_delta": metrics["governance_confidence_delta"],
        "constitutional_pressure_delta": metrics["constitutional_pressure_delta"],
        "drift_score_delta": metrics["drift_score_delta"],
        "recovery_state": current["recovery_state"],
        "drift_state": current["drift_state"],
        "n_recommendations": len(recommendations),
        "effectiveness_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance recovery effectiveness engine (Step 37). "
            "Audits whether recovery is restoring stability. No broker calls."
        ),
    )
    p.add_argument("--recovery-summary", default=str(DEFAULT_RECOVERY_SUMMARY))
    p.add_argument("--recovery-record", default=str(DEFAULT_RECOVERY_RECORD))
    p.add_argument("--recovery-mem", default=str(DEFAULT_RECOVERY_MEM))
    p.add_argument("--drift-summary", default=str(DEFAULT_DRIFT_SUMMARY))
    p.add_argument("--drift-mem", default=str(DEFAULT_DRIFT_MEM))
    p.add_argument("--violation-mem", default=str(DEFAULT_VIOLATION_MEM))
    p.add_argument("--precedent-mem", default=str(DEFAULT_PRECEDENT_MEM))
    p.add_argument("--shadow-perf", default=str(DEFAULT_SHADOW_PERF))
    p.add_argument("--progression-mem", default=str(DEFAULT_PROGRESSION_MEM))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--health", default=str(DEFAULT_HEALTH))
    p.add_argument("--readiness", default=str(DEFAULT_READINESS))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-mem-csv", default=str(DEFAULT_OUT_MEM_CSV))
    p.add_argument("--out-mem-parquet", default=str(DEFAULT_OUT_MEM_PQ))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[ARM_RECOVERY_EFFECTIVENESS] starting "
        "(read-only recovery effectiveness audit; no broker calls)",
        flush=True,
    )

    recovery_summary = _safe_read_json(
        Path(args.recovery_summary), label="arm_governance_recovery_summary.json"
    )
    recovery_record = _safe_read_json(
        Path(args.recovery_record), label="arm_governance_recovery.json"
    )
    recovery_mem = _safe_read_csv_rows(
        Path(args.recovery_mem), label="arm_governance_recovery_memory.csv"
    )
    drift_summary = _safe_read_json(
        Path(args.drift_summary), label="arm_governance_drift_detection_summary.json"
    )
    drift_mem = _safe_read_csv_rows(Path(args.drift_mem), label="arm_governance_drift_memory.csv")
    violation_mem = _safe_read_csv_rows(
        Path(args.violation_mem), label="arm_constitution_violation_memory.csv"
    )
    precedent_mem = _safe_read_csv_rows(
        Path(args.precedent_mem), label="arm_constitutional_precedent_memory.csv"
    )
    shadow_perf = _safe_read_json(
        Path(args.shadow_perf), label="arm_shadow_performance_summary.json"
    )
    progression_mem = _safe_read_csv_rows(
        Path(args.progression_mem), label="arm_autonomy_progression_memory.csv"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    health = _safe_read_json(Path(args.health), label="autonomous_system_health_summary.json")
    readiness = _safe_read_json(Path(args.readiness), label="autonomous_readiness_summary.json")
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_recovery_effectiveness_memory.csv"
    )

    record, summary, md, merged_memory = build_effectiveness_audit(
        recovery_summary=recovery_summary,
        recovery_record=recovery_record,
        recovery_mem=recovery_mem,
        drift_summary=drift_summary,
        drift_mem=drift_mem,
        violation_mem=violation_mem,
        precedent_mem=precedent_mem,
        shadow_perf=shadow_perf,
        progression_mem=progression_mem,
        scorecard=scorecard,
        health=health,
        readiness=readiness,
        existing_effectiveness_mem=existing_mem,
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
            merged_memory, Path(args.out_mem_csv), columns=EFFECTIVENESS_MEMORY_COLUMNS
        )
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    booleans = record["governance_booleans"]
    print(
        "[ARM_RECOVERY_EFFECTIVENESS] "
        f"state={record['effectiveness_state']} "
        f"score={record['effectiveness_score']:.3f} "
        f"working={booleans['recovery_working']} "
        f"adjustment={booleans['recovery_adjustment_required']} "
        f"confidence={summary['governance_confidence_delta']:+.3f}",
        flush=True,
    )
    print(
        "[ARM_RECOVERY_EFFECTIVENESS_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False",
        flush=True,
    )
    print(
        f"[ARM_RECOVERY_EFFECTIVENESS_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_RECOVERY_EFFECTIVENESS_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
