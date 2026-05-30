"""
ARM Governance Doctrine Simulation Engine -- Step 43.

Reads:
    data/results/arm_governance_doctrine_activation_board_summary.json  (Step 42)
    data/results/arm_governance_doctrine_activation_board.json          (Step 42)
    data/results/arm_governance_doctrine_activation_memory.csv          (Step 42)
    data/results/runtime_policy_governed.json                         (Step 18)
    data/results/autonomous_governance_scorecard.json                 (Step 19)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/arm_supreme_governance_council_summary.json          (Step 34)
    data/results/arm_constitutional_court_summary.json                (Step 33)

Writes:
    data/results/arm_governance_doctrine_simulation.json
    data/results/arm_governance_doctrine_simulation.md
    data/results/arm_governance_doctrine_simulation_summary.json
    data/results/arm_governance_doctrine_simulation_memory.csv
    data/results/arm_governance_doctrine_simulation_memory.parquet

Purpose
-------
This engine answers:

    "What would happen if governance doctrine activation occurred?"

It simulates governance doctrine activation in a read-only sandbox before any
future runtime activation. Activation eligible != Activated. Simulated != Applied.
Simulation NEVER activates runtime policy. Constitutional law remains supreme.

Simulation state cascade
------------------------
    1. DOCTRINE_SIMULATION_INSTITUTIONAL  mature governance simulation process
    2. DOCTRINE_SIMULATION_RESTRICTED     elevated pressure; defensive simulation only
    3. DOCTRINE_SIMULATION_ACTIVE          doctrine simulated normally
    4. DOCTRINE_SIMULATION_FORMING         limited doctrine; weak confidence
    5. DOCTRINE_SIMULATION_DORMANT         no eligible doctrine to simulate

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens are never written literally; an import-time
self-check raises if they ever appear.

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* runtime_mutation_allowed is ALWAYS false. simulated_only is ALWAYS true.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only simulation memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to DOCTRINE_SIMULATION_DORMANT.
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

DEFAULT_ACTIVATION_SUM = RESULTS_DIR / "arm_governance_doctrine_activation_board_summary.json"
DEFAULT_ACTIVATION_REC = RESULTS_DIR / "arm_governance_doctrine_activation_board.json"
DEFAULT_ACTIVATION_MEM = RESULTS_DIR / "arm_governance_doctrine_activation_memory.csv"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_SUM = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_doctrine_simulation.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_doctrine_simulation.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_doctrine_simulation_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_doctrine_simulation_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_doctrine_simulation_memory.parquet"


# -----------------------------------------------------------
# Simulation state constants
# -----------------------------------------------------------
SIMULATION_DORMANT = "DOCTRINE_SIMULATION_DORMANT"
SIMULATION_FORMING = "DOCTRINE_SIMULATION_FORMING"
SIMULATION_RESTRICTED = "DOCTRINE_SIMULATION_RESTRICTED"
SIMULATION_ACTIVE = "DOCTRINE_SIMULATION_ACTIVE"
SIMULATION_INSTITUTIONAL = "DOCTRINE_SIMULATION_INSTITUTIONAL"

DECISION_SIMULATED = "SIMULATED"
DECISION_SIMULATED_DEFENSIVE = "SIMULATED_DEFENSIVE"
DECISION_DEFERRED = "SIMULATION_DEFERRED"
DECISION_REJECTED = "SIMULATION_REJECTED"
DECISION_OPERATOR_OBS = "SIMULATION_OPERATOR_OBSERVATION"

ACTIVATION_ELIGIBLE = frozenset({"ACTIVATION_ELIGIBLE", "ACTIVATION_LIMITED"})
ACTIVATION_DEFERRED = frozenset({"ACTIVATION_DEFERRED"})
ACTIVATION_REJECTED = frozenset({"ACTIVATION_REJECTED"})
ACTIVATION_OPERATOR = frozenset({"OPERATOR_ACTIVATION_REQUIRED"})

DEFENSIVE_POLICIES = frozenset(
    {
        "confidence_threshold",
        "deployment_threshold",
        "target_cash_pct",
        "max_position_pct",
        "persistence_threshold",
        "min_observations_before_graduation",
        "autonomy_readiness_threshold",
        "skepticism_threshold",
        "auto_lock_manual_after_overruling",
        "governance_monitoring_frequency_multiplier",
    }
)

LOOSENING_POLICIES = frozenset(
    {
        "autonomy_loosen_threshold",
        "reduce_confidence_threshold",
        "reduce_target_cash_pct",
    }
)

POLICY_EFFECTS: Dict[str, str] = {
    "confidence_threshold": "raises signal confidence floor; fewer low-confidence deployments",
    "deployment_threshold": "tightens capital deployment gate; reduces aggressive deployment",
    "target_cash_pct": "increases cash reserve posture; improves capital preservation",
    "max_position_pct": "reduces single-position concentration; improves capital preservation",
    "persistence_threshold": "extends signal persistence requirement; slows deployment cadence",
    "min_observations_before_graduation": "lengthens apprenticeship; graduation becomes harder",
    "autonomy_readiness_threshold": "raises autonomy readiness bar; operator approval frequency increases",
    "skepticism_threshold": "increases governance skepticism; tighter oversight on autonomy signals",
    "auto_lock_manual_after_overruling": "persists manual lock after court overruling; operator supremacy reinforced",
    "governance_monitoring_frequency_multiplier": "increases governance monitoring frequency; faster drift detection",
}

POLICY_CATEGORIES: Dict[str, str] = {
    "confidence_threshold": "confidence",
    "deployment_threshold": "confidence",
    "target_cash_pct": "capital_preservation",
    "max_position_pct": "capital_preservation",
    "persistence_threshold": "autonomy",
    "min_observations_before_graduation": "autonomy",
    "autonomy_readiness_threshold": "autonomy",
    "skepticism_threshold": "governance",
    "auto_lock_manual_after_overruling": "governance",
    "governance_monitoring_frequency_multiplier": "governance",
}

SIMULATION_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "simulation_state",
    "simulated_count",
    "defensive_count",
    "confidence",
    "simulated_effect",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_DOCTRINE_SIMULATION_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(SIMULATION_MEMORY_COLUMNS))
        for col in ("confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("simulated_count", "defensive_count"):
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


def _compute_delta(current: Any, simulated: Any) -> Any:
    c, s = _to_float(current), _to_float(simulated)
    if c is not None and s is not None:
        return round(s - c, 6)
    if isinstance(current, bool) or isinstance(simulated, bool):
        return simulated
    return None


def _is_defensive_policy(name: str, current: Any, simulated: Any) -> bool:
    if name in LOOSENING_POLICIES:
        return False
    if name not in DEFENSIVE_POLICIES:
        return False
    if name == "max_position_pct":
        c, s = _to_float(current), _to_float(simulated)
        return c is not None and s is not None and s < c
    if name == "auto_lock_manual_after_overruling":
        return simulated is True or str(simulated).lower() == "true"
    c, s = _to_float(current), _to_float(simulated)
    if c is not None and s is not None:
        return s >= c
    return True


def _risk_impact(name: str, activation_decision: str, conf: float, defensive: bool) -> str:
    if activation_decision in ACTIVATION_OPERATOR:
        return "HIGH"
    if name in LOOSENING_POLICIES or not defensive:
        return "HIGH"
    if activation_decision == "ACTIVATION_ELIGIBLE" and conf >= 0.70:
        return "LOW"
    if activation_decision == "ACTIVATION_LIMITED" or conf >= 0.58:
        return "MODERATE"
    return "HIGH"


def _system_health_score(health: Dict[str, Any]) -> float:
    status = _norm_upper(health.get("overall_status"))
    n_total = _to_float(health.get("n_artifacts_total")) or 1.0
    n_fresh = _to_float(health.get("n_artifacts_fresh")) or 0.0
    n_sub = _to_float(health.get("n_subsystems_total")) or 1.0
    n_healthy = _to_float(health.get("n_subsystems_healthy")) or 0.0
    freshness = n_fresh / max(n_total, 1.0)
    subsystem = n_healthy / max(n_sub, 1.0)
    base = freshness * 0.55 + subsystem * 0.45
    if status == "STALE":
        base *= 0.55
    elif status == "DEGRADED":
        base *= 0.75
    elif status == "HEALTHY":
        base = max(base, 0.75)
    return _clamp(base, 0.0, 1.0)


def _court_stability_score(ctx: Dict[str, Any]) -> float:
    score = 1.0
    if ctx["constitution_violated"]:
        score -= 0.35
    if ctx["court_ruling"] == "COURT_OVERRULED":
        score -= 0.25
    if ctx["council_ruling"] in ("GOVERNANCE_REVOKE_AUTONOMY", "GOVERNANCE_LOCKDOWN"):
        score -= 0.15
    if ctx["operator_pressure"]:
        score -= 0.05
    return _clamp(score, 0.0, 1.0)


def _runtime_value(runtime_policy: Dict[str, Any], name: str, fallback: Any) -> Any:
    if name in runtime_policy:
        return runtime_policy[name]
    aliases = runtime_policy.get("aliases") or {}
    alias_map = {
        "target_cash_pct": "target_cash_reserve_pct",
        "max_position_pct": "max_single_position_pct",
        "confidence_threshold": "min_deploy_confidence",
        "deployment_threshold": "deployment_threshold",
        "persistence_threshold": "deploy_persistence_floor",
    }
    alias = alias_map.get(name)
    if alias and alias in aliases:
        return aliases[alias]
    return fallback


# -----------------------------------------------------------
# Context extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    activation_summary: Dict[str, Any],
    activation_record: Dict[str, Any],
    activation_mem: List[Dict[str, str]],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    constitution_state = _norm_upper(
        court_summary.get("constitution_state") or council_summary.get("constitution_state")
    )
    mem_conf = [
        _to_float(r.get("activation_confidence"))
        for r in activation_mem
        if _to_float(r.get("activation_confidence")) is not None
    ]
    activation_memory_stability = 0.0
    if len(mem_conf) >= 2:
        mean_c = sum(mem_conf) / len(mem_conf)
        var = sum((c - mean_c) ** 2 for c in mem_conf) / len(mem_conf)
        activation_memory_stability = _clamp(1.0 - min(var * 4.0, 1.0), 0.0, 1.0)

    ctx: Dict[str, Any] = {
        "activation_state": _norm_upper(
            activation_summary.get("activation_state") or activation_record.get("activation_state")
        ),
        "activation_confidence": _clamp(
            _to_float(activation_summary.get("activation_confidence"))
            or _to_float(activation_record.get("activation_confidence"))
            or 0.0,
            0.0,
            1.0,
        ),
        "activated_doctrines": activation_record.get("activated_doctrines") or [],
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
            or activation_summary.get("operator_activation_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "readiness_score": _clamp(
            _to_float(readiness_summary.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "governance_quality": _clamp(
            _to_float(scores.get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "runtime_policy": runtime_policy,
        "activation_memory_depth": len(activation_mem),
        "activation_memory_stability": activation_memory_stability,
        "limited_activation_available": bool(
            activation_summary.get("limited_activation_available")
        ),
        "eligible_activation_available": bool(
            activation_summary.get("activation_eligible_doctrines_available")
        ),
    }
    ctx["court_stability"] = _court_stability_score(ctx)
    return ctx


def _simulatable_doctrines(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        d
        for d in ctx["activated_doctrines"]
        if _norm_upper(d.get("activation_decision")) in ACTIVATION_ELIGIBLE
    ]


# -----------------------------------------------------------
# Doctrine simulation
# -----------------------------------------------------------
def _simulate_doctrine(
    doctrine: Dict[str, Any],
    *,
    restricted: bool,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(doctrine.get("policy_name", ""))
    activation_decision = _norm_upper(doctrine.get("activation_decision"))
    conf = _to_float(doctrine.get("confidence")) or 0.0
    const_safe = bool(doctrine.get("constitutional_safe", True))
    current = _runtime_value(ctx["runtime_policy"], name, doctrine.get("current_value"))
    simulated = doctrine.get("proposed_value")
    delta = _compute_delta(current, simulated)
    defensive = _is_defensive_policy(name, current, simulated)
    effect = POLICY_EFFECTS.get(name, f"simulated governance adjustment for {name}")

    simulation_decision = DECISION_DEFERRED
    if activation_decision in ACTIVATION_OPERATOR:
        simulation_decision = DECISION_OPERATOR_OBS
    elif activation_decision in ACTIVATION_REJECTED or not const_safe:
        simulation_decision = DECISION_REJECTED
    elif activation_decision in ACTIVATION_DEFERRED:
        simulation_decision = DECISION_DEFERRED
    elif restricted and not defensive:
        simulation_decision = DECISION_DEFERRED
    elif activation_decision == "ACTIVATION_ELIGIBLE" and not restricted:
        simulation_decision = DECISION_SIMULATED
    elif activation_decision in ACTIVATION_ELIGIBLE:
        simulation_decision = DECISION_SIMULATED_DEFENSIVE
    else:
        simulation_decision = DECISION_DEFERRED

    return {
        "policy_name": name,
        "simulation_decision": simulation_decision,
        "current_value": current,
        "simulated_value": simulated,
        "delta": delta,
        "expected_governance_effect": effect,
        "risk_impact": _risk_impact(name, activation_decision, conf, defensive),
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "runtime_mutation_allowed": False,
        "simulated_only": True,
        "activation_decision": activation_decision,
        "policy_category": POLICY_CATEGORIES.get(name, "governance"),
    }


def _simulate_operator_observation(
    doctrine: Dict[str, Any],
    *,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(doctrine.get("policy_name", ""))
    conf = _to_float(doctrine.get("confidence")) or 0.0
    const_safe = bool(doctrine.get("constitutional_safe", True))
    current = _runtime_value(ctx["runtime_policy"], name, doctrine.get("current_value"))
    simulated = doctrine.get("proposed_value")
    delta = _compute_delta(current, simulated)
    effect = POLICY_EFFECTS.get(
        name,
        f"observed governance impact if {name} doctrine were activated under operator supervision",
    )
    return {
        "policy_name": name,
        "simulation_decision": DECISION_OPERATOR_OBS if const_safe else DECISION_REJECTED,
        "current_value": current,
        "simulated_value": simulated,
        "delta": delta,
        "expected_governance_effect": effect,
        "risk_impact": "HIGH",
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "runtime_mutation_allowed": False,
        "simulated_only": True,
        "activation_decision": _norm_upper(doctrine.get("activation_decision")),
        "policy_category": POLICY_CATEGORIES.get(name, "governance"),
    }


def _simulate_all(ctx: Dict[str, Any], restricted: bool) -> List[Dict[str, Any]]:
    simulated: List[Dict[str, Any]] = []
    for doctrine in ctx["activated_doctrines"]:
        act = _norm_upper(doctrine.get("activation_decision"))
        if act in ACTIVATION_ELIGIBLE:
            simulated.append(_simulate_doctrine(doctrine, restricted=restricted, ctx=ctx))
        elif act in ACTIVATION_OPERATOR:
            simulated.append(_simulate_operator_observation(doctrine, ctx=ctx))
        elif act in ACTIVATION_DEFERRED and _is_defensive_policy(
            str(doctrine.get("policy_name", "")),
            _runtime_value(
                ctx["runtime_policy"],
                str(doctrine.get("policy_name", "")),
                doctrine.get("current_value"),
            ),
            doctrine.get("proposed_value"),
        ):
            row = _simulate_doctrine(doctrine, restricted=restricted, ctx=ctx)
            row["simulation_decision"] = DECISION_DEFERRED
            simulated.append(row)
    return simulated


# -----------------------------------------------------------
# Simulation confidence and state
# -----------------------------------------------------------
def _simulation_confidence(ctx: Dict[str, Any], simulated: List[Dict[str, Any]]) -> float:
    active = [
        r
        for r in simulated
        if r["simulation_decision"] in (DECISION_SIMULATED, DECISION_SIMULATED_DEFENSIVE)
    ]
    avg_conf = sum(r["confidence"] for r in active) / max(len(active), 1) if active else 0.0
    const_safety = (
        sum(1 for r in active if r["constitutional_safe"]) / max(len(active), 1) if active else 0.0
    )

    raw = (
        ctx["activation_confidence"] * 0.22
        + const_safety * 0.18
        + ctx["governance_quality"] * 0.16
        + ctx["system_health_score"] * 0.16
        + ctx["readiness_score"] * 0.12
        + ctx["court_stability"] * 0.10
        + avg_conf * 0.06
    )

    penalty = ctx["constitutional_pressure"] * 0.18
    if ctx["constitution_violated"]:
        penalty += 0.10
    if ctx["court_ruling"] == "COURT_OVERRULED":
        penalty += 0.08
    if ctx["system_health_stale"]:
        penalty += 0.07
    if ctx["activation_confidence"] < 0.10:
        penalty += 0.05

    return round(_clamp(raw - penalty, 0.0, 1.0), 6)


def _count_simulations(simulated: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "simulated": sum(
            1
            for r in simulated
            if r["simulation_decision"] in (DECISION_SIMULATED, DECISION_SIMULATED_DEFENSIVE)
        ),
        "defensive": sum(
            1 for r in simulated if r["simulation_decision"] == DECISION_SIMULATED_DEFENSIVE
        ),
        "deferred": sum(1 for r in simulated if r["simulation_decision"] == DECISION_DEFERRED),
        "rejected": sum(1 for r in simulated if r["simulation_decision"] == DECISION_REJECTED),
        "operator_obs": sum(
            1 for r in simulated if r["simulation_decision"] == DECISION_OPERATOR_OBS
        ),
    }


def _classify_simulation_state(
    *,
    ctx: Dict[str, Any],
    simulation_confidence: float,
    simulated: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> Tuple[str, List[str], bool]:
    reasons: List[str] = []
    simulatable = _simulatable_doctrines(ctx)

    if not simulatable and counts["simulated"] == 0:
        reasons.append("no activation-eligible doctrine available for simulation")
        return SIMULATION_DORMANT, reasons, False

    restricted = (
        ctx["constitutional_pressure"] >= 0.55
        or ctx["constitution_violated"]
        or ctx["court_ruling"] == "COURT_OVERRULED"
        or ctx["activation_state"] == "DOCTRINE_ACTIVATION_RESTRICTED"
    )

    if (
        ctx["activation_state"] == "DOCTRINE_ACTIVATION_INSTITUTIONAL"
        and simulation_confidence >= 0.60
        and ctx["activation_memory_depth"] >= 3
        and counts["simulated"] >= 2
    ):
        reasons.append("mature governance simulation process with repeatable sandbox quality")
        return SIMULATION_INSTITUTIONAL, reasons, restricted

    if restricted:
        reasons.append("constitutional pressure elevated; defensive simulation only")
        return SIMULATION_RESTRICTED, reasons, True

    if simulation_confidence < 0.35 or ctx["activation_confidence"] < 0.20:
        reasons.append("limited doctrine available; simulation confidence weak")
        return SIMULATION_FORMING, reasons, False

    if simulation_confidence >= 0.40 and counts["simulated"] >= 1:
        reasons.append("doctrine simulated under normal governance sandbox process")
        return SIMULATION_ACTIVE, reasons, False

    reasons.append("simulation process forming institutional sandbox posture")
    return SIMULATION_FORMING, reasons, restricted


# -----------------------------------------------------------
# Governance impact, booleans, recommendations, rationale
# -----------------------------------------------------------
def _governance_impact(simulated: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    impact: Dict[str, List[str]] = {
        "confidence": [],
        "capital_preservation": [],
        "autonomy": [],
        "governance": [],
    }
    for r in simulated:
        if r["simulation_decision"] not in (
            DECISION_SIMULATED,
            DECISION_SIMULATED_DEFENSIVE,
            DECISION_OPERATOR_OBS,
        ):
            continue
        cat = r.get("policy_category", "governance")
        line = f"{r['policy_name']}: {r['expected_governance_effect']} (risk={r['risk_impact']})"
        impact.setdefault(cat, []).append(line)
    return impact


def _simulation_booleans(
    state: str,
    simulated: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, bool]:
    return {
        "doctrine_simulation_available": counts["simulated"] > 0 or counts["operator_obs"] > 0,
        "defensive_simulation_only": state == SIMULATION_RESTRICTED or counts["defensive"] > 0,
        "operator_review_required": (counts["operator_obs"] > 0 or ctx["operator_pressure"]),
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "simulated_only": True,
    }


def _aggregate_effect(simulated: List[Dict[str, Any]]) -> str:
    active = [
        r
        for r in simulated
        if r["simulation_decision"] in (DECISION_SIMULATED, DECISION_SIMULATED_DEFENSIVE)
    ]
    if not active:
        return "no doctrine activation simulated"
    names = ", ".join(r["policy_name"] for r in active[:3])
    return f"simulated defensive governance posture shift via {names}"


def _recommendations(
    state: str, simulated: List[Dict[str, Any]], counts: Dict[str, int]
) -> List[str]:
    recs: List[str] = []
    if counts["defensive"] > 0 or state == SIMULATION_RESTRICTED:
        recs.append("Continue defensive doctrine simulation")
    if any(
        r.get("policy_category") == "autonomy" and r["simulation_decision"] == DECISION_OPERATOR_OBS
        for r in simulated
    ):
        recs.append("Avoid autonomy loosening simulation")
    if counts["deferred"] > 0:
        recs.append("Increase observation before activation consideration")
    recs.append("Maintain runtime mutation lock")
    if counts["operator_obs"] > 0:
        recs.append("Escalate high-impact doctrine to operator review")
    if state == SIMULATION_RESTRICTED:
        recs.append("Require constitutional review before any future activation")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(simulated: List[Dict[str, Any]], state: str) -> str:
    cash = [
        r
        for r in simulated
        if r["policy_name"] == "target_cash_pct"
        and r["simulation_decision"] in (DECISION_SIMULATED, DECISION_SIMULATED_DEFENSIVE)
    ]
    if cash:
        return (
            "Triton simulated elevated cash doctrine and observed improved capital "
            "preservation posture under constitutional stress."
        )
    active = [
        r
        for r in simulated
        if r["simulation_decision"] in (DECISION_SIMULATED, DECISION_SIMULATED_DEFENSIVE)
    ]
    if active:
        names = ", ".join(r["policy_name"] for r in active[:3])
        return f"Triton simulated governance doctrine changes for {names} without runtime mutation."
    if state == SIMULATION_RESTRICTED:
        return (
            "Simulation operates in restricted mode due to elevated constitutional "
            "pressure; only defensive doctrine outcomes are modeled."
        )
    return "Governance doctrine simulation completed in read-only sandbox mode."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    simulation_confidence: float,
    simulated: List[Dict[str, Any]],
    impact: Dict[str, List[str]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Doctrine Simulation",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Simulation State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| simulation_confidence | {simulation_confidence:.3f} |",
        f"| simulated | {counts['simulated']} |",
        f"| defensive | {counts['defensive']} |",
        f"| deferred | {counts['deferred']} |",
        f"| operator_observation | {counts['operator_obs']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        f"| simulated_only | {booleans['simulated_only']} |",
        "",
        "## Simulated Doctrine Effects",
        "",
    ]
    if simulated:
        lines.append("| policy | decision | current | simulated | delta | risk |")
        lines.append("|---|---|---|---|---|---|")
        for r in simulated:
            lines.append(
                f"| {r['policy_name']} | {r['simulation_decision']} | {r['current_value']} | "
                f"{r['simulated_value']} | {r['delta']} | {r['risk_impact']} |"
            )
        lines.append("")
        for r in simulated:
            lines.append(
                f"- **{r['policy_name']}** ({r['simulation_decision']}): "
                f"{r['expected_governance_effect']}"
            )
    else:
        lines.append("_No doctrines simulated this cycle._")

    lines.extend(["", "## Governance Impact", ""])
    category_titles = {
        "confidence": "Confidence doctrine",
        "capital_preservation": "Capital preservation",
        "autonomy": "Autonomy doctrine",
        "governance": "Governance doctrine",
    }
    any_impact = False
    for cat, title in category_titles.items():
        items = impact.get(cat) or []
        if items:
            any_impact = True
            lines.append(f"### {title}")
            lines.append("")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")
    if not any_impact:
        lines.append("_No governance impact modeled this cycle._")
        lines.append("")

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
            "This is a governance sandbox only. Activation eligible ≠ Activated. "
            "Simulated ≠ Applied. No runtime policy is mutated. Constitutional law, "
            "court rulings, capital preservation doctrine, and operator supremacy remain supreme.",
            "",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------
# Simulation memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    simulation_confidence: float,
    counts: Dict[str, int],
    simulated_effect: str,
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "simulation_state": state,
        "simulated_count": counts["simulated"],
        "defensive_count": counts["defensive"],
        "confidence": round(simulation_confidence, 6),
        "simulated_effect": simulated_effect,
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
        for c in SIMULATION_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_doctrine_simulation(
    *,
    activation_summary: Dict[str, Any],
    activation_record: Dict[str, Any],
    activation_mem: List[Dict[str, str]],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    court_summary: Dict[str, Any],
    existing_simulation_mem: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        activation_summary=activation_summary,
        activation_record=activation_record,
        activation_mem=activation_mem,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        council_summary=council_summary,
        court_summary=court_summary,
    )

    pre_restricted = (
        ctx["constitutional_pressure"] >= 0.55
        or ctx["constitution_violated"]
        or ctx["court_ruling"] == "COURT_OVERRULED"
    )
    simulated = _simulate_all(ctx, restricted=pre_restricted)
    simulation_confidence = _simulation_confidence(ctx, simulated)
    counts = _count_simulations(simulated)

    state, reasons, restricted = _classify_simulation_state(
        ctx=ctx,
        simulation_confidence=simulation_confidence,
        simulated=simulated,
        counts=counts,
    )

    if restricted != pre_restricted:
        simulated = _simulate_all(ctx, restricted=restricted)
        simulation_confidence = _simulation_confidence(ctx, simulated)
        counts = _count_simulations(simulated)

    impact = _governance_impact(simulated)
    booleans = _simulation_booleans(state, simulated, ctx, counts)
    simulated_effect = _aggregate_effect(simulated)
    recommendations = _recommendations(state, simulated, counts)
    rationale = _build_rationale(simulated, state)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        simulation_confidence=simulation_confidence,
        counts=counts,
        simulated_effect=simulated_effect,
        rationale=rationale,
    )
    merged_memory = _merge_memory(existing_simulation_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        simulation_confidence=simulation_confidence,
        simulated=simulated,
        impact=impact,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_simulation_engine",
        "engine_version": 1,
        "simulation_state": state,
        "simulation_confidence": simulation_confidence,
        "simulation_reasons": reasons,
        "simulated_doctrines": simulated,
        "governance_impact": impact,
        "simulation_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "simulated_effect_summary": simulated_effect,
        "activation_vs_simulated_note": (
            "Activation eligible ≠ Activated. Simulated ≠ Applied. "
            "This engine models outcomes only; it never mutates runtime policy."
        ),
        "constitutional_supremacy_note": (
            "Simulation cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "simulation_memory_size_after_append": len(merged_memory),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "simulated_only": True,
            "simulation_sandbox_only": True,
        },
        "inputs_seen": {
            "arm_governance_doctrine_activation_board_summary": bool(activation_summary),
            "arm_governance_doctrine_activation_board_record": bool(activation_record),
            "arm_governance_doctrine_activation_memory_rows": len(activation_mem),
            "runtime_policy_governed": bool(runtime_policy),
            "autonomous_governance_scorecard": bool(scorecard),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "arm_constitutional_court_summary": bool(court_summary),
            "existing_simulation_memory_rows": len(existing_simulation_mem),
            "n_simulatable_doctrines": len(_simulatable_doctrines(ctx)),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_simulation_engine",
        "simulation_state": state,
        "simulation_confidence": simulation_confidence,
        "doctrine_simulation_available": booleans["doctrine_simulation_available"],
        "defensive_simulation_only": booleans["defensive_simulation_only"],
        "operator_review_required": booleans["operator_review_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "simulated_only": booleans["simulated_only"],
        "simulated_count": counts["simulated"],
        "defensive_count": counts["defensive"],
        "deferred_count": counts["deferred"],
        "operator_observation_count": counts["operator_obs"],
        "n_doctrines_modeled": len(simulated),
        "n_recommendations": len(recommendations),
        "simulation_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance doctrine simulation engine (Step 43). "
            "Simulates doctrine activation outcomes without runtime mutation. No broker calls."
        ),
    )
    p.add_argument("--activation-summary", default=str(DEFAULT_ACTIVATION_SUM))
    p.add_argument("--activation-record", default=str(DEFAULT_ACTIVATION_REC))
    p.add_argument("--activation-mem", default=str(DEFAULT_ACTIVATION_MEM))
    p.add_argument("--runtime-policy", default=str(DEFAULT_RUNTIME_POLICY))
    p.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUM))
    p.add_argument("--council-summary", default=str(DEFAULT_COUNCIL_SUMMARY))
    p.add_argument("--court-summary", default=str(DEFAULT_COURT_SUMMARY))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--out-summary", default=str(DEFAULT_OUT_SUMMARY))
    p.add_argument("--out-mem-csv", default=str(DEFAULT_OUT_MEM_CSV))
    p.add_argument("--out-mem-parquet", default=str(DEFAULT_OUT_MEM_PQ))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        "[ARM_DOCTRINE_SIMULATION] starting "
        "(read-only governance doctrine sandbox; no runtime mutation; no broker calls)",
        flush=True,
    )

    activation_summary = _safe_read_json(
        Path(args.activation_summary), label="arm_governance_doctrine_activation_board_summary.json"
    )
    activation_record = _safe_read_json(
        Path(args.activation_record), label="arm_governance_doctrine_activation_board.json"
    )
    activation_mem = _safe_read_csv_rows(
        Path(args.activation_mem), label="arm_governance_doctrine_activation_memory.csv"
    )
    runtime_policy = _safe_read_json(
        Path(args.runtime_policy), label="runtime_policy_governed.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="autonomous_readiness_summary.json"
    )
    council_summary = _safe_read_json(
        Path(args.council_summary), label="arm_supreme_governance_council_summary.json"
    )
    court_summary = _safe_read_json(
        Path(args.court_summary), label="arm_constitutional_court_summary.json"
    )
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_doctrine_simulation_memory.csv"
    )

    record, summary, md, merged_memory = build_doctrine_simulation(
        activation_summary=activation_summary,
        activation_record=activation_record,
        activation_mem=activation_mem,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        council_summary=council_summary,
        court_summary=court_summary,
        existing_simulation_mem=existing_mem,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=SIMULATION_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["simulation_counts"]
    restricted_flag = 1 if record["simulation_state"] == SIMULATION_RESTRICTED else 0
    print(
        "[ARM_DOCTRINE_SIMULATION] "
        f"state={record['simulation_state']} "
        f"simulated={counts['simulated']} "
        f"restricted={restricted_flag} "
        f"confidence={record['simulation_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_DOCTRINE_SIMULATION_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False "
        "runtime_mutation=False simulated_only=True",
        flush=True,
    )
    print(
        f"[ARM_DOCTRINE_SIMULATION_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_DOCTRINE_SIMULATION_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
