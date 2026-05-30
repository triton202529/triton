"""
ARM Governance Doctrine Activation Board Engine -- Step 42.

Reads:
    data/results/arm_governance_policy_ratification_summary.json  (Step 41)
    data/results/arm_governance_policy_ratification.json          (Step 41)
    data/results/arm_governance_ratification_memory.csv           (Step 41)
    data/results/arm_constitutional_court_summary.json            (Step 33)
    data/results/arm_autonomy_constitution_summary.json           (Step 32)
    data/results/arm_supreme_governance_council_summary.json      (Step 34)
    data/results/autonomous_system_health_summary.json
    data/results/autonomous_readiness_summary.json
    data/results/runtime_policy_governed.json                     (Step 18)
    data/results/autonomous_governance_scorecard.json             (Step 19)

Writes:
    data/results/arm_governance_doctrine_activation_board.json
    data/results/arm_governance_doctrine_activation_board.md
    data/results/arm_governance_doctrine_activation_board_summary.json
    data/results/arm_governance_doctrine_activation_memory.csv
    data/results/arm_governance_doctrine_activation_memory.parquet

Purpose
-------
This engine answers:

    "Which ratified governance doctrines are activation-eligible?"

It evaluates ratified governance doctrines and determines which are safe for
activation consideration. Ratified != Activated. Activation-eligible !=
Runtime-mutated. This engine NEVER mutates runtime policy.

Activation board state cascade
------------------------------
    1. DOCTRINE_ACTIVATION_INSTITUTIONAL  mature doctrine activation process
    2. DOCTRINE_ACTIVATION_RESTRICTED     elevated pressure; defensive only
    3. DOCTRINE_ACTIVATION_ACTIVE          eligibility evaluated normally
    4. DOCTRINE_ACTIVATION_FORMING         ratified doctrines; weak confidence
    5. DOCTRINE_ACTIVATION_DORMANT         no ratified doctrines

Hard safety rule
----------------
This file MUST NOT import or reference any broker/execution module.
Forbidden tokens are never written literally; an import-time
self-check raises if they ever appear.

Safety
------
* STRICT READ ONLY. No broker calls, no portfolio mutation.
* runtime_mutation_allowed is ALWAYS false on every doctrine record.
* Atomic writes for JSON / MD / CSV / Parquet.
* Append-only activation memory keyed by timestamp.
* Missing inputs warn-and-continue; defaults to DOCTRINE_ACTIVATION_DORMANT.
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

DEFAULT_RATIFICATION_SUM = RESULTS_DIR / "arm_governance_policy_ratification_summary.json"
DEFAULT_RATIFICATION_REC = RESULTS_DIR / "arm_governance_policy_ratification.json"
DEFAULT_RATIFICATION_MEM = RESULTS_DIR / "arm_governance_ratification_memory.csv"
DEFAULT_COURT_SUMMARY = RESULTS_DIR / "arm_constitutional_court_summary.json"
DEFAULT_CONSTITUTION_SUM = RESULTS_DIR / "arm_autonomy_constitution_summary.json"
DEFAULT_COUNCIL_SUMMARY = RESULTS_DIR / "arm_supreme_governance_council_summary.json"
DEFAULT_HEALTH_SUMMARY = RESULTS_DIR / "autonomous_system_health_summary.json"
DEFAULT_READINESS_SUM = RESULTS_DIR / "autonomous_readiness_summary.json"
DEFAULT_RUNTIME_POLICY = RESULTS_DIR / "runtime_policy_governed.json"
DEFAULT_SCORECARD = RESULTS_DIR / "autonomous_governance_scorecard.json"

DEFAULT_OUT_JSON = RESULTS_DIR / "arm_governance_doctrine_activation_board.json"
DEFAULT_OUT_MD = RESULTS_DIR / "arm_governance_doctrine_activation_board.md"
DEFAULT_OUT_SUMMARY = RESULTS_DIR / "arm_governance_doctrine_activation_board_summary.json"
DEFAULT_OUT_MEM_CSV = RESULTS_DIR / "arm_governance_doctrine_activation_memory.csv"
DEFAULT_OUT_MEM_PQ = RESULTS_DIR / "arm_governance_doctrine_activation_memory.parquet"


# -----------------------------------------------------------
# Activation state constants
# -----------------------------------------------------------
ACTIVATION_DORMANT = "DOCTRINE_ACTIVATION_DORMANT"
ACTIVATION_FORMING = "DOCTRINE_ACTIVATION_FORMING"
ACTIVATION_RESTRICTED = "DOCTRINE_ACTIVATION_RESTRICTED"
ACTIVATION_ACTIVE = "DOCTRINE_ACTIVATION_ACTIVE"
ACTIVATION_INSTITUTIONAL = "DOCTRINE_ACTIVATION_INSTITUTIONAL"

DECISION_ELIGIBLE = "ACTIVATION_ELIGIBLE"
DECISION_LIMITED = "ACTIVATION_LIMITED"
DECISION_DEFERRED = "ACTIVATION_DEFERRED"
DECISION_REJECTED = "ACTIVATION_REJECTED"
DECISION_OPERATOR = "OPERATOR_ACTIVATION_REQUIRED"

SCOPE_NONE = "NONE"
SCOPE_LIMITED = "LIMITED"
SCOPE_FULL = "FULL"
SCOPE_OPERATOR = "OPERATOR"

RATIFIED_DECISIONS = frozenset({"RATIFIED", "RATIFIED_LIMITED"})
OPERATOR_RATIFICATION = frozenset({"OPERATOR_RATIFICATION_REQUIRED"})

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

HIGH_IMPACT_POLICIES = frozenset(
    {
        "confidence_threshold",
        "deployment_threshold",
        "autonomy_readiness_threshold",
        "min_observations_before_graduation",
        "auto_lock_manual_after_overruling",
        "skepticism_threshold",
        "persistence_threshold",
    }
)

ACTIVATION_MEMORY_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "activation_state",
    "eligible_count",
    "limited_count",
    "deferred_count",
    "rejected_count",
    "operator_required_count",
    "activation_confidence",
    "rationale",
)


# -----------------------------------------------------------
# Safe IO
# -----------------------------------------------------------
def _warn(msg: str) -> None:
    print(f"[ARM_DOCTRINE_ACTIVATION_WARN] {msg}", flush=True)


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
        df = pd.DataFrame(rows, columns=list(ACTIVATION_MEMORY_COLUMNS))
        for col in ("activation_confidence",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in (
            "eligible_count",
            "limited_count",
            "deferred_count",
            "rejected_count",
            "operator_required_count",
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


def _is_defensive_doctrine(name: str, current: Any, proposed: Any) -> bool:
    if name in LOOSENING_POLICIES:
        return False
    if name not in DEFENSIVE_POLICIES:
        return False
    if name == "max_position_pct":
        c, p = _to_float(current), _to_float(proposed)
        return c is not None and p is not None and p < c
    if name == "auto_lock_manual_after_overruling":
        return proposed is True or str(proposed).lower() == "true"
    c, p = _to_float(current), _to_float(proposed)
    if c is not None and p is not None:
        return p >= c
    return True


def _is_aggressive_doctrine(name: str, current: Any, proposed: Any) -> bool:
    if name in LOOSENING_POLICIES:
        return True
    if name == "max_position_pct":
        c, p = _to_float(current), _to_float(proposed)
        return c is not None and p is not None and p > c
    if name in (
        "confidence_threshold",
        "deployment_threshold",
        "target_cash_pct",
        "persistence_threshold",
        "skepticism_threshold",
        "autonomy_readiness_threshold",
    ):
        c, p = _to_float(current), _to_float(proposed)
        return c is not None and p is not None and p < c
    return False


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


# -----------------------------------------------------------
# Context extraction
# -----------------------------------------------------------
def _extract_context(
    *,
    ratification_summary: Dict[str, Any],
    ratification_record: Dict[str, Any],
    ratification_mem: List[Dict[str, str]],
    court_summary: Dict[str, Any],
    constitution_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
) -> Dict[str, Any]:
    scores = scorecard.get("scores") or {}
    mem_conf = [
        _to_float(r.get("ratification_confidence"))
        for r in ratification_mem
        if _to_float(r.get("ratification_confidence")) is not None
    ]
    ratification_memory_stability = 0.0
    if len(mem_conf) >= 2:
        mean_c = sum(mem_conf) / len(mem_conf)
        var = sum((c - mean_c) ** 2 for c in mem_conf) / len(mem_conf)
        ratification_memory_stability = _clamp(1.0 - min(var * 4.0, 1.0), 0.0, 1.0)

    constitution_state = _norm_upper(
        constitution_summary.get("constitution_state")
        or court_summary.get("constitution_state")
        or council_summary.get("constitution_state")
    )

    ctx: Dict[str, Any] = {
        "ratification_state": _norm_upper(
            ratification_summary.get("ratification_state")
            or ratification_record.get("ratification_state")
        ),
        "ratification_confidence": _clamp(
            _to_float(ratification_summary.get("ratification_confidence"))
            or _to_float(ratification_record.get("ratification_confidence"))
            or 0.0,
            0.0,
            1.0,
        ),
        "governance_doctrine": ratification_record.get("governance_doctrine") or [],
        "ratified_policies": ratification_record.get("ratified_policies") or [],
        "constitutional_pressure": _clamp(
            0.75 if constitution_state == "CONSTITUTION_VIOLATED" else 0.30,
            0.0,
            1.0,
        ),
        "constitution_violated": (
            constitution_state == "CONSTITUTION_VIOLATED"
            or bool(constitution_summary.get("constitution_violated"))
        ),
        "court_ruling": _norm_upper(court_summary.get("judicial_ruling")),
        "council_ruling": _norm_upper(council_summary.get("governance_ruling")),
        "operator_pressure": bool(
            court_summary.get("operator_review_required")
            or council_summary.get("operator_supervision_required")
            or constitution_summary.get("operator_override_required")
            or ratification_summary.get("operator_ratification_required")
        ),
        "system_health_score": _system_health_score(health_summary),
        "system_health_stale": _norm_upper(health_summary.get("overall_status")) == "STALE",
        "readiness_score": _clamp(
            _to_float(readiness_summary.get("readiness_score")) or 0.0,
            0.0,
            1.0,
        ),
        "readiness_state": _norm_upper(readiness_summary.get("readiness_state")),
        "governance_quality": _clamp(
            _to_float(scores.get("governance_quality_score")) or 0.5,
            0.0,
            1.0,
        ),
        "regime": _norm_upper(runtime_policy.get("regime") or scorecard.get("regime")),
        "ratification_memory_depth": len(ratification_mem),
        "ratification_memory_stability": ratification_memory_stability,
        "defensive_constraints_required": bool(
            constitution_summary.get("defensive_constraints_required")
        ),
        "autonomy_allowed": bool(
            constitution_summary.get("autonomy_constitutionally_allowed")
            and court_summary.get("autonomy_judicially_allowed")
        ),
    }
    ctx["court_stability"] = _court_stability_score(ctx)
    return ctx


def _policy_lookup(ctx: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in ctx["ratified_policies"]:
        name = str(p.get("policy_name", ""))
        if name:
            out[name] = p
    return out


# -----------------------------------------------------------
# Doctrine activation evaluation
# -----------------------------------------------------------
def _activation_scope(decision: str) -> str:
    if decision == DECISION_ELIGIBLE:
        return SCOPE_FULL
    if decision == DECISION_LIMITED:
        return SCOPE_LIMITED
    if decision == DECISION_OPERATOR:
        return SCOPE_OPERATOR
    return SCOPE_NONE


def _evaluate_doctrine(
    doctrine: Dict[str, Any],
    policy: Dict[str, Any],
    *,
    restricted: bool,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(doctrine.get("policy_name", ""))
    rat_scope = _norm_upper(doctrine.get("ratification_scope"))
    conf = _to_float(doctrine.get("confidence")) or _to_float(policy.get("confidence")) or 0.0
    const_safe = bool(policy.get("constitutional_safe", True))
    op_flag = bool(policy.get("operator_ratification_required"))
    current = doctrine.get("prior_value", policy.get("current_value"))
    proposed = doctrine.get("doctrine_value", policy.get("proposed_value"))
    defensive = _is_defensive_doctrine(name, current, proposed)
    aggressive = _is_aggressive_doctrine(name, current, proposed)

    decision = DECISION_DEFERRED
    activation_rationale = "insufficient stability for doctrine activation eligibility"

    if rat_scope in OPERATOR_RATIFICATION or op_flag:
        decision = DECISION_OPERATOR
        activation_rationale = (
            "operator activation required for high-impact doctrine with runtime implications"
        )
    elif not const_safe:
        decision = DECISION_REJECTED
        activation_rationale = "rejected due to constitutional safety concern"
    elif aggressive:
        decision = DECISION_REJECTED
        activation_rationale = "rejected: doctrine loosens governance under instability"
    elif (
        ctx["constitution_violated"]
        and ctx["council_ruling"] == "GOVERNANCE_REVOKE_AUTONOMY"
        and (aggressive or not defensive)
    ):
        decision = DECISION_REJECTED
        activation_rationale = (
            "rejected due to court/council conflict under constitutional violation"
        )
    elif ctx["system_health_stale"] and conf < 0.65:
        decision = DECISION_DEFERRED
        activation_rationale = (
            "deferred: stale system health requires pipeline refresh before activation"
        )
    elif restricted and not defensive:
        decision = DECISION_DEFERRED
        activation_rationale = (
            "deferred: restricted board permits only defensive doctrine activation"
        )
    elif name in HIGH_IMPACT_POLICIES and (
        ctx["constitutional_pressure"] >= 0.55 or ctx["constitution_violated"]
    ):
        decision = DECISION_OPERATOR
        activation_rationale = (
            "operator activation required for constitutionally sensitive high-impact doctrine"
        )
    elif (
        rat_scope == "RATIFIED"
        and conf >= 0.72
        and const_safe
        and defensive
        and ctx["system_health_score"] >= 0.60
        and ctx["readiness_score"] >= 0.55
        and not restricted
    ):
        decision = DECISION_ELIGIBLE
        activation_rationale = f"activation eligible: {name} is constitutionally safe with strong confidence {conf:.2f}"
    elif defensive and conf >= 0.58 and const_safe:
        if restricted or rat_scope == "RATIFIED_LIMITED" or ctx["system_health_stale"]:
            decision = DECISION_LIMITED
            activation_rationale = "limited activation eligible: defensive doctrine safe for gradual activation consideration"
        elif conf >= 0.68 and ctx["readiness_score"] >= 0.50:
            decision = DECISION_ELIGIBLE
            activation_rationale = (
                f"activation eligible: defensive {name} aligned with capital preservation"
            )
        else:
            decision = DECISION_LIMITED
            activation_rationale = "limited activation eligible; gradual activation recommended"
    elif conf >= 0.50 and const_safe and not restricted:
        decision = DECISION_LIMITED
        activation_rationale = "limited activation eligible pending additional stability evidence"
    else:
        decision = DECISION_DEFERRED
        activation_rationale = (
            "deferred pending stronger activation confidence and system stability"
        )

    scope = _activation_scope(decision)
    return {
        "policy_name": name,
        "activation_decision": decision,
        "current_value": current,
        "proposed_value": proposed,
        "activation_scope": scope,
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "operator_activation_required": op_flag or decision == DECISION_OPERATOR,
        "activation_rationale": activation_rationale,
        "runtime_mutation_allowed": False,
        "ratification_scope": rat_scope,
    }


def _evaluate_operator_pending(
    policy: Dict[str, Any],
    *,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(policy.get("policy_name", ""))
    conf = _to_float(policy.get("confidence")) or 0.0
    const_safe = bool(policy.get("constitutional_safe", True))
    current = policy.get("current_value")
    proposed = policy.get("proposed_value")
    rat_dec = _norm_upper(policy.get("ratification_decision"))

    if rat_dec not in OPERATOR_RATIFICATION:
        return {}

    decision = DECISION_OPERATOR
    rationale = (
        "operator activation required; doctrine not fully ratified for autonomous activation"
    )
    if not const_safe:
        decision = DECISION_REJECTED
        rationale = "rejected: operator-pending doctrine fails constitutional safety"

    return {
        "policy_name": name,
        "activation_decision": decision,
        "current_value": current,
        "proposed_value": proposed,
        "activation_scope": SCOPE_OPERATOR if decision == DECISION_OPERATOR else SCOPE_NONE,
        "confidence": round(conf, 4),
        "constitutional_safe": const_safe,
        "operator_activation_required": True,
        "activation_rationale": rationale,
        "runtime_mutation_allowed": False,
        "ratification_scope": rat_dec,
    }


def _evaluate_all(ctx: Dict[str, Any], restricted: bool) -> List[Dict[str, Any]]:
    policy_map = _policy_lookup(ctx)
    evaluated: List[Dict[str, Any]] = []
    seen: set = set()

    for doctrine in ctx["governance_doctrine"]:
        name = str(doctrine.get("policy_name", ""))
        if not name:
            continue
        policy = policy_map.get(name, {})
        evaluated.append(_evaluate_doctrine(doctrine, policy, restricted=restricted, ctx=ctx))
        seen.add(name)

    for policy in ctx["ratified_policies"]:
        name = str(policy.get("policy_name", ""))
        if not name or name in seen:
            continue
        rat_dec = _norm_upper(policy.get("ratification_decision"))
        if rat_dec in OPERATOR_RATIFICATION:
            row = _evaluate_operator_pending(policy, ctx=ctx)
            if row:
                evaluated.append(row)
                seen.add(name)

    return evaluated


# -----------------------------------------------------------
# Activation confidence and board state
# -----------------------------------------------------------
def _activation_confidence(
    ctx: Dict[str, Any],
    evaluated: List[Dict[str, Any]],
) -> float:
    avg_conf = 0.0
    if evaluated:
        avg_conf = sum(r["confidence"] for r in evaluated) / len(evaluated)
    const_safety = sum(1 for r in evaluated if r["constitutional_safe"]) / max(len(evaluated), 1)

    raw = (
        ctx["ratification_confidence"] * 0.22
        + const_safety * 0.18
        + ctx["system_health_score"] * 0.18
        + ctx["readiness_score"] * 0.15
        + ctx["governance_quality"] * 0.12
        + ctx["court_stability"] * 0.10
        + avg_conf * 0.05
    )

    penalty = ctx["constitutional_pressure"] * 0.20
    if ctx["constitution_violated"]:
        penalty += 0.10
    if ctx["court_ruling"] == "COURT_OVERRULED":
        penalty += 0.08
    if ctx["system_health_stale"]:
        penalty += 0.07
    rejected = sum(1 for r in evaluated if r["activation_decision"] == DECISION_REJECTED)
    if evaluated and rejected / len(evaluated) > 0.25:
        penalty += 0.05
    if ctx["ratification_confidence"] < 0.15:
        penalty += 0.05

    return round(_clamp(raw - penalty, 0.0, 1.0), 6)


def _classify_activation_state(
    *,
    ctx: Dict[str, Any],
    activation_confidence: float,
    evaluated: List[Dict[str, Any]],
) -> Tuple[str, List[str], bool]:
    reasons: List[str] = []
    n_doctrine = len(ctx["governance_doctrine"])

    if n_doctrine == 0:
        reasons.append("no ratified governance doctrines available for activation evaluation")
        return ACTIVATION_DORMANT, reasons, False

    restricted = (
        ctx["constitutional_pressure"] >= 0.55
        or ctx["constitution_violated"]
        or ctx["court_ruling"] == "COURT_OVERRULED"
        or ctx["council_ruling"] in ("GOVERNANCE_REVOKE_AUTONOMY", "GOVERNANCE_LOCKDOWN")
        or ctx["ratification_state"] == "POLICY_RATIFICATION_CONSERVATIVE"
    )

    n_eligible = sum(
        1 for r in evaluated if r["activation_decision"] in (DECISION_ELIGIBLE, DECISION_LIMITED)
    )

    if (
        ctx["ratification_state"] == "POLICY_RATIFICATION_INSTITUTIONAL"
        and activation_confidence >= 0.62
        and ctx["ratification_memory_depth"] >= 3
        and n_eligible >= 2
    ):
        reasons.append("mature doctrine activation process with repeatable eligibility quality")
        return ACTIVATION_INSTITUTIONAL, reasons, restricted

    if restricted:
        reasons.append("constitutional pressure elevated; only defensive doctrines eligible")
        return ACTIVATION_RESTRICTED, reasons, True

    if activation_confidence < 0.35 or ctx["ratification_confidence"] < 0.25:
        reasons.append("ratified doctrines exist; activation confidence weak")
        return ACTIVATION_FORMING, reasons, False

    if activation_confidence >= 0.45 and n_doctrine >= 1:
        reasons.append("activation eligibility evaluated under normal board process")
        return ACTIVATION_ACTIVE, reasons, False

    reasons.append("activation board forming institutional eligibility posture")
    return ACTIVATION_FORMING, reasons, restricted


# -----------------------------------------------------------
# Booleans, recommendations, rationale
# -----------------------------------------------------------
def _activation_booleans(
    state: str,
    evaluated: List[Dict[str, Any]],
    ctx: Dict[str, Any],
) -> Dict[str, bool]:
    eligible = [r for r in evaluated if r["activation_decision"] == DECISION_ELIGIBLE]
    limited = [r for r in evaluated if r["activation_decision"] == DECISION_LIMITED]
    op_required = any(
        r["activation_decision"] == DECISION_OPERATOR or r["operator_activation_required"]
        for r in evaluated
    )
    return {
        "activation_eligible_doctrines_available": len(eligible) > 0,
        "limited_activation_available": len(limited) > 0,
        "operator_activation_required": op_required or ctx["operator_pressure"],
        "constitutional_review_required": (
            ctx["constitution_violated"] or ctx["court_ruling"] == "COURT_OVERRULED"
        ),
        "runtime_mutation_allowed": False,
        "activation_memory_reliable": state == ACTIVATION_INSTITUTIONAL,
    }


def _recommendations(evaluated: List[Dict[str, Any]], state: str) -> List[str]:
    recs: List[str] = []
    by_name = {r["policy_name"]: r for r in evaluated}

    if any(r["activation_decision"] == DECISION_LIMITED for r in evaluated):
        recs.append("Activate defensive doctrine only after operator confirmation")
    if any(
        _is_aggressive_doctrine(r["policy_name"], r["current_value"], r["proposed_value"])
        for r in evaluated
    ):
        recs.append("Defer autonomy-loosening doctrine")
    recs.append("Maintain runtime mutation lock")
    if any(r["activation_decision"] == DECISION_DEFERRED for r in evaluated):
        recs.append("Continue doctrine observation")
    if any(r["activation_decision"] == DECISION_OPERATOR for r in evaluated):
        recs.append("Require operator confirmation before any activation consideration")
    if state == ACTIVATION_RESTRICTED:
        recs.append("Require constitutional review before activation")
    if by_name.get("target_cash_pct", {}).get("activation_decision") == DECISION_LIMITED:
        recs.append("Consider limited activation of elevated cash posture doctrine")
    seen: set = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _build_rationale(evaluated: List[Dict[str, Any]], state: str) -> str:
    limited = [
        r
        for r in evaluated
        if r["activation_decision"] == DECISION_LIMITED and r["policy_name"] == "target_cash_pct"
    ]
    if limited:
        return (
            "Triton marked elevated cash posture as limited activation eligible because it is "
            "defensive, constitutionally safe, and aligned with capital preservation under stress."
        )
    eligible = [r for r in evaluated if r["activation_decision"] == DECISION_ELIGIBLE]
    if eligible:
        names = ", ".join(r["policy_name"] for r in eligible[:3])
        return f"Triton marked {names} as activation eligible under current governance stability."
    if state == ACTIVATION_RESTRICTED:
        return (
            "The activation board operates in restricted mode due to elevated constitutional "
            "pressure; only defensive doctrine may be considered for limited activation."
        )
    deferred = sum(1 for r in evaluated if r["activation_decision"] == DECISION_DEFERRED)
    if deferred:
        return f"The board deferred {deferred} doctrine(s) pending stronger system stability."
    return "The activation board completed eligibility evaluation of ratified governance doctrines."


# -----------------------------------------------------------
# Markdown
# -----------------------------------------------------------
def _render_markdown(
    *,
    generated_at: str,
    state: str,
    activation_confidence: float,
    evaluated: List[Dict[str, Any]],
    booleans: Dict[str, bool],
    reasons: List[str],
    rationale: str,
    recommendations: List[str],
    counts: Dict[str, int],
) -> str:
    lines = [
        "# Triton Governance Doctrine Activation Board",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Activation Board State",
        "",
        f"**{state}**",
        "",
        "| field | value |",
        "|---|---|",
        f"| activation_confidence | {activation_confidence:.3f} |",
        f"| eligible | {counts['eligible']} |",
        f"| limited | {counts['limited']} |",
        f"| deferred | {counts['deferred']} |",
        f"| rejected | {counts['rejected']} |",
        f"| operator_activation | {counts['operator_required']} |",
        f"| runtime_mutation_allowed | {booleans['runtime_mutation_allowed']} |",
        "",
        "## Doctrine Activation Decisions",
        "",
    ]
    if evaluated:
        lines.append("| policy | decision | scope | confidence | operator |")
        lines.append("|---|---|---|---|---|")
        for r in evaluated:
            lines.append(
                f"| {r['policy_name']} | {r['activation_decision']} | {r['activation_scope']} | "
                f"{r['confidence']:.2f} | {r['operator_activation_required']} |"
            )
        lines.append("")
        for r in evaluated:
            lines.append(
                f"- **{r['policy_name']}** ({r['activation_decision']}): {r['activation_rationale']}"
            )
    else:
        lines.append("_No ratified doctrines to evaluate._")

    eligible_rows = [r for r in evaluated if r["activation_decision"] == DECISION_ELIGIBLE]
    limited_rows = [r for r in evaluated if r["activation_decision"] == DECISION_LIMITED]
    activation_rows = eligible_rows + limited_rows

    lines.extend(["", "## Activation-Eligible Doctrines", ""])
    if activation_rows:
        for r in activation_rows:
            lines.append(
                f"- {r['policy_name']}: {r['current_value']} → {r['proposed_value']} "
                f"({r['activation_decision']}, scope={r['activation_scope']})"
            )
    else:
        lines.append("_None activation-eligible this cycle._")

    restricted_rows = [
        r
        for r in evaluated
        if r["activation_decision"] in (DECISION_DEFERRED, DECISION_REJECTED, DECISION_OPERATOR)
    ]
    lines.extend(["", "## Deferred or Restricted Doctrines", ""])
    if restricted_rows:
        for r in restricted_rows:
            lines.append(
                f"- {r['policy_name']} ({r['activation_decision']}): {r['activation_rationale']}"
            )
    else:
        lines.append("_None deferred or restricted._")

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
            "The activation board evaluates eligibility only. Ratified ≠ Activated. "
            "Activation-eligible ≠ Runtime-mutated. This engine never mutates runtime policy. "
            "Constitutional law, court rulings, capital preservation doctrine, and operator "
            "supremacy remain supreme.",
            "",
        ]
    )
    return "\n".join(lines)


def _count_decisions(evaluated: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "eligible": sum(1 for r in evaluated if r["activation_decision"] == DECISION_ELIGIBLE),
        "limited": sum(1 for r in evaluated if r["activation_decision"] == DECISION_LIMITED),
        "deferred": sum(1 for r in evaluated if r["activation_decision"] == DECISION_DEFERRED),
        "rejected": sum(1 for r in evaluated if r["activation_decision"] == DECISION_REJECTED),
        "operator_required": sum(
            1 for r in evaluated if r["activation_decision"] == DECISION_OPERATOR
        ),
    }


# -----------------------------------------------------------
# Activation memory
# -----------------------------------------------------------
def _build_memory_row(
    *,
    timestamp: str,
    state: str,
    activation_confidence: float,
    counts: Dict[str, int],
    rationale: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "activation_state": state,
        "eligible_count": counts["eligible"],
        "limited_count": counts["limited"],
        "deferred_count": counts["deferred"],
        "rejected_count": counts["rejected"],
        "operator_required_count": counts["operator_required"],
        "activation_confidence": round(activation_confidence, 6),
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
        for c in ACTIVATION_MEMORY_COLUMNS:
            r.setdefault(c, None)
    return out


# -----------------------------------------------------------
# Orchestration
# -----------------------------------------------------------
def build_doctrine_activation(
    *,
    ratification_summary: Dict[str, Any],
    ratification_record: Dict[str, Any],
    ratification_mem: List[Dict[str, str]],
    court_summary: Dict[str, Any],
    constitution_summary: Dict[str, Any],
    council_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
    readiness_summary: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    scorecard: Dict[str, Any],
    existing_activation_mem: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    timestamp = _now_iso_utc()

    ctx = _extract_context(
        ratification_summary=ratification_summary,
        ratification_record=ratification_record,
        ratification_mem=ratification_mem,
        court_summary=court_summary,
        constitution_summary=constitution_summary,
        council_summary=council_summary,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
    )

    pre_restricted = (
        ctx["constitutional_pressure"] >= 0.55
        or ctx["constitution_violated"]
        or ctx["court_ruling"] == "COURT_OVERRULED"
    )
    evaluated = _evaluate_all(ctx, restricted=pre_restricted)
    activation_confidence = _activation_confidence(ctx, evaluated)

    state, reasons, restricted = _classify_activation_state(
        ctx=ctx,
        activation_confidence=activation_confidence,
        evaluated=evaluated,
    )

    if restricted != pre_restricted:
        evaluated = _evaluate_all(ctx, restricted=restricted)
        activation_confidence = _activation_confidence(ctx, evaluated)

    counts = _count_decisions(evaluated)
    booleans = _activation_booleans(state, evaluated, ctx)
    recommendations = _recommendations(evaluated, state)
    rationale = _build_rationale(evaluated, state)

    mem_row = _build_memory_row(
        timestamp=timestamp,
        state=state,
        activation_confidence=activation_confidence,
        counts=counts,
        rationale=rationale,
    )
    merged_memory = _merge_memory(existing_activation_mem, mem_row)

    md = _render_markdown(
        generated_at=timestamp,
        state=state,
        activation_confidence=activation_confidence,
        evaluated=evaluated,
        booleans=booleans,
        reasons=reasons,
        rationale=rationale,
        recommendations=recommendations,
        counts=counts,
    )

    record: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_activation_board_engine",
        "engine_version": 1,
        "activation_state": state,
        "activation_confidence": activation_confidence,
        "activation_reasons": reasons,
        "activated_doctrines": evaluated,
        "decision_counts": counts,
        "governance_booleans": booleans,
        "recommendations": recommendations,
        "rationale": rationale,
        "ratified_vs_activated_note": (
            "Ratified ≠ Activated. This board determines activation eligibility only. "
            "Activation-eligible ≠ Runtime-mutated. No runtime policy is changed here."
        ),
        "constitutional_supremacy_note": (
            "The activation board evaluates eligibility only. It NEVER mutates runtime policy. "
            "Activation cannot override the constitution, constitutional court, "
            "capital preservation doctrine, or operator supremacy."
        ),
        "activation_memory_size_after_append": len(merged_memory),
        "safety": {
            "broker_calls_made": False,
            "orders_placed": False,
            "portfolio_mutated": False,
            "runtime_mutation_allowed": False,
            "activation_board_only": True,
        },
        "inputs_seen": {
            "arm_governance_policy_ratification_summary": bool(ratification_summary),
            "arm_governance_policy_ratification_record": bool(ratification_record),
            "arm_governance_ratification_memory_rows": len(ratification_mem),
            "arm_constitutional_court_summary": bool(court_summary),
            "arm_autonomy_constitution_summary": bool(constitution_summary),
            "arm_supreme_governance_council_summary": bool(council_summary),
            "autonomous_system_health_summary": bool(health_summary),
            "autonomous_readiness_summary": bool(readiness_summary),
            "runtime_policy_governed": bool(runtime_policy),
            "autonomous_governance_scorecard": bool(scorecard),
            "existing_activation_memory_rows": len(existing_activation_mem),
            "n_ratified_doctrine_entries": len(ctx["governance_doctrine"]),
        },
    }

    summary: Dict[str, Any] = {
        "generated_at_utc": timestamp,
        "engine": "arm_governance_doctrine_activation_board_engine",
        "activation_state": state,
        "activation_confidence": activation_confidence,
        "activation_eligible_doctrines_available": booleans[
            "activation_eligible_doctrines_available"
        ],
        "limited_activation_available": booleans["limited_activation_available"],
        "operator_activation_required": booleans["operator_activation_required"],
        "constitutional_review_required": booleans["constitutional_review_required"],
        "runtime_mutation_allowed": booleans["runtime_mutation_allowed"],
        "activation_memory_reliable": booleans["activation_memory_reliable"],
        "eligible_count": counts["eligible"],
        "limited_count": counts["limited"],
        "deferred_count": counts["deferred"],
        "rejected_count": counts["rejected"],
        "operator_required_count": counts["operator_required"],
        "n_doctrines_evaluated": len(evaluated),
        "n_recommendations": len(recommendations),
        "activation_memory_size": len(merged_memory),
    }

    return record, summary, md, merged_memory


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only ARM governance doctrine activation board engine (Step 42). "
            "Evaluates ratified doctrines for activation eligibility. "
            "Never mutates runtime policy. No broker calls."
        ),
    )
    p.add_argument("--ratification-summary", default=str(DEFAULT_RATIFICATION_SUM))
    p.add_argument("--ratification-record", default=str(DEFAULT_RATIFICATION_REC))
    p.add_argument("--ratification-mem", default=str(DEFAULT_RATIFICATION_MEM))
    p.add_argument("--court-summary", default=str(DEFAULT_COURT_SUMMARY))
    p.add_argument("--constitution-summary", default=str(DEFAULT_CONSTITUTION_SUM))
    p.add_argument("--council-summary", default=str(DEFAULT_COUNCIL_SUMMARY))
    p.add_argument("--health-summary", default=str(DEFAULT_HEALTH_SUMMARY))
    p.add_argument("--readiness-summary", default=str(DEFAULT_READINESS_SUM))
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
        "[ARM_DOCTRINE_ACTIVATION] starting "
        "(read-only doctrine activation eligibility; no runtime mutation; no broker calls)",
        flush=True,
    )

    ratification_summary = _safe_read_json(
        Path(args.ratification_summary), label="arm_governance_policy_ratification_summary.json"
    )
    ratification_record = _safe_read_json(
        Path(args.ratification_record), label="arm_governance_policy_ratification.json"
    )
    ratification_mem = _safe_read_csv_rows(
        Path(args.ratification_mem), label="arm_governance_ratification_memory.csv"
    )
    court_summary = _safe_read_json(
        Path(args.court_summary), label="arm_constitutional_court_summary.json"
    )
    constitution_summary = _safe_read_json(
        Path(args.constitution_summary), label="arm_autonomy_constitution_summary.json"
    )
    council_summary = _safe_read_json(
        Path(args.council_summary), label="arm_supreme_governance_council_summary.json"
    )
    health_summary = _safe_read_json(
        Path(args.health_summary), label="autonomous_system_health_summary.json"
    )
    readiness_summary = _safe_read_json(
        Path(args.readiness_summary), label="autonomous_readiness_summary.json"
    )
    runtime_policy = _safe_read_json(
        Path(args.runtime_policy), label="runtime_policy_governed.json"
    )
    scorecard = _safe_read_json(Path(args.scorecard), label="autonomous_governance_scorecard.json")
    existing_mem = _safe_read_csv_rows(
        Path(args.out_mem_csv), label="arm_governance_doctrine_activation_memory.csv"
    )

    record, summary, md, merged_memory = build_doctrine_activation(
        ratification_summary=ratification_summary,
        ratification_record=ratification_record,
        ratification_mem=ratification_mem,
        court_summary=court_summary,
        constitution_summary=constitution_summary,
        council_summary=council_summary,
        health_summary=health_summary,
        readiness_summary=readiness_summary,
        runtime_policy=runtime_policy,
        scorecard=scorecard,
        existing_activation_mem=existing_mem,
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
        _atomic_write_csv(merged_memory, Path(args.out_mem_csv), columns=ACTIVATION_MEMORY_COLUMNS)
    except Exception as e:
        _warn(f"failed to write {args.out_mem_csv}: {type(e).__name__}: {e}")
        return 2
    parquet_ok = _atomic_write_parquet(merged_memory, Path(args.out_mem_parquet))

    counts = record["decision_counts"]
    print(
        "[ARM_DOCTRINE_ACTIVATION] "
        f"state={record['activation_state']} "
        f"eligible={counts['eligible']} "
        f"limited={counts['limited']} "
        f"operator={counts['operator_required']} "
        f"confidence={record['activation_confidence']:.3f}",
        flush=True,
    )
    print(
        "[ARM_DOCTRINE_ACTIVATION_SAFETY] "
        "broker_calls=0 orders_placed=0 portfolio_mutated=False runtime_mutation=False",
        flush=True,
    )
    print(
        f"[ARM_DOCTRINE_ACTIVATION_OUT] json={Path(args.out_json).as_posix()} "
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
                f"[ARM_DOCTRINE_ACTIVATION_SAFETY] forbidden broker token detected: {tok!r}"
            )


_self_check_no_broker_tokens()


if __name__ == "__main__":
    raise SystemExit(main())
