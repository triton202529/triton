"""
TRITON Phase 203 — Governance Recovery Action Readiness & Execution Feasibility.

Validates whether Phase 202 recovery acceleration scenarios are executable under
current market, lifecycle, portfolio, and governance conditions.

Read-only — does NOT modify governance, execution, trading, allocation, or risk.

Outputs (data/results/):
    governance_recovery_execution_feasibility.json
    recovery_action_readiness.csv
    recovery_execution_dependencies.csv
    recovery_execution_constraints.csv

Run:
    python -m services.governance_recovery_execution_feasibility
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.concentration_governance_analysis import (
    CPI_WEIGHTS,
    _concentration_score,
    _estimate_cps,
    _escalation_from_cps,
    _f,
    _load_positions,
    _read_csv,
    _read_json,
)
from services.concentration_recovery_simulator import (
    CPS_GREEN,
    CPS_YELLOW,
    _simulate_scenario,
)
from services.diversification_feasibility_analysis import (
    MIN_OPEN_NEW_CONFIDENCE,
    _classify_feasibility,
    _lifecycle_row,
)
from services.pnl_ledger_builder import _norm_sym, _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "governance_recovery_execution_feasibility.json"
OUT_ACTIONS = RESULTS / "recovery_action_readiness.csv"
OUT_DEPS = RESULTS / "recovery_execution_dependencies.csv"
OUT_CONSTRAINTS = RESULTS / "recovery_execution_constraints.csv"

RECOVERY_FOCUS = ["QQQ", "HD", "XLK", "BITO"]
PHASE202_SCENARIOS = [
    {
        "name": "QQQ 60% / HD 15% / XLK 25% (Balanced Exit)",
        "weights": {"QQQ": 60.0, "HD": 15.0, "XLK": 25.0},
        "rank": "MOST_REALISTIC_PATH",
        "days_to_yellow": 12,
    },
    {
        "name": "QQQ 60% / HD 20% / XLK 20%",
        "weights": {"QQQ": 60.0, "HD": 20.0, "XLK": 20.0},
        "rank": "FASTEST_PATH",
        "days_to_yellow": 8,
    },
    {
        "name": "Dynamic Recovery Basket (HD+XLK+BITO)",
        "weights": {"QQQ": 25.0, "HD": 25.0, "XLK": 25.0, "BITO": 25.0},
        "rank": "FASTEST_PATH",
        "days_to_yellow": 15,
        "requires_bito": True,
    },
    {
        "name": "CPS +9 via concentration only (simulated)",
        "weights": {"QQQ": 60.0, "HD": 15.0, "XLK": 25.0},
        "rank": "HIGHEST_CONFIDENCE_PATH",
        "days_to_yellow": 5,
    },
]


def _log(msg: str) -> None:
    print(f"[governance_recovery_execution_feasibility] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _date_offset(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")


def _readiness_band(score: float) -> str:
    if score >= 90:
        return "READY_TO_RECOVER"
    if score >= 75:
        return "HIGH_READINESS"
    if score >= 50:
        return "LIMITED_READINESS"
    if score >= 25:
        return "LOW_READINESS"
    return "RECOVERY_NOT_FEASIBLE"


def _map_readiness_status(
    readiness_status: str,
    feasibility: str,
    action_type: str,
    sym: str,
    in_held: bool,
    exec_auth: bool,
    gov_auth: bool,
) -> str:
    """Map Phase 180 readiness / feasibility to Phase 203 execution eligibility."""
    if action_type == "trim" and sym == "QQQ" and in_held:
        return "IMMEDIATELY_EXECUTABLE" if exec_auth else "EXECUTABLE_AFTER_RECOVERY"
    rs = (readiness_status or "").upper()
    if rs == "READY":
        return "IMMEDIATELY_EXECUTABLE"
    if rs == "READY_WITH_CONDITIONS":
        return "IMMEDIATELY_EXECUTABLE"
    if feasibility == "IMMEDIATELY_FEASIBLE" and exec_auth:
        return "IMMEDIATELY_EXECUTABLE"
    if feasibility == "FEASIBLE_WITH_TIME":
        return "EXECUTABLE_AFTER_RECOVERY"
    if rs == "BLOCKED":
        return "BLOCKED_BY_LIFECYCLE"
    if not gov_auth and feasibility != "CURRENTLY_UNAVAILABLE":
        return "EXECUTABLE_AFTER_RECOVERY"
    return "BLOCKED_BY_LIFECYCLE"


def _exec_snapshot_map(exec_readiness: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        _norm_sym(c.get("ticker", "")): c
        for c in (exec_readiness.get("candidate_readiness") or [])
        if c.get("ticker")
    }


def _evaluate_recovery_action(
    sym: str,
    action_type: str,
    action_label: str,
    recommendations: pd.DataFrame,
    lifecycle: pd.DataFrame,
    alloc_summary: Dict[str, Any],
    auth: Dict[str, Any],
    held: Set[str],
    plan_weight: Optional[float],
    base_components: Dict[str, int],
    total_mv: float,
    baseline_largest: float,
    current_cps: int,
    exec_snap: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sym = _norm_sym(sym)
    rec_row = recommendations[recommendations["ticker"].map(_norm_sym) == sym]
    rec = rec_row.iloc[0].to_dict() if not rec_row.empty else {"ticker": sym}
    lc = _lifecycle_row(lifecycle, sym)
    blocked = set(alloc_summary.get("blocked_new_buy_symbols") or [])

    feasibility, blockers = _classify_feasibility(rec, lc, blocked, sym in RECOVERY_FOCUS)
    lc_action = str(lc.get("lifecycle_action") or rec.get("lifecycle_action") or "—")
    stance = str(lc.get("effective_stance") or rec.get("effective_stance") or "—")
    allocation_action = str(rec.get("recommended_action") or "—").upper()
    conf = _to_float(rec.get("confidence") or lc.get("confidence"), np.nan)
    edge = _to_float(rec.get("edge_score"), np.nan)
    alloc_score = _to_float(rec.get("allocation_score"), np.nan)
    sys_exec = bool(auth.get("execution_authorized"))
    sys_gov = bool(auth.get("governance_authorized"))
    exec_blocked = lc.get("execution_blocked") is True
    data_source = "live"
    lifecycle_drift = False

    if exec_snap:
        data_source = "execution_readiness_gate"
        snap_feas = exec_snap.get("feasibility_status")
        if snap_feas:
            feasibility = snap_feas
        snap_lc = exec_snap.get("lifecycle_status")
        snap_stance = exec_snap.get("effective_stance")
        snap_alloc = exec_snap.get("allocation_status")
        if snap_lc and snap_lc != "—":
            if snap_lc != lc_action:
                lifecycle_drift = True
            lc_action = snap_lc
        if snap_stance and snap_stance != "—":
            if snap_stance != stance:
                lifecycle_drift = True
            stance = snap_stance
        if snap_alloc and snap_alloc != "—":
            if snap_alloc != allocation_action:
                lifecycle_drift = True
            allocation_action = snap_alloc
        conf = _to_float(exec_snap.get("confidence"), conf)
        edge = _to_float(exec_snap.get("edge_score"), edge)
        alloc_score = _to_float(exec_snap.get("allocation_score"), alloc_score)
        snap_blockers = exec_snap.get("blockers")
        if snap_blockers and not lifecycle_drift:
            blockers = [b.strip() for b in str(snap_blockers).split(";") if b.strip()]
        elif lifecycle_drift:
            blockers = list(blockers) + ["live lifecycle drifted from execution readiness snapshot"]

    if action_type == "trim" and sym == "QQQ":
        blockers = [b for b in blockers if "buy-ready" not in b.lower()]
        if sym in held:
            feasibility = "IMMEDIATELY_FEASIBLE"

    classification = _map_readiness_status(
        (exec_snap or {}).get("readiness_status", ""),
        feasibility,
        action_type,
        sym,
        sym in held,
        sys_exec and not exec_blocked,
        sys_gov,
    )

    # Expected CPS impact for concentration actions
    cps_delta = 0.0
    if action_type == "add" and plan_weight and plan_weight > 0:
        weights = {"QQQ": max(0, 100 - plan_weight)}
        weights[sym] = plan_weight
        sim = _simulate_scenario(weights, total_mv, base_components, baseline_largest, current_cps)
        cps_delta = float(sim.get("cps_improvement") or 0)
    elif action_type == "trim" and sym == "QQQ":
        target_qqq = plan_weight or 60.0
        trim_pct = max(0, baseline_largest - target_qqq)
        weights = {"QQQ": target_qqq, "HD": trim_pct * 0.375, "XLK": trim_pct * 0.625}
        sim = _simulate_scenario(weights, total_mv, base_components, baseline_largest, current_cps)
        cps_delta = float(sim.get("cps_improvement") or 0)

    div_contrib = 0.0
    if sym not in held and plan_weight:
        div_contrib = plan_weight
    elif sym == "QQQ" and action_type == "trim":
        div_contrib = max(0, baseline_largest - (plan_weight or 60))

    gov_dep = "NONE"
    if classification == "IMMEDIATELY_EXECUTABLE" and not sys_gov:
        gov_dep = "Governance clears post-concentration reduction (ORANGE→YELLOW)"
    elif classification == "BLOCKED_BY_GOVERNANCE":
        gov_dep = "Requires governance authorization before execution"
    elif classification == "BLOCKED_BY_LIFECYCLE":
        gov_dep = f"Requires {allocation_action}→OPEN_NEW promotion"

    return {
        "action": action_label,
        "symbol": sym,
        "action_type": action_type,
        "lifecycle_status": lc_action,
        "effective_stance": stance,
        "allocation_status": allocation_action,
        "confidence_score": _f(conf, 4),
        "edge_score": _f(edge, 4),
        "allocation_score": _f(alloc_score, 4),
        "execution_eligibility": classification,
        "governance_dependency": gov_dep,
        "expected_cps_impact": _f(cps_delta, 1),
        "diversification_contribution_pct": _f(div_contrib, 1),
        "blockers": "; ".join(blockers) if blockers else "",
        "feasibility_status": feasibility,
        "execution_authorized_system": sys_exec,
        "governance_authorized_system": sys_gov,
        "data_source": data_source,
        "lifecycle_drift_detected": lifecycle_drift,
    }


def _build_actions(
    recommendations: pd.DataFrame,
    lifecycle: pd.DataFrame,
    alloc_summary: Dict[str, Any],
    auth: Dict[str, Any],
    held: Set[str],
    plan_weights: Dict[str, float],
    base_components: Dict[str, int],
    total_mv: float,
    baseline_largest: float,
    current_cps: int,
    exec_readiness: Dict[str, Any],
) -> List[Dict[str, Any]]:
    snap_map = _exec_snapshot_map(exec_readiness)
    actions: List[Dict[str, Any]] = []

    actions.append(
        _evaluate_recovery_action(
            "QQQ",
            "trim",
            "Reduce QQQ concentration toward 60%",
            recommendations,
            lifecycle,
            alloc_summary,
            auth,
            held,
            plan_weights.get("QQQ", 60.0),
            base_components,
            total_mv,
            baseline_largest,
            current_cps,
            snap_map.get("QQQ"),
        )
    )

    for sym in ("HD", "XLK", "BITO"):
        actions.append(
            _evaluate_recovery_action(
                sym,
                "add",
                f"Add {sym} to recovery basket",
                recommendations,
                lifecycle,
                alloc_summary,
                auth,
                held,
                plan_weights.get(sym),
                base_components,
                total_mv,
                baseline_largest,
                current_cps,
                snap_map.get(sym),
            )
        )

    open_new = recommendations[
        recommendations["recommended_action"].astype(str).str.upper() == "OPEN_NEW"
    ]
    for sym in open_new["ticker"].map(_norm_sym).unique():
        if sym in RECOVERY_FOCUS:
            continue
        actions.append(
            _evaluate_recovery_action(
                sym,
                "add",
                f"BUY-effective candidate: {sym}",
                recommendations,
                lifecycle,
                alloc_summary,
                auth,
                held,
                None,
                base_components,
                total_mv,
                baseline_largest,
                current_cps,
                snap_map.get(sym),
            )
        )

    order = {
        "IMMEDIATELY_EXECUTABLE": 0,
        "EXECUTABLE_AFTER_RECOVERY": 1,
        "BLOCKED_BY_LIFECYCLE": 2,
        "BLOCKED_BY_GOVERNANCE": 3,
        "NOT_A_VALID_RECOVERY_ACTION": 4,
    }
    actions.sort(
        key=lambda a: (
            order.get(a["execution_eligibility"], 9),
            -(a.get("expected_cps_impact") or 0),
        )
    )
    for i, a in enumerate(actions, 1):
        a["rank"] = i
    return actions


def _validate_scenario(
    scenario: Dict[str, Any],
    actions: List[Dict[str, Any]],
    bito_ok: bool,
) -> Dict[str, Any]:
    weights = scenario["weights"]
    required = [s for s in weights if s != "QQQ" and weights[s] > 0]
    action_map = {a["symbol"]: a for a in actions}

    blockers: List[str] = []
    executable: List[str] = []
    for sym in required:
        act = action_map.get(sym)
        if not act:
            blockers.append(f"{sym}: not evaluated")
            continue
        elig = act["execution_eligibility"]
        if elig in ("IMMEDIATELY_EXECUTABLE", "EXECUTABLE_AFTER_RECOVERY"):
            executable.append(sym)
        elif elig == "BLOCKED_BY_LIFECYCLE":
            blockers.append(f"{sym}: lifecycle blocked")
        else:
            blockers.append(f"{sym}: {elig}")

    if scenario.get("requires_bito") and not bito_ok:
        blockers.append("BITO: not OPEN_NEW — GREEN path blocked")

    qqq_act = action_map.get("QQQ", {})
    qqq_ok = qqq_act.get("execution_eligibility") in (
        "IMMEDIATELY_EXECUTABLE",
        "EXECUTABLE_AFTER_RECOVERY",
    )

    feasible = len(blockers) == 0 and qqq_ok
    partial = len(executable) >= 2 and qqq_ok and not scenario.get("requires_bito")

    if feasible:
        status = "EXECUTABLE"
    elif partial:
        status = "PARTIALLY_EXECUTABLE"
    else:
        status = "NOT_EXECUTABLE"

    return {
        "scenario_name": scenario["name"],
        "rank_classification": scenario["rank"],
        "target_weights": weights,
        "execution_status": status,
        "executable_components": executable,
        "blocking_factors": blockers,
        "phase202_days_to_yellow": scenario["days_to_yellow"],
        "realistic_days_to_yellow": scenario["days_to_yellow"] if feasible else None,
        "qqq_trim_required": True,
        "qqq_trim_feasible": qqq_ok,
    }


def _recovery_sequences(
    actions: List[Dict[str, Any]],
    exit_strategy: Dict[str, Any],
    scenario_validations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    hd = next((a for a in actions if a["symbol"] == "HD"), {})
    xlk = next((a for a in actions if a["symbol"] == "XLK"), {})
    qqq = next((a for a in actions if a["symbol"] == "QQQ"), {})
    bito = next((a for a in actions if a["symbol"] == "BITO"), {})

    fastest_yellow = next(
        (s for s in scenario_validations if s["execution_status"] == "EXECUTABLE"), {}
    )
    if not fastest_yellow:
        fastest_yellow = next(
            (s for s in scenario_validations if s["execution_status"] == "PARTIALLY_EXECUTABLE"),
            scenario_validations[0] if scenario_validations else {},
        )

    balanced = next(
        (s for s in scenario_validations if "Balanced" in s.get("scenario_name", "")),
        {},
    )
    conservative_days = ((exit_strategy.get("timeline") or {}).get("conservative") or {}).get(
        "estimated_days_to_yellow"
    ) or 28

    return {
        "FASTEST_SEQUENCE": {
            "steps": [
                {
                    "step": 1,
                    "action": "Trim QQQ by 20% (100%→80%)",
                    "feasibility": qqq.get("execution_eligibility"),
                    "rationale": "Fastest concentration reduction lever",
                },
                {
                    "step": 2,
                    "action": "Deploy HD + XLK at OPEN_NEW weights",
                    "feasibility": f"HD={hd.get('execution_eligibility')}, XLK={xlk.get('execution_eligibility')}",
                    "rationale": "Both candidates lifecycle-ready at allocation layer",
                },
                {
                    "step": 3,
                    "action": "Continue trim to 60% QQQ target",
                    "feasibility": qqq.get("execution_eligibility"),
                    "rationale": "Reach YELLOW CPS threshold",
                },
            ],
            "target_scenario": fastest_yellow.get("scenario_name"),
            "estimated_days_to_yellow": fastest_yellow.get("realistic_days_to_yellow"),
            "projected_yellow_date": _date_offset(
                int(fastest_yellow.get("realistic_days_to_yellow") or 12)
            ),
        },
        "LOWEST_RISK_SEQUENCE": {
            "steps": [
                {
                    "step": 1,
                    "action": "Conservative 5% QQQ trim per 7-day cadence",
                    "feasibility": qqq.get("execution_eligibility"),
                    "rationale": "Minimize step disruption per concentration exit plan",
                },
                {
                    "step": 2,
                    "action": "Add HD when trim creates slot",
                    "feasibility": hd.get("execution_eligibility"),
                    "rationale": "Non-Tech sector diversification first",
                },
                {
                    "step": 3,
                    "action": "Add XLK at subsequent step",
                    "feasibility": xlk.get("execution_eligibility"),
                    "rationale": "Complete YELLOW recovery basket",
                },
                {
                    "step": 4,
                    "action": "Await BITO OPEN_NEW before GREEN extension",
                    "feasibility": bito.get("execution_eligibility"),
                    "rationale": "GREEN requires four-name equal weight",
                },
            ],
            "target_scenario": "Conservative 7-day steps",
            "estimated_days_to_yellow": conservative_days,
            "projected_yellow_date": _date_offset(int(conservative_days)),
        },
        "HIGHEST_CONFIDENCE_SEQUENCE": {
            "steps": [
                {
                    "step": 1,
                    "action": "Execute balanced exit: QQQ 60% / HD 15% / XLK 25%",
                    "feasibility": balanced.get("execution_status", "PARTIALLY_EXECUTABLE"),
                    "rationale": "Aligns Phase 179/180/181 threshold plan with Phase 202 realistic path",
                },
                {
                    "step": 2,
                    "action": "Document concentration reduction evidence",
                    "feasibility": "IMMEDIATELY_EXECUTABLE",
                    "rationale": "Governance evidence for ORANGE→YELLOW transition",
                },
                {
                    "step": 3,
                    "action": "Monitor CPS ≥75 for 14 days",
                    "feasibility": "EXECUTABLE_AFTER_RECOVERY",
                    "rationale": "Stabilization evidence before GREEN pursuit",
                },
            ],
            "target_scenario": balanced.get("scenario_name"),
            "estimated_days_to_yellow": balanced.get("realistic_days_to_yellow"),
            "projected_yellow_date": _date_offset(
                int(balanced.get("realistic_days_to_yellow") or 12)
            ),
        },
    }


def _paradox_recheck(
    paradox: Dict[str, Any],
    exec_readiness: Dict[str, Any],
    actions: List[Dict[str, Any]],
    auth: Dict[str, Any],
) -> Dict[str, Any]:
    hd = next((a for a in actions if a["symbol"] == "HD"), {})
    xlk = next((a for a in actions if a["symbol"] == "XLK"), {})
    can_yellow = (exec_readiness.get("execution_validation") or {}).get(
        "can_yellow_recovery_plan_execute_now"
    )
    exec_val = exec_readiness.get("execution_validation") or {}

    paradox_detected = paradox.get("paradox_detected", False)
    hd_exec = hd.get("execution_eligibility") == "IMMEDIATELY_EXECUTABLE"
    xlk_exec = xlk.get("execution_eligibility") == "IMMEDIATELY_EXECUTABLE"

    if (can_yellow or (hd_exec and xlk_exec)) and hd_exec and xlk_exec:
        status = "PARTIALLY_RESOLVED"
        explanation = (
            "Governance still blocks authorization (ORANGE), but HD/XLK diversification "
            "is executable at the allocation/paper layer. Authorization is expected to clear "
            "after concentration reduction — the paradox is walkable, not circular."
        )
        changed = [
            "Phase 180 confirmed HD/XLK IMMEDIATELY_FEASIBLE at allocation layer",
            "Execution layer permits orders while governance withholds authorization",
            "Recovery path defined: diversify first → CPS rises → governance clears",
        ]
    elif paradox_detected:
        status = "UNRESOLVED"
        explanation = paradox.get("paradox_summary", "Paradox persists — diversification blocked.")
        changed = []
    else:
        status = "RESOLVED"
        explanation = "Governance and diversification constraints are aligned."
        changed = []

    remaining = []
    if not auth.get("governance_authorized"):
        remaining.append("Governance ORANGE blocks authorization until CPS ≥75")
    if xlk.get("diversification_contribution_pct") and xlk.get("sector") != "Consumer":
        remaining.append("XLK adds Tech exposure — HD required for sector diversification")
    bito = next((a for a in actions if a["symbol"] == "BITO"), {})
    if bito.get("execution_eligibility") == "BLOCKED_BY_LIFECYCLE":
        remaining.append("BITO WATCH blocks GREEN recovery path")

    return {
        "status": status,
        "prior_paradox_detected": paradox_detected,
        "explanation": explanation,
        "what_changed_since_prior_phases": changed,
        "remaining_constraints": remaining,
        "recovery_execution_realistically_possible": can_yellow is True,
        "hd_executable": hd.get("execution_eligibility") == "IMMEDIATELY_EXECUTABLE",
        "xlk_executable": xlk.get("execution_eligibility") == "IMMEDIATELY_EXECUTABLE",
    }


def _failure_modes(
    actions: List[Dict[str, Any]],
    base: Dict[str, Any],
    daily: Dict[str, Any],
) -> List[Dict[str, Any]]:
    hd = next((a for a in actions if a["symbol"] == "HD"), {})
    xlk = next((a for a in actions if a["symbol"] == "XLK"), {})
    bito = next((a for a in actions if a["symbol"] == "BITO"), {})

    momentum = (daily.get("governance_recovery_score") or {}).get("classification") or "STALLED"

    modes = [
        {
            "risk": "HD/XLK lifecycle deterioration",
            "probability": 0.15,
            "severity": "HIGH",
            "impact_on_yellow_days": 14,
            "impact_on_green_days": 21,
            "mitigation_evidence": "Monitor lifecycle_action=BUY and confidence ≥0.50 daily",
            "current_status": f"HD={hd.get('lifecycle_status')}, XLK={xlk.get('lifecycle_status')}",
        },
        {
            "risk": "BITO remains WATCH",
            "probability": 0.45,
            "severity": "MODERATE",
            "impact_on_yellow_days": 0,
            "impact_on_green_days": 30,
            "mitigation_evidence": "Track confidence gap (need ≥0.50 for OPEN_NEW)",
            "current_status": f"BITO confidence={bito.get('confidence_score')}, status=WATCH",
        },
        {
            "risk": "No BUY-effective replacements if HD/XLK degrade",
            "probability": 0.20,
            "severity": "HIGH",
            "impact_on_yellow_days": 20,
            "impact_on_green_days": 40,
            "mitigation_evidence": "Maintain OPEN_NEW candidate pipeline monitoring",
            "current_status": "HD/XLK currently OPEN_NEW",
        },
        {
            "risk": "Concentration persistence at 100%",
            "probability": 0.55 if momentum == "STALLED" else 0.25,
            "severity": "CRITICAL",
            "impact_on_yellow_days": 20,
            "impact_on_green_days": 60,
            "mitigation_evidence": "Execute QQQ trim + HD/XLK deployment per exit strategy",
            "current_status": f"QQQ={base.get('qqq_concentration_pct')}%",
        },
        {
            "risk": "Recovery evidence stagnation",
            "probability": 0.50 if momentum == "STALLED" else 0.20,
            "severity": "HIGH",
            "impact_on_yellow_days": 15,
            "impact_on_green_days": 30,
            "mitigation_evidence": "Daily evidence monitor — document CPS/concentration deltas",
            "current_status": f"Recovery score momentum={momentum}",
        },
    ]
    return modes


def _execution_dependencies(
    actions: List[Dict[str, Any]],
    exec_readiness: Dict[str, Any],
    promotion: Dict[str, Any],
) -> List[Dict[str, Any]]:
    deps: List[Dict[str, Any]] = []
    for a in actions:
        if a["symbol"] not in RECOVERY_FOCUS:
            continue
        prereqs: List[str] = []
        if a["symbol"] == "QQQ":
            prereqs = ["Operator approval for concentration reduction"]
        elif a["symbol"] in ("HD", "XLK"):
            prereqs = ["QQQ trim creates capital/slot", "Lifecycle BUY + OPEN_NEW"]
            if not a.get("governance_authorized_system"):
                prereqs.append("Governance clears post-diversification (not pre-requisite)")
        elif a["symbol"] == "BITO":
            prereqs = list(promotion.get("required_lifecycle_transitions") or [])
            prereqs.append("YELLOW achieved before GREEN extension")

        deps.append(
            {
                "symbol": a["symbol"],
                "action": a["action"],
                "execution_eligibility": a["execution_eligibility"],
                "prerequisites": "; ".join(prereqs),
                "unblocks": (
                    "YELLOW governance authorization"
                    if a["symbol"] in ("HD", "XLK", "QQQ")
                    else "GREEN recovery path"
                ),
                "dependency_complexity": (
                    "LOW"
                    if a["execution_eligibility"] == "IMMEDIATELY_EXECUTABLE"
                    else (
                        "MODERATE"
                        if a["execution_eligibility"] == "EXECUTABLE_AFTER_RECOVERY"
                        else "HIGH"
                    )
                ),
            }
        )
    return deps


def _constraints(
    auth: Dict[str, Any],
    actions: List[Dict[str, Any]],
    exec_readiness: Dict[str, Any],
    base: Dict[str, Any],
) -> List[Dict[str, Any]]:
    constraints: List[Dict[str, Any]] = []

    if not auth.get("governance_authorized"):
        constraints.append(
            {
                "constraint": "Governance ORANGE blocks authorization",
                "category": "governance",
                "severity": "CRITICAL",
                "blocks": "Immediate authorization restoration",
                "resolution_path": "Concentration reduction → CPS ≥75 → YELLOW",
                "estimated_clearance_days": int(base.get("distance_to_yellow", 9) * 1.5),
            }
        )

    bito = next((a for a in actions if a["symbol"] == "BITO"), {})
    if bito.get("execution_eligibility") == "BLOCKED_BY_LIFECYCLE":
        constraints.append(
            {
                "constraint": "BITO allocation WATCH — not OPEN_NEW",
                "category": "lifecycle",
                "severity": "HIGH",
                "blocks": "GREEN recovery path",
                "resolution_path": "Confidence ≥0.50 → OPEN_NEW promotion",
                "estimated_clearance_days": 14,
            }
        )

    if base.get("qqq_concentration_pct", 100) >= 95:
        constraints.append(
            {
                "constraint": "QQQ concentration at 100%",
                "category": "concentration",
                "severity": "CRITICAL",
                "blocks": "CPS improvement and YELLOW threshold",
                "resolution_path": "Trim QQQ + deploy HD/XLK per exit strategy",
                "estimated_clearance_days": 12,
            }
        )

    bottlenecks = exec_readiness.get("bottleneck_analysis") or {}
    if bottlenecks.get("primary_bottleneck") == "AUTHORIZATION_BOTTLENECK":
        constraints.append(
            {
                "constraint": "Authorization bottleneck (paradox)",
                "category": "governance",
                "severity": "HIGH",
                "blocks": "Pre-emptive governance clearance",
                "resolution_path": "Execute diversification — authorization follows evidence",
                "estimated_clearance_days": 8,
            }
        )

    return constraints


def _readiness_score(
    actions: List[Dict[str, Any]],
    scenario_validations: List[Dict[str, Any]],
    exec_readiness: Dict[str, Any],
    paradox_status: Dict[str, Any],
) -> Dict[str, Any]:
    core = [a for a in actions if a["symbol"] in ("HD", "XLK", "QQQ")]
    imm = sum(1 for a in core if a["execution_eligibility"] == "IMMEDIATELY_EXECUTABLE")
    cond = sum(1 for a in core if a["execution_eligibility"] == "EXECUTABLE_AFTER_RECOVERY")
    action_avail = (imm * 100 + cond * 60) / max(len(core), 1)

    confs = [_to_float(a.get("confidence_score"), 0) for a in core if a.get("confidence_score")]
    edges = [_to_float(a.get("edge_score"), 0) for a in core if a.get("edge_score")]
    cand_quality = (
        (np.mean(confs) / MIN_OPEN_NEW_CONFIDENCE * 50 + np.mean(edges) / 0.65 * 50)
        if confs and edges
        else 70.0
    )

    lifecycle_ready = (
        sum(1 for a in core if a.get("lifecycle_status") in ("BUY", "ADD"))
        / max(len(core), 1)
        * 100
    )

    exec_val = exec_readiness.get("execution_validation") or {}
    conc_feas = 85.0 if exec_val.get("can_yellow_recovery_plan_execute_now") else 55.0
    if any(a.get("execution_eligibility") == "IMMEDIATELY_EXECUTABLE" for a in core):
        conc_feas = max(conc_feas, 80.0)

    dep_complex = 100.0
    if paradox_status.get("status") == "UNRESOLVED":
        dep_complex = 30.0
    elif paradox_status.get("status") == "PARTIALLY_RESOLVED":
        dep_complex = 65.0

    executable_scenarios = sum(
        1
        for s in scenario_validations
        if s["execution_status"] in ("EXECUTABLE", "PARTIALLY_EXECUTABLE")
    )
    scenario_factor = executable_scenarios / max(len(scenario_validations), 1) * 100

    score = float(
        np.clip(
            action_avail * 0.25
            + cand_quality * 0.20
            + lifecycle_ready * 0.15
            + conc_feas * 0.20
            + dep_complex * 0.10
            + scenario_factor * 0.10,
            0,
            100,
        )
    )

    return {
        "execution_readiness_score": round(score, 1),
        "classification": _readiness_band(score),
        "components": {
            "available_recovery_actions": round(action_avail, 1),
            "candidate_quality": round(cand_quality, 1),
            "lifecycle_readiness": round(lifecycle_ready, 1),
            "concentration_reduction_feasibility": round(conc_feas, 1),
            "execution_dependency_complexity": round(dep_complex, 1),
            "scenario_executability": round(scenario_factor, 1),
        },
    }


def _executive_report(
    base: Dict[str, Any],
    readiness: Dict[str, Any],
    actions: List[Dict[str, Any]],
    sequences: Dict[str, Any],
    paradox: Dict[str, Any],
    failures: List[Dict[str, Any]],
    scenario_validations: List[Dict[str, Any]],
    constraints: List[Dict[str, Any]],
) -> Dict[str, Any]:
    executable = [a for a in actions if a["execution_eligibility"] == "IMMEDIATELY_EXECUTABLE"]
    blocked = [a for a in actions if a["execution_eligibility"].startswith("BLOCKED")]
    fastest = sequences.get("FASTEST_SEQUENCE") or {}
    safest = sequences.get("LOWEST_RISK_SEQUENCE") or {}
    confident = sequences.get("HIGHEST_CONFIDENCE_SEQUENCE") or {}

    phase202_executable = [
        s
        for s in scenario_validations
        if s["execution_status"] in ("EXECUTABLE", "PARTIALLY_EXECUTABLE")
    ]

    return {
        "report_type": "GOVERNANCE_RECOVERY_EXECUTION_FEASIBILITY_REPORT",
        "current_recovery_feasibility": {
            "cps": base["cps"],
            "qqq_concentration_pct": base["qqq_concentration_pct"],
            "governance_status": "BLOCKED" if base.get("gov_blocked") else "AUTHORIZED",
            "phase202_scenarios_executable": len(phase202_executable),
            "phase202_scenarios_total": len(scenario_validations),
        },
        "execution_readiness_score": readiness["execution_readiness_score"],
        "readiness_classification": readiness["classification"],
        "top_executable_actions": [a["action"] for a in executable[:5]],
        "top_blocking_constraints": [c["constraint"] for c in constraints[:4]],
        "recovery_sequences_summary": {
            "fastest": fastest.get("target_scenario"),
            "fastest_days": fastest.get("estimated_days_to_yellow"),
            "safest": safest.get("target_scenario"),
            "highest_confidence": confident.get("target_scenario"),
        },
        "governance_paradox_status": paradox.get("status"),
        "governance_paradox_explanation": paradox.get("explanation"),
        "failure_risks_top3": [f["risk"] for f in failures[:3]],
        "recommended_recovery_preparation": [
            "Confirm HD/XLK OPEN_NEW readiness before next rebalance window",
            "Execute QQQ trim per balanced exit plan (10% steps toward 60%)",
            "Document daily concentration + CPS evidence for governance review",
            "Monitor BITO confidence for OPEN_NEW promotion (GREEN path)",
            "Prepare operator concentration review sign-off",
        ],
        "executive_questions": {
            "can_phase202_scenarios_occur": len(phase202_executable) >= 2,
            "executable_actions_today": [a["symbol"] for a in executable],
            "blocked_actions": [a["symbol"] for a in blocked],
            "paradox_solved": paradox.get("status") in ("RESOLVED", "PARTIALLY_RESOLVED"),
            "fastest_realistic_yellow_path": fastest.get("target_scenario"),
            "fastest_realistic_yellow_days": fastest.get("estimated_days_to_yellow"),
            "recovery_failure_top_risk": failures[0]["risk"] if failures else "Unknown",
            "prepared_for_execution_when_authorized": readiness["execution_readiness_score"] >= 50,
        },
    }


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    acceleration = _read_json(RESULTS / "governance_recovery_acceleration.json")
    exec_readiness = _read_json(RESULTS / "diversification_execution_readiness.json")
    exit_strategy = _read_json(RESULTS / "concentration_exit_strategy.json")
    paradox_doc = _read_json(RESULTS / "concentration_governance_analysis.json")
    auth = _read_json(RESULTS / "governance_authorization.json")
    cpe = _read_json(RESULTS / "capital_preservation_escalation.json")
    cpi = _read_json(RESULTS / "capital_preservation_intelligence.json")
    daily = _read_json(RESULTS / "governance_daily_evidence.json")
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    lifecycle = _read_csv(RESULTS / "signal_lifecycle_effective.csv")
    alloc_summary = _read_json(RESULTS / "portfolio_allocation_summary.json")
    thresholds = _read_json(RESULTS / "governance_recovery_thresholds.json")

    positions = _load_positions()
    held = {_norm_sym(p["symbol"]) for p in positions}
    total_mv = sum(p["market_value"] for p in positions)
    _, baseline_largest, _ = _concentration_score(positions)

    base_components = dict(cpi.get("component_scores") or {})
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)
    current_cps = int(
        cpi.get("capital_preservation_score") or _estimate_cps(positions, base_components)[0]
    )

    accel_base = acceleration.get("baseline") or {}
    yellow_plan = (thresholds.get("yellow_threshold") or {}).get(
        "minimum_yellow_recovery_plan"
    ) or {}
    plan_weights = dict(yellow_plan.get("weights") or {"QQQ": 60.0, "HD": 15.0, "XLK": 25.0})

    bito_row = recommendations[recommendations["ticker"].map(_norm_sym) == "BITO"]
    bito_action = (
        str(bito_row.iloc[0]["recommended_action"]).upper() if not bito_row.empty else "WATCH"
    )
    bito_ok = bito_action == "OPEN_NEW"

    base = {
        "cps": float(accel_base.get("cps") or current_cps),
        "distance_to_yellow": float(
            accel_base.get("distance_to_yellow") or max(0, CPS_YELLOW - current_cps)
        ),
        "distance_to_green": float(
            accel_base.get("distance_to_green") or max(0, CPS_GREEN - current_cps)
        ),
        "qqq_concentration_pct": float(accel_base.get("qqq_concentration_pct") or baseline_largest),
        "recovery_score": float(accel_base.get("recovery_score") or 47.4),
        "gov_blocked": not auth.get("governance_authorized"),
        "escalation": str(cpe.get("escalation_state") or _escalation_from_cps(current_cps)),
    }

    actions = _build_actions(
        recommendations,
        lifecycle,
        alloc_summary,
        auth,
        held,
        plan_weights,
        base_components,
        total_mv,
        baseline_largest,
        current_cps,
        exec_readiness,
    )

    scenario_validations = [_validate_scenario(s, actions, bito_ok) for s in PHASE202_SCENARIOS]
    sequences = _recovery_sequences(actions, exit_strategy, scenario_validations)
    paradox = _paradox_recheck(
        paradox_doc.get("paradox_analysis") or {},
        exec_readiness,
        actions,
        auth,
    )
    failures = _failure_modes(actions, base, daily)
    deps = _execution_dependencies(
        actions,
        exec_readiness,
        exec_readiness.get("promotion_dependencies") or {},
    )
    constraints = _constraints(auth, actions, exec_readiness, base)
    readiness = _readiness_score(actions, scenario_validations, exec_readiness, paradox)
    executive = _executive_report(
        base,
        readiness,
        actions,
        sequences,
        paradox,
        failures,
        scenario_validations,
        constraints,
    )

    report = {
        "generated_at": ts,
        "phase": "203",
        "mode": "governance_recovery_execution_feasibility",
        "disclaimer": (
            "Execution feasibility intelligence only — does NOT modify governance rules, "
            "execution permissions, trading, allocation, or risk controls."
        ),
        "baseline": base,
        "execution_readiness": readiness,
        "recovery_action_readiness": actions,
        "phase202_scenario_validation": scenario_validations,
        "recovery_sequences": sequences,
        "governance_paradox_recheck": paradox,
        "failure_mode_analysis": failures,
        "execution_dependencies": deps,
        "execution_constraints": constraints,
        "executive_recommendation": executive,
        "summary": {
            "readiness_score": readiness["execution_readiness_score"],
            "readiness_classification": readiness["classification"],
            "paradox_status": paradox.get("status"),
            "executable_actions": len(
                [a for a in actions if a["execution_eligibility"] == "IMMEDIATELY_EXECUTABLE"]
            ),
            "phase202_executable_scenarios": len(
                [
                    s
                    for s in scenario_validations
                    if s["execution_status"] in ("EXECUTABLE", "PARTIALLY_EXECUTABLE")
                ]
            ),
            "fastest_realistic_yellow_days": (sequences.get("FASTEST_SEQUENCE") or {}).get(
                "estimated_days_to_yellow"
            ),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(actions).to_csv(OUT_ACTIONS, index=False, encoding="utf-8")
    pd.DataFrame(deps).to_csv(OUT_DEPS, index=False, encoding="utf-8")
    pd.DataFrame(constraints).to_csv(OUT_CONSTRAINTS, index=False, encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_ACTIONS.name}, {OUT_DEPS.name}, {OUT_CONSTRAINTS.name}")
    _log(
        f"readiness={readiness['execution_readiness_score']} ({readiness['classification']}) "
        f"paradox={paradox.get('status')} executable_scenarios="
        f"{report['summary']['phase202_executable_scenarios']}"
    )
    return report


if __name__ == "__main__":
    run()
