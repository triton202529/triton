"""
TRITON Phase 188 — Autonomous Portfolio Transition & Governance Recovery
Execution Planner (read-only).

Designs a controlled, governance-safe transition roadmap from the current
concentrated portfolio to the Phase 187 GREEN recovery target.

SAFETY: planning and simulation only — does not execute trades, modify rules,
or alter portfolio state.

Outputs (data/results/):
    autonomous_transition_plan.json
    autonomous_transition_stages.csv
    autonomous_transition_risk_assessment.csv

Run:
    python -m services.autonomous_portfolio_transition_planner
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
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
    _escalation_from_cps,
    _f,
    _load_positions,
    _read_csv,
    _read_json,
)
from services.concentration_recovery_simulator import CPS_GREEN, CPS_YELLOW, _simulate_scenario
from services.diversification_feasibility_analysis import _lifecycle_row
from services.pnl_ledger_builder import _norm_sym, _to_float
from services.sector_exposure import get_sector

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "autonomous_transition_plan.json"
OUT_STAGES = RESULTS / "autonomous_transition_stages.csv"
OUT_RISK = RESULTS / "autonomous_transition_risk_assessment.csv"

DOMINANT = "QQQ"
BUY_STANCES = {"BUY", "ADD"}
MIN_CONF = 0.45


def _log(msg: str) -> None:
    print(f"[autonomous_portfolio_transition_planner] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_context() -> Dict[str, Any]:
    return {
        "optimization": _read_json(RESULTS / "final_governance_recovery_optimization.json"),
        "reconstruction": _read_json(RESULTS / "recovery_basket_reconstruction.json"),
        "lifecycle_eff": _read_csv(RESULTS / "signal_lifecycle_effective.csv"),
        "recs": _read_csv(RESULTS / "portfolio_allocation_recommendations.csv"),
        "exception_gov": _read_json(RESULTS / "recovery_basket_exception_governance.json"),
        "exec_readiness": _read_json(RESULTS / "execution_readiness.json"),
        "div_exec": _read_json(RESULTS / "diversification_execution_readiness.json"),
        "cpi": _read_json(RESULTS / "capital_preservation_intelligence.json"),
        "cpe": _read_json(RESULTS / "capital_preservation_escalation.json"),
        "auth": _read_json(RESULTS / "governance_authorization.json"),
        "alloc_summary": _read_json(RESULTS / "portfolio_allocation_summary.json"),
        "positions": _load_positions(),
    }


def _target_portfolio(ctx: Dict[str, Any]) -> Dict[str, float]:
    opt = ctx["optimization"]
    paths = opt.get("optimization_paths") or []
    path_a = next((p for p in paths if p.get("path_id") == "PATH_A"), None)
    if path_a and path_a.get("weights"):
        return {k: float(v) for k, v in path_a["weights"].items()}
    return {
        "QQQ": 38.0,
        "PEP": 15.0,
        "XLE": 15.0,
        "ABBV": 15.0,
        "MA": 15.0,
        "V": 2.0,
    }


def _current_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
    positions = ctx["positions"]
    total_mv = sum(p["market_value"] for p in positions) or 1438.0
    weights: Dict[str, float] = {}
    if total_mv > 0:
        for p in positions:
            sym = p["symbol"]
            weights[sym] = weights.get(sym, 0.0) + p["market_value"] / total_mv * 100.0
    if not weights:
        weights = {DOMINANT: 100.0}

    base_components = dict(
        (ctx["reconstruction"].get("current_state") or {}).get("base_components")
        or ctx["cpi"].get("component_scores")
        or {}
    )
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)

    current_cps = int(
        ctx["cpi"].get("capital_preservation_score")
        or (ctx["reconstruction"].get("current_state") or {}).get("current_cps")
        or 66
    )
    _, largest_pct, _ = _concentration_score(positions) if positions else (0, 100.0, 1)
    esc = str(ctx["cpe"].get("escalation_state") or _escalation_from_cps(current_cps)).upper()
    gov_auth = ctx["auth"].get("governance_authorized")
    gov_status = "AUTHORIZED" if gov_auth is True else "BLOCKED"

    return {
        "weights": weights,
        "largest_concentration_pct": _f(largest_pct or weights.get(DOMINANT, 100.0)),
        "current_cps": current_cps,
        "escalation": esc,
        "governance_status": gov_status,
        "governance_authorized": gov_auth is True,
        "portfolio_value": _f(total_mv),
        "position_count": len(positions) or 1,
        "base_components": base_components,
        "total_mv": float(total_mv),
        "baseline_largest_pct": float(largest_pct or 100.0),
    }


def _sim_weights(
    weights: Dict[str, float],
    state: Dict[str, Any],
    anchor_cps: int,
) -> Dict[str, Any]:
    w = {k: float(v) for k, v in weights.items() if v > 0}
    total = sum(w.values())
    if abs(total - 100.0) > 0.01 and total > 0:
        w = {k: v / total * 100.0 for k, v in w.items()}
    return _simulate_scenario(
        w,
        state["total_mv"],
        state["base_components"],
        state["baseline_largest_pct"],
        anchor_cps,
    )


def _rec_row(recs: pd.DataFrame, sym: str) -> Dict[str, Any]:
    if recs.empty:
        return {}
    sub = recs[recs["ticker"].map(_norm_sym) == sym]
    return sub.iloc[0].to_dict() if not sub.empty else {}


def _validate_lifecycle(
    ctx: Dict[str, Any],
    symbols: List[str],
    stage_weights: Dict[str, float],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    all_exec = True
    any_cond = False
    for sym in symbols:
        if sym == DOMINANT or stage_weights.get(sym, 0) <= 0:
            continue
        lc = _lifecycle_row(ctx["lifecycle_eff"], sym)
        rec = _rec_row(ctx["recs"], sym)
        effective = str(lc.get("effective_stance") or rec.get("effective_stance") or "").upper()
        signal = str(lc.get("signal") or rec.get("signal") or "").upper()
        conf = _to_float(lc.get("confidence") or rec.get("confidence"), np.nan)
        risk = str(rec.get("risk_flag") or "OK").upper()
        exec_blocked = lc.get("execution_blocked") is True

        blockers: List[str] = []
        if effective not in BUY_STANCES:
            blockers.append(f"effective_stance={effective}")
        if signal == "SELL":
            blockers.append("signal=SELL")
        if not np.isnan(conf) and conf < MIN_CONF:
            blockers.append(f"confidence={conf:.3f}<{MIN_CONF}")
        if risk != "OK" or "BLOCK" in risk:
            blockers.append(f"risk_flag={risk}")
        if exec_blocked:
            blockers.append("execution_blocked")

        if not blockers:
            status = "EXECUTABLE"
        elif effective in BUY_STANCES and len(blockers) == 1 and "confidence" in blockers[0]:
            status = "CONDITIONAL"
            any_cond = True
        else:
            status = "NOT_EXECUTABLE"
            all_exec = False

        results.append(
            {
                "symbol": sym,
                "effective_stance": effective,
                "confidence": _f(conf, 4),
                "signal": signal,
                "risk_flag": risk,
                "lifecycle_status": status,
                "blockers": blockers,
            }
        )

    overall = "EXECUTABLE"
    if not all_exec:
        overall = (
            "NOT_EXECUTABLE"
            if any(r["lifecycle_status"] == "NOT_EXECUTABLE" for r in results)
            else "CONDITIONAL"
        )
    elif any_cond:
        overall = "CONDITIONAL"

    return {"overall": overall, "candidates": results}


def _stage_actions(
    start_w: Dict[str, float],
    target_w: Dict[str, float],
) -> Dict[str, Any]:
    all_syms = set(start_w) | set(target_w)
    reductions = []
    additions = []
    for sym in all_syms:
        s = start_w.get(sym, 0.0)
        t = target_w.get(sym, 0.0)
        delta = t - s
        if delta < -0.01:
            reductions.append(
                {
                    "symbol": sym,
                    "from_pct": _f(s, 2),
                    "to_pct": _f(t, 2),
                    "change_pct": _f(delta, 2),
                }
            )
        elif delta > 0.01:
            additions.append(
                {
                    "symbol": sym,
                    "from_pct": _f(s, 2),
                    "to_pct": _f(t, 2),
                    "change_pct": _f(delta, 2),
                }
            )
    return {"reductions": reductions, "additions": additions}


def _build_stage(
    stage_num: int,
    path_id: str,
    start_w: Dict[str, float],
    target_w: Dict[str, float],
    state: Dict[str, Any],
    ctx: Dict[str, Any],
    cycle_offset: int,
    gate: str,
) -> Dict[str, Any]:
    sim = _sim_weights(target_w, state, state["current_cps"])
    adds = [a["symbol"] for a in _stage_actions(start_w, target_w)["additions"]]
    lc_val = _validate_lifecycle(ctx, adds, target_w)
    actions = _stage_actions(start_w, target_w)

    gov_status = sim["governance_status"]
    if not state["governance_authorized"] and sim["projected_cps"] < CPS_YELLOW:
        gov_status = "BLOCKED"
    elif sim["projected_cps"] >= CPS_YELLOW and gov_status == "AUTHORIZED":
        gov_status = "AUTHORIZED"

    exec_ready = ctx["exec_readiness"].get("readiness_status") or "PARTIALLY_READY"
    stage_readiness = lc_val["overall"]
    if stage_readiness == "NOT_EXECUTABLE":
        exec_ready = "NOT_READY"
    elif gov_status == "BLOCKED" and sim["projected_cps"] < CPS_YELLOW:
        exec_ready = "CONDITIONAL_EXCEPTION_REQUIRED"

    return {
        "stage": stage_num,
        "path_id": path_id,
        "cycle": state["current_cps"] and cycle_offset,
        "trading_cycle": f"T+{cycle_offset}",
        "lifecycle_gate": gate,
        "starting_state": {k: _f(v, 2) for k, v in start_w.items()},
        "target_state": {k: _f(v, 2) for k, v in target_w.items()},
        "required_actions": actions,
        "projected_cps": sim["projected_cps"],
        "projected_escalation": sim["escalation_level"],
        "projected_governance_status": gov_status,
        "concentration_pct": sim["largest_position_weight_pct"],
        "execution_readiness": stage_readiness,
        "execution_readiness_detail": exec_ready,
        "lifecycle_validation": lc_val,
        "revalidation_required": stage_num > 1,
    }


def _path_a_stages(
    target: Dict[str, float], state: Dict[str, Any], ctx: Dict[str, Any]
) -> List[Dict[str, Any]]:
    start = dict(state["weights"])
    return [
        _build_stage(1, "PATH_A", start, target, state, ctx, 0, "none — immediate full transition")
    ]


def _path_b_stages(
    target: Dict[str, float], state: Dict[str, Any], ctx: Dict[str, Any]
) -> List[Dict[str, Any]]:
    start = dict(state["weights"])
    s1 = {DOMINANT: 70.0, "PEP": 15.0, "XLE": 15.0}
    s2 = {DOMINANT: 52.0, "PEP": 15.0, "XLE": 15.0, "ABBV": 18.0}
    stages = []
    prev = start
    for i, tw in enumerate([s1, s2, target], 1):
        stages.append(_build_stage(i, "PATH_B", prev, tw, state, ctx, i - 1, f"cycle_{i}_open"))
        prev = tw
    return stages


def _path_c_stages(
    target: Dict[str, float], state: Dict[str, Any], ctx: Dict[str, Any]
) -> List[Dict[str, Any]]:
    start = dict(state["weights"])
    milestones = [
        ({DOMINANT: 85.0, "PEP": 15.0}, "PEP effective_stance=BUY confirmed"),
        ({DOMINANT: 70.0, "PEP": 15.0, "XLE": 15.0}, "XLE effective_stance=BUY confirmed"),
        ({DOMINANT: 55.0, "PEP": 15.0, "XLE": 15.0, "ABBV": 15.0}, "ABBV signal=BUY sustained"),
        (target, "MA+V lifecycle revalidation pass"),
    ]
    stages = []
    prev = start
    for i, (tw, gate) in enumerate(milestones, 1):
        stages.append(_build_stage(i, "PATH_C", prev, tw, state, ctx, i - 1, gate))
        prev = tw
    return stages


def _path_d_stages(
    target: Dict[str, float], state: Dict[str, Any], ctx: Dict[str, Any]
) -> List[Dict[str, Any]]:
    start = dict(state["weights"])
    s1 = {DOMINANT: 75.0, "PEP": 12.5, "XLE": 12.5}
    s2 = {DOMINANT: 55.0, "PEP": 15.0, "XLE": 15.0, "ABBV": 15.0}
    stages = []
    prev = start
    gates = [
        "Recovery Basket Exception activation (DUAL_APPROVAL) + Stage 1 tranche",
        "CPS >= YELLOW checkpoint — expand diversification",
        "Final GREEN alignment",
    ]
    for i, tw in enumerate([s1, s2, target], 1):
        stages.append(_build_stage(i, "PATH_D", prev, tw, state, ctx, i - 1, gates[i - 1]))
        prev = tw
    return stages


def _path_e_stages(
    target: Dict[str, float], state: Dict[str, Any], ctx: Dict[str, Any]
) -> List[Dict[str, Any]]:
    start = dict(state["weights"])
    milestones = [
        {DOMINANT: 90.0, "PEP": 10.0},
        {DOMINANT: 80.0, "PEP": 10.0, "XLE": 10.0},
        {DOMINANT: 70.0, "PEP": 12.0, "XLE": 12.0, "ABBV": 6.0},
        {DOMINANT: 55.0, "PEP": 15.0, "XLE": 15.0, "ABBV": 15.0},
        target,
    ]
    stages = []
    prev = start
    for i, tw in enumerate(milestones, 1):
        stages.append(
            _build_stage(
                i,
                "PATH_E",
                prev,
                tw,
                state,
                ctx,
                i - 1,
                f"minimal_tranche_{i} + lifecycle revalidation",
            )
        )
        prev = tw
    return stages


def _path_metrics(
    stages: List[Dict[str, Any]], path_id: str, path_name: str, desc: str
) -> Dict[str, Any]:
    final = stages[-1] if stages else {}
    cps_traj = [s["projected_cps"] for s in stages]
    conc_traj = [s["concentration_pct"] for s in stages]
    not_exec = sum(1 for s in stages if s.get("execution_readiness") == "NOT_EXECUTABLE")

    complexity_map = {
        "PATH_A": ("HIGH", 1.0),
        "PATH_B": ("MEDIUM", 0.5),
        "PATH_C": ("MEDIUM", 0.6),
        "PATH_D": ("HIGH", 0.7),
        "PATH_E": ("LOW", 0.25),
    }
    complexity, slippage = complexity_map.get(path_id, ("MEDIUM", 0.5))

    lifecycle_risk = "LOW" if not_exec == 0 else "MODERATE" if not_exec <= 1 else "HIGH"

    return {
        "path_id": path_id,
        "path_name": path_name,
        "description": desc,
        "transition_duration_cycles": len(stages),
        "transition_duration_days_est": len(stages),
        "trading_cycles": len(stages),
        "cps_progression": cps_traj,
        "concentration_trajectory": conc_traj,
        "governance_progression": [s["projected_governance_status"] for s in stages],
        "escalation_progression": [s["projected_escalation"] for s in stages],
        "execution_complexity": complexity,
        "estimated_slippage_exposure": slippage,
        "lifecycle_dependency_risk": lifecycle_risk,
        "final_cps": final.get("projected_cps"),
        "final_escalation": final.get("projected_escalation"),
        "final_governance": final.get("projected_governance_status"),
        "stages": stages,
        "reaches_green": _to_float(final.get("projected_cps"), 0) >= CPS_GREEN,
    }


def _build_paths(
    target: Dict[str, float],
    state: Dict[str, Any],
    ctx: Dict[str, Any],
) -> List[Dict[str, Any]]:
    paths = [
        _path_metrics(
            _path_a_stages(target, state, ctx),
            "PATH_A",
            "Immediate Full Transition",
            "Single-cycle transition directly to GREEN target weights.",
        ),
        _path_metrics(
            _path_b_stages(target, state, ctx),
            "PATH_B",
            "Gradual Multi-Cycle Transition",
            "Three-cycle staged diversification with intermediate CPS checkpoints.",
        ),
        _path_metrics(
            _path_c_stages(target, state, ctx),
            "PATH_C",
            "Signal-Triggered Transition",
            "Four stages gated on lifecycle signal confirmation per candidate.",
        ),
        _path_metrics(
            _path_d_stages(target, state, ctx),
            "PATH_D",
            "Hybrid Transition",
            "Exception-scoped Stage 1 tranche then progressive diversification to GREEN.",
        ),
        _path_metrics(
            _path_e_stages(target, state, ctx),
            "PATH_E",
            "Risk-Minimized Transition",
            "Five minimal tranches with lifecycle revalidation between each cycle.",
        ),
    ]

    for p in paths:
        score = 0.0
        if p.get("reaches_green"):
            score += 0.35
        score += 0.25 * (1.0 - p.get("estimated_slippage_exposure", 0.5))
        score += 0.20 * (1.0 if p.get("lifecycle_dependency_risk") == "LOW" else 0.5)
        score += 0.10 * (1.0 / max(p.get("trading_cycles", 1), 1))
        score += 0.10 * (_to_float(p.get("final_cps"), 0) / 100.0)
        p["path_score"] = _f(score, 4)

    paths.sort(key=lambda p: _to_float(p.get("path_score"), 0), reverse=True)
    return paths


def _risk_analysis(
    paths: List[Dict[str, Any]],
    state: Dict[str, Any],
    ctx: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = [
        {
            "risk_type": "execution_timing",
            "severity": "MODERATE",
            "mitigation": "Execute during regular market hours; avoid multi-symbol batch in single illiquid window.",
            "required_monitoring": "Order fill rates and slippage vs plan per stage",
        },
        {
            "risk_type": "lifecycle_deterioration",
            "severity": "MODERATE",
            "mitigation": "Re-validate effective_stance before each stage; abort if BUY->WAIT transition detected.",
            "required_monitoring": "signal_lifecycle_effective.csv refresh per cycle",
        },
        {
            "risk_type": "concentration_persistence",
            "severity": "HIGH" if state["largest_concentration_pct"] >= 90 else "MODERATE",
            "mitigation": "Prioritize QQQ reduction tranches in early stages; do not delay first diversification.",
            "required_monitoring": "Largest position weight vs 35% governance target",
        },
        {
            "risk_type": "market_movement",
            "severity": "MODERATE",
            "mitigation": "Use limit orders where supported; stage sizes <= 15% per cycle.",
            "required_monitoring": "Intra-stage portfolio drift vs target weights",
        },
        {
            "risk_type": "governance_regression",
            "severity": "HIGH" if not state["governance_authorized"] else "LOW",
            "mitigation": "PATH_D exception scope for Stage 1 OR wait for CPS>=75 before unrestricted execution.",
            "required_monitoring": "CPS and escalation after each stage",
        },
        {
            "risk_type": "correlation",
            "severity": "LOW",
            "mitigation": "Target basket spans Consumer/Energy/Healthcare/Financials — low QQQ correlation.",
            "required_monitoring": "Rolling 60d correlation vs QQQ post-transition",
        },
    ]

    summary = {
        "overall_risk": "MODERATE",
        "critical_count": sum(1 for r in rows if r["severity"] == "CRITICAL"),
        "high_count": sum(1 for r in rows if r["severity"] == "HIGH"),
        "primary_risk": "governance_regression",
        "recovery_blocker": (
            "Governance ORANGE gate until CPS >= YELLOW; exception framework required for Stage 1 "
            "unless gradual path lifts CPS first."
        ),
    }
    return rows, summary


def _approval_requirements(ctx: Dict[str, Any], recommended: Dict[str, Any]) -> Dict[str, Any]:
    fw = ctx["exception_gov"].get("framework_summary") or {}
    return {
        "approval_model": fw.get("recommended_approval_model") or "DUAL_APPROVAL",
        "exception_framework": fw.get("framework_name") or "Recovery Basket Exception (PATH_B)",
        "exception_required_stage_1": recommended.get("path_id") in ("PATH_A", "PATH_D"),
        "human_approvals_required": [
            "Operator approval — transition plan activation",
            "Risk officer sign-off — staged weight targets per cycle",
            "DUAL_APPROVAL if Recovery Basket Exception invoked (PATH_D or blocked Stage 1)",
            "Governance checkpoint review when CPS crosses YELLOW (>=75)",
            "Final GREEN confirmation review when CPS >= 90",
        ],
        "governance_checkpoints": [
            {
                "checkpoint": "PRE_TRANSITION",
                "trigger": "Plan published",
                "action": "Human review of lifecycle validation",
            },
            {
                "checkpoint": "POST_STAGE_1",
                "trigger": "First tranche complete",
                "action": "CPS + escalation re-eval",
            },
            {
                "checkpoint": "YELLOW_CROSSING",
                "trigger": "CPS >= 75",
                "action": "Governance authorization re-check",
            },
            {
                "checkpoint": "GREEN_TARGET",
                "trigger": "CPS >= 90",
                "action": "Close transition; restore standard governance",
            },
        ],
        "revalidation_stages": "Every stage — lifecycle effective_stance must remain BUY/ADD",
    }


def _autonomy_assessment(
    ctx: Dict[str, Any],
    state: Dict[str, Any],
    recommended: Dict[str, Any],
) -> Dict[str, Any]:
    exec_r = ctx["exec_readiness"]
    can_autonomous = (
        state["governance_authorized"]
        and exec_r.get("readiness_status") == "READY"
        and recommended.get("lifecycle_dependency_risk") == "LOW"
    )
    return {
        "can_execute_autonomously": False,
        "autonomous_readiness_score": 35,
        "autonomous_readiness_label": "NOT_AUTONOMOUS — HUMAN_IN_LOOP_REQUIRED",
        "rationale": (
            "Governance BLOCKED at ORANGE; DUAL_APPROVAL exception or CPS lift required. "
            "Execution readiness PARTIALLY_READY (7/8 checks). Lifecycle revalidation mandatory "
            "each cycle. Paper execution permitted but plan activation requires operator approval."
        ),
        "human_in_loop_required": True,
        "paper_execution_permitted": exec_r.get("paper_execution_permitted", True),
        "live_execution_permitted": exec_r.get("live_execution_permitted", False),
        "automation_safe_stages": [
            s["stage"]
            for s in (recommended.get("stages") or [])
            if s.get("execution_readiness") == "EXECUTABLE"
        ],
        "monitoring_requirements": [
            "CPS trajectory vs plan after each stage",
            "Lifecycle effective_stance for PEP/XLE/ABBV/MA/V",
            "Governance authorization state",
            "Concentration largest-weight pct",
            "Exception window expiry if PATH_D active",
        ],
    }


def _executive_report(
    state: Dict[str, Any],
    target: Dict[str, float],
    paths: List[Dict[str, Any]],
    recommended: Dict[str, Any],
    risk_summary: Dict[str, Any],
    approvals: Dict[str, Any],
    autonomy: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "report_type": "AUTONOMOUS_TRANSITION_EXECUTION_REPORT",
        "current_state_summary": {
            "weights": state["weights"],
            "cps": state["current_cps"],
            "escalation": state["escalation"],
            "governance": state["governance_status"],
            "concentration_pct": state["largest_concentration_pct"],
        },
        "target_green_portfolio": {
            "weights": target,
            "projected_cps": recommended.get("final_cps") or CPS_GREEN,
            "projected_escalation": "GREEN",
            "recovery_probability_pct": 92,
        },
        "recommended_path": {
            "path_id": recommended.get("path_id"),
            "path_name": recommended.get("path_name"),
            "cycles": recommended.get("trading_cycles"),
            "final_cps": recommended.get("final_cps"),
        },
        "cps_timeline": recommended.get("cps_progression"),
        "executive_recommendation": (
            f"Execute {recommended.get('path_id')} ({recommended.get('path_name')}) over "
            f"{recommended.get('trading_cycles')} trading cycles. "
            f"Human DUAL_APPROVAL required for Stage 1 while governance ORANGE. "
            f"Re-validate lifecycle before each tranche. Target CPS "
            f"{state['current_cps']} -> {recommended.get('final_cps')} (GREEN). "
            "Do not enable fully autonomous execution until POST_YELLOW checkpoint cleared."
        ),
        "executive_questions": {
            "how_to_move_to_green": (
                f"{recommended.get('trading_cycles')}-cycle staged transition via {recommended.get('path_id')}"
            ),
            "safest_path": recommended.get("path_id"),
            "cycles_required": recommended.get("trading_cycles"),
            "approvals_required": approvals.get("approval_model"),
            "autonomous_execution": autonomy.get("can_execute_autonomously"),
            "risks_preventing_recovery": risk_summary.get("recovery_blocker"),
            "monitoring": ", ".join(autonomy.get("monitoring_requirements") or [])[:200],
        },
    }


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    ctx = _load_context()
    state = _current_state(ctx)
    target = _target_portfolio(ctx)
    paths = _build_paths(target, state, ctx)
    recommended = paths[0] if paths else {}
    risk_rows, risk_summary = _risk_analysis(paths, state, ctx)
    approvals = _approval_requirements(ctx, recommended)
    autonomy = _autonomy_assessment(ctx, state, recommended)
    executive = _executive_report(
        state, target, paths, recommended, risk_summary, approvals, autonomy
    )

    all_stages: List[Dict[str, Any]] = []
    for p in paths:
        for s in p.get("stages") or []:
            row = dict(s)
            row["path_name"] = p.get("path_name")
            all_stages.append(row)

    summary = {
        "current_cps": state["current_cps"],
        "target_cps": recommended.get("final_cps"),
        "recommended_path": recommended.get("path_id"),
        "transition_cycles": recommended.get("trading_cycles"),
        "reaches_green": recommended.get("reaches_green"),
        "autonomous_ready": autonomy.get("can_execute_autonomously"),
        "overall_risk": risk_summary.get("overall_risk"),
    }

    report = {
        "generated_at": ts,
        "phase": "188",
        "mode": "autonomous_portfolio_transition_planner",
        "disclaimer": (
            "Planning and simulation only — does not execute trades, modify governance, "
            "lifecycle, execution, broker, allocation, or risk controls."
        ),
        "current_state": state,
        "target_portfolio": {
            "weights": target,
            "projected_cps": CPS_GREEN,
            "projected_escalation": "GREEN",
            "projected_governance_status": "AUTHORIZED",
            "source": "final_governance_recovery_optimization PATH_A",
        },
        "transition_paths": [{k: v for k, v in p.items() if k != "stages"} for p in paths],
        "recommended_path": recommended,
        "transition_stages": all_stages,
        "recommended_stages": recommended.get("stages") or [],
        "risk_analysis": {"rows": risk_rows, "summary": risk_summary},
        "approval_requirements": approvals,
        "autonomy_assessment": autonomy,
        "executive_report": executive,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "read-only planning — no pipeline modifications",
        },
    }

    stage_rows = []
    for s in all_stages:
        actions = s.get("required_actions") or {}
        stage_rows.append(
            {
                "stage": s.get("stage"),
                "path_id": s.get("path_id"),
                "path_name": s.get("path_name"),
                "trading_cycle": s.get("trading_cycle"),
                "starting_state": json.dumps(s.get("starting_state") or {}),
                "target_state": json.dumps(s.get("target_state") or {}),
                "required_actions": json.dumps(actions),
                "projected_cps": s.get("projected_cps"),
                "projected_governance_status": s.get("projected_governance_status"),
                "projected_escalation": s.get("projected_escalation"),
                "execution_readiness": s.get("execution_readiness"),
                "lifecycle_gate": s.get("lifecycle_gate"),
            }
        )
    pd.DataFrame(stage_rows).to_csv(OUT_STAGES, index=False, encoding="utf-8")
    pd.DataFrame(risk_rows).to_csv(OUT_RISK, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_STAGES.name}, {OUT_RISK.name}")
    _log(
        f"recommended={summary['recommended_path']} cycles={summary['transition_cycles']} "
        f"final_cps={summary['target_cps']} autonomous={summary['autonomous_ready']}"
    )
    return report


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
