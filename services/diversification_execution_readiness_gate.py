"""
TRITON Phase 180 — Diversification Execution Readiness Gate (analysis only).

Determines whether the governance-approved diversification recovery plan can be
executed under current lifecycle, authorization, allocation, and risk constraints.

SAFETY: analysis only — no governance, lifecycle, execution, broker, allocation,
or risk control changes. Does not place trades or submit orders.

Outputs (data/results/):
    diversification_execution_readiness.json
    diversification_execution_readiness.csv

Run:
    python -m services.diversification_execution_readiness_gate
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
    PERMIT_ESCALATION,
    _open_new_tickers,
    _simulate_scenario,
)
from services.diversification_feasibility_analysis import (
    MIN_OPEN_NEW_CONFIDENCE,
    STRONG_EDGE_SCORE,
    _classify_feasibility,
    _lifecycle_row,
)
from services.pnl_ledger_builder import _norm_sym, _to_float
from services.sector_exposure import get_sector

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "diversification_execution_readiness.json"
OUT_CSV = RESULTS / "diversification_execution_readiness.csv"

DEFAULT_RECOVERY_PLAN = {"QQQ": 60.0, "HD": 15.0, "XLK": 25.0}
RECOVERY_FOCUS = ["QQQ", "HD", "XLK", "BITO"]


def _log(msg: str) -> None:
    print(f"[diversification_execution_readiness_gate] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prob_label(score: float) -> str:
    if score >= 0.80:
        return "VERY_HIGH"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.35:
        return "MODERATE"
    return "LOW"


def _concentration_impact(sym: str, held: Set[str]) -> str:
    if sym in held:
        return "NEUTRAL"
    sec = get_sector(sym)
    if sec != "Tech":
        return "REDUCES_CONCENTRATION"
    return "PARTIAL_DIVERSIFICATION"


def _position_limit_status(sym: str, alloc_summary: Dict[str, Any], held: Set[str]) -> str:
    if sym in held:
        return "HELD"
    slots = _to_float(alloc_summary.get("available_slots"), 0)
    if sym in set(alloc_summary.get("blocked_new_buy_symbols") or []):
        return "BLOCKED"
    if slots <= 0:
        return "NO_SLOTS"
    return "SLOT_AVAILABLE"


def _classify_readiness(
    feasibility: str,
    blockers: List[str],
    action: str,
    exec_auth: bool,
    gov_auth: bool,
    in_plan: bool,
    position_status: str,
) -> str:
    if not in_plan and sym_not_eligible(action, feasibility):
        return "NOT_ELIGIBLE"
    if position_status == "BLOCKED" or blockers:
        if any("risk" in b.lower() or "block" in b.lower() for b in blockers):
            return "BLOCKED"
    if feasibility == "IMMEDIATELY_FEASIBLE" and exec_auth and position_status == "SLOT_AVAILABLE":
        if not gov_auth:
            return "READY_WITH_CONDITIONS"
        return "READY"
    if feasibility == "FEASIBLE_WITH_TIME":
        return "READY_WITH_CONDITIONS"
    if blockers:
        return "BLOCKED"
    return "NOT_ELIGIBLE"


def sym_not_eligible(action: str, feasibility: str) -> bool:
    return action not in ("OPEN_NEW", "ADD", "WATCH") and feasibility == "CURRENTLY_UNAVAILABLE"


def _readiness_score(
    readiness_status: str,
    conf: float,
    edge: float,
    alloc_score: float,
) -> float:
    base = {
        "READY": 95.0,
        "READY_WITH_CONDITIONS": 72.0,
        "BLOCKED": 25.0,
        "NOT_ELIGIBLE": 10.0,
    }.get(readiness_status, 40.0)
    conf_b = min(conf / MIN_OPEN_NEW_CONFIDENCE, 1.0) * 10 if np.isfinite(conf) else 0
    edge_b = min(edge / STRONG_EDGE_SCORE, 1.0) * 5 if np.isfinite(edge) else 0
    alloc_b = min(alloc_score, 1.0) * 5 if np.isfinite(alloc_score) else 0
    return float(np.clip(base + conf_b + edge_b + alloc_b, 0, 100))


def _evaluate_candidate(
    sym: str,
    recommendations: pd.DataFrame,
    lifecycle: pd.DataFrame,
    alloc_summary: Dict[str, Any],
    auth: Dict[str, Any],
    held: Set[str],
    in_recovery_plan: bool,
    plan_weight_pct: Optional[float],
) -> Dict[str, Any]:
    sym = _norm_sym(sym)
    rec_row = recommendations[recommendations["ticker"].map(_norm_sym) == sym]
    rec = rec_row.iloc[0].to_dict() if not rec_row.empty else {"ticker": sym}
    lc = _lifecycle_row(lifecycle, sym)
    blocked = set(alloc_summary.get("blocked_new_buy_symbols") or [])

    feasibility, blockers = _classify_feasibility(rec, lc, blocked, sym in RECOVERY_FOCUS)
    action = str(rec.get("recommended_action") or "—").upper()
    lc_action = str(lc.get("lifecycle_action") or rec.get("lifecycle_action") or "—")
    stance = str(lc.get("effective_stance") or rec.get("effective_stance") or "—")
    conf = _to_float(rec.get("confidence") or lc.get("confidence"), np.nan)
    edge = _to_float(rec.get("edge_score"), np.nan)
    alloc_score = _to_float(rec.get("allocation_score"), np.nan)
    risk = str(rec.get("risk_flag") or "OK")

    sys_exec = bool(auth.get("execution_authorized"))
    sys_gov = bool(auth.get("governance_authorized"))
    exec_blocked = lc.get("execution_blocked") is True

    position_status = _position_limit_status(sym, alloc_summary, held)
    cand_exec = (
        sys_exec
        and not exec_blocked
        and sym not in blocked
        and action in ("OPEN_NEW", "ADD")
        and feasibility == "IMMEDIATELY_FEASIBLE"
        and position_status == "SLOT_AVAILABLE"
    )

    readiness_status = _classify_readiness(
        feasibility,
        blockers,
        action,
        cand_exec,
        sys_gov,
        in_recovery_plan or action == "OPEN_NEW",
        position_status,
    )

    score = _readiness_score(readiness_status, conf, edge, alloc_score)

    return {
        "ticker": sym,
        "in_recovery_plan": in_recovery_plan,
        "plan_target_weight_pct": _f(plan_weight_pct, 2),
        "lifecycle_status": lc_action,
        "effective_stance": stance,
        "allocation_status": action,
        "allocation_score": _f(alloc_score, 4),
        "confidence": _f(conf, 4),
        "edge_score": _f(edge, 4),
        "risk_flags": risk,
        "execution_authorized": cand_exec,
        "governance_authorized": sys_gov,
        "system_execution_authorized": sys_exec,
        "position_limit_status": position_status,
        "concentration_impact": _concentration_impact(sym, held),
        "feasibility_status": feasibility,
        "readiness_status": readiness_status,
        "readiness_score": _f(score, 1),
        "blockers": "; ".join(blockers) if blockers else "",
        "sector": get_sector(sym),
    }


def _identify_bottlenecks(
    candidates: List[Dict[str, Any]],
    auth: Dict[str, Any],
    can_yellow: bool,
    can_green: bool,
) -> Dict[str, Any]:
    primary = "NO_BOTTLENECK"
    secondary = "NO_BOTTLENECK"

    if not bool(auth.get("governance_authorized")):
        primary = "AUTHORIZATION_BOTTLENECK"
    elif not can_yellow:
        blocked_c = [c for c in candidates if c.get("readiness_status") == "BLOCKED"]
        if blocked_c:
            primary = "ALLOCATION_BOTTLENECK"
        else:
            primary = "LIFECYCLE_BOTTLENECK"

    bito = next((c for c in candidates if c.get("ticker") == "BITO"), None)
    if not can_green and bito:
        if secondary == "NO_BOTTLENECK":
            secondary = "PROMOTION_BOTTLENECK"
        if bito.get("allocation_status") == "WATCH":
            if primary == "NO_BOTTLENECK":
                primary = "PROMOTION_BOTTLENECK"

    risk_blocked = [c for c in candidates if "BLOCK" in str(c.get("risk_flags", "")).upper()]
    if risk_blocked and primary == "NO_BOTTLENECK":
        primary = "RISK_BOTTLENECK"

    if not can_yellow and primary == "AUTHORIZATION_BOTTLENECK":
        secondary = "CONCENTRATION_BOTTLENECK"

    return {
        "primary_bottleneck": primary,
        "secondary_bottleneck": secondary,
        "governance_gate_reason": (auth.get("gate_reasons") or {}).get("governance"),
        "details": (
            "Governance ORANGE blocks authorization while concentration paradoxically "
            "requires diversification trades that would reduce concentration."
            if primary == "AUTHORIZATION_BOTTLENECK"
            else None
        ),
    }


def _promotion_dependencies(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    non_ready = [
        c
        for c in candidates
        if c.get("readiness_status") not in ("READY",) and c.get("ticker") != "QQQ"
    ]
    closest = None
    best_gap = 999.0
    for c in non_ready:
        conf = _to_float(c.get("confidence"), 0)
        gap = max(0, MIN_OPEN_NEW_CONFIDENCE - conf)
        if gap < best_gap:
            best_gap = gap
            closest = c

    bito = next((c for c in candidates if c.get("ticker") == "BITO"), {})
    bito_conf = _to_float(bito.get("confidence"), 0)
    bito_gap = max(0, MIN_OPEN_NEW_CONFIDENCE - bito_conf)

    green_blocker = "BITO" if bito.get("readiness_status") != "READY" else None

    events: List[str] = []
    if bito_gap > 0:
        events.append(f"BITO WATCH → OPEN_NEW (confidence gap {bito_gap:.3f})")
    for c in candidates:
        if c.get("allocation_status") == "WATCH" and c.get("ticker") != "BITO":
            events.append(f"{c['ticker']} promotion to OPEN_NEW")

    transitions = []
    if bito.get("lifecycle_status") == "BUY" and bito.get("allocation_status") == "WATCH":
        transitions.append("BITO: allocation WATCH → OPEN_NEW when confidence >= 0.50")
    transitions.append("Execute HD+XLK diversification to clear ORANGE → YELLOW governance")

    return {
        "closest_to_promotion": {
            "ticker": closest.get("ticker") if closest else "—",
            "confidence": closest.get("confidence") if closest else None,
            "confidence_gap_to_open_new": _f(best_gap, 4) if closest else None,
            "current_status": closest.get("readiness_status") if closest else None,
        },
        "green_recovery_blocker": green_blocker,
        "bito_eligibility": {
            "confidence": _f(bito_conf, 4),
            "confidence_gap": _f(bito_gap, 4),
            "allocation_status": bito.get("allocation_status"),
            "promotion_required": bito_gap > 0,
        },
        "promotion_events_that_would_help": events,
        "required_lifecycle_transitions": transitions,
    }


def _overall_classification(
    can_yellow: bool,
    can_green: bool,
    candidates: List[Dict[str, Any]],
) -> str:
    plan_adds = [c for c in candidates if c.get("in_recovery_plan") and c.get("ticker") != "QQQ"]
    ready = sum(1 for c in plan_adds if c.get("readiness_status") == "READY")
    cond = sum(1 for c in plan_adds if c.get("readiness_status") == "READY_WITH_CONDITIONS")

    if can_green and ready == len(plan_adds):
        return "FULLY_READY"
    if can_yellow and (ready + cond) >= len(plan_adds) and plan_adds:
        return "READY"
    if cond >= 1 or ready >= 1:
        return "PARTIALLY_READY"
    return "NOT_READY"


def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()

    thresholds = _read_json(RESULTS / "governance_recovery_thresholds.json")
    simulation = _read_json(RESULTS / "concentration_recovery_simulation.json")
    monitor = _read_json(RESULTS / "recovery_basket_monitor.json")
    feasibility_doc = _read_json(RESULTS / "diversification_feasibility_analysis.json")
    auth = _read_json(RESULTS / "governance_authorization.json")
    cpe = _read_json(RESULTS / "capital_preservation_escalation.json")
    cpi = _read_json(RESULTS / "capital_preservation_intelligence.json")
    watchdog = _read_json(RESULTS / "watchdog_status.json")
    alloc_summary = _read_json(RESULTS / "portfolio_allocation_summary.json")
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    lifecycle = _read_csv(RESULTS / "signal_lifecycle_effective.csv")

    positions = _load_positions()
    held = {_norm_sym(p["symbol"]) for p in positions}
    total_mv = sum(p["market_value"] for p in positions)

    yellow_plan = (
        (thresholds.get("yellow_threshold") or {}).get("minimum_yellow_recovery_plan")
        or (thresholds.get("executive_report") or {}).get("minimum_yellow_recovery_plan")
        or {}
    )
    plan_weights = dict(yellow_plan.get("weights") or DEFAULT_RECOVERY_PLAN)
    green_plan = (thresholds.get("green_threshold") or {}).get("minimum_green_recovery_plan") or {}

    base_components = dict(cpi.get("component_scores") or {})
    for k in CPI_WEIGHTS:
        base_components.setdefault(k, 70)
    current_cps = int(
        cpi.get("capital_preservation_score") or _estimate_cps(positions, base_components)[0]
    )
    current_esc = str(cpe.get("escalation_state") or _escalation_from_cps(current_cps))
    current_gov = "AUTHORIZED" if auth.get("governance_authorized") else "BLOCKED"

    open_new = _open_new_tickers(recommendations)
    eval_syms = list(dict.fromkeys(RECOVERY_FOCUS + open_new))

    candidate_rows: List[Dict[str, Any]] = []
    for sym in eval_syms:
        in_plan = sym in plan_weights and (plan_weights.get(sym) or 0) > 0
        weight = plan_weights.get(sym) if in_plan else None
        candidate_rows.append(
            _evaluate_candidate(
                sym, recommendations, lifecycle, alloc_summary, auth, held, in_plan, weight
            )
        )

    hd = next((c for c in candidate_rows if c["ticker"] == "HD"), {})
    xlk = next((c for c in candidate_rows if c["ticker"] == "XLK"), {})
    bito = next((c for c in candidate_rows if c["ticker"] == "BITO"), {})

    can_hd = bool(hd.get("execution_authorized")) or hd.get("readiness_status") in (
        "READY",
        "READY_WITH_CONDITIONS",
    )
    can_xlk = bool(xlk.get("execution_authorized")) or xlk.get("readiness_status") in (
        "READY",
        "READY_WITH_CONDITIONS",
    )
    can_bito = bool(bito.get("execution_authorized")) and bito.get("readiness_status") == "READY"

    plan_adds_ready = (
        hd.get("readiness_status") in ("READY", "READY_WITH_CONDITIONS")
        and xlk.get("readiness_status") in ("READY", "READY_WITH_CONDITIONS")
        and bool(auth.get("execution_authorized"))
    )
    can_yellow = plan_adds_ready
    can_green = can_yellow and can_bito and bool(green_plan.get("weights"))

    sim_yellow = _simulate_scenario(plan_weights, total_mv, base_components, 100.0, current_cps)

    lifecycle_feas = np.mean(
        [
            (
                1.0
                if c.get("feasibility_status") == "IMMEDIATELY_FEASIBLE"
                else 0.5 if c.get("feasibility_status") == "FEASIBLE_WITH_TIME" else 0.0
            )
            for c in candidate_rows
            if c.get("ticker") in ("HD", "XLK", "BITO")
        ]
    )
    alloc_feas = np.mean(
        [
            (
                1.0
                if c.get("allocation_status") == "OPEN_NEW"
                else 0.4 if c.get("allocation_status") == "WATCH" else 0.0
            )
            for c in candidate_rows
            if c.get("ticker") in ("HD", "XLK", "BITO")
        ]
    )
    auth_feas = 1.0 if auth.get("execution_authorized") else 0.0
    if not auth.get("governance_authorized"):
        auth_feas *= 0.5
    risk_feas = (
        1.0
        if str(watchdog.get("watchdog_status", "")).upper() in ("OK", "WARN", "WARNING")
        else 0.3
    )

    exec_prob = float(
        np.clip(
            0.30 * alloc_feas + 0.25 * lifecycle_feas + 0.25 * auth_feas + 0.20 * risk_feas, 0, 1
        )
    )
    gov_rec_prob = (
        _to_float((monitor.get("executive_report") or {}).get("recovery_probability_score"), 50)
        / 100.0
    )
    if not np.isfinite(gov_rec_prob):
        gov_rec_prob = exec_prob

    recovery_prob = float(np.clip(0.5 * exec_prob + 0.5 * gov_rec_prob, 0, 1))

    bottlenecks = _identify_bottlenecks(candidate_rows, auth, can_yellow, can_green)
    promotions = _promotion_dependencies(candidate_rows)

    avg_readiness = float(np.mean([_to_float(c.get("readiness_score"), 0) for c in candidate_rows]))
    execution_score = float(exec_prob * 100)
    recovery_score = float(sim_yellow.get("projected_cps") or current_cps)
    governance_score = float(100 if current_gov == "AUTHORIZED" else 35)

    overall = _overall_classification(can_yellow, can_green, candidate_rows)

    required_actions: List[str] = []
    if not auth.get("governance_authorized"):
        required_actions.append(
            "Execute diversification (HD+XLK) to reduce concentration and lift escalation ORANGE → YELLOW"
        )
    if not can_bito:
        required_actions.append(
            f"Await BITO OPEN_NEW promotion (confidence gap {promotions['bito_eligibility'].get('confidence_gap')})"
        )
    if can_yellow and not auth.get("governance_authorized"):
        required_actions.append(
            "Paper execution layer permits orders; governance gate clears post-diversification (paradox resolution)"
        )

    execution_validation = {
        "can_hd_execute_now": can_hd,
        "can_xlk_execute_now": can_xlk,
        "can_bito_execute_now": can_bito,
        "can_yellow_recovery_plan_execute_now": can_yellow,
        "can_green_recovery_plan_execute_now": can_green,
        "yellow_plan_executable_at_allocation_layer": plan_adds_ready,
        "yellow_plan_governance_cleared_after_execution": sim_yellow.get("governance_status")
        == "AUTHORIZED",
        "projected_post_execution_cps": sim_yellow.get("projected_cps"),
        "projected_post_execution_escalation": sim_yellow.get("escalation_level"),
    }

    readiness_scores = {
        "readiness_score": _f(avg_readiness, 1),
        "execution_score": _f(execution_score, 1),
        "recovery_score": _f(recovery_score, 1),
        "governance_score": _f(governance_score, 1),
        "overall_readiness_classification": overall,
    }

    recovery_plan = {
        "name": "MINIMUM_YELLOW_RECOVERY_PLAN",
        "weights": plan_weights,
        "target_escalation": "YELLOW",
        "projected_cps": sim_yellow.get("projected_cps"),
        "projected_governance_status": sim_yellow.get("governance_status"),
        "green_plan_available": bool(green_plan.get("weights")),
        "green_plan_note": green_plan.get("note"),
    }

    current_state = {
        "generated_at": ts,
        "current_cps": current_cps,
        "current_escalation": current_esc,
        "current_governance_status": current_gov,
        "largest_concentration_pct": (simulation.get("current_state") or {}).get(
            "largest_concentration_pct"
        ),
        "execution_authorized_system": bool(auth.get("execution_authorized")),
        "governance_authorized_system": bool(auth.get("governance_authorized")),
        "open_new_candidates": open_new,
        "watchdog_status": watchdog.get("watchdog_status"),
    }

    executive = {
        "report_type": "DIVERSIFICATION_EXECUTION_READINESS_REPORT",
        "current_recovery_plan": plan_weights,
        "current_governance_state": current_gov,
        "current_escalation": current_esc,
        "execution_readiness": overall,
        "recovery_readiness": (monitor.get("executive_report") or {}).get(
            "estimated_recovery_readiness"
        ),
        "primary_bottleneck": bottlenecks.get("primary_bottleneck"),
        "secondary_bottleneck": bottlenecks.get("secondary_bottleneck"),
        "execution_probability": _prob_label(exec_prob),
        "recovery_probability": _prob_label(recovery_prob),
        "required_actions": required_actions,
        "expected_governance_outcome": (
            f"Post-plan: CPS {sim_yellow.get('projected_cps')} → "
            f"{sim_yellow.get('escalation_level')} ({sim_yellow.get('governance_status')})"
        ),
        "recommended_action": (
            "Proceed with HD+XLK diversification at allocation layer; "
            "governance authorization expected after concentration reduction"
            if can_yellow
            else "Resolve blockers before execution"
        ),
        "readiness_classification": overall,
        "governance_impact": "ORANGE → YELLOW upon successful plan execution",
    }

    probability = {
        "execution_probability": _prob_label(exec_prob),
        "execution_probability_score": _f(exec_prob * 100, 1),
        "governance_recovery_probability": _prob_label(gov_rec_prob),
        "governance_recovery_probability_score": _f(gov_rec_prob * 100, 1),
        "combined_recovery_probability": _prob_label(recovery_prob),
        "allocation_feasibility": _f(alloc_feas * 100, 1),
        "lifecycle_feasibility": _f(lifecycle_feas * 100, 1),
        "authorization_feasibility": _f(auth_feas * 100, 1),
        "risk_feasibility": _f(risk_feas * 100, 1),
    }

    summary = {
        "candidates_evaluated": len(candidate_rows),
        "yellow_executable": can_yellow,
        "green_executable": can_green,
        "overall_readiness": overall,
        "primary_bottleneck": bottlenecks.get("primary_bottleneck"),
        "hd_readiness": hd.get("readiness_status"),
        "xlk_readiness": xlk.get("readiness_status"),
        "bito_readiness": bito.get("readiness_status"),
    }

    pd.DataFrame(candidate_rows).to_csv(OUT_CSV, index=False, encoding="utf-8")

    report = {
        "generated_at": ts,
        "executive_report": executive,
        "current_state": current_state,
        "recovery_plan": recovery_plan,
        "candidate_readiness": candidate_rows,
        "execution_validation": execution_validation,
        "bottleneck_analysis": bottlenecks,
        "promotion_dependencies": promotions,
        "readiness_scores": readiness_scores,
        "execution_probability": probability,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "analysis-only — no trades, orders, governance, lifecycle, or authorization modified",
        },
        "disclaimer": (
            "Execution readiness evaluation only — does not place trades, submit orders, or "
            "modify governance, lifecycle, broker, authorization, or allocation rules."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_CSV.name}")
    _log(
        f"overall={overall} yellow={can_yellow} green={can_green} bottleneck={bottlenecks.get('primary_bottleneck')}"
    )
    return report


if __name__ == "__main__":
    run()
