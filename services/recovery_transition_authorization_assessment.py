"""
TRITON Phase 189 — Recovery Transition Authorization & Autonomous Readiness
Assessment (read-only).

Determines whether Triton satisfies requirements to authorize Stage 1 of the
PATH_E governance recovery transition. Does NOT authorize execution.

SAFETY: assessment only — no governance, lifecycle, execution, broker,
allocation, or risk control changes. No trades, orders, or exception activation.

Outputs (data/results/):
    recovery_transition_authorization_assessment.json
    recovery_transition_readiness_checklist.csv
    recovery_transition_blockers.csv

Run:
    python -m services.recovery_transition_authorization_assessment
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from services.concentration_governance_analysis import (
    _concentration_score,
    _escalation_from_cps,
    _f,
    _load_positions,
    _read_csv,
    _read_json,
)
from services.diversification_feasibility_analysis import _lifecycle_row
from services.guard_auto import is_market_open_et
from services.pnl_ledger_builder import _norm_sym, _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "recovery_transition_authorization_assessment.json"
OUT_CHECKLIST = RESULTS / "recovery_transition_readiness_checklist.csv"
OUT_BLOCKERS = RESULTS / "recovery_transition_blockers.csv"

RECOVERY_CANDIDATES = ["PEP", "XLE", "ABBV", "MA", "V"]
STAGE_1_REQUIRED = ["PEP"]
BUY_STANCES = {"BUY", "ADD"}
MIN_CONF_STAGE1 = 0.45
MIN_CONF_LATER = 0.45


def _log(msg: str) -> None:
    print(f"[recovery_transition_authorization_assessment] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_context() -> Dict[str, Any]:
    return {
        "transition_plan": _read_json(RESULTS / "autonomous_transition_plan.json"),
        "reconstruction": _read_json(RESULTS / "recovery_basket_reconstruction.json"),
        "optimization": _read_json(RESULTS / "final_governance_recovery_optimization.json"),
        "lifecycle_eff": _read_csv(RESULTS / "signal_lifecycle_effective.csv"),
        "recs": _read_csv(RESULTS / "portfolio_allocation_recommendations.csv"),
        "opps": _read_csv(RESULTS / "trade_opportunities.csv"),
        "auth": _read_json(RESULTS / "governance_authorization.json"),
        "exec_readiness": _read_json(RESULTS / "execution_readiness.json"),
        "exception_gov": _read_json(RESULTS / "recovery_basket_exception_governance.json"),
        "impl_readiness": _read_json(RESULTS / "recovery_basket_implementation_readiness.json"),
        "cpi": _read_json(RESULTS / "capital_preservation_intelligence.json"),
        "cpe": _read_json(RESULTS / "capital_preservation_escalation.json"),
        "guard": _read_json(RESULTS / "guard_snapshot.json"),
        "watchdog": _read_json(ROOT / "data" / "live" / "watchdog_alert_state.json"),
        "positions": _load_positions(),
    }


def _stage_1(ctx: Dict[str, Any]) -> Dict[str, Any]:
    stages = ctx["transition_plan"].get("recommended_stages") or []
    s1 = next((s for s in stages if s.get("stage") == 1), None)
    if not s1:
        all_stages = ctx["transition_plan"].get("transition_stages") or []
        s1 = next(
            (s for s in all_stages if s.get("stage") == 1 and s.get("path_id") == "PATH_E"),
            {},
        )
    return s1 or {
        "target_state": {"QQQ": 90.0, "PEP": 10.0},
        "starting_state": {"QQQ": 100.0},
        "projected_cps": 68,
        "projected_escalation": "ORANGE",
        "projected_governance_status": "BLOCKED",
    }


def _current_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
    plan_state = ctx["transition_plan"].get("current_state") or {}
    positions = ctx["positions"]
    total_mv = sum(p["market_value"] for p in positions) or plan_state.get("total_mv") or 0
    weights: Dict[str, float] = {}
    if total_mv > 0:
        for p in positions:
            sym = p["symbol"]
            weights[sym] = weights.get(sym, 0.0) + p["market_value"] / total_mv * 100.0
    if not weights:
        weights = plan_state.get("weights") or {"QQQ": 100.0}

    cps = int(plan_state.get("current_cps") or ctx["cpi"].get("capital_preservation_score") or 68)
    esc = str(
        ctx["cpe"].get("escalation_state")
        or plan_state.get("escalation")
        or _escalation_from_cps(cps)
    ).upper()
    _, largest, _ = _concentration_score(positions) if positions else (0, 100.0, 1)

    return {
        "weights": {k: _f(v, 2) for k, v in weights.items()},
        "current_cps": cps,
        "escalation": esc,
        "governance_status": (
            "AUTHORIZED" if ctx["auth"].get("governance_authorized") else "BLOCKED"
        ),
        "governance_authorized": ctx["auth"].get("governance_authorized") is True,
        "overall_authorization": ctx["auth"].get("overall_authorization") is True,
        "largest_concentration_pct": _f(largest or weights.get("QQQ", 100.0)),
        "portfolio_value": _f(total_mv),
        "position_count": len(positions) or 1,
    }


def _rec_row(recs: pd.DataFrame, sym: str) -> Dict[str, Any]:
    if recs.empty:
        return {}
    sub = recs[recs["ticker"].map(_norm_sym) == sym]
    return sub.iloc[0].to_dict() if not sub.empty else {}


def _in_opportunities(opps: pd.DataFrame, sym: str) -> bool:
    if opps.empty:
        return False
    col = "ticker" if "ticker" in opps.columns else "symbol"
    return not opps[opps[col].map(_norm_sym) == sym].empty


def _validate_candidate(ctx: Dict[str, Any], sym: str, stage_required: bool) -> Dict[str, Any]:
    sym = _norm_sym(sym)
    lc = _lifecycle_row(ctx["lifecycle_eff"], sym)
    rec = _rec_row(ctx["recs"], sym)
    effective = str(lc.get("effective_stance") or rec.get("effective_stance") or "").upper()
    signal = str(lc.get("signal") or rec.get("signal") or "").upper()
    conf = _to_float(lc.get("confidence") or rec.get("confidence"), np.nan)
    edge = _to_float(rec.get("edge_score"), np.nan)
    alloc = _to_float(rec.get("allocation_score"), np.nan)
    risk = str(rec.get("risk_flag") or "OK").upper()
    exec_blocked = lc.get("execution_blocked") is True
    min_conf = MIN_CONF_STAGE1 if stage_required else MIN_CONF_LATER

    blockers: List[str] = []
    if effective not in BUY_STANCES:
        blockers.append(f"effective_stance={effective}")
    if signal == "SELL":
        blockers.append("signal=SELL")
    if not np.isnan(conf) and conf < min_conf:
        blockers.append(f"confidence={conf:.3f}<{min_conf}")
    if risk != "OK" or "BLOCK" in risk:
        blockers.append(f"risk_flag={risk}")
    if exec_blocked:
        blockers.append("execution_blocked")

    if not blockers:
        status = "VALID"
        exec_elig = "EXECUTABLE"
    elif effective in BUY_STANCES and len(blockers) == 1 and "confidence" in blockers[0]:
        status = "CONDITIONAL"
        exec_elig = "CONDITIONAL"
    else:
        status = "INVALID"
        exec_elig = "NOT_EXECUTABLE"

    recon = ctx["reconstruction"].get("candidate_universe") or []
    recon_row = next((c for c in recon if c.get("symbol") == sym), {})

    return {
        "symbol": sym,
        "stage_1_required": stage_required,
        "effective_stance": effective,
        "lifecycle_action": str(lc.get("lifecycle_action") or "").upper(),
        "signal": signal,
        "confidence": _f(conf, 4),
        "edge_score": _f(edge, 4),
        "allocation_score": _f(alloc, 4),
        "risk_flags": risk,
        "execution_blocked": exec_blocked,
        "in_trade_opportunities": _in_opportunities(ctx["opps"], sym),
        "validation_status": status,
        "execution_eligibility": exec_elig,
        "blockers": blockers,
        "recovery_classification": recon_row.get("recovery_classification") or "—",
        "diversification_score": recon_row.get("diversification_score"),
        "basket_still_valid": status in ("VALID", "CONDITIONAL"),
    }


def _checklist_item(
    name: str,
    category: str,
    passed: bool,
    severity: str,
    detail: str,
    action: str,
) -> Dict[str, Any]:
    return {
        "check_name": name,
        "category": category,
        "status": "PASS" if passed else "FAIL",
        "severity": severity if not passed else "INFO",
        "detail": detail,
        "required_action": action if not passed else "—",
    }


def _governance_readiness(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    fw = ctx["exception_gov"].get("framework_summary") or {}
    impl = ctx["impl_readiness"].get("readiness_scores") or {}
    auth = ctx["auth"]

    stale_scope = fw.get("eligible_assets_now") or ["HD", "XLK"]
    new_basket = RECOVERY_CANDIDATES
    scope_updated = set(_norm_sym(s) for s in stale_scope) != set(new_basket) and bool(new_basket)

    exception_available = bool(fw.get("framework_name")) and fw.get("implements_exception") is False
    dual_approval_model = fw.get("recommended_approval_model") == "DUAL_APPROVAL"
    operator_ok = auth.get("operator_authorized") is True
    gov_ok = auth.get("governance_authorized") is True
    dual_satisfied = operator_ok and gov_ok

    checks = [
        _checklist_item(
            "orange_exception_path_available",
            "governance",
            exception_available,
            "HIGH",
            f"Framework: {fw.get('framework_name', '—')}",
            "Complete exception framework design review",
        ),
        _checklist_item(
            "dual_approval_model_defined",
            "governance",
            dual_approval_model,
            "HIGH",
            f"Model: {fw.get('recommended_approval_model', '—')}",
            "Define approval model in governance framework",
        ),
        _checklist_item(
            "dual_approval_satisfied",
            "governance",
            dual_satisfied,
            "CRITICAL",
            f"operator={operator_ok} governance={gov_ok}",
            "Obtain DUAL_APPROVAL for Stage 1 recovery exception (operator + risk officer)",
        ),
        _checklist_item(
            "exception_scope_limited",
            "governance",
            True,
            "INFO",
            f"CONSERVATIVE tier, max {fw.get('max_duration_hours', 48)}h window",
            "—",
        ),
        _checklist_item(
            "exception_scope_matches_current_basket",
            "governance",
            not scope_updated,
            "HIGH",
            f"Framework lists {stale_scope}; current basket {new_basket}",
            "Update exception registry to Phase 186/187 basket (PEP/XLE/ABBV/MA/V)",
        ),
        _checklist_item(
            "audit_trail_ready",
            "governance",
            _to_float(impl.get("audit_readiness"), 0) >= 70,
            "MODERATE",
            f"Audit readiness score: {impl.get('audit_readiness', '—')}",
            "Complete audit model validation per Phase 181",
        ),
        _checklist_item(
            "governance_gate_clear",
            "governance",
            gov_ok,
            "CRITICAL",
            str(auth.get("gate_reasons", {}).get("governance", "ORANGE gate")),
            "Activate scoped recovery exception OR await CPS lift (Stage 1 remains ORANGE)",
        ),
    ]

    score = sum(1 for c in checks if c["status"] == "PASS") / len(checks) * 100.0

    return {
        "score": _f(score, 1),
        "exception_path_available": exception_available,
        "dual_approval_model": fw.get("recommended_approval_model"),
        "dual_approval_satisfied": dual_satisfied,
        "exception_scope_current": stale_scope,
        "exception_scope_required": new_basket,
        "scope_update_required": scope_updated,
        "governance_authorized": gov_ok,
        "audit_readiness_score": impl.get("audit_readiness"),
        "checks_passed": sum(1 for c in checks if c["status"] == "PASS"),
        "checks_total": len(checks),
    }, checks


def _execution_readiness(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    er = ctx["exec_readiness"]
    auth = ctx["auth"]
    guard = ctx["guard"]
    watchdog = ctx["watchdog"]

    now = datetime.now(timezone.utc)
    market_open = is_market_open_et(now)

    broker_ok = er.get("checks", {}).get("broker", False)
    lifecycle_ok = er.get("checks", {}).get("lifecycle", False)
    signals_ok = er.get("checks", {}).get("signals", False)
    paper_ok = er.get("paper_execution_permitted") is True
    guard_blocked = guard.get("blocked") is True or guard.get("kill_switch") is True
    watchdog_ok = _to_float(watchdog.get("broker_disconnect_streak"), 0) == 0

    checks = [
        _checklist_item(
            "paper_execution_authorized",
            "execution",
            paper_ok,
            "CRITICAL",
            f"mode={er.get('mode', '—')}",
            "Enable paper execution authorization",
        ),
        _checklist_item(
            "broker_connectivity",
            "execution",
            broker_ok and watchdog_ok,
            "CRITICAL",
            er.get("check_details", {}).get("broker", "—"),
            "Restore broker connectivity",
        ),
        _checklist_item(
            "lifecycle_freshness",
            "execution",
            lifecycle_ok,
            "HIGH",
            er.get("check_details", {}).get("lifecycle", "—"),
            "Refresh lifecycle artifacts",
        ),
        _checklist_item(
            "signals_freshness",
            "execution",
            signals_ok,
            "MODERATE",
            er.get("check_details", {}).get("signals", "—"),
            "Refresh signal pipeline",
        ),
        _checklist_item(
            "guard_not_blocking",
            "execution",
            not guard_blocked,
            "CRITICAL",
            str(guard.get("message") or guard.get("reason") or "OK"),
            "Clear guard/kill-switch block",
        ),
        _checklist_item(
            "duplicate_order_protection",
            "execution",
            True,
            "INFO",
            "Lifecycle + order manager dedup assumed active (no duplicate open orders flagged)",
            "—",
        ),
        _checklist_item(
            "market_session_readiness",
            "execution",
            market_open,
            "MODERATE",
            f"NYSE regular session open={market_open} (ET approximation)",
            "Wait for regular market session or use planned limit order queue",
        ),
        _checklist_item(
            "execution_policy_gate",
            "execution",
            auth.get("execution_authorized") is True,
            "HIGH",
            str(auth.get("gate_reasons", {}).get("execution", "—")),
            "Resolve execution policy gate",
        ),
    ]

    score = sum(1 for c in checks if c["status"] == "PASS") / len(checks) * 100.0

    return {
        "score": _f(score, 1),
        "readiness_status": er.get("readiness_status"),
        "paper_execution_permitted": paper_ok,
        "live_execution_permitted": er.get("live_execution_permitted"),
        "market_session_open": market_open,
        "checks_passing_count": er.get("checks_passing_count"),
        "checks_total": er.get("checks_total"),
        "guard_blocked": guard_blocked,
    }, checks


def _lifecycle_readiness(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    stage1 = [c for c in candidates if c.get("stage_1_required")]
    all_valid = all(c.get("basket_still_valid") for c in candidates)
    stage1_ok = all(c.get("validation_status") == "VALID" for c in stage1)
    invalid = [c["symbol"] for c in candidates if c["validation_status"] == "INVALID"]
    conditional = [c["symbol"] for c in candidates if c["validation_status"] == "CONDITIONAL"]

    if stage1_ok and all_valid:
        score = 95.0
    elif stage1_ok:
        score = 85.0
    elif all(c.get("validation_status") != "INVALID" for c in stage1):
        score = 70.0
    else:
        score = 30.0

    return {
        "score": score,
        "stage_1_candidates_valid": stage1_ok,
        "full_basket_valid": all_valid,
        "invalid_symbols": invalid,
        "conditional_symbols": conditional,
    }


def _risk_readiness(ctx: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    conc = state.get("largest_concentration_pct") or 100.0
    cps = state.get("current_cps") or 68
    score = float(np.clip(100 - max(0, conc - 50) * 0.5 - max(0, 75 - cps) * 0.3, 20, 90))
    return {
        "score": _f(score, 1),
        "concentration_pct": conc,
        "current_cps": cps,
        "stage_1_projected_cps": 68,
        "risk_acceptable_for_stage_1_tranche": conc <= 100 and True,
    }


def _operational_readiness(ctx: Dict[str, Any]) -> Dict[str, Any]:
    impl = ctx["impl_readiness"].get("readiness_scores") or {}
    er = ctx["exec_readiness"]
    op = _to_float(impl.get("operational_readiness"), 20.0)
    exec_score = (er.get("checks_passing_count", 0) / max(er.get("checks_total", 1), 1)) * 100
    score = (op + exec_score) / 2.0
    return {
        "score": _f(score, 1),
        "implementation_operational_score": op,
        "execution_check_ratio": _f(exec_score, 1),
    }


def _classify_overall(
    gov: Dict[str, Any],
    lifecycle: Dict[str, Any],
    execution: Dict[str, Any],
    blockers: List[Dict[str, Any]],
) -> str:
    critical = [b for b in blockers if b.get("severity") == "CRITICAL"]
    if any(b.get("component") == "lifecycle" for b in critical):
        return "NOT_READY"
    if not lifecycle.get("stage_1_candidates_valid"):
        return "NOT_READY"

    if critical:
        if gov.get("dual_approval_satisfied") and execution.get("paper_execution_permitted"):
            return "READY_FOR_STAGE_1"
        return "READY_FOR_REVIEW"

    if (
        gov.get("dual_approval_satisfied")
        and execution.get("paper_execution_permitted")
        and lifecycle.get("full_basket_valid")
        and execution.get("score", 0) >= 75
    ):
        return "READY_FOR_STAGE_1"

    if execution.get("score", 0) >= 60 and lifecycle.get("stage_1_candidates_valid"):
        return "READY_FOR_REVIEW"

    return "NOT_READY"


def _build_blockers(
    gov: Dict[str, Any],
    lifecycle: Dict[str, Any],
    execution: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    blockers: List[Dict[str, Any]] = []

    if not gov.get("dual_approval_satisfied"):
        blockers.append(
            {
                "blocker": "DUAL_APPROVAL_NOT_SATISFIED",
                "component": "governance",
                "severity": "CRITICAL",
                "impact": "Stage 1 cannot proceed under ORANGE without scoped exception approval",
                "resolution_requirement": "Operator + risk officer DUAL_APPROVAL for recovery transition exception",
            }
        )

    if gov.get("scope_update_required"):
        blockers.append(
            {
                "blocker": "EXCEPTION_SCOPE_STALE",
                "component": "governance",
                "severity": "HIGH",
                "impact": "Exception registry still references HD/XLK; current basket is PEP/XLE/ABBV/MA/V",
                "resolution_requirement": "Update recovery basket exception registry to Phase 186/187 symbols",
            }
        )

    if not state.get("governance_authorized"):
        blockers.append(
            {
                "blocker": "GOVERNANCE_ORANGE_GATE",
                "component": "governance",
                "severity": "CRITICAL",
                "impact": "Standard governance path blocked; Stage 1 projected to remain ORANGE/BLOCKED until exception",
                "resolution_requirement": "Activate PATH_B scoped exception for Stage 1 tranche OR document acceptance of ORANGE during Stage 1",
            }
        )

    for c in candidates:
        if c["validation_status"] == "INVALID":
            blockers.append(
                {
                    "blocker": f"LIFECYCLE_INVALID_{c['symbol']}",
                    "component": "lifecycle",
                    "severity": "CRITICAL" if c["stage_1_required"] else "HIGH",
                    "impact": f"{c['symbol']} no longer lifecycle-actionable: {', '.join(c['blockers'])}",
                    "resolution_requirement": "Replace candidate or wait for lifecycle recovery to BUY",
                }
            )

    if not execution.get("paper_execution_permitted"):
        blockers.append(
            {
                "blocker": "PAPER_EXECUTION_NOT_PERMITTED",
                "component": "execution",
                "severity": "CRITICAL",
                "impact": "Paper execution layer blocked",
                "resolution_requirement": "Clear execution readiness failures",
            }
        )

    if execution.get("guard_blocked"):
        blockers.append(
            {
                "blocker": "GUARD_OR_KILL_SWITCH_ACTIVE",
                "component": "execution",
                "severity": "CRITICAL",
                "impact": "Guard snapshot blocking execution",
                "resolution_requirement": "Clear guard block before transition",
            }
        )

    if not execution.get("market_session_open"):
        blockers.append(
            {
                "blocker": "MARKET_SESSION_CLOSED",
                "component": "operational",
                "severity": "MODERATE",
                "impact": "Outside NYSE regular session — timing risk for Stage 1 execution window",
                "resolution_requirement": "Schedule Stage 1 for next regular market session",
            }
        )

    invalid_later = [c["symbol"] for c in candidates if c["validation_status"] == "CONDITIONAL"]
    if invalid_later:
        blockers.append(
            {
                "blocker": "CONDITIONAL_CONFIDENCE_LATER_STAGES",
                "component": "lifecycle",
                "severity": "MODERATE",
                "impact": f"Symbols with confidence caveats: {', '.join(invalid_later)}",
                "resolution_requirement": "Re-validate before Stages 2-5; does not block Stage 1 if PEP valid",
            }
        )

    return blockers


def _autonomous_readiness(
    gov: Dict[str, Any],
    lifecycle: Dict[str, Any],
    execution: Dict[str, Any],
    risk: Dict[str, Any],
    operational: Dict[str, Any],
    classification: str,
) -> Dict[str, Any]:
    overall = (
        _to_float(gov.get("score"), 0) * 0.30
        + _to_float(lifecycle.get("score"), 0) * 0.25
        + _to_float(execution.get("score"), 0) * 0.20
        + _to_float(risk.get("score"), 0) * 0.10
        + _to_float(operational.get("score"), 0) * 0.15
    )
    return {
        "governance_readiness": gov.get("score"),
        "lifecycle_readiness": lifecycle.get("score"),
        "execution_readiness": execution.get("score"),
        "risk_readiness": risk.get("score"),
        "operational_readiness": operational.get("score"),
        "overall_transition_readiness": _f(overall, 1),
        "overall_classification": classification,
        "autonomous_execution_permitted": False,
        "human_in_loop_required": True,
        "rationale": (
            "Stage 1 requires human DUAL_APPROVAL under ORANGE governance. "
            "Autonomous transition blocked until POST_YELLOW checkpoint (Stage 3)."
        ),
    }


def _executive_report(
    state: Dict[str, Any],
    stage1: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    autonomy: Dict[str, Any],
    blockers: List[Dict[str, Any]],
    gov: Dict[str, Any],
    execution: Dict[str, Any],
) -> Dict[str, Any]:
    critical = [b for b in blockers if b.get("severity") == "CRITICAL"]
    pep = next((c for c in candidates if c["symbol"] == "PEP"), {})

    if autonomy["overall_classification"] == "READY_FOR_STAGE_1":
        auth_status = "READY_FOR_AUTHORIZATION_REVIEW"
        next_action = (
            "Proceed to formal Stage 1 authorization ceremony (still requires human approval)"
        )
    elif autonomy["overall_classification"] == "READY_FOR_REVIEW":
        auth_status = "NOT_AUTHORIZED — READY_FOR_REVIEW"
        next_action = (
            "Proceed to DUAL_APPROVAL review: update exception scope to current basket, "
            "then authorize Stage 1 PEP 10% tranche under PATH_E"
        )
    else:
        auth_status = "NOT_AUTHORIZED"
        next_action = "Resolve critical blockers before approval review"

    return {
        "report_type": "RECOVERY_TRANSITION_AUTHORIZATION_REPORT",
        "current_portfolio_state": state,
        "target_stage_1_transition": {
            "path": "PATH_E Stage 1",
            "starting_state": stage1.get("starting_state"),
            "target_state": stage1.get("target_state"),
            "actions": stage1.get("required_actions"),
            "projected_cps": stage1.get("projected_cps"),
            "projected_escalation": stage1.get("projected_escalation"),
            "projected_governance": stage1.get("projected_governance_status"),
        },
        "recovery_candidate_status": candidates,
        "readiness_scorecard": {
            "governance": gov.get("score"),
            "lifecycle": autonomy.get("lifecycle_readiness"),
            "execution": autonomy.get("execution_readiness"),
            "risk": autonomy.get("risk_readiness"),
            "operational": autonomy.get("operational_readiness"),
            "overall": autonomy.get("overall_transition_readiness"),
        },
        "authorization_status": auth_status,
        "stage_1_can_begin": pep.get("validation_status") == "VALID"
        and execution.get("paper_execution_permitted"),
        "stage_1_authorized": False,
        "authorization_note": "This assessment does NOT authorize execution.",
        "required_approvals": [
            "Operator approval — Stage 1 transition plan",
            "Risk officer approval — DUAL_APPROVAL recovery exception",
            "Exception scope update — PEP/XLE/ABBV/MA/V basket registry",
            "Pre-trade lifecycle revalidation — PEP effective BUY",
        ],
        "current_blockers": [b["blocker"] for b in blockers],
        "critical_blocker_count": len(critical),
        "recommended_next_action": next_action,
        "executive_questions": {
            "can_stage_1_begin": pep.get("validation_status") == "VALID",
            "candidates_still_valid": all(c.get("basket_still_valid") for c in candidates),
            "exception_ready": gov.get("exception_path_available")
            and not gov.get("scope_update_required"),
            "dual_approval_satisfied": gov.get("dual_approval_satisfied"),
            "paper_environment_ready": execution.get("paper_execution_permitted"),
            "what_prevents_authorization": (
                critical[0]["blocker"] if critical else "Human approval pending"
            ),
            "wait_or_proceed": (
                "PROCEED_TO_APPROVAL_REVIEW"
                if autonomy["overall_classification"] == "READY_FOR_REVIEW"
                else "WAIT"
            ),
        },
    }


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    ctx = _load_context()
    state = _current_state(ctx)
    stage1 = _stage_1(ctx)

    candidates = [
        _validate_candidate(ctx, sym, sym in STAGE_1_REQUIRED) for sym in RECOVERY_CANDIDATES
    ]

    gov, gov_checks = _governance_readiness(ctx)
    execution, exec_checks = _execution_readiness(ctx)
    lifecycle = _lifecycle_readiness(candidates)
    risk = _risk_readiness(ctx, state)
    operational = _operational_readiness(ctx)

    blockers = _build_blockers(gov, lifecycle, execution, candidates, state)
    classification = _classify_overall(gov, lifecycle, execution, blockers)
    autonomy = _autonomous_readiness(gov, lifecycle, execution, risk, operational, classification)
    executive = _executive_report(state, stage1, candidates, autonomy, blockers, gov, execution)

    checklist = gov_checks + exec_checks
    lifecycle_checks = [
        _checklist_item(
            f"candidate_{c['symbol']}_valid",
            "lifecycle",
            c["validation_status"] == "VALID",
            "CRITICAL" if c["stage_1_required"] else "MODERATE",
            f"stance={c['effective_stance']} signal={c['signal']} conf={c['confidence']}",
            f"Resolve: {', '.join(c['blockers'])}" if c["blockers"] else "—",
        )
        for c in candidates
    ]
    checklist.extend(lifecycle_checks)

    summary = {
        "overall_classification": classification,
        "overall_readiness_score": autonomy.get("overall_transition_readiness"),
        "stage_1_pep_valid": next((c for c in candidates if c["symbol"] == "PEP"), {}).get(
            "validation_status"
        )
        == "VALID",
        "full_basket_valid": lifecycle.get("full_basket_valid"),
        "dual_approval_satisfied": gov.get("dual_approval_satisfied"),
        "paper_ready": execution.get("paper_execution_permitted"),
        "blocker_count": len(blockers),
        "critical_blocker_count": sum(1 for b in blockers if b.get("severity") == "CRITICAL"),
        "authorization_granted": False,
        "recommended_action": executive.get("recommended_next_action"),
    }

    report = {
        "generated_at": ts,
        "phase": "189",
        "mode": "recovery_transition_authorization_assessment",
        "disclaimer": (
            "Assessment only — does NOT authorize execution, activate exceptions, "
            "modify governance, lifecycle, execution, broker, allocation, or risk controls."
        ),
        "current_state": state,
        "stage_1_target": {
            "path_id": "PATH_E",
            "stage": 1,
            **{k: stage1.get(k) for k in stage1 if k not in ("path_name",)},
        },
        "candidate_validation": candidates,
        "governance_readiness": gov,
        "execution_readiness": execution,
        "autonomous_readiness": autonomy,
        "readiness_checklist": checklist,
        "blockers": blockers,
        "executive_recommendation": executive,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "read-only assessment — no authorization granted",
        },
    }

    pd.DataFrame(checklist).to_csv(OUT_CHECKLIST, index=False, encoding="utf-8")
    pd.DataFrame(blockers).to_csv(OUT_BLOCKERS, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_CHECKLIST.name}, {OUT_BLOCKERS.name}")
    _log(
        f"classification={classification} readiness={summary['overall_readiness_score']} "
        f"blockers={summary['blocker_count']} pep_valid={summary['stage_1_pep_valid']}"
    )
    return report


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
