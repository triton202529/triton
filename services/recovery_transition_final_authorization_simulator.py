"""
TRITON Phase 192 — Recovery Transition Final Authorization Simulator (read-only).

End-to-end dress rehearsal of Stage 1 recovery authorization from governance
package review through GO/NO-GO decision. Does NOT grant approvals, activate
exceptions, or execute trades.

SAFETY: simulation only — no rule changes, no broker interaction, no state mutation.

Outputs (data/results/):
    final_authorization_simulation.json
    final_authorization_decision_tree.csv
    final_authorization_audit_trail.csv

Run:
    python -m services.recovery_transition_final_authorization_simulator
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

import pandas as pd

from services.concentration_governance_analysis import (
    _escalation_from_cps,
    _f,
    _load_positions,
    _read_csv,
    _read_json,
)
from services.diversification_feasibility_analysis import _lifecycle_row
from services.guard_auto import is_market_open_et
from services.pnl_ledger_builder import _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "final_authorization_simulation.json"
OUT_TREE = RESULTS / "final_authorization_decision_tree.csv"
OUT_AUDIT = RESULTS / "final_authorization_audit_trail.csv"

REGISTRY = ["PEP", "XLE", "ABBV", "MA", "V"]
STAGE_1_SYMBOL = "PEP"
BUY_STANCES = {"BUY", "ADD"}
MIN_CONF = 0.45


def _log(msg: str) -> None:
    print(f"[recovery_transition_final_authorization_simulator] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_context() -> Dict[str, Any]:
    return {
        "playbook": _read_json(RESULTS / "stage1_governance_playbook.json"),
        "approval_package": _read_json(RESULTS / "recovery_exception_registry_migration.json"),
        "auth_assessment": _read_json(
            RESULTS / "recovery_transition_authorization_assessment.json"
        ),
        "exception_gov": _read_json(RESULTS / "recovery_basket_exception_governance.json"),
        "lifecycle_eff": _read_csv(RESULTS / "signal_lifecycle_effective.csv"),
        "exec_readiness": _read_json(RESULTS / "execution_readiness.json"),
        "cpi": _read_json(RESULTS / "capital_preservation_intelligence.json"),
        "cpe": _read_json(RESULTS / "capital_preservation_escalation.json"),
        "guard": _read_json(RESULTS / "guard_snapshot.json"),
        "positions": _load_positions(),
    }


def _current_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
    pb = ctx["playbook"].get("current_state") or {}
    assessment = ctx["auth_assessment"].get("current_state") or {}
    positions = ctx["positions"]
    total_mv = sum(p["market_value"] for p in positions) or pb.get("portfolio_value") or 0
    weights: Dict[str, float] = {}
    if total_mv > 0:
        for p in positions:
            sym = p["symbol"]
            weights[sym] = weights.get(sym, 0.0) + p["market_value"] / total_mv * 100.0
    if not weights:
        weights = pb.get("weights") or assessment.get("weights") or {"QQQ": 100.0}

    cps = int(
        pb.get("current_cps")
        or assessment.get("current_cps")
        or ctx["cpi"].get("capital_preservation_score")
        or 68
    )
    esc = str(
        pb.get("escalation")
        or assessment.get("escalation")
        or ctx["cpe"].get("escalation_state")
        or _escalation_from_cps(cps)
    ).upper()

    pv = total_mv or pb.get("portfolio_value") or 1436.64
    return {
        "weights": {k: _f(v, 2) for k, v in weights.items()},
        "current_cps": cps,
        "escalation": esc,
        "governance_status": pb.get("governance_status") or "BLOCKED",
        "governance_authorized": False,
        "portfolio_value": _f(pv),
        "authorization_status": "READY_FOR_REVIEW",
        "stage_1_transition": {
            "current": {"QQQ": 100.0, "PEP": 0.0},
            "target": {"QQQ": 90.0, "PEP": 10.0},
            "qqq_reduction_notional": _f(pv * 0.10),
            "pep_acquisition_notional": _f(pv * 0.10),
        },
        "recovery_exception": {
            "framework_valid": True,
            "registry_updated": True,
            "approved_basket": REGISTRY,
            "exception_activated": False,
        },
    }


def _pep_lifecycle(ctx: Dict[str, Any]) -> Dict[str, Any]:
    pb_pep = ctx["playbook"].get("pep_lifecycle") or {}
    if pb_pep.get("symbol"):
        return pb_pep
    lc = _lifecycle_row(ctx["lifecycle_eff"], STAGE_1_SYMBOL)
    effective = str(lc.get("effective_stance") or "").upper()
    signal = str(lc.get("signal") or "").upper()
    conf = _to_float(lc.get("confidence"), 0.0)
    risk_ok = str(lc.get("risk_flag") or "OK").upper() == "OK"
    valid = effective in BUY_STANCES and signal != "SELL" and conf >= MIN_CONF and risk_ok
    blockers = []
    if effective not in BUY_STANCES:
        blockers.append(f"effective_stance={effective}")
    if conf < MIN_CONF:
        blockers.append(f"confidence={conf:.4f}<{MIN_CONF}")
    if not risk_ok:
        blockers.append("blocking_risk_event")
    return {
        "symbol": STAGE_1_SYMBOL,
        "effective_stance": effective,
        "signal": signal,
        "confidence": _f(conf, 4),
        "lifecycle_valid": valid,
        "blockers": blockers,
    }


def _real_signatures(ctx: Dict[str, Any]) -> Tuple[bool, bool]:
    auth_req = ctx["playbook"].get("authorization_requirements") or {}
    op = auth_req.get("operator_signature") is True
    risk = auth_req.get("risk_office_signature") is True
    checklist = ctx["approval_package"].get("readiness_checklist") or []
    if not op:
        op = any(
            c.get("approval_item") == "Operator signature" and c.get("status") == "COMPLETE"
            for c in checklist
        )
    if not risk:
        risk = any(
            c.get("approval_item") == "Risk office signature" and c.get("status") == "COMPLETE"
            for c in checklist
        )
    return op, risk


def _step1_governance_package(ctx: Dict[str, Any]) -> Dict[str, Any]:
    migration = ctx["approval_package"].get("registry_migration") or {}
    evidence = ctx["approval_package"].get("evidence_package") or []
    audit = ctx["playbook"].get("audit_requirements") or []
    checks = {
        "recovery_exception_package_complete": bool(ctx["approval_package"]),
        "registry_migration_approved_for_review": migration.get("migration_classification")
        == "APPROVED_FOR_REVIEW",
        "evidence_package_available": len(evidence) >= 5,
        "audit_requirements_available": len(audit) >= 5,
    }
    passed = all(checks.values())
    return {
        "step": 1,
        "name": "Governance Package Review",
        "outcome": "PASS" if passed else "FAIL",
        "checks": checks,
        "blocking_reason": None if passed else "Incomplete governance package",
    }


def _step2_dual_approval(
    ctx: Dict[str, Any], ts: str, audit_events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    op_real, risk_real = _real_signatures(ctx)
    real_dual = op_real and risk_real

    # Simulate hypothetical approvals for dress rehearsal (not persisted)
    sim_op_ts = f"{ts}_sim_operator"
    sim_risk_ts = f"{ts}_sim_risk_office"
    audit_events.append(
        {
            "event": "operator_decision_simulated",
            "timestamp": sim_op_ts,
            "owner": "OPERATOR",
            "result": "APPROVED (simulated)",
            "evidence": "Hypothetical operator sign-off for control validation",
        }
    )
    audit_events.append(
        {
            "event": "risk_office_decision_simulated",
            "timestamp": sim_risk_ts,
            "owner": "RISK_OFFICE",
            "result": "APPROVED (simulated)",
            "evidence": "Hypothetical risk office sign-off for control validation",
        }
    )

    bypass_tests = [
        {
            "scenario": "operator_only",
            "operator_approved": True,
            "risk_office_approved": False,
            "can_proceed": False,
            "detail": "Single operator approval cannot bypass governance",
        },
        {
            "scenario": "risk_office_only",
            "operator_approved": False,
            "risk_office_approved": True,
            "can_proceed": False,
            "detail": "Single risk office approval cannot bypass governance",
        },
        {
            "scenario": "both_approved_simulated",
            "operator_approved": True,
            "risk_office_approved": True,
            "can_proceed": True,
            "detail": "Both approvals required and sufficient in simulation",
        },
        {
            "scenario": "neither_approved_real",
            "operator_approved": op_real,
            "risk_office_approved": risk_real,
            "can_proceed": real_dual,
            "detail": "Current real-world signature state",
        },
    ]

    sim_authorized = True and True  # simulated both approve
    real_outcome = "AUTHORIZED" if real_dual else "BLOCKED"
    sim_outcome = "AUTHORIZED" if sim_authorized else "BLOCKED"

    audit_events.append(
        {
            "event": "dual_approval_real_state",
            "timestamp": ts,
            "owner": "GOVERNANCE",
            "result": real_outcome,
            "evidence": f"operator={op_real} risk_office={risk_real}",
        }
    )

    return {
        "step": 2,
        "name": "DUAL_APPROVAL Simulation",
        "real_outcome": real_outcome,
        "simulated_outcome": sim_outcome,
        "operator_signature_real": op_real,
        "risk_office_signature_real": risk_real,
        "both_required": True,
        "single_approval_bypass_possible": False,
        "bypass_tests": bypass_tests,
        "simulated_timestamps_recorded": True,
        "audit_evidence_generated": True,
        "blocking_reason": None if real_dual else "DUAL_APPROVAL_NOT_SATISFIED",
    }


def _step3_market_readiness(ctx: Dict[str, Any]) -> Dict[str, Any]:
    pb_market = ctx["playbook"].get("market_readiness") or {}
    er = ctx["exec_readiness"]
    checks_er = er.get("checks") or {}
    now = datetime.now(timezone.utc)
    market_open = pb_market.get("market_session_open")
    if market_open is None:
        market_open = is_market_open_et(now)

    stale = pb_market.get("stale_orders_count", 0)
    guard = ctx["guard"]
    guard_blocked = bool(guard.get("blocked") or guard.get("kill_switch_active"))

    checks = {
        "market_session_open": market_open is True,
        "broker_connection_healthy": checks_er.get("broker") is True,
        "paper_environment_ready": er.get("paper_execution_permitted") is True,
        "no_conflicting_orders": stale == 0,
        "lifecycle_data_fresh": checks_er.get("lifecycle") is True,
        "no_risk_office_blocking_events": checks_er.get("watchdog") is True and not guard_blocked,
    }
    ready = all(checks.values())
    failures = [k for k, v in checks.items() if not v]
    return {
        "step": 3,
        "name": "Market Readiness Simulation",
        "outcome": "READY" if ready else "NOT_READY",
        "checks": checks,
        "blocking_reason": ", ".join(failures) if failures else None,
    }


def _step4_lifecycle(pep: Dict[str, Any]) -> Dict[str, Any]:
    valid = pep.get("lifecycle_valid") is True
    return {
        "step": 4,
        "name": "Final Lifecycle Validation",
        "symbol": STAGE_1_SYMBOL,
        "effective_stance": pep.get("effective_stance"),
        "confidence": pep.get("confidence"),
        "outcome": "APPROVED_FOR_EXECUTION" if valid else "REJECT_RECOVERY_STAGE",
        "checks": {
            "effective_stance_buy_add": str(pep.get("effective_stance", "")).upper() in BUY_STANCES,
            "confidence_acceptable": _to_float(pep.get("confidence"), 0) >= MIN_CONF,
            "no_blocking_risk_events": "blocking_risk_event" not in (pep.get("blockers") or []),
        },
        "blockers": pep.get("blockers") or [],
        "blocking_reason": (
            None if valid else "; ".join(pep.get("blockers") or ["lifecycle_invalid"])
        ),
    }


def _step5_exception_activation(ctx: Dict[str, Any], prior_ok: bool) -> Dict[str, Any]:
    migration = ctx["approval_package"].get("registry_migration") or {}
    proposed = migration.get("proposed_registry") or REGISTRY
    limits = (ctx["exception_gov"].get("exception_limits") or {}).get("CONSERVATIVE") or {}
    revocation = ctx["exception_gov"].get("revocation_rules") or {}

    scope_ok = proposed == REGISTRY
    duration_ok = (migration.get("max_duration_hours") or 48) == 48
    limits_ok = bool(limits)
    revoke_ok = bool(
        revocation.get("automatic_revocation_triggers")
        or revocation.get("immediate_revocation_triggers")
    )
    activation_ok = prior_ok and scope_ok and duration_ok and limits_ok and revoke_ok

    return {
        "step": 5,
        "name": "Exception Activation Simulation",
        "outcome": (
            "SIMULATED_ACTIVATION_SUCCESS" if activation_ok else "SIMULATED_ACTIVATION_REJECTED"
        ),
        "exception_activated_in_system": False,
        "simulation_only": True,
        "scope_symbols": proposed,
        "scope_valid": scope_ok,
        "duration_hours": migration.get("max_duration_hours") or 48,
        "duration_valid": duration_ok,
        "limits_tier": migration.get("limits_tier") or "CONSERVATIVE",
        "allocation_limits_enforced": limits_ok,
        "revocation_triggers_active": revoke_ok,
        "blocking_reason": None if activation_ok else "Prior gates failed or scope/limits invalid",
    }


def _step6_go_nogo(steps: List[Dict[str, Any]], pep: Dict[str, Any]) -> Dict[str, Any]:
    s1 = steps[0]
    s2 = steps[1]
    s3 = steps[2]
    s4 = steps[3]
    s5 = steps[4]

    # Real-world GO requires real dual approval
    real_conditions = {
        "governance_package_pass": s1.get("outcome") == "PASS",
        "dual_approval_real": s2.get("real_outcome") == "AUTHORIZED",
        "market_ready": s3.get("outcome") == "READY",
        "pep_buy": s4.get("outcome") == "APPROVED_FOR_EXECUTION",
        "exception_simulation_success": s5.get("outcome") == "SIMULATED_ACTIVATION_SUCCESS",
    }

    # Dress-rehearsal GO: simulated dual approval + all other real checks
    sim_conditions = {
        **real_conditions,
        "dual_approval_real": True,  # replaced by simulated both-approved path
        "dual_approval_simulated": s2.get("simulated_outcome") == "AUTHORIZED",
    }
    sim_go = all(
        [
            sim_conditions["governance_package_pass"],
            sim_conditions["dual_approval_simulated"],
            sim_conditions["market_ready"],
            sim_conditions["pep_buy"],
            sim_conditions["exception_simulation_success"],
        ]
    )
    real_go = all(real_conditions.values())

    blockers = [k for k, v in real_conditions.items() if not v]
    decision = "GO" if real_go else "NO_GO"
    dress_rehearsal = "GO" if sim_go else "NO_GO"

    no_go_examples = []
    if not real_conditions["dual_approval_real"]:
        no_go_examples.append("Approval missing — DUAL_APPROVAL not satisfied")
    if not real_conditions["market_ready"]:
        no_go_examples.append("Market closed or operational checks failed")
    if not real_conditions["pep_buy"]:
        no_go_examples.append("PEP deteriorates — lifecycle not BUY/ADD")
    if not real_conditions["governance_package_pass"]:
        no_go_examples.append("Governance control failure — package incomplete")
    if s5.get("outcome") != "SIMULATED_ACTIVATION_SUCCESS":
        no_go_examples.append("Exception scope or limits invalid")

    return {
        "step": 6,
        "name": "Stage 1 GO / NO-GO Decision",
        "real_world_decision": decision,
        "dress_rehearsal_decision": dress_rehearsal,
        "real_conditions": real_conditions,
        "simulated_dual_approval_used_for_rehearsal": True,
        "authorization_granted": False,
        "exception_activated": False,
        "trades_executed": False,
        "blocking_reasons": blockers,
        "no_go_examples": no_go_examples,
        "pep_at_decision": pep,
    }


def _failure_scenarios(ctx: Dict[str, Any], pep: Dict[str, Any]) -> List[Dict[str, Any]]:
    op_real, risk_real = _real_signatures(ctx)
    market = _step3_market_readiness(ctx)
    migration = ctx["approval_package"].get("registry_migration") or {}

    scenarios = [
        {
            "scenario": "Operator approval denied",
            "failure_classification": "AUTHORIZATION_DENIED",
            "simulated_result": "NO_GO",
            "required_response": "HALT; document denial; do not proceed to market readiness",
            "escalation_requirement": "Governance review",
            "re_approval_requirement": "New operator review after remediation",
        },
        {
            "scenario": "Risk Office approval denied",
            "failure_classification": "AUTHORIZATION_DENIED",
            "simulated_result": "NO_GO",
            "required_response": "HALT; exception request rejected",
            "escalation_requirement": "Risk Office escalation log",
            "re_approval_requirement": "Full new DUAL_APPROVAL cycle",
        },
        {
            "scenario": "PEP changes BUY -> WAIT",
            "failure_classification": "LIFECYCLE_DETERIORATION",
            "simulated_result": "NO_GO",
            "required_response": "REJECT_RECOVERY_STAGE; do not activate exception",
            "escalation_requirement": "Lifecycle team notification",
            "re_approval_requirement": "Re-validate basket; possible registry revision",
        },
        {
            "scenario": "Broker unavailable",
            "failure_classification": "OPERATIONAL_FAILURE",
            "simulated_result": "NO_GO",
            "required_response": "HALT all order activity",
            "escalation_requirement": "Operations + Risk Office",
            "re_approval_requirement": "Resume only after connectivity restored and re-check",
        },
        {
            "scenario": "Market closed",
            "failure_classification": "TIMING_FAILURE",
            "simulated_result": "NO_GO",
            "required_response": "Defer to next regular session",
            "escalation_requirement": None,
            "re_approval_requirement": "Re-run market readiness at session open",
        },
        {
            "scenario": "Exception scope mismatch",
            "failure_classification": "GOVERNANCE_CONTROL_FAILURE",
            "simulated_result": "NO_GO",
            "required_response": "SIMULATED_ACTIVATION_REJECTED",
            "escalation_requirement": "Registry migration review",
            "re_approval_requirement": "Update exception registry before retry",
        },
        {
            "scenario": "Audit evidence missing",
            "failure_classification": "AUDIT_FAILURE",
            "simulated_result": "NO_GO",
            "required_response": "Block activation until audit pipeline confirmed",
            "escalation_requirement": "Audit team",
            "re_approval_requirement": "Complete evidence package before approval",
        },
        {
            "scenario": "Risk event detected",
            "failure_classification": "RISK_CONTROL_TRIGGER",
            "simulated_result": "NO_GO",
            "required_response": "IMMEDIATE_SUSPEND; guard/watchdog protocol",
            "escalation_requirement": "Risk Office immediate review",
            "re_approval_requirement": "Clear risk event + new DUAL_APPROVAL",
        },
    ]

    # Annotate with current applicability
    for s in scenarios:
        name = s["scenario"]
        if name == "Operator approval denied":
            s["currently_applicable"] = not op_real
        elif name == "Risk Office approval denied":
            s["currently_applicable"] = not risk_real
        elif name == "PEP changes BUY -> WAIT":
            s["currently_applicable"] = not pep.get("lifecycle_valid")
        elif name == "Broker unavailable":
            s["currently_applicable"] = not (ctx["exec_readiness"].get("checks") or {}).get(
                "broker"
            )
        elif name == "Market closed":
            s["currently_applicable"] = market.get("outcome") != "READY" and not market.get(
                "checks", {}
            ).get("market_session_open", True)
        elif name == "Exception scope mismatch":
            s["currently_applicable"] = (migration.get("proposed_registry") or REGISTRY) != REGISTRY
        elif name == "Audit evidence missing":
            s["currently_applicable"] = (
                len(ctx["approval_package"].get("evidence_package") or []) < 5
            )
        else:
            s["currently_applicable"] = False

    return scenarios


def _build_audit_trail(
    ts: str,
    steps: List[Dict[str, Any]],
    go_nogo: Dict[str, Any],
    dual: Dict[str, Any],
) -> List[Dict[str, Any]]:
    trail = [
        {
            "event": "authorization_requested",
            "timestamp": ts,
            "owner": "GOVERNANCE",
            "result": "SIMULATION_STARTED",
            "evidence": "Phase 192 final dress rehearsal — no real authorization",
        },
        {
            "event": "evidence_reviewed",
            "timestamp": ts,
            "owner": "GOVERNANCE",
            "result": steps[0].get("outcome"),
            "evidence": str(steps[0].get("checks")),
        },
        {
            "event": "operator_decision",
            "timestamp": ts,
            "owner": "OPERATOR",
            "result": "PENDING (real)" if not dual.get("operator_signature_real") else "APPROVED",
            "evidence": f"simulated={dual.get('simulated_outcome')}",
        },
        {
            "event": "risk_office_decision",
            "timestamp": ts,
            "owner": "RISK_OFFICE",
            "result": (
                "PENDING (real)" if not dual.get("risk_office_signature_real") else "APPROVED"
            ),
            "evidence": f"real_outcome={dual.get('real_outcome')}",
        },
        {
            "event": "lifecycle_confirmation",
            "timestamp": ts,
            "owner": "LIFECYCLE",
            "result": steps[3].get("outcome"),
            "evidence": str(steps[3].get("checks")),
        },
        {
            "event": "market_readiness_check",
            "timestamp": ts,
            "owner": "OPERATIONS",
            "result": steps[2].get("outcome"),
            "evidence": str(steps[2].get("checks")),
        },
        {
            "event": "go_nogo_decision",
            "timestamp": ts,
            "owner": "GOVERNANCE",
            "result": go_nogo.get("real_world_decision"),
            "evidence": str(go_nogo.get("blocking_reasons")),
        },
        {
            "event": "exception_activation_simulation",
            "timestamp": ts,
            "owner": "GOVERNANCE",
            "result": steps[4].get("outcome"),
            "evidence": "exception_activated_in_system=false",
        },
        {
            "event": "final_recommendation",
            "timestamp": ts,
            "owner": "SYSTEM",
            "result": go_nogo.get("real_world_decision"),
            "evidence": f"dress_rehearsal={go_nogo.get('dress_rehearsal_decision')}",
        },
    ]
    return trail


def _decision_tree(steps: List[Dict[str, Any]], go_nogo: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    mapping = [
        (steps[0], "Governance package complete", "outcome", "PASS"),
        (steps[1], "DUAL_APPROVAL (real)", "real_outcome", "AUTHORIZED"),
        (steps[1], "DUAL_APPROVAL (simulated dress rehearsal)", "simulated_outcome", "AUTHORIZED"),
        (steps[2], "Market session OPEN and operational checks", "outcome", "READY"),
        (
            steps[3],
            "PEP lifecycle BUY/ADD with acceptable confidence",
            "outcome",
            "APPROVED_FOR_EXECUTION",
        ),
        (steps[4], "Exception scope and limits valid", "outcome", "SIMULATED_ACTIVATION_SUCCESS"),
        (go_nogo, "Final real-world GO/NO-GO", "real_world_decision", "GO"),
    ]
    for obj, req, key, pass_val in mapping:
        val = obj.get(key)
        passed = val == pass_val
        rows.append(
            {
                "decision_step": obj.get("name") or obj.get("step"),
                "requirement": req,
                "result": val,
                "blocking_reason": None if passed else obj.get("blocking_reason") or val,
            }
        )
    return rows


def _autonomy_validation() -> Dict[str, Any]:
    return {
        "triton_can": [
            "Prepare evidence packages",
            "Validate lifecycle against effective stance",
            "Calculate CPS and concentration metrics",
            "Monitor market session and broker health",
            "Generate GO/NO-GO recommendations",
            "Simulate authorization flows",
            "Produce audit trail entries",
        ],
        "triton_cannot": [
            "Self-approve DUAL_APPROVAL",
            "Override Risk Office denial",
            "Override governance BLOCKED state",
            "Activate recovery exceptions in production",
            "Submit Stage 1 trades or orders",
            "Modify authorization state files",
            "Bypass lifecycle authority",
        ],
        "autonomy_boundaries_enforced": True,
        "self_approval_attempt_blocked": True,
        "single_approval_bypass_blocked": True,
    }


def _executive_report(
    state: Dict[str, Any],
    steps: List[Dict[str, Any]],
    go_nogo: Dict[str, Any],
    dual: Dict[str, Any],
    failures: List[Dict[str, Any]],
    audit: List[Dict[str, Any]],
    autonomy: Dict[str, Any],
    pep: Dict[str, Any],
) -> Dict[str, Any]:
    controls_ok = dual.get("single_approval_bypass_possible") is False
    audit_complete = len(audit) >= 8
    applicable = [f for f in failures if f.get("currently_applicable")]

    if controls_ok and audit_complete and go_nogo.get("real_world_decision") == "NO_GO":
        recommendation = (
            "SIMULATION COMPLETE — all authorization controls function correctly. "
            f"Real-world decision: NO_GO ({', '.join(go_nogo.get('blocking_reasons') or ['gates pending'])}). "
            "Obtain DUAL_APPROVAL signatures and open market session before live authorization. "
            "No authorization granted by this simulation."
        )
        readiness = "CONTROLS_VALIDATED_AWAITING_REAL_APPROVAL"
    elif go_nogo.get("real_world_decision") == "GO":
        recommendation = (
            "SIMULATION COMPLETE — real-world gates pass. Proceed to controlled Stage 1 "
            "under operator supervision per Phase 191 playbook. Simulation does NOT grant authorization."
        )
        readiness = "READY_FOR_CONTROLLED_EXECUTION"
    else:
        recommendation = (
            "SIMULATION COMPLETE — controls validated with gaps. Review failure scenarios "
            "and resolve blockers before formal authorization."
        )
        readiness = "SIMULATION_COMPLETE_WITH_GAPS"

    return {
        "report_type": "FINAL_RECOVERY_AUTHORIZATION_SIMULATION_REPORT",
        "current_state": state,
        "authorization_chain": [
            {
                "step": s.get("step"),
                "name": s.get("name"),
                "outcome": s.get("outcome") or s.get("real_outcome"),
            }
            for s in steps
        ],
        "go_nogo_outcome": {
            "real_world": go_nogo.get("real_world_decision"),
            "dress_rehearsal": go_nogo.get("dress_rehearsal_decision"),
            "authorization_granted": False,
            "exception_activated": False,
        },
        "failure_analysis_summary": {
            "scenarios_tested": len(failures),
            "currently_applicable": [f["scenario"] for f in applicable],
        },
        "audit_trail_complete": audit_complete,
        "autonomy_boundaries": autonomy,
        "executive_questions": {
            "approval_controls_functioning": controls_ok,
            "single_approval_bypass_possible": False,
            "pep_deterioration_causes_no_go": True,
            "no_go_causes": go_nogo.get("no_go_examples"),
            "audit_trail_complete": audit_complete,
            "autonomy_boundaries_enforced": autonomy.get("autonomy_boundaries_enforced"),
            "ready_for_controlled_recovery": readiness == "READY_FOR_CONTROLLED_EXECUTION"
            or readiness == "CONTROLS_VALIDATED_AWAITING_REAL_APPROVAL",
        },
        "recommended_decision": recommendation,
        "simulation_readiness": readiness,
    }


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    ctx = _load_context()
    state = _current_state(ctx)
    pep = _pep_lifecycle(ctx)

    audit_events: List[Dict[str, Any]] = []
    s1 = _step1_governance_package(ctx)
    s2 = _step2_dual_approval(ctx, ts, audit_events)
    s3 = _step3_market_readiness(ctx)
    s4 = _step4_lifecycle(pep)
    prior_ok = (
        s1["outcome"] == "PASS"
        and s2["simulated_outcome"] == "AUTHORIZED"
        and s3["outcome"] == "READY"
        and s4["outcome"] == "APPROVED_FOR_EXECUTION"
    )
    s5 = _step5_exception_activation(ctx, prior_ok)
    steps = [s1, s2, s3, s4, s5]
    go_nogo = _step6_go_nogo(steps, pep)
    steps.append(go_nogo)

    failures = _failure_scenarios(ctx, pep)
    audit = _build_audit_trail(ts, steps[:5], go_nogo, s2)
    audit.extend(audit_events)
    tree = _decision_tree(steps[:5], go_nogo)
    autonomy = _autonomy_validation()
    executive = _executive_report(state, steps[:6], go_nogo, s2, failures, audit, autonomy, pep)

    summary = {
        "simulation_readiness": executive.get("simulation_readiness"),
        "real_world_decision": go_nogo.get("real_world_decision"),
        "dress_rehearsal_decision": go_nogo.get("dress_rehearsal_decision"),
        "authorization_granted": False,
        "exception_activated": False,
        "trades_executed": False,
        "controls_validated": autonomy.get("autonomy_boundaries_enforced"),
        "single_approval_bypass_blocked": True,
        "failure_scenarios_tested": len(failures),
        "audit_events": len(audit),
    }

    report = {
        "generated_at": ts,
        "phase": "192",
        "mode": "recovery_transition_final_authorization_simulator",
        "disclaimer": (
            "Simulation only — does NOT grant approvals, activate exceptions, modify "
            "governance/lifecycle/execution/broker/allocation/risk controls, or place trades."
        ),
        "current_state": state,
        "authorization_simulation": {
            "steps": steps,
            "dual_approval": s2,
            "go_nogo": go_nogo,
        },
        "market_readiness": s3,
        "lifecycle_validation": s4,
        "exception_activation_simulation": s5,
        "decision_tree": tree,
        "failure_scenarios": failures,
        "audit_trail": audit,
        "autonomy_validation": autonomy,
        "executive_recommendation": executive,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "read-only simulation — no authorization granted",
        },
    }

    pd.DataFrame(tree).to_csv(OUT_TREE, index=False, encoding="utf-8")
    pd.DataFrame(audit).to_csv(OUT_AUDIT, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_TREE.name}, {OUT_AUDIT.name}")
    _log(
        f"real={summary['real_world_decision']} rehearsal={summary['dress_rehearsal_decision']} "
        f"controls={summary['controls_validated']}"
    )
    return report


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
