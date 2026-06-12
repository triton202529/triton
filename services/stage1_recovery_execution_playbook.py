"""
TRITON Phase 191 — Stage 1 Recovery Execution Governance Playbook (read-only).

Defines the operational governance playbook for Stage 1 recovery transition
(QQQ 100% -> 90% + PEP 10%) under human supervision. Does NOT authorize
execution, activate exceptions, or submit trades.

SAFETY: governance playbook only — no rule changes, no broker interaction,
no authorization grants.

Outputs (data/results/):
    stage1_governance_playbook.json
    stage1_execution_checklist.csv
    stage1_failure_scenarios.csv
    stage1_audit_requirements.csv

Run:
    python -m services.stage1_recovery_execution_playbook
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
from services.pnl_ledger_builder import _norm_sym, _to_float

RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
OUT_JSON = RESULTS / "stage1_governance_playbook.json"
OUT_CHECKLIST = RESULTS / "stage1_execution_checklist.csv"
OUT_FAILURES = RESULTS / "stage1_failure_scenarios.csv"
OUT_AUDIT = RESULTS / "stage1_audit_requirements.csv"

REGISTRY = ["PEP", "XLE", "ABBV", "MA", "V"]
STAGE_1_SYMBOL = "PEP"
BUY_STANCES = {"BUY", "ADD"}


def _log(msg: str) -> None:
    print(f"[stage1_recovery_execution_playbook] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_context() -> Dict[str, Any]:
    fw_path = RESULTS / "governance_exception_framework.json"
    exception_gov = _read_json(RESULTS / "recovery_basket_exception_governance.json")
    if fw_path.is_file():
        exception_gov = {**exception_gov, **_read_json(fw_path)}

    return {
        "approval_package": _read_json(RESULTS / "recovery_exception_registry_migration.json"),
        "auth_assessment": _read_json(
            RESULTS / "recovery_transition_authorization_assessment.json"
        ),
        "transition_plan": _read_json(RESULTS / "autonomous_transition_plan.json"),
        "exception_gov": exception_gov,
        "lifecycle_eff": _read_csv(RESULTS / "signal_lifecycle_effective.csv"),
        "exec_readiness": _read_json(RESULTS / "execution_readiness.json"),
        "auth": _read_json(RESULTS / "governance_authorization.json"),
        "cpi": _read_json(RESULTS / "capital_preservation_intelligence.json"),
        "cpe": _read_json(RESULTS / "capital_preservation_escalation.json"),
        "guard": _read_json(RESULTS / "guard_snapshot.json"),
        "positions": _load_positions(),
        "stale_orders": _read_csv(RESULTS / "stale_open_orders.csv"),
        "manage_orders": _read_csv(LIVE / "manage_orders.csv"),
    }


def _current_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
    assessment = ctx["auth_assessment"].get("current_state") or {}
    approval = ctx["approval_package"].get("current_state") or {}
    positions = ctx["positions"]
    total_mv = sum(p["market_value"] for p in positions) or assessment.get("portfolio_value") or 0
    weights: Dict[str, float] = {}
    if total_mv > 0:
        for p in positions:
            sym = p["symbol"]
            weights[sym] = weights.get(sym, 0.0) + p["market_value"] / total_mv * 100.0
    if not weights:
        weights = assessment.get("weights") or approval.get("weights") or {"QQQ": 100.0}

    cps = int(
        assessment.get("current_cps")
        or approval.get("current_cps")
        or ctx["cpi"].get("capital_preservation_score")
        or 68
    )
    esc = str(
        assessment.get("escalation")
        or approval.get("escalation")
        or ctx["cpe"].get("escalation_state")
        or _escalation_from_cps(cps)
    ).upper()

    migration = ctx["approval_package"].get("registry_migration") or {}
    return {
        "weights": {k: _f(v, 2) for k, v in weights.items()},
        "current_cps": cps,
        "escalation": esc,
        "governance_status": assessment.get("governance_status") or "BLOCKED",
        "governance_authorized": assessment.get("governance_authorized") is True,
        "largest_concentration_pct": assessment.get("largest_concentration_pct") or 100.0,
        "portfolio_value": _f(total_mv),
        "authorization_status": "READY_FOR_REVIEW",
        "recovery_exception": {
            "framework_valid": True,
            "registry_updated_for_review": migration.get("migration_classification")
            == "APPROVED_FOR_REVIEW",
            "approved_basket": REGISTRY,
            "exception_activated": False,
            "stage_1_target": {"QQQ": 90.0, "PEP": 10.0},
        },
    }


def _pep_lifecycle(ctx: Dict[str, Any]) -> Dict[str, Any]:
    lc = _lifecycle_row(ctx["lifecycle_eff"], STAGE_1_SYMBOL)
    effective = str(lc.get("effective_stance") or "").upper()
    signal = str(lc.get("signal") or "").upper()
    conf = _to_float(lc.get("confidence"), 0.0)
    valid = effective in BUY_STANCES and signal != "SELL"
    return {
        "symbol": STAGE_1_SYMBOL,
        "effective_stance": effective,
        "signal": signal,
        "confidence": _f(conf, 4),
        "lifecycle_valid": valid,
        "blockers": [] if valid else [f"effective_stance={effective}", f"signal={signal}"],
    }


def _dual_approval_status(ctx: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    checklist = ctx["approval_package"].get("readiness_checklist") or []
    op_sig = any(
        c.get("approval_item") == "Operator signature" and c.get("status") == "COMPLETE"
        for c in checklist
    )
    risk_sig = any(
        c.get("approval_item") == "Risk office signature" and c.get("status") == "COMPLETE"
        for c in checklist
    )
    return op_sig and risk_sig, op_sig, risk_sig


def _stale_order_count(ctx: Dict[str, Any]) -> int:
    stale = ctx["stale_orders"]
    if not stale.empty:
        return len(stale)
    manage = ctx["manage_orders"]
    if manage.empty:
        return 0
    status_col = next((c for c in manage.columns if "status" in c.lower()), None)
    if status_col:
        return int((manage[status_col].astype(str).str.upper() == "STALE").sum())
    return 0


def _authorization_requirements(ctx: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    dual_ok, op_sig, risk_sig = _dual_approval_status(ctx)
    migration = ctx["approval_package"].get("registry_migration") or {}
    fw = ctx["exception_gov"].get("framework_summary") or {}
    limits = (ctx["exception_gov"].get("exception_limits") or {}).get("CONSERVATIVE") or {}

    pre_auth_items = [
        (
            "DUAL_APPROVAL completed",
            "GOVERNANCE",
            dual_ok,
            "Obtain Operator + Risk Office signatures",
        ),
        (
            "Operator signature recorded",
            "OPERATOR",
            op_sig,
            "Operator sign-off on Stage 1 playbook",
        ),
        (
            "Risk Office signature recorded",
            "RISK_OFFICE",
            risk_sig,
            "Risk office sign-off on exception scope",
        ),
        (
            "Exception scope matches registry",
            "REGISTRY",
            migration.get("proposed_registry") == REGISTRY,
            "Confirm PEP/XLE/ABBV/MA/V registry",
        ),
        (
            "Exception duration valid",
            "GOVERNANCE",
            bool(migration.get("max_duration_hours")),
            "Verify 48h CONSERVATIVE window",
        ),
        (
            "Exception limits valid",
            "RISK_OFFICE",
            bool(limits),
            "Verify allocation caps per CONSERVATIVE tier",
        ),
        (
            "Audit system ready",
            "AUDIT",
            bool((ctx["exception_gov"].get("audit_requirements") or {})),
            "Confirm audit event pipeline active",
        ),
    ]

    items = []
    for name, owner, ok, action in pre_auth_items:
        items.append(
            {
                "phase": "pre_authorization",
                "check_item": name,
                "owner": owner,
                "status": "PASS" if ok else "PENDING",
                "required_action": action if not ok else "Complete",
            }
        )

    return {
        "dual_approval_satisfied": dual_ok,
        "operator_signature": op_sig,
        "risk_office_signature": risk_sig,
        "exception_scope": REGISTRY,
        "stage_1_symbols": [STAGE_1_SYMBOL],
        "max_duration_hours": migration.get("max_duration_hours")
        or fw.get("max_duration_hours")
        or 48,
        "limits_tier": migration.get("limits_tier") or "CONSERVATIVE",
        "pre_authorization_checklist": items,
        "remaining_requirements": [
            r
            for r in [
                "Operator signature" if not op_sig else None,
                "Risk Office signature" if not risk_sig else None,
                "Market session open",
                "Final lifecycle revalidation",
            ]
            if r
        ],
        "authorization_granted": False,
        "stage_1_authorized": False,
    }


def _market_readiness(ctx: Dict[str, Any], pep: Dict[str, Any]) -> Dict[str, Any]:
    er = ctx["exec_readiness"]
    checks = er.get("checks") or {}
    market_open = is_market_open_et(datetime.now(timezone.utc))
    stale_n = _stale_order_count(ctx)
    guard = ctx["guard"]
    guard_blocked = bool(guard.get("blocked") or guard.get("kill_switch_active"))

    pre_market = [
        (
            "Broker connection healthy",
            checks.get("broker") is True,
            er.get("check_details", {}).get("broker", "—"),
        ),
        (
            "Paper execution authorization active",
            er.get("paper_execution_permitted") is True,
            f"mode={er.get('mode', 'PAPER_ONLY')}",
        ),
        ("No stale orders", stale_n == 0, f"stale_orders={stale_n}"),
        (
            "Lifecycle data fresh",
            checks.get("lifecycle") is True,
            er.get("check_details", {}).get("lifecycle", "—"),
        ),
        (
            "PEP remains effective BUY",
            pep.get("lifecycle_valid") is True,
            f"stance={pep.get('effective_stance')} conf={pep.get('confidence')}",
        ),
        (
            "Risk Office has no blocking events",
            checks.get("watchdog") is True and not guard_blocked,
            er.get("check_details", {}).get("watchdog", "—"),
        ),
        (
            "Market session is OPEN",
            market_open,
            f"NYSE regular session open={market_open}",
        ),
    ]

    rows = []
    for name, ok, detail in pre_market:
        rows.append(
            {
                "phase": "pre_market",
                "check_item": name,
                "owner": "OPERATIONS",
                "status": "PASS" if ok else "FAIL",
                "required_action": detail if not ok else "Complete",
            }
        )

    all_pass = all(ok for _, ok, _ in pre_market)
    return {
        "market_session_open": market_open,
        "pre_market_validation": rows,
        "pre_market_ready": all_pass,
        "broker_healthy": checks.get("broker") is True,
        "paper_execution_permitted": er.get("paper_execution_permitted") is True,
        "stale_orders_count": stale_n,
    }


def _stage1_from_plan(ctx: Dict[str, Any]) -> Dict[str, Any]:
    stages = ctx["transition_plan"].get("recommended_stages") or []
    return next((s for s in stages if s.get("stage") == 1), {})


def _execution_sequence(
    ctx: Dict[str, Any], state: Dict[str, Any], stage1: Dict[str, Any]
) -> List[Dict[str, Any]]:
    pv = state.get("portfolio_value") or 1436.64
    qqq_reduction_pct = 10.0
    pep_addition_pct = 10.0
    qqq_notional = _f(pv * qqq_reduction_pct / 100.0)
    pep_notional = _f(pv * pep_addition_pct / 100.0)

    return [
        {
            "step": 1,
            "action": "Final lifecycle refresh",
            "owner": "SYSTEM",
            "human_required": False,
            "detail": "Refresh signal_lifecycle_effective.csv and reconcile portfolio_allocation_recommendations",
            "gate": "lifecycle artifacts <= 24h old",
        },
        {
            "step": 2,
            "action": "Validate PEP still qualifies",
            "owner": "OPERATOR",
            "human_required": True,
            "detail": "Confirm PEP effective_stance=BUY/ADD, signal!=SELL, confidence >= 0.45",
            "gate": "PEP lifecycle_valid=true",
        },
        {
            "step": 3,
            "action": "Activate scoped Recovery Exception",
            "owner": "OPERATOR + RISK_OFFICE",
            "human_required": True,
            "detail": (
                "DUAL_APPROVAL recorded; activate PATH_B exception scoped to registry "
                f"{REGISTRY} for Stage 1 tranche only"
            ),
            "gate": "dual_approval_satisfied AND pre_market_ready",
            "note": "PLAYBOOK ONLY — activation not performed by this phase",
        },
        {
            "step": 4,
            "action": "Submit QQQ reduction tranche",
            "owner": "OPERATOR",
            "human_required": True,
            "detail": f"Sell QQQ to reduce concentration 100% -> 90% (~${qqq_notional} notional)",
            "gate": "exception active; market session OPEN",
            "order_type": "limit_or_market_per_policy",
            "symbol": "QQQ",
            "target_change_pct": -qqq_reduction_pct,
        },
        {
            "step": 5,
            "action": "Submit PEP acquisition tranche",
            "owner": "OPERATOR",
            "human_required": True,
            "detail": f"Buy PEP to reach 10% weight (~${pep_notional} notional)",
            "gate": "QQQ reduction submitted or queued; PEP still lifecycle-valid",
            "order_type": "limit_or_market_per_policy",
            "symbol": "PEP",
            "target_change_pct": pep_addition_pct,
        },
        {
            "step": 6,
            "action": "Monitor fills",
            "owner": "OPERATOR",
            "human_required": True,
            "detail": "Track partial fills, slippage, and order status until both tranches complete",
            "gate": "watchdog healthy; no guard block",
        },
        {
            "step": 7,
            "action": "Confirm portfolio state",
            "owner": "OPERATOR",
            "human_required": True,
            "detail": "Verify weights approximate QQQ 90% / PEP 10% (+/- 1% tolerance)",
            "gate": "both orders filled or failure protocol engaged",
            "target_state": stage1.get("target_state") or {"QQQ": 90.0, "PEP": 10.0},
        },
        {
            "step": 8,
            "action": "Calculate updated metrics",
            "owner": "SYSTEM",
            "human_required": False,
            "detail": "Recompute QQQ concentration, CPS, governance status; write audit trail",
            "gate": "post-trade verification complete",
            "expected_outcomes": {
                "qqq_concentration_pct": 90.0,
                "projected_cps": stage1.get("projected_cps") or 68,
                "projected_escalation": stage1.get("projected_escalation") or "ORANGE",
                "projected_governance": stage1.get("projected_governance_status") or "BLOCKED",
            },
        },
    ]


def _failure_scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "scenario": "PEP lifecycle deteriorates before execution",
            "severity": "CRITICAL",
            "immediate_response": "HALT Stage 1; do NOT activate exception or submit orders",
            "allowed_actions": [
                "Re-run lifecycle refresh",
                "Defer to next session",
                "Escalate to Risk Office",
            ],
            "required_approvals": ["Risk Office review before any retry"],
            "audit_requirements": [
                "Log lifecycle deterioration event",
                "Record halt decision and timestamp",
            ],
            "exception_action": "DO NOT ACTIVATE — revoke pending request if already submitted",
        },
        {
            "scenario": "QQQ order partially fills",
            "severity": "HIGH",
            "immediate_response": "Pause PEP acquisition until QQQ tranche reconciled",
            "allowed_actions": [
                "Cancel remainder or wait for fill",
                "Adjust PEP tranche to matched notional",
                "Abort and revert plan",
            ],
            "required_approvals": ["Operator decision; Risk Office if >2% drift from target"],
            "audit_requirements": [
                "Log partial fill quantities and prices",
                "Document adjusted target weights",
            ],
            "exception_action": "Exception remains active within scope; no additional symbols",
        },
        {
            "scenario": "PEP order partially fills",
            "severity": "HIGH",
            "immediate_response": "Assess drift from 10% PEP target; do not exceed registry cap",
            "allowed_actions": [
                "Complete fill with remainder order",
                "Accept partial Stage 1 completion",
                "Halt and schedule retry",
            ],
            "required_approvals": ["Operator; Risk Office if PEP weight <5% or >10%"],
            "audit_requirements": ["Log fill ratio", "Record final achieved weights"],
            "exception_action": "Revoke if PEP leaves BUY/ADD lifecycle",
        },
        {
            "scenario": "Order rejected",
            "severity": "HIGH",
            "immediate_response": "Capture rejection reason; do not retry without diagnosis",
            "allowed_actions": [
                "Diagnose broker/policy rejection",
                "Retry once with corrected parameters",
                "Abort Stage 1",
            ],
            "required_approvals": ["Operator + Risk Office for retry after policy rejection"],
            "audit_requirements": ["Log rejection code", "Record retry or abort decision"],
            "exception_action": "Suspend exception if rejection indicates policy violation",
        },
        {
            "scenario": "Market volatility spike",
            "severity": "MODERATE",
            "immediate_response": "Switch to limit orders; widen tolerance or defer",
            "allowed_actions": [
                "Pause execution until volatility subsides",
                "Reduce tranche size (not below governance minimum)",
                "Abort and wait next session",
            ],
            "required_approvals": ["Operator discretion; Risk Office if deferring >1 session"],
            "audit_requirements": ["Log VIX/spread context", "Document deferral rationale"],
            "exception_action": "Exception clock continues; monitor 48h expiry",
        },
        {
            "scenario": "Broker connectivity failure",
            "severity": "CRITICAL",
            "immediate_response": "HALT all order activity immediately",
            "allowed_actions": [
                "Verify open orders via broker portal",
                "Restore connectivity",
                "Cancel orphaned orders if unreachable",
            ],
            "required_approvals": ["Risk Office before resuming after connectivity loss"],
            "audit_requirements": [
                "Log connectivity failure window",
                "Reconcile order state post-recovery",
            ],
            "exception_action": "IMMEDIATE_SUSPEND — exception frozen until connectivity restored",
        },
        {
            "scenario": "Exception expires before completion",
            "severity": "CRITICAL",
            "immediate_response": "STOP all recovery trades; auto-revoke exception",
            "allowed_actions": [
                "Document partial completion state",
                "Request new DUAL_APPROVAL for continuation",
                "Hold portfolio at current weights",
            ],
            "required_approvals": ["New DUAL_APPROVAL required for any further tranches"],
            "audit_requirements": [
                "Log expiry event",
                "Record partial vs target weights",
                "Archive exception lifecycle",
            ],
            "exception_action": "AUTO_REVOKE at 48h or CPS milestone per framework",
        },
        {
            "scenario": "CPS does not improve as expected",
            "severity": "MODERATE",
            "immediate_response": "Continue to post-execution governance review; do not force additional trades",
            "allowed_actions": [
                "Document CPS trajectory vs projection",
                "Proceed to Stage 2 only after full governance review",
                "Escalate if CPS deteriorates",
            ],
            "required_approvals": ["Governance review before Stage 2 authorization"],
            "audit_requirements": [
                "Log post-trade CPS calculation",
                "Compare to projected_cps=68",
            ],
            "exception_action": "Revoke if CPS drops below pre-trade baseline",
        },
    ]


def _audit_requirements(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    gov_audit = ctx["exception_gov"].get("audit_requirements") or {}
    events = (
        gov_audit.get("required_events")
        or gov_audit.get("event_types")
        or [
            "exception_request",
            "dual_approval",
            "pre_trade_simulation",
            "execution",
            "revocation_event",
        ]
    )

    rows = []
    for evt in events:
        rows.append(
            {
                "audit_event": evt,
                "timing": "pre_trade" if "pre" in evt or "approval" in evt else "post_trade",
                "owner": "SYSTEM",
                "retention": "permanent",
                "required_fields": "timestamp, operator_id, exception_id, portfolio_snapshot",
            }
        )

    post_review = [
        {
            "audit_event": "post_execution_governance_review",
            "timing": "post_trade",
            "owner": "RISK_OFFICE",
            "retention": "permanent",
            "required_fields": "exception_used_correctly, limits_respected, concentration_delta",
        },
        {
            "audit_event": "stage1_completion_classification",
            "timing": "post_trade",
            "owner": "GOVERNANCE",
            "retention": "permanent",
            "required_fields": "classification, cps_before, cps_after, weights_achieved",
        },
        {
            "audit_event": "lifecycle_authority_confirmation",
            "timing": "post_trade",
            "owner": "LIFECYCLE",
            "retention": "permanent",
            "required_fields": "pep_stance_at_execution, revalidation_timestamp",
        },
        {
            "audit_event": "human_intervention_log",
            "timing": "continuous",
            "owner": "OPERATOR",
            "retention": "permanent",
            "required_fields": "intervention_type, decision, approver",
        },
    ]
    rows.extend(post_review)
    return rows


def _post_execution_review() -> Dict[str, Any]:
    criteria = [
        ("exception_used_correctly", "Exception limited to PEP tranche within registry scope"),
        ("limits_respected", "QQQ reduction <=10%, PEP addition <=10%, duration <=48h"),
        ("lifecycle_authoritative", "PEP was BUY/ADD at execution; no bypass"),
        ("risk_controls_intact", "No guard bypass; CONSERVATIVE tier enforced"),
        ("concentration_decreased", "QQQ weight reduced from 100% toward 90%"),
        ("audit_evidence_complete", "All required audit events logged with timestamps"),
    ]
    return {
        "evaluation_criteria": [{"criterion": c, "description": d} for c, d in criteria],
        "classifications": {
            "SUCCESSFUL_RECOVERY_STAGE": "Target weights achieved within tolerance; audit complete",
            "PARTIAL_COMPLETION": "One tranche incomplete but within recovery pathway",
            "FAILED_STAGE": "Execution aborted or limits breached",
            "REQUIRES_GOVERNANCE_REVIEW": "Anomaly detected; Risk Office review mandatory",
        },
    }


def _autonomy_boundaries(ctx: Dict[str, Any]) -> Dict[str, Any]:
    revocation = ctx["exception_gov"].get("revocation_rules") or {}
    auto_triggers = revocation.get("automatic_revocation_triggers") or [
        "CPS >= 75 (YELLOW milestone)",
        "CPS >= 90 (GREEN milestone)",
        "48h duration expiry",
        "lifecycle != BUY/ADD for registry symbol",
    ]
    immediate = revocation.get("immediate_revocation_triggers") or [
        "guard_or_kill_switch_active",
        "broker_connectivity_loss",
        "unauthorized_symbol_attempt",
    ]

    return {
        "triton_may_do_automatically": [
            "Refresh lifecycle artifacts on schedule",
            "Compute CPS and concentration metrics",
            "Generate pre-trade simulation reports",
            "Log audit events to accountability registry",
            "Alert on watchdog/guard triggers",
            "Detect stale orders and flag for review",
        ],
        "requires_human_approval": [
            "DUAL_APPROVAL for exception activation",
            "Final PEP lifecycle confirmation before orders",
            "Order submission (QQQ sell, PEP buy)",
            "Retry after order rejection",
            "Continuation to Stage 2",
            "Exception extension beyond 48h",
            "Any symbol outside registry scope",
        ],
        "causes_immediate_suspension": [
            "Guard or kill-switch activation",
            "Broker connectivity failure",
            "PEP lifecycle != BUY/ADD",
            "Order policy violation",
            "Unauthorized asset in tranche",
            "Dual approval revoked",
        ],
        "exception_must_be_revoked_when": auto_triggers + immediate,
        "live_execution": "BLOCKED — paper only per execution_readiness",
        "autonomous_order_submission": False,
    }


def _execution_checklist(
    auth_req: Dict[str, Any],
    market: Dict[str, Any],
    sequence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = list(auth_req.get("pre_authorization_checklist") or [])
    rows.extend(market.get("pre_market_validation") or [])
    for step in sequence:
        rows.append(
            {
                "phase": f"execution_step_{step['step']}",
                "check_item": step["action"],
                "owner": step["owner"],
                "status": "PENDING",
                "required_action": step.get("detail", ""),
            }
        )
    rows.append(
        {
            "phase": "post_execution",
            "check_item": "Post-execution governance review",
            "owner": "RISK_OFFICE",
            "status": "PENDING",
            "required_action": "Classify SUCCESSFUL_RECOVERY_STAGE / PARTIAL / FAILED / REVIEW",
        }
    )
    return rows


def _executive_report(
    state: Dict[str, Any],
    auth_req: Dict[str, Any],
    market: Dict[str, Any],
    sequence: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    audit: List[Dict[str, Any]],
    autonomy: Dict[str, Any],
    pep: Dict[str, Any],
) -> Dict[str, Any]:
    pre_auth_pass = all(
        i.get("status") == "PASS" for i in auth_req.get("pre_authorization_checklist") or []
    )
    pre_market_pass = market.get("pre_market_ready") is True

    if pre_auth_pass and pre_market_pass and pep.get("lifecycle_valid"):
        recommendation = (
            "PLAYBOOK COMPLETE — all pre-conditions documented. "
            "Obtain DUAL_APPROVAL signatures, then execute Stage 1 per sequence under operator supervision. "
            "This phase does NOT authorize execution."
        )
        readiness = "PLAYBOOK_READY"
    elif pep.get("lifecycle_valid"):
        recommendation = (
            "PLAYBOOK READY WITH GAPS — review pending authorization and market readiness items "
            "before scheduling Stage 1 execution window."
        )
        readiness = "PLAYBOOK_READY_WITH_GAPS"
    else:
        recommendation = (
            "BLOCKED — PEP lifecycle invalid; do not proceed until lifecycle recovers to BUY/ADD."
        )
        readiness = "NOT_EXECUTABLE"

    return {
        "report_type": "STAGE_1_RECOVERY_EXECUTION_GOVERNANCE_REPORT",
        "current_recovery_status": {
            "portfolio": state.get("weights"),
            "cps": state.get("current_cps"),
            "escalation": state.get("escalation"),
            "governance": state.get("governance_status"),
            "authorization": state.get("authorization_status"),
            "recovery_exception": state.get("recovery_exception"),
        },
        "approval_checklist_summary": {
            "pre_authorization_pass": pre_auth_pass,
            "pre_market_pass": pre_market_pass,
            "remaining": auth_req.get("remaining_requirements"),
        },
        "execution_timeline": {
            "path": "PATH_E",
            "stage": 1,
            "trading_cycle": "T+0",
            "target_transition": "QQQ 100% -> 90%, PEP 0% -> 10%",
            "projected_cps": 68,
            "projected_escalation": "ORANGE",
        },
        "trade_sequence_summary": [
            {"step": s["step"], "action": s["action"], "symbol": s.get("symbol")} for s in sequence
        ],
        "failure_responses_count": len(failures),
        "audit_requirements_count": len(audit),
        "autonomy_boundaries": autonomy,
        "executive_questions": {
            "what_must_happen_before_stage_1": (
                "DUAL_APPROVAL signatures, pre-market validation pass, PEP lifecycle BUY, market OPEN"
            ),
            "exact_trade_sequence": "QQQ -10% sell tranche, then PEP +10% buy tranche (Steps 4-5)",
            "market_conditions_change": "See failure_scenarios: volatility spike, session close, lifecycle drift",
            "partial_fill_handling": "Pause complementary tranche; adjust or abort per failure matrix",
            "human_intervention_required": (
                "Steps 2-7, all exception activation, all order submission, all failure responses"
            ),
            "exception_expiry": "48h CONSERVATIVE tier or CPS milestone auto-revoke",
            "audit_trail": "dual_approval, pre_trade_simulation, execution, post_execution_governance_review",
        },
        "recommended_decision": recommendation,
        "playbook_readiness": readiness,
        "stage_1_authorized": False,
        "exception_activated": False,
    }


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    ctx = _load_context()
    state = _current_state(ctx)
    pep = _pep_lifecycle(ctx)
    stage1 = _stage1_from_plan(ctx)

    auth_req = _authorization_requirements(ctx, state)
    market = _market_readiness(ctx, pep)
    sequence = _execution_sequence(ctx, state, stage1)
    failures = _failure_scenarios()
    audit = _audit_requirements(ctx)
    post_review = _post_execution_review()
    autonomy = _autonomy_boundaries(ctx)
    checklist = _execution_checklist(auth_req, market, sequence)
    executive = _executive_report(state, auth_req, market, sequence, failures, audit, autonomy, pep)

    summary = {
        "playbook_readiness": executive.get("playbook_readiness"),
        "pre_authorization_complete": auth_req.get("dual_approval_satisfied"),
        "pre_market_ready": market.get("pre_market_ready"),
        "pep_lifecycle_valid": pep.get("lifecycle_valid"),
        "execution_steps": len(sequence),
        "failure_scenarios": len(failures),
        "audit_events": len(audit),
        "stage_1_authorized": False,
        "exception_activated": False,
        "authorization_granted": False,
    }

    report = {
        "generated_at": ts,
        "phase": "191",
        "mode": "stage1_recovery_execution_playbook",
        "disclaimer": (
            "Governance playbook only — does NOT authorize Stage 1, activate exceptions, "
            "modify governance/lifecycle/execution/broker/allocation/risk controls, or place trades."
        ),
        "current_state": state,
        "authorization_requirements": auth_req,
        "market_readiness": market,
        "pep_lifecycle": pep,
        "execution_sequence": sequence,
        "failure_scenarios": failures,
        "post_execution_review": post_review,
        "audit_requirements": audit,
        "autonomy_boundaries": autonomy,
        "executive_recommendation": executive,
        "execution_checklist": checklist,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "read-only playbook — no execution authorized",
        },
    }

    pd.DataFrame(checklist).to_csv(OUT_CHECKLIST, index=False, encoding="utf-8")
    pd.DataFrame(failures).to_csv(OUT_FAILURES, index=False, encoding="utf-8")
    pd.DataFrame(audit).to_csv(OUT_AUDIT, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_CHECKLIST.name}, {OUT_FAILURES.name}, {OUT_AUDIT.name}")
    _log(
        f"readiness={summary['playbook_readiness']} "
        f"steps={summary['execution_steps']} failures={summary['failure_scenarios']}"
    )
    return report


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
