"""
TRITON Phase 181 — Recovery Basket Exception Governance Framework (design only).

Designs the governance control framework required to safely approve, monitor,
audit, and revoke PATH_B Recovery Basket Exceptions. Does NOT implement exceptions.

SAFETY: framework design only — no governance, lifecycle, execution, broker,
allocation, or risk control changes. No trades or orders.

Outputs (data/results/):
    recovery_basket_exception_governance.json
    recovery_basket_exception_controls.csv
    recovery_basket_exception_audit_model.csv

Run:
    python -m services.recovery_basket_exception_governance
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from services.concentration_governance_analysis import (
    _concentration_score,
    _load_positions,
    _read_json,
)
from services.concentration_recovery_simulator import CPS_GREEN, CPS_YELLOW
from services.pnl_ledger_builder import _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "recovery_basket_exception_governance.json"
OUT_CONTROLS = RESULTS / "recovery_basket_exception_controls.csv"
OUT_AUDIT = RESULTS / "recovery_basket_exception_audit_model.csv"

REGISTRY_CORE = ["HD", "XLK"]
REGISTRY_EXTENDED = ["HD", "XLK", "BITO"]
REGISTRY_FUTURE = ["HD", "XLK", "BITO", "IWM", "XLV"]

APPROVAL_MODELS = [
    "OPERATOR",
    "RISK_OFFICE",
    "GOVERNANCE",
    "DUAL_APPROVAL",
    "TRIPLE_APPROVAL",
]

ABUSE_LEVELS = {"LOW", "MODERATE", "HIGH", "CRITICAL"}
LIMIT_TIERS = {"CONSERVATIVE", "MODERATE", "AGGRESSIVE"}


def _log(msg: str) -> None:
    print(f"[recovery_basket_exception_governance] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _first_json(*paths: Path) -> Tuple[Path, Dict[str, Any]]:
    for p in paths:
        if p.is_file():
            return p, _read_json(p)
    return paths[0], {}


def _load_context() -> Dict[str, Any]:
    phase180_path, phase180 = _first_json(RESULTS / "governance_recovery_exception_analysis.json")
    auth_path, auth = _first_json(RESULTS / "governance_authorization.json")
    cpe_path, cpe = _first_json(RESULTS / "capital_preservation_escalation.json")
    cpi_path, cpi = _first_json(RESULTS / "capital_preservation_intelligence.json")
    basket_path, basket = _first_json(RESULTS / "diversification_recovery_basket.json")
    thresh_path, thresh = _first_json(RESULTS / "governance_recovery_thresholds.json")

    positions = _load_positions()
    _, largest_pct, _ = _concentration_score(positions)
    current_cps = int(cpi.get("capital_preservation_score") or 65)

    yellow_weights = (
        (thresh.get("executive_report") or {}).get("minimum_yellow_recovery_plan") or {}
    ).get("weights") or {"QQQ": 60.0, "HD": 15.0, "XLK": 25.0}

    path_b = next(
        (p for p in (phase180.get("exception_paths") or []) if p.get("exception_path") == "PATH_B"),
        {},
    )

    return {
        "paths": {
            "phase_180_analysis": str(phase180_path),
            "governance_authorization": str(auth_path),
            "capital_preservation_escalation": str(cpe_path),
            "capital_preservation_intelligence": str(cpi_path),
            "diversification_recovery_basket": str(basket_path),
            "governance_recovery_thresholds": str(thresh_path),
        },
        "phase180": phase180,
        "auth": auth,
        "cpe": cpe,
        "cpi": cpi,
        "basket": basket,
        "thresh": thresh,
        "path_b": path_b,
        "current_cps": current_cps,
        "largest_pct": largest_pct,
        "escalation": str(cpe.get("escalation_state") or "ORANGE"),
        "yellow_weights": yellow_weights,
        "deadlock": (phase180.get("deadlock_analysis") or {}),
    }


def _score_label(score: float) -> str:
    if score >= 85:
        return "VERY_HIGH"
    if score >= 70:
        return "HIGH"
    if score >= 50:
        return "MODERATE"
    return "LOW"


def _approval_models() -> List[Dict[str, Any]]:
    specs = {
        "OPERATOR": {
            "approvers": ["OPERATOR"],
            "description": "Single operator approval for recovery basket exception",
            "approval_strength": 55,
            "operational_speed": 95,
            "governance_integrity": 45,
            "abuse_risk": 65,
        },
        "RISK_OFFICE": {
            "approvers": ["RISK_OFFICE"],
            "description": "Risk office sole approval with concentration mandate",
            "approval_strength": 75,
            "operational_speed": 70,
            "governance_integrity": 78,
            "abuse_risk": 35,
        },
        "GOVERNANCE": {
            "approvers": ["GOVERNANCE_BOARD"],
            "description": "Governance board approval — highest integrity, slowest",
            "approval_strength": 90,
            "operational_speed": 40,
            "governance_integrity": 92,
            "abuse_risk": 20,
        },
        "DUAL_APPROVAL": {
            "approvers": ["OPERATOR", "RISK_OFFICE"],
            "description": "Operator initiates; Risk Office confirms concentration-reducing scope",
            "approval_strength": 82,
            "operational_speed": 75,
            "governance_integrity": 88,
            "abuse_risk": 25,
        },
        "TRIPLE_APPROVAL": {
            "approvers": ["OPERATOR", "RISK_OFFICE", "GOVERNANCE"],
            "description": "Operator + Risk Office + Governance for extended or repeat exceptions",
            "approval_strength": 95,
            "operational_speed": 35,
            "governance_integrity": 96,
            "abuse_risk": 12,
        },
    }
    rows = []
    for model in APPROVAL_MODELS:
        s = specs[model]
        composite = (
            0.30 * s["approval_strength"]
            + 0.25 * s["governance_integrity"]
            + 0.25 * (100 - s["abuse_risk"])
            + 0.20 * s["operational_speed"]
        )
        rows.append(
            {
                "model": model,
                "approvers": s["approvers"],
                "description": s["description"],
                "approval_strength": s["approval_strength"],
                "approval_strength_label": _score_label(s["approval_strength"]),
                "operational_speed": s["operational_speed"],
                "operational_speed_label": _score_label(s["operational_speed"]),
                "governance_integrity": s["governance_integrity"],
                "governance_integrity_label": _score_label(s["governance_integrity"]),
                "abuse_risk": s["abuse_risk"],
                "abuse_risk_label": _score_label(100 - s["abuse_risk"]),
                "composite_score": round(composite, 1),
                "recommended_for_path_b": model in ("DUAL_APPROVAL", "RISK_OFFICE"),
            }
        )
    rows.sort(key=lambda r: -r["composite_score"])
    return rows


def _authorization_scope() -> Dict[str, Any]:
    scopes = [
        {
            "scope_type": "basket_scoped",
            "description": "Exception applies only to registry-listed recovery basket symbols",
            "safest": True,
            "integrity_score": 95,
            "flexibility_score": 60,
            "recommended": True,
        },
        {
            "scope_type": "ticker_scoped",
            "description": "Per-ticker exception tokens with explicit symbol binding",
            "safest": False,
            "integrity_score": 88,
            "flexibility_score": 75,
            "recommended": False,
            "note": "Use as sub-scope within basket exception, not standalone",
        },
        {
            "scope_type": "allocation_scoped",
            "description": "Cap max weight per symbol per minimum_yellow_recovery_plan",
            "safest": False,
            "integrity_score": 85,
            "flexibility_score": 70,
            "recommended": True,
            "note": "Mandatory companion constraint to basket scope",
        },
        {
            "scope_type": "time_scoped",
            "description": "Exception expires after fixed window (48–72h default)",
            "safest": True,
            "integrity_score": 92,
            "flexibility_score": 55,
            "recommended": True,
        },
    ]
    return {
        "recommended_model": "basket_scoped + allocation_scoped + time_scoped",
        "safest_combination": "Basket registry match AND max-weight caps AND time-box expiry",
        "scope_options": scopes,
        "authorization_token_fields": [
            "exception_id",
            "approved_basket_version",
            "approved_symbols",
            "max_weight_per_symbol",
            "max_total_diversification_pct",
            "valid_from_utc",
            "valid_until_utc",
            "approver_ids",
            "pre_exception_cps",
            "target_cps_yellow",
            "revocation_triggers",
        ],
    }


def _recovery_basket_registry(ctx: Dict[str, Any]) -> Dict[str, Any]:
    basket_v1 = (ctx["basket"].get("recovery_basket_v1") or {}).get("members") or []
    return {
        "registry_name": "APPROVED_RECOVERY_BASKET_REGISTRY",
        "version": "v1.0-design",
        "core_members": [
            {
                "symbol": "HD",
                "status": "ACTIVE",
                "sector": "Consumer",
                "role": "PRIMARY_DIVERSIFIER",
                "max_weight_pct": ctx["yellow_weights"].get("HD", 15.0),
                "requires_lifecycle_buy": True,
            },
            {
                "symbol": "XLK",
                "status": "ACTIVE",
                "sector": "Tech",
                "role": "SECONDARY_DIVERSIFIER",
                "max_weight_pct": ctx["yellow_weights"].get("XLK", 25.0),
                "requires_lifecycle_buy": True,
            },
        ],
        "conditional_members": [
            {
                "symbol": "BITO",
                "status": "CONDITIONAL",
                "promotion_requirement": "OPEN_NEW lifecycle + confidence >= 0.50",
                "max_weight_pct": 25.0,
                "role": "GREEN_RECOVERY",
            },
        ],
        "future_candidates": [
            {
                "symbol": s,
                "status": "CANDIDATE",
                "note": "Governance review required before addition",
            }
            for s in REGISTRY_FUTURE
            if s not in REGISTRY_EXTENDED
        ],
        "registry_processes": {
            "maintenance": "Quarterly review by Risk Office; versioned registry artifact",
            "approval": "Dual approval (Risk Office + Governance) for registry version changes",
            "addition": "Feasibility analysis + CPS simulation + abuse review before ACTIVE status",
            "removal": "Immediate suspension on lifecycle/risk deterioration; formal removal via registry version bump",
        },
        "basket_v1_reference": basket_v1,
        "eligible_for_exception_now": REGISTRY_CORE,
    }


def _exception_limits() -> Dict[str, Any]:
    tiers = {
        "CONSERVATIVE": {
            "max_allocation_increase_pct_per_symbol": 15.0,
            "max_total_diversification_pct": 40.0,
            "max_concentration_reduction_pct_per_cycle": 40.0,
            "max_exception_duration_hours": 48,
            "max_active_exceptions": 1,
            "max_exceptions_per_30_days": 2,
            "auto_expiry": "valid_until_utc OR CPS >= YELLOW threshold",
            "recommended": True,
        },
        "MODERATE": {
            "max_allocation_increase_pct_per_symbol": 25.0,
            "max_total_diversification_pct": 50.0,
            "max_concentration_reduction_pct_per_cycle": 50.0,
            "max_exception_duration_hours": 72,
            "max_active_exceptions": 1,
            "max_exceptions_per_30_days": 3,
            "auto_expiry": "valid_until_utc OR CPS >= YELLOW OR escalation change",
            "recommended": False,
        },
        "AGGRESSIVE": {
            "max_allocation_increase_pct_per_symbol": 35.0,
            "max_total_diversification_pct": 65.0,
            "max_concentration_reduction_pct_per_cycle": 65.0,
            "max_exception_duration_hours": 96,
            "max_active_exceptions": 2,
            "max_exceptions_per_30_days": 5,
            "auto_expiry": "valid_until_utc only",
            "recommended": False,
        },
    }
    return {
        "recommended_tier": "CONSERVATIVE",
        "tiers": tiers,
        "rationale": (
            "CONSERVATIVE tier aligns with minimum_yellow_recovery_plan (HD 15%, XLK 25%) "
            "and limits exception duration to 48h with single active exception"
        ),
    }


def _revocation_rules() -> Dict[str, Any]:
    rules = [
        {
            "trigger": "CPS_REACHES_YELLOW",
            "condition": f"CPS >= {CPS_YELLOW}",
            "action": "AUTO_REVOKE",
            "priority": 1,
            "mandatory": True,
        },
        {
            "trigger": "CPS_REACHES_GREEN",
            "condition": f"CPS >= {CPS_GREEN}",
            "action": "AUTO_REVOKE_AND_ARCHIVE",
            "priority": 1,
            "mandatory": True,
        },
        {
            "trigger": "ESCALATION_IMPROVES",
            "condition": "Escalation transitions ORANGE → YELLOW or YELLOW → GREEN",
            "action": "AUTO_REVOKE",
            "priority": 2,
            "mandatory": True,
        },
        {
            "trigger": "RECOVERY_OBJECTIVE_ACHIEVED",
            "condition": "Post-trade simulation confirms governance_authorized=true",
            "action": "AUTO_REVOKE",
            "priority": 2,
            "mandatory": True,
        },
        {
            "trigger": "ASSET_LEAVES_REGISTRY",
            "condition": "Symbol removed from ACTIVE registry or lifecycle != BUY/OPEN_NEW",
            "action": "IMMEDIATE_REVOKE_SYMBOL",
            "priority": 1,
            "mandatory": True,
        },
        {
            "trigger": "RISK_EVENT",
            "condition": "Watchdog CRITICAL, sector hard block, or FORCE_EXIT risk flag",
            "action": "IMMEDIATE_FULL_REVOKE",
            "priority": 1,
            "mandatory": True,
        },
        {
            "trigger": "LIFECYCLE_DETERIORATION",
            "condition": "effective_stance degrades from BUY to WAIT/HOLD",
            "action": "REVOKE_SYMBOL",
            "priority": 2,
            "mandatory": True,
        },
        {
            "trigger": "EXCEPTION_EXPIRY",
            "condition": "valid_until_utc elapsed",
            "action": "AUTO_REVOKE",
            "priority": 3,
            "mandatory": True,
        },
        {
            "trigger": "MANUAL_REVOCATION",
            "condition": "Risk Office or Governance revokes via control plane",
            "action": "IMMEDIATE_FULL_REVOKE",
            "priority": 1,
            "mandatory": False,
        },
        {
            "trigger": "CONCENTRATION_INCREASE_DETECTED",
            "condition": "Pre-trade simulation shows largest weight increase",
            "action": "BLOCK_AND_REVOKE",
            "priority": 1,
            "mandatory": True,
        },
    ]
    return {
        "automatic_revocation_triggers": [r for r in rules if r["action"].startswith("AUTO")],
        "immediate_revocation_triggers": [
            r for r in rules if "IMMEDIATE" in r["action"] or r["action"] == "BLOCK_AND_REVOKE"
        ],
        "all_rules": rules,
        "default_on_expiry": "AUTO_REVOKE with audit log entry and governance notification",
    }


def _audit_requirements() -> Dict[str, Any]:
    return {
        "mandatory_logs": [
            "approval_request",
            "approval_decision",
            "exception_token_issued",
            "pre_exception_cps_snapshot",
            "pre_exception_concentration_snapshot",
            "execution_attempt",
            "execution_outcome",
            "post_exception_cps_snapshot",
            "post_exception_concentration_snapshot",
            "recovery_progress_update",
            "revocation_event",
            "registry_version_reference",
        ],
        "retention_policy": {
            "approval_and_revocation": "7 years",
            "execution_events": "7 years",
            "monitoring_snapshots": "2 years",
            "registry_versions": "indefinite",
        },
        "integrity_controls": [
            "Append-only audit ledger",
            "Exception ID cross-reference to paper_execution_audit",
            "Immutable registry version hash on each token",
            "Pre/post CPS mandatory delta field",
        ],
        "reporting_cadence": {
            "real_time": ["revocation", "risk_event", "concentration_increase_block"],
            "daily": ["recovery_progress", "exception_status"],
            "weekly": ["exception_effectiveness", "abuse_vector_review"],
        },
    }


def _audit_model_rows() -> List[Dict[str, Any]]:
    return [
        {
            "event_type": "EXCEPTION_APPROVAL_REQUEST",
            "required_fields": "exception_id,requestor,requested_symbols,pre_cps,justification,timestamp",
            "retention_requirement": "7 years",
            "severity": "INFO",
        },
        {
            "event_type": "EXCEPTION_APPROVAL_GRANTED",
            "required_fields": "exception_id,approver_id,approval_model,token_hash,valid_until,approved_symbols",
            "retention_requirement": "7 years",
            "severity": "WARNING",
        },
        {
            "event_type": "EXCEPTION_APPROVAL_DENIED",
            "required_fields": "exception_id,approver_id,denial_reason,timestamp",
            "retention_requirement": "7 years",
            "severity": "INFO",
        },
        {
            "event_type": "EXCEPTION_TOKEN_ISSUED",
            "required_fields": "exception_id,registry_version,max_weights,valid_from,valid_until,revocation_triggers",
            "retention_requirement": "7 years",
            "severity": "WARNING",
        },
        {
            "event_type": "PRE_EXCEPTION_SNAPSHOT",
            "required_fields": "exception_id,cps,escalation,largest_weight_pct,holdings_hash",
            "retention_requirement": "2 years",
            "severity": "INFO",
        },
        {
            "event_type": "EXECUTION_ATTEMPT",
            "required_fields": "exception_id,symbol,side,qty,simulation_pass,concentration_delta",
            "retention_requirement": "7 years",
            "severity": "WARNING",
        },
        {
            "event_type": "EXECUTION_OUTCOME",
            "required_fields": "exception_id,symbol,status,order_id,fill_qty,post_cps",
            "retention_requirement": "7 years",
            "severity": "WARNING",
        },
        {
            "event_type": "POST_EXCEPTION_SNAPSHOT",
            "required_fields": "exception_id,cps,escalation,largest_weight_pct,cps_delta,recovery_pct",
            "retention_requirement": "2 years",
            "severity": "INFO",
        },
        {
            "event_type": "RECOVERY_PROGRESS",
            "required_fields": "exception_id,cps,target_cps,days_elapsed,effectiveness_score",
            "retention_requirement": "2 years",
            "severity": "INFO",
        },
        {
            "event_type": "EXCEPTION_REVOKED",
            "required_fields": "exception_id,revocation_trigger,revoked_by,final_cps,timestamp",
            "retention_requirement": "7 years",
            "severity": "CRITICAL",
        },
        {
            "event_type": "REGISTRY_VERSION_CHANGE",
            "required_fields": "registry_version,added_symbols,removed_symbols,approver_ids",
            "retention_requirement": "indefinite",
            "severity": "WARNING",
        },
        {
            "event_type": "ABUSE_BLOCK",
            "required_fields": "exception_id,block_reason,symbol,attempted_weight,simulation_result",
            "retention_requirement": "7 years",
            "severity": "CRITICAL",
        },
    ]


def _monitoring_requirements() -> Dict[str, Any]:
    return {
        "tracked_metrics": [
            {"metric": "pre_exception_cps", "frequency": "on_issue", "alert_threshold": None},
            {
                "metric": "post_exception_cps",
                "frequency": "post_trade",
                "alert_threshold": f"< {CPS_YELLOW} after 72h",
            },
            {
                "metric": "concentration_reduction_pct",
                "frequency": "post_trade",
                "alert_threshold": "< 5% after execution",
            },
            {
                "metric": "recovery_progress_score",
                "frequency": "daily",
                "alert_threshold": "< 25 after 7 days",
            },
            {
                "metric": "governance_status",
                "frequency": "on_change",
                "alert_threshold": "BLOCKED after YELLOW reached",
            },
            {
                "metric": "exception_effectiveness",
                "frequency": "weekly",
                "alert_threshold": "LOW for 2 consecutive weeks",
            },
            {
                "metric": "largest_position_weight_pct",
                "frequency": "continuous",
                "alert_threshold": "increase vs pre-exception",
            },
            {
                "metric": "active_exception_count",
                "frequency": "continuous",
                "alert_threshold": "> max_active_exceptions",
            },
        ],
        "dashboards": [
            "Recovery Basket Exception Control Panel",
            "CPS pre/post exception timeline",
            "Concentration reduction tracker",
            "Revocation event log",
        ],
        "escalation_path": "Monitoring alert → Risk Office review → optional Governance notification → manual revoke",
    }


def _abuse_analysis() -> Dict[str, Any]:
    vectors = [
        {
            "question": "Could exception become a loophole?",
            "assessment": "Without basket registry + time-box + revocation, yes. Framework closes loop with mandatory auto-revoke.",
            "risk_level": "MODERATE",
            "controls": ["Basket registry", "Time scope", "Auto-revocation", "Dual approval"],
        },
        {
            "question": "Could concentration increase?",
            "assessment": "Pre-trade simulation must show largest-weight decrease; CONSERVATIVE caps prevent overweight adds.",
            "risk_level": "LOW",
            "controls": ["Concentration monotonic decrease check", "Max weight caps"],
        },
        {
            "question": "Could governance be bypassed?",
            "assessment": "Exception token is additive — standard governance re-engages after revocation or expiry.",
            "risk_level": "LOW",
            "controls": ["Default-deny outside token scope", "No permanent rule changes"],
        },
        {
            "question": "Could unauthorized assets enter?",
            "assessment": "Only ACTIVE registry symbols permitted; registry version hash bound to token.",
            "risk_level": "LOW",
            "controls": ["Registry version binding", "Symbol allowlist"],
        },
        {
            "question": "Could allocation discipline weaken?",
            "assessment": "Allocation-scoped max weights mirror minimum_yellow_recovery_plan; no open-ended adds.",
            "risk_level": "MODERATE",
            "controls": ["Max diversification pct", "Per-cycle limits", "Audit trail"],
        },
    ]
    levels = [v["risk_level"] for v in vectors]
    overall = "MODERATE" if "MODERATE" in levels else "LOW"
    if "HIGH" in levels:
        overall = "HIGH"
    if "CRITICAL" in levels:
        overall = "CRITICAL"

    return {
        "overall_abuse_risk": overall,
        "vectors": vectors,
        "residual_risk_with_framework": "LOW",
        "governance_integrity_preserved": True,
    }


def _controls_catalog(
    approval_models: List[Dict[str, Any]],
    limits: Dict[str, Any],
    revocation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    conservative = limits["tiers"]["CONSERVATIVE"]
    controls = [
        {
            "control_name": "Dual Approval (Operator + Risk Office)",
            "control_type": "approval",
            "control_strength": 88,
            "risk_reduction": 75,
            "implementation_priority": "P0",
        },
        {
            "control_name": "Approved Recovery Basket Registry",
            "control_type": "authorization_scope",
            "control_strength": 95,
            "risk_reduction": 85,
            "implementation_priority": "P0",
        },
        {
            "control_name": "Time-Boxed Exception Token (48h)",
            "control_type": "duration",
            "control_strength": 92,
            "risk_reduction": 80,
            "implementation_priority": "P0",
        },
        {
            "control_name": "Pre-Trade Concentration Decrease Simulation",
            "control_type": "execution_gate",
            "control_strength": 90,
            "risk_reduction": 88,
            "implementation_priority": "P0",
        },
        {
            "control_name": "Max Weight Caps (CONSERVATIVE tier)",
            "control_type": "allocation_limit",
            "control_strength": 85,
            "risk_reduction": 70,
            "implementation_priority": "P0",
        },
        {
            "control_name": "Auto-Revoke on CPS >= YELLOW",
            "control_type": "revocation",
            "control_strength": 93,
            "risk_reduction": 82,
            "implementation_priority": "P0",
        },
        {
            "control_name": "Auto-Revoke on Lifecycle Deterioration",
            "control_type": "revocation",
            "control_strength": 80,
            "risk_reduction": 65,
            "implementation_priority": "P1",
        },
        {
            "control_name": "Append-Only Exception Audit Ledger",
            "control_type": "audit",
            "control_strength": 88,
            "risk_reduction": 60,
            "implementation_priority": "P1",
        },
        {
            "control_name": "Single Active Exception Limit",
            "control_type": "rate_limit",
            "control_strength": 78,
            "risk_reduction": 55,
            "implementation_priority": "P1",
        },
        {
            "control_name": "Registry Version Hash on Token",
            "control_type": "integrity",
            "control_strength": 86,
            "risk_reduction": 72,
            "implementation_priority": "P1",
        },
        {
            "control_name": "Triple Approval for Repeat Exceptions",
            "control_type": "approval",
            "control_strength": 96,
            "risk_reduction": 90,
            "implementation_priority": "P2",
        },
        {
            "control_name": "Weekly Exception Effectiveness Review",
            "control_type": "monitoring",
            "control_strength": 70,
            "risk_reduction": 45,
            "implementation_priority": "P2",
        },
    ]
    best_approval = next(
        (m for m in approval_models if m["model"] == "DUAL_APPROVAL"), approval_models[0]
    )
    controls[0]["control_strength"] = best_approval["governance_integrity"]
    return controls


def _executive_recommendation(
    ctx: Dict[str, Any],
    approval_models: List[Dict[str, Any]],
    scope: Dict[str, Any],
    limits: Dict[str, Any],
    revocation: Dict[str, Any],
    audit: Dict[str, Any],
    abuse: Dict[str, Any],
    registry: Dict[str, Any],
) -> Dict[str, Any]:
    recommended_model = "DUAL_APPROVAL"
    dual = next((m for m in approval_models if m["model"] == recommended_model), approval_models[0])
    conservative = limits["tiers"]["CONSERVATIVE"]

    return {
        "report_type": "RECOVERY_BASKET_EXCEPTION_GOVERNANCE_REPORT",
        "recommended_governance_model": recommended_model,
        "approval_authority": {
            "primary": "OPERATOR (initiate)",
            "confirming": "RISK_OFFICE (approve)",
            "escalation": "GOVERNANCE (repeat or extended exceptions)",
            "model_scores": dual,
        },
        "authorization_scope": scope["recommended_model"],
        "exception_limits_tier": limits["recommended_tier"],
        "exception_limits_summary": conservative,
        "revocation_summary": [r["trigger"] for r in revocation["all_rules"] if r.get("mandatory")],
        "audit_summary": audit["mandatory_logs"],
        "abuse_risk": abuse["overall_abuse_risk"],
        "residual_abuse_risk_with_controls": abuse["residual_risk_with_framework"],
        "governance_integrity_assessment": {
            "preserved": True,
            "score": dual["governance_integrity"],
            "rationale": (
                "Exception is scoped, time-boxed, revocable, and auditable. "
                "Standard ORANGE block remains default for all non-registry actions."
            ),
        },
        "executive_questions": {
            "who_can_approve": "OPERATOR initiates; RISK_OFFICE must confirm (DUAL_APPROVAL). "
            "GOVERNANCE required for repeat exceptions within 30 days.",
            "which_assets_qualify": f"ACTIVE registry: {', '.join(registry['eligible_for_exception_now'])}. "
            "BITO conditional on OPEN_NEW promotion.",
            "how_long_active": f"{conservative['max_exception_duration_hours']}h maximum (CONSERVATIVE tier)",
            "how_revoked": "Auto-revoke on CPS>=YELLOW, escalation improvement, expiry, lifecycle/risk events",
            "how_audited": "Append-only ledger with 12 mandatory event types; 7-year retention for approvals/executions",
            "governance_integrity_preserved": True,
            "abuse_prevention": "Basket registry + dual approval + simulation gate + auto-revocation",
        },
        "recommended_approach": (
            "Adopt PATH_B with DUAL_APPROVAL governance: Operator requests, Risk Office approves "
            "against APPROVED_RECOVERY_BASKET_REGISTRY v1 (HD, XLK). Apply CONSERVATIVE limits "
            "(48h, single active exception, max HD 15% / XLK 25%). Mandatory pre/post CPS audit. "
            "Auto-revoke when CPS reaches YELLOW. Do NOT implement in this phase."
        ),
        "phase_180_reference": {
            "deadlock": ctx["deadlock"].get("deadlock_classification"),
            "path_b_safety": "SAFE",
            "recovery_without_exception": False,
        },
    }


def run_framework() -> Dict[str, Any]:
    ctx = _load_context()
    ts = _now_iso()

    approval_models = _approval_models()
    authorization_scope = _authorization_scope()
    registry = _recovery_basket_registry(ctx)
    exception_limits = _exception_limits()
    revocation_rules = _revocation_rules()
    audit_requirements = _audit_requirements()
    monitoring_requirements = _monitoring_requirements()
    abuse_analysis = _abuse_analysis()
    controls = _controls_catalog(approval_models, exception_limits, revocation_rules)
    executive = _executive_recommendation(
        ctx,
        approval_models,
        authorization_scope,
        exception_limits,
        revocation_rules,
        audit_requirements,
        abuse_analysis,
        registry,
    )

    framework_summary = {
        "framework_name": "Recovery Basket Exception Governance Framework (PATH_B)",
        "design_phase": "181",
        "implements_exception": False,
        "recommended_approval_model": executive["recommended_governance_model"],
        "recommended_limits_tier": exception_limits["recommended_tier"],
        "eligible_assets_now": registry["eligible_for_exception_now"],
        "max_duration_hours": exception_limits["tiers"]["CONSERVATIVE"][
            "max_exception_duration_hours"
        ],
        "governance_integrity_score": executive["governance_integrity_assessment"]["score"],
        "abuse_risk": abuse_analysis["overall_abuse_risk"],
        "deadlock_context": ctx["deadlock"].get("deadlock_classification"),
    }

    report = {
        "generated_at": ts,
        "phase": "181",
        "mode": "recovery_basket_exception_governance_design",
        "disclaimer": (
            "Framework design only — governs the exception conceptually. "
            "Does not implement exceptions, modify rules, or place orders."
        ),
        "framework_summary": framework_summary,
        "current_state": {
            "cps": ctx["current_cps"],
            "escalation": ctx["escalation"],
            "largest_concentration_pct": ctx["largest_pct"],
            "governance_authorized": ctx["auth"].get("governance_authorized"),
            "deadlock_classification": ctx["deadlock"].get("deadlock_classification"),
            "path_b_reference": ctx["path_b"],
            "input_paths": ctx["paths"],
        },
        "approval_models": approval_models,
        "authorization_scope": authorization_scope,
        "recovery_basket_registry": registry,
        "exception_limits": exception_limits,
        "revocation_rules": revocation_rules,
        "audit_requirements": audit_requirements,
        "monitoring_requirements": monitoring_requirements,
        "abuse_analysis": abuse_analysis,
        "controls_catalog": controls,
        "executive_recommendation": executive,
        "certification": {
            "status": "CERTIFIED",
            "safety": "design-only — no governance, lifecycle, execution, or broker modifications",
        },
    }

    pd.DataFrame(controls).to_csv(OUT_CONTROLS, index=False, encoding="utf-8")
    pd.DataFrame(_audit_model_rows()).to_csv(OUT_AUDIT, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_CONTROLS.name}, {OUT_AUDIT.name}")
    _log(
        f"model={framework_summary['recommended_approval_model']} "
        f"tier={framework_summary['recommended_limits_tier']} "
        f"integrity={framework_summary['governance_integrity_score']}"
    )
    return report


def main() -> None:
    run_framework()


if __name__ == "__main__":
    main()
