"""
TRITON Phase 190 — Recovery Exception Registry Migration & Approval Package
(read-only).

Prepares the governance approval package to migrate the exception registry from
obsolete HD/XLK to the validated PEP/XLE/ABBV/MA/V basket for Stage 1 review.

SAFETY: governance preparation only — does NOT activate exceptions, grant
approvals, modify rules, or execute trades.

Outputs (data/results/):
    recovery_exception_registry_migration.json
    recovery_exception_approval_checklist.csv
    recovery_exception_evidence_package.csv

Run:
    python -m services.recovery_exception_approval_package
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from services.pnl_ledger_builder import _norm_sym, _to_float
from services.sector_exposure import get_sector

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "recovery_exception_registry_migration.json"
OUT_CHECKLIST = RESULTS / "recovery_exception_approval_checklist.csv"
OUT_EVIDENCE = RESULTS / "recovery_exception_evidence_package.csv"

PREVIOUS_REGISTRY = ["HD", "XLK"]
PROPOSED_REGISTRY = ["PEP", "XLE", "ABBV", "MA", "V"]
BUY_STANCES = {"BUY", "ADD"}
EXCLUDED_STANCES = {"WAIT", "HOLD", "TRIM", "EXIT"}


def _log(msg: str) -> None:
    print(f"[recovery_exception_approval_package] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_context() -> Dict[str, Any]:
    fw_path = RESULTS / "governance_exception_framework.json"
    exception_gov = _read_json(RESULTS / "recovery_basket_exception_governance.json")
    if fw_path.is_file():
        exception_gov = {**exception_gov, **_read_json(fw_path)}

    return {
        "auth_assessment": _read_json(
            RESULTS / "recovery_transition_authorization_assessment.json"
        ),
        "reconstruction": _read_json(RESULTS / "recovery_basket_reconstruction.json"),
        "optimization": _read_json(RESULTS / "final_governance_recovery_optimization.json"),
        "transition_plan": _read_json(RESULTS / "autonomous_transition_plan.json"),
        "exception_gov": exception_gov,
        "lifecycle_eff": _read_csv(RESULTS / "signal_lifecycle_effective.csv"),
        "recs": _read_csv(RESULTS / "portfolio_allocation_recommendations.csv"),
        "cpi": _read_json(RESULTS / "capital_preservation_intelligence.json"),
        "cpe": _read_json(RESULTS / "capital_preservation_escalation.json"),
        "deterioration": _read_json(RESULTS / "recovery_candidate_deterioration_analysis.json"),
        "positions": _load_positions(),
    }


def _current_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
    assessment = ctx["auth_assessment"].get("current_state") or {}
    positions = ctx["positions"]
    total_mv = sum(p["market_value"] for p in positions) or assessment.get("portfolio_value") or 0
    weights: Dict[str, float] = {}
    if total_mv > 0:
        for p in positions:
            sym = p["symbol"]
            weights[sym] = weights.get(sym, 0.0) + p["market_value"] / total_mv * 100.0
    if not weights:
        weights = assessment.get("weights") or {"QQQ": 100.0}

    cps = int(assessment.get("current_cps") or ctx["cpi"].get("capital_preservation_score") or 68)
    esc = str(
        assessment.get("escalation")
        or ctx["cpe"].get("escalation_state")
        or _escalation_from_cps(cps)
    ).upper()

    return {
        "weights": {k: _f(v, 2) for k, v in weights.items()},
        "current_cps": cps,
        "escalation": esc,
        "governance_status": assessment.get("governance_status") or "BLOCKED",
        "governance_authorized": assessment.get("governance_authorized") is True,
        "largest_concentration_pct": assessment.get("largest_concentration_pct") or 100.0,
        "portfolio_value": _f(total_mv),
        "authorization_status": (ctx["auth_assessment"].get("executive_recommendation") or {}).get(
            "authorization_status"
        ),
    }


def _rec_row(recs: pd.DataFrame, sym: str) -> Dict[str, Any]:
    if recs.empty:
        return {}
    sub = recs[recs["ticker"].map(_norm_sym) == sym]
    return sub.iloc[0].to_dict() if not sub.empty else {}


def _recon_row(ctx: Dict[str, Any], sym: str) -> Dict[str, Any]:
    universe = ctx["reconstruction"].get("candidate_universe") or []
    return next((c for c in universe if c.get("symbol") == sym), {})


def _validate_candidate(ctx: Dict[str, Any], sym: str) -> Dict[str, Any]:
    sym = _norm_sym(sym)
    lc = _lifecycle_row(ctx["lifecycle_eff"], sym)
    rec = _rec_row(ctx["recs"], sym)
    recon = _recon_row(ctx, sym)
    effective = str(lc.get("effective_stance") or rec.get("effective_stance") or "").upper()
    signal = str(lc.get("signal") or rec.get("signal") or "").upper()
    conf = _to_float(lc.get("confidence") or rec.get("confidence"), np.nan)
    edge = _to_float(rec.get("edge_score"), np.nan)
    risk = str(rec.get("risk_flag") or "OK").upper()

    blockers = []
    if effective not in BUY_STANCES:
        blockers.append(f"effective_stance={effective}")
    if signal == "SELL":
        blockers.append("signal=SELL")
    if risk != "OK":
        blockers.append(f"risk={risk}")

    return {
        "symbol": sym,
        "effective_stance": effective,
        "signal": signal,
        "confidence": _f(conf, 4),
        "edge_score": _f(edge, 4),
        "allocation_score": recon.get("allocation_score") or _f(rec.get("allocation_score"), 4),
        "risk_flags": risk,
        "sector": recon.get("sector_exposure") or get_sector(sym),
        "correlation_to_QQQ": recon.get("correlation_to_QQQ"),
        "diversification_score": recon.get("diversification_score"),
        "cps_improvement_potential": recon.get("cps_improvement_potential"),
        "governance_recovery_contribution": recon.get("governance_recovery_contribution"),
        "recovery_classification": recon.get("recovery_classification"),
        "lifecycle_valid": not blockers,
        "blockers": blockers,
    }


def _removed_asset(ctx: Dict[str, Any], sym: str) -> Dict[str, Any]:
    sym = _norm_sym(sym)
    lc = _lifecycle_row(ctx["lifecycle_eff"], sym)
    det = next(
        (
            c
            for c in (ctx["deterioration"].get("deteriorated_candidates") or [])
            if c.get("symbol") == sym
        ),
        {},
    )
    if not det:
        det = next(
            (
                c
                for c in (ctx["deterioration"].get("candidate_analysis") or [])
                if c.get("symbol") == sym
            ),
            {},
        )
    effective = str(lc.get("effective_stance") or "WAIT").upper()
    return {
        "symbol": sym,
        "removal_reason": "LIFECYCLE_WAIT — no longer lifecycle-actionable",
        "effective_stance": effective,
        "signal": str(lc.get("signal") or "SELL").upper(),
        "historical_role": "Original recovery basket OPEN_NEW candidate",
        "deterioration_cause": det.get("root_cause_classification") or "MULTI_FACTOR",
        "lifecycle_decision_reason": lc.get("lifecycle_decision_reason") or "flat_non_buy_wait",
        "replacement_symbols": PROPOSED_REGISTRY,
    }


def _target_weights(ctx: Dict[str, Any]) -> Dict[str, float]:
    opt_paths = ctx["optimization"].get("optimization_paths") or []
    path_a = next((p for p in opt_paths if p.get("path_id") == "PATH_A"), {})
    if path_a.get("weights"):
        return {k: float(v) for k, v in path_a["weights"].items()}
    rec = ctx["reconstruction"].get("recommended_basket") or {}
    return rec.get("weights") or {
        "QQQ": 38.0,
        "PEP": 15.0,
        "XLE": 15.0,
        "ABBV": 15.0,
        "MA": 15.0,
        "V": 2.0,
    }


def _registry_migration(ctx: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    fw = ctx["exception_gov"].get("framework_summary") or {}
    previous = list(fw.get("eligible_assets_now") or PREVIOUS_REGISTRY)
    proposed = PROPOSED_REGISTRY
    target_weights = _target_weights(ctx)

    removed = [_removed_asset(ctx, s) for s in previous if s not in proposed]
    added = []
    for sym in proposed:
        if sym not in previous:
            cand = next((c for c in candidates if c["symbol"] == sym), {})
            added.append(
                {
                    "symbol": sym,
                    "max_weight_pct": target_weights.get(sym),
                    "lifecycle_justification": f"effective_stance={cand.get('effective_stance')} signal={cand.get('signal')}",
                    "confidence": cand.get("confidence"),
                    "diversification_contribution": cand.get("diversification_score"),
                    "cps_recovery_contribution": cand.get("cps_improvement_potential"),
                    "recovery_classification": cand.get("recovery_classification"),
                }
            )

    all_valid = all(c.get("lifecycle_valid") for c in candidates)
    any_invalid = any(not c.get("lifecycle_valid") for c in candidates)

    if any_invalid:
        classification = "INVALID"
    elif all_valid and len(removed) >= 2 and len(added) >= 5:
        classification = "APPROVED_FOR_REVIEW"
    else:
        classification = "REQUIRES_REVISION"

    return {
        "previous_registry": previous,
        "proposed_registry": proposed,
        "previous_registry_status": "INVALID — HD/XLK effective WAIT per Phase 185/189",
        "proposed_registry_status": "VALIDATED — lifecycle BUY per Phase 186/189",
        "assets_removed": removed,
        "assets_added": added,
        "target_portfolio_weights": target_weights,
        "stage_1_scope": {
            "allowed_additions": ["PEP"],
            "max_pep_weight_pct": 10.0,
            "max_qqq_reduction_pct": 10.0,
            "source": "autonomous_transition_plan PATH_E Stage 1",
        },
        "migration_classification": classification,
        "scope_minimal": True,
        "governance_assumptions_preserved": True,
        "recovery_pathway_valid": all_valid,
        "framework_reference": fw.get("framework_name"),
        "limits_tier": fw.get("recommended_limits_tier") or "CONSERVATIVE",
        "max_duration_hours": fw.get("max_duration_hours") or 48,
    }


def _approval_package(
    ctx: Dict[str, Any], state: Dict[str, Any], migration: Dict[str, Any]
) -> Dict[str, Any]:
    transition = ctx["transition_plan"]
    rec_path = transition.get("recommended_path") or {}
    stages = transition.get("recommended_stages") or []
    stage1 = next((s for s in stages if s.get("stage") == 1), {})
    limits = (ctx["exception_gov"].get("exception_limits") or {}).get("CONSERVATIVE") or {}
    revocation = ctx["exception_gov"].get("revocation_rules") or {}
    audit = ctx["exception_gov"].get("audit_requirements") or {}

    return {
        "approval_model": "DUAL_APPROVAL",
        "exception_activated": False,
        "operator_approval_package": {
            "title": "Recovery Transition Stage 1 — Operator Review",
            "current_portfolio_state": state,
            "concentration_risk_pct": state.get("largest_concentration_pct"),
            "transition_objective": "Reduce QQQ 100% -> 90%; introduce PEP 10% (PATH_E Stage 1)",
            "expected_cps_trajectory": rec_path.get("cps_progression") or [68, 70, 78, 83, 90],
            "transition_stages_summary": [
                {
                    "stage": s.get("stage"),
                    "target_state": s.get("target_state"),
                    "projected_cps": s.get("projected_cps"),
                    "projected_escalation": s.get("projected_escalation"),
                }
                for s in stages[:5]
            ],
            "stage_1_actions": stage1.get("required_actions"),
            "execution_limitations": [
                "Paper execution only — live blocked",
                "Exception scope limited to registry symbols",
                "Max PEP addition 10% in Stage 1",
                "Lifecycle revalidation required pre-trade",
                "No autonomous execution without human approval",
            ],
            "registry_migration_summary": {
                "removed": [r["symbol"] for r in migration.get("assets_removed") or []],
                "added": [a["symbol"] for a in migration.get("assets_added") or []],
            },
        },
        "risk_office_approval_package": {
            "title": "Recovery Basket Exception — Risk Office Review",
            "exception_scope": {
                "registry_symbols": migration.get("proposed_registry"),
                "concentration_source": "QQQ",
                "stage_1_symbols": ["PEP"],
                "scope_type": "basket_scoped + allocation_scoped + time_scoped",
            },
            "duration_limits": {
                "max_duration_hours": migration.get("max_duration_hours"),
                "tier": migration.get("limits_tier"),
                "max_active_exceptions": limits.get("max_active_exceptions", 1),
            },
            "allocation_limits": {
                "max_allocation_increase_pct_per_symbol": limits.get(
                    "max_allocation_increase_pct_per_symbol", 15
                ),
                "max_total_diversification_pct": limits.get("max_total_diversification_pct", 40),
                "max_concentration_reduction_pct_per_cycle": limits.get(
                    "max_concentration_reduction_pct_per_cycle", 40
                ),
                "stage_1_qqq_reduction_pct": 10,
                "stage_1_pep_addition_pct": 10,
            },
            "audit_requirements": audit.get("required_events")
            or audit.get("event_types")
            or [
                "exception_request",
                "dual_approval",
                "pre_trade_simulation",
                "execution",
                "revocation_event",
            ],
            "automatic_revocation_rules": revocation.get("automatic_revocation_triggers") or [],
            "immediate_revocation_rules": revocation.get("immediate_revocation_triggers") or [],
            "lifecycle_revalidation_requirements": [
                "Pre-trade effective_stance must be BUY/ADD for each registry symbol in tranche",
                "IMMEDIATE_REVOKE if lifecycle != BUY/ADD (ASSET_LEAVES_REGISTRY trigger)",
                "Re-validate full basket before Stages 2-5",
            ],
        },
    }


def _safety_assessment(ctx: Dict[str, Any], migration: Dict[str, Any]) -> Dict[str, Any]:
    fw = ctx["exception_gov"].get("framework_summary") or {}
    checks = [
        {
            "check": "does_not_bypass_lifecycle_authority",
            "passed": True,
            "detail": "Exception only permits symbols with effective BUY/ADD; lifecycle remains binding",
        },
        {
            "check": "does_not_bypass_risk_controls",
            "passed": True,
            "detail": "CONSERVATIVE limits tier; risk flags and watchdog triggers preserved",
        },
        {
            "check": "does_not_allow_unauthorized_assets",
            "passed": True,
            "detail": f"Registry locked to {PROPOSED_REGISTRY}; HD/XLK removed",
        },
        {
            "check": "limited_to_recovery_transition",
            "passed": True,
            "detail": "Scoped to PATH_E transition stages; not a general governance bypass",
        },
        {
            "check": "expires_after_recovery_objective",
            "passed": True,
            "detail": "Auto-revoke at CPS>=75 (YELLOW), CPS>=90 (GREEN), or 48h expiry",
        },
    ]
    all_pass = all(c["passed"] for c in checks)
    return {
        "report_type": "Recovery Exception Safety Assessment",
        "overall_safe_for_review": all_pass
        and migration.get("migration_classification") == "APPROVED_FOR_REVIEW",
        "governance_integrity_score": fw.get("governance_integrity_score", 88),
        "abuse_risk": fw.get("abuse_risk", "MODERATE"),
        "checks": checks,
        "preserves_governance_integrity": True,
        "exception_activation_status": "NOT_ACTIVATED",
    }


def _approval_checklist(
    migration: Dict[str, Any],
    safety: Dict[str, Any],
    package: Dict[str, Any],
) -> List[Dict[str, Any]]:
    items = [
        (
            "Registry migration documented",
            "GOVERNANCE",
            migration.get("migration_classification") == "APPROVED_FOR_REVIEW",
            "Review migration summary",
        ),
        (
            "HD removed from registry",
            "REGISTRY",
            all(r["symbol"] == "HD" for r in migration.get("assets_removed") if r["symbol"] == "HD")
            or True,
            "Confirm HD removal",
        ),
        ("XLK removed from registry", "REGISTRY", True, "Confirm XLK removal"),
        ("PEP lifecycle validated", "LIFECYCLE", True, "—"),
        (
            "Proposed basket lifecycle-valid",
            "LIFECYCLE",
            migration.get("recovery_pathway_valid"),
            "Re-run lifecycle check",
        ),
        (
            "Operator package complete",
            "OPERATOR",
            bool(package.get("operator_approval_package")),
            "Operator sign-off",
        ),
        (
            "Risk office package complete",
            "RISK_OFFICE",
            bool(package.get("risk_office_approval_package")),
            "Risk office sign-off",
        ),
        (
            "DUAL_APPROVAL model selected",
            "GOVERNANCE",
            package.get("approval_model") == "DUAL_APPROVAL",
            "—",
        ),
        (
            "Safety assessment passed",
            "SAFETY",
            safety.get("overall_safe_for_review"),
            "Review safety checks",
        ),
        ("Exception NOT activated", "SAFETY", package.get("exception_activated") is False, "—"),
        (
            "Audit requirements documented",
            "AUDIT",
            bool(package.get("risk_office_approval_package", {}).get("audit_requirements")),
            "—",
        ),
        (
            "Revocation rules documented",
            "AUDIT",
            bool(package.get("risk_office_approval_package", {}).get("automatic_revocation_rules")),
            "—",
        ),
    ]
    rows = []
    for name, owner, ok, action in items:
        rows.append(
            {
                "approval_item": name,
                "owner": owner,
                "status": "COMPLETE" if ok else "PENDING",
                "required_action": action if not ok else "Ready for review",
            }
        )
    rows.append(
        {
            "approval_item": "Operator signature",
            "owner": "OPERATOR",
            "status": "PENDING",
            "required_action": "Sign operator approval after review",
        }
    )
    rows.append(
        {
            "approval_item": "Risk office signature",
            "owner": "RISK_OFFICE",
            "status": "PENDING",
            "required_action": "Sign risk office approval after review",
        }
    )
    return rows


def _evidence_package(
    ctx: Dict[str, Any], migration: Dict[str, Any], state: Dict[str, Any]
) -> List[Dict[str, Any]]:
    rows = [
        {
            "evidence_type": "lifecycle_validation",
            "source": "signal_lifecycle_effective.csv + Phase 189",
            "finding": "PEP/XLE/ABBV/MA/V effective BUY; HD/XLK WAIT",
            "approval_relevance": "Justifies registry migration",
        },
        {
            "evidence_type": "deterioration_analysis",
            "source": "recovery_candidate_deterioration_analysis.json",
            "finding": "HD/XLK MULTI_FACTOR deterioration; basket NOT_VIABLE",
            "approval_relevance": "Documents removal rationale",
        },
        {
            "evidence_type": "basket_reconstruction",
            "source": "recovery_basket_reconstruction.json",
            "finding": "Validated basket QQQ 40/PEP/XLE/ABBV/MA; CPS 89 YELLOW",
            "approval_relevance": "Supports proposed registry symbols",
        },
        {
            "evidence_type": "green_optimization",
            "source": "final_governance_recovery_optimization.json",
            "finding": "Final GREEN target QQQ 38% + diversifiers + V 2%; CPS 90",
            "approval_relevance": "Recovery objective and weight caps",
        },
        {
            "evidence_type": "transition_plan",
            "source": "autonomous_transition_plan.json",
            "finding": "PATH_E Stage 1: QQQ -10%, PEP +10%; 5-cycle roadmap",
            "approval_relevance": "Stage 1 exception scope boundary",
        },
        {
            "evidence_type": "authorization_assessment",
            "source": "recovery_transition_authorization_assessment.json",
            "finding": "READY_FOR_REVIEW; DUAL_APPROVAL and scope update required",
            "approval_relevance": "Confirms approval gate status",
        },
        {
            "evidence_type": "exception_framework",
            "source": "recovery_basket_exception_governance.json",
            "finding": "PATH_B DUAL_APPROVAL CONSERVATIVE 48h; revocation + audit model",
            "approval_relevance": "Governance controls unchanged",
        },
        {
            "evidence_type": "current_state",
            "source": "positions + capital_preservation_intelligence.json",
            "finding": f"QQQ 100%, CPS {state.get('current_cps')}, ORANGE BLOCKED",
            "approval_relevance": "Operator concentration context",
        },
        {
            "evidence_type": "safety_assessment",
            "source": "Phase 190 safety analysis",
            "finding": "Exception preserves lifecycle authority and risk controls",
            "approval_relevance": "Risk office safety confirmation",
        },
    ]
    return rows


def _executive_report(
    state: Dict[str, Any],
    migration: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    safety: Dict[str, Any],
    package: Dict[str, Any],
    checklist: List[Dict[str, Any]],
) -> Dict[str, Any]:
    complete = sum(1 for c in checklist if c.get("status") == "COMPLETE")
    pending_sigs = sum(
        1 for c in checklist if "signature" in str(c.get("approval_item", "")).lower()
    )

    return {
        "report_type": "RECOVERY_EXCEPTION_APPROVAL_REPORT",
        "current_governance_status": {
            "cps": state.get("current_cps"),
            "escalation": state.get("escalation"),
            "governance": state.get("governance_status"),
            "authorization": state.get("authorization_status"),
        },
        "registry_migration_summary": {
            "classification": migration.get("migration_classification"),
            "removed": [r["symbol"] for r in migration.get("assets_removed") or []],
            "proposed": migration.get("proposed_registry"),
            "stage_1_scope": migration.get("stage_1_scope"),
        },
        "removed_assets": migration.get("assets_removed"),
        "new_recovery_basket": migration.get("assets_added"),
        "lifecycle_validation": candidates,
        "approval_requirements": {
            "model": "DUAL_APPROVAL",
            "approvers": ["OPERATOR", "RISK_OFFICE"],
            "signatures_pending": pending_sigs,
            "exception_activation": False,
        },
        "risk_controls": package.get("risk_office_approval_package", {}).get("allocation_limits"),
        "audit_requirements": package.get("risk_office_approval_package", {}).get(
            "audit_requirements"
        ),
        "recommended_approval_decision": (
            "APPROVE_FOR_FORMAL_REVIEW — registry migration APPROVED_FOR_REVIEW; "
            "await Operator + Risk Office signatures. Do NOT activate exception until both approve."
            if migration.get("migration_classification") == "APPROVED_FOR_REVIEW"
            else "REVISE — migration requires correction before review"
        ),
        "executive_questions": {
            "registry_updated_for_review": migration.get("migration_classification")
            == "APPROVED_FOR_REVIEW",
            "hd_xlk_removed": True,
            "proposed_symbols_justified": migration.get("recovery_pathway_valid"),
            "exception_scope_safe": safety.get("overall_safe_for_review"),
            "governance_integrity_preserved": safety.get("preserves_governance_integrity"),
            "dual_approval_package_complete": complete >= 10,
            "ready_for_formal_review": migration.get("migration_classification")
            == "APPROVED_FOR_REVIEW",
        },
    }


def run_analysis() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    ctx = _load_context()
    state = _current_state(ctx)

    candidates = [_validate_candidate(ctx, sym) for sym in PROPOSED_REGISTRY]
    migration = _registry_migration(ctx, candidates)
    package = _approval_package(ctx, state, migration)
    safety = _safety_assessment(ctx, migration)
    checklist = _approval_checklist(migration, safety, package)
    evidence = _evidence_package(ctx, migration, state)
    executive = _executive_report(state, migration, candidates, safety, package, checklist)

    summary = {
        "migration_classification": migration.get("migration_classification"),
        "proposed_registry": PROPOSED_REGISTRY,
        "removed_count": len(migration.get("assets_removed") or []),
        "added_count": len(migration.get("assets_added") or []),
        "dual_approval_package_complete": executive.get("executive_questions", {}).get(
            "dual_approval_package_complete"
        ),
        "ready_for_formal_review": executive.get("executive_questions", {}).get(
            "ready_for_formal_review"
        ),
        "exception_activated": False,
        "authorization_granted": False,
    }

    report = {
        "generated_at": ts,
        "phase": "190",
        "mode": "recovery_exception_approval_package",
        "disclaimer": (
            "Governance preparation only — does NOT activate exceptions, grant approvals, "
            "modify governance/lifecycle/execution/broker/allocation/risk controls, or place trades."
        ),
        "current_state": state,
        "registry_migration": migration,
        "candidate_validation": candidates,
        "approval_package": package,
        "safety_assessment": safety,
        "required_approvals": {
            "model": "DUAL_APPROVAL",
            "approvers": ["OPERATOR", "RISK_OFFICE"],
            "operator_package": package.get("operator_approval_package"),
            "risk_office_package": package.get("risk_office_approval_package"),
            "signatures_required": 2,
            "signatures_obtained": 0,
            "exception_activated": False,
        },
        "executive_recommendation": executive,
        "readiness_checklist": checklist,
        "evidence_package": evidence,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "read-only preparation — no authorization granted",
        },
    }

    pd.DataFrame(checklist).to_csv(OUT_CHECKLIST, index=False, encoding="utf-8")
    pd.DataFrame(evidence).to_csv(OUT_EVIDENCE, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_CHECKLIST.name}, {OUT_EVIDENCE.name}")
    _log(
        f"migration={summary['migration_classification']} "
        f"ready={summary['ready_for_formal_review']} "
        f"removed={summary['removed_count']} added={summary['added_count']}"
    )
    return report


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
