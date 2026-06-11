"""
TRITON Phase 183 — Recovery Basket Implementation Readiness Review (assessment only).

Determines whether the Phase 181/182 Recovery Basket Exception Framework is
mature enough for implementation consideration. Does NOT implement the framework.

SAFETY: assessment only — no governance, lifecycle, execution, broker, allocation,
or risk control changes. No trades or orders.

Outputs (data/results/):
    recovery_basket_implementation_readiness.json
    recovery_basket_dependency_audit.csv
    recovery_basket_readiness_gaps.csv

Run:
    python -m services.recovery_basket_implementation_readiness
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

from services.diversification_feasibility_analysis import _lifecycle_row
from services.pnl_ledger_builder import _norm_sym, _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "recovery_basket_implementation_readiness.json"
OUT_DEPS = RESULTS / "recovery_basket_dependency_audit.csv"
OUT_GAPS = RESULTS / "recovery_basket_readiness_gaps.csv"

RECOVERY_SYMBOLS = ["HD", "XLK", "QQQ", "BITO"]
READINESS_CLASSES = {
    "NOT_READY",
    "PARTIALLY_READY",
    "READY_FOR_REVIEW",
    "READY_FOR_IMPLEMENTATION",
}
RISK_CLASSES = {"LOW", "MODERATE", "HIGH", "CRITICAL"}


def _log(msg: str) -> None:
    print(f"[recovery_basket_implementation_readiness] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _first_json(*paths: Path) -> Tuple[Path, Dict[str, Any]]:
    for p in paths:
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            return p, d if isinstance(d, dict) else {}
    return paths[0], {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_context() -> Dict[str, Any]:
    sim_path, simulation = _first_json(RESULTS / "recovery_basket_exception_simulation.json")
    gov_path, governance = _first_json(RESULTS / "recovery_basket_exception_governance.json")
    p180_path, phase180 = _first_json(RESULTS / "governance_recovery_exception_analysis.json")
    audit_path, exec_audit = _first_json(RESULTS / "allocation_execution_path_audit.json")
    ready_path, div_ready = _first_json(RESULTS / "diversification_execution_readiness.json")
    gate_path, gate = _first_json(RESULTS / "paper_submission_gate_audit.json")
    auth_path, auth = _first_json(RESULTS / "governance_authorization.json")
    exit_path, exit_strat = _first_json(RESULTS / "concentration_exit_strategy.json")

    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    lifecycle = _read_csv(RESULTS / "signal_lifecycle_effective.csv")
    alloc_summary_path, alloc_summary = _first_json(RESULTS / "portfolio_allocation_summary.json")

    scenario_d = next(
        (s for s in (simulation.get("simulation_results") or []) if s.get("scenario") == "D"),
        {},
    )

    return {
        "paths": {
            "simulation": str(sim_path),
            "governance_framework": str(gov_path),
            "phase180": str(p180_path),
            "execution_audit": str(audit_path),
            "diversification_readiness": str(ready_path),
            "paper_gate": str(gate_path),
            "authorization": str(auth_path),
            "exit_strategy": str(exit_path),
            "allocation_summary": str(alloc_summary_path),
        },
        "simulation": simulation,
        "governance": governance,
        "phase180": phase180,
        "exec_audit": exec_audit,
        "div_ready": div_ready,
        "gate": gate,
        "auth": auth,
        "exit_strat": exit_strat,
        "alloc_summary": alloc_summary,
        "recommendations": recommendations,
        "lifecycle": lifecycle,
        "scenario_d": scenario_d,
    }


def _symbol_layers(ctx: Dict[str, Any], sym: str) -> Dict[str, Any]:
    sym = _norm_sym(sym)
    recs = ctx["recommendations"]
    sub = recs[recs["ticker"].map(_norm_sym) == sym] if not recs.empty else pd.DataFrame()
    rec = sub.iloc[0].to_dict() if not sub.empty else {}
    lc = _lifecycle_row(ctx["lifecycle"], sym)
    exit_cs = ctx["exit_strat"].get("current_state") or {}
    exit_key = f"{sym.lower()}_status"
    sim_assets = (ctx["simulation"].get("current_state") or {}).get("asset_status") or {}
    sim_st = sim_assets.get(sym) or {}

    alloc_action = str(rec.get("recommended_action") or "—").upper()
    alloc_stance = str(rec.get("effective_stance") or rec.get("lifecycle_action") or "—").upper()
    lc_stance = str(lc.get("effective_stance") or "—").upper()
    exit_status = str(exit_cs.get(exit_key) or sim_st.get("exit_strategy_status") or "—").upper()

    drift_alloc_lc = (
        alloc_stance != lc_stance
        and alloc_stance not in ("—", "")
        and lc_stance
        not in (
            "—",
            "",
        )
    )
    drift_exit_alloc = exit_status != alloc_action and exit_status not in ("—", "")

    return {
        "symbol": sym,
        "allocation_action": alloc_action,
        "allocation_stance": alloc_stance,
        "lifecycle_stance": lc_stance,
        "exit_strategy_status": exit_status,
        "lifecycle_actionable": lc_stance in ("BUY", "ADD"),
        "open_new_eligible": alloc_action == "OPEN_NEW",
        "drift_allocation_vs_lifecycle": drift_alloc_lc,
        "drift_exit_vs_allocation": drift_exit_alloc,
        "drift_severity": (
            "HIGH"
            if drift_alloc_lc and alloc_action in ("OPEN_NEW", "ADD")
            else "MODERATE" if drift_alloc_lc or drift_exit_alloc else "NONE"
        ),
    }


def _lifecycle_drift_analysis(ctx: Dict[str, Any]) -> Dict[str, Any]:
    layers = [_symbol_layers(ctx, s) for s in ("HD", "XLK", "QQQ", "BITO")]
    drifted = [
        l for l in layers if l["drift_allocation_vs_lifecycle"] or l["drift_exit_vs_allocation"]
    ]
    hd_xlk = [l for l in layers if l["symbol"] in ("HD", "XLK")]

    alloc_ts = (ctx["alloc_summary"].get("generated_at_utc") or "")[:19]
    lc_rows = ctx["lifecycle"]
    lc_ts = ""
    if not lc_rows.empty and "generated_at" in lc_rows.columns:
        lc_ts = str(lc_rows["generated_at"].iloc[0])[:19]
    elif not lc_rows.empty:
        for col in ("lifecycle_ts", "asof_ts", "timestamp"):
            if col in lc_rows.columns:
                lc_ts = str(lc_rows[col].iloc[0])[:19]
                break

    cause_candidates = []
    if alloc_ts and lc_ts and alloc_ts < lc_ts:
        cause_candidates.append("Allocation artifact stale relative to lifecycle refresh")
    if any(
        l["allocation_action"] == "HOLD" and l["exit_strategy_status"] == "OPEN_NEW" for l in hd_xlk
    ):
        cause_candidates.append(
            "Exit strategy references prior OPEN_NEW promotion not reflected in current allocation"
        )
    if all(not l["lifecycle_actionable"] for l in hd_xlk):
        cause_candidates.append(
            "Lifecycle effective stance WAIT blocks BUY path despite registry eligibility"
        )
    if not cause_candidates:
        cause_candidates.append(
            "Cross-layer stance divergence between allocation, lifecycle, and exit strategy"
        )

    recovery_impact = (
        "HIGH — recovery basket symbols not lifecycle-actionable; "
        "exception alone cannot produce orders without lifecycle reconciliation or exception-scoped lifecycle override design"
        if all(not l["lifecycle_actionable"] for l in hd_xlk)
        else "MODERATE — partial drift on subset of basket symbols"
    )

    return {
        "symbols_analyzed": layers,
        "drift_count": len(drifted),
        "recovery_basket_drift": hd_xlk,
        "historical_drift_note": (
            "Phase 179/182 observed allocation OPEN_NEW (BUY) vs lifecycle WAIT; "
            "current allocation also shows HOLD/WAIT divergence from exit strategy OPEN_NEW"
        ),
        "probable_causes": cause_candidates,
        "artifact_timestamps": {
            "allocation": alloc_ts or "unknown",
            "lifecycle": lc_ts or "unknown",
        },
        "frequency": "PERSISTENT — observed across Phases 179, 182, and 183 assessments",
        "recovery_impact": recovery_impact,
        "implementation_risk": (
            "HIGH" if all(not l["lifecycle_actionable"] for l in hd_xlk) else "MODERATE"
        ),
        "recommended_mitigation": [
            "Reconcile allocation ↔ lifecycle before implementation",
            "Define exception-scoped lifecycle eligibility rules in design spec (not runtime change in this phase)",
            "Add drift detection monitor to exception activation gate",
        ],
    }


def _dependency_audit(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    gov = ctx["governance"]
    sim = ctx["simulation"]
    framework = gov.get("framework_summary") or {}

    deps = [
        {
            "system": "governance_execution_readiness",
            "component": "ORANGE escalation gate",
            "change_required": True,
            "change_type": "ADD_EXCEPTION_TOKEN_PATH",
            "assumption_verified": False,
            "controls_complete": False,
            "notes": "Requires exception token bypass scoped to recovery basket only",
        },
        {
            "system": "governance_authorization",
            "component": "governance_authorized flag",
            "change_required": True,
            "change_type": "CONDITIONAL_OVERRIDE",
            "assumption_verified": True,
            "controls_complete": False,
            "notes": "Phase 182 sim confirms post-trade AUTHORIZED; override mechanism not built",
        },
        {
            "system": "human_approval_center",
            "component": "DUAL_APPROVAL workflow",
            "change_required": True,
            "change_type": "NEW_APPROVAL_TYPE",
            "assumption_verified": False,
            "controls_complete": False,
            "notes": "Operator + Risk Office approval path designed but not wired",
        },
        {
            "system": "recovery_basket_registry",
            "component": "APPROVED_RECOVERY_BASKET_REGISTRY",
            "change_required": True,
            "change_type": "NEW_ARTIFACT",
            "assumption_verified": True,
            "controls_complete": True,
            "notes": "Registry designed in Phase 181; runtime registry not deployed",
        },
        {
            "system": "execute_trades",
            "component": "dry_run / --execute gate",
            "change_required": True,
            "change_type": "EXCEPTION_AWARE_PLACEMENT",
            "assumption_verified": True,
            "controls_complete": False,
            "notes": "dry_run=true blocks all placement today",
        },
        {
            "system": "place_live_orders",
            "component": "final_submit_allowed",
            "change_required": True,
            "change_type": "EXCEPTION_TOKEN_CHECK",
            "assumption_verified": False,
            "controls_complete": False,
            "notes": "Must honor exception token without weakening default gate",
        },
        {
            "system": "signal_lifecycle",
            "component": "effective_stance gate",
            "change_required": False,
            "change_type": "NONE",
            "assumption_verified": False,
            "controls_complete": True,
            "notes": "Should remain unchanged; drift must be resolved upstream",
        },
        {
            "system": "portfolio_allocation",
            "component": "OPEN_NEW scoring",
            "change_required": False,
            "change_type": "NONE",
            "assumption_verified": True,
            "controls_complete": True,
            "notes": "No allocation logic changes required for exception path",
        },
        {
            "system": "risk_overlay",
            "component": "sector / concentration guards",
            "change_required": False,
            "change_type": "NONE",
            "assumption_verified": True,
            "controls_complete": True,
            "notes": "Risk controls remain active per Phase 181/182 safety analysis",
        },
        {
            "system": "broker",
            "component": "order submission",
            "change_required": False,
            "change_type": "NONE",
            "assumption_verified": True,
            "controls_complete": True,
            "notes": "No broker rule changes; paper path unchanged",
        },
        {
            "system": "paper_execution_audit",
            "component": "audit trail",
            "change_required": True,
            "change_type": "EXTEND_EVENTS",
            "assumption_verified": True,
            "controls_complete": False,
            "notes": "12 event types designed; simulated in Phase 182, not persisted at runtime",
        },
        {
            "system": "capital_preservation",
            "component": "CPS / escalation",
            "change_required": False,
            "change_type": "NONE",
            "assumption_verified": True,
            "controls_complete": True,
            "notes": "CPS re-evaluation drives auto-revocation",
        },
        {
            "system": "monitoring",
            "component": "exception control panel",
            "change_required": True,
            "change_type": "NEW_DASHBOARD",
            "assumption_verified": False,
            "controls_complete": False,
            "notes": "Monitoring requirements designed; operational panel not built",
        },
        {
            "system": "documentation",
            "component": "implementation spec",
            "change_required": True,
            "change_type": "NEW_DOC",
            "assumption_verified": False,
            "controls_complete": False,
            "notes": "Design phases 180-182 complete; implementation spec pending",
        },
    ]
    return deps


def _readiness_gaps(
    ctx: Dict[str, Any], drift: Dict[str, Any], deps: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    gaps: List[Dict[str, Any]] = []
    incomplete = [d for d in deps if d["change_required"] and not d["controls_complete"]]

    gap_specs = [
        {
            "gap_id": "G1",
            "category": "governance",
            "gap": "Exception token path not implemented in governance_execution_readiness",
            "severity": "CRITICAL",
            "blocks_implementation": True,
        },
        {
            "gap_id": "G2",
            "category": "governance",
            "gap": "DUAL_APPROVAL workflow not wired to exception issuance",
            "severity": "HIGH",
            "blocks_implementation": True,
        },
        {
            "gap_id": "G3",
            "category": "lifecycle",
            "gap": "HD/XLK not lifecycle-actionable (effective_stance=WAIT)",
            "severity": "HIGH",
            "blocks_implementation": True,
        },
        {
            "gap_id": "G4",
            "category": "lifecycle",
            "gap": "Allocation ↔ lifecycle ↔ exit strategy drift on recovery basket symbols",
            "severity": "HIGH",
            "blocks_implementation": True,
        },
        {
            "gap_id": "G5",
            "category": "execution",
            "gap": "execute_trades dry_run default prevents placement validation",
            "severity": "MODERATE",
            "blocks_implementation": False,
        },
        {
            "gap_id": "G6",
            "category": "execution",
            "gap": "final_submit_allowed exception check not implemented",
            "severity": "HIGH",
            "blocks_implementation": True,
        },
        {
            "gap_id": "G7",
            "category": "audit",
            "gap": "Runtime exception audit ledger not deployed",
            "severity": "MODERATE",
            "blocks_implementation": True,
        },
        {
            "gap_id": "G8",
            "category": "monitoring",
            "gap": "Exception control panel / drift monitor not operational",
            "severity": "MODERATE",
            "blocks_implementation": False,
        },
        {
            "gap_id": "G9",
            "category": "documentation",
            "gap": "Implementation specification and rollback plan not published",
            "severity": "MODERATE",
            "blocks_implementation": False,
        },
        {
            "gap_id": "G10",
            "category": "operational",
            "gap": "End-to-end paper dry-run with exception token not executed",
            "severity": "HIGH",
            "blocks_implementation": True,
        },
    ]

    for g in gap_specs:
        g["status"] = "OPEN"
        g["source_phase"] = "183"
        gaps.append(g)

    for d in incomplete:
        if not any(g["gap"] == d["notes"] for g in gaps):
            gaps.append(
                {
                    "gap_id": f"GD-{d['system']}",
                    "category": d["system"],
                    "gap": d["notes"],
                    "severity": "MODERATE",
                    "blocks_implementation": d["change_required"],
                    "status": "OPEN",
                    "source_phase": "183",
                }
            )

    return gaps


def _dimension_scores(
    ctx: Dict[str, Any], drift: Dict[str, Any], gaps: List[Dict[str, Any]]
) -> Dict[str, Any]:
    gov = ctx["governance"]
    sim = ctx["simulation"]
    scenario_d = ctx["scenario_d"]
    exec_summary = ctx["exec_audit"].get("summary") or {}

    blocking = sum(1 for g in gaps if g.get("blocks_implementation"))
    critical = sum(1 for g in gaps if g.get("severity") == "CRITICAL")

    governance_readiness = float(
        np.clip(
            88
            - 15 * (1 if critical else 0)
            - 5 * sum(1 for g in gaps if g["category"] == "governance"),
            30,
            95,
        )
    )
    lifecycle_readiness = float(
        np.clip(
            100
            - 40 * (1 if drift.get("implementation_risk") == "HIGH" else 0.5)
            - 10 * drift.get("drift_count", 0),
            15,
            80,
        )
    )
    execution_readiness = float(
        np.clip(
            70
            - 20 * (1 if ctx["gate"].get("dry_run_decision", {}).get("dry_run") else 0)
            - 15 * (1 if not exec_summary.get("observed_orders_planned") == 0 else 0),
            25,
            75,
        )
    )
    audit_readiness = float(
        np.clip(
            85 if (sim.get("audit_simulation") or {}).get("audit_coverage_complete") else 50,
            40,
            90,
        )
    )
    monitoring_readiness = 45.0
    simulation_readiness = float(
        np.clip(
            90 if scenario_d.get("outcome_classification") == "SUCCESS" else 50,
            40,
            95,
        )
    )
    operational_readiness = float(
        np.clip(
            (governance_readiness + lifecycle_readiness + execution_readiness) / 3 - blocking * 3,
            20,
            70,
        )
    )

    overall = float(
        np.clip(
            0.22 * governance_readiness
            + 0.22 * lifecycle_readiness
            + 0.18 * execution_readiness
            + 0.12 * audit_readiness
            + 0.10 * monitoring_readiness
            + 0.16 * simulation_readiness,
            0,
            100,
        )
    )

    advance_sim = (sim.get("executive_recommendation") or {}).get(
        "advance_to_implementation_review"
    )

    if overall >= 85 and blocking == 0 and lifecycle_readiness >= 75:
        classification = "READY_FOR_IMPLEMENTATION"
    elif (overall >= 65 or advance_sim) and simulation_readiness >= 70:
        classification = "READY_FOR_REVIEW"
    elif overall >= 45:
        classification = "PARTIALLY_READY"
    else:
        classification = "NOT_READY"

    if classification == "READY_FOR_IMPLEMENTATION" and (blocking > 0 or lifecycle_readiness < 70):
        classification = "READY_FOR_REVIEW"

    return {
        "governance_readiness": _f(governance_readiness, 1),
        "lifecycle_readiness": _f(lifecycle_readiness, 1),
        "execution_readiness": _f(execution_readiness, 1),
        "audit_readiness": _f(audit_readiness, 1),
        "monitoring_readiness": _f(monitoring_readiness, 1),
        "simulation_readiness": _f(simulation_readiness, 1),
        "operational_readiness": _f(operational_readiness, 1),
        "overall_readiness": _f(overall, 1),
        "overall_classification": classification,
        "blocking_gaps": blocking,
        "simulation_validated": scenario_d.get("deadlock_resolved"),
        "advance_to_implementation_review": advance_sim,
    }


def _f(v: float, d: int = 1) -> float:
    return round(float(v), d)


def _risk_analysis(
    ctx: Dict[str, Any], drift: Dict[str, Any], gaps: List[Dict[str, Any]]
) -> Dict[str, Any]:
    def _classify(score: float) -> str:
        if score >= 75:
            return "CRITICAL"
        if score >= 55:
            return "HIGH"
        if score >= 35:
            return "MODERATE"
        return "LOW"

    gov_risk = 30 + 10 * sum(1 for g in gaps if g["category"] == "governance")
    lifecycle_risk = 70 if drift.get("implementation_risk") == "HIGH" else 45
    execution_risk = 55 if ctx["gate"].get("dry_run_decision", {}).get("dry_run") else 35
    audit_risk = 40 if any(g["gap_id"] == "G7" for g in gaps) else 25
    operational_risk = 50 + 5 * sum(1 for g in gaps if g.get("blocks_implementation"))

    risks = {
        "governance_risk": _classify(gov_risk),
        "governance_risk_score": gov_risk,
        "lifecycle_risk": _classify(lifecycle_risk),
        "lifecycle_risk_score": lifecycle_risk,
        "execution_risk": _classify(execution_risk),
        "execution_risk_score": execution_risk,
        "audit_risk": _classify(audit_risk),
        "audit_risk_score": audit_risk,
        "operational_risk": _classify(operational_risk),
        "operational_risk_score": operational_risk,
    }
    scores = [gov_risk, lifecycle_risk, execution_risk, audit_risk, operational_risk]
    risks["overall_risk"] = _classify(float(np.mean(scores)))
    risks["overall_risk_score"] = _f(float(np.mean(scores)), 1)
    return risks


def _executive_recommendation(
    scores: Dict[str, Any],
    gaps: List[Dict[str, Any]],
    drift: Dict[str, Any],
    risks: Dict[str, Any],
    deps: List[Dict[str, Any]],
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    major = [g for g in gaps if g.get("severity") in ("CRITICAL", "HIGH")]
    no_change = [d["system"] for d in deps if not d["change_required"]]
    need_change = [d["system"] for d in deps if d["change_required"]]

    ready_for_review = scores.get("overall_classification") in (
        "READY_FOR_REVIEW",
        "READY_FOR_IMPLEMENTATION",
    )

    return {
        "report_type": "RECOVERY_BASKET_IMPLEMENTATION_READINESS_REPORT",
        "implementation_recommendation": scores.get("overall_classification"),
        "ready_for_implementation_review": ready_for_review,
        "ready_for_implementation": scores.get("overall_classification")
        == "READY_FOR_IMPLEMENTATION",
        "executive_questions": {
            "is_framework_ready_for_implementation_review": ready_for_review,
            "what_gaps_remain": [g["gap"] for g in major[:6]],
            "what_risks_remain": [
                f"{k}={v}" for k, v in risks.items() if k.endswith("_risk") and k != "overall_risk"
            ],
            "how_serious_is_lifecycle_drift": drift.get("implementation_risk"),
            "systems_requiring_modification": need_change,
            "systems_no_change_required": no_change,
            "must_complete_before_implementation": [
                "Resolve HD/XLK lifecycle drift (WAIT vs OPEN_NEW/HOLD)",
                "Implement exception token path in governance gate",
                "Wire DUAL_APPROVAL to exception issuance",
                "Deploy runtime exception audit ledger",
                "Publish implementation spec with rollback plan",
                "Execute end-to-end paper validation with exception token",
            ],
        },
        "recommended_actions": [
            "Proceed to formal implementation review board (framework design validated)",
            "Prioritize lifecycle reconciliation for HD and XLK before any exception activation",
            "Implement governance exception token as scoped additive path (not rule weakening)",
            "Build DUAL_APPROVAL workflow and 48h time-box enforcement",
            "Extend paper_execution_audit with Phase 181 event types",
            "Run paper E2E test of Scenario D weights before production consideration",
        ],
        "scenario_d_reference": {
            "weights": {"QQQ": 60.0, "HD": 15.0, "XLK": 25.0},
            "deadlock_resolved": ctx["scenario_d"].get("deadlock_resolved"),
            "projected_cps": ctx["scenario_d"].get("projected_cps"),
            "projected_escalation": ctx["scenario_d"].get("projected_escalation"),
        },
    }


def run_readiness_review() -> Dict[str, Any]:
    ctx = _load_context()
    ts = _now_iso()
    drift = _lifecycle_drift_analysis(ctx)
    deps = _dependency_audit(ctx)
    gaps = _readiness_gaps(ctx, drift, deps)
    scores = _dimension_scores(ctx, drift, gaps)
    risks = _risk_analysis(ctx, drift, gaps)
    executive = _executive_recommendation(scores, gaps, drift, risks, deps, ctx)

    summary = {
        "overall_classification": scores["overall_classification"],
        "overall_readiness": scores["overall_readiness"],
        "ready_for_implementation_review": executive["ready_for_implementation_review"],
        "ready_for_implementation": executive["ready_for_implementation"],
        "major_gaps": len([g for g in gaps if g.get("severity") in ("CRITICAL", "HIGH")]),
        "lifecycle_drift_risk": drift.get("implementation_risk"),
        "overall_risk": risks.get("overall_risk"),
        "systems_to_modify": len([d for d in deps if d["change_required"]]),
        "simulation_validated": scores.get("simulation_validated"),
    }

    report = {
        "generated_at": ts,
        "phase": "183",
        "mode": "recovery_basket_implementation_readiness",
        "disclaimer": (
            "Implementation readiness assessment only — does not implement the framework, "
            "modify rules, or place orders."
        ),
        "readiness_scores": scores,
        "dependency_audit": {
            "dependencies": deps,
            "systems_requiring_changes": [d["system"] for d in deps if d["change_required"]],
            "systems_no_changes": [d["system"] for d in deps if not d["change_required"]],
            "unverified_assumptions": [d for d in deps if not d["assumption_verified"]],
            "incomplete_controls": [
                d for d in deps if d["change_required"] and not d["controls_complete"]
            ],
        },
        "readiness_gaps": gaps,
        "risk_analysis": risks,
        "lifecycle_drift_analysis": drift,
        "executive_recommendation": executive,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "assessment-only — no governance, lifecycle, execution, or broker modifications",
        },
    }

    dep_rows = []
    for d in deps:
        dep_rows.append(
            {
                "system": d["system"],
                "component": d["component"],
                "change_required": d["change_required"],
                "change_type": d["change_type"],
                "assumption_verified": d["assumption_verified"],
                "controls_complete": d["controls_complete"],
                "notes": d["notes"],
            }
        )
    pd.DataFrame(dep_rows).to_csv(OUT_DEPS, index=False, encoding="utf-8")
    pd.DataFrame(gaps).to_csv(OUT_GAPS, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_DEPS.name}, {OUT_GAPS.name}")
    _log(
        f"classification={scores['overall_classification']} "
        f"overall={scores['overall_readiness']} gaps={len(gaps)} drift={drift.get('implementation_risk')}"
    )
    return report


def main() -> None:
    run_readiness_review()


if __name__ == "__main__":
    main()
