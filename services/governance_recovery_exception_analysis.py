"""
TRITON Phase 180 — Governance Recovery Exception Path Design (analysis only).

Designs hypothetical governance-safe exception frameworks that could permit
concentration-reducing diversification while ORANGE escalation blocks normal
authorization. Does NOT implement any exception.

SAFETY: analysis and design only — no governance, lifecycle, execution, broker,
allocation, or risk control changes. No trades or orders.

Outputs (data/results/):
    governance_recovery_exception_analysis.json
    governance_recovery_exception_paths.csv
    governance_recovery_deadlock_analysis.csv

Run:
    python -m services.governance_recovery_exception_analysis
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
    PERMIT_ESCALATION,
    _open_new_tickers,
    _simulate_scenario,
)
from services.pnl_ledger_builder import _norm_sym, _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "governance_recovery_exception_analysis.json"
OUT_PATHS = RESULTS / "governance_recovery_exception_paths.csv"
OUT_DEADLOCK = RESULTS / "governance_recovery_deadlock_analysis.csv"

RECOVERY_BASKET = ["HD", "XLK", "BITO"]
DEFAULT_YELLOW_WEIGHTS = {"QQQ": 60.0, "HD": 15.0, "XLK": 25.0}
DEFAULT_BALANCED_WEIGHTS = {"QQQ": 50.0, "HD": 25.0, "XLK": 25.0}
DEFAULT_BASKET_EQUAL = {"QQQ": 33.33, "HD": 33.33, "XLK": 33.34}

DEADLOCK_CLASSES = {"NO_DEADLOCK", "PARTIAL_DEADLOCK", "FULL_RECOVERY_DEADLOCK"}
SAFETY_CLASSES = {"SAFE", "CONDITIONALLY_SAFE", "HIGH_RISK", "UNACCEPTABLE"}


def _log(msg: str) -> None:
    print(f"[governance_recovery_exception_analysis] {msg}", flush=True)


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


def _first_json(*paths: Path) -> Tuple[Path, Dict[str, Any]]:
    for p in paths:
        if p.is_file():
            return p, _read_json(p)
    return paths[0], {}


def _load_context() -> Dict[str, Any]:
    auth_path, auth = _first_json(
        RESULTS / "governance_authorization.json",
        RESULTS / "execution_authorization.json",
    )
    cpe_path, cpe = _first_json(RESULTS / "capital_preservation_escalation.json")
    cpi_path, cpi = _first_json(RESULTS / "capital_preservation_intelligence.json")
    alloc_path, alloc = _first_json(RESULTS / "portfolio_allocation_summary.json")
    audit_path, audit = _first_json(RESULTS / "allocation_execution_path_audit.json")
    basket_path, basket = _first_json(RESULTS / "diversification_recovery_basket.json")
    thresh_path, thresh = _first_json(RESULTS / "governance_recovery_thresholds.json")
    sim_path, sim = _first_json(RESULTS / "concentration_recovery_simulation.json")
    ready_path, ready = _first_json(RESULTS / "diversification_execution_readiness.json")
    gate_path, gate = _first_json(RESULTS / "paper_submission_gate_audit.json")

    positions = _load_positions()
    recommendations = _read_csv(RESULTS / "portfolio_allocation_recommendations.csv")
    lifecycle = _read_csv(RESULTS / "signal_lifecycle_effective.csv")

    total_mv = sum(p.get("market_value", 0) for p in positions) or _to_float(
        alloc.get("total_position_value"), 1400.0
    )
    _, largest_pct, _ = _concentration_score(positions)
    base_components = dict(cpi.get("component_scores") or {})
    for k in ("drawdown", "concentration", "exposure", "operational", "execution"):
        base_components.setdefault(k, 70)
    current_cps = int(
        cpi.get("capital_preservation_score") or _estimate_cps(positions, base_components)[0]
    )
    esc = str(cpe.get("escalation_state") or _escalation_from_cps(current_cps))

    open_new = _open_new_tickers(recommendations)
    if not open_new:
        open_new = [
            _norm_sym(t)
            for t in (alloc.get("top_5_opportunities") or [])
            if isinstance(t, dict) and str(t.get("recommended_action") or "").upper() == "OPEN_NEW"
            for t in [t.get("ticker")]
            if t
        ]
    if not open_new:
        diag = alloc.get("allocation_diagnostics") or {}
        n = int(diag.get("open_new_candidates") or alloc.get("open_new_candidates") or 0)
        if n > 0:
            open_new = ["HD", "XLK"][:n]

    yellow_weights = (
        (thresh.get("executive_report") or {}).get("minimum_yellow_recovery_plan") or {}
    ).get("weights") or DEFAULT_YELLOW_WEIGHTS
    balanced_weights = (
        (thresh.get("executive_report") or {}).get("optimal_recovery_path") or {}
    ).get("weights") or DEFAULT_BALANCED_WEIGHTS

    return {
        "paths": {
            "governance_authorization": str(auth_path),
            "capital_preservation_escalation": str(cpe_path),
            "capital_preservation_intelligence": str(cpi_path),
            "portfolio_allocation_summary": str(alloc_path),
            "allocation_execution_path_audit": str(audit_path),
            "diversification_recovery_basket": str(basket_path),
            "governance_recovery_thresholds": str(thresh_path),
            "concentration_recovery_simulation": str(sim_path),
            "diversification_execution_readiness": str(ready_path),
            "paper_submission_gate_audit": str(gate_path),
        },
        "auth": auth,
        "cpe": cpe,
        "cpi": cpi,
        "alloc": alloc,
        "audit": audit,
        "basket": basket,
        "thresh": thresh,
        "sim": sim,
        "ready": ready,
        "gate": gate,
        "recommendations": recommendations,
        "lifecycle": lifecycle,
        "positions": positions,
        "total_mv": total_mv,
        "largest_pct": largest_pct,
        "current_cps": current_cps,
        "base_components": base_components,
        "escalation": esc,
        "yellow_weights": {k: float(v) for k, v in yellow_weights.items()},
        "balanced_weights": {k: float(v) for k, v in balanced_weights.items()},
        "open_new": open_new,
    }


def _governance_model(ctx: Dict[str, Any]) -> Dict[str, Any]:
    auth = ctx["auth"]
    cpe = ctx["cpe"]
    return {
        "concentration_model": "CPS-weighted escalation with PERMIT_ESCALATION={GREEN,YELLOW}",
        "escalation_rules": {
            "current_escalation": ctx["escalation"],
            "escalation_reasons": cpe.get("escalation_reason")
            or cpe.get("escalation_reason_labels"),
            "cps_yellow_threshold": CPS_YELLOW,
            "cps_green_threshold": CPS_GREEN,
            "permit_escalation": sorted(PERMIT_ESCALATION),
        },
        "authorization_rules": {
            "governance_authorized": bool(auth.get("governance_authorized")),
            "execution_authorized": bool(auth.get("execution_authorized")),
            "overall_authorization": bool(auth.get("overall_authorization")),
            "gate_reasons": auth.get("gate_reasons") or {},
            "governance_requirement": "Escalation must be GREEN or YELLOW for governance_authorized=true",
        },
        "recovery_basket": {
            "members": RECOVERY_BASKET,
            "basket_v1": (ctx["basket"].get("recovery_basket_v1") or {}).get("members"),
            "minimum_yellow_plan": ctx["yellow_weights"],
            "optimal_recovery_plan": ctx["balanced_weights"],
        },
        "diversification_requirements": {
            "open_new_candidates": ctx["open_new"],
            "available_slots": ctx["alloc"].get("available_slots"),
            "minimum_for_yellow": (ctx["basket"].get("executive_report") or {}).get(
                "orange_to_yellow_minimum_names"
            ),
            "minimum_for_green": (ctx["basket"].get("executive_report") or {}).get(
                "yellow_to_green_minimum_names"
            ),
        },
    }


def _analyze_deadlock(ctx: Dict[str, Any]) -> Dict[str, Any]:
    auth = ctx["auth"]
    audit_summary = (ctx["audit"].get("summary") or {}) if ctx["audit"] else {}
    ready = ctx["ready"]
    paradox = bool(audit_summary.get("paradox_detected"))
    gov_blocked = not bool(auth.get("governance_authorized"))
    exec_ok = bool(auth.get("execution_authorized"))
    open_new = ctx["open_new"]
    slots = _to_float(ctx["alloc"].get("available_slots"), 0)

    blocking_controls = [
        {
            "control": "ORANGE escalation gate",
            "layer": "governance",
            "effect": "governance_authorized=false blocks final_submit_allowed",
            "blocks_recovery": True,
        },
        {
            "control": "Green-governance requirement",
            "layer": "governance_execution_readiness",
            "effect": "Readiness fails governance check while ORANGE",
            "blocks_recovery": True,
        },
        {
            "control": "Concentration CPS scoring",
            "layer": "capital_preservation",
            "effect": "100% QQQ concentration triggers ORANGE",
            "blocks_recovery": False,
            "protects_recovery": True,
            "note": "Correctly detects problem; does not block recovery actions if authorized",
        },
        {
            "control": "Execution authorization (paper)",
            "layer": "execution",
            "effect": "execution_authorized=true permits paper layer",
            "blocks_recovery": False,
            "protects_recovery": True,
        },
        {
            "control": "Allocation OPEN_NEW scoring",
            "layer": "allocation",
            "effect": f"Identifies {len(open_new)} OPEN_NEW candidates with {int(slots)} slots",
            "blocks_recovery": False,
            "protects_recovery": True,
        },
        {
            "control": "Lifecycle effective stance",
            "layer": "lifecycle",
            "effect": "Allocation/lifecycle drift (BUY→WAIT) blocks even if governance cleared",
            "blocks_recovery": True,
        },
        {
            "control": "execute_trades dry_run default",
            "layer": "execution",
            "effect": "Plan-only mode prevents order placement without --execute",
            "blocks_recovery": True,
        },
        {
            "control": "Sector/risk overlay",
            "layer": "risk",
            "effect": "Sector hard caps may block additional entries",
            "blocks_recovery": paradox,
        },
    ]

    protecting = [c for c in blocking_controls if c.get("protects_recovery")]
    preventing = [c for c in blocking_controls if c.get("blocks_recovery")]

    recovery_possible_now = (
        bool(auth.get("governance_authorized"))
        and len(open_new) > 0
        and slots > 0
        and not bool((ctx["gate"].get("dry_run_decision") or {}).get("dry_run", True))
    )

    if recovery_possible_now:
        classification = "NO_DEADLOCK"
        exists = False
    elif gov_blocked and exec_ok and open_new and paradox:
        classification = "FULL_RECOVERY_DEADLOCK"
        exists = True
    elif gov_blocked and paradox and (open_new or slots > 0):
        classification = "FULL_RECOVERY_DEADLOCK"
        exists = True
    elif gov_blocked and (open_new or ready):
        classification = "PARTIAL_DEADLOCK"
        exists = True
    else:
        classification = "PARTIAL_DEADLOCK"
        exists = gov_blocked

    cycle = [
        "Concentration Risk (100% QQQ)",
        "ORANGE Escalation",
        "Governance Block (governance_authorized=false)",
        "Diversification Orders Blocked",
        "Concentration Persists",
    ]

    return {
        "deadlock_exists": exists,
        "deadlock_classification": classification,
        "recovery_possible_under_current_rules": recovery_possible_now,
        "governance_paradox_detected": paradox,
        "self_reinforcing_cycle": cycle,
        "controls_blocking_recovery": [c["control"] for c in preventing],
        "controls_protecting_recovery": [c["control"] for c in protecting],
        "control_details": blocking_controls,
        "evidence": {
            "available_slots": slots,
            "open_new_candidates": open_new,
            "orders_planned": audit_summary.get("observed_orders_planned"),
            "orders_executed": audit_summary.get("observed_orders_executed"),
            "governance_authorized": auth.get("governance_authorized"),
            "execution_authorized": auth.get("execution_authorized"),
            "readiness_status": (ctx["gate"].get("governance_context") or {}).get("readiness")
            or (ctx["ready"].get("current_state") or {}).get("current_governance_status"),
        },
    }


def _path_simulation(
    path_id: str,
    name: str,
    description: str,
    weights: Dict[str, float],
    ctx: Dict[str, Any],
    *,
    scope: str,
    authority_required: str,
    exception_risk_base: float,
    days_multiplier: float,
) -> Dict[str, Any]:
    sim = _simulate_scenario(
        weights,
        ctx["total_mv"],
        ctx["base_components"],
        ctx["largest_pct"],
        ctx["current_cps"],
    )
    deadlock_resolved = sim["escalation_level"] in PERMIT_ESCALATION
    cps_delta = sim["cps_improvement"]
    conc_red = sim["concentration_reduction_pct"]
    gov_risk_change = -cps_delta / 100.0 if cps_delta > 0 else 0.05

    lifecycle_penalty = 0.15 if ctx["audit"] else 0.10
    exec_penalty = 0.10 if (ctx["gate"].get("dry_run_decision") or {}).get("dry_run") else 0.0
    recovery_prob = float(
        np.clip(
            0.40 * (1.0 if deadlock_resolved else 0.2)
            + 0.25 * min(cps_delta / 20.0, 1.0)
            + 0.20 * min(conc_red / 50.0, 1.0)
            + 0.15 * (1.0 - exception_risk_base)
            - lifecycle_penalty
            - exec_penalty,
            0.05,
            0.95,
        )
    )
    exception_risk = float(
        np.clip(exception_risk_base + (0.1 if not deadlock_resolved else -0.05), 0, 1)
    )

    if deadlock_resolved:
        days_yellow = max(1, int(3 * days_multiplier))
        days_green = max(days_yellow + 7, int(14 * days_multiplier))
        if sim["projected_cps"] >= CPS_GREEN:
            days_green = days_yellow + 3
    else:
        days_yellow = 999
        days_green = 999

    baseline_days_yellow = 999 if not bool(ctx["auth"].get("governance_authorized")) else 30
    recovery_acceleration = max(0, baseline_days_yellow - days_yellow)

    return {
        "exception_path": path_id,
        "path_name": name,
        "description": description,
        "scope": scope,
        "authority_required": authority_required,
        "target_weights": sim["weights"],
        "projected_cps_change": cps_delta,
        "projected_cps": sim["projected_cps"],
        "projected_concentration_reduction_pct": conc_red,
        "projected_largest_weight_pct": sim["largest_position_weight_pct"],
        "projected_escalation": sim["escalation_level"],
        "projected_governance_status": sim["governance_status"],
        "governance_risk_change": _f(gov_risk_change, 3),
        "recovery_probability": _prob_label(recovery_prob),
        "recovery_probability_score": _f(recovery_prob * 100, 1),
        "exception_risk": _prob_label(exception_risk),
        "exception_risk_score": _f(exception_risk * 100, 1),
        "expected_governance_impact": (
            f"ORANGE → {sim['escalation_level']} (CPS {ctx['current_cps']} → {sim['projected_cps']})"
        ),
        "deadlock_resolved": deadlock_resolved,
        "days_to_yellow": days_yellow if days_yellow < 999 else "UNREACHABLE",
        "days_to_green": days_green if days_green < 999 else "UNREACHABLE",
        "recovery_acceleration_days": recovery_acceleration,
        "concentration_risk_reduction_pct": conc_red,
        "governance_stability_impact": (
            "IMPROVES" if deadlock_resolved else "NEUTRAL" if cps_delta > 0 else "WORSENS"
        ),
    }


def _model_exception_paths(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    paths = [
        _path_simulation(
            "PATH_A",
            "Concentration-Reducing Exception Only",
            "Permit only trades where post-trade simulation shows reduced largest-position "
            "weight and CPS improvement. No new symbols outside concentration-reducing set.",
            ctx["yellow_weights"],
            ctx,
            scope="OPEN_NEW/ADD on HD, XLK only when projected largest weight decreases",
            authority_required="Automated pre-trade simulation gate + audit log",
            exception_risk_base=0.20,
            days_multiplier=1.2,
        ),
        _path_simulation(
            "PATH_B",
            "Recovery Basket Exception",
            "Pre-scoped exception limited to approved recovery basket (HD, XLK, BITO) "
            "with fixed maximum weights from minimum YELLOW plan.",
            ctx["yellow_weights"],
            ctx,
            scope="Recovery basket members only; max weights per governance_recovery_thresholds",
            authority_required="Operator approval + basket registry match",
            exception_risk_base=0.15,
            days_multiplier=1.0,
        ),
        _path_simulation(
            "PATH_C",
            "Governance-Approved Diversification Window",
            "Time-limited (48–72h) operator-approved window permitting recovery basket "
            "execution while ORANGE; auto-expires and reverts to standard gates.",
            ctx["balanced_weights"],
            ctx,
            scope="Time-boxed window; recovery basket only; mandatory post-window CPS re-eval",
            authority_required="Human operator approval + time-box token",
            exception_risk_base=0.25,
            days_multiplier=0.8,
        ),
        _path_simulation(
            "PATH_D",
            "Threshold-Based Recovery Authorization",
            "Auto-permit when pre-trade simulation projects CPS ≥ YELLOW threshold and "
            "concentration reduction ≥ minimum; no manual window required.",
            ctx["yellow_weights"],
            ctx,
            scope="Simulation-gated auto-authorization when CPS ≥ 75 post-trade",
            authority_required="Automated threshold gate only (no operator)",
            exception_risk_base=0.35,
            days_multiplier=0.7,
        ),
    ]
    return paths


def _safety_analysis(paths: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
    vectors = [
        {
            "vector": "Exception authority abuse",
            "question": "Could exception authority be abused?",
            "assessment": "PATH_A/B scope-limited to concentration-reducing symbols reduces abuse surface. "
            "PATH_C/D require operator or simulation gates.",
            "mitigation": "Basket registry, audit trail, time-box expiry, pre-trade simulation",
        },
        {
            "vector": "Concentration increase",
            "question": "Could concentration increase?",
            "assessment": "All paths require post-trade CPS simulation; PATH_A explicitly blocks "
            "non-reducing trades.",
            "mitigation": "Mandatory largest-weight monotonic decrease check",
        },
        {
            "vector": "Risk control bypass",
            "question": "Could risk controls be bypassed?",
            "assessment": "Sector overlays, position caps, and lifecycle gates remain active; "
            "exception only bypasses governance escalation gate.",
            "mitigation": "Exception does not disable lifecycle, risk overlay, or position manager",
        },
        {
            "vector": "Governance weakening",
            "question": "Could governance protections weaken?",
            "assessment": "Permanent bypass would weaken governance; time-boxed/scoped exceptions preserve "
            "ORANGE block for non-recovery actions.",
            "mitigation": "Default-deny outside exception scope; auto-revert post-recovery",
        },
        {
            "vector": "Authorization integrity",
            "question": "Would authorization integrity remain intact?",
            "assessment": "Dual authorization preserved: exception token + execution_authorized + "
            "operator approval where required.",
            "mitigation": "Separate exception audit artifact; no silent rule changes",
        },
    ]

    path_safety: List[Dict[str, Any]] = []
    for p in paths:
        pid = p["exception_path"]
        risk = _to_float(p.get("exception_risk_score"), 50) / 100.0
        if pid == "PATH_A":
            cls = "CONDITIONALLY_SAFE"
            notes = "Narrowest scope; simulation gate required; lifecycle drift still blocks"
        elif pid == "PATH_B":
            cls = "SAFE"
            notes = "Fixed basket + operator approval; lowest abuse surface"
        elif pid == "PATH_C":
            cls = "CONDITIONALLY_SAFE"
            notes = "Effective but requires operator discipline and window expiry enforcement"
        elif pid == "PATH_D":
            cls = "HIGH_RISK" if risk > 0.30 else "CONDITIONALLY_SAFE"
            notes = "Auto-authorization without operator; simulation gaming risk"
        else:
            cls = "CONDITIONALLY_SAFE"
            notes = ""
        path_safety.append(
            {
                "exception_path": pid,
                "safety_classification": cls,
                "exception_risk_score": p.get("exception_risk_score"),
                "notes": notes,
            }
        )

    overall = "CONDITIONALLY_SAFE"
    if any(s["safety_classification"] == "UNACCEPTABLE" for s in path_safety):
        overall = "UNACCEPTABLE"
    elif all(s["safety_classification"] == "SAFE" for s in path_safety):
        overall = "SAFE"
    elif any(s["safety_classification"] == "HIGH_RISK" for s in path_safety):
        overall = "CONDITIONALLY_SAFE"

    return {
        "overall_classification": overall,
        "abuse_vectors": vectors,
        "path_classifications": path_safety,
        "governance_integrity_preserved": True,
        "requires_implementation_phase": True,
        "disclaimer": "Design analysis only — no exception implemented",
    }


def _recovery_impact(paths: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
    baseline = {
        "days_to_yellow": "UNREACHABLE",
        "days_to_green": "UNREACHABLE",
        "recovery_probability": "LOW",
        "note": "Under current rules governance blocks diversification required for recovery",
    }
    best_effective = max(paths, key=lambda p: _to_float(p.get("recovery_probability_score"), 0))
    safest = next(
        (p for p in paths if p["exception_path"] == "PATH_B"),
        paths[0],
    )
    return {
        "baseline_without_exception": baseline,
        "best_effective_path": best_effective["exception_path"],
        "safest_path": safest["exception_path"],
        "path_impacts": [
            {
                "exception_path": p["exception_path"],
                "days_to_yellow": p["days_to_yellow"],
                "days_to_green": p["days_to_green"],
                "recovery_acceleration_days": p["recovery_acceleration_days"],
                "concentration_risk_reduction_pct": p["concentration_risk_reduction_pct"],
                "governance_stability_impact": p["governance_stability_impact"],
                "deadlock_resolved": p["deadlock_resolved"],
            }
            for p in paths
        ],
    }


def _executive_recommendation(
    deadlock: Dict[str, Any],
    paths: List[Dict[str, Any]],
    safety: Dict[str, Any],
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    safest = next((p for p in paths if p["exception_path"] == "PATH_B"), paths[0])
    most_effective = max(paths, key=lambda p: _to_float(p.get("recovery_probability_score"), 0))

    can_recover_without = deadlock.get("recovery_possible_under_current_rules", False)
    exception_helps = any(p.get("deadlock_resolved") for p in paths)

    return {
        "report_type": "GOVERNANCE_RECOVERY_EXCEPTION_REPORT",
        "executive_questions": {
            "can_recovery_occur_without_exceptions": not deadlock.get("deadlock_exists", True),
            "can_recovery_occur_without_exceptions_detail": (
                "No — ORANGE governance block prevents diversification orders that would clear ORANGE"
                if deadlock.get("deadlock_exists")
                else "Yes — current authorization permits recovery actions"
            ),
            "would_exception_materially_improve_recovery": exception_helps,
            "would_exception_materially_improve_recovery_detail": (
                f"Yes — {most_effective['exception_path']} projects "
                f"{most_effective['expected_governance_impact']} with "
                f"{most_effective['recovery_probability']} recovery probability"
                if exception_helps
                else "No material improvement projected"
            ),
            "safest_exception_design": safest["path_name"],
            "safest_exception_path": safest["exception_path"],
            "most_effective_exception_design": most_effective["path_name"],
            "most_effective_exception_path": most_effective["exception_path"],
            "minimum_authority_required": safest["authority_required"],
            "governance_remains_intact": safety.get("governance_integrity_preserved", True),
        },
        "recommended_approach": (
            "Implement PATH_B (Recovery Basket Exception) as a design candidate: "
            "operator-approved, basket-scoped, time-limited exception that permits HD+XLK "
            "diversification while ORANGE, with mandatory post-trade CPS re-evaluation. "
            "Pair with lifecycle reconciliation to resolve allocation drift. "
            "Do NOT implement in this phase."
        ),
        "deadlock_status": deadlock.get("deadlock_classification"),
        "current_governance_state": {
            "governance_authorized": ctx["auth"].get("governance_authorized"),
            "escalation": ctx["escalation"],
            "cps": ctx["current_cps"],
        },
        "current_concentration_state": {
            "largest_position_pct": _f(ctx["largest_pct"], 2),
            "holdings_count": len(ctx["positions"]),
            "open_new_candidates": ctx["open_new"],
        },
        "safest_exception_path": safest["exception_path"],
        "most_effective_exception_path": most_effective["exception_path"],
        "recovery_impact_summary": most_effective["expected_governance_impact"],
        "governance_impact_summary": (
            "Scoped exception preserves ORANGE block for non-recovery actions; "
            "clears paradox for pre-approved basket only"
        ),
    }


def _deadlock_csv(deadlock: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for c in deadlock.get("control_details") or []:
        rows.append(
            {
                "control": c.get("control"),
                "layer": c.get("layer"),
                "blocks_recovery": c.get("blocks_recovery"),
                "protects_recovery": c.get("protects_recovery", False),
                "effect": c.get("effect"),
                "note": c.get("note", ""),
            }
        )
    rows.append(
        {
            "control": "DEADLOCK_SUMMARY",
            "layer": "system",
            "blocks_recovery": deadlock.get("deadlock_exists"),
            "protects_recovery": False,
            "effect": deadlock.get("deadlock_classification"),
            "note": " | ".join(deadlock.get("self_reinforcing_cycle") or []),
        }
    )
    return pd.DataFrame(rows)


def _paths_csv(paths: List[Dict[str, Any]], safety: Dict[str, Any]) -> pd.DataFrame:
    safety_map = {
        s["exception_path"]: s["safety_classification"]
        for s in (safety.get("path_classifications") or [])
    }
    rows = []
    for p in paths:
        rows.append(
            {
                "exception_path": p["exception_path"],
                "path_name": p["path_name"],
                "deadlock_resolved": p["deadlock_resolved"],
                "recovery_probability": p["recovery_probability"],
                "recovery_probability_score": p["recovery_probability_score"],
                "exception_risk": p["exception_risk"],
                "exception_risk_score": p["exception_risk_score"],
                "governance_risk_change": p["governance_risk_change"],
                "projected_cps_change": p["projected_cps_change"],
                "projected_concentration_reduction_pct": p["projected_concentration_reduction_pct"],
                "days_to_yellow": p["days_to_yellow"],
                "days_to_green": p["days_to_green"],
                "safety_classification": safety_map.get(p["exception_path"], "CONDITIONALLY_SAFE"),
                "authority_required": p["authority_required"],
            }
        )
    return pd.DataFrame(rows)


def run_analysis() -> Dict[str, Any]:
    ctx = _load_context()
    ts = _now_iso()
    governance_model = _governance_model(ctx)
    deadlock = _analyze_deadlock(ctx)
    paths = _model_exception_paths(ctx)
    safety = _safety_analysis(paths, ctx)
    recovery_impact = _recovery_impact(paths, ctx)
    executive = _executive_recommendation(deadlock, paths, safety, ctx)

    for p in paths:
        sc = next(
            (
                s["safety_classification"]
                for s in safety["path_classifications"]
                if s["exception_path"] == p["exception_path"]
            ),
            "CONDITIONALLY_SAFE",
        )
        p["safety_classification"] = sc

    summary = {
        "deadlock_classification": deadlock["deadlock_classification"],
        "deadlock_exists": deadlock["deadlock_exists"],
        "exception_paths_evaluated": len(paths),
        "safest_path": executive["safest_exception_path"],
        "most_effective_path": executive["most_effective_exception_path"],
        "recovery_without_exception": executive["executive_questions"][
            "can_recovery_occur_without_exceptions"
        ],
        "exception_improves_recovery": executive["executive_questions"][
            "would_exception_materially_improve_recovery"
        ],
        "overall_safety": safety["overall_classification"],
        "minimum_authority": executive["executive_questions"]["minimum_authority_required"],
    }

    report = {
        "generated_at": ts,
        "phase": "180",
        "mode": "governance_recovery_exception_design",
        "disclaimer": (
            "Analysis and design only — no governance exception implemented. "
            "No trades, orders, or rule modifications."
        ),
        "current_state": {
            "governance_model": governance_model,
            "cps": ctx["current_cps"],
            "escalation": ctx["escalation"],
            "largest_concentration_pct": _f(ctx["largest_pct"], 2),
            "governance_authorized": ctx["auth"].get("governance_authorized"),
            "execution_authorized": ctx["auth"].get("execution_authorized"),
            "open_new_candidates": ctx["open_new"],
            "available_slots": ctx["alloc"].get("available_slots"),
            "recovery_basket": RECOVERY_BASKET,
            "input_paths": ctx["paths"],
        },
        "deadlock_analysis": deadlock,
        "exception_paths": paths,
        "safety_analysis": safety,
        "recovery_impact": recovery_impact,
        "executive_recommendation": executive,
        "summary": summary,
        "certification": {
            "status": "CERTIFIED",
            "safety": "design-only — no governance, lifecycle, execution, or broker modifications",
        },
    }

    _paths_csv(paths, safety).to_csv(OUT_PATHS, index=False, encoding="utf-8")
    _deadlock_csv(deadlock).to_csv(OUT_DEADLOCK, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _log(f"wrote {OUT_JSON.name}, {OUT_PATHS.name}, {OUT_DEADLOCK.name}")
    _log(
        f"deadlock={deadlock['deadlock_classification']} "
        f"safest={summary['safest_path']} effective={summary['most_effective_path']}"
    )
    return report


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
