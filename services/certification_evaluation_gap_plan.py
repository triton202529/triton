"""
certification_evaluation_gap_plan.py
Phase 196U — Read-only certification Evaluation gap analysis and closure plan.
Does not modify certification artifacts or weaken gates.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

CERT_PATH = RESULTS / "capital_preservation_certification.json"
EVAL_PATH = RESULTS / "protective_action_evaluation.json"
TRIALS_PATH = RESULTS / "protective_action_trials.json"
DEF_SIM_PATH = RESULTS / "defensive_simulation_results.json"
GOVERNOR_PATH = RESULTS / "capital_preservation_governor.json"
CPI_PATH = RESULTS / "capital_preservation_intelligence.json"
CPE_PATH = RESULTS / "capital_preservation_escalation.json"
GOV_AUTH_PATH = RESULTS / "governance_authorization.json"
READINESS_PATH = RESULTS / "execution_readiness.json"
RISK_COMMITTEE_PATH = RESULTS / "risk_committee_oversight.json"
PGB_PATH = RESULTS / "preservation_governance_board.json"
ICR_PATH = RESULTS / "investment_committee_review.json"
ALLOC_SUMMARY_PATH = RESULTS / "portfolio_allocation_summary.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _artifact_age_minutes(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 60.0


def _evaluation_threshold_label(eval_doc: Dict[str, Any]) -> Tuple[float, str]:
    baseline = str(
        eval_doc.get("baseline_context_label")
        or (eval_doc.get("baseline_context") or {}).get("baseline_context_label")
        or "STANDARD"
    )
    high_cps_adj = bool(eval_doc.get("high_cps_adjustment_applied"))
    if baseline == "HIGH_CPS_PRESERVATION" and high_cps_adj:
        return 40.0, "high_cps_baseline_aware>=40"
    if baseline == "MODERATE_CPS_IMPROVEMENT":
        return 45.0, "moderate_cps>=45"
    return 50.0, "standard>=50"


def build_gap_map_rows() -> List[Dict[str, str]]:
    cert = _load_json(CERT_PATH)
    eval_doc = _load_json(EVAL_PATH)
    cpi = _load_json(CPI_PATH)
    cpe = _load_json(CPE_PATH)
    governor = _load_json(GOVERNOR_PATH)
    gov = _load_json(GOV_AUTH_PATH)
    readiness = _load_json(READINESS_PATH)
    def_sim = _load_json(DEF_SIM_PATH)
    alloc = _load_json(ALLOC_SUMMARY_PATH)

    eval_area = (cert.get("areas") or {}).get("Evaluation") or {}
    eval_score = int(eval_area.get("score") or 0)
    avg_eff = float(eval_doc.get("average_effectiveness") or 0)
    threshold, threshold_label = _evaluation_threshold_label(eval_doc)
    gap_points = max(0.0, threshold - avg_eff)

    conc_pct = float(
        (def_sim.get("baseline") or {}).get("largest_concentration_pct")
        or alloc.get("largest_position_weight_pct")
        or 0
    )
    conc_component = int((cpi.get("component_scores") or {}).get("concentration") or 0)

    rows: List[Dict[str, str]] = []

    def add(
        gap_name: str,
        source: str,
        current: str,
        required: str,
        blocker_type: str,
        root_cause: str,
        action: str,
        *,
        market: bool = False,
        arm: bool = False,
        manual: bool = False,
        code: bool = False,
        observe: bool = True,
        now: bool = False,
        phase: str = "196V",
    ) -> None:
        rows.append(
            {
                "gap_name": gap_name,
                "source_artifact": source,
                "current_value": current,
                "required_value": required,
                "blocker_type": blocker_type,
                "root_cause": root_cause,
                "safe_closure_action": action,
                "requires_market_hours": "true" if market else "false",
                "requires_paper_arm": "true" if arm else "false",
                "requires_manual_approval": "true" if manual else "false",
                "requires_code_change": "true" if code else "false",
                "requires_more_observation": "true" if observe else "false",
                "can_be_closed_now": "true" if now else "false",
                "recommended_phase": phase,
            }
        )

    add(
        "Evaluation area below threshold",
        "data/results/capital_preservation_certification.json",
        f"score={eval_score} avg_effectiveness={avg_eff}",
        f">={threshold} ({threshold_label})",
        "TRUE_EVALUATION_GAP",
        (
            f"4 trials avg={avg_eff}; beneficial=1 neutral=1 negative=2; "
            f"high_cps_adjustment_applied=false uses standard>=50"
        ),
        (
            "Improve protective trial outcomes via natural cycle refresh; "
            "do not edit scores or lower threshold"
        ),
        observe=True,
        now=False,
        phase="196V",
    )

    add(
        "mandatory_preservation_gate_failed veto",
        "data/results/capital_preservation_certification.json",
        "triggered (failed_gates=Evaluation)",
        "Evaluation area certified",
        "TRUE_EVALUATION_GAP",
        "Veto is consequence of uncertified Evaluation mandatory gate",
        "Close Evaluation gap first; veto clears when Evaluation certifies",
        observe=True,
        now=False,
        phase="196V",
    )

    add(
        "Weak trial: paper_drawdown_reduction",
        "data/results/protective_action_evaluation.json",
        "effectiveness_score=25 NEGATIVE",
        "NEUTRAL or BENEFICIAL (score>=42 typical)",
        "TRUE_EVALUATION_GAP",
        "Drawdown protection simulation shows weak protective outcome at current portfolio",
        "Observe additional cycles; review drawdown trial inputs when concentration improves",
        observe=True,
        phase="196V",
    )

    add(
        "Weak trial: paper_exposure_reduction",
        "data/results/protective_action_evaluation.json",
        "effectiveness_score=5 NEGATIVE",
        "NEUTRAL or BENEFICIAL",
        "TRUE_EVALUATION_GAP",
        "Exposure reduction trial scores minimal benefit under HIGH_CPS_PRESERVATION baseline",
        "Continue observation; do not force trades to inflate trial scores",
        observe=True,
        phase="196V",
    )

    add(
        "Portfolio concentration (WFC overweight)",
        "data/results/capital_preservation_intelligence.json",
        f"largest_weight~{conc_pct:.1f}% concentration_component={conc_component}",
        "Lower concentration / escalation reason cleared",
        "TRUE_CONCENTRATION_RISK",
        (
            "Two-name Financials book (JPM+WFC); EXCESS_CONCENTRATION escalation; "
            "governor best_protection=CONCENTRATION_REDUCTION"
        ),
        (
            "Review concentration reduction path via governance-approved protective policy; "
            "simulation-only until policy+approval aligned — no forced trades"
        ),
        manual=True,
        observe=True,
        phase="197A",
    )

    add(
        "REDUCE_CONCENTRATION policy disabled",
        "data/results/governance_authorization.json",
        "policy_authorized=false for REDUCE_CONCENTRATION candidates",
        "policy enabled with operator approval for concentration actions",
        "TRUE_GOVERNANCE_GAP",
        "JPM/WFC concentration candidates exist but protective policy disabled",
        "Operator reviews policy enablement after concentration plan — not automatic",
        manual=True,
        observe=False,
        phase="197A",
    )

    add(
        "Governor preservation posture ORANGE",
        "data/results/capital_preservation_governor.json",
        f"posture={governor.get('preservation_posture', 'ORANGE')}",
        "GREEN or YELLOW with weak-trial concern resolved",
        "TRUE_GOVERNANCE_GAP",
        "Top drivers: Concentration Risk, Weak Trial Outcomes",
        "Advisory posture; improves when evaluation and concentration gaps close",
        observe=True,
        phase="196V",
    )

    add(
        "Escalation EXCESS_CONCENTRATION",
        "data/results/capital_preservation_escalation.json",
        f"state={cpe.get('escalation_state', 'YELLOW')}",
        "No EXCESS_CONCENTRATION in escalation_reason",
        "TRUE_CONCENTRATION_RISK",
        str(cpe.get("escalation_reason_labels") or ["Concentration Risk"]),
        "Concentration reduction evidence via approved protective simulations",
        observe=True,
        phase="197A",
    )

    beneficial = int(eval_doc.get("beneficial_count") or 0)
    if beneficial >= 1:
        add(
            "Beneficial concentration trial exists",
            "data/results/protective_action_evaluation.json",
            "paper_concentration_reduction score=67 BENEFICIAL",
            "Raise portfolio average effectiveness >=50",
            "TRUE_EVALUATION_GAP",
            "Single beneficial trial insufficient vs two negative trials",
            "Use as evidence path; need improved weak trials or additional beneficial cycles",
            observe=True,
            phase="196V",
        )

    readiness_status = str(readiness.get("readiness_status") or "NOT_READY")
    if readiness_status == "READY":
        add(
            "Execution readiness",
            "data/results/execution_readiness.json",
            "READY (8/8 checks)",
            "READY",
            "READINESS_TRANSIENT",
            "Passing — not a certification blocker",
            "Maintain runtime health; no action required",
            observe=False,
            now=True,
            phase="n/a",
        )

    if gov.get("overall_authorization"):
        add(
            "Governance authorization",
            "data/results/governance_authorization.json",
            "overall_authorization=true (paper)",
            "authorized for paper evaluation",
            "AUTHORIZATION_TRANSIENT",
            "Passing for paper policy; distinct from certification Evaluation area",
            "No bypass; certification remains separate governance signal",
            observe=False,
            now=True,
            phase="n/a",
        )

    stale_candidates: List[str] = []
    for path in (CERT_PATH, EVAL_PATH, TRIALS_PATH, CPI_PATH):
        age = _artifact_age_minutes(path)
        if age is not None and age > 120:
            stale_candidates.append(f"{path.name}~{age:.0f}m")

    if stale_candidates:
        add(
            "Potentially stale certification artifacts",
            ",".join(stale_candidates),
            "age>120min",
            "cycle-aligned <40min",
            "STALE_ARTIFACT",
            "Artifact timestamps older than freshness window",
            "Allow natural continuous-loop refresh; do not hand-edit JSON",
            observe=True,
            phase="196K",
        )

    return rows


def build_certification_gap_summary() -> Dict[str, Any]:
    cert = _load_json(CERT_PATH)
    eval_doc = _load_json(EVAL_PATH)
    readiness = _load_json(READINESS_PATH)
    gov = _load_json(GOV_AUTH_PATH)

    eval_area = (cert.get("areas") or {}).get("Evaluation") or {}
    eval_score = int(eval_area.get("score") or 0)
    avg_eff = float(eval_doc.get("average_effectiveness") or 0)
    threshold, threshold_label = _evaluation_threshold_label(eval_doc)
    gap_points = round(max(0.0, threshold - avg_eff), 1)

    failed = list(cert.get("failed_requirements") or [])
    vetoes = list(cert.get("veto_triggered") or [])

    readiness_status = str(readiness.get("readiness_status") or "REVIEW")
    auth_status = "AUTHORIZED" if gov.get("overall_authorization") else "NOT_AUTHORIZED"

    gap_rows = build_gap_map_rows()
    primary_types = {r["blocker_type"] for r in gap_rows if r["blocker_type"].startswith("TRUE_")}

    safe_actions = sorted(
        {
            r["safe_closure_action"]
            for r in gap_rows
            if r["can_be_closed_now"] == "false" and r["blocker_type"].startswith("TRUE_")
        }
    )

    return {
        "certification_status": str(cert.get("certification_status") or "REVIEW"),
        "certification_score": cert.get("certification_score"),
        "certified_area_count": cert.get("certified_area_count"),
        "failed_requirements": failed,
        "evaluation_score": eval_score,
        "evaluation_threshold": threshold,
        "evaluation_threshold_label": threshold_label,
        "evaluation_gap_points": gap_points,
        "evaluation_average_effectiveness": avg_eff,
        "evaluation_beneficial_count": eval_doc.get("beneficial_count"),
        "evaluation_neutral_count": eval_doc.get("neutral_count"),
        "evaluation_negative_count": eval_doc.get("negative_count"),
        "mandatory_vetoes": vetoes,
        "primary_gap": "TRUE_EVALUATION_GAP",
        "secondary_gaps": sorted(t for t in primary_types if t != "TRUE_EVALUATION_GAP"),
        "true_governance_gap_detected": "TRUE_GOVERNANCE_GAP" in primary_types,
        "true_evaluation_gap_detected": "TRUE_EVALUATION_GAP" in primary_types,
        "true_concentration_risk_detected": "TRUE_CONCENTRATION_RISK" in primary_types,
        "stale_artifact_detected": "STALE_ARTIFACT" in {r["blocker_type"] for r in gap_rows},
        "readiness_current_status": readiness_status,
        "authorization_current_status": auth_status,
        "paper_certification_blocks_paper_entry": False,
        "live_certification_blocks_live_execution": True,
        "safe_closure_actions": safe_actions,
        "requires_paper_arm_to_close": False,
        "requires_market_hours_to_close": False,
        "requires_manual_approval_to_close": any(
            r["requires_manual_approval"] == "true"
            for r in gap_rows
            if r["blocker_type"].startswith("TRUE_")
        ),
        "requires_code_change_to_close": False,
        "recommended_next_phase": "196V",
        "gap_map_rows": gap_rows,
        "evaluation_trials": eval_doc.get("evaluations") or [],
        "pre_veto_status": cert.get("pre_veto_certification_status"),
        "high_cps_adjustment_applied": eval_doc.get("high_cps_adjustment_applied"),
    }


def write_gap_map_csv(path: Path) -> int:
    rows = build_gap_map_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
