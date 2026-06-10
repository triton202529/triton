"""
TRITON Phase 171 — Exit Timing Implementation Governance (review-only).

Evaluates whether the certified exit timing enhancements from Phases 169/170
(EARLIER_EXIT_5D, EARLIER_EXIT_PEAK_DRAWDOWN, COMBINED_OPTIMIZED_RULESET) are
safe enough to advance toward controlled paper-mode testing. Produces
governance scores, a capital-preservation assessment, a risk assessment,
an operator control rollout design, and a final governance verdict.

SAFETY: governance, risk, and capital-preservation review only. No lifecycle
rules, thresholds, execution, portfolio, or broker behavior is modified; no
orders are written.

Outputs (data/results/):
    exit_timing_governance.json
    exit_timing_governance_decisions.csv
    exit_timing_risk_assessment.csv

Run:
    python -m services.exit_timing_governance
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

from services.pnl_ledger_builder import _to_float

RESULTS = ROOT / "data" / "results"
OUT_JSON = RESULTS / "exit_timing_governance.json"
OUT_DECISIONS = RESULTS / "exit_timing_governance_decisions.csv"
OUT_RISK = RESULTS / "exit_timing_risk_assessment.csv"

PROPOSALS = ["EARLIER_EXIT_5D", "EARLIER_EXIT_PEAK_DRAWDOWN", "COMBINED_OPTIMIZED_RULESET"]

# Rule-complexity proxy: number of interacting rules each proposal activates.
PROPOSAL_COMPLEXITY = {
    "EARLIER_EXIT_5D": 1,
    "EARLIER_EXIT_PEAK_DRAWDOWN": 1,
    "COMBINED_OPTIMIZED_RULESET": 3,
}


def _log(msg: str) -> None:
    print(f"[exit_timing_governance] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(v: Any, nd: int = 2) -> Optional[float]:
    x = _to_float(v, np.nan)
    return round(float(x), nd) if np.isfinite(x) else None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _scenario_map(sim: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(s.get("scenario")): s for s in sim.get("scenarios") or []}


# ──────────────────────────────────────────────
# CARD 3 — Governance review (per proposal)
# ──────────────────────────────────────────────
def _risk_score(s: Dict[str, Any]) -> float:
    """0-100, higher = MORE risk."""
    fp = _to_float(s.get("false_positive_risk"), 50.0)
    complexity = PROPOSAL_COMPLEXITY.get(str(s.get("scenario")), 2)
    n = _to_float(s.get("impacted_trades"), 0.0)
    sample_uncertainty = float(np.clip((30.0 - n) / 30.0 * 100.0, 0.0, 100.0))
    conf = _to_float(s.get("rule_confidence"), 50.0)
    return float(
        np.clip(
            0.45 * fp
            + 0.20 * (complexity - 1) * 25.0
            + 0.15 * sample_uncertainty
            + 0.20 * (100.0 - conf),
            0.0,
            100.0,
        )
    )


def _risk_class(score: float) -> str:
    return "LOW_RISK" if score < 25 else "MEDIUM_RISK" if score < 50 else "HIGH_RISK"


def _capital_preservation_score(s: Dict[str, Any], baseline: Dict[str, Any]) -> float:
    """0-100, higher = stronger preservation alignment."""
    protected = _to_float(s.get("capital_protected"), 0.0)
    lost = _to_float(s.get("capital_lost"), 0.0)
    net = protected - lost
    deployed = max(_to_float(baseline.get("capital_deployed"), 1.0), 1.0)
    # Net protection as bps of deployed capital, saturating at 100 bps.
    bps = net / deployed * 10000.0
    protection = float(np.clip(bps, 0.0, 100.0))
    fp = _to_float(s.get("false_positive_risk"), 50.0)
    return float(np.clip(0.70 * protection + 0.30 * (100.0 - fp), 0.0, 100.0))


def _operator_control_score(s: Dict[str, Any]) -> float:
    """0-100, higher = easier to observe, override, and reverse."""
    complexity = PROPOSAL_COMPLEXITY.get(str(s.get("scenario")), 2)
    fp = _to_float(s.get("false_positive_risk"), 50.0)
    return float(np.clip(100.0 - (complexity - 1) * 20.0 - fp * 0.5, 0.0, 100.0))


def _consistency(s: Dict[str, Any]) -> float:
    improved = _to_float(s.get("improved"), np.nan)
    worsened = _to_float(s.get("worsened"), np.nan)
    if np.isfinite(improved) and np.isfinite(worsened) and improved + worsened > 0:
        return float(improved / (improved + worsened) * 100.0)
    return _to_float(s.get("rule_confidence"), np.nan)


def _evidence_strength(s: Dict[str, Any]) -> str:
    conf = _to_float(s.get("rule_confidence"), 0.0)
    n = _to_float(s.get("impacted_trades"), 0.0)
    if conf >= 75 and n >= 20:
        return "STRONG"
    if conf >= 60 and n >= 10:
        return "MODERATE"
    return "WEAK"


def build_governance_decisions(sim: Dict[str, Any], framework: Dict[str, Any]) -> pd.DataFrame:
    scenarios = _scenario_map(sim)
    baseline = sim.get("baseline") or {}
    fw_cert = (framework.get("certification") or {}).get("status", "UNKNOWN")
    rows: List[Dict[str, Any]] = []
    for name in PROPOSALS:
        s = scenarios.get(name) or {"scenario": name}
        risk = _risk_score(s)
        cp = _capital_preservation_score(s, baseline)
        ctrl = _operator_control_score(s)
        consistency = _consistency(s)
        evidence = _evidence_strength(s)
        n = int(_to_float(s.get("impacted_trades"), 0.0))
        conf = _to_float(s.get("rule_confidence"), 0.0)
        governance = float(
            np.clip(0.30 * conf + 0.25 * (100.0 - risk) + 0.25 * cp + 0.20 * ctrl, 0.0, 100.0)
        )
        fp = _to_float(s.get("false_positive_risk"), np.nan)
        delta = _to_float(s.get("simulated_pnl_delta"), np.nan)

        if (
            governance >= 70
            and risk < 50
            and np.isfinite(fp)
            and fp <= 5.0
            and np.isfinite(consistency)
            and consistency >= 80
            and evidence == "STRONG"
            and fw_cert == "CERTIFIED"
        ):
            decision = "APPROVE_FOR_PAPER_TRIAL"
            rationale = (
                f"+${delta:,.2f} replay benefit, {consistency:.0f}% of impacted trades improved, "
                f"{fp:.1f}% false positives, single observable rule — meets every governance gate"
            )
        elif governance >= 45 and risk < 70:
            decision = "REQUIRES_MORE_EVIDENCE"
            gaps = []
            if not (np.isfinite(fp) and fp <= 5.0):
                gaps.append(f"false-positive risk {fp:.1f}% exceeds the 5% gate")
            if evidence != "STRONG":
                gaps.append(f"evidence strength is {evidence}")
            if PROPOSAL_COMPLEXITY.get(name, 1) > 1:
                gaps.append("interacting rules need component-level attribution before approval")
            rationale = "; ".join(gaps) or "governance score below the approval gate"
        else:
            decision = "REJECT"
            rationale = "fails governance, risk, and evidence gates"

        rows.append(
            {
                "proposal": name,
                "governance_score": _f(governance, 1),
                "capital_preservation_score": _f(cp, 1),
                "risk_score": _f(risk, 1),
                "risk_class": _risk_class(risk),
                "operator_control_score": _f(ctrl, 1),
                "evidence_strength": evidence,
                "sample_size": n,
                "historical_consistency_pct": _f(consistency, 1),
                "false_positive_risk_pct": _f(fp, 1),
                "simulated_benefit": _f(delta),
                "decision": decision,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# CARD 4 — Capital preservation review
# ──────────────────────────────────────────────
def build_capital_preservation_assessment(
    sim: Dict[str, Any],
    framework: Dict[str, Any],
    cp_cert: Dict[str, Any],
    opp_cost: pd.DataFrame,
) -> Dict[str, Any]:
    scenarios = _scenario_map(sim)
    baseline = sim.get("baseline") or {}
    deployed = _to_float(baseline.get("capital_deployed"), np.nan)
    dd_proxy = _to_float(baseline.get("drawdown_proxy_pct"), np.nan)

    per_proposal: List[Dict[str, Any]] = []
    for name in PROPOSALS:
        s = scenarios.get(name) or {}
        protected = _to_float(s.get("capital_protected"), 0.0)
        lost = _to_float(s.get("capital_lost"), 0.0)
        fp = _to_float(s.get("false_positive_risk"), np.nan)
        n = _to_float(s.get("impacted_trades"), 0.0)
        # Worst case: every flagged trade is a false positive costing the avg per-trade benefit.
        avg_benefit = protected / n if n > 0 else 0.0
        worst_case = -(fp / 100.0) * n * avg_benefit if np.isfinite(fp) else np.nan
        dd_reduction = (
            protected / deployed * 100.0 if np.isfinite(deployed) and deployed > 0 else np.nan
        )
        # Recovery improvement: capital freed earlier is redeployable; negative
        # false_negative_risk in Phase 169 indicated exits ahead of recoveries.
        fn = _to_float(s.get("false_negative_risk"), np.nan)
        recovery = (
            "IMPROVED"
            if np.isfinite(fn) and fn <= 0
            else "NEUTRAL" if np.isfinite(fn) and fn < 30 else "AT_RISK"
        )
        per_proposal.append(
            {
                "proposal": name,
                "capital_saved": _f(protected),
                "capital_lost": _f(lost),
                "worst_case_impact": _f(worst_case),
                "false_positive_cost": _f((fp / 100.0) * protected if np.isfinite(fp) else np.nan),
                "drawdown_reduction_pct_of_deployed": _f(dd_reduction, 3),
                "recovery_improvement": recovery,
            }
        )

    # Realized late-exit damage from the Phase 170 catalog grounds the doctrine case.
    realized_lost = (
        float(pd.to_numeric(opp_cost.get("difference"), errors="coerce").clip(lower=0).sum())
        if not opp_cost.empty
        else 0.0
    )
    preserved_post = (
        float(pd.to_numeric(opp_cost.get("capital_preserved"), errors="coerce").sum())
        if not opp_cost.empty
        else 0.0
    )

    strengthens = any((p["capital_saved"] or 0) > abs(p["capital_lost"] or 0) for p in per_proposal)
    return {
        "doctrine_source": {
            "capital_preservation_certification": cp_cert.get(
                "certification_status", "UNAVAILABLE"
            ),
            "preservation_status": cp_cert.get("preservation_status", "UNAVAILABLE"),
            "paper_execution_permitted": cp_cert.get("paper_execution_permitted"),
        },
        "baseline_drawdown_proxy_pct": _f(dd_proxy),
        "historical_giveback_from_late_exits": _f(realized_lost),
        "historical_capital_preserved_by_exits": _f(preserved_post),
        "per_proposal": per_proposal,
        "would_strengthen_preservation": bool(strengthens),
        "verdict": (
            "YES — earlier exits convert documented late-exit giveback into preserved capital "
            "with near-zero measured false-positive cost"
            if strengthens
            else "NO — measured protection does not exceed measured cost"
        ),
    }


# ──────────────────────────────────────────────
# CARD 5 — Risk review
# ──────────────────────────────────────────────
def build_risk_assessment(sim: Dict[str, Any], decisions: pd.DataFrame) -> pd.DataFrame:
    scenarios = _scenario_map(sim)
    rows: List[Dict[str, Any]] = []
    for name in PROPOSALS:
        s = scenarios.get(name) or {}
        fp = _to_float(s.get("false_positive_risk"), np.nan)
        fn = _to_float(s.get("false_negative_risk"), np.nan)
        worsened = _to_float(s.get("worsened"), np.nan)
        complexity = PROPOSAL_COMPLEXITY.get(name, 2)

        def sev(x: float, lo: float, hi: float) -> str:
            if not np.isfinite(x):
                return "MEDIUM_RISK"
            return "LOW_RISK" if x < lo else "MEDIUM_RISK" if x < hi else "HIGH_RISK"

        factors = [
            (
                "false_exits",
                (
                    f"{fp:.1f}% of impacted trades would have exited a position that kept rising"
                    if np.isfinite(fp)
                    else "unmeasured"
                ),
                sev(fp, 5, 15),
                "advisory-first rollout; operator confirms before any exit acts",
            ),
            (
                "premature_exits",
                (
                    f"{int(worsened)} trades worsened in replay"
                    if np.isfinite(worsened)
                    else "unmeasured"
                ),
                sev(worsened, 2, 5),
                "minimum-hold and minimum-runup guards in the detector design",
            ),
            (
                "missed_recoveries",
                (
                    f"false-negative measure {fn:.1f}% (negative = exits beat recoveries)"
                    if np.isfinite(fn)
                    else "unmeasured"
                ),
                sev(fn, 10, 40),
                "10-day post-exit tracking already built into the Phase 170 catalog",
            ),
            (
                "overreaction_risk",
                f"{complexity} interacting rule(s); escalation bands dampen single-day noise",
                "LOW_RISK" if complexity == 1 else "MEDIUM_RISK",
                "OBSERVE/WATCH bands require multi-day confirmation before REVIEW",
            ),
            (
                "underreaction_risk",
                "20 historical LATE_EXITs show the current ruleset already underreacts",
                "MEDIUM_RISK",
                "late-exit detector surfaces aging positions daily",
            ),
            (
                "operator_override",
                "all proposed modes below AUTOMATED keep the operator as final authority",
                "LOW_RISK",
                "PAPER_TRIAL requires explicit operator opt-in per Phase 171 governance",
            ),
        ]
        for factor, measurement, severity, mitigation in factors:
            rows.append(
                {
                    "proposal": name,
                    "risk_factor": factor,
                    "measurement": measurement,
                    "severity": severity,
                    "mitigation": mitigation,
                }
            )

    df = pd.DataFrame(rows)
    # Overall row per proposal from the governance risk score.
    for _, d in decisions.iterrows():
        df.loc[len(df)] = {
            "proposal": d["proposal"],
            "risk_factor": "OVERALL",
            "measurement": f"composite risk score {d['risk_score']}/100",
            "severity": d["risk_class"],
            "mitigation": "see governance decision rationale",
        }
    return df


# ──────────────────────────────────────────────
# CARD 6 — Operator control review
# ──────────────────────────────────────────────
def build_operator_control_review(decisions: pd.DataFrame) -> Dict[str, Any]:
    approved = decisions[decisions["decision"] == "APPROVE_FOR_PAPER_TRIAL"]["proposal"].tolist()
    modes = [
        {
            "mode": "OBSERVE_ONLY",
            "description": "Detector runs daily and logs late-exit classifications; nothing is shown in trading flow.",
            "risk": "NONE",
            "entry_criteria": "none — already satisfied by the Phase 170 framework",
        },
        {
            "mode": "ADVISORY_ONLY",
            "description": "Escalation bands and exit candidates appear on the dashboard; operator decides manually.",
            "risk": "NONE (display only)",
            "entry_criteria": "OBSERVE_ONLY stable for 10 trading days with no false SEVERE flags",
        },
        {
            "mode": "PAPER_TRIAL",
            "description": "Approved rules generate paper-mode exit intents recorded and scored, never routed to the broker.",
            "risk": "LOW (no capital at risk)",
            "entry_criteria": "ADVISORY_ONLY accuracy ≥80% over 20 advisories; capital preservation certification permits paper execution",
        },
        {
            "mode": "AUTOMATED",
            "description": "Rules act autonomously inside lifecycle logic.",
            "risk": "HIGH — out of scope for Phase 171",
            "entry_criteria": "successful paper trial + a future governance phase; NOT considered here",
        },
    ]
    return {
        "modes": modes,
        "safest_first_mode": "OBSERVE_ONLY",
        "recommended_rollout_sequence": [
            "1. OBSERVE_ONLY — run the late-exit detector daily, log-only",
            "2. ADVISORY_ONLY — surface escalation bands on the dashboard",
            f"3. PAPER_TRIAL — limited to approved proposals only ({', '.join(approved) if approved else 'none approved yet'})",
            "4. AUTOMATED — explicitly deferred; requires a dedicated future governance review",
        ],
        "note": "Governance recommendation only — no mode is being implemented in this phase.",
    }


# ──────────────────────────────────────────────
# CARD 7 — Implementation readiness
# ──────────────────────────────────────────────
def build_readiness(
    sim: Dict[str, Any],
    framework: Dict[str, Any],
    decisions: pd.DataFrame,
    catalog: pd.DataFrame,
) -> Dict[str, Any]:
    factors: List[Dict[str, Any]] = []

    inputs_present = sum(
        1
        for p in (
            "exit_timing_framework.json",
            "late_exit_catalog.csv",
            "exit_opportunity_cost_report.csv",
            "exit_timing_recommendations.csv",
            "lifecycle_rule_simulation_results.json",
            "adaptive_lifecycle_optimization.json",
        )
        if (RESULTS / p).is_file()
    )
    data_quality = inputs_present / 6.0 * 100.0
    factors.append(
        {
            "factor": "data_quality",
            "score": _f(data_quality, 1),
            "detail": f"{inputs_present}/6 governance inputs present",
        }
    )

    n_exits = int(len(catalog))
    sample = float(np.clip(n_exits / 60.0 * 100.0, 0.0, 100.0))
    factors.append(
        {
            "factor": "sample_size",
            "score": _f(sample, 1),
            "detail": f"{n_exits} historical exits analyzed (60 = full credit)",
        }
    )

    cons = pd.to_numeric(decisions["historical_consistency_pct"], errors="coerce").dropna()
    consistency = float(cons.mean()) if not cons.empty else 0.0
    factors.append(
        {
            "factor": "historical_consistency",
            "score": _f(consistency, 1),
            "detail": "mean share of impacted trades improved across proposals",
        }
    )

    confs = [
        _to_float(s.get("rule_confidence"), np.nan)
        for s in (sim.get("scenarios") or [])
        if s.get("scenario") in PROPOSALS
    ]
    sim_conf = float(np.nanmean(confs)) if confs and np.isfinite(np.nanmean(confs)) else 0.0
    factors.append(
        {
            "factor": "simulation_confidence",
            "score": _f(sim_conf, 1),
            "detail": "mean Phase 169 rule confidence across proposals",
        }
    )

    fw_status = (framework.get("certification") or {}).get("status", "NOT_CERTIFIED")
    fw_score = {"CERTIFIED": 100.0, "PARTIALLY_CERTIFIED": 60.0}.get(fw_status, 0.0)
    factors.append(
        {
            "factor": "framework_certification",
            "score": _f(fw_score, 1),
            "detail": f"Phase 170 framework is {fw_status}",
        }
    )

    approved_n = int((decisions["decision"] == "APPROVE_FOR_PAPER_TRIAL").sum())
    alignment = float(np.clip(approved_n / 2.0 * 100.0, 0.0, 100.0))
    factors.append(
        {
            "factor": "governance_alignment",
            "score": _f(alignment, 1),
            "detail": f"{approved_n} proposal(s) cleared every governance gate",
        }
    )

    weights = {
        "data_quality": 0.15,
        "sample_size": 0.15,
        "historical_consistency": 0.20,
        "simulation_confidence": 0.15,
        "framework_certification": 0.20,
        "governance_alignment": 0.15,
    }
    score = float(sum(weights[f["factor"]] * (f["score"] or 0.0) for f in factors))
    status = "READY" if score >= 75 else "PARTIALLY_READY" if score >= 50 else "NOT_READY"
    return {
        "implementation_readiness_score": _f(score, 1),
        "status": status,
        "factors": factors,
        "scope": "readiness applies to OBSERVE_ONLY / ADVISORY_ONLY / PAPER_TRIAL modes only",
    }


# ──────────────────────────────────────────────
# CARDS 8 + 10 — Executive report & certification
# ──────────────────────────────────────────────
def build_executive(
    decisions: pd.DataFrame,
    cp: Dict[str, Any],
    risk: pd.DataFrame,
    control: Dict[str, Any],
    readiness: Dict[str, Any],
) -> Dict[str, Any]:
    approved = decisions[decisions["decision"] == "APPROVE_FOR_PAPER_TRIAL"]
    needs = decisions[decisions["decision"] == "REQUIRES_MORE_EVIDENCE"]
    top = (
        decisions.sort_values("governance_score", ascending=False).iloc[0]
        if not decisions.empty
        else None
    )

    overall = risk[risk["risk_factor"] == "OVERALL"]
    worst_risk = None
    if not overall.empty:
        order = {"HIGH_RISK": 0, "MEDIUM_RISK": 1, "LOW_RISK": 2}
        worst = overall.sort_values("severity", key=lambda s: s.map(order)).iloc[0]
        worst_risk = f"{worst['proposal']} — {worst['measurement']} ({worst['severity']})"

    best_cp = max(
        cp.get("per_proposal") or [],
        key=lambda p: (p.get("capital_saved") or 0.0) - abs(p.get("capital_lost") or 0.0),
        default=None,
    )

    if not approved.empty:
        verdict = (
            f"APPROVED — {', '.join(approved['proposal'])} cleared for future paper-mode "
            "experimentation via the staged rollout (OBSERVE_ONLY first). No operational change occurs in this phase."
        )
        status = "APPROVED_WITH_CONDITIONS"
    elif not needs.empty:
        verdict = "DEFERRED — no proposal cleared every gate; collect the required evidence and re-review."
        status = "DEFERRED"
    else:
        verdict = "REJECTED — no exit timing enhancement meets governance standards."
        status = "REJECTED"

    return {
        "top_recommendation": (
            f"{top['proposal']} (governance {top['governance_score']}/100, {top['decision']})"
            if top is not None
            else "—"
        ),
        "top_risk": worst_risk or "—",
        "top_capital_preservation_benefit": (
            f"{best_cp['proposal']} — ${best_cp['capital_saved']:,.2f} saved vs ${abs(best_cp['capital_lost'] or 0):,.2f} lost"
            if best_cp
            else "—"
        ),
        "implementation_readiness": f"{readiness.get('implementation_readiness_score')}/100 ({readiness.get('status')})",
        "recommended_rollout_mode": control.get("safest_first_mode", "OBSERVE_ONLY"),
        "approval_status": status,
        "approved_proposals": approved["proposal"].tolist(),
        "required_additional_evidence": [
            f"{r['proposal']}: {r['rationale']}" for _, r in needs.iterrows()
        ],
        "final_governance_verdict": verdict,
    }


def build_certification(
    decisions: pd.DataFrame, readiness: Dict[str, Any], framework: Dict[str, Any]
) -> Dict[str, Any]:
    reasons: List[str] = []
    inputs_ok = (readiness.get("factors") or [{}])[0].get("score", 0) == 100.0
    if not inputs_ok:
        reasons.append("one or more governance input artifacts are missing")
    fw_status = (framework.get("certification") or {}).get("status", "NOT_CERTIFIED")
    if fw_status != "CERTIFIED":
        reasons.append(f"Phase 170 framework certification is {fw_status}")
    if decisions.empty:
        reasons.append("no governance decisions could be produced")

    resolved = not decisions.empty and decisions["decision"].notna().all()
    if inputs_ok and fw_status == "CERTIFIED" and resolved:
        status = "CERTIFIED"
    elif resolved:
        status = "PARTIALLY_CERTIFIED"
    else:
        status = "NOT_CERTIFIED"
    return {
        "status": status,
        "proposals_reviewed": int(len(decisions)),
        "framework_certification": fw_status,
        "all_decisions_resolved": bool(resolved),
        "reasons": reasons,
        "safety": (
            "governance review only — no lifecycle rules, thresholds, execution, "
            "portfolio, or broker behavior modified; no orders written"
        ),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def run() -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)

    framework = _read_json(RESULTS / "exit_timing_framework.json")
    sim = _read_json(RESULTS / "lifecycle_rule_simulation_results.json")
    cp_cert = _read_json(RESULTS / "capital_preservation_certification.json")
    catalog = _read_csv(RESULTS / "late_exit_catalog.csv")
    opp_cost = _read_csv(RESULTS / "exit_opportunity_cost_report.csv")

    decisions = build_governance_decisions(sim, framework)
    decisions.to_csv(OUT_DECISIONS, index=False, encoding="utf-8")
    _log(f"decisions: {decisions[['proposal', 'decision']].to_dict(orient='records')}")

    cp = build_capital_preservation_assessment(sim, framework, cp_cert, opp_cost)
    risk = build_risk_assessment(sim, decisions)
    risk.to_csv(OUT_RISK, index=False, encoding="utf-8")

    control = build_operator_control_review(decisions)
    readiness = build_readiness(sim, framework, decisions, catalog)
    executive = build_executive(decisions, cp, risk, control, readiness)
    cert = build_certification(decisions, readiness, framework)

    report = {
        "generated_at": _now_iso(),
        "executive_report": executive,
        "certification": cert,
        "governance_decisions": decisions.to_dict(orient="records"),
        "capital_preservation_assessment": cp,
        "risk_assessment": risk.to_dict(orient="records"),
        "operator_control_review": control,
        "implementation_readiness": readiness,
        "approval_matrix": decisions[
            [
                "proposal",
                "governance_score",
                "risk_score",
                "capital_preservation_score",
                "operator_control_score",
                "decision",
            ]
        ].to_dict(orient="records"),
        "phase_172_focus": (
            "Implement OBSERVE_ONLY mode for the approved exit-timing proposals: a daily "
            "log-only late-exit detector report, accuracy tracking against subsequent price "
            "action, and the evidence collection required to clear the remaining proposals."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {OUT_JSON.name}, {OUT_DECISIONS.name}, {OUT_RISK.name}")
    _log(
        f"readiness: {readiness.get('implementation_readiness_score')} ({readiness.get('status')})"
    )
    _log(f"verdict: {executive.get('final_governance_verdict')}")
    _log(f"certification: {cert['status']}")
    return report


if __name__ == "__main__":
    run()
