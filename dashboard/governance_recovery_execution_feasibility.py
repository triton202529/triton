"""
TRITON Phase 203 — Governance Recovery Execution Feasibility dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st


def _read_json_safe(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚙️ [GOVERNANCE_RECOVERY_EXECUTION_FEASIBILITY_REPORT]")
    ex = report.get("executive_recommendation") or {}
    readiness = report.get("execution_readiness") or {}
    paradox = report.get("governance_paradox_recheck") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Readiness Score", str(readiness.get("execution_readiness_score", "—")))
    c2.metric("Classification", str(readiness.get("classification", "—")))
    c3.metric("Paradox Status", str(paradox.get("status", "—")))
    c4.metric(
        "Phase 202 Executable",
        f"{(ex.get('current_recovery_feasibility') or {}).get('phase202_scenarios_executable', '—')}"
        f"/{(ex.get('current_recovery_feasibility') or {}).get('phase202_scenarios_total', '—')}",
    )
    st.info(paradox.get("explanation", "—"), icon="💡")


def _render_readiness_score(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Recovery Readiness Score")
    readiness = report.get("execution_readiness") or {}
    components = readiness.get("components") or {}
    if components:
        df = pd.DataFrame([{"component": k, "score": v} for k, v in components.items()])
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_action_matrix(results_dir: Path) -> None:
    st.markdown("#### 🎯 Action Feasibility Matrix")
    df = _read_csv_safe(results_dir / "recovery_action_readiness.csv")
    if df.empty:
        st.info("No action readiness data.", icon="ℹ️")
        return
    cols = [
        c
        for c in [
            "rank",
            "action",
            "symbol",
            "execution_eligibility",
            "lifecycle_status",
            "allocation_status",
            "expected_cps_impact",
            "diversification_contribution_pct",
            "governance_dependency",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_candidates(report: Dict[str, Any]) -> None:
    st.markdown("#### ✅ Candidate Eligibility")
    actions = report.get("recovery_action_readiness") or []
    focus = [a for a in actions if a.get("symbol") in ("QQQ", "HD", "XLK", "BITO")]
    if not focus:
        st.info("No candidate data.", icon="ℹ️")
        return
    df = pd.DataFrame(focus)
    cols = [
        c
        for c in [
            "symbol",
            "execution_eligibility",
            "confidence_score",
            "edge_score",
            "allocation_score",
            "blockers",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_concentration_paths(report: Dict[str, Any]) -> None:
    st.markdown("#### 📉 Concentration Reduction Paths")
    validations = report.get("phase202_scenario_validation") or []
    if not validations:
        st.info("No scenario validation.", icon="ℹ️")
        return
    df = pd.DataFrame(
        [
            {
                "scenario": v.get("scenario_name"),
                "status": v.get("execution_status"),
                "rank": v.get("rank_classification"),
                "days_to_yellow": v.get("realistic_days_to_yellow"),
                "blocking": "; ".join(v.get("blocking_factors") or []),
            }
            for v in validations
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_sequences(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛤 Recovery Sequences")
    sequences = report.get("recovery_sequences") or {}
    for name, seq in sequences.items():
        st.markdown(
            f"**{name.replace('_', ' ')}** — {seq.get('target_scenario', '—')} "
            f"({seq.get('estimated_days_to_yellow', '—')} days)"
        )
        for step in seq.get("steps") or []:
            st.markdown(
                f"  {step.get('step')}. {step.get('action')} "
                f"_[{step.get('feasibility')}]_ — {step.get('rationale')}"
            )


def _render_constraints(results_dir: Path) -> None:
    st.markdown("#### 🚧 Remaining Constraints")
    df = _read_csv_safe(results_dir / "recovery_execution_constraints.csv")
    if df.empty:
        st.info("No constraints data.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_failure_modes(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚠️ Failure Modes")
    failures = report.get("failure_mode_analysis") or []
    if not failures:
        st.info("No failure mode data.", icon="ℹ️")
        return
    df = pd.DataFrame(failures)
    cols = [
        c
        for c in [
            "risk",
            "probability",
            "severity",
            "impact_on_yellow_days",
            "impact_on_green_days",
            "mitigation_evidence",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_recommendations(report: Dict[str, Any]) -> None:
    st.markdown("#### 💡 Executive Recommendation")
    ex = report.get("executive_recommendation") or {}
    qs = ex.get("executive_questions") or {}
    st.markdown(
        f"**Can Phase 202 scenarios occur?** {qs.get('can_phase202_scenarios_occur', '—')} · "
        f"**Paradox solved?** {qs.get('paradox_solved', '—')} · "
        f"**Prepared when authorized?** {qs.get('prepared_for_execution_when_authorized', '—')}"
    )
    for p in ex.get("recommended_recovery_preparation") or []:
        st.markdown(f"- {p}")


def render_governance_recovery_execution_feasibility(results_dir: Path) -> None:
    st.markdown("### ⚙️ Governance Recovery Action Readiness & Execution Feasibility Intelligence")
    st.caption(
        "Validates whether Phase 202 recovery scenarios are executable under current conditions. "
        "Refresh with `python -m services.governance_recovery_execution_feasibility`."
    )
    report = _read_json_safe(results_dir / "governance_recovery_execution_feasibility.json")
    if not report:
        st.warning(
            "No feasibility report. Run `python -m services.governance_recovery_execution_feasibility`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_readiness_score(report)
    _render_action_matrix(results_dir)
    _render_candidates(report)
    _render_concentration_paths(report)
    _render_sequences(report)
    _render_constraints(results_dir)
    _render_failure_modes(report)
    _render_recommendations(report)
