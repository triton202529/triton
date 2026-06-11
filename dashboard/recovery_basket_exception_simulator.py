"""
TRITON Phase 182 — Recovery Basket Exception Simulation dashboard (read-only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _plotly_bar


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


_OUTCOME = {
    "SUCCESS": "🟢 SUCCESS",
    "PARTIAL_SUCCESS": "🟡 PARTIAL_SUCCESS",
    "FAILED": "🔴 FAILED",
}

_SAFETY = {
    "SAFE": "🟢 SAFE",
    "CONDITIONALLY_SAFE": "🟡 CONDITIONALLY_SAFE",
    "HIGH_RISK": "🟠 HIGH_RISK",
    "UNACCEPTABLE": "🔴 UNACCEPTABLE",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 [RECOVERY_BASKET_EXCEPTION_SIMULATION_REPORT]")
    ex = report.get("executive_recommendation") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Deadlock Resolved?",
        "✅ Yes" if ex.get("framework_resolves_deadlock") else "❌ No",
    )
    c2.metric(
        "Governance Intact?",
        "✅ Yes" if ex.get("governance_remains_intact") else "❌ No",
    )
    c3.metric(
        "Safety",
        _SAFETY.get(
            str((report.get("governance_safety_analysis") or {}).get("classification")), "—"
        ),
    )
    c4.metric(
        "Implementation Review?",
        "✅ Yes" if ex.get("advance_to_implementation_review") else "⏸ No",
    )
    st.markdown(str(ex.get("executive_summary") or ""))
    for c in ex.get("caveats") or []:
        st.caption(f"⚠ {c}")


def _render_deadlock_status(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Current Deadlock Status")
    cs = report.get("current_state") or {}
    dl = report.get("deadlock_analysis") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("CPS", str(cs.get("cps", "—")))
    c2.metric("Escalation", str(cs.get("escalation", "—")))
    c3.metric("Deadlock", str(cs.get("deadlock_classification", "—")))
    assets = cs.get("asset_status") or {}
    if assets:
        rows = [{"symbol": k, **v} for k, v in assets.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown(
        f"**Framework resolves deadlock (Scenario D):** "
        f"{'Yes' if dl.get('deadlock_resolved_by_framework') else 'No'}"
    )


def _render_scenarios(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Scenario Comparison")
    df = _read_csv_safe(results_dir / "recovery_basket_exception_scenarios.csv")
    if df.empty:
        df = pd.DataFrame(report.get("simulation_results") or [])
    if df.empty:
        st.info("No scenarios.", icon="ℹ️")
        return
    view = df.copy()
    if "outcome_classification" in view.columns:
        view["outcome_classification"] = view["outcome_classification"].map(
            lambda x: _OUTCOME.get(str(x), str(x))
        )
    st.dataframe(view, use_container_width=True, hide_index=True)
    if "exception_effectiveness_score" in df.columns:
        chart = df[["scenario", "exception_effectiveness_score"]].copy()
        if not _plotly_bar(
            chart,
            "scenario",
            "exception_effectiveness_score",
            "Exception Effectiveness by Scenario",
        ):
            st.bar_chart(chart.set_index("scenario")["exception_effectiveness_score"])


def _render_deadlock_resolution(report: Dict[str, Any]) -> None:
    st.markdown("#### ✅ Deadlock Resolution Analysis")
    dl = report.get("deadlock_analysis") or {}
    eff = report.get("effectiveness_analysis") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended Scenario", str(dl.get("recommended_scenario", "—")))
    c2.metric(
        "Outcome", _OUTCOME.get(str(dl.get("overall_outcome")), str(dl.get("overall_outcome", "—")))
    )
    c3.metric("CPS Improvement", str(eff.get("cps_improvement", "—")))
    checks = [
        ("Deadlock Resolved", dl.get("deadlock_resolved_by_framework")),
        ("Recovery Enabled", dl.get("recovery_enabled")),
        ("Governance Preserved", dl.get("governance_preserved")),
        ("Authorization Preserved", dl.get("authorization_preserved")),
        ("Risk Controls Preserved", dl.get("risk_controls_preserved")),
        ("Lifecycle Controls Preserved", dl.get("lifecycle_controls_preserved")),
    ]
    for label, ok in checks:
        st.markdown(f"- **{label}:** {'✅' if ok else '❌'}")


def _render_safety(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Governance Safety Analysis")
    sa = report.get("governance_safety_analysis") or {}
    st.metric("Classification", _SAFETY.get(str(sa.get("classification")), "—"))
    for t in sa.get("control_tests") or []:
        icon = "✅" if t.get("passed") else "❌"
        st.markdown(f"{icon} **{t.get('control')}** — {t.get('note')}")
    st.markdown("**Safety vectors**")
    for v in sa.get("safety_vectors") or []:
        icon = "✅" if v.get("passed") else "❌"
        st.markdown(f"{icon} {v.get('question')}: {v.get('answer')}")
    if sa.get("lifecycle_drift_detected"):
        st.warning(str(sa.get("lifecycle_drift_note") or "Lifecycle drift detected"), icon="⚠️")


def _render_audit(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📜 Audit Simulation")
    audit = report.get("audit_simulation") or {}
    c1, c2 = st.columns(2)
    c1.metric("Events Simulated", str(audit.get("event_count", "—")))
    c2.metric(
        "Coverage Complete",
        "✅ Yes" if audit.get("audit_coverage_complete") else "❌ No",
    )
    df = _read_csv_safe(results_dir / "recovery_basket_exception_audit_simulation.csv")
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_recovery_impact(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Recovery Impact")
    eff = report.get("effectiveness_analysis") or {}
    comp = eff.get("baseline_vs_recommended") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Concentration Reduction", f"{eff.get('concentration_reduction_pct', '—')}%")
    c2.metric("Recovery Acceleration", f"{eff.get('recovery_acceleration_days', '—')} days")
    c3.metric("Yellow Prob Δ", str(eff.get("yellow_probability_change", "—")))
    st.markdown(
        f"Escalation: {comp.get('escalation_baseline')} → {comp.get('escalation_recommended')} · "
        f"Governance: {comp.get('governance_baseline')} → {comp.get('governance_recommended')}"
    )


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_recovery_basket_exception_simulator(results_dir: Path) -> None:
    st.markdown("### 🧪 Recovery Basket Exception Simulation")
    st.caption(
        "Validates Phase 181 framework under simulated scenarios — no implementation. "
        "Refresh with `python -m services.recovery_basket_exception_simulator`."
    )
    report = _read_json_safe(results_dir / "recovery_basket_exception_simulation.json")
    if not report:
        st.warning(
            "No simulation found. Run `python -m services.recovery_basket_exception_simulator`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_deadlock_status(report)
    _render_scenarios(results_dir, report)
    _render_deadlock_resolution(report)
    _render_recovery_impact(report)
    _render_safety(report)
    _render_audit(results_dir, report)
    _render_certification(report)
