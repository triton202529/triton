"""
TRITON Phase 179 — 🎯 Governance Recovery Threshold Optimizer dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _num, _plotly_bar


def _read_json_safe(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


_EFF = {
    "OPTIMAL": "🟢 OPTIMAL",
    "EFFICIENT": "🟢 EFFICIENT",
    "MODERATE": "🟡 MODERATE",
    "INEFFICIENT": "🔴 INEFFICIENT",
}


def _render_current_state(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Current State")
    cs = report.get("current_state") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current CPS", str(cs.get("current_cps", "—")))
    c2.metric("Governance", str(cs.get("current_governance_status", "—")))
    c3.metric("Escalation", str(cs.get("current_escalation", "—")))
    c4.metric("Concentration %", str(cs.get("largest_concentration_pct", "—")))
    c5, c6 = st.columns(2)
    c5.metric("Distance To YELLOW", str(cs.get("distance_to_yellow", "—")))
    c6.metric("Distance To GREEN", str(cs.get("distance_to_green", "—")))


def _render_thresholds(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Threshold Discovery")
    yt = report.get("yellow_threshold") or {}
    gt = report.get("green_threshold") or {}
    st.markdown("**YELLOW Threshold**")
    c1, c2, c3 = st.columns(3)
    c1.metric("CPS Required", str(yt.get("cps_required", 75)))
    c2.metric("Min HD %", str(yt.get("minimum_hd_allocation_pct", "—")))
    c3.metric("Min XLK %", str(yt.get("minimum_xlk_allocation_pct", "—")))
    st.json(yt.get("minimum_yellow_recovery_plan") or {})
    st.markdown("**GREEN Threshold**")
    c4, c5 = st.columns(2)
    c4.metric("CPS Required", str(gt.get("cps_required", 90)))
    c5.metric("BITO Required", "Yes" if gt.get("bito_required") else "No")
    st.json(gt.get("minimum_green_recovery_plan") or {})


def _render_optimization(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛤️ Recovery Optimization")
    paths = report.get("recovery_paths") or {}
    labels = [
        ("Optimal Recovery Path", "OPTIMAL_RECOVERY_PATH"),
        ("Fastest Recovery Path", "FASTEST_RECOVERY_PATH"),
        ("Lowest Risk Path", "LOWEST_RISK_RECOVERY_PATH"),
        ("Lowest Disruption Path", "LOWEST_DISRUPTION_PATH"),
        ("Highest Probability Path", "HIGHEST_PROBABILITY_PATH"),
    ]
    for title, key in labels:
        p = paths.get(key) or {}
        if not p:
            continue
        with st.expander(f"**{title}**", expanded=key == "OPTIMAL_RECOVERY_PATH"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Allocation Change %", str(p.get("allocation_change_pct", "—")))
            c2.metric("Projected CPS", str(p.get("projected_cps", "—")))
            c3.metric("Efficiency", str(p.get("recovery_efficiency_score", "—")))
            st.json(p.get("weights") or {})


def _render_curve(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Recovery Curve")
    df = _read_csv_safe(results_dir / "recovery_efficiency_curve.csv")
    if df.empty:
        df = pd.DataFrame(report.get("recovery_curve") or [])
    if df.empty:
        st.info("No curve data.", icon="ℹ️")
        return
    df = df.sort_values("diversification_pct")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if "diversification_pct" in df.columns and "projected_cps" in df.columns:
        try:
            import plotly.express as px

            fig = px.line(
                df,
                x="diversification_pct",
                y="projected_cps",
                color="escalation_level",
                title="Diversification % vs Projected CPS",
                markers=True,
            )
            fig.add_hline(y=75, line_dash="dash", annotation_text="YELLOW")
            fig.add_hline(y=90, line_dash="dash", annotation_text="GREEN")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.line_chart(df.set_index("diversification_pct")[["projected_cps"]])


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Recommendation")
    ex = report.get("executive_report") or {}
    st.markdown(f"**[{ex.get('report_type', 'GOVERNANCE_RECOVERY_THRESHOLD_REPORT')}]**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current CPS", str(ex.get("current_cps", "—")))
    c2.metric("Expected Improvement", str(ex.get("expected_cps_improvement", "—")))
    c3.metric("Recovery Efficiency", str(ex.get("recovery_efficiency", "—")))
    c4.metric("Threshold Confidence", f"{_num(ex.get('threshold_confidence_pct'))}%")
    st.markdown(f"**Recommended plan:** `{ex.get('recommended_recovery_plan')}`")
    st.markdown(f"**Expected outcome:** {ex.get('expected_outcome') or '—'}")
    st.markdown(f"**Probability of success:** {ex.get('probability_of_success') or '—'}")
    rp = report.get("recovery_probability") or {}
    st.json(rp)


def _render_rankings(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Efficiency Rankings")
    df = _read_csv_safe(results_dir / "governance_recovery_thresholds.csv")
    if df.empty:
        df = pd.DataFrame(report.get("efficiency_rankings") or [])
    if df.empty:
        st.info("No rankings.", icon="ℹ️")
        return
    if "classification" in df.columns:
        df["classification"] = df["classification"].map(lambda x: _EFF.get(str(x), str(x)))
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)
    if "recovery_efficiency_score" in df.columns:
        top = df.nlargest(15, "recovery_efficiency_score")
        _plotly_bar(top, "allocation_change_pct", "recovery_efficiency_score", "Top Efficiency")


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification")
    cert = report.get("certification") or {}
    st.metric("Status", str(cert.get("status", "—")))
    st.success(str(cert.get("safety") or ""), icon="🔒")


def render_governance_recovery_threshold_optimizer(results_dir: Path) -> None:
    st.markdown("### 🎯 Governance Recovery Threshold Optimizer")
    st.caption(
        "Minimum diversification thresholds for governance recovery — analysis only. "
        "Refresh with `python -m services.governance_recovery_threshold_optimizer`."
    )

    report = _read_json_safe(results_dir / "governance_recovery_thresholds.json")
    if not report:
        st.info(
            "No threshold report yet. Run "
            "`python -m services.governance_recovery_threshold_optimizer`.",
            icon="ℹ️",
        )
        return

    _render_current_state(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "🎯 Threshold Discovery",
            "🛤️ Recovery Optimization",
            "📈 Recovery Curve",
            "📋 Executive Recommendation",
            "📊 Efficiency Rankings",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="grto_section",
    )
    if section == "🎯 Threshold Discovery":
        _render_thresholds(report)
    elif section == "🛤️ Recovery Optimization":
        _render_optimization(report)
    elif section == "📈 Recovery Curve":
        _render_curve(results_dir, report)
    elif section == "📋 Executive Recommendation":
        _render_executive(report)
    elif section == "📊 Efficiency Rankings":
        _render_rankings(results_dir, report)
    else:
        _render_certification(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
