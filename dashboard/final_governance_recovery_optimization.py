"""
TRITON Phase 187 — Final Governance Recovery Optimization dashboard (read-only).
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


_CLASS = {
    "OPTIMAL": "🟢 OPTIMAL",
    "STRONG": "🟡 STRONG",
    "ACCEPTABLE": "🟠 ACCEPTABLE",
    "INEFFICIENT": "🔴 INEFFICIENT",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 [FINAL_GOVERNANCE_RECOVERY_OPTIMIZATION_REPORT]")
    ex = report.get("executive_recommendation") or {}
    gap = ex.get("current_cps_gap") or {}
    st.markdown(
        f"**CPS gap:** {gap.get('baseline_cps')} -> GREEN requires {gap.get('green_threshold')} "
        f"({gap.get('gap_points')} point(s))"
    )
    best = ex.get("best_recovery_path") or {}
    st.markdown(
        f"**Best path:** {best.get('path_id')} — {best.get('path_name')} "
        f"(CPS {best.get('projected_cps')}, {best.get('classification')})"
    )
    st.markdown(f"**Minimum change:** {ex.get('minimum_required_change') or '—'}")
    st.info(str(ex.get("executive_recommendation") or ""), icon="💡")
    qs = ex.get("executive_questions") or {}
    with st.expander("Executive Q&A"):
        for q, a in qs.items():
            st.markdown(f"**{q.replace('_', ' ').title()}** — {a}")


def _render_gap(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Current CPS Gap")
    base = report.get("baseline") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline CPS", str(base.get("projected_cps", "—")))
    c2.metric("GREEN Threshold", "90")
    c3.metric("Gap", str(base.get("cps_gap_to_green", "—")))
    c4.metric("Baseline Escalation", str(base.get("escalation", "—")))


def _render_sensitivity(results_dir: Path) -> None:
    st.markdown("#### 📈 CPS Sensitivity Analysis")
    df = _read_csv_safe(results_dir / "final_cps_sensitivity_matrix.csv")
    if df.empty:
        st.info("No sensitivity data.", icon="ℹ️")
        return
    green_hits = (
        df[df["reaches_green"] == True] if "reaches_green" in df.columns else df
    )  # noqa: E712
    st.markdown("**Minimum shifts reaching GREEN**")
    if not green_hits.empty and "allocation_change_pct" in green_hits.columns:
        st.dataframe(
            green_hits.sort_values("allocation_change_pct").head(15),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    if "cps_change" in df.columns:
        chart = df[df["allocation_change_pct"] == 5][["candidate", "cps_change"]].copy()
        if not chart.empty and not _plotly_bar(
            chart, "candidate", "cps_change", "CPS Change @ +5%"
        ):
            st.bar_chart(chart.set_index("candidate")["cps_change"])


def _render_paths(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🛤️ Recovery Path Comparison")
    df = _read_csv_safe(results_dir / "final_recovery_adjustments.csv")
    if df.empty:
        df = pd.DataFrame(report.get("optimization_paths") or [])
    if df.empty:
        st.info("No paths.", icon="ℹ️")
        return
    view = df.copy()
    if "classification" in view.columns:
        view["classification"] = view["classification"].map(lambda x: _CLASS.get(str(x), str(x)))
    cols = [
        c
        for c in [
            "adjustment",
            "path_name",
            "projected_cps",
            "projected_status",
            "projected_governance",
            "risk_score",
            "allocation_change_pct",
            "classification",
            "rank",
        ]
        if c in view.columns
    ]
    st.dataframe(view[cols] if cols else view, use_container_width=True, hide_index=True)
    if "projected_cps" in df.columns:
        chart = df[["adjustment", "projected_cps"]].copy()
        if not _plotly_bar(chart, "adjustment", "projected_cps", "Projected CPS by Path"):
            st.bar_chart(chart.set_index("adjustment")["projected_cps"])


def _render_green_prob(report: Dict[str, Any]) -> None:
    st.markdown("#### 🟢 GREEN Probability Forecast")
    gp = report.get("green_recovery_projection") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Allocation-Only GREEN", f"{gp.get('allocation_only_green_probability', '—')}%")
    c2.metric("BITO-Assisted @30d", f"{gp.get('bito_assisted_green_probability_30d', '—')}%")
    c3.metric("Best Scenario", str(gp.get("highest_confidence_scenario", "—")))


def _render_risk(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚠️ Risk Comparison")
    paths = report.get("optimization_paths") or []
    rows = [
        {
            "path": p.get("path_id"),
            "risk_score": p.get("additional_risk"),
            "complexity": p.get("implementation_complexity"),
            "reaches_green": p.get("reaches_green"),
        }
        for p in paths
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_final_governance_recovery_optimization(results_dir: Path) -> None:
    st.markdown("### 🎯 Final Governance Recovery Optimization Engine")
    st.caption(
        "Closes the final CPS gap from YELLOW to GREEN. "
        "Refresh with `python -m services.final_governance_recovery_optimization`."
    )
    report = _read_json_safe(results_dir / "final_governance_recovery_optimization.json")
    if not report:
        st.warning(
            "No optimization found. Run `python -m services.final_governance_recovery_optimization`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_gap(report)
    _render_sensitivity(results_dir)
    _render_paths(results_dir, report)
    _render_green_prob(report)
    _render_risk(report)
    _render_certification(report)
