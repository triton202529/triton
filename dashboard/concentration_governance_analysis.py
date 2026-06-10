"""
TRITON Phase 174 — Concentration Governance Paradox Analysis dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _money, _num, _plotly_bar


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


_CLASS_BADGE = {
    "REDUCES_CONCENTRATION": "🟢 REDUCES_CONCENTRATION",
    "INCREASES_CONCENTRATION": "🔴 INCREASES_CONCENTRATION",
    "NEUTRAL": "🟡 NEUTRAL",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Summary")
    ex = report.get("executive_report") or {}
    paradox = report.get("paradox_analysis") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Concentration", f"{_num(ex.get('current_concentration_pct'))}%")
    c2.metric("Escalation State", str(ex.get("current_escalation_state", "—")))
    c3.metric("Gov Authorized", str(ex.get("current_governance_authorized", "—")))
    c4.metric("Paradox Detected", "YES" if paradox.get("paradox_detected") else "NO")
    st.markdown(f"- **Best Candidate:** {ex.get('best_diversification_candidate') or '—'}")
    st.markdown(
        f"- **Reduction Potential:** {_num(ex.get('concentration_reduction_potential_pp'))} pp"
    )
    st.markdown(f"- **XLK Reduces Concentration:** {ex.get('xlk_reduces_concentration')}")
    st.markdown(f"- **Fastest Path to YELLOW:** {ex.get('fastest_path_to_yellow') or '—'}")
    st.markdown(f"- **Fastest Path to GREEN:** {ex.get('fastest_path_to_green') or '—'}")
    if paradox.get("paradox_summary"):
        st.warning(str(paradox["paradox_summary"]), icon="⚖️")


def _render_candidates(results_dir: Path) -> None:
    st.markdown("#### 🎯 Diversification Candidates (OPEN_NEW)")
    df = _read_csv_safe(results_dir / "diversification_candidates.csv")
    if df.empty:
        st.info("No OPEN_NEW candidates in the allocation pipeline.", icon="ℹ️")
        return
    view = df.copy()
    view["classification"] = view["classification"].map(lambda c: _CLASS_BADGE.get(str(c), str(c)))
    st.dataframe(view, use_container_width=True, hide_index=True)
    if "diversification_benefit_pct_points" in df.columns:
        chart = df[["ticker", "diversification_benefit_pct_points"]].sort_values(
            "diversification_benefit_pct_points", ascending=False
        )
        if not _plotly_bar(
            chart,
            "ticker",
            "diversification_benefit_pct_points",
            "Concentration Reduction Benefit (pp)",
        ):
            st.bar_chart(chart.set_index("ticker")["diversification_benefit_pct_points"])


def _render_scenarios(results_dir: Path) -> None:
    st.markdown("#### 📊 Scenario Analysis")
    df = _read_csv_safe(results_dir / "concentration_scenarios.csv")
    if df.empty:
        st.info("No scenarios generated.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    if "capital_preservation_score" in df.columns:
        chart = df[["scenario", "capital_preservation_score"]].copy()
        chart["scenario"] = chart["scenario"].str.replace("Scenario ", "S")
        if not _plotly_bar(chart, "scenario", "capital_preservation_score", "CPS by Scenario"):
            st.bar_chart(chart.set_index("scenario")["capital_preservation_score"])


def _render_xlk(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔬 XLK Key Question")
    xlk = (report.get("paradox_analysis") or {}).get("xlk_analysis") or {}
    if not xlk:
        st.info("No XLK analysis available.", icon="ℹ️")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Reduces Name Concentration", str(xlk.get("would_reduce_name_concentration", "—")))
    c2.metric("Projected Largest Weight", f"{_num(xlk.get('projected_largest_weight_pct'))}%")
    c3.metric("Tech Sector After", f"{_num(xlk.get('sector_concentration_after_pct'))}%")
    st.info(str(xlk.get("note") or ""), icon="💡")


def _render_governance_impact(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚖️ Governance Impact")
    ex = report.get("executive_report") or {}
    c1, c2 = st.columns(2)
    c1.metric("Would Trade Improve Gov?", str(ex.get("would_trade_improve_governance", "—")))
    c2.metric(
        "Blocking Beneficial Trade?", str(ex.get("governance_blocking_beneficial_trade", "—"))
    )
    st.markdown(f"**Projected impact:** {ex.get('projected_governance_impact') or '—'}")
    cur = report.get("current_portfolio") or {}
    st.caption(f"Block reason: {cur.get('governance_block_reason') or '—'}")


def _render_methodology(report: Dict[str, Any]) -> None:
    st.markdown("#### 📐 Methodology")
    st.json(report.get("methodology") or {})
    st.caption(str(report.get("disclaimer") or ""))


def render_concentration_governance_analysis(results_dir: Path) -> None:
    st.markdown("### ⚖️ Concentration Governance Paradox Analysis")
    st.caption(
        "Analysis only — investigates whether governance blocks beneficial diversification. "
        "Refresh with `python -m services.concentration_governance_analysis`."
    )

    report = _read_json_safe(results_dir / "concentration_governance_analysis.json")
    if not report:
        st.info(
            "No analysis report yet. Run "
            "`python -m services.concentration_governance_analysis` to generate it.",
            icon="ℹ️",
        )
        return

    _render_executive(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "🎯 Candidates",
            "📊 Scenarios",
            "🔬 XLK Analysis",
            "⚖️ Governance Impact",
            "📐 Methodology",
        ],
        horizontal=True,
        key="cga_section",
    )
    if section == "🎯 Candidates":
        _render_candidates(results_dir)
    elif section == "📊 Scenarios":
        _render_scenarios(results_dir)
    elif section == "🔬 XLK Analysis":
        _render_xlk(report)
    elif section == "⚖️ Governance Impact":
        _render_governance_impact(report)
    else:
        _render_methodology(report)
