"""
TRITON Phase 176 — Diversification Basket Feasibility Analysis dashboard.
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


_FEAS_BADGE = {
    "IMMEDIATELY_FEASIBLE": "🟢 IMMEDIATELY_FEASIBLE",
    "FEASIBLE_WITH_TIME": "🟡 FEASIBLE_WITH_TIME",
    "CURRENTLY_UNAVAILABLE": "🔴 CURRENTLY_UNAVAILABLE",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Summary")
    ex = report.get("executive_report") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Feasibility", str(ex.get("current_feasibility", "—")))
    c2.metric("Assemblable Now", str(ex.get("basket_assemblable_now", "—")))
    c3.metric("Feasible CPS", str(ex.get("projected_cps_feasible_only", "—")))
    c4.metric("Full Basket CPS", str(ex.get("projected_cps_full_basket", "—")))
    st.markdown(f"**Recovery basket:** {', '.join(ex.get('recovery_basket') or [])}")
    st.markdown(f"**Missing:** {', '.join(ex.get('missing_components') or []) or 'none'}")
    st.markdown(f"**Fastest path:** {ex.get('fastest_achievable_recovery_path') or '—'}")


def _render_key_questions(report: Dict[str, Any]) -> None:
    st.markdown("#### ❓ Key Questions")
    kq = report.get("key_questions") or {}
    for label, key in [
        ("Can HD be acquired?", "can_hd_be_acquired"),
        ("Can XLK be acquired?", "can_xlk_be_acquired"),
        ("Can BITO be acquired?", "can_bito_be_acquired"),
    ]:
        ans = kq.get(key) or {}
        badge = _FEAS_BADGE.get(str(ans.get("feasibility")), str(ans.get("feasibility")))
        st.markdown(f"- **{label}** {badge}")
        if ans.get("blockers"):
            st.caption(str(ans["blockers"]))
    asm = kq.get("can_recovery_basket_be_assembled") or {}
    st.markdown(
        f"- **Basket assemblable now:** {'✅' if asm.get('now') else '❌'} "
        f"(available: {', '.join(asm.get('immediately_available') or []) or 'none'})"
    )
    if asm.get("missing_reason"):
        st.warning(f"Missing: {asm.get('missing_reason')}", icon="⚠️")


def _render_candidates(results_dir: Path) -> None:
    st.markdown("#### 🎯 Feasibility Candidates")
    df = _read_csv_safe(results_dir / "diversification_feasibility_candidates.csv")
    if df.empty:
        st.info("No candidates evaluated.", icon="ℹ️")
        return
    view = df.copy()
    view["feasibility"] = view["feasibility"].map(lambda f: _FEAS_BADGE.get(str(f), str(f)))
    st.dataframe(view, use_container_width=True, hide_index=True)


def _render_scenarios(results_dir: Path) -> None:
    st.markdown("#### 🔬 Scenario Analysis")
    df = _read_csv_safe(results_dir / "diversification_feasibility_scenarios.csv")
    if df.empty:
        st.info("No scenarios.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    if "recovery_probability_pct" in df.columns:
        chart = df[["scenario", "recovery_probability_pct"]].copy()
        chart["scenario"] = chart["scenario"].str.replace("Scenario ", "S")
        if not _plotly_bar(chart, "scenario", "recovery_probability_pct", "Recovery Probability"):
            st.bar_chart(chart.set_index("scenario")["recovery_probability_pct"])


def _render_lifecycle(report: Dict[str, Any]) -> None:
    st.markdown("#### 🧬 Lifecycle Status")
    summary = (report.get("executive_report") or {}).get("lifecycle_status_summary") or {}
    for label, key in [
        ("Immediately feasible", "immediately_feasible"),
        ("Feasible with time", "feasible_with_time"),
        ("Currently unavailable", "currently_unavailable"),
    ]:
        items = summary.get(key) or []
        st.markdown(f"**{label}:** {', '.join(items) if items else '—'}")


def render_diversification_feasibility_analysis(results_dir: Path) -> None:
    st.markdown("### ✅ Diversification Basket Feasibility")
    st.caption(
        "Validates whether the Phase 175 recovery basket can be built from current "
        "lifecycle-approved opportunities. Refresh with "
        "`python -m services.diversification_feasibility_analysis`."
    )

    report = _read_json_safe(results_dir / "diversification_feasibility_analysis.json")
    if not report:
        st.info(
            "No report yet. Run `python -m services.diversification_feasibility_analysis`.",
            icon="ℹ️",
        )
        return

    _render_executive(report)
    _render_key_questions(report)
    st.divider()

    section = st.radio(
        "Section",
        ["🎯 Candidates", "🔬 Scenarios", "🧬 Lifecycle Status"],
        horizontal=True,
        key="dfa_section",
    )
    if section == "🎯 Candidates":
        _render_candidates(results_dir)
    elif section == "🔬 Scenarios":
        _render_scenarios(results_dir)
    else:
        _render_lifecycle(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
