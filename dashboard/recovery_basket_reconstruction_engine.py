"""
TRITON Phase 186 — Recovery Basket Reconstruction Engine dashboard (read-only).
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
    "ELITE_RECOVERY_CANDIDATE": "🟢 ELITE",
    "STRONG_RECOVERY_CANDIDATE": "🟡 STRONG",
    "ACCEPTABLE_RECOVERY_CANDIDATE": "🟠 ACCEPTABLE",
    "UNSUITABLE": "🔴 UNSUITABLE",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 [RECOVERY_BASKET_RECONSTRUCTION_REPORT]")
    ex = report.get("executive_report") or {}
    fail = ex.get("current_recovery_failure") or {}
    st.error(
        f"Legacy basket {fail.get('legacy_basket')} NOT EXECUTABLE — {fail.get('reason', '—')[:120]}…",
        icon="⛔",
    )
    rec = ex.get("recommended_basket") or report.get("recommended_basket") or {}
    st.markdown(
        f"**Recommended:** {rec.get('basket_name') or '—'} · "
        f"CPS → **{rec.get('projected_cps') or ex.get('projected_cps')}** · "
        f"Escalation → **{rec.get('projected_escalation') or ex.get('projected_escalation')}** · "
        f"Recovery prob: **{rec.get('recovery_probability') or ex.get('recovery_probability')}**"
    )
    st.info(str(ex.get("executive_recommendation") or ""), icon="💡")
    qs = ex.get("executive_questions") or {}
    with st.expander("Executive Q&A"):
        for q, a in qs.items():
            st.markdown(f"**{q.replace('_', ' ').title()}** — {a}")


def _render_failure(report: Dict[str, Any]) -> None:
    st.markdown("#### ⛔ Current Recovery Failure")
    state = report.get("current_state") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("QQQ Weight", f"{(state.get('holdings') or {}).get('QQQ', 100):.1f}%")
    c2.metric("CPS", str(state.get("current_cps", "—")))
    c3.metric("Escalation", str(state.get("escalation", "—")))
    c4.metric("Governance", str(state.get("governance_status", "—")))
    st.caption(str(state.get("legacy_failure_reason") or ""))
    legacy = state.get("legacy_plan_weights") or {}
    if legacy:
        st.markdown(f"**Obsolete plan:** {legacy}")


def _render_candidates(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Candidate Rankings")
    df = _read_csv_safe(results_dir / "recovery_basket_candidates.csv")
    if df.empty:
        df = pd.DataFrame(report.get("candidate_universe") or [])
    if df.empty:
        st.info("No candidates.", icon="ℹ️")
        return
    viable = (
        df[df["recovery_classification"] != "UNSUITABLE"]
        if "recovery_classification" in df.columns
        else df
    )
    view = viable.head(20).copy() if not viable.empty else df.head(20).copy()
    if "recovery_classification" in view.columns:
        view["recovery_classification"] = view["recovery_classification"].map(
            lambda x: _CLASS.get(str(x), str(x))
        )
    cols = [
        c
        for c in [
            "symbol",
            "effective_stance",
            "confidence",
            "edge_score",
            "diversification_score",
            "correlation_to_QQQ",
            "cps_improvement_potential",
            "recovery_classification",
        ]
        if c in view.columns
    ]
    st.dataframe(view[cols] if cols else view, use_container_width=True, hide_index=True)
    if "diversification_score" in viable.columns and not viable.empty:
        chart = viable.head(10)[["symbol", "diversification_score"]].copy()
        if not _plotly_bar(chart, "symbol", "diversification_score", "Diversification Score"):
            st.bar_chart(chart.set_index("symbol")["diversification_score"])


def _render_baskets(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🧺 Basket Comparisons")
    df = _read_csv_safe(results_dir / "recovery_basket_scenarios.csv")
    if df.empty:
        df = pd.DataFrame(report.get("basket_options") or [])
    if df.empty:
        st.info("No basket scenarios.", icon="ℹ️")
        return
    show_cols = [
        c
        for c in [
            "basket_name",
            "projected_cps",
            "projected_escalation",
            "projected_governance_status",
            "recovery_probability",
            "required_qqq_reduction",
            "candidates_required",
            "risk_score",
        ]
        if c in df.columns
    ]
    st.dataframe(df[show_cols] if show_cols else df, use_container_width=True, hide_index=True)
    if "projected_cps" in df.columns:
        chart = df[["basket_name", "projected_cps"]].copy()
        if not _plotly_bar(chart, "basket_name", "projected_cps", "Projected CPS by Basket"):
            st.bar_chart(chart.set_index("basket_name")["projected_cps"])


def _render_governance_recovery(report: Dict[str, Any]) -> None:
    st.markdown("#### 🟢 Projected Governance Recovery")
    proj = report.get("recovery_projection") or {}
    val = report.get("recovery_validation") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Current CPS", str(proj.get("current_cps", "—")))
    c2.metric("Projected CPS", str(proj.get("projected_cps_recommended", "—")))
    c3.metric("CPS Delta", str(proj.get("cps_delta", "—")))
    c4, c5 = st.columns(2)
    c4.metric("YELLOW Reachable", "Yes" if val.get("yellow_reachable") else "No")
    c5.metric("GREEN Reachable", "Yes" if val.get("green_reachable") else "No")
    st.caption(
        f"Fastest: {val.get('fastest_recovery_basket')} · "
        f"Lowest risk: {val.get('lowest_risk_basket')} · "
        f"Most efficient: {val.get('most_efficient_basket')}"
    )


def _render_replacement(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔄 Replacement Analysis")
    repl = report.get("replacement_analysis") or {}
    c1, c2 = st.columns(2)
    c1.markdown(f"**HD →** {repl.get('hd_replacement', '—')}")
    c1.caption(", ".join(repl.get("hd_replacement_alternatives") or []))
    c2.markdown(f"**XLK →** {repl.get('xlk_replacement', '—')}")
    c2.caption(", ".join(repl.get("xlk_replacement_alternatives") or []))
    st.markdown(str(repl.get("thesis_change_summary") or ""))


def _render_risk(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚠️ Risk Analysis")
    risk = report.get("risk_analysis") or {}
    st.json(risk.get("basket_risk_comparison") or {})


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_recovery_basket_reconstruction_engine(results_dir: Path) -> None:
    st.markdown("### 🧺 Recovery Basket Reconstruction Engine")
    st.caption(
        "Rebuilds recovery basket from authoritative lifecycle BUY/ADD candidates. "
        "Refresh with `python -m services.recovery_basket_reconstruction_engine`."
    )
    report = _read_json_safe(results_dir / "recovery_basket_reconstruction.json")
    if not report:
        st.warning(
            "No reconstruction found. Run `python -m services.recovery_basket_reconstruction_engine`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_failure(report)
    _render_candidates(results_dir, report)
    _render_baskets(results_dir, report)
    _render_governance_recovery(report)
    _render_replacement(report)
    _render_risk(report)
    _render_certification(report)
