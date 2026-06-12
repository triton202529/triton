"""
TRITON Phase 188 — Autonomous Portfolio Transition Planner dashboard (read-only).
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


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 [AUTONOMOUS_TRANSITION_EXECUTION_REPORT]")
    ex = report.get("executive_report") or {}
    cur = ex.get("current_state_summary") or {}
    tgt = ex.get("target_green_portfolio") or {}
    rec = ex.get("recommended_path") or {}
    st.markdown(
        f"**Current:** CPS {cur.get('cps')} · {cur.get('escalation')} · "
        f"QQQ {cur.get('weights', {}).get('QQQ', 100)}%"
    )
    st.markdown(
        f"**Target:** CPS {tgt.get('projected_cps')} · GREEN · "
        f"recovery prob {tgt.get('recovery_probability_pct')}%"
    )
    st.markdown(
        f"**Recommended:** {rec.get('path_id')} — {rec.get('path_name')} "
        f"({rec.get('cycles')} cycles)"
    )
    st.info(str(ex.get("executive_recommendation") or ""), icon="💡")
    qs = ex.get("executive_questions") or {}
    with st.expander("Executive Q&A"):
        for q, a in qs.items():
            st.markdown(f"**{q.replace('_', ' ').title()}** — {a}")


def _render_portfolio_comparison(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚖️ Current Portfolio vs Target")
    cur = report.get("current_state") or {}
    tgt = report.get("target_portfolio") or {}
    rows = []
    all_syms = set((cur.get("weights") or {}).keys()) | set((tgt.get("weights") or {}).keys())
    for sym in sorted(all_syms):
        rows.append(
            {
                "symbol": sym,
                "current_pct": (cur.get("weights") or {}).get(sym, 0),
                "target_pct": (tgt.get("weights") or {}).get(sym, 0),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_path_comparison(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛤️ Transition Path Comparison")
    paths = report.get("transition_paths") or []
    if not paths:
        st.info("No paths.", icon="ℹ️")
        return
    df = pd.DataFrame(paths)
    cols = [
        c
        for c in [
            "path_id",
            "path_name",
            "trading_cycles",
            "final_cps",
            "final_escalation",
            "execution_complexity",
            "estimated_slippage_exposure",
            "lifecycle_dependency_risk",
            "path_score",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_stages(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📅 Stage Progression (Recommended Path)")
    rec_id = (report.get("recommended_path") or {}).get("path_id")
    df = _read_csv_safe(results_dir / "autonomous_transition_stages.csv")
    if df.empty:
        stages = report.get("recommended_stages") or []
        df = pd.DataFrame(stages)
    if df.empty:
        st.info("No stages.", icon="ℹ️")
        return
    if rec_id and "path_id" in df.columns:
        view = df[df["path_id"] == rec_id]
    else:
        view = df
    cols = [
        c
        for c in [
            "stage",
            "trading_cycle",
            "projected_cps",
            "projected_escalation",
            "projected_governance_status",
            "execution_readiness",
            "lifecycle_gate",
        ]
        if c in view.columns
    ]
    st.dataframe(view[cols] if cols else view, use_container_width=True, hide_index=True)


def _render_cps_timeline(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 CPS Recovery Timeline")
    rec = report.get("recommended_path") or {}
    traj = rec.get("cps_progression") or (report.get("executive_report") or {}).get("cps_timeline")
    if not traj:
        st.info("No CPS timeline.", icon="ℹ️")
        return
    df = pd.DataFrame({"cycle": list(range(len(traj))), "projected_cps": traj})
    if not _plotly_bar(df, "cycle", "projected_cps", "CPS Progression by Cycle"):
        st.line_chart(df.set_index("cycle")["projected_cps"])
    st.caption(
        f"Start CPS: {(report.get('current_state') or {}).get('current_cps')} -> Final: {traj[-1]}"
    )


def _render_readiness(report: Dict[str, Any]) -> None:
    st.markdown("#### ✅ Execution Readiness & Autonomy")
    auto = report.get("autonomy_assessment") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Autonomous Ready", "No" if not auto.get("can_execute_autonomously") else "Yes")
    c2.metric("Readiness Score", str(auto.get("autonomous_readiness_score", "—")))
    c3.metric("Paper Permitted", str(auto.get("paper_execution_permitted", "—")))
    st.markdown(str(auto.get("rationale") or ""))


def _render_risk(results_dir: Path) -> None:
    st.markdown("#### ⚠️ Risk Assessment")
    df = _read_csv_safe(results_dir / "autonomous_transition_risk_assessment.csv")
    if df.empty:
        st.info("No risk data.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_checkpoints(report: Dict[str, Any]) -> None:
    st.markdown("#### 🏛 Governance Checkpoints")
    approvals = report.get("approval_requirements") or {}
    for cp in approvals.get("governance_checkpoints") or []:
        st.markdown(f"**{cp.get('checkpoint')}** — {cp.get('trigger')}: {cp.get('action')}")


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_autonomous_portfolio_transition_planner(results_dir: Path) -> None:
    st.markdown("### 🚀 Autonomous Portfolio Transition Planner")
    st.caption(
        "Governance-safe transition roadmap to GREEN target. "
        "Refresh with `python -m services.autonomous_portfolio_transition_planner`."
    )
    report = _read_json_safe(results_dir / "autonomous_transition_plan.json")
    if not report:
        st.warning(
            "No transition plan found. Run `python -m services.autonomous_portfolio_transition_planner`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_portfolio_comparison(report)
    _render_path_comparison(report)
    _render_stages(results_dir, report)
    _render_cps_timeline(report)
    _render_readiness(report)
    _render_risk(results_dir)
    _render_checkpoints(report)
    _render_certification(report)
