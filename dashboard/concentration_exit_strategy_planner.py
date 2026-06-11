"""
TRITON Phase 181 — 📉 Concentration Exit Strategy Planner dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _num


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


_RISK = {
    "LOW_RISK": "🟢 LOW_RISK",
    "MODERATE_RISK": "🟡 MODERATE_RISK",
    "HIGH_RISK": "🔴 HIGH_RISK",
}


def _render_concentration(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Current Concentration")
    cs = report.get("current_state") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("QQQ Weight %", str(cs.get("qqq_concentration_pct", "—")))
    c2.metric("Current CPS", str(cs.get("current_cps", "—")))
    c3.metric("Escalation", str(cs.get("current_escalation", "—")))
    c4.metric("Governance", str(cs.get("current_governance_status", "—")))
    st.caption(
        f"HD: {cs.get('hd_status')} | XLK: {cs.get('xlk_status')} | BITO: {cs.get('bito_status')}"
    )


def _render_exit_plans(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Exit Plans")
    for key, label in [
        ("conservative_plan", "Conservative (5% steps)"),
        ("balanced_plan", "Balanced (10% steps)"),
        ("aggressive_plan", "Aggressive (15% steps)"),
    ]:
        plan = report.get(key) or {}
        risk = _RISK.get(str(plan.get("risk_classification")), str(plan.get("risk_classification")))
        with st.expander(f"**{label}** — {risk}", expanded=key == "balanced_plan"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Steps", str(plan.get("total_steps", "—")))
            c2.metric("Steps to YELLOW", str(plan.get("steps_to_yellow", "—")))
            c3.metric("Final CPS", str(plan.get("final_cps", "—")))
            st.caption(
                f"Final: {plan.get('final_escalation')} / {plan.get('final_governance_status')}"
            )


def _render_stepwise(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🪜 Stepwise Recovery Path")
    df = _read_csv_safe(results_dir / "concentration_exit_steps.csv")
    if df.empty:
        st.info("No step data.", icon="ℹ️")
        return
    plan = st.selectbox(
        "Plan",
        sorted(df["plan_name"].unique()) if "plan_name" in df.columns else [],
        key="cesp_plan",
    )
    sub = df[df["plan_name"] == plan] if plan else df
    st.dataframe(sub, use_container_width=True, hide_index=True)


def _render_cps_path(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Projected CPS Path")
    rec = (
        (report.get("recommended_plan") or {}).get("plan_detail")
        or report.get("balanced_plan")
        or {}
    )
    steps = rec.get("steps") or []
    if not steps:
        st.info("No CPS path data.", icon="ℹ️")
        return
    path_df = pd.DataFrame(
        [
            {
                "Step": s.get("step_number"),
                "CPS": s.get("projected_cps"),
                "QQQ %": s.get("qqq_weight_pct"),
            }
            for s in steps
        ]
    )
    st.line_chart(path_df.set_index("Step")[["CPS"]])
    st.dataframe(path_df, use_container_width=True, hide_index=True)


def _render_governance(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚖️ Governance Impact")
    gi = report.get("governance_impact") or {}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Current**")
        st.json(gi.get("current") or {})
    with c2:
        st.markdown("**Post-Recommended Plan**")
        st.json(gi.get("post_recommended") or {})
    st.metric("CPS Improvement", str(gi.get("cps_improvement", "—")))


def _render_risk(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Risk Assessment")
    tl = report.get("timeline") or {}
    for name in ("conservative", "balanced", "aggressive"):
        t = tl.get(name) or {}
        st.markdown(
            f"- **{name.title()}:** YELLOW step {t.get('steps_to_yellow')} "
            f"(~{t.get('estimated_days_to_yellow')} days)"
        )
    if tl.get("green_requires_bito_promotion"):
        st.warning("GREEN requires BITO OPEN_NEW promotion before extension step.", icon="⚠️")


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Recommendation")
    ex = report.get("executive_report") or {}
    st.markdown(f"**[{ex.get('report_type', 'CONCENTRATION_EXIT_STRATEGY_REPORT')}]**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended Strategy", str(ex.get("recommended_strategy", "—")))
    c2.metric("Steps to YELLOW", str(ex.get("steps_required_for_yellow", "—")))
    c3.metric("Timeline (days)", str(ex.get("expected_recovery_timeline_days", "—")))
    st.markdown(f"**Safest path:** {ex.get('safest_path')}")
    st.markdown(f"**Fastest path:** {ex.get('fastest_path')}")
    st.markdown(f"**Expected governance:** {ex.get('expected_governance_outcome')}")
    st.markdown(f"**GREEN possible when:** {ex.get('green_possible_when')}")


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification")
    cert = report.get("certification") or {}
    st.metric("Status", str(cert.get("status", "—")))
    st.success(str(cert.get("safety") or ""), icon="🔒")


def render_concentration_exit_strategy_planner(results_dir: Path) -> None:
    st.markdown("### 📉 Concentration Exit Strategy Planner")
    st.caption(
        "Governance-safe QQQ concentration exit planning — analysis only. "
        "Refresh with `python -m services.concentration_exit_strategy_planner`."
    )

    report = _read_json_safe(results_dir / "concentration_exit_strategy.json")
    if not report:
        st.info(
            "No exit strategy report yet. Run "
            "`python -m services.concentration_exit_strategy_planner`.",
            icon="ℹ️",
        )
        return

    _render_concentration(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "📋 Exit Plans",
            "🪜 Stepwise Recovery Path",
            "📈 Projected CPS Path",
            "⚖️ Governance Impact",
            "🛡️ Risk Assessment",
            "📋 Executive Recommendation",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="cesp_section",
    )
    if section == "📋 Exit Plans":
        _render_exit_plans(report)
    elif section == "🪜 Stepwise Recovery Path":
        _render_stepwise(results_dir, report)
    elif section == "📈 Projected CPS Path":
        _render_cps_path(report)
    elif section == "⚖️ Governance Impact":
        _render_governance(report)
    elif section == "🛡️ Risk Assessment":
        _render_risk(report)
    elif section == "📋 Executive Recommendation":
        _render_executive(report)
    else:
        _render_certification(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
