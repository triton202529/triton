"""
TRITON Phase 180 — 🚦 Diversification Execution Readiness dashboard.
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


_READY = {
    "READY": "🟢 READY",
    "READY_WITH_CONDITIONS": "🟡 READY_WITH_CONDITIONS",
    "BLOCKED": "🔴 BLOCKED",
    "NOT_ELIGIBLE": "⚪ NOT_ELIGIBLE",
}
_OVERALL = {
    "FULLY_READY": "🟢 FULLY_READY",
    "READY": "🟢 READY",
    "PARTIALLY_READY": "🟡 PARTIALLY_READY",
    "NOT_READY": "🔴 NOT_READY",
}


def _render_recovery_plan(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Recovery Plan")
    rp = report.get("recovery_plan") or {}
    cs = report.get("current_state") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current CPS", str(cs.get("current_cps", "—")))
    c2.metric("Governance", str(cs.get("current_governance_status", "—")))
    c3.metric("Escalation", str(cs.get("current_escalation", "—")))
    c4.metric("Target CPS", str(rp.get("projected_cps", "—")))
    st.markdown(f"**Plan weights:** `{rp.get('weights') or {}}`")
    st.markdown(f"**Post-execution governance:** {rp.get('projected_governance_status') or '—'}")


def _render_candidates(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Candidate Readiness")
    df = _read_csv_safe(results_dir / "diversification_execution_readiness.csv")
    if df.empty:
        df = pd.DataFrame(report.get("candidate_readiness") or [])
    if df.empty:
        st.info("No candidate data.", icon="ℹ️")
        return
    focus = df[df["ticker"].isin(["HD", "XLK", "BITO", "QQQ"])].copy()
    if "readiness_status" in focus.columns:
        focus["readiness_status"] = focus["readiness_status"].map(
            lambda x: _READY.get(str(x), str(x))
        )
    st.markdown("**Recovery basket**")
    st.dataframe(focus, use_container_width=True, hide_index=True)
    open_new = (
        df[df["allocation_status"] == "OPEN_NEW"]
        if "allocation_status" in df.columns
        else pd.DataFrame()
    )
    if not open_new.empty:
        st.markdown("**OPEN_NEW candidates**")
        st.dataframe(open_new, use_container_width=True, hide_index=True)


def _render_validation(report: Dict[str, Any]) -> None:
    st.markdown("#### ✅ Execution Validation")
    ev = report.get("execution_validation") or {}
    prob = report.get("execution_probability") or {}
    c1, c2 = st.columns(2)
    c1.metric(
        "YELLOW Executable?",
        "✅ Yes" if ev.get("can_yellow_recovery_plan_execute_now") else "❌ No",
    )
    c2.metric(
        "GREEN Executable?",
        "✅ Yes" if ev.get("can_green_recovery_plan_execute_now") else "❌ No",
    )
    c3, c4 = st.columns(2)
    c3.metric("Execution Probability", str(prob.get("execution_probability", "—")))
    c4.metric("Recovery Probability", str(prob.get("combined_recovery_probability", "—")))
    st.json(ev)


def _render_bottlenecks(report: Dict[str, Any]) -> None:
    st.markdown("#### 🚧 Bottleneck Analysis")
    bn = report.get("bottleneck_analysis") or {}
    prom = report.get("promotion_dependencies") or {}
    c1, c2 = st.columns(2)
    c1.metric("Primary Bottleneck", str(bn.get("primary_bottleneck", "—")))
    c2.metric("Secondary Bottleneck", str(bn.get("secondary_bottleneck", "—")))
    if bn.get("details"):
        st.warning(str(bn["details"]), icon="⚠️")
    st.markdown("**Required actions**")
    for a in (report.get("executive_report") or {}).get("required_actions") or []:
        st.markdown(f"- {a}")
    st.markdown("**Promotion dependencies**")
    st.json(prom)


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Recommendation")
    ex = report.get("executive_report") or {}
    st.markdown(f"**[{ex.get('report_type', 'DIVERSIFICATION_EXECUTION_READINESS_REPORT')}]**")
    rs = report.get("readiness_scores") or {}
    overall = str(rs.get("overall_readiness_classification", "—"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Readiness", _OVERALL.get(overall, overall))
    c2.metric("Execution Score", f"{_num(rs.get('execution_score'))}")
    c3.metric("Recovery Score", str(rs.get("recovery_score", "—")))
    st.markdown(f"**Recommended action:** {ex.get('recommended_action') or '—'}")
    st.markdown(f"**Expected outcome:** {ex.get('expected_governance_outcome') or '—'}")
    st.markdown(f"**Governance impact:** {ex.get('governance_impact') or '—'}")


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification")
    cert = report.get("certification") or {}
    st.metric("Status", str(cert.get("status", "—")))
    st.success(str(cert.get("safety") or ""), icon="🔒")


def render_diversification_execution_readiness_gate(results_dir: Path) -> None:
    st.markdown("### 🚦 Diversification Execution Readiness")
    st.caption(
        "Execution readiness gate for diversification recovery — analysis only. "
        "Refresh with `python -m services.diversification_execution_readiness_gate`."
    )

    report = _read_json_safe(results_dir / "diversification_execution_readiness.json")
    if not report:
        st.info(
            "No readiness report yet. Run "
            "`python -m services.diversification_execution_readiness_gate`.",
            icon="ℹ️",
        )
        return

    _render_recovery_plan(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "🎯 Candidate Readiness",
            "✅ Execution Validation",
            "🚧 Bottleneck Analysis",
            "📋 Executive Recommendation",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="derg_section",
    )
    if section == "🎯 Candidate Readiness":
        _render_candidates(results_dir, report)
    elif section == "✅ Execution Validation":
        _render_validation(report)
    elif section == "🚧 Bottleneck Analysis":
        _render_bottlenecks(report)
    elif section == "📋 Executive Recommendation":
        _render_executive(report)
    else:
        _render_certification(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
