"""
TRITON Phase 171 — ⚖️ Exit Timing Governance (read-only dashboard).

Displays the governance review of the Phase 169/170 exit timing proposals:
governance decisions, capital preservation review, risk review, operator
control rollout design, readiness assessment, and the approval matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _money, _num, _pct, _plotly_bar


def _read_json_safe(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _table(records: Any) -> pd.DataFrame:
    try:
        return pd.DataFrame(records or [])
    except Exception:
        return pd.DataFrame()


def _plotly_grouped_bar(df: pd.DataFrame, x: str, y: str, color: str, title: str) -> bool:
    if df is None or df.empty or any(c not in df.columns for c in (x, y, color)):
        return False
    try:
        import plotly.express as px

        fig = px.bar(df, x=x, y=y, color=color, barmode="group", title=title)
        st.plotly_chart(fig, use_container_width=True)
        return True
    except Exception:
        return False


_DECISION_BADGE = {
    "APPROVE_FOR_PAPER_TRIAL": "🟢 APPROVE_FOR_PAPER_TRIAL",
    "APPROVED_FOR_PAPER_TRIAL": "🟢 APPROVED_FOR_PAPER_TRIAL",
    "REQUIRES_MORE_EVIDENCE": "🟡 REQUIRES_MORE_EVIDENCE",
    "NEEDS_MORE_EVIDENCE": "🟡 NEEDS_MORE_EVIDENCE",
    "REJECT": "🔴 REJECT",
    "REJECTED": "🔴 REJECTED",
}
_RISK_BADGE = {
    "LOW_RISK": "🟢 LOW_RISK",
    "MEDIUM_RISK": "🟡 MEDIUM_RISK",
    "HIGH_RISK": "🔴 HIGH_RISK",
    "NONE": "🟢 NONE",
}
_STATUS_BADGE = {
    "CERTIFIED": "🟢",
    "PARTIALLY_CERTIFIED": "🟡",
    "NOT_CERTIFIED": "🔴",
    "READY": "🟢",
    "PARTIALLY_READY": "🟡",
    "NOT_READY": "🔴",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Summary")
    ex = report.get("executive_report") or {}
    ready = report.get("implementation_readiness") or {}
    c1, c2, c3, c4 = st.columns(4)
    status = str(ex.get("approval_status", "—"))
    badge = {"APPROVED_WITH_CONDITIONS": "🟢", "DEFERRED": "🟡", "REJECTED": "🔴"}.get(status, "⚪")
    c1.metric("Approval Status", f"{badge} {status}")
    c2.metric(
        "Readiness",
        f"{_STATUS_BADGE.get(str(ready.get('status')), '⚪')} {ready.get('implementation_readiness_score', '—')}/100",
    )
    c3.metric("Rollout Mode (First)", str(ex.get("recommended_rollout_mode", "—")))
    c4.metric("Approved Proposals", str(len(ex.get("approved_proposals") or [])))
    st.markdown(f"- **Top Recommendation:** {ex.get('top_recommendation') or '—'}")
    st.markdown(f"- **Top Risk:** {ex.get('top_risk') or '—'}")
    st.markdown(
        f"- **Top Capital Preservation Benefit:** {ex.get('top_capital_preservation_benefit') or '—'}"
    )
    evidence = ex.get("required_additional_evidence") or []
    if evidence:
        st.markdown("**Required additional evidence:**")
        for e in evidence:
            st.markdown(f"- {e}")
    verdict = str(ex.get("final_governance_verdict") or "—")
    if status == "APPROVED_WITH_CONDITIONS":
        st.success(f"Final Governance Verdict: {verdict}", icon="⚖️")
    elif status == "DEFERRED":
        st.warning(f"Final Governance Verdict: {verdict}", icon="⚖️")
    else:
        st.error(f"Final Governance Verdict: {verdict}", icon="⚖️")
    focus = report.get("phase_172_focus")
    if focus:
        st.info(f"Phase 172 focus: {focus}", icon="🧭")


def _render_decisions(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚖️ Governance Decisions")
    df = _table(report.get("governance_decisions"))
    if df.empty:
        st.info("No governance decisions available.", icon="ℹ️")
        return
    view = df.copy()
    view["decision"] = view["decision"].map(lambda d: _DECISION_BADGE.get(str(d), str(d)))
    if "risk_class" in view.columns:
        view["risk_class"] = view["risk_class"].map(lambda r: _RISK_BADGE.get(str(r), str(r)))
    st.dataframe(view, use_container_width=True, hide_index=True)
    melted = df.melt(
        id_vars=["proposal"],
        value_vars=[
            c
            for c in (
                "governance_score",
                "capital_preservation_score",
                "operator_control_score",
                "risk_score",
            )
            if c in df.columns
        ],
        var_name="score",
        value_name="value",
    )
    if not _plotly_grouped_bar(
        melted, "proposal", "value", "score", "Governance Scores by Proposal"
    ):
        st.bar_chart(df.set_index("proposal")[["governance_score", "risk_score"]])
    st.caption("Risk score: higher = more risk. All other scores: higher = better.")


def _render_capital(report: Dict[str, Any]) -> None:
    st.markdown("#### 🏦 Capital Preservation Review")
    cp = report.get("capital_preservation_assessment") or {}
    doctrine = cp.get("doctrine_source") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("CP Certification", str(doctrine.get("capital_preservation_certification", "—")))
    c2.metric(
        "Late-Exit Giveback (Historical)", _money(cp.get("historical_giveback_from_late_exits"))
    )
    c3.metric("Capital Preserved by Exits", _money(cp.get("historical_capital_preserved_by_exits")))
    df = _table(cp.get("per_proposal"))
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    verdict = str(cp.get("verdict") or "—")
    if cp.get("would_strengthen_preservation"):
        st.success(f"Would implementation strengthen preservation? {verdict}", icon="🛡️")
    else:
        st.error(f"Would implementation strengthen preservation? {verdict}", icon="🛡️")


def _render_risk(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚠️ Risk Review")
    df = _table(report.get("risk_assessment"))
    if df.empty:
        st.info("No risk assessment available.", icon="ℹ️")
        return
    proposals = sorted(df["proposal"].unique().tolist())
    pick = st.selectbox("Proposal", proposals, key="etg_risk_pick")
    view = df[df["proposal"] == pick].copy()
    overall = view[view["risk_factor"] == "OVERALL"]
    if not overall.empty:
        sev = str(overall.iloc[0]["severity"])
        st.metric("Overall Risk", _RISK_BADGE.get(sev, sev))
    view["severity"] = view["severity"].map(lambda s: _RISK_BADGE.get(str(s), str(s)))
    st.dataframe(
        view[view["risk_factor"] != "OVERALL"][
            ["risk_factor", "measurement", "severity", "mitigation"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    counts = (
        df[df["risk_factor"] != "OVERALL"]
        .groupby(["proposal", "severity"])
        .size()
        .reset_index(name="factors")
    )
    if not _plotly_grouped_bar(
        counts, "proposal", "factors", "severity", "Risk Factor Severity by Proposal"
    ):
        st.bar_chart(counts.pivot(index="proposal", columns="severity", values="factors").fillna(0))


def _render_control(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎛️ Operator Control Review")
    ctrl = report.get("operator_control_review") or {}
    modes = _table(ctrl.get("modes"))
    if not modes.empty:
        st.dataframe(modes, use_container_width=True, hide_index=True)
    st.markdown(f"**Safest first mode:** `{ctrl.get('safest_first_mode', '—')}`")
    st.markdown("**Recommended rollout sequence:**")
    for step in ctrl.get("recommended_rollout_sequence") or []:
        st.markdown(f"- {step}")
    st.warning(str(ctrl.get("note") or "Governance recommendation only."), icon="🔒")


def _render_readiness(report: Dict[str, Any]) -> None:
    st.markdown("#### 🚦 Readiness Assessment")
    ready = report.get("implementation_readiness") or {}
    score = ready.get("implementation_readiness_score")
    status = str(ready.get("status", "—"))
    c1, c2 = st.columns(2)
    c1.metric("Implementation Readiness", f"{score}/100")
    c2.metric("Status", f"{_STATUS_BADGE.get(status, '⚪')} {status}")
    factors = _table(ready.get("factors"))
    if not factors.empty:
        if not _plotly_bar(factors, "factor", "score", "Readiness Factors"):
            st.bar_chart(factors.set_index("factor")["score"])
        st.dataframe(factors, use_container_width=True, hide_index=True)
    st.caption(str(ready.get("scope") or ""))
    st.caption("Bands: 75+ READY · 50+ PARTIALLY_READY · <50 NOT_READY")


def _render_matrix(report: Dict[str, Any]) -> None:
    st.markdown("#### ✅ Approval Matrix")
    df = _table(report.get("approval_matrix"))
    if df.empty:
        st.info("No approval matrix available.", icon="ℹ️")
        return
    ready = report.get("implementation_readiness") or {}
    view = df.copy()
    view["readiness_score"] = ready.get("implementation_readiness_score")
    view["decision"] = view["decision"].map(
        lambda d: _DECISION_BADGE.get(str(d), str(d)).replace("APPROVE_FOR", "APPROVED_FOR")
    )
    view = view.rename(
        columns={
            "proposal": "Proposal",
            "governance_score": "Governance Score",
            "risk_score": "Risk Score",
            "capital_preservation_score": "Capital Preservation Score",
            "operator_control_score": "Operator Control Score",
            "readiness_score": "Readiness Score",
            "decision": "Decision",
        }
    )
    st.dataframe(view, use_container_width=True, hide_index=True)


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification Status")
    cert = report.get("certification") or {}
    if not cert:
        st.info("No certification data.", icon="ℹ️")
        return
    status = str(cert.get("status", "—"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", f"{_STATUS_BADGE.get(status, '⚪')} {status}")
    c2.metric("Proposals Reviewed", str(cert.get("proposals_reviewed", "—")))
    c3.metric("Framework Certification", str(cert.get("framework_certification", "—")))
    for r in cert.get("reasons") or []:
        st.warning(r, icon="⚠️")
    st.success(str(cert.get("safety") or "Governance review only."), icon="🔒")


def render_exit_timing_governance(results_dir: Path) -> None:
    st.markdown("### ⚖️ Exit Timing Governance")
    st.caption(
        "Phase 171 governance review of the certified exit timing enhancements — "
        "governance, risk, and capital-preservation review only; nothing operational "
        "changes. Refresh with `python -m services.exit_timing_governance`."
    )

    report = _read_json_safe(results_dir / "exit_timing_governance.json")
    if not report:
        st.info(
            "No governance report yet. Run "
            "`python -m services.exit_timing_governance` to generate it.",
            icon="ℹ️",
        )
        return

    _render_executive(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "⚖️ Governance Decisions",
            "🏦 Capital Preservation",
            "⚠️ Risk Review",
            "🎛️ Operator Control",
            "🚦 Readiness",
            "✅ Approval Matrix",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="etg_section",
    )
    if section == "⚖️ Governance Decisions":
        _render_decisions(report)
    elif section == "🏦 Capital Preservation":
        _render_capital(report)
    elif section == "⚠️ Risk Review":
        _render_risk(report)
    elif section == "🎛️ Operator Control":
        _render_control(report)
    elif section == "🚦 Readiness":
        _render_readiness(report)
    elif section == "✅ Approval Matrix":
        _render_matrix(report)
    else:
        _render_certification(report)
