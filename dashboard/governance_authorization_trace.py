"""
TRITON Phase 173 — 🔍 Governance Authorization Trace (read-only dashboard).

Explains why governance_authorized may differ from execution_authorized,
with dependency tree, block reasons, readiness trace, and remediation guidance.
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


def _bool_badge(v: Any) -> str:
    return "🟢 true" if v is True else "🔴 false" if v is False else f"⚪ {v}"


_SEV_BADGE = {
    "CRITICAL": "🔴 CRITICAL",
    "BLOCKING": "🟠 BLOCKING",
    "WARNING": "🟡 WARNING",
    "INFO": "🔵 INFO",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Summary")
    ex = report.get("executive_report") or {}
    state = ex.get("current_authorization_state") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Governance Authorized", _bool_badge(state.get("governance_authorized")))
    c2.metric("Execution Authorized", _bool_badge(state.get("execution_authorized")))
    c3.metric("Overall Authorization", _bool_badge(state.get("overall_authorization")))
    c4.metric("Readiness", str(state.get("readiness_status", "—")))
    st.markdown(f"- **Primary Cause:** {ex.get('primary_cause') or '—'}")
    st.markdown(f"- **Highest Severity Block:** {ex.get('highest_severity_block') or '—'}")
    st.markdown(f"- **Recommended Next Action:** {ex.get('recommended_next_action') or '—'}")
    st.markdown(f"- **Fastest Path to READY:** {ex.get('fastest_path_to_ready') or '—'}")
    mismatch = (report.get("authorization_trace") or {}).get("mismatch_explanation")
    if mismatch:
        st.info(mismatch, icon="🔍")


def _render_authorization_status(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚖️ Authorization Status")
    trace = report.get("authorization_trace") or {}
    rows = []
    for gate in (
        "execution_authorized",
        "operator_authorized",
        "policy_authorized",
        "governance_authorized",
        "overall_authorization",
    ):
        item = trace.get(gate) or {}
        rows.append(
            {
                "Gate": gate,
                "Value": _bool_badge(item.get("value")),
                "Why": item.get("why"),
                "Source Rule": item.get("source_rule"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    submit = trace.get("paper_final_submit_allowed") or {}
    st.metric("Paper Final Submit Allowed", _bool_badge(submit.get("value")))
    if submit.get("why"):
        st.caption(str(submit.get("why")))


def _render_tree(results_dir: Path) -> None:
    st.markdown("#### 🌳 Dependency Tree")
    tree = _read_json_safe(results_dir / "governance_dependency_tree.json")
    if not tree:
        st.info("No dependency tree available.", icon="ℹ️")
        return
    st.json(tree)


def _render_blocks(results_dir: Path) -> None:
    st.markdown("#### 🚫 Block Reasons")
    df = _read_csv_safe(results_dir / "governance_block_reasons.csv")
    if df.empty:
        st.info("No block reasons recorded.", icon="ℹ️")
        return
    view = df.copy()
    if "severity" in view.columns:
        view["severity"] = view["severity"].map(lambda s: _SEV_BADGE.get(str(s), str(s)))
    sev_filter = st.multiselect(
        "Severity filter",
        options=sorted(df["severity"].unique().tolist()),
        default=sorted(df["severity"].unique().tolist()),
        key="gat_sev",
    )
    if sev_filter:
        view = view[df["severity"].isin(sev_filter)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    counts = df["severity"].value_counts().reset_index()
    counts.columns = ["severity", "count"]
    if not _plotly_bar(counts, "severity", "count", "Block Severity Distribution"):
        st.bar_chart(counts.set_index("severity")["count"])


def _render_readiness(report: Dict[str, Any]) -> None:
    st.markdown("#### 🚦 Readiness Trace")
    rt = report.get("readiness_trace") or {}
    c1, c2 = st.columns(2)
    c1.metric("Status", str(rt.get("readiness_status", "—")))
    c2.metric("Checks", f"{rt.get('checks_passing_count', '—')}/{rt.get('checks_total', '—')}")
    rows = rt.get("trace_rows") or []
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.success("All readiness checks passing.", icon="✅")


def _render_risk(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚠️ Risk Trace")
    risk = report.get("risk_trace") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Governance Status", str(risk.get("governance_status", "—")))
    c2.metric("Escalation State", str(risk.get("escalation_state", "—")))
    c3.metric("Escalation Reasons", ", ".join(risk.get("escalation_reasons") or []) or "—")
    gov = report.get("governance_trace") or {}
    if gov.get("checks"):
        st.dataframe(pd.DataFrame(gov["checks"]), use_container_width=True, hide_index=True)


def _render_governance(report: Dict[str, Any]) -> None:
    st.markdown("#### 🏛 Governance Trace")
    gov = report.get("governance_trace") or {}
    st.markdown(f"**Reason:** {gov.get('reason') or '—'}")
    st.markdown(
        f"Escalation: **{gov.get('escalation_state')}** · "
        f"Awareness: **{gov.get('governance_awareness_level') or '—'}**"
    )
    chain = (
        (report.get("authorization_trace") or {})
        .get("governance_authorized", {})
        .get("dependency_chain")
    )
    if chain:
        st.markdown("**Dependency chain:**")
        for step in chain:
            st.markdown(f"- {step}")


def _render_remediation(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛠 Remediation Actions")
    st.caption("Recommendations only — the trace engine does not perform remediation.")
    actions = report.get("remediation_actions") or []
    if not actions:
        st.info("No remediation actions suggested.", icon="ℹ️")
        return
    df = pd.DataFrame(actions)
    if "severity" in df.columns:
        df["severity"] = df["severity"].map(lambda s: _SEV_BADGE.get(str(s), str(s)))
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_governance_authorization_trace(results_dir: Path) -> None:
    st.markdown("### 🔍 Governance Authorization Trace")
    st.caption(
        "Diagnostic trace — explains authorization gates without modifying any "
        "decisions. Refresh with `python -m services.governance_authorization_trace`."
    )

    report = _read_json_safe(results_dir / "governance_authorization_trace.json")
    if not report:
        st.info(
            "No trace report yet. Run "
            "`python -m services.governance_authorization_trace` to generate it.",
            icon="ℹ️",
        )
        return

    _render_executive(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "⚖️ Authorization Status",
            "🌳 Dependency Tree",
            "🚫 Block Reasons",
            "🚦 Readiness Trace",
            "⚠️ Risk Trace",
            "🏛 Governance Trace",
            "🛠 Remediation Actions",
        ],
        horizontal=True,
        key="gat_section",
    )
    if section == "⚖️ Authorization Status":
        _render_authorization_status(report)
    elif section == "🌳 Dependency Tree":
        _render_tree(results_dir)
    elif section == "🚫 Block Reasons":
        _render_blocks(results_dir)
    elif section == "🚦 Readiness Trace":
        _render_readiness(report)
    elif section == "⚠️ Risk Trace":
        _render_risk(report)
    elif section == "🏛 Governance Trace":
        _render_governance(report)
    else:
        _render_remediation(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
