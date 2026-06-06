"""
TRITON Institutional Protection dashboards — Phases 25–27.
Read-only / simulation only. No live execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _committee_badge(status: str) -> str:
    return {
        "CLEAR": "🟢 CLEAR",
        "MONITOR": "🟡 MONITOR",
        "REVIEW_REQUIRED": "🟠 REVIEW REQUIRED",
        "ESCALATE": "🔴 ESCALATE",
    }.get(str(status).upper(), status)


def _domain_badge(status: str) -> str:
    return {
        "CLEAR": "🟢",
        "MONITOR": "🟡",
        "CONCERN": "🟠",
        "CRITICAL": "🔴",
    }.get(str(status).upper(), status)


def _board_badge(status: str) -> str:
    return {
        "ACTIVE": "🟢 ACTIVE",
        "REVIEW": "🟠 REVIEW",
        "SUSPENDED": "🔴 SUSPENDED",
        "STANDBY": "🟡 STANDBY",
    }.get(str(status).upper(), status)


def render_risk_committee_oversight(results_dir: Path) -> None:
    st.markdown("### 🏛 Risk Committee Oversight")
    st.caption(
        "Committee assessments across portfolio, governance, preservation, readiness, "
        "and certification. **Advisory only — no execution.**"
    )

    doc = _load_json(results_dir / "risk_committee_oversight.json")
    if not doc:
        st.warning("Risk committee data not found. Run `python -m services.risk_watchdog`.")
        return

    status = str(doc.get("committee_status") or "MONITOR")
    st.metric("Committee Status", _committee_badge(status))
    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Assessment", doc.get("overall_assessment", "—"))
    c2.metric("Avg Domain Score", doc.get("average_domain_score", 0))
    c3.metric("CPS", doc.get("capital_preservation_score", "—"))
    c4.metric("Escalation", doc.get("escalation_state", "—"))

    concerns = doc.get("top_concerns") or []
    if concerns:
        st.markdown("#### Top Concerns")
        for c in concerns:
            st.markdown(f"- **{c}**")

    domains: List[Dict[str, Any]] = doc.get("domains") or []
    if domains:
        st.markdown("#### Domain Assessments")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Domain": d.get("domain"),
                        "Score": d.get("score"),
                        "Status": d.get("status"),
                        "Concerns": ", ".join(d.get("concerns") or []) or "—",
                    }
                    for d in domains
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        for d in domains:
            with st.expander(
                f"{_domain_badge(d.get('status', ''))} {d.get('domain')} — {d.get('score')}/100",
                expanded=False,
            ):
                for c in d.get("concerns") or []:
                    st.markdown(f"- {c}")
                if not d.get("concerns"):
                    st.caption("No active concerns in this domain.")

    history_path = results_dir / "risk_committee_oversight_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "average_domain_score" in hist.columns:
                st.markdown("#### Domain Score Trend")
                trend = hist.tail(30).copy()
                if "timestamp" in trend.columns:
                    trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
                st.line_chart(trend.set_index("timestamp")["average_domain_score"])
        except Exception:
            pass

    st.info("Committee oversight informs human review only. No broker actions are taken.")


def render_accountability_registry(results_dir: Path) -> None:
    st.markdown("### 📑 Accountability Registry")
    st.caption(
        "Traceability for protective decision paths across governor, approvals, "
        "candidates, evaluations, and audit. **Read-only.**"
    )

    doc = _load_json(results_dir / "accountability_registry.json")
    if not doc:
        st.warning("Accountability registry not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Registry Entries", doc.get("entry_count", 0))
    c2.metric("Not Certified / Pending", doc.get("not_certified_count", 0))
    c3.metric("Overall Authorization", "Yes" if doc.get("overall_authorization") else "No")

    summary = doc.get("summary_by_origin") or {}
    if summary:
        st.markdown("#### Entries by Origin")
        st.dataframe(
            pd.DataFrame([{"Origin": k, "Count": v} for k, v in sorted(summary.items())]),
            use_container_width=True,
            hide_index=True,
        )

    entries: List[Dict[str, Any]] = doc.get("entries") or []
    if entries:
        st.markdown("#### Decision Traceability")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Decision ID": (e.get("decision_id") or "")[:36],
                        "Origin": e.get("origin"),
                        "Type": e.get("decision_type"),
                        "Result": e.get("decision_result"),
                        "Certification": e.get("certification_status"),
                        "Governance Source": e.get("governance_source"),
                        "Approval Source": e.get("approval_source"),
                        "Policy Source": e.get("policy_source"),
                        "Timestamp": e.get("timestamp"),
                    }
                    for e in entries[:100]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        for e in entries[:15]:
            with st.expander(
                f"{e.get('origin')} — {e.get('decision_type')} ({e.get('decision_result')})",
                expanded=False,
            ):
                st.markdown(f"**Decision ID:** `{e.get('decision_id')}`")
                st.markdown(f"**Certification:** {e.get('certification_status')}")
                st.markdown(f"**Governance Source:** {e.get('governance_source')}")
                st.markdown(f"**Approval Source:** {e.get('approval_source')}")
                st.markdown(f"**Policy Source:** {e.get('policy_source')}")
                if e.get("details"):
                    st.markdown(f"**Details:** {e.get('details')}")
                st.caption("Execution permitted: No")

    st.warning("Registry tracks accountability only. No trades or orders are executed.")


def render_preservation_governance_board(results_dir: Path) -> None:
    st.markdown("### 👑 Preservation Governance Board")
    st.caption(
        "Unified authority layer aggregating committee, accountability, certification, "
        "and governor. **Advisory only — human-controlled.**"
    )

    doc = _load_json(results_dir / "preservation_governance_board.json")
    if not doc:
        st.warning("Governance board data not found. Run the watchdog first.")
        return

    board_status = str(doc.get("board_status") or "ACTIVE")
    st.metric("Board Status", _board_badge(board_status))
    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Governance Confidence", f"{doc.get('governance_confidence', 0)}%")
    c2.metric("Preservation Authority", doc.get("preservation_authority", "HUMAN_CONTROLLED"))
    c3.metric(
        "Automation Authorized",
        "Yes" if doc.get("automation_authorized") else "No",
    )
    c4.metric("Committee Status", doc.get("committee_status", "—"))

    c_a, c_b, c_c = st.columns(3)
    c_a.metric("Overall Assessment", doc.get("overall_assessment", "—"))
    c_b.metric("Preservation Posture", doc.get("preservation_posture", "—"))
    c_c.metric("Certification", doc.get("certification_status", "—"))

    recommendations = doc.get("board_recommendations") or []
    if recommendations:
        st.markdown("#### Board Recommendations (Advisory Only)")
        for rec in recommendations:
            st.markdown(f"- {rec}")

    c_x, c_y = st.columns(2)
    c_x.metric("Accountability Entries", doc.get("accountability_entries", 0))
    c_y.metric(
        "Live Execution Permitted",
        "No" if not doc.get("live_execution_permitted") else "Yes",
    )

    history_path = results_dir / "preservation_governance_board_history.csv"
    if history_path.is_file():
        try:
            hist = pd.read_csv(history_path)
            if not hist.empty and "governance_confidence" in hist.columns:
                st.markdown("#### Governance Confidence Trend")
                trend = hist.tail(30).copy()
                if "timestamp" in trend.columns:
                    trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
                st.line_chart(trend.set_index("timestamp")["governance_confidence"])
        except Exception:
            pass

    st.info(
        "Preservation authority defaults to HUMAN_CONTROLLED. "
        "automation_authorized remains false. No portfolio modifications."
    )
