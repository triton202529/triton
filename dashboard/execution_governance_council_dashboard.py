"""
TRITON THP-008 — Execution Governance Council Dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from services.execution_approval_workflow import (
    APPROVAL_REQUESTS_JSON,
    approve_request,
    evaluate_governed_execution,
    expire_stale_requests,
    load_approval_requests,
    reject_request,
)
from services.execution_escalation_engine import (
    ESCALATIONS_JSON,
    load_escalations,
    run_escalation_scan,
)
from services.execution_governance_council import (
    COUNCIL_STATUS_JSON,
    load_council_status,
    run_governance_council,
)
from services.execution_gate_degradation_manager import load_operational_status
from services.triton_execution_gate import EXECUTION_PATHS

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"
APPROVAL_HISTORY_CSV = GOVERNANCE_DIR / "execution_approval_history.csv"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _mode_badge(mode: str) -> str:
    return {
        "NORMAL_AUTOMATION": "🟢 Normal Automation",
        "ELEVATED_OVERSIGHT": "🟡 Elevated Oversight",
        "EMERGENCY_ONLY": "🔴 Emergency Only",
    }.get(mode, mode)


def page_execution_governance_council() -> None:
    st.title("🏛 Execution Governance Council")
    st.info(
        "**Investment committee-grade execution governance.** Human approval workflows, "
        "escalation management, and authority level enforcement."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Refresh Council Status", key="egc_refresh"):
            run_governance_council()
            st.success("Council status updated.")
    with col2:
        if st.button("Scan Escalations", key="egc_escalate"):
            run_escalation_scan()
            st.success("Escalation scan complete.")
    with col3:
        if st.button("Expire Stale Requests", key="egc_expire"):
            n = expire_stale_requests()
            st.success(f"Expired {n} stale request(s).")

    council = load_council_status() or _read_json(COUNCIL_STATUS_JSON)
    operational = load_operational_status()
    approvals = load_approval_requests() or _read_json(APPROVAL_REQUESTS_JSON)
    escalations = load_escalations() or _read_json(ESCALATIONS_JSON)

    st.markdown("### Authority Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_mode_badge(council.get("current_authority_mode", "UNKNOWN")))
    c2.metric("Health Score", council.get("health_score", "—"))
    c3.metric("Operational State", operational.get("operational_state", "—"))
    c4.metric("Certification", council.get("certification_level", "—"))

    restrictions = council.get("active_restrictions") or {}
    if restrictions.get("capital_deployment_blocked"):
        st.error("🛑 LOCKDOWN: Capital deployment blocked except emergency authority.")
    elif restrictions.get("new_capital_deployment_restricted"):
        st.warning("🟠 RESTRICT: New capital deployment requires governance review.")

    st.markdown("### Approval Queue")
    approval_records = approvals.get("records") or []
    summary = approvals.get("summary") or {}
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Pending", summary.get("pending", 0))
    a2.metric("Approved", summary.get("approved", 0))
    a3.metric("Rejected", summary.get("rejected", 0))
    a4.metric("Expired", summary.get("expired", 0))

    pending = [r for r in approval_records if r.get("approval_status") == "PENDING"]
    if pending:
        st.dataframe(pd.DataFrame(pending), use_container_width=True, hide_index=True)
        st.markdown("#### Review Pending Request")
        selected = st.selectbox(
            "Select approval_id",
            [r.get("approval_id") for r in pending],
            key="egc_pending_select",
        )
        approver = st.text_input("Approving authority", key="egc_approver")
        rationale = st.text_area("Rationale", key="egc_rationale")
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("Approve", key="egc_approve"):
                if approver and rationale:
                    approve_request(
                        selected, approving_authority=approver, approval_rationale=rationale
                    )
                    st.success("Request approved.")
                    st.rerun()
                else:
                    st.warning("Approver and rationale required.")
        with bc2:
            if st.button("Reject", key="egc_reject"):
                if approver and rationale:
                    reject_request(
                        selected, rejecting_authority=approver, rejection_rationale=rationale
                    )
                    st.success("Request rejected.")
                    st.rerun()
                else:
                    st.warning("Authority and rationale required.")
    else:
        st.caption("No pending approval requests.")

    st.markdown("### Escalation Center")
    esc_summary = escalations.get("summary") or {}
    e1, e2, e3 = st.columns(3)
    e1.metric("Total Escalations", esc_summary.get("total", 0))
    e2.metric("Open", esc_summary.get("open", 0))
    e3.metric("Resolved", esc_summary.get("resolved", 0))
    if escalations.get("records"):
        open_esc = [r for r in escalations["records"] if r.get("resolution_status") == "OPEN"]
        if open_esc:
            st.dataframe(pd.DataFrame(open_esc[-20:]), use_container_width=True, hide_index=True)

    st.markdown("### Decision Analytics")
    analytics = council.get("analytics") or {}
    d1, d2, d3 = st.columns(3)
    d1.metric("Automation Rate", f"{analytics.get('automation_rate_pct', 0)}%")
    d2.metric("Human Intervention Rate", f"{analytics.get('human_intervention_rate_pct', 0)}%")
    d3.metric("Review Required", analytics.get("review_required_count", 0))

    st.markdown("### Governed Execution Test")
    st.caption("Simulates governed evaluation (record_evidence=False).")
    test_path = st.selectbox(
        "Execution path",
        list(EXECUTION_PATHS.values()),
        index=list(EXECUTION_PATHS.values()).index(EXECUTION_PATHS["PAPER_EXECUTION"]),
        key="egc_test_path",
    )
    if st.button("Evaluate with Governance", key="egc_test_eval"):
        ctx = {"paperMode": True, "lifecycleApproved": True}
        if test_path == EXECUTION_PATHS["LIVE_EXECUTION"]:
            ctx = {"liveMode": True, "liveModeAuthorized": False, "governanceApproved": False}
        result = evaluate_governed_execution(
            test_path,
            ctx,
            record_evidence=False,
            source="council_dashboard_test",
            execution_identifier=f"dashboard-{test_path}",
        )
        st.json(result)

    st.markdown("### Approval History")
    if APPROVAL_HISTORY_CSV.is_file():
        try:
            st.dataframe(pd.read_csv(APPROVAL_HISTORY_CSV).tail(20), use_container_width=True)
        except Exception:
            st.caption("History unavailable.")
    else:
        st.caption("No approval history yet.")

    st.markdown("### Council Status Artifact")
    if council:
        st.json(council)


def render() -> None:
    page_execution_governance_council()
