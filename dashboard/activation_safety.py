"""
TRITON Activation Safety dashboards — Phases 13–15 (read-only + queue status updates).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from services.activation_safety import set_approval_request_status


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def render_defensive_automation_sandbox(results_dir: Path) -> None:
    st.markdown("### 🧱 Defensive Automation Sandbox")
    st.caption(
        "Hypothetical defensive actions if automation were enabled. "
        "**Status: SIMULATION_ONLY — no execution.**"
    )

    doc = _load_json(results_dir / "defensive_action_candidates.json")
    if not doc:
        st.warning("No sandbox candidates yet. Run `python -m services.risk_watchdog`.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}**")
    st.metric("Candidate Actions", doc.get("candidate_count", 0))

    candidates: List[Dict[str, Any]] = doc.get("candidates") or []
    if not candidates:
        st.success("No hypothetical defensive actions required at this time.")
        return

    rows = []
    for c in candidates:
        rows.append(
            {
                "Action": c.get("candidate_action"),
                "Policy Type": c.get("policy_type"),
                "Reason": c.get("reason"),
                "Est. Risk Reduction": c.get("estimated_risk_reduction"),
                "Status": c.get("status"),
                "Execution": "BLOCKED" if not c.get("execution_permitted") else "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Candidate details", expanded=False):
        for c in candidates:
            st.json(c)


def render_human_approval_center(results_dir: Path) -> None:
    st.markdown("### 👤 Human Approval Center")
    st.caption(
        "Institutional approval gates. Approving or rejecting updates queue status only — "
        "**no trades, orders, or portfolio changes are executed.**"
    )

    queue_path = results_dir / "human_approval_queue.json"
    doc = _load_json(queue_path)
    if not doc:
        st.warning("Approval queue not initialized. Run the watchdog first.")
        return

    counts = doc.get("counts") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pending Review", counts.get("PENDING_REVIEW", 0))
    c2.metric("Approved", counts.get("APPROVED", 0))
    c3.metric("Rejected", counts.get("REJECTED", 0))
    c4.metric("Expired", counts.get("EXPIRED", 0))

    requests: List[Dict[str, Any]] = doc.get("requests") or []

    def _show_group(title: str, status: str) -> None:
        items = [r for r in requests if str(r.get("status")) == status]
        st.markdown(f"#### {title} ({len(items)})")
        if not items:
            st.caption(f"No {title.lower()} requests.")
            return
        st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)

    pending = [r for r in requests if str(r.get("status")) == "PENDING_REVIEW"]
    if pending:
        st.markdown("#### Pending Actions — Review Required")
        for req in pending:
            with st.container(border=True):
                st.markdown(f"**{req.get('candidate_action')}** — {req.get('reason')}")
                st.caption(
                    f"Request ID: `{req.get('request_id')}` | "
                    f"Est. risk reduction: {req.get('estimated_risk_reduction')} | "
                    f"Created: {req.get('created_at')}"
                )
                note = st.text_input(
                    "Reviewer note (optional)",
                    key=f"note_{req.get('request_id')}",
                    value="",
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Approve (queue only)", key=f"approve_{req.get('request_id')}"):
                        ok, msg = set_approval_request_status(
                            queue_path,
                            str(req.get("request_id")),
                            "APPROVED",
                            reviewer_note=note,
                        )
                        if ok:
                            st.success("Marked APPROVED — no execution performed.")
                            st.rerun()
                        else:
                            st.error(msg)
                with col_b:
                    if st.button("Reject", key=f"reject_{req.get('request_id')}"):
                        ok, msg = set_approval_request_status(
                            queue_path,
                            str(req.get("request_id")),
                            "REJECTED",
                            reviewer_note=note,
                        )
                        if ok:
                            st.info("Marked REJECTED — no execution performed.")
                            st.rerun()
                        else:
                            st.error(msg)

    _show_group("Approved Actions", "APPROVED")
    _show_group("Rejected Actions", "REJECTED")
    _show_group("Expired Actions", "EXPIRED")


def render_protective_action_policy_center(results_dir: Path) -> None:
    st.markdown("### 🛡️ Protective Action Policy Center")
    st.caption(
        "Future protective action policy definitions. **All policies default to disabled.** "
        "No automated execution."
    )

    doc = _load_json(results_dir / "protective_action_policy.json")
    if not doc:
        st.warning("Policy file not found. Run the watchdog first.")
        return

    st.caption(f"Generated: **{doc.get('generated_at', '—')}** | Mode: **{doc.get('mode', '—')}**")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Global Execution Enabled", str(doc.get("global_execution_enabled", False)))
    g2.metric("Paper Execution Enabled", str(doc.get("paper_execution_enabled", False)))
    g3.metric("Live Execution Enabled", str(doc.get("live_execution_enabled", False)))
    g4.metric("Enabled Policies", doc.get("enabled_policy_count", 0))

    policies = doc.get("policies") or []
    if policies:
        df = pd.DataFrame(
            [
                {
                    "Action": p.get("action"),
                    "Policy Type": p.get("policy_type"),
                    "Enabled": p.get("enabled"),
                    "Requires Approval": p.get("requires_human_approval"),
                    "Requires Green Governance": p.get("requires_green_governance"),
                    "Description": p.get("description"),
                }
                for p in policies
                if isinstance(p, dict)
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.warning(
        "Live execution remains **disabled**. Paper execution requires "
        "`TRITON_ENABLE_PAPER_EXECUTION=1` plus governance/operator approval. "
        "This layer does not place orders."
    )

    with st.expander("Full policy document", expanded=False):
        st.json(doc)
