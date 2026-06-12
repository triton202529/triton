"""
TRITON Phase 189 — Recovery Transition Authorization Assessment dashboard (read-only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st


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
    "NOT_READY": "🔴 NOT_READY",
    "READY_FOR_REVIEW": "🟡 READY_FOR_REVIEW",
    "READY_FOR_STAGE_1": "🟢 READY_FOR_STAGE_1",
    "READY_FOR_AUTONOMOUS_TRANSITION": "🟢 AUTONOMOUS",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 [RECOVERY_TRANSITION_AUTHORIZATION_REPORT]")
    ex = report.get("executive_recommendation") or {}
    st.markdown(f"**Authorization status:** {ex.get('authorization_status') or '—'}")
    st.warning(str(ex.get("authorization_note") or ""), icon="⚠️")
    st.markdown(f"**Stage 1 can begin (technical)?** {ex.get('stage_1_can_begin')}")
    st.info(str(ex.get("recommended_next_action") or ""), icon="💡")
    qs = ex.get("executive_questions") or {}
    with st.expander("Executive Q&A"):
        for q, a in qs.items():
            st.markdown(f"**{q.replace('_', ' ').title()}** — {a}")


def _render_current_vs_target(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚖️ Current vs Stage 1 Target")
    cur = report.get("current_state") or {}
    tgt = (report.get("executive_recommendation") or {}).get("target_stage_1_transition") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current CPS", str(cur.get("current_cps", "—")))
    c2.metric("Escalation", str(cur.get("escalation", "—")))
    c3.metric("QQQ Weight", f"{(cur.get('weights') or {}).get('QQQ', 100)}%")
    c4.metric("Governance", str(cur.get("governance_status", "—")))
    st.markdown(f"**Stage 1 target:** `{tgt.get('target_state')}`")
    st.caption(
        f"Projected after Stage 1: CPS {tgt.get('projected_cps')} · {tgt.get('projected_escalation')}"
    )


def _render_readiness(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Readiness Scorecard")
    auto = report.get("autonomous_readiness") or {}
    classification = auto.get("overall_classification") or report.get("summary", {}).get(
        "overall_classification"
    )
    st.markdown(f"**Overall:** {_CLASS.get(str(classification), classification)}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Governance", str(auto.get("governance_readiness", "—")))
    c2.metric("Lifecycle", str(auto.get("lifecycle_readiness", "—")))
    c3.metric("Execution", str(auto.get("execution_readiness", "—")))
    c4.metric("Risk", str(auto.get("risk_readiness", "—")))
    c5.metric("Operational", str(auto.get("operational_readiness", "—")))
    st.caption(f"Composite: {auto.get('overall_transition_readiness')} · Autonomous: No")


def _render_lifecycle(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔄 Lifecycle Validation")
    df = pd.DataFrame(report.get("candidate_validation") or [])
    if df.empty:
        st.info("No candidate data.", icon="ℹ️")
        return
    cols = [
        c
        for c in [
            "symbol",
            "stage_1_required",
            "effective_stance",
            "confidence",
            "edge_score",
            "validation_status",
            "execution_eligibility",
            "basket_still_valid",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_governance(report: Dict[str, Any]) -> None:
    st.markdown("#### 🏛 Governance Approval Status")
    gov = report.get("governance_readiness") or {}
    c1, c2 = st.columns(2)
    c1.metric("Dual Approval Satisfied", "Yes" if gov.get("dual_approval_satisfied") else "No")
    c2.metric("Scope Update Required", "Yes" if gov.get("scope_update_required") else "No")
    st.markdown(f"**Exception model:** {gov.get('dual_approval_model') or '—'}")
    st.markdown(f"**Framework scope (stale):** {gov.get('exception_scope_current')}")
    st.markdown(f"**Required scope:** {gov.get('exception_scope_required')}")


def _render_execution(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚙️ Execution Readiness")
    ex = report.get("execution_readiness") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Paper Permitted", str(ex.get("paper_execution_permitted", "—")))
    c2.metric("Market Open", str(ex.get("market_session_open", "—")))
    c3.metric("Status", str(ex.get("readiness_status", "—")))


def _render_checklist(results_dir: Path) -> None:
    st.markdown("#### ✅ Readiness Checklist")
    df = _read_csv_safe(results_dir / "recovery_transition_readiness_checklist.csv")
    if df.empty:
        st.info("No checklist.", icon="ℹ️")
        return
    fails = df[df["status"] == "FAIL"] if "status" in df.columns else df
    st.dataframe(fails if not fails.empty else df, use_container_width=True, hide_index=True)


def _render_blockers(results_dir: Path) -> None:
    st.markdown("#### ⛔ Blocker Analysis")
    df = _read_csv_safe(results_dir / "recovery_transition_blockers.csv")
    if df.empty:
        st.info("No blockers.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_recovery_transition_authorization_assessment(results_dir: Path) -> None:
    st.markdown("### 🛂 Recovery Transition Authorization Assessment")
    st.caption(
        "Stage 1 authorization readiness gate (assessment only — does not authorize). "
        "Refresh with `python -m services.recovery_transition_authorization_assessment`."
    )
    report = _read_json_safe(results_dir / "recovery_transition_authorization_assessment.json")
    if not report:
        st.warning(
            "No assessment found. Run `python -m services.recovery_transition_authorization_assessment`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_current_vs_target(report)
    _render_readiness(report)
    _render_lifecycle(report)
    _render_governance(report)
    _render_execution(report)
    _render_checklist(results_dir)
    _render_blockers(results_dir)
    _render_certification(report)
