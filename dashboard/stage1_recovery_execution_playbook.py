"""
TRITON Phase 191 — Stage 1 Recovery Execution Governance Playbook dashboard (read-only).
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


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ [STAGE_1_RECOVERY_EXECUTION_GOVERNANCE_REPORT]")
    ex = report.get("executive_recommendation") or {}
    status = ex.get("current_recovery_status") or {}
    st.markdown(
        f"**Portfolio:** {status.get('portfolio')} · CPS {status.get('cps')} · "
        f"{status.get('escalation')} · {status.get('governance')}"
    )
    st.info(str(ex.get("recommended_decision") or ""), icon="💡")
    st.warning(
        "Playbook only — Stage 1 NOT authorized, exception NOT activated.",
        icon="⚠️",
    )
    qs = ex.get("executive_questions") or {}
    with st.expander("Executive Q&A"):
        for q, a in qs.items():
            st.markdown(f"**{q.replace('_', ' ').title()}** — {a}")


def _render_auth_gate(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔐 Authorization Gate")
    auth = report.get("authorization_requirements") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dual Approval", "YES" if auth.get("dual_approval_satisfied") else "NO")
    c2.metric("Operator Sig", "YES" if auth.get("operator_signature") else "PENDING")
    c3.metric("Risk Office Sig", "YES" if auth.get("risk_office_signature") else "PENDING")
    c4.metric("Stage 1 Auth", "NO")
    remaining = auth.get("remaining_requirements") or []
    if remaining:
        st.caption(f"Remaining: {', '.join(remaining)}")


def _render_market(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Market Readiness")
    market = report.get("market_readiness") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Session Open", "YES" if market.get("market_session_open") else "NO")
    c2.metric("Broker OK", "YES" if market.get("broker_healthy") else "NO")
    c3.metric("Pre-Market Ready", "YES" if market.get("pre_market_ready") else "NO")
    df = pd.DataFrame(market.get("pre_market_validation") or [])
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_checklist(results_dir: Path) -> None:
    st.markdown("#### ✅ Execution Checklist")
    df = _read_csv_safe(results_dir / "stage1_execution_checklist.csv")
    if df.empty:
        st.info("No checklist.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_sequence(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Execution Sequence")
    df = pd.DataFrame(report.get("execution_sequence") or [])
    if df.empty:
        st.info("No sequence.", icon="ℹ️")
        return
    cols = [
        c
        for c in [
            "step",
            "action",
            "owner",
            "human_required",
            "symbol",
            "target_change_pct",
            "gate",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_failures(results_dir: Path) -> None:
    st.markdown("#### ⚠️ Failure Scenario Matrix")
    df = _read_csv_safe(results_dir / "stage1_failure_scenarios.csv")
    if df.empty:
        st.info("No failure scenarios.", icon="ℹ️")
        return
    show = [
        c
        for c in ["scenario", "severity", "immediate_response", "exception_action"]
        if c in df.columns
    ]
    st.dataframe(df[show] if show else df, use_container_width=True, hide_index=True)


def _render_audit(results_dir: Path) -> None:
    st.markdown("#### 📎 Audit Requirements")
    df = _read_csv_safe(results_dir / "stage1_audit_requirements.csv")
    if df.empty:
        st.info("No audit requirements.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_oversight(report: Dict[str, Any]) -> None:
    st.markdown("#### 👤 Human Oversight Requirements")
    auto = report.get("autonomy_boundaries") or {}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Requires human approval**")
        for item in auto.get("requires_human_approval") or []:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("**Immediate suspension triggers**")
        for item in auto.get("causes_immediate_suspension") or []:
            st.markdown(f"- {item}")


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_stage1_recovery_execution_playbook(results_dir: Path) -> None:
    st.markdown("### 🛡️ Stage 1 Recovery Execution Governance Playbook")
    st.caption(
        "Operational playbook for Stage 1 recovery (QQQ 90% + PEP 10%) under human supervision. "
        "Refresh with `python -m services.stage1_recovery_execution_playbook`."
    )
    report = _read_json_safe(results_dir / "stage1_governance_playbook.json")
    if not report:
        st.warning(
            "No playbook found. Run `python -m services.stage1_recovery_execution_playbook`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_auth_gate(report)
    _render_market(report)
    _render_checklist(results_dir)
    _render_sequence(report)
    _render_failures(results_dir)
    _render_audit(results_dir)
    _render_oversight(report)
    _render_certification(report)
