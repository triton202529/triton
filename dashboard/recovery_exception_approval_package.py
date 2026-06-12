"""
TRITON Phase 190 — Recovery Exception Approval Package dashboard (read-only).
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
    st.markdown("#### 📋 [RECOVERY_EXCEPTION_APPROVAL_REPORT]")
    ex = report.get("executive_recommendation") or {}
    gov = ex.get("current_governance_status") or {}
    mig = ex.get("registry_migration_summary") or {}
    st.markdown(
        f"**Governance:** CPS {gov.get('cps')} · {gov.get('escalation')} · {gov.get('governance')}"
    )
    st.markdown(
        f"**Migration:** {mig.get('classification')} · Removed {mig.get('removed')} · "
        f"Proposed {mig.get('proposed')}"
    )
    st.info(str(ex.get("recommended_approval_decision") or ""), icon="💡")
    st.warning("Exception NOT activated — preparation package only.", icon="⚠️")
    qs = ex.get("executive_questions") or {}
    with st.expander("Executive Q&A"):
        for q, a in qs.items():
            st.markdown(f"**{q.replace('_', ' ').title()}** — {a}")


def _render_block(report: Dict[str, Any]) -> None:
    st.markdown("#### ⛔ Current Governance Block")
    state = report.get("current_state") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPS", str(state.get("current_cps", "—")))
    c2.metric("Escalation", str(state.get("escalation", "—")))
    c3.metric("Governance", str(state.get("governance_status", "—")))
    c4.metric("QQQ %", f"{(state.get('weights') or {}).get('QQQ', 100)}%")


def _render_migration(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔄 Registry Migration")
    mig = report.get("registry_migration") or {}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Removed**")
        for r in mig.get("assets_removed") or []:
            st.markdown(f"- **{r.get('symbol')}** — {r.get('removal_reason')}")
    with c2:
        st.markdown("**Added**")
        for a in mig.get("assets_added") or []:
            st.markdown(
                f"- **{a.get('symbol')}** — max {a.get('max_weight_pct')}% · "
                f"conf {a.get('confidence')}"
            )
    st.caption(f"Classification: **{mig.get('migration_classification')}**")


def _render_lifecycle(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔄 Lifecycle Evidence")
    df = pd.DataFrame(report.get("candidate_validation") or [])
    if df.empty:
        st.info("No validation data.", icon="ℹ️")
        return
    cols = [
        c
        for c in [
            "symbol",
            "effective_stance",
            "confidence",
            "diversification_score",
            "cps_improvement_potential",
            "recovery_classification",
            "lifecycle_valid",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_checklist(results_dir: Path) -> None:
    st.markdown("#### ✅ Approval Checklist")
    df = _read_csv_safe(results_dir / "recovery_exception_approval_checklist.csv")
    if df.empty:
        st.info("No checklist.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_risk_office(report: Dict[str, Any]) -> None:
    st.markdown("#### 🏛 Risk Office Review")
    pkg = (report.get("approval_package") or {}).get("risk_office_approval_package") or {}
    st.markdown(f"**Scope:** {pkg.get('exception_scope')}")
    st.markdown(f"**Duration limits:** {pkg.get('duration_limits')}")
    st.markdown(f"**Allocation limits:** {pkg.get('allocation_limits')}")


def _render_operator(report: Dict[str, Any]) -> None:
    st.markdown("#### 👤 Operator Review")
    pkg = (report.get("approval_package") or {}).get("operator_approval_package") or {}
    st.markdown(f"**Objective:** {pkg.get('transition_objective')}")
    st.markdown(f"**CPS trajectory:** {pkg.get('expected_cps_trajectory')}")
    for lim in pkg.get("execution_limitations") or []:
        st.markdown(f"- {lim}")


def _render_safety(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Safety Assessment")
    safety = report.get("safety_assessment") or {}
    st.markdown(f"**Safe for review:** {safety.get('overall_safe_for_review')}")
    for chk in safety.get("checks") or []:
        icon = "✅" if chk.get("passed") else "❌"
        st.markdown(f"{icon} {chk.get('check')} — {chk.get('detail')}")


def _render_evidence(results_dir: Path) -> None:
    st.markdown("#### 📎 Evidence Package")
    df = _read_csv_safe(results_dir / "recovery_exception_evidence_package.csv")
    if df.empty:
        st.info("No evidence.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_recovery_exception_approval_package(results_dir: Path) -> None:
    st.markdown("### 📋 Recovery Exception Approval Package")
    st.caption(
        "Registry migration and DUAL_APPROVAL package for Stage 1 review. "
        "Refresh with `python -m services.recovery_exception_approval_package`."
    )
    report = _read_json_safe(results_dir / "recovery_exception_registry_migration.json")
    if not report:
        st.warning(
            "No approval package found. Run `python -m services.recovery_exception_approval_package`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_block(report)
    _render_migration(report)
    _render_lifecycle(report)
    _render_checklist(results_dir)
    _render_risk_office(report)
    _render_operator(report)
    _render_safety(report)
    _render_evidence(results_dir)
    _render_certification(report)
