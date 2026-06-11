"""
TRITON Phase 181 — Recovery Basket Exception Governance Framework dashboard (read-only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _plotly_bar


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


_ABUSE = {
    "LOW": "🟢 LOW",
    "MODERATE": "🟡 MODERATE",
    "HIGH": "🟠 HIGH",
    "CRITICAL": "🔴 CRITICAL",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 [RECOVERY_BASKET_EXCEPTION_GOVERNANCE_REPORT]")
    ex = report.get("executive_recommendation") or {}
    qs = ex.get("executive_questions") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended Model", str(ex.get("recommended_governance_model", "—")))
    c2.metric("Limits Tier", str(ex.get("exception_limits_tier", "—")))
    c3.metric(
        "Governance Integrity",
        str((ex.get("governance_integrity_assessment") or {}).get("score", "—")),
    )
    st.markdown(f"**Who can approve?** {qs.get('who_can_approve') or '—'}")
    st.markdown(f"**Eligible assets:** {qs.get('which_assets_qualify') or '—'}")
    st.markdown(f"**Max duration:** {qs.get('how_long_active') or '—'}")
    st.markdown(f"**Revocation:** {qs.get('how_revoked') or '—'}")
    st.markdown(f"**Audit:** {qs.get('how_audited') or '—'}")
    st.info(str(ex.get("recommended_approach") or ""), icon="💡")


def _render_framework(report: Dict[str, Any]) -> None:
    st.markdown("#### 🏛 Governance Framework")
    fs = report.get("framework_summary") or {}
    cs = report.get("current_state") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPS", str(cs.get("cps", "—")))
    c2.metric("Escalation", str(cs.get("escalation", "—")))
    c3.metric("Deadlock", str(cs.get("deadlock_classification", "—")))
    c4.metric("Abuse Risk", _ABUSE.get(str(fs.get("abuse_risk")), str(fs.get("abuse_risk", "—"))))
    reg = report.get("recovery_basket_registry") or {}
    st.markdown(
        f"**Registry (active now):** {', '.join(reg.get('eligible_for_exception_now') or [])}"
    )
    st.markdown(f"**Max duration:** {fs.get('max_duration_hours', '—')}h")


def _render_approval_models(report: Dict[str, Any]) -> None:
    st.markdown("#### ✅ Approval Models")
    df = pd.DataFrame(report.get("approval_models") or [])
    if df.empty:
        st.info("No approval models.", icon="ℹ️")
        return
    view = df.copy()
    if "recommended_for_path_b" in view.columns:
        view["recommended_for_path_b"] = view["recommended_for_path_b"].map(
            lambda x: "✅" if x else "—"
        )
    st.dataframe(view, use_container_width=True, hide_index=True)
    if "composite_score" in df.columns:
        chart = df[["model", "composite_score"]].copy()
        if not _plotly_bar(chart, "model", "composite_score", "Approval Model Composite Score"):
            st.bar_chart(chart.set_index("model")["composite_score"])


def _render_authorization_scope(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔐 Authorization Scope")
    scope = report.get("authorization_scope") or {}
    st.markdown(f"**Recommended:** {scope.get('recommended_model') or '—'}")
    st.markdown(f"**Safest combination:** {scope.get('safest_combination') or '—'}")
    opts = pd.DataFrame(scope.get("scope_options") or [])
    if not opts.empty:
        st.dataframe(opts, use_container_width=True, hide_index=True)
    st.markdown("**Token fields**")
    for f in scope.get("authorization_token_fields") or []:
        st.markdown(f"- `{f}`")


def _render_exception_limits(report: Dict[str, Any]) -> None:
    st.markdown("#### ⏱ Exception Limits")
    limits = report.get("exception_limits") or {}
    st.markdown(f"**Recommended tier:** {limits.get('recommended_tier') or '—'}")
    tiers = limits.get("tiers") or {}
    rows = [{"tier": k, **v} for k, v in tiers.items()]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_revocation(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛑 Revocation Rules")
    rev = report.get("revocation_rules") or {}
    df = pd.DataFrame(rev.get("all_rules") or [])
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_audit(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📜 Audit Requirements")
    audit = report.get("audit_requirements") or {}
    st.markdown("**Mandatory logs**")
    for log in audit.get("mandatory_logs") or []:
        st.markdown(f"- `{log}`")
    df = _read_csv_safe(results_dir / "recovery_basket_exception_audit_model.csv")
    if not df.empty:
        st.markdown("**Audit event model**")
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_controls(results_dir: Path) -> None:
    st.markdown("#### 🛡️ Controls Catalog")
    df = _read_csv_safe(results_dir / "recovery_basket_exception_controls.csv")
    if df.empty:
        st.info("No controls catalog.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    if "control_strength" in df.columns:
        chart = df[["control_name", "control_strength"]].head(8)
        if not _plotly_bar(chart, "control_name", "control_strength", "Top Control Strength"):
            st.bar_chart(chart.set_index("control_name")["control_strength"])


def _render_abuse(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚠️ Abuse Analysis")
    abuse = report.get("abuse_analysis") or {}
    c1, c2 = st.columns(2)
    c1.metric("Overall Risk", _ABUSE.get(str(abuse.get("overall_abuse_risk")), "—"))
    c2.metric(
        "Residual (with controls)",
        _ABUSE.get(str(abuse.get("residual_risk_with_framework")), "—"),
    )
    for v in abuse.get("vectors") or []:
        with st.expander(str(v.get("question"))):
            st.markdown(f"**Assessment:** {v.get('assessment')}")
            st.markdown(f"**Risk:** {_ABUSE.get(str(v.get('risk_level')), v.get('risk_level'))}")
            st.markdown(f"**Controls:** {', '.join(v.get('controls') or [])}")


def _render_monitoring(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Monitoring")
    mon = report.get("monitoring_requirements") or {}
    df = pd.DataFrame(mon.get("tracked_metrics") or [])
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_recovery_basket_exception_governance(results_dir: Path) -> None:
    st.markdown("### 🧺 Recovery Basket Exception Governance Framework")
    st.caption(
        "PATH_B governance control design — approval, scope, limits, audit, and revocation. "
        "Refresh with `python -m services.recovery_basket_exception_governance`."
    )
    report = _read_json_safe(results_dir / "recovery_basket_exception_governance.json")
    if not report:
        st.warning(
            "No framework found. Run `python -m services.recovery_basket_exception_governance`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_framework(report)
    _render_approval_models(report)
    _render_authorization_scope(report)
    _render_exception_limits(report)
    _render_revocation(report)
    _render_audit(results_dir, report)
    _render_controls(results_dir)
    _render_monitoring(report)
    _render_abuse(report)
    _render_certification(report)
