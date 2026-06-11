"""
TRITON Phase 183 — Recovery Basket Implementation Readiness dashboard (read-only).
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


_CLASS = {
    "NOT_READY": "🔴 NOT_READY",
    "PARTIALLY_READY": "🟡 PARTIALLY_READY",
    "READY_FOR_REVIEW": "🟢 READY_FOR_REVIEW",
    "READY_FOR_IMPLEMENTATION": "🟢 READY_FOR_IMPLEMENTATION",
}

_RISK = {
    "LOW": "🟢 LOW",
    "MODERATE": "🟡 MODERATE",
    "HIGH": "🟠 HIGH",
    "CRITICAL": "🔴 CRITICAL",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 [RECOVERY_BASKET_IMPLEMENTATION_READINESS_REPORT]")
    ex = report.get("executive_recommendation") or {}
    qs = ex.get("executive_questions") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Recommendation", _CLASS.get(str(ex.get("implementation_recommendation")), "—"))
    c2.metric(
        "Ready for Review?",
        "✅ Yes" if ex.get("ready_for_implementation_review") else "❌ No",
    )
    c3.metric(
        "Ready for Implementation?",
        "✅ Yes" if ex.get("ready_for_implementation") else "❌ No",
    )
    st.markdown("**Must complete before implementation:**")
    for a in qs.get("must_complete_before_implementation") or []:
        st.markdown(f"- {a}")
    st.markdown("**Recommended actions:**")
    for a in ex.get("recommended_actions") or []:
        st.markdown(f"- {a}")


def _render_scorecard(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Readiness Scorecard")
    rs = report.get("readiness_scores") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Overall",
        f"{rs.get('overall_readiness', '—')} ({_CLASS.get(str(rs.get('overall_classification')), '—')})",
    )
    c2.metric("Governance", str(rs.get("governance_readiness", "—")))
    c3.metric("Lifecycle", str(rs.get("lifecycle_readiness", "—")))
    c4.metric("Execution", str(rs.get("execution_readiness", "—")))
    c5, c6, c7 = st.columns(3)
    c5.metric("Audit", str(rs.get("audit_readiness", "—")))
    c6.metric("Simulation", str(rs.get("simulation_readiness", "—")))
    c7.metric("Operational", str(rs.get("operational_readiness", "—")))
    dims = pd.DataFrame(
        [
            {"dimension": k.replace("_readiness", ""), "score": v}
            for k, v in rs.items()
            if k.endswith("_readiness") and k != "overall_readiness"
        ]
    )
    if not dims.empty:
        if not _plotly_bar(dims, "dimension", "score", "Readiness by Dimension"):
            st.bar_chart(dims.set_index("dimension")["score"])


def _render_dependencies(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🔗 Dependency Audit")
    df = _read_csv_safe(results_dir / "recovery_basket_dependency_audit.csv")
    if df.empty:
        deps = (report.get("dependency_audit") or {}).get("dependencies") or []
        df = pd.DataFrame(deps)
    if df.empty:
        st.info("No dependency audit.", icon="ℹ️")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Requires changes**")
        chg = (
            df[df["change_required"] == True] if "change_required" in df.columns else df
        )  # noqa: E712
        st.dataframe(chg, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**No changes required**")
        nc = (
            df[df["change_required"] == False]
            if "change_required" in df.columns
            else pd.DataFrame()
        )  # noqa: E712
        st.dataframe(nc, use_container_width=True, hide_index=True)


def _render_drift(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔄 Lifecycle Drift Analysis")
    drift = report.get("lifecycle_drift_analysis") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Drift Count", str(drift.get("drift_count", "—")))
    c2.metric("Implementation Risk", _RISK.get(str(drift.get("implementation_risk")), "—"))
    c3.metric("Frequency", str(drift.get("frequency", "—"))[:30])
    st.markdown(f"**Recovery impact:** {drift.get('recovery_impact') or '—'}")
    st.markdown("**Probable causes:**")
    for c in drift.get("probable_causes") or []:
        st.markdown(f"- {c}")
    rows = drift.get("symbols_analyzed") or drift.get("recovery_basket_drift") or []
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("**Mitigation:**")
    for m in drift.get("recommended_mitigation") or []:
        st.markdown(f"- {m}")


def _render_gaps(results_dir: Path) -> None:
    st.markdown("#### 🚧 Readiness Gaps")
    df = _read_csv_safe(results_dir / "recovery_basket_readiness_gaps.csv")
    if df.empty:
        st.info("No gaps recorded.", icon="ℹ️")
        return
    view = df.copy()
    if "severity" in view.columns:
        view["severity"] = view["severity"].map(lambda x: _RISK.get(str(x), str(x)))
    st.dataframe(view, use_container_width=True, hide_index=True)


def _render_risks(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚠️ Risk Analysis")
    ra = report.get("risk_analysis") or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Governance", _RISK.get(str(ra.get("governance_risk")), "—"))
    c2.metric("Lifecycle", _RISK.get(str(ra.get("lifecycle_risk")), "—"))
    c3.metric("Execution", _RISK.get(str(ra.get("execution_risk")), "—"))
    c4.metric("Audit", _RISK.get(str(ra.get("audit_risk")), "—"))
    c5.metric("Overall", _RISK.get(str(ra.get("overall_risk")), "—"))


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_recovery_basket_implementation_readiness(results_dir: Path) -> None:
    st.markdown("### 📋 Recovery Basket Implementation Readiness")
    st.caption(
        "Implementation readiness assessment for PATH_B Recovery Basket Exception — no implementation. "
        "Refresh with `python -m services.recovery_basket_implementation_readiness`."
    )
    report = _read_json_safe(results_dir / "recovery_basket_implementation_readiness.json")
    if not report:
        st.warning(
            "No readiness report found. Run `python -m services.recovery_basket_implementation_readiness`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_scorecard(report)
    _render_dependencies(results_dir, report)
    _render_drift(report)
    _render_gaps(results_dir)
    _render_risks(report)
    _render_certification(report)
