"""
TRITON Phase 184 — Lifecycle Drift Root Cause Analysis dashboard (read-only).
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


_SEV = {
    "NO_DRIFT": "🟢 NO_DRIFT",
    "MINOR_DRIFT": "🟡 MINOR_DRIFT",
    "MODERATE_DRIFT": "🟠 MODERATE_DRIFT",
    "MAJOR_DRIFT": "🟠 MAJOR_DRIFT",
    "CRITICAL_DRIFT": "🔴 CRITICAL_DRIFT",
}


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 [LIFECYCLE_DRIFT_ROOT_CAUSE_REPORT]")
    ex = report.get("executive_recommendation") or {}
    qs = ex.get("executive_questions") or {}
    st.markdown(f"**Current drift:** {ex.get('current_drift_summary') or '—'}")
    st.markdown(f"**Root cause:** {ex.get('root_cause') or '—'}")
    st.markdown(f"**Authoritative layer:** {ex.get('authoritative_layer') or '—'}")
    st.markdown(f"**Execution impact:** {ex.get('execution_impact') or '—'}")
    st.markdown(f"**Recovery impact:** {ex.get('recovery_impact') or '—'}")
    st.markdown("**Resolution path:**")
    for step in ex.get("recommended_resolution_path") or []:
        st.markdown(f"- {step}")
    st.info(str(qs.get("why_is_drift_occurring") or ""), icon="🔍")


def _render_summary(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Drift Summary")
    ds = report.get("drift_summary") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbols Analyzed", str(ds.get("symbols_analyzed", "—")))
    c2.metric("Drift Count", str(ds.get("drift_count", "—")))
    c3.metric("Critical", str(ds.get("critical_count", "—")))
    c4.metric("Primary Cause", str(ds.get("primary_root_cause", "—")))
    ts = ds.get("artifact_timestamps") or {}
    st.caption(
        f"Timestamps — allocation: {ts.get('allocation_recommendations', '—')} · "
        f"lifecycle: {ts.get('signal_lifecycle_effective', '—')}"
    )


def _render_candidates(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Affected Symbols")
    df = _read_csv_safe(results_dir / "lifecycle_drift_candidates.csv")
    if df.empty:
        df = pd.DataFrame(report.get("candidate_analysis") or [])
    if df.empty:
        st.info("No candidates.", icon="ℹ️")
        return
    focus = df[df["symbol"].isin(["HD", "XLK", "QQQ", "BITO"])] if "symbol" in df.columns else df
    view = focus.copy() if not focus.empty else df.copy()
    if "drift_severity" in view.columns:
        view["drift_severity"] = view["drift_severity"].map(lambda x: _SEV.get(str(x), str(x)))
    st.dataframe(view, use_container_width=True, hide_index=True)
    if "drift_severity" in df.columns:
        counts = df["drift_severity"].value_counts().reset_index()
        counts.columns = ["drift_severity", "count"]
        if not _plotly_bar(counts, "drift_severity", "count", "Drift Severity Distribution"):
            st.bar_chart(counts.set_index("drift_severity")["count"])


def _render_root_causes(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔬 Root Causes")
    rc = report.get("root_causes") or {}
    st.markdown(f"**Primary:** {rc.get('primary_root_cause') or '—'}")
    st.markdown(f"**Expected behavior?** {rc.get('is_expected_behavior')}")
    st.caption(str(rc.get("expected_behavior_note") or ""))
    for cause in rc.get("ranked_causes") or []:
        icon = "✅" if cause.get("confirmed") else "⚠️"
        st.markdown(
            f"{icon} **{cause.get('label')}** (weight={cause.get('weight')}) — {cause.get('description')}"
        )
    if rc.get("code_reference"):
        st.code(str(rc["code_reference"]))


def _render_impact(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Impact Analysis")
    imp = report.get("impact_analysis") or {}
    c1, c2 = st.columns(2)
    c1.metric("Execution Readiness", str(imp.get("execution_readiness_impact", "—")))
    c2.metric("Recovery Basket", str(imp.get("recovery_basket_viability", "—")))
    c3, c4 = st.columns(2)
    c3.metric("YELLOW Timeline", str(imp.get("yellow_timeline_impact_days", "—")))
    c4.metric("Forecast Confidence", str(imp.get("forecast_confidence_impact", "—")))


def _render_authority(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚖️ Authority Analysis")
    auth = report.get("authority_analysis") or {}
    st.markdown(
        f"**Authoritative for execution:** `{auth.get('authoritative_layer_for_execution')}`"
    )
    st.markdown(f"**Source:** `{auth.get('authoritative_source')}`")
    st.markdown(str(auth.get("recommendation") or ""))
    ranking = pd.DataFrame(auth.get("layer_ranking") or [])
    if not ranking.empty:
        st.dataframe(ranking, use_container_width=True, hide_index=True)


def _render_history(results_dir: Path) -> None:
    st.markdown("#### 🕐 Drift History")
    df = _read_csv_safe(results_dir / "lifecycle_drift_history.csv")
    if df.empty:
        st.info("No history.", icon="ℹ️")
        return
    focus = df[df["symbol"].isin(["HD", "XLK", "QQQ"])] if "symbol" in df.columns else df
    st.dataframe(focus if not focus.empty else df, use_container_width=True, hide_index=True)


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔒 Certification")
    cert = report.get("certification") or {}
    st.success(str(cert.get("safety") or ""), icon="✅")
    st.caption(str(report.get("disclaimer") or ""))


def render_lifecycle_drift_root_cause_analysis(results_dir: Path) -> None:
    st.markdown("### 🔄 Lifecycle Drift Root Cause Analysis")
    st.caption(
        "Compares allocation, lifecycle, execution, and recovery basket layers. "
        "Refresh with `python -m services.lifecycle_drift_root_cause_analysis`."
    )
    report = _read_json_safe(results_dir / "lifecycle_drift_analysis.json")
    if not report:
        st.warning(
            "No analysis found. Run `python -m services.lifecycle_drift_root_cause_analysis`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_summary(report)
    _render_candidates(results_dir, report)
    _render_root_causes(report)
    _render_impact(report)
    _render_authority(report)
    _render_history(results_dir)
    _render_certification(report)
