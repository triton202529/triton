"""
TRITON Phase 200 — Governance Recovery Roadmap dashboard.
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


def _band_color(band: str) -> str:
    return {
        "GOVERNANCE_READY": "🟢",
        "NEAR_AUTHORIZATION": "🟡",
        "RECOVERING": "🟠",
        "NOT_READY": "🔴",
    }.get(band, "⚪")


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛣 [GOVERNANCE_RECOVERY_ROADMAP_REPORT]")
    ex = report.get("executive_recommendation") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Governance State", str(ex.get("current_governance_state", "—")))
    c2.metric("CPS", str(ex.get("current_cps", "—")))
    c3.metric("P(YELLOW)", str(ex.get("recovery_probability_yellow", "—")))
    c4.metric("Progress", f"{report.get('recovery_progress_pct', '—')}%")
    st.info(
        f"YELLOW est. **{ex.get('estimated_authorization_yellow_date', '—')}** · "
        f"GREEN est. **{ex.get('estimated_authorization_green_date', '—')}**",
        icon="💡",
    )


def _render_progress(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Recovery Progress Meter")
    pct = float(report.get("recovery_progress_pct") or 0)
    st.progress(min(1.0, pct / 100.0))
    st.markdown(f"Overall roadmap completion: **{pct:.1f}%**")
    readiness = report.get("governance_readiness_score") or {}
    band = str(readiness.get("classification", "—"))
    st.markdown(
        f"{_band_color(band)} Readiness: **{band}** ({readiness.get('governance_readiness_score', '—')})"
    )


def _render_cps_journey(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 CPS Journey (ORANGE → YELLOW → GREEN)")
    m = report.get("metrics") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Current CPS", str(m.get("cps", "—")))
    c2.metric("Δ to YELLOW (75)", str(m.get("distance_to_yellow", "—")))
    c3.metric("Δ to GREEN (90)", str(m.get("distance_to_green", "—")))
    st.markdown(
        f"Escalation: **{m.get('escalation', '—')}** · QQQ concentration: **{m.get('qqq_concentration_pct', '—')}%**"
    )


def _render_evidence(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Evidence Completion Status")
    summary = report.get("evidence_summary") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Collected", str(summary.get("collected", 0)))
    c2.metric("Pending", str(summary.get("pending", 0)))
    c3.metric("Total", str(summary.get("total", 0)))
    df = _read_csv_safe(results_dir / "governance_recovery_evidence_log.csv")
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_milestones(results_dir: Path) -> None:
    st.markdown("#### 🏁 Milestone Tracker")
    df = _read_csv_safe(results_dir / "governance_recovery_milestones.csv")
    if df.empty:
        st.info("No milestones.", icon="ℹ️")
        return
    cols = [
        c
        for c in [
            "milestone_id",
            "title",
            "status",
            "priority",
            "owner",
            "completion_percentage",
            "expected_completion_date",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_roadmap(report: Dict[str, Any]) -> None:
    st.markdown("#### 🗺 Recovery Roadmap Stages")
    stages = report.get("recovery_roadmap") or []
    if stages:
        st.dataframe(pd.DataFrame(stages), use_container_width=True, hide_index=True)


def _render_probability(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Authorization Probability")
    prob = report.get("authorization_probability") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("YELLOW", f"{prob.get('probability_yellow_pct', '—')}%")
    c2.metric("GREEN", f"{prob.get('probability_green_pct', '—')}%")
    c3.metric("Confidence", str(prob.get("confidence_level", "—")))
    st.markdown(
        f"Restoration: **{prob.get('probability_governance_restoration_pct')}%** · "
        f"Delay risk: **{prob.get('probability_delayed_recovery_pct')}%** · "
        f"Failure risk: **{prob.get('probability_recovery_failure_pct')}%**"
    )


def _render_recommendations(report: Dict[str, Any]) -> None:
    st.markdown("#### 💡 Executive Recommendations")
    ex = report.get("executive_recommendation") or {}
    for action in ex.get("recommended_next_actions") or []:
        st.markdown(f"- {action}")
    risks = ex.get("top_risks") or []
    if risks:
        with st.expander("Top risks"):
            for r in risks:
                st.markdown(f"- {r}")


def _render_history(results_dir: Path) -> None:
    st.markdown("#### 📉 Progress History")
    df = _read_csv_safe(results_dir / "governance_recovery_progress_history.csv")
    if not df.empty:
        st.dataframe(df.tail(15).iloc[::-1], use_container_width=True, hide_index=True)


def render_governance_recovery_roadmap_tracker(results_dir: Path) -> None:
    st.markdown("### 🛣 Governance Recovery Roadmap & Evidence Tracker")
    st.caption(
        "Measurable institutional recovery pathway with evidence tracking. "
        "Refresh with `python -m services.governance_recovery_roadmap_tracker`."
    )
    report = _read_json_safe(results_dir / "governance_recovery_roadmap.json")
    if not report:
        st.warning(
            "No roadmap report. Run `python -m services.governance_recovery_roadmap_tracker`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_progress(report)
    _render_cps_journey(report)
    _render_roadmap(report)
    _render_milestones(results_dir)
    _render_evidence(results_dir, report)
    _render_probability(report)
    _render_history(results_dir)
    _render_recommendations(report)
