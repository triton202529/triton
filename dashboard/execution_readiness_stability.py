"""
TRITON Phase 198 — Execution Readiness Stability dashboard.
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
        "INSTITUTIONAL_GRADE": "🟢",
        "STABLE": "🟡",
        "DEVELOPING": "🟠",
        "UNSTABLE": "🔴",
    }.get(band, "⚪")


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔎 [EXECUTION_READINESS_STABILITY_REPORT]")
    ex = report.get("executive_recommendation") or {}
    stab = report.get("stability_score") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stability Score", str(stab.get("stability_score", "—")))
    c2.metric("Classification", str(stab.get("classification", "—")))
    c3.metric("Readiness", str(report.get("current_readiness", {}).get("status", "—")))
    cert = report.get("certification_impact") or {}
    c4.metric(
        "Gov Block Only",
        "YES" if cert.get("intentional_governance_block_only") else "NO",
    )
    st.info(ex.get("institutional_learning", ""), icon="💡")
    actions = ex.get("recommended_actions") or []
    if actions:
        with st.expander("Recommended actions"):
            for a in actions:
                st.markdown(f"- {a}")


def _render_stability(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Stability Score")
    stab = report.get("stability_score") or {}
    band = str(stab.get("classification", "—"))
    st.markdown(f"{_band_color(band)} **{band}** — score **{stab.get('stability_score', '—')}**")
    factors = stab.get("factors") or {}
    if factors:
        st.dataframe(
            pd.DataFrame([{"factor": k, "score": v} for k, v in factors.items()]),
            use_container_width=True,
            hide_index=True,
        )


def _render_timeline(report: Dict[str, Any]) -> None:
    st.markdown("#### ⏱ Readiness Failure Timeline")
    timeline = report.get("failure_timeline") or []
    if not timeline:
        st.info("No failure episodes recorded.", icon="ℹ️")
        return
    df = pd.DataFrame(timeline)
    cols = [
        c
        for c in [
            "check_name",
            "failure_type",
            "first_seen",
            "last_seen",
            "duration_minutes",
            "operational_impact",
            "recovery_behavior",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


def _render_classification(results_dir: Path) -> None:
    st.markdown("#### 🏷 Failure Classification")
    df = _read_csv_safe(results_dir / "readiness_failure_patterns.csv")
    if df.empty:
        st.info("No patterns file.", icon="ℹ️")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_governance(report: Dict[str, Any]) -> None:
    st.markdown("#### 🏛 Governance Restrictions")
    gov = report.get("governance_awareness") or {}
    st.markdown(f"**Classification:** {gov.get('classification', '—')}")
    st.markdown(f"**Escalation:** {gov.get('escalation_state', '—')}")
    st.markdown(f"**Detail:** {gov.get('detail', '—')}")
    st.caption(gov.get("note", ""))


def _render_technical(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚙️ Technical Failures")
    cert = report.get("certification_impact") or {}
    tech = cert.get("technical_failures") or []
    if tech:
        for t in tech:
            st.error(f"Failed: {t}", icon="⚠️")
    else:
        st.success("No technical operational failures in current snapshot.", icon="✅")
    cur = report.get("current_readiness") or {}
    failed = cur.get("failed_checks") or []
    if failed:
        st.markdown(f"Current failed checks: **{', '.join(failed)}**")


def _render_watchdog(report: Dict[str, Any]) -> None:
    st.markdown("#### 🐕 Watchdog Reliability")
    wd = report.get("watchdog_reliability") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Stability Score", str(wd.get("watchdog_stability_score", "—")))
    c2.metric("False Positive Prob", str(wd.get("false_positive_probability", "—")))
    c3.metric("Max Consecutive Fails", str(wd.get("consecutive_failures_max", "—")))
    st.markdown(wd.get("assessment", "—"))


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 📜 Certification Impact")
    cert = report.get("certification_impact") or {}
    rows = [
        ("Actual status", cert.get("actual_readiness_status")),
        ("If temporary anomalies removed", cert.get("status_if_temporary_anomalies_removed")),
        ("Operational readiness (auth ignored)", cert.get("operational_readiness_status")),
        ("Under governance-only block", cert.get("status_with_intentional_governance_block")),
    ]
    for label, val in rows:
        st.markdown(f"- **{label}:** {val or '—'}")
    st.info(cert.get("certification_summary", ""), icon="ℹ️")


def _render_roadmap(report: Dict[str, Any]) -> None:
    st.markdown("#### 🗺 Improvement Roadmap")
    ex = report.get("executive_recommendation") or {}
    eq = ex.get("executive_questions") or {}
    st.markdown(f"**Actually unhealthy?** {eq.get('is_triton_actually_unhealthy')}")
    st.markdown(f"**Failure character:** {eq.get('are_failures_temporary_or_critical')}")
    st.markdown(f"**Governance intentional?** {eq.get('is_governance_blocking_intentionally')}")


def render_execution_readiness_stability(results_dir: Path) -> None:
    st.markdown("### 🔎 Execution Readiness Stability & Certification Integrity")
    st.caption(
        "Separates governance discipline from operational failure. "
        "Refresh with `python -m services.execution_readiness_stability`."
    )
    report = _read_json_safe(results_dir / "execution_readiness_stability.json")
    if not report:
        st.warning(
            "No stability report. Run `python -m services.execution_readiness_stability`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_stability(report)
    _render_timeline(report)
    _render_classification(results_dir)
    _render_governance(report)
    _render_technical(report)
    _render_watchdog(report)
    _render_certification(report)
    _render_roadmap(report)
