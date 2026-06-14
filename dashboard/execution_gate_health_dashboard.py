"""
TRITON THP-007 — Execution Gate Health & Continuous Supervision Dashboard (read-only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from services.execution_gate_anomaly_detector import ANOMALIES_JSON, load_anomalies
from services.execution_gate_degradation_manager import (
    DEGRADATION_EVENTS_JSON,
    OPERATIONAL_STATUS_JSON,
    load_degradation_events,
    load_operational_status,
    run_continuous_supervision,
)
from services.execution_gate_health_monitor import (
    HEALTH_HISTORY_CSV,
    HEALTH_REPORT_JSON,
    load_health_report,
)

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _health_badge(classification: str) -> str:
    return {
        "GREEN": "🟢 GREEN — Institutional Grade",
        "YELLOW": "🟡 YELLOW — Warning",
        "ORANGE": "🟠 ORANGE — Degraded",
        "RED": "🔴 RED — Critical",
    }.get(classification, classification)


def _state_badge(state: str) -> str:
    return {
        "MONITOR": "✅ MONITOR",
        "WARNING": "⚠️ WARNING",
        "RESTRICT": "🟠 RESTRICT",
        "LOCKDOWN": "🛑 LOCKDOWN",
    }.get(state, state)


def page_execution_gate_health() -> None:
    st.title("🛡 Execution Gate Health & Continuous Supervision")
    st.info(
        "**Read-only institutional supervision.** Monitors gate health, detects anomalies, "
        "and classifies degradation. No broker calls or trading behavior changes."
    )

    if st.button("Run Continuous Supervision", key="eg_run_supervision"):
        with st.spinner("Running health monitor, anomaly detection, degradation manager..."):
            run_continuous_supervision()
        st.success("Supervision cycle complete.")

    status = load_operational_status() or _read_json(OPERATIONAL_STATUS_JSON)
    health = load_health_report() or _read_json(HEALTH_REPORT_JSON)
    anomalies = load_anomalies() or _read_json(ANOMALIES_JSON)
    degradation = load_degradation_events() or _read_json(DEGRADATION_EVENTS_JSON)

    score = health.get("health_score", status.get("health_score", 0))
    classification = health.get("health_classification", "UNKNOWN")
    op_state = status.get("operational_state", "MONITOR")

    st.markdown("### Overall Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Health Score", f"{score}/100")
    c2.markdown(_health_badge(classification))
    c3.markdown(_state_badge(op_state))
    c4.metric("Gate Available", "Yes" if status.get("gate_available") else "No")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Certification", status.get("certification_level", "—"))
    c6.metric("Resilience Score", status.get("resilience_score", "—"))
    c7.metric("Active Anomalies", status.get("anomaly_count", 0))
    c8.metric("Drift Events", status.get("drift_count", 0))

    st.markdown("### Decision Analytics")
    analytics = health.get("decision_analytics") or {}
    d1, d2, d3 = st.columns(3)
    d1.metric("Recent ALLOW (24h)", analytics.get("recent_allow_count", 0))
    d2.metric("Recent BLOCK (24h)", analytics.get("recent_block_count", 0))
    d3.metric("Recent Total", analytics.get("recent_total", 0))
    paths = analytics.get("most_active_paths") or []
    if paths:
        path_df = pd.DataFrame(paths, columns=["execution_path", "count"])
        st.bar_chart(path_df.set_index("execution_path")["count"])

    st.markdown("### Health Components")
    components = health.get("components") or []
    if components:
        comp_df = pd.DataFrame(components)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        if "component" in comp_df.columns and "score" in comp_df.columns:
            st.bar_chart(comp_df.set_index("component")["score"])

    st.markdown("### Governance Health")
    gov = health.get("governance_health") or {}
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Decisions", gov.get("decision_count", 0))
    g2.metric("Evidence Records", gov.get("evidence_count", 0))
    g3.metric("Evidence Age (h)", gov.get("evidence_age_hours") or "—")
    g4.metric(
        "Paper/Live Boundary",
        "Intact" if status.get("paper_live_boundary_intact") else "VIOLATED",
    )

    st.markdown("### Active Restrictions")
    restrictions = status.get("restrictions") or {}
    if restrictions.get("capital_deployment_blocked"):
        st.error(
            "🛑 LOCKDOWN: New capital deployment blocked. Emergency risk actions remain available."
        )
    elif restrictions.get("new_capital_deployment_restricted"):
        st.warning("🟠 RESTRICT: New capital deployment restricted pending governance review.")
    elif restrictions.get("enhanced_monitoring"):
        st.warning("⚠️ WARNING: Enhanced monitoring active.")
    else:
        st.success("✅ MONITOR: Normal supervision cadence.")

    st.markdown("### Anomalies")
    anomaly_summary = anomalies.get("summary") or {}
    a1, a2, a3 = st.columns(3)
    a1.metric("Total", anomaly_summary.get("total", 0))
    a2.metric("Critical", anomaly_summary.get("critical", 0))
    a3.metric("High", anomaly_summary.get("high", 0))
    if anomalies.get("records"):
        st.dataframe(pd.DataFrame(anomalies["records"]), use_container_width=True, hide_index=True)

    st.markdown("### Degradation Events")
    if degradation.get("records"):
        st.dataframe(
            pd.DataFrame(degradation["records"][-10:]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No degradation state transitions recorded.")

    st.markdown("### Health History")
    if HEALTH_HISTORY_CSV.is_file():
        try:
            hist_df = pd.read_csv(HEALTH_HISTORY_CSV)
            st.dataframe(hist_df.tail(20), use_container_width=True, hide_index=True)
            if "health_score" in hist_df.columns and "timestamp" in hist_df.columns:
                chart_df = hist_df.tail(50).copy()
                chart_df["health_score"] = pd.to_numeric(chart_df["health_score"], errors="coerce")
                st.line_chart(chart_df.set_index("timestamp")["health_score"])
        except Exception:
            st.caption("Health history unavailable.")
    else:
        st.caption("Run supervision to populate health history.")

    st.markdown("### Operational Status Artifact")
    if status:
        st.json(status)


def render() -> None:
    page_execution_gate_health()
