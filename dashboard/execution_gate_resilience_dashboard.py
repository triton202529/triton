"""
TRITON THP-006 — Execution Gate Resilience & Certification Dashboard (read-only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from services.execution_gate_chaos_engine import CHAOS_JSON, run_chaos_engine
from services.execution_gate_resilience_certification import (
    CERTIFICATION_JSON,
    REPORT_JSON,
    load_certification,
    run_full_certification,
)
from services.execution_gate_stress_suite import STRESS_JSON, run_stress_suite

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


def _level_badge(level: str) -> str:
    return {
        "INSTITUTIONAL_GRADE": "🏛 INSTITUTIONAL GRADE",
        "PRODUCTION_READY": "✅ PRODUCTION READY",
        "REQUIRES_IMPROVEMENT": "⚠️ REQUIRES IMPROVEMENT",
        "FAILED": "❌ FAILED",
    }.get(level, level)


def page_execution_gate_resilience() -> None:
    st.title("🛡 Execution Gate Resilience Certification")
    st.info(
        "**Read-only resilience lab.** Stress testing, chaos engineering, and "
        "institutional certification. No broker calls or capital deployment."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Run Stress Suite", key="eg_run_stress"):
            with st.spinner("Running stress tests..."):
                run_stress_suite()
            st.success("Stress suite complete.")
    with col2:
        if st.button("Run Chaos Engine", key="eg_run_chaos"):
            with st.spinner("Injecting controlled faults..."):
                run_chaos_engine()
            st.success("Chaos engine complete.")
    with col3:
        if st.button("Run Full Certification", key="eg_run_cert"):
            with st.spinner("Running full certification pipeline..."):
                run_full_certification()
            st.success("Certification complete.")

    cert = load_certification() or _read_json(CERTIFICATION_JSON)
    report = _read_json(REPORT_JSON)
    stress = _read_json(STRESS_JSON)
    chaos = _read_json(CHAOS_JSON)

    score = cert.get("resilience_score", 0)
    level = cert.get("certification_level", "UNKNOWN")

    st.markdown("### Resilience Score")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resilience Score", f"{score}/100")
    c2.markdown(_level_badge(level))
    c3.metric("Accidental Allows", cert.get("accidental_authorizations", 0))
    c4.metric(
        "Paper/Live Boundary",
        "Intact" if cert.get("paper_live_boundary_intact") else "VIOLATED",
    )

    st.markdown("### Category Scores")
    cats = cert.get("category_scores") or report.get("categories") or []
    if cats:
        cat_df = pd.DataFrame(cats)
        st.dataframe(cat_df, use_container_width=True, hide_index=True)
        if "category" in cat_df.columns and "score" in cat_df.columns:
            st.bar_chart(cat_df.set_index("category")["score"])
    else:
        st.caption("Run **Full Certification** to generate category scores.")

    st.markdown("### Stress Test Results")
    stress_summary = stress.get("summary") or {}
    s1, s2, s3 = st.columns(3)
    s1.metric("Stress Total", stress_summary.get("total", 0))
    s2.metric("Stress Passed", stress_summary.get("passed", 0))
    s3.metric("Stress Failed", stress_summary.get("failed", 0))
    if stress.get("records"):
        st.dataframe(pd.DataFrame(stress["records"]), use_container_width=True, hide_index=True)

    st.markdown("### Chaos Engineering Results")
    chaos_summary = chaos.get("summary") or {}
    ch1, ch2, ch3 = st.columns(3)
    ch1.metric("Chaos Total", chaos_summary.get("total", 0))
    ch2.metric("Chaos Passed", chaos_summary.get("passed", 0))
    ch3.metric("All Fail-Safe", "Yes" if chaos_summary.get("all_fail_safe") else "No")
    if chaos.get("records"):
        st.dataframe(pd.DataFrame(chaos["records"]), use_container_width=True, hide_index=True)

    st.markdown("### Failed Scenarios & Remediation")
    failed = cert.get("failed_scenarios") or []
    if failed:
        for item in failed[:8]:
            st.markdown(f"**{item.get('scenario_name')}** ({item.get('source')})")
            st.write(item.get("failure_description"))
            st.caption(f"Remediation: {item.get('recommended_remediation')}")
    else:
        st.success("No failed scenarios in latest certification run.")

    st.markdown("### Certification Artifact")
    if cert:
        st.json(cert)
    else:
        st.caption("No certification artifact yet.")


def render() -> None:
    page_execution_gate_resilience()
