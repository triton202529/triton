"""
TRITON THP-005 — Execution Gate Simulation & Replay Dashboard (read-only).

Flight recorder and simulator for execution authorization validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from services.execution_gate_failure_investigator import (
    FAILURE_JSON,
    run_failure_investigation,
)
from services.execution_gate_replay_engine import (
    DRIFT_REPORT_JSON,
    REPLAY_CSV,
    REPLAY_JSON,
    run_replay,
)
from services.execution_gate_simulator import (
    SIMULATION_CSV,
    SIMULATION_JSON,
    run_simulations,
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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def page_execution_gate_simulation() -> None:
    st.title("🔬 Execution Gate Simulation Laboratory")
    st.info(
        "**Read-only validation lab.** Replay historical decisions, run authorization "
        "scenarios, and investigate failures. No broker calls or capital deployment."
    )

    col_run1, col_run2, col_run3 = st.columns(3)
    with col_run1:
        if st.button("Run Replay Engine", key="eg_run_replay"):
            with st.spinner("Replaying historical decisions..."):
                run_replay()
            st.success("Replay complete.")
    with col_run2:
        if st.button("Run Scenario Simulator", key="eg_run_sim"):
            with st.spinner("Running simulation catalog..."):
                run_simulations()
            st.success("Simulation complete.")
    with col_run3:
        if st.button("Run Failure Investigation", key="eg_run_inv"):
            with st.spinner("Analyzing failures..."):
                run_failure_investigation()
            st.success("Investigation complete.")

    replay_doc = _read_json(REPLAY_JSON)
    replay_summary = replay_doc.get("summary") or {}
    sim_doc = _read_json(SIMULATION_JSON)
    sim_summary = sim_doc.get("summary") or {}
    failure_doc = _read_json(FAILURE_JSON)
    failure_summary = failure_doc.get("summary") or {}
    drift_doc = _read_json(DRIFT_REPORT_JSON)

    st.markdown("### Replay Summary")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Replayed", replay_summary.get("total", 0))
    r2.metric("Consistent", replay_summary.get("consistent", 0))
    r3.metric("Drift Detected", replay_summary.get("drift_detected", 0))
    r4.metric("Requires Review", replay_summary.get("requires_review", 0))

    replay_df = _read_csv(REPLAY_CSV)
    if replay_df.empty and replay_doc.get("records"):
        replay_df = pd.DataFrame(replay_doc["records"])
    if not replay_df.empty:
        st.dataframe(replay_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No replay results yet. Click **Run Replay Engine**.")

    st.markdown("### Simulation Results")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Simulations", sim_summary.get("total", 0))
    s2.metric("Passed", sim_summary.get("passed", 0))
    s3.metric("Failed", sim_summary.get("failed", 0))
    s4.metric("Unexpected Allows", sim_summary.get("unexpected_authorizations", 0))
    s5.metric("Unexpected Blocks", sim_summary.get("unexpected_blocks", 0))

    sim_df = _read_csv(SIMULATION_CSV)
    if sim_df.empty and sim_doc.get("records"):
        sim_df = pd.DataFrame(sim_doc["records"])
    if not sim_df.empty:
        st.dataframe(sim_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No simulation results yet. Click **Run Scenario Simulator**.")

    st.markdown("### Failure Analysis")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total Analyses", failure_summary.get("total_analyses", 0))
    f2.metric("Risk Failures", failure_summary.get("risk_related_failures", 0))
    f3.metric("Governance Failures", failure_summary.get("governance_related_failures", 0))
    f4.metric("Top Reasons", len(failure_summary.get("top_rejection_reasons") or {}))

    failures = failure_doc.get("records") or []
    if failures:
        for item in failures[:5]:
            st.markdown(f"**{item.get('execution_path')}** — `{item.get('decision')}`")
            st.write(item.get("human_explanation"))
        if len(failures) > 5:
            st.caption(f"... and {len(failures) - 5} more analyses in {FAILURE_JSON.name}")
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
    else:
        st.caption("No failure analyses yet.")

    st.markdown("### Drift Monitoring")
    if drift_doc:
        st.metric("Drift Events", drift_doc.get("drift_count", 0))
        st.metric("Review Queue", drift_doc.get("review_count", 0))
        drift_by_path = drift_doc.get("drift_by_path") or {}
        if drift_by_path:
            st.bar_chart(pd.Series(drift_by_path))
        observations = drift_doc.get("policy_observations") or []
        for obs in observations:
            st.caption(f"• {obs}")
    else:
        st.caption("No drift report yet. Run replay to generate drift monitoring.")


def render() -> None:
    page_execution_gate_simulation()
