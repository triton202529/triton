"""
TRITON Phase 178 — Concentration Recovery Simulation dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.pnl_ledger import _num, _plotly_bar


def _read_json_safe(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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
    "NO_IMPROVEMENT": "⚪ NO_IMPROVEMENT",
    "PARTIAL_RECOVERY": "🟡 PARTIAL_RECOVERY",
    "YELLOW_RECOVERY": "🟠 YELLOW_RECOVERY",
    "GREEN_RECOVERY": "🟢 GREEN_RECOVERY",
}


def _render_current_state(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Current State")
    cs = report.get("current_state") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current CPS", str(cs.get("current_cps", "—")))
    c2.metric("Escalation", str(cs.get("current_escalation", "—")))
    c3.metric("Governance", str(cs.get("current_governance_status", "—")))
    c4.metric("Largest Weight %", str(cs.get("largest_concentration_pct", "—")))
    st.caption(f"Holdings: {cs.get('holdings') or {}}")
    st.caption(f"OPEN_NEW: {', '.join(cs.get('open_new_opportunities') or []) or 'none'}")


def _render_scenario_comparison(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 Scenario Comparison")
    df = _read_csv_safe(results_dir / "concentration_recovery_scenarios.csv")
    if df.empty:
        df = pd.DataFrame(report.get("scenario_results") or [])
    if df.empty:
        st.info("No scenario data.", icon="ℹ️")
        return
    df = df.sort_values("recovery_score", ascending=False)
    if "classification" in df.columns:
        df["classification"] = df["classification"].map(lambda x: _CLASS.get(str(x), str(x)))
    st.dataframe(df, use_container_width=True, hide_index=True)
    if "recovery_score" in df.columns and "scenario" in df.columns:
        chart_df = df.sort_values("recovery_score")
        _plotly_bar(chart_df, "scenario", "recovery_score", "Recovery Score by Scenario")


def _render_recovery_paths(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛤️ Recovery Paths")
    paths = report.get("executive_paths") or {}
    labels = [
        ("Best Recovery Path", "BEST_RECOVERY_PATH"),
        ("Fastest Path To Yellow", "FASTEST_PATH_TO_YELLOW"),
        ("Fastest Path To Green", "FASTEST_PATH_TO_GREEN"),
        ("Lowest Risk Path", "LOWEST_RISK_RECOVERY_PATH"),
        ("Highest Probability Path", "HIGHEST_PROBABILITY_RECOVERY_PATH"),
    ]
    for title, key in labels:
        p = paths.get(key) or {}
        if not p:
            continue
        with st.expander(
            f"**{title}** — {p.get('scenario', '—')}", expanded=key == "BEST_RECOVERY_PATH"
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Projected CPS", str(p.get("projected_cps", "—")))
            c2.metric("Escalation", str(p.get("escalation_level", "—")))
            c3.metric(
                "Classification",
                _CLASS.get(str(p.get("classification")), str(p.get("classification"))),
            )
            st.json(p.get("weights") or {})


def _render_thresholds(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Threshold Discovery")
    td = report.get("threshold_discovery") or {}
    c1, c2 = st.columns(2)
    c1.metric("Min HD % for YELLOW", str(td.get("minimum_hd_allocation_for_yellow_pct", "—")))
    c2.metric("Min XLK % for YELLOW", str(td.get("minimum_xlk_allocation_for_yellow_pct", "—")))
    st.markdown(f"**Combined HD+XLK for YELLOW:** `{td.get('minimum_combined_hd_xlk_for_yellow')}`")
    st.markdown(f"**Combined for GREEN:** `{td.get('minimum_combined_for_green')}`")
    st.info(td.get("estimated_path_to_clear_governance") or "—", icon="ℹ️")


def _render_forecasting(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔮 Forecasting Foundation")
    ff = report.get("forecasting_foundation") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Recovery Probability", str(ff.get("recovery_probability", "—")))
    c2.metric("Probability Score", f"{_num(ff.get('recovery_probability_score'))}%")
    c3.metric("Forecast Confidence", f"{_num(ff.get('forecast_confidence'))}%")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Current State**")
        st.json(ff.get("current_state") or {})
    with col_b:
        st.markdown("**Projected State**")
        st.json(ff.get("projected_state") or {})
    st.markdown("**Recovery Delta**")
    st.json(ff.get("recovery_delta") or {})


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification")
    cert = report.get("certification") or {}
    st.metric("Status", str(cert.get("status", "—")))
    st.success(str(cert.get("safety") or ""), icon="🔒")


def render_concentration_recovery_simulator(results_dir: Path) -> None:
    st.markdown("### 📊 Concentration Recovery Simulation")
    st.caption(
        "Governance recovery forecasting under diversification scenarios — analysis only. "
        "Refresh with `python -m services.concentration_recovery_simulator`."
    )

    report = _read_json_safe(results_dir / "concentration_recovery_simulation.json")
    if not report:
        st.info(
            "No simulation report yet. Run `python -m services.concentration_recovery_simulator`.",
            icon="ℹ️",
        )
        return

    _render_current_state(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "📈 Scenario Comparison",
            "🛤️ Recovery Paths",
            "🎯 Threshold Discovery",
            "🔮 Forecasting Foundation",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="crs_section",
    )
    if section == "📈 Scenario Comparison":
        _render_scenario_comparison(results_dir, report)
    elif section == "🛤️ Recovery Paths":
        _render_recovery_paths(report)
    elif section == "🎯 Threshold Discovery":
        _render_thresholds(report)
    elif section == "🔮 Forecasting Foundation":
        _render_forecasting(report)
    else:
        _render_certification(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
