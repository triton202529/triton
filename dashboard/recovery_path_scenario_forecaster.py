"""
TRITON Phase 184 — 🧭 Recovery Path Scenario Forecasting dashboard.
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


_OUTCOME = {
    "RECOVERY_CERTAIN": "🟢 RECOVERY_CERTAIN",
    "RECOVERY_LIKELY": "🟢 RECOVERY_LIKELY",
    "RECOVERY_POSSIBLE": "🟡 RECOVERY_POSSIBLE",
    "RECOVERY_UNLIKELY": "🟠 RECOVERY_UNLIKELY",
    "RECOVERY_FAILED": "🔴 RECOVERY_FAILED",
}


def _render_matrix(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🧮 Scenario Matrix")
    df = _read_csv_safe(results_dir / "recovery_path_scenarios.csv")
    if df.empty:
        df = pd.DataFrame(report.get("scenarios") or [])
    if df.empty:
        st.info("No scenarios.", icon="ℹ️")
        return
    cols = [
        c
        for c in [
            "scenario",
            "scenario_probability",
            "yellow_probability",
            "green_probability",
            "timeline_to_yellow_days",
            "timeline_to_green_days",
            "projected_cps",
            "projected_escalation",
            "outcome_classification",
            "forecast_confidence",
        ]
        if c in df.columns
    ]
    view = df[cols].copy()
    if "outcome_classification" in view.columns:
        view["outcome_classification"] = view["outcome_classification"].map(
            lambda x: _OUTCOME.get(str(x), str(x))
        )
    st.dataframe(view, use_container_width=True, hide_index=True)


def _render_rankings(report: Dict[str, Any]) -> None:
    st.markdown("#### 🏆 Probability Rankings")
    scenarios = sorted(
        report.get("scenarios") or [],
        key=lambda s: float(s.get("scenario_probability") or 0),
        reverse=True,
    )
    for s in scenarios:
        st.markdown(
            f"- **{s.get('scenario')}** — {s.get('scenario_probability')}% scenario weight, "
            f"YELLOW {s.get('yellow_probability')}%, GREEN {s.get('green_probability')}%"
        )


def _render_dependencies(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔗 Recovery Dependencies")
    st.json(report.get("dependencies") or {})


def _render_yellow(report: Dict[str, Any]) -> None:
    st.markdown("#### 🟡 Yellow Forecast")
    fs = report.get("forecast_summary") or {}
    st.metric("Aggregate YELLOW Probability", f"{fs.get('aggregate_yellow_probability', '—')}%")
    fastest = fs.get("fastest_scenario")
    st.markdown(f"**Fastest path:** {fastest} (~{fs.get('fastest_yellow_days')} days)")


def _render_green(report: Dict[str, Any]) -> None:
    st.markdown("#### 🟢 Green Forecast")
    fs = report.get("forecast_summary") or {}
    st.metric("Aggregate GREEN Probability", f"{fs.get('aggregate_green_probability', '—')}%")
    deps = report.get("dependencies") or {}
    bito = deps.get("bito_promotion") or {}
    st.markdown(f"**BITO promotion prob 30d:** {bito.get('promotion_prob_30d')}%")
    st.caption("GREEN requires BITO OPEN_NEW promotion + extension step.")


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Recommendation")
    ex = report.get("executive_report") or {}
    st.markdown(f"**[{ex.get('report_type', 'RECOVERY_PATH_SCENARIO_REPORT')}]**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Most Likely", str(ex.get("most_likely_scenario", "—")))
    c2.metric("P(YELLOW)", f"{ex.get('probability_of_yellow', '—')}%")
    c3.metric("P(GREEN)", f"{ex.get('probability_of_green', '—')}%")
    st.markdown(f"**Recommendation:** {ex.get('executive_recommendation')}")
    fs = report.get("forecast_summary") or {}
    st.markdown(f"**Biggest risk:** {fs.get('biggest_risk')}")
    st.markdown(f"**Failure mode:** {fs.get('most_likely_failure_mode')}")


def _render_probability_matrix(results_dir: Path) -> None:
    st.markdown("#### 📊 Probability Matrix")
    mx = _read_csv_safe(results_dir / "recovery_path_probability_matrix.csv")
    if mx.empty:
        st.info("No matrix data.", icon="ℹ️")
        return
    if "scenario" in mx.columns and "outcome" in mx.columns:
        pivot = mx.pivot_table(
            index="scenario", columns="outcome", values="probability", aggfunc="first"
        )
        st.dataframe(pivot, use_container_width=True)


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification")
    cert = report.get("certification") or {}
    st.metric("Status", str(cert.get("status", "—")))
    st.success(str(cert.get("safety") or ""), icon="🔒")


def render_recovery_path_scenario_forecaster(results_dir: Path) -> None:
    st.markdown("### 🧭 Recovery Path Scenario Forecasting")
    st.caption(
        "Multi-path governance recovery scenario forecasting — analysis only. "
        "Refresh with `python -m services.recovery_path_scenario_forecaster`."
    )

    report = _read_json_safe(results_dir / "recovery_path_scenarios.json")
    if not report:
        st.info(
            "No scenario report yet. Run "
            "`python -m services.recovery_path_scenario_forecaster`.",
            icon="ℹ️",
        )
        return

    cs = report.get("current_state") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPS", str(cs.get("current_cps", "—")))
    c2.metric("Escalation", str(cs.get("current_escalation", "—")))
    c3.metric("HD", str(cs.get("hd_status", "—")))
    c4.metric("BITO", str(cs.get("bito_status", "—")))
    st.divider()

    section = st.radio(
        "Section",
        [
            "🧮 Scenario Matrix",
            "🏆 Probability Rankings",
            "🔗 Recovery Dependencies",
            "🟡 Yellow Forecast",
            "🟢 Green Forecast",
            "📊 Probability Matrix",
            "📋 Executive Recommendation",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="rpsf_section",
    )
    if section == "🧮 Scenario Matrix":
        _render_matrix(results_dir, report)
    elif section == "🏆 Probability Rankings":
        _render_rankings(report)
    elif section == "🔗 Recovery Dependencies":
        _render_dependencies(report)
    elif section == "🟡 Yellow Forecast":
        _render_yellow(report)
    elif section == "🟢 Green Forecast":
        _render_green(report)
    elif section == "📊 Probability Matrix":
        _render_probability_matrix(results_dir)
    elif section == "📋 Executive Recommendation":
        _render_executive(report)
    else:
        _render_certification(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
