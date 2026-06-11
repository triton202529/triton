"""
TRITON Phase 185 — 🎯 Governance Outcome Prediction Engine dashboard.
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


_CONF = {
    "VERY_HIGH": "🟢 VERY_HIGH",
    "HIGH": "🟢 HIGH",
    "MODERATE": "🟡 MODERATE",
    "LOW": "🔴 LOW",
}
_STATE = {
    "RED": "🔴 RED",
    "ORANGE": "🟠 ORANGE",
    "YELLOW": "🟡 YELLOW",
    "GREEN": "🟢 GREEN",
}


def _render_current(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Current Governance State")
    cs = report.get("current_state") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPS", str(cs.get("cps", "—")))
    c2.metric("Escalation", str(cs.get("escalation", "—")))
    c3.metric("Governance", str(cs.get("governance_status", "—")))
    c4.metric("QQQ %", str(cs.get("qqq_concentration_pct", "—")))


def _render_predictions(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔮 Future Governance Predictions")
    preds = pd.DataFrame(report.get("predictions") or [])
    if preds.empty:
        st.info("No predictions.", icon="ℹ️")
        return
    view = preds.copy()
    if "most_likely_state" in view.columns:
        view["most_likely_state"] = view["most_likely_state"].map(
            lambda x: _STATE.get(str(x), str(x))
        )
    st.dataframe(view, use_container_width=True, hide_index=True)
    if "forecast_horizon" in preds.columns and "prob_green" in preds.columns:
        chart = preds.set_index("forecast_horizon")[
            ["prob_red", "prob_orange", "prob_yellow", "prob_green"]
        ]
        st.area_chart(chart)


def _render_matrix(results_dir: Path) -> None:
    st.markdown("#### 🧮 State Probability Matrix")
    mx = _read_csv_safe(results_dir / "governance_state_probability_matrix.csv")
    if mx.empty:
        st.info("No matrix.", icon="ℹ️")
        return
    if "forecast_horizon" in mx.columns and "state" in mx.columns:
        pivot = mx.pivot_table(
            index="forecast_horizon", columns="state", values="probability", aggfunc="first"
        )
        st.dataframe(pivot, use_container_width=True)


def _render_transitions(report: Dict[str, Any]) -> None:
    st.markdown("#### ↔️ Transition Forecasts")
    tf = report.get("transition_forecasts") or {}
    for key, val in tf.items():
        label = key.replace("_", " → ").replace("to ", "→ ")
        with st.expander(f"**{label}**"):
            st.json(val)


def _render_resilience(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Resilience Analysis")
    ra = report.get("resilience_analysis") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Governance Stability", str(ra.get("governance_stability", "—")))
    c2.metric("Recovery Stability", str(ra.get("recovery_stability", "—")))
    c3.metric("Forecast Resilience", str(ra.get("forecast_resilience", "—")))
    st.json(ra)


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Recommendation")
    ex = report.get("executive_report") or {}
    st.markdown(f"**[{ex.get('report_type', 'GOVERNANCE_OUTCOME_PREDICTION_REPORT')}]**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("30d State", _STATE.get(str(ex.get("most_likely_future_state_30d")), "—"))
    c2.metric("90d State", _STATE.get(str(ex.get("most_likely_future_state_90d")), "—"))
    c3.metric("P(YELLOW)", f"{ex.get('probability_of_yellow', '—')}%")
    c4.metric("P(GREEN)", f"{ex.get('probability_of_green', '—')}%")
    st.markdown(f"**Transition path:** {ex.get('most_likely_transition_path')}")
    st.markdown(f"**Recommendation:** {ex.get('executive_recommendation')}")


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification")
    cert = report.get("certification") or {}
    st.metric("Status", str(cert.get("status", "—")))
    st.success(str(cert.get("safety") or ""), icon="🔒")


def render_governance_outcome_prediction_engine(results_dir: Path) -> None:
    st.markdown("### 🎯 Governance Outcome Prediction Engine")
    st.caption(
        "Consolidated governance outcome forecasting from Phases 178–184 — analysis only. "
        "Refresh with `python -m services.governance_outcome_prediction_engine`."
    )

    report = _read_json_safe(results_dir / "governance_outcome_predictions.json")
    if not report:
        st.info(
            "No prediction report yet. Run "
            "`python -m services.governance_outcome_prediction_engine`.",
            icon="ℹ️",
        )
        return

    _render_current(report)
    st.divider()

    section = st.radio(
        "Section",
        [
            "🔮 Future Predictions",
            "🧮 State Probability Matrix",
            "↔️ Transition Forecasts",
            "🛡️ Resilience Analysis",
            "📋 Executive Recommendation",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="gope_section",
    )
    if section == "🔮 Future Predictions":
        _render_predictions(report)
    elif section == "🧮 State Probability Matrix":
        _render_matrix(results_dir)
    elif section == "↔️ Transition Forecasts":
        _render_transitions(report)
    elif section == "🛡️ Resilience Analysis":
        _render_resilience(report)
    elif section == "📋 Executive Recommendation":
        _render_executive(report)
    else:
        _render_certification(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
