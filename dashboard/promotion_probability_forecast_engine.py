"""
TRITON Phase 183 — 🔮 Promotion Probability Forecast dashboard.
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


_CLASS = {
    "VERY_LIKELY": "🟢 VERY_LIKELY",
    "LIKELY": "🟢 LIKELY",
    "POSSIBLE": "🟡 POSSIBLE",
    "UNLIKELY": "🟠 UNLIKELY",
    "VERY_UNLIKELY": "🔴 VERY_UNLIKELY",
}


def _render_candidates(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Candidate Forecasts")
    rows = report.get("candidate_forecasts") or []
    if not rows:
        st.info("No forecasts.", icon="ℹ️")
        return
    df = pd.DataFrame(rows)
    show = [
        "ticker",
        "current_status",
        "confidence",
        "edge_score",
        "confidence_trend",
        "promotion_probability_30d",
        "promotion_classification_30d",
        "expected_promotion_date",
        "recovery_impact",
    ]
    cols = [c for c in show if c in df.columns]
    view = df[cols].copy()
    if "promotion_classification_30d" in view.columns:
        view["promotion_classification_30d"] = view["promotion_classification_30d"].map(
            lambda x: _CLASS.get(str(x), str(x))
        )
    st.dataframe(view, use_container_width=True, hide_index=True)


def _render_rankings(report: Dict[str, Any]) -> None:
    st.markdown("#### 🏆 Promotion Rankings")
    ranks = pd.DataFrame(report.get("promotion_rankings") or [])
    if ranks.empty:
        st.info("No rankings.", icon="ℹ️")
        return
    if "classification" in ranks.columns:
        ranks["classification"] = ranks["classification"].map(lambda x: _CLASS.get(str(x), str(x)))
    st.dataframe(ranks, use_container_width=True, hide_index=True)


def _render_dependencies(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔗 Recovery Dependencies")
    st.json(report.get("promotion_dependencies") or {})
    st.markdown("**Recovery Impact**")
    st.json(report.get("recovery_impact") or {})


def _render_bito(report: Dict[str, Any]) -> None:
    st.markdown("#### ₿ BITO Forecast")
    ex = report.get("executive_report") or {}
    bf = ex.get("bito_forecast") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("30d Probability", f"{bf.get('probability_30d', '—')}%")
    c2.metric("60d Probability", f"{bf.get('probability_60d', '—')}%")
    cls = str(bf.get("classification", "—"))
    c3.metric("Classification", _CLASS.get(cls, cls))
    st.markdown(f"**Expected promotion date:** {bf.get('expected_date') or '—'}")
    bito = next(
        (c for c in (report.get("candidate_forecasts") or []) if c.get("ticker") == "BITO"),
        {},
    )
    if bito.get("horizon_forecasts"):
        hf = pd.DataFrame([{"horizon": k, **v} for k, v in bito["horizon_forecasts"].items()])
        if "promotion_probability" in hf.columns:
            st.line_chart(hf.set_index("horizon")[["promotion_probability"]])


def _render_timeline_impact(report: Dict[str, Any]) -> None:
    st.markdown("#### 🗓️ Timeline Impact")
    ti = report.get("timeline_impact") or {}
    c1, c2 = st.columns(2)
    c1.metric("Days To YELLOW", str(ti.get("days_to_yellow", "—")))
    c2.metric("Days To GREEN (baseline)", str(ti.get("days_to_green_baseline", "—")))
    st.json(ti)


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📋 Executive Recommendation")
    ex = report.get("executive_report") or {}
    st.markdown(f"**[{ex.get('report_type', 'PROMOTION_PROBABILITY_FORECAST_REPORT')}]**")
    st.markdown(f"**Next promotion candidate:** `{ex.get('next_promotion_candidate')}`")
    st.markdown(f"**Recommendation:** {ex.get('executive_recommendation')}")
    rows = pd.DataFrame(ex.get("candidates") or [])
    if not rows.empty:
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_history(results_dir: Path) -> None:
    st.markdown("#### 📜 Forecast History")
    hist = _read_csv_safe(results_dir / "promotion_probability_history.csv")
    if hist.empty:
        st.caption("History builds as the engine runs over time.")
        return
    st.dataframe(hist.tail(20), use_container_width=True, hide_index=True)


def _render_certification(report: Dict[str, Any]) -> None:
    st.markdown("#### 🛡️ Certification")
    cert = report.get("certification") or {}
    st.metric("Status", str(cert.get("status", "—")))
    st.success(str(cert.get("safety") or ""), icon="🔒")


def render_promotion_probability_forecast_engine(results_dir: Path) -> None:
    st.markdown("### 🔮 Promotion Probability Forecast")
    st.caption(
        "Lifecycle promotion probability forecasting — analysis only. "
        "Refresh with `python -m services.promotion_probability_forecast_engine`."
    )

    report = _read_json_safe(results_dir / "promotion_probability_forecast.json")
    if not report:
        st.info(
            "No forecast report yet. Run "
            "`python -m services.promotion_probability_forecast_engine`.",
            icon="ℹ️",
        )
        return

    section = st.radio(
        "Section",
        [
            "🎯 Candidate Forecasts",
            "🏆 Promotion Rankings",
            "🔗 Recovery Dependencies",
            "₿ BITO Forecast",
            "🗓️ Timeline Impact",
            "📋 Executive Recommendation",
            "📜 Forecast History",
            "🛡️ Certification",
        ],
        horizontal=True,
        key="ppfe_section",
    )
    if section == "🎯 Candidate Forecasts":
        _render_candidates(report)
    elif section == "🏆 Promotion Rankings":
        _render_rankings(report)
    elif section == "🔗 Recovery Dependencies":
        _render_dependencies(report)
    elif section == "₿ BITO Forecast":
        _render_bito(report)
    elif section == "🗓️ Timeline Impact":
        _render_timeline_impact(report)
    elif section == "📋 Executive Recommendation":
        _render_executive(report)
    elif section == "📜 Forecast History":
        _render_history(results_dir)
    else:
        _render_certification(report)

    st.divider()
    st.caption(str(report.get("disclaimer") or ""))
