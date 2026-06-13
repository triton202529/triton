"""
TRITON Phase 201 — Governance Recovery Daily Evidence Monitor dashboard.
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
        "AUTHORIZATION_READY": "🟢",
        "NEAR_YELLOW": "🟡",
        "RECOVERING": "🟠",
        "STALLED": "🔴",
        "CRITICAL": "⚫",
    }.get(band, "⚪")


def _render_executive(report: Dict[str, Any]) -> None:
    st.markdown("#### 📈 [GOVERNANCE_DAILY_EVIDENCE_REPORT]")
    ex = report.get("executive_recommendation") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPS", str(ex.get("current_cps", "—")))
    c2.metric("Daily Δ CPS", str(ex.get("daily_cps_change", "—")))
    c3.metric("Direction", str(ex.get("recovery_direction", "—")))
    c4.metric("Velocity", str(ex.get("recovery_velocity", "—")))
    forecast = ex.get("updated_authorization_forecast") or {}
    st.info(
        f"YELLOW forecast: **{forecast.get('yellow_date', '—')}** · "
        f"GREEN: **{forecast.get('green_date', '—')}**",
        icon="💡",
    )


def _render_score(report: Dict[str, Any]) -> None:
    st.markdown("#### 📊 Recovery Score")
    score = report.get("governance_recovery_score") or {}
    band = str(score.get("classification", "—"))
    st.markdown(
        f"{_band_color(band)} **{band}** — score **{score.get('governance_recovery_score', '—')}**"
    )
    factors = score.get("factors") or {}
    if factors:
        st.dataframe(
            pd.DataFrame([{"factor": k, "value": v} for k, v in factors.items()]),
            use_container_width=True,
            hide_index=True,
        )


def _render_cps_trend(results_dir: Path) -> None:
    st.markdown("#### 📉 CPS Trend")
    df = _read_csv_safe(results_dir / "governance_recovery_trend.csv")
    if df.empty:
        st.info("No trend data yet.", icon="ℹ️")
        return
    if "cps" in df.columns:
        st.line_chart(df.set_index("date")["cps"] if "date" in df.columns else df["cps"])


def _render_timelines(report: Dict[str, Any]) -> None:
    st.markdown("#### ⏱ Distance Timelines")
    ex = report.get("executive_recommendation") or {}
    c1, c2 = st.columns(2)
    c1.metric("Δ to YELLOW", str(ex.get("distance_to_yellow", "—")))
    c2.metric("Δ to GREEN", str(ex.get("distance_to_green", "—")))
    vel = report.get("recovery_velocity") or {}
    st.markdown(
        f"Updated YELLOW: **{vel.get('updated_yellow_forecast_date', '—')}** "
        f"({vel.get('updated_days_to_yellow', '—')} days) · "
        f"GREEN: **{vel.get('updated_green_forecast_date', '—')}**"
    )


def _render_concentration(report: Dict[str, Any]) -> None:
    st.markdown("#### 🎯 Concentration Reduction Progress")
    pr = (report.get("daily_evidence") or {}).get("portfolio_recovery") or {}
    st.metric("QQQ %", str(pr.get("qqq_concentration_pct", "—")))
    st.metric("Daily change", str(pr.get("qqq_concentration_change", "—")))


def _render_velocity(results_dir: Path, report: Dict[str, Any]) -> None:
    st.markdown("#### 🚀 Recovery Velocity")
    vel = report.get("recovery_velocity") or {}
    st.markdown(
        f"Direction: **{vel.get('recovery_direction')}** · Velocity: **{vel.get('recovery_velocity')}** · "
        f"CPS/day: **{vel.get('cps_change_per_day')}**"
    )
    df = _read_csv_safe(results_dir / "governance_recovery_velocity.csv")
    if not df.empty and "recovery_direction" in df.columns:
        st.dataframe(df.tail(5).iloc[::-1], use_container_width=True, hide_index=True)


def _render_momentum(report: Dict[str, Any]) -> None:
    st.markdown("#### 💨 Evidence Momentum")
    mom = report.get("evidence_momentum") or {}
    st.markdown(f"**Classification:** {mom.get('classification', '—')}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Positive**")
        for p in mom.get("positive_evidence") or []:
            st.markdown(f"- {p}")
    with c2:
        st.markdown("**Negative**")
        for n in mom.get("negative_evidence") or []:
            st.markdown(f"- {n}")


def _render_warnings(report: Dict[str, Any]) -> None:
    st.markdown("#### ⚠️ Warning Center")
    warnings = report.get("early_warnings") or []
    for w in warnings:
        st.warning(
            f"**{w.get('warning_code')}** — {w.get('root_cause')} · Action: {w.get('recommended_action')}",
            icon="⚠️",
        )


def _render_forecast(report: Dict[str, Any]) -> None:
    st.markdown("#### 🔮 Authorization Forecast")
    ex = report.get("executive_recommendation") or {}
    fc = ex.get("updated_authorization_forecast") or {}
    st.markdown(f"- YELLOW: **{fc.get('yellow_date')}** (baseline {fc.get('baseline_yellow')})")
    st.markdown(f"- GREEN: **{fc.get('green_date')}** (baseline {fc.get('baseline_green')})")
    st.markdown(f"- Evidence completion: **{ex.get('evidence_completion_pct')}%**")


def render_governance_daily_evidence_monitor(results_dir: Path) -> None:
    st.markdown("### 📈 Governance Recovery Daily Evidence Monitor")
    st.caption(
        "Continuous daily evidence tracking for governance recovery. "
        "Refresh with `python -m services.governance_daily_evidence_monitor`."
    )
    report = _read_json_safe(results_dir / "governance_daily_evidence.json")
    if not report:
        st.warning(
            "No daily evidence report. Run `python -m services.governance_daily_evidence_monitor`.",
            icon="⚠️",
        )
        return
    _render_executive(report)
    _render_score(report)
    _render_cps_trend(results_dir)
    _render_timelines(report)
    _render_concentration(report)
    _render_velocity(results_dir, report)
    _render_momentum(report)
    _render_warnings(report)
    _render_forecast(report)
