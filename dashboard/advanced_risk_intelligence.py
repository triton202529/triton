"""
TRITON Advanced Risk Intelligence dashboards — Phases 10–12 (read-only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def render_defensive_simulation_lab(results_dir: Path) -> None:
    st.markdown("### 🧪 Defensive Simulation Lab")
    st.caption(
        "Counterfactual defensive control simulations. **No trades or portfolio changes are executed.**"
    )

    sim = _load_json(results_dir / "defensive_simulation_results.json")
    if not sim:
        st.warning("No simulation results yet. Run `python -m services.risk_watchdog` first.")
        return

    baseline = sim.get("baseline") or {}
    st.caption(f"Generated: **{sim.get('generated_at', '—')}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline CPS", baseline.get("capital_preservation_score", "—"))
    c2.metric("Baseline Escalation", baseline.get("escalation_state", "—"))
    c3.metric("Largest Concentration %", baseline.get("largest_concentration_pct", "—"))

    simulations: List[Dict[str, Any]] = sim.get("simulations") or []
    if not simulations:
        st.info("No simulations computed.")
        return

    rows = []
    for s in simulations:
        rows.append(
            {
                "Simulation": s.get("simulation_name"),
                "Type": s.get("simulation_type"),
                "Return Δ": s.get("portfolio_return_delta"),
                "Drawdown Δ": s.get("max_drawdown_delta"),
                "Volatility Δ": s.get("volatility_delta"),
                "Concentration Δ": s.get("concentration_delta"),
                "Risk Score Δ": s.get("risk_score_delta"),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    try:
        import plotly.express as px  # type: ignore

        fig = px.bar(
            df,
            x="Simulation",
            y="Risk Score Δ",
            title="Simulated Risk Score Impact",
            color="Type",
        )
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(df.set_index("Simulation")["Risk Score Δ"])

    with st.expander("Simulation details", expanded=False):
        for s in simulations:
            st.markdown(f"**{s.get('simulation_name')}**")
            st.json(s)


def render_predictive_risk_section(results_dir: Path, *, embedded: bool = False) -> None:
    if not embedded:
        st.markdown("### 🔮 Predictive Risk Intelligence")
    else:
        st.markdown("#### 🔮 Predictive Risk Outlook")
    st.caption("Early-warning estimates from historical watchdog trends — not trade signals.")

    pred = _load_json(results_dir / "predictive_risk_intelligence.json")
    if not pred:
        st.warning("Predictive intelligence not generated yet.")
        return

    st.caption(f"Generated: **{pred.get('generated_at', '—')}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Direction", pred.get("risk_direction", "—"))
    c2.metric("Risk Momentum", pred.get("risk_momentum", "—"))
    c3.metric("Forecast Confidence", f"{pred.get('forecast_confidence', '—')}%")
    days = pred.get("estimated_days_to_threshold")
    c4.metric("Est. Days to Threshold", days if days is not None else "—")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Alert Velocity (24h)", pred.get("alert_velocity", "—"))
    c6.metric("Alert Acceleration", pred.get("alert_acceleration", "—"))
    forecast = pred.get("portfolio_health_forecast") or {}
    esc = pred.get("escalation_forecast") or {}
    c7.metric("Projected CPS", forecast.get("projected_cps", "—"))
    c8.metric("Projected Escalation", esc.get("projected_escalation", "—"))

    hist = _load_csv(results_dir / "capital_preservation_history.csv")
    if not hist.empty and "capital_preservation_score" in hist.columns:
        hist = hist.copy()
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce", utc=True)
        hist["capital_preservation_score"] = pd.to_numeric(
            hist["capital_preservation_score"], errors="coerce"
        )
        hist = hist.dropna(subset=["timestamp"]).sort_values("timestamp")
        try:
            import plotly.express as px  # type: ignore

            fig = px.line(
                hist,
                x="timestamp",
                y="capital_preservation_score",
                title="CPS History + Forecast Context",
                markers=True,
            )
            proj = forecast.get("projected_cps")
            if proj is not None and not hist.empty:
                fig.add_scatter(
                    x=[hist["timestamp"].iloc[-1]],
                    y=[proj],
                    mode="markers+text",
                    name="Projected",
                    text=["Projected"],
                    textposition="top center",
                )
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart(hist.set_index("timestamp")["capital_preservation_score"])

    with st.expander("Full predictive payload", expanded=False):
        st.json(pred)


def render_executive_risk_command_center(results_dir: Path) -> None:
    st.markdown("### 🏛 Executive Risk Command Center")
    st.caption("Boardroom-level read-only risk briefing for leadership and governance committees.")

    summary = _load_json(results_dir / "executive_risk_summary.json")
    report = _load_json(results_dir / "executive_risk_report.json")
    if not summary:
        st.warning("Executive summary not generated yet. Run the watchdog first.")
        return

    ex = summary.get("executive_summary") or {}
    st.caption(f"Generated: **{summary.get('generated_at', '—')}**")

    st.markdown("#### Executive Summary")
    st.markdown(
        f"""
| Field | Value |
|-------|-------|
| **Portfolio Health** | {ex.get('portfolio_health', '—')} |
| **Capital Preservation Score** | {ex.get('capital_preservation_score', '—')} |
| **Escalation State** | {ex.get('escalation_state', '—')} |
| **Governance Awareness** | {ex.get('governance_awareness', '—')} |
| **Risk Direction** | {ex.get('risk_direction', '—')} |
| **Projected Escalation** | {ex.get('projected_escalation', '—')} |
"""
    )

    top_risks = ex.get("top_risks") or summary.get("top_risks") or []
    if top_risks:
        st.markdown("**Top Risks**")
        for r in top_risks:
            st.markdown(f"- {r}")

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("CPS", summary.get("capital_preservation_score", "—"))
    r1c2.metric("Escalation", summary.get("escalation_state", "—"))
    r1c3.metric("Governance", summary.get("governance_awareness_label", "—"))
    r1c4.metric("Risk Trend", summary.get("risk_trend", "—"))

    st.markdown("---")
    render_predictive_risk_section(results_dir, embedded=True)

    st.markdown("---")
    st.markdown("#### Incident Intelligence")
    incident = summary.get("incident_intelligence") or {}
    i1, i2, i3 = st.columns(3)
    i1.metric("Incidents (24h)", incident.get("incident_count_24h", 0))
    i2.metric("Broker Disconnects", incident.get("broker_disconnect_count", 0))
    i3.metric("Last Disconnect", incident.get("last_disconnect") or "—")

    st.markdown("#### Strategic Watchlist")
    watch = summary.get("strategic_watchlist") or []
    if watch:
        st.dataframe(pd.DataFrame(watch), use_container_width=True, hide_index=True)
    else:
        st.caption("No items on strategic watchlist.")

    st.markdown("#### Portfolio Health Timeline Snapshot")
    hist = _load_csv(results_dir / "capital_preservation_history.csv")
    if not hist.empty:
        st.dataframe(hist.tail(10), use_container_width=True, hide_index=True)
    else:
        st.caption("No history available.")

    with st.expander("Full executive report", expanded=False):
        st.json(report if report else summary)
