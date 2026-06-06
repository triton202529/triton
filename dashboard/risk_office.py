"""
TRITON Risk Office Dashboard — read-only operator view (Phases 7–8).

Consumes existing capital preservation / watchdog JSON and CSV artifacts only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

ESCALATION_ORDER = ["GREEN", "YELLOW", "ORANGE", "RED", "CRITICAL"]
TREND_DELTA = 3.0


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


def _parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _classify_trend(first_score: float, last_score: float) -> str:
    if not pd.notna(first_score) or not pd.notna(last_score):
        return "STABLE"
    if last_score >= first_score + TREND_DELTA:
        return "IMPROVING"
    if last_score <= first_score - TREND_DELTA:
        return "DETERIORATING"
    return "STABLE"


def _window_trend(
    df: pd.DataFrame, days: int, score_col: str = "capital_preservation_score"
) -> str:
    if df.empty or score_col not in df.columns:
        return "STABLE"
    work = df.dropna(subset=["timestamp"]).copy()
    if work.empty:
        return "STABLE"
    cutoff = work["timestamp"].max() - pd.Timedelta(days=days)
    window = work[work["timestamp"] >= cutoff]
    if len(window) < 2:
        return "STABLE"
    window = window.sort_values("timestamp")
    return _classify_trend(
        float(window[score_col].iloc[0]),
        float(window[score_col].iloc[-1]),
    )


def _prepare_cpi_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["timestamp"] = _parse_ts(out.get("timestamp", pd.Series(dtype=object)))
    if "capital_preservation_score" in out.columns:
        out["capital_preservation_score"] = pd.to_numeric(
            out["capital_preservation_score"], errors="coerce"
        )
    if "active_alerts" in out.columns:
        out["active_alerts"] = pd.to_numeric(out["active_alerts"], errors="coerce")
    return out.dropna(subset=["timestamp"]).sort_values("timestamp")


def _count_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "count"])
    work = df.copy()
    work["timestamp"] = _parse_ts(work["timestamp"])
    work = work.dropna(subset=["timestamp"])
    if work.empty:
        return pd.DataFrame(columns=["timestamp", "count"])
    counts = work.groupby("timestamp", as_index=False).size()
    counts.columns = ["timestamp", "count"]
    return counts.sort_values("timestamp")


def _escalation_rank(state: str) -> int:
    try:
        return ESCALATION_ORDER.index(str(state).upper())
    except ValueError:
        return -1


def render_risk_office_dashboard(results_dir: Path) -> None:
    """Render Risk Office tab — portfolio health, timeline, governance (read-only)."""
    results_dir = Path(results_dir)
    cpi = _load_json(results_dir / "capital_preservation_intelligence.json")
    cpe = _load_json(results_dir / "capital_preservation_escalation.json")
    alerts = _load_json(results_dir / "watchdog_alerts.json")
    cpa = _load_json(results_dir / "capital_preservation_advisory.json")
    cpd = _load_json(results_dir / "capital_preservation_decision_support.json")
    gov = _load_json(results_dir / "governance_risk_summary.json")
    status = _load_json(results_dir / "watchdog_status.json")

    st.markdown("### 🛡️ Risk Office")
    st.caption(
        "Read-only capital preservation posture from watchdog artifacts. "
        "No trading actions are initiated from this view."
    )

    generated = cpi.get("generated_at") or cpe.get("generated_at") or status.get("timestamp") or "—"
    st.caption(f"Last artifact update: **{generated}**")

    cps = cpi.get("capital_preservation_score")
    health_band = cpi.get("health_band") or "—"
    risk_trend = cpi.get("risk_trend") or "—"
    escalation_state = cpe.get("escalation_state") or cpi.get("escalation_state") or "—"
    active_alerts = alerts.get("active_alerts") or []
    incident = alerts.get("incident_intelligence") or {}
    cpa_summary = cpa.get("summary") or {}
    cpd_summary = cpd.get("summary") or {}

    # ── Row 1: core cards ──────────────────────────────────────────────
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.markdown("#### Portfolio Health")
        st.metric("Capital Preservation Score", cps if cps is not None else "—")
        st.metric("Health Band", health_band)
        st.metric("Risk Trend", risk_trend)

    with r1c2:
        st.markdown("#### Escalation")
        st.metric("Escalation State", escalation_state)
        reasons = cpe.get("escalation_reason") or []
        if reasons:
            st.markdown("**Escalation reasons**")
            for r in reasons:
                st.markdown(f"- `{r}`")
        else:
            st.caption("No active escalation reasons.")

    with r1c3:
        st.markdown("#### Governance Summary")
        st.metric("Awareness Level", gov.get("governance_awareness_label") or "—")
        st.metric("Governance Status", gov.get("governance_status") or "—")
        drivers = gov.get("governance_drivers") or gov.get("governance_summary") or []
        if drivers:
            st.markdown("**Governance drivers**")
            for d in drivers:
                st.markdown(f"- {d}")
        else:
            st.caption("No governance drivers recorded.")

    st.markdown("---")

    # ── Row 2: alert / advisory / decision / incident ───────────────────
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)

    with r2c1:
        st.markdown("#### Alert Summary")
        st.metric("Active Alerts", len(active_alerts))
        severity_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for a in active_alerts:
            if not isinstance(a, dict):
                continue
            sev = str(a.get("severity") or "UNKNOWN")
            typ = str(a.get("alert_type") or "UNKNOWN")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            type_counts[typ] = type_counts.get(typ, 0) + 1
        if type_counts:
            st.markdown("**By type**")
            for k, v in sorted(type_counts.items()):
                st.markdown(f"- {k}: {v}")
        if severity_counts:
            st.markdown("**By severity**")
            for k, v in sorted(severity_counts.items()):
                st.markdown(f"- {k}: {v}")

    with r2c2:
        st.markdown("#### Advisory")
        st.metric(
            "Advisory Count", cpa_summary.get("advisory_count", len(cpa.get("advisories") or []))
        )
        st.metric("Highest Priority", cpa_summary.get("highest_priority") or "—")
        st.metric("Top Issue", cpa_summary.get("top_issue") or "—")

    with r2c3:
        st.markdown("#### Decision Support")
        st.metric("Issue Count", cpd_summary.get("issue_count", 0))
        st.metric("Top Issue", cpd_summary.get("top_issue") or "—")
        paths: List[str] = []
        for item in cpd.get("decision_support_items") or []:
            if not isinstance(item, dict):
                continue
            for opt in item.get("available_review_options") or []:
                if opt not in paths:
                    paths.append(str(opt))
        if paths:
            st.markdown("**Review paths**")
            for p in paths[:6]:
                st.markdown(f"- {p}")
            if len(paths) > 6:
                st.caption(f"+ {len(paths) - 6} more")

    with r2c4:
        st.markdown("#### Incident Intelligence")
        st.metric("Incidents (24h)", incident.get("incident_count_24h", 0))
        st.metric("Broker Disconnects", incident.get("broker_disconnect_count", 0))
        st.metric("Last Disconnect", incident.get("last_disconnect") or "—")
        max_d = incident.get("max_disconnect_duration_minutes")
        if max_d is not None:
            st.caption(f"Max disconnect duration: {max_d} min")

    st.markdown("---")
    st.markdown("### 📈 Portfolio Health Timeline")

    cpi_hist = _prepare_cpi_history(_load_csv(results_dir / "capital_preservation_history.csv"))
    cpe_hist = _load_csv(results_dir / "capital_preservation_escalation_history.csv")
    cpa_hist = _load_csv(results_dir / "capital_preservation_advisory_history.csv")
    cpd_hist = _load_csv(results_dir / "capital_preservation_decision_history.csv")

    if not cpi_hist.empty:
        t7 = _window_trend(cpi_hist, 7)
        t30 = _window_trend(cpi_hist, 30)
        t90 = _window_trend(cpi_hist, 90)
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("7-Day Trend", t7)
        tc2.metric("30-Day Trend", t30)
        tc3.metric("90-Day Trend", t90)
        tc4.metric("Current Risk Trend", risk_trend)
    else:
        st.info(
            "No capital preservation history yet — run the watchdog loop to build timeline data."
        )

    try:
        import plotly.express as px  # type: ignore
    except ImportError:
        px = None

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if px and not cpi_hist.empty:
            fig = px.line(
                cpi_hist,
                x="timestamp",
                y="capital_preservation_score",
                title="CPS Over Time",
                markers=True,
            )
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        elif not cpi_hist.empty:
            st.line_chart(cpi_hist.set_index("timestamp")["capital_preservation_score"])

        if not cpe_hist.empty and "escalation_state" in cpe_hist.columns:
            esc = cpe_hist.copy()
            esc["timestamp"] = _parse_ts(esc["timestamp"])
            esc["esc_rank"] = esc["escalation_state"].map(_escalation_rank)
            esc = esc.dropna(subset=["timestamp"]).sort_values("timestamp")
            if px and not esc.empty:
                fig2 = px.line(
                    esc,
                    x="timestamp",
                    y="esc_rank",
                    title="Escalation State Over Time",
                    markers=True,
                )
                fig2.update_yaxes(
                    tickmode="array",
                    tickvals=list(range(len(ESCALATION_ORDER))),
                    ticktext=ESCALATION_ORDER,
                )
                fig2.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig2, use_container_width=True)

        if not cpi_hist.empty and "active_alerts" in cpi_hist.columns:
            if px:
                fig3 = px.line(
                    cpi_hist,
                    x="timestamp",
                    y="active_alerts",
                    title="Alert Count Over Time",
                    markers=True,
                )
                fig3.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig3, use_container_width=True)

    with chart_col2:
        adv_counts = _count_by_timestamp(cpa_hist)
        if px and not adv_counts.empty:
            fig4 = px.line(
                adv_counts,
                x="timestamp",
                y="count",
                title="Advisory Count Over Time",
                markers=True,
            )
            fig4.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig4, use_container_width=True)
        elif adv_counts.empty:
            st.caption("Advisory history will appear after watchdog cycles.")

        dec_counts = _count_by_timestamp(cpd_hist)
        if px and not dec_counts.empty:
            fig5 = px.line(
                dec_counts,
                x="timestamp",
                y="count",
                title="Decision Support Count Over Time",
                markers=True,
            )
            fig5.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig5, use_container_width=True)

    with st.expander("Active advisories (detail)", expanded=False):
        adv_list = cpa.get("advisories") or []
        if adv_list:
            st.dataframe(pd.DataFrame(adv_list), use_container_width=True, hide_index=True)
        else:
            st.caption("No active advisories.")

    with st.expander("Decision support items (detail)", expanded=False):
        dse_items = cpd.get("decision_support_items") or []
        if dse_items:
            st.dataframe(pd.DataFrame(dse_items), use_container_width=True, hide_index=True)
        else:
            st.caption("No decision support items.")

    with st.expander("Raw governance risk summary", expanded=False):
        st.json(gov if gov else {"status": "not generated yet"})
