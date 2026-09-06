"""
operator_reports.py
Phase UIR-002 — Operator Console Reports page (read-only presentation).

All financial values come from services.reconciled_portfolio_reporting.
No broker calls, no runtime controls, no artifact writes.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Mapping, Optional

import streamlit as st

from services.reconciled_portfolio_reporting import (
    DEFAULT_PERIOD,
    PERIOD_TYPES,
    REPORTING_TZ_NAME,
    ReportRequest,
    build_reconciled_report,
    format_money,
    tone_for_report_status,
)


def _metric_caption(metric: Mapping[str, Any]) -> str:
    avail = str(metric.get("availability") or "UNAVAILABLE")
    if metric.get("value") is None:
        reason = metric.get("reason") or "No authoritative evidence"
        return f"{avail} — {reason}"
    method = metric.get("method") or "n/a"
    return f"{avail} · {method}"


def _render_metric_card(label: str, metric: Mapping[str, Any]) -> None:
    from dashboard import operator_ui as oui

    display = (
        format_money(metric)
        if metric.get("currency") == "USD"
        or metric.get("unit")
        in (
            None,
            "USD_equity_change",
        )
        else ("Unavailable" if metric.get("value") is None else str(metric.get("value")))
    )
    if metric.get("unit") == "count":
        display = "Unavailable" if metric.get("value") is None else str(metric.get("value"))
    if metric.get("unit") == "percent":
        display = "Unavailable" if metric.get("value") is None else f"{metric.get('value')}%"

    tone = "muted"
    avail = str(metric.get("availability") or "")
    if avail == "AVAILABLE":
        tone = "ok"
    elif avail == "PARTIAL":
        tone = "warn"
    elif avail == "CONFLICTED":
        tone = "crit"

    oui.render_compact_panel(
        label,
        [
            {"label": "Value", "value": display},
            {"label": "Availability", "value": avail or "UNAVAILABLE"},
            {"label": "Evidence", "value": _metric_caption(metric)},
        ],
    )
    st.markdown(
        oui.status_badge(avail or "UNAVAILABLE", tone=tone),
        unsafe_allow_html=True,
    )


def render_operator_reports_page() -> None:
    """Full Reports page — local evidence only."""
    from dashboard import operator_ui as oui

    oui.inject_operator_theme()

    # Controls first (compact).
    hdr_l, hdr_r = st.columns([6.2, 1.0])
    with hdr_l:
        oui.render_command_header(
            title="Reports",
            context=(
                f"Read-only reconciled portfolio reporting · timezone {REPORTING_TZ_NAME} · "
                "unsupported values stay Unavailable"
            ),
            market_state="n/a",
            evidence_ts="local evidence",
        )
    with hdr_r:
        st.write("")
        if st.button("Refresh", key="cc_reports_refresh", use_container_width=True):
            # Reload local evidence only — no broker call.
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

    st.caption(f"Reporting timezone: **{REPORTING_TZ_NAME}**")

    period = st.radio(
        "Reporting period",
        list(PERIOD_TYPES),
        index=list(PERIOD_TYPES).index(DEFAULT_PERIOD),
        horizontal=True,
        key="cc_reports_period",
    )

    anchor: Optional[date] = None
    custom_start: Optional[date] = None
    custom_end: Optional[date] = None

    if period == "Custom":
        c1, c2 = st.columns(2)
        with c1:
            custom_start = st.date_input("Start date", key="cc_reports_custom_start")
        with c2:
            custom_end = st.date_input("End date", key="cc_reports_custom_end")
    elif period == "Daily":
        use_default = st.checkbox(
            "Use latest supported reporting date (weekend/holiday safe)",
            value=True,
            key="cc_reports_daily_default",
        )
        if not use_default:
            anchor = st.date_input("Reporting date", key="cc_reports_daily_anchor")
    elif period == "Weekly":
        anchor = st.date_input("Week containing", key="cc_reports_week_anchor")
    elif period == "Monthly":
        anchor = st.date_input("Month containing", key="cc_reports_month_anchor")
    elif period == "Quarterly":
        anchor = st.date_input("Quarter containing", key="cc_reports_quarter_anchor")
    elif period == "Half-year":
        anchor = st.date_input("Half-year containing", key="cc_reports_half_anchor")
    elif period == "Yearly":
        anchor = st.date_input("Year containing", key="cc_reports_year_anchor")

    request = ReportRequest(
        period_type=period,
        anchor_date=anchor,
        custom_start=custom_start,
        custom_end=custom_end,
        as_of=datetime.now().astimezone(),
    )
    report = build_reconciled_report(request)

    st.caption(f"Report as_of: **{report.get('as_of') or 'UNKNOWN'}**")
    try:
        from services.paper_order_ledger import proceeds_summary

        sale = proceeds_summary()
        st.caption(
            f"Paper SELL filled proceeds (ledger): **${sale.get('filled_proceeds') or 0:,.2f}** "
            f"from {sale.get('filled_count') or 0} terminal fill(s). "
            f"Fees {sale.get('fees_commissions')}. "
            f"Realized P/L {sale.get('realized_pl_availability')} — "
            f"{sale.get('realized_pl_reason')}"
        )
    except Exception:
        pass
    if report.get("label"):
        st.caption(
            f"Selected: **{report.get('period_type')}** · {report.get('label')} · "
            f"{report.get('start_date')} → {report.get('end_date')}"
        )

    # Status strip
    freshness = report.get("freshness") or {}
    oui.render_safety_strip(
        [
            {
                "label": "Report availability",
                "value": report.get("availability") or "UNAVAILABLE",
                "tone": tone_for_report_status(report.get("availability")),
            },
            {
                "label": "Period completeness",
                "value": report.get("completeness") or "UNAVAILABLE",
                "tone": tone_for_report_status(report.get("completeness")),
            },
            {
                "label": "Reconciliation",
                "value": (report.get("reconciliation") or {}).get("status") or "UNAVAILABLE",
                "tone": tone_for_report_status((report.get("reconciliation") or {}).get("status")),
            },
            {
                "label": "Data freshness",
                "value": freshness.get("status") or "UNKNOWN",
                "tone": tone_for_report_status(freshness.get("status")),
            },
        ]
    )
    if freshness.get("reason"):
        st.caption(f"Freshness aggregate: {freshness.get('reason')}")
    src_fresh = freshness.get("sources") or []
    if src_fresh:
        with st.expander("Source-level freshness provenance", expanded=False):
            rows = []
            for s in src_fresh:
                status = str(s.get("status") or "UNKNOWN")
                post = bool(s.get("post_as_of") or s.get("excluded_from_as_of_freshness"))
                rows.append(
                    {
                        "source": s.get("source"),
                        "status": status,
                        "post_as_of": post,
                        "green_fresh": status == "FRESH" and not post,
                        "source_timestamp": s.get("source_timestamp"),
                        "age_minutes": s.get("age_minutes"),
                        "minutes_after_as_of": s.get("minutes_after_as_of"),
                        "date_only": bool(s.get("date_only")),
                        "mtime_advisory_only": s.get("filesystem_mtime_advisory"),
                        "reason": s.get("reason"),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption(
                "Post-as-of / future-dated / date-only / UNKNOWN sources are never shown as "
                "authoritative green FRESH. Filesystem mtime is advisory only."
            )

    if not report.get("ok"):
        st.warning(report.get("error") or "Report request invalid")
        missing = (report.get("evidence") or {}).get("missing_requirements") or []
        if missing:
            st.caption("Missing requirements: " + "; ".join(str(m) for m in missing))
        return

    if report.get("availability") == "UNAVAILABLE":
        st.info(
            "No authoritative period data is available for this selection. "
            "No financial values are displayed."
        )
        cov = report.get("effective_coverage") or {}
        st.caption(f"Latest known source date: {cov.get('latest_known_source_date') or 'n/a'}")
        missing = (report.get("evidence") or {}).get("missing_requirements") or []
        if missing:
            st.caption("Missing requirements: " + "; ".join(str(m) for m in missing))

    metrics = report.get("metrics") or {}
    cov = report.get("effective_coverage") or {}
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        _render_metric_card("Opening equity", metrics.get("opening_equity") or {})
        st.caption(
            "Authoritative session-open equity is Unavailable — "
            "portfolio_history is date-only / mixed-producer."
        )
    with m2:
        _render_metric_card("Closing / as-of equity", metrics.get("closing_equity") or {})
        st.caption(
            "Authoritative session-close equity is Unavailable — "
            "no certified exchange-close boundary."
        )
    with m3:
        _render_metric_card("Net equity change", metrics.get("net_equity_change") or {})
        st.caption(
            "Session net equity change withheld without authoritative open/close boundaries."
        )
    with m4:
        _render_metric_card("Period P/L", metrics.get("period_pl") or {})

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        _render_metric_card(
            "Previous recorded equity", metrics.get("previous_recorded_equity") or {}
        )
        st.caption(
            f"Observation date: {cov.get('previous_recorded_date') or 'n/a'} · "
            "not a certified session open."
        )
    with r2:
        _render_metric_card("Latest recorded equity", metrics.get("latest_recorded_equity") or {})
        st.caption(
            f"Observation date: {cov.get('latest_recorded_date') or 'n/a'} · "
            "not a certified exchange close."
        )
    with r3:
        _render_metric_card(
            "Change between recorded observations",
            metrics.get("recorded_observation_change") or {},
        )
        st.caption("Not Period P/L, Daily P/L, investment return, or session equity change.")
    with r4:
        st.caption(cov.get("portfolio_history_disclosure") or "")

    m5, m6, m7, m8 = st.columns(4)
    with m5:
        _render_metric_card("Realized P/L", metrics.get("realized_pl") or {})
        st.caption("Realized P/L withheld unless a governed accounting authority is proven.")
    with m6:
        _render_metric_card("Unrealized P/L", metrics.get("unrealized_pl") or {})
    with m7:
        _render_metric_card("Net external cash flows", metrics.get("net_external_cash_flows") or {})
    with m8:
        _render_metric_card("Position count", metrics.get("position_count") or {})

    tab_summary, tab_holdings, tab_activity, tab_recon = st.tabs(
        ["Summary", "Holdings", "Activity", "Reconciliation & Evidence"]
    )

    with tab_summary:
        st.markdown("**What this period covers**")
        st.write(
            {
                "period": report.get("period_type"),
                "start": report.get("start_date"),
                "end": report.get("end_date"),
                "is_market_session": cov.get("is_market_session"),
                "previous_recorded_date": cov.get("previous_recorded_date"),
                "latest_recorded_date": cov.get("latest_recorded_date"),
                "non_session_history_observations": cov.get("non_session_history_observations"),
                "history_dates_in_range": cov.get("history_dates"),
                "portfolio_history_disclosure": cov.get("portfolio_history_disclosure"),
            }
        )
        chart = report.get("chart") or {}
        if chart.get("available"):
            try:
                import pandas as pd

                df = pd.DataFrame(chart.get("points") or [])
                if not df.empty:
                    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
                    st.line_chart(df.set_index("date")["total_value"], height=220)
            except Exception:
                st.caption("Chart render failed — values remain in evidence JSON.")
        else:
            st.caption(chart.get("reason") or "Chart unavailable")

        c_orders, c_trades = st.columns(2)
        with c_orders:
            _render_metric_card("Order count", metrics.get("order_count") or {})
        with c_trades:
            _render_metric_card(
                "Trade count (fills/partials only)", metrics.get("trade_count") or {}
            )

        with st.expander("Metric provenance", expanded=False):
            st.json(metrics)

    with tab_holdings:
        holdings = report.get("holdings") or {}
        st.markdown(f"**{holdings.get('section_title') or 'Latest position snapshot'}**")
        disclosure = holdings.get("disclosure") or holdings.get("reason") or ""
        if disclosure:
            st.info(disclosure)
        st.caption(
            f"snapshot_ts={holdings.get('snapshot_ts') or 'UNKNOWN'} · "
            f"snapshot_date={holdings.get('snapshot_date') or 'UNKNOWN'} · "
            f"selected_period_end={holdings.get('selected_period_end') or 'UNKNOWN'} · "
            f"relationship={holdings.get('period_relationship') or 'UNKNOWN'}"
        )
        rows = holdings.get("rows") or []
        if not rows:
            st.info("No holding rows available for this period.")
        else:
            # Shared disclosure above replaces long per-row Notes in the visible grid.
            cols = [c for c in (holdings.get("columns") or []) if c != "Notes"]
            display_rows = []
            for r in rows:
                item = {}
                for c in cols:
                    val = r.get(c)
                    # Preserve signed unrealized strings (negatives remain distinct).
                    item[c] = "Unavailable" if val is None else val
                display_rows.append(item)
            # HTML table avoids Glide-grid tab-width collapse and keeps all labels visible.
            st.table(display_rows)
            st.caption(
                "Per-row Notes omitted from the grid — see shared snapshot disclosure above. "
                "Beginning quantity and net quantity change remain Unavailable without "
                "authoritative opening and ending holdings."
            )

    with tab_activity:
        activity = report.get("activity") or {}
        if activity.get("no_activity_message"):
            st.info(activity["no_activity_message"])
        if activity.get("note"):
            st.caption(activity["note"])
        st.caption(
            "Coverage: "
            f"**{activity.get('coverage_kind') or 'UNKNOWN'}** · "
            "counts_are_complete_period_totals="
            f"**{activity.get('counts_are_complete_period_totals')}**"
        )
        st.write(activity.get("counts") or {})

        st.markdown("**Broker execution activity (recorded under partial coverage)**")
        broker_rows = activity.get("broker_orders") or []
        if broker_rows:
            st.dataframe(broker_rows, use_container_width=True, hide_index=True)
        else:
            st.caption(
                activity.get("empty_state_broker")
                or (
                    "No broker order rows were found in the available local evidence "
                    "for this period."
                )
            )

        st.markdown("**TRITON decision activity**")
        decisions = activity.get("triton_decisions") or []
        if decisions:
            st.dataframe(decisions, use_container_width=True, hide_index=True)
        else:
            st.caption(
                activity.get("empty_state_triton")
                or (
                    "No TRITON decision-audit rows were found in the available local "
                    "evidence for this period."
                )
            )

        st.caption(
            "Unlinked broker activity is labeled UNLINKED_BROKER_ACTIVITY and is not "
            "described as a TRITON action. Canceled/rejected orders are not trades."
        )

    with tab_recon:
        recon = report.get("reconciliation") or {}
        st.caption(f"Reconciliation status: **{recon.get('status') or 'UNAVAILABLE'}**")
        disc = recon.get("discrepancies") or []
        if disc:
            st.error(f"{len(disc)} evidence-backed discrepancy(ies)")
            for d in disc:
                st.write(d)
        checks = recon.get("checks") or []
        with st.expander("Reconciliation details", expanded=False):
            st.json(checks)
        with st.expander("Source coverage", expanded=False):
            st.json(report.get("effective_coverage") or {})
            st.write((report.get("evidence") or {}).get("missing_requirements") or [])
        with st.expander("Raw source references", expanded=False):
            st.write((report.get("evidence") or {}).get("sources") or [])
        with st.expander("Technical JSON", expanded=False):
            st.json(
                {
                    "availability": report.get("availability"),
                    "completeness": report.get("completeness"),
                    "freshness": report.get("freshness"),
                    "writes_performed": report.get("writes_performed"),
                    "network_calls_performed": report.get("network_calls_performed"),
                    "notes": (report.get("evidence") or {}).get("notes"),
                }
            )
        with st.expander("Calculation descriptions", expanded=False):
            st.markdown(
                """
- **Opening / closing equity**: Unavailable without certified session-open/close boundaries.
- **Recorded observations**: previous/latest `portfolio_history.total_value` (date-only).
- **Recorded observation change**: not Period P/L, return, or session equity change.
- **Period P/L / return %**: withheld unless an approved reconciled authority exists.
- **Realized P/L**: Unavailable — `trade_outcomes` local FIFO is not a governed authority.
- **Order/fill counts**: recorded activity inside the limited `recent_orders` window only.
- **Unrealized P/L**: sum of `positions_snapshot.unrealized_pl` when present.
                """
            )
