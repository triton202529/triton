"""Isolated Performance page. Presentation only. No ledger writes or broker calls."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import streamlit as st

from dashboard import operator_ui as oui
from dashboard.operator_table import render_operator_table
from dashboard.performance_presentation import (
    CHART_CAPTION,
    CHART_TITLE,
    FAIL_CLOSED_TITLE,
    GROSS_PROVISIONAL_LABEL,
    KIND_CERTIFIED,
    KIND_OPENING,
    MORE_METRICS_NOTE,
    NET_PL_NOTE,
    PAGE_HEADING,
    PAGE_SUPPORTING_COPY,
    SECOND_CLOSE_NOTE,
    STATUS_AVAILABLE,
    STATUS_BUILDING,
    STATUS_CONFLICT,
    UNAVAILABLE_LABEL,
    PerformanceDataUnavailable,
    load_performance_view,
    overview_performance_summary,
)
from dashboard.preview_data import expected_accounting_ledger_sha256, isolation_active
from dashboard.preview_data import preview_banner_text

LINE_COLORS = {
    "positive": "#39C98A",
    "negative": "#EF6461",
    "neutral": "#95A2B2",
}

CHIP_TONES = {
    KIND_OPENING: "info",
    KIND_CERTIFIED: "ok",
    STATUS_BUILDING: "warn",
    STATUS_AVAILABLE: "ok",
    UNAVAILABLE_LABEL: "muted",
    STATUS_CONFLICT: "crit",
    GROSS_PROVISIONAL_LABEL: "warn",
    "Pending fee reconciliation": "warn",
    "Evidence conflict": "crit",
}


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def _root():
    from dashboard.preview_data import isolation_active, preview_data_root

    repo = _repo_root()
    return preview_data_root(repo) if isolation_active(repo) else repo


def _isolation_banner() -> None:
    if isolation_active(_repo_root()):
        st.markdown(
            f'<div class="triton-iso-banner">{preview_banner_text(_repo_root())}</div>',
            unsafe_allow_html=True,
        )


def _badge(text: str, *, tone: str = "muted") -> str:
    chip_tone = CHIP_TONES.get(text, tone if tone in CHIP_TONES.values() or tone else "muted")
    if tone in {"info", "ok", "warn", "crit", "muted"}:
        chip_tone = tone
    return f'<span class="triton-status-chip {chip_tone}">{oui.escape(text)}</span>'


def _metric_cards(items: Sequence[Mapping[str, Any]]) -> None:
    parts = []
    for item in items:
        value = str(item.get("value") or UNAVAILABLE_LABEL)
        tone = "muted" if value == UNAVAILABLE_LABEL else str(item.get("tone") or "info")
        parts.append(
            f'<div class="triton-perf-card {tone}">'
            f'<div class="triton-perf-label">{oui.escape(item.get("label"))}</div>'
            f'<div class="triton-perf-value">{oui.escape(value)}</div>'
            "</div>"
        )
    st.markdown(f'<div class="triton-perf-grid">{"".join(parts)}</div>', unsafe_allow_html=True)


def _equity_figure(view: Mapping[str, Any]) -> Any:
    import plotly.graph_objects as go

    points = list(view.get("checkpoints") or [])
    fig = go.Figure()
    xs = [p.get("axis_label") or p.get("kind") for p in points]
    ys = []
    for p in points:
        try:
            ys.append(float(p.get("equity")))
        except (TypeError, ValueError):
            ys.append(None)
    color = LINE_COLORS.get(str(view.get("chart_tone") or "neutral"), LINE_COLORS["neutral"])
    hover = []
    for point, y in zip(points, ys):
        ytxt = f"${y:,.2f}" if y is not None else UNAVAILABLE_LABEL
        when = point.get("timestamp_et") or point.get("timestamp")
        hover.append(f"{point.get('kind')}<br>{when}<br>Equity: {ytxt}")
    if xs and any(y is not None for y in ys):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line={"color": color, "width": 2.4, "shape": "linear"},
                marker={"size": 10, "color": color, "line": {"width": 2, "color": "#0B0F14"}},
                hovertext=hover,
                hoverinfo="text",
                connectgaps=False,
                name="Verified equity",
            )
        )
        valid = [y for y in ys if y is not None]
        lo, hi = min(valid), max(valid)
        pad = max((hi - lo) * 0.12, 8.0)
        fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_layout(
        title=CHART_TITLE,
        margin={"l": 16, "r": 12, "t": 36, "b": 16},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E7ECF3", "family": "IBM Plex Sans, Segoe UI, sans-serif"},
        hovermode="closest",
        showlegend=False,
        autosize=True,
        height=340,
        xaxis={
            "title": {"text": ""},
            "gridcolor": "rgba(38,49,61,0.65)",
            "zeroline": False,
            "type": "category",
            "categoryorder": "array",
            "categoryarray": xs,
            "tickmode": "array",
            "tickvals": xs,
            "ticktext": xs,
        },
        yaxis={
            "title": {"text": ""},
            "gridcolor": "rgba(38,49,61,0.65)",
            "zeroline": False,
            "tickprefix": "$",
            "tickformat": ",.0f",
        },
    )
    return fig


def _load_view(root=None) -> Dict[str, Any]:
    repo = _repo_root()
    data_root = root or _root()
    expected = None
    if isolation_active(repo):
        expected = expected_accounting_ledger_sha256(repo)
    return load_performance_view(root=data_root, expected_sha256=expected)


def render_overview_performance_summary(root=None) -> None:
    try:
        view = _load_view(root)
    except PerformanceDataUnavailable:
        st.markdown("#### Performance")
        st.caption(FAIL_CLOSED_TITLE)
        return
    summary = overview_performance_summary(view)
    st.markdown("#### Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest certified equity", summary["latest_certified_equity"])
    c2.metric("Certified closes", summary["certified_close_count"])
    c3.metric("Status", summary["status"])
    st.caption(summary["guidance"])


def render_performance_page(root=None) -> None:
    oui.inject_operator_theme()
    _isolation_banner()
    oui.render_page_intro(PAGE_HEADING, PAGE_SUPPORTING_COPY)
    try:
        view = _load_view(root)
    except PerformanceDataUnavailable:
        oui.render_empty_state(
            FAIL_CLOSED_TITLE,
            "The copied accounting ledger is missing, truncated, duplicated, or does not match the frozen hash.",
        )
        return

    chips = " ".join(
        _badge(str(item.get("text")), tone=str(item.get("tone") or "muted"))
        for item in view.get("header_badges") or []
    )
    st.markdown(f'<div class="triton-chip-row">{chips}</div>', unsafe_allow_html=True)

    _metric_cards(
        [
            {"label": "Latest equity", "value": view.get("latest_equity"), "tone": "info"},
            {"label": "Cash", "value": view.get("cash"), "tone": "info"},
            {"label": "Market value", "value": view.get("market_value"), "tone": "info"},
            {"label": "Unrealized P/L", "value": view.get("unrealized_pl"), "tone": "info"},
            {
                "label": "Certified closes",
                "value": str(view.get("certified_close_count")),
                "tone": "ok",
            },
            {
                "label": "Accounting status",
                "value": view.get("accounting_status"),
                "tone": "warn" if view.get("accounting_status") == STATUS_BUILDING else "ok",
            },
        ]
    )
    if view.get("building_note"):
        st.caption(view["building_note"])
    st.caption("The opening-checkpoint-to-close difference is not an investment return.")

    try:
        st.plotly_chart(
            _equity_figure(view),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )
    except Exception:
        st.caption("Interactive chart unavailable — verified checkpoint values remain below.")
        rows = [
            {
                "Checkpoint": p.get("kind"),
                "Timestamp": p.get("timestamp_et") or p.get("timestamp"),
                "Equity": money_safe(p.get("equity")),
            }
            for p in view.get("checkpoints") or []
        ]
        if rows:
            render_operator_table(rows)
    st.caption(CHART_CAPTION)

    st.markdown("#### Verified metrics")
    _metric_cards(
        [
            {"label": "Time-weighted return", "value": view.get("time_weighted_return")},
            {"label": "Maximum drawdown", "value": view.get("maximum_drawdown")},
            {
                "label": view.get("spy_comparison_label") or "SPY return",
                "value": view.get("spy_comparison"),
            },
            {
                "label": view.get("qqq_comparison_label") or "QQQ return",
                "value": view.get("qqq_comparison"),
            },
        ]
    )
    if view.get("comparison_period"):
        st.caption(f"SPY and QQQ use the identical certified dates ({view['comparison_period']}).")
    with st.expander("More performance metrics", expanded=False):
        _metric_cards(
            [
                {"label": "Realized P/L", "value": view.get("net_realized_pl")},
                {"label": "Win rate", "value": view.get("win_rate")},
                {"label": "Average winner", "value": view.get("average_winner")},
                {"label": "Average loser", "value": view.get("average_loser")},
                {"label": "Profit factor", "value": view.get("profit_factor")},
                {"label": "Average market exposure", "value": view.get("average_market_exposure")},
                {
                    "label": "Exposure-adjusted return",
                    "value": view.get("exposure_adjusted_return"),
                },
                {"label": "Slippage", "value": view.get("decision_slippage")},
                {"label": "Fees", "value": view.get("fees")},
            ]
        )
        st.caption(view.get("grouped_metrics_note") or MORE_METRICS_NOTE)
        if view.get("exposure_adjusted_note"):
            st.caption(f"Exposure-adjusted return: {view['exposure_adjusted_note']}")
    if view.get("submit_to_fill_slippage") != UNAVAILABLE_LABEL:
        st.caption(f"Submit-to-fill slippage: {view.get('submit_to_fill_slippage')}")
    st.caption(f"Decision slippage: {view.get('decision_slippage')}")

    activity = list(view.get("activity") or [])
    if activity:
        st.markdown("#### Post-cutover activity")
        shown = []
        for row in activity:
            shown.append(
                {
                    "Symbol": row.get("symbol"),
                    "Side": row.get("side"),
                    "Quantity": row.get("quantity"),
                    "Fill price": row.get("fill_price"),
                    "Proceeds": row.get("proceeds"),
                    "Position": row.get("position"),
                    "Gross provisional": row.get("gross_result"),
                    "Result label": row.get("gross_label"),
                    "Net realized P/L": row.get("net_realized_pl"),
                }
            )
        render_operator_table(shown)
        st.caption("Proceeds are not profit.")
        if any(row.get("gross_label") == GROSS_PROVISIONAL_LABEL for row in activity):
            st.markdown(
                f'<span class="triton-status-chip warn">{oui.escape(GROSS_PROVISIONAL_LABEL)}</span>',
                unsafe_allow_html=True,
            )
            st.caption(NET_PL_NOTE)

    st.markdown("#### Holdings at latest close")
    holdings = [
        {
            "Symbol": row.get("symbol"),
            "Quantity": row.get("quantity"),
            "Market value": row.get("market_value"),
            "Average basis": row.get("average_basis"),
            "Unrealized gain/loss": row.get("unrealized"),
            "Portfolio weight": row.get("weight"),
        }
        for row in view.get("holdings") or []
    ]
    if holdings:
        render_operator_table(holdings, signed_columns=("Unrealized gain/loss",))
    else:
        st.caption("No certified holdings.")

    st.markdown("#### Benchmarks")
    benches = [
        {
            "Symbol": row.get("symbol"),
            "Session date": row.get("session_date"),
            "Adjusted close": row.get("adjusted_close"),
            "Methodology": row.get("methodology"),
            "Comparison": row.get("comparison"),
        }
        for row in view.get("benchmarks") or []
    ]
    if benches:
        render_operator_table(benches)
    else:
        st.caption("No benchmark evidence.")

    st.markdown("#### Cash flows and income")
    flows = view.get("cash_flows") or {}
    render_operator_table(
        [
            {
                "Type": "External flow",
                "Amount": flows.get("external_flow"),
                "Note": "Certified period",
            },
            {
                "Type": "Dividends",
                "Amount": (
                    UNAVAILABLE_LABEL
                    if (view.get("dividends") or [{}])[0].get("none")
                    else (view.get("dividends") or [{}])[0].get("amount")
                ),
                "Note": "Separate from external contributions",
            },
            {
                "Type": "Fees",
                "Amount": view.get("fees"),
                "Note": "Separate from external contributions",
            },
        ]
    )

    st.markdown("#### Accounting evidence")
    st.markdown(
        " ".join(
            [
                f'<span class="triton-status-chip info">{oui.escape(KIND_OPENING)}</span>',
                f'<span class="triton-status-chip ok">{oui.escape(KIND_CERTIFIED)}</span>',
                f'<span class="triton-status-chip warn">{oui.escape(STATUS_BUILDING)}</span>',
                f'<span class="triton-status-chip ok">{oui.escape(STATUS_AVAILABLE)}</span>',
                f'<span class="triton-status-chip muted">{oui.escape(UNAVAILABLE_LABEL)}</span>',
                f'<span class="triton-status-chip warn">{oui.escape(GROSS_PROVISIONAL_LABEL)}</span>',
                '<span class="triton-status-chip warn">Pending fee reconciliation</span>',
                f'<span class="triton-status-chip crit">{oui.escape(STATUS_CONFLICT)}</span>',
            ]
        ),
        unsafe_allow_html=True,
    )
    if view.get("earliest_first_return_date"):
        st.caption(
            "Earliest possible one-period return: "
            f"{view['earliest_first_return_date']} after a second certified close."
        )

    with st.expander("Technical details", expanded=False):
        st.json(view.get("technical") or {})


def money_safe(value: Any) -> str:
    from dashboard.display_labels import money_text

    if value in (None, "", "UNAVAILABLE"):
        return UNAVAILABLE_LABEL
    return money_text(value)
