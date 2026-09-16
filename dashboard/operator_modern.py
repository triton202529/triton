"""
UI-A modern operator pages.

Presentation only. Reuses existing status loaders, ledger reducers, and
authority-control callbacks. Does not write artifacts, contact a broker,
or arm/start TRITON by itself.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import streamlit as st

from dashboard import operator_ui as oui
from dashboard.display_labels import (
    display_label,
    display_status,
    is_raw_identifier,
    money_text,
    quantity_text,
    signed_text,
    technical_payload,
)
from dashboard.operator_table import render_operator_table
from dashboard.portfolio_chart import (
    available_periods,
    build_plotly_figure,
    chart_state,
    filter_period,
    normalize_history_rows,
    period_change,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _root() -> Path:
    from dashboard.preview_data import isolation_active, preview_data_root

    repo = _repo_root()
    return preview_data_root(repo) if isolation_active(repo) else repo


def _isolation_banner() -> None:
    from dashboard.preview_data import isolation_active, preview_banner_text

    if isolation_active(_repo_root()):
        st.markdown(
            f'<div class="triton-iso-banner">{preview_banner_text(_repo_root())}</div>',
            unsafe_allow_html=True,
        )


def _load_lifecycle_actions(root: Optional[Path] = None) -> Dict[str, str]:
    path = (root or _root()) / "data" / "results" / "signal_lifecycle.csv"
    if not path.is_file():
        return {}
    out: Dict[str, str] = {}
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                sym = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
                action = (
                    row.get("lifecycle_action") or row.get("stance") or row.get("decision_action")
                )
                if sym and action:
                    out[sym] = str(action)
    except Exception:
        return {}
    return out


def _human_loop_display(card: Mapping[str, Any]) -> str:
    """Operator-facing loop card text. Underlying UIR labels stay unchanged."""
    label = display_status(card.get("label") or "Unknown")
    count_label = str(card.get("count_label") or "").strip()
    if not count_label:
        return label
    if count_label.upper() == "COUNT UNAVAILABLE":
        return f"{label} · count unavailable"
    pretty = (
        count_label.replace(" / ", " of ")
        .replace("INSTANCES", "instances")
        .replace("INSTANCE", "instance")
    )
    return f"{label} · {pretty}"


def _buy_authority_display(status: Mapping[str, Any]) -> str:
    paper = status.get("paper_arm") or {}
    return display_status(paper.get("display_status") or paper.get("status") or "Unknown")


def _sell_authority_state() -> Dict[str, str]:
    try:
        from dashboard.paper_sell_arm_controls import (
            COLOR_AMBER,
            COLOR_BLUE,
            COLOR_GREEN,
            COLOR_RED,
            map_sell_arm_visual_status,
        )
        from services.paper_sell_arm_console import build_presentation

        pres = build_presentation()
        visual = map_sell_arm_visual_status(pres)
        color = str(visual.get("color") or COLOR_AMBER)
        tone = {
            COLOR_GREEN: "ok",
            COLOR_AMBER: "warn",
            COLOR_BLUE: "info",
            COLOR_RED: "crit",
        }.get(color, "muted")
        raw_label = str(visual.get("label") or "SELL — Unknown")
        if raw_label == "SELL completed" or visual.get("code") == "COMPLETED":
            raw_label = "SELL completed"
        elif "UNAVAILABLE" in raw_label.upper():
            raw_label = "SELL — Unavailable"
        elif "SELL ARM" in raw_label.upper():
            raw_label = raw_label.replace("SELL ARM", "SELL").replace("  ", " ")
        return {"label": raw_label, "tone": tone}
    except Exception:
        return {"label": "SELL — Unavailable", "tone": "muted"}


def _market_session_card() -> Dict[str, Any]:
    from dashboard.market_session_card import market_session_card
    from dashboard.preview_data import isolation_active, preview_as_of_datetime

    now = preview_as_of_datetime() if isolation_active(_repo_root()) else None
    return market_session_card(now=now)


def _certified_totals():
    from dashboard.performance_presentation import try_certified_page_totals
    from dashboard.preview_data import expected_accounting_ledger_sha256, isolation_active

    expected = None
    if isolation_active(_repo_root()):
        expected = expected_accounting_ledger_sha256(_repo_root())
    return try_certified_page_totals(root=_root(), expected_sha256=expected)


def _freshness_view(status: Mapping[str, Any]) -> Dict[str, Any]:
    from dashboard.freshness_presentation import freshness_for_render
    from dashboard.preview_data import isolation_active

    return dict(freshness_for_render(status, isolated=isolation_active(_repo_root())))


def render_status_bar(status: Mapping[str, Any]) -> None:
    """Safety states — blocked/unknown never styled healthy."""
    live = status.get("live_execution") or {}
    paper = status.get("paper_arm") or {}
    live_perm = status.get("live_permission") or {}
    freshness = status.get("data_freshness") or {}
    cont = status.get("continuous") or {}
    wd = status.get("watchdog") or {}
    mode = "Paper Only"
    if live_perm.get("live_execution_permitted") is True or live.get("blocked") is False:
        mode = "Live mode indicated"
    paper_exec = display_status(paper.get("display_status") or "Unknown")
    sell_auth = _sell_authority_state()
    market = _market_session_card()
    fresh = _freshness_view(status)
    live_value = "Blocked" if live.get("blocked") else "Not blocked"
    pipeline = oui.present_pipeline_heartbeat_label(
        freshness_class=freshness.get("class"),
        process_count=cont.get("process_count"),
        discovery_available=cont.get("process_discovery_available", True),
        lock_held=bool(cont.get("lock_exists")),
        expected_after_hours_stop=bool((status.get("runtime_expectation") or {}).get("expected")),
        age_minutes=freshness.get("heartbeat_age_minutes"),
        evidence_path=freshness.get("evidence_path") or "heartbeat.json",
    )
    from services.dashboard_status_sync import load_runtime_heartbeat

    watchdog_hb = oui.present_watchdog_heartbeat_label(
        load_runtime_heartbeat(),
        process_count=wd.get("process_count"),
        discovery_available=wd.get("process_discovery_available", True),
        lock_held=bool(wd.get("lock_exists")),
        expected_after_hours_stop=bool((status.get("runtime_expectation") or {}).get("expected")),
    )
    cont_card = oui.loop_process_card_state(
        process_running=cont.get("process_running"),
        process_count=cont.get("process_count"),
        discovery_available=cont.get("process_discovery_available", True),
        heartbeat_class=pipeline["label"],
        heartbeat_age_minutes=freshness.get("heartbeat_age_minutes"),
    )
    wd_card = oui.loop_process_card_state(
        process_running=wd.get("process_running"),
        process_count=wd.get("process_count"),
        discovery_available=wd.get("process_discovery_available", True),
        heartbeat_class=watchdog_hb["label"],
        heartbeat_age_minutes=watchdog_hb.get("age_minutes"),
    )
    oui.render_safety_strip(
        [
            {
                "label": "Trading mode",
                "value": mode,
                "tone": "crit" if mode != "Paper Only" else "info",
            },
            {
                "label": "Market session",
                "value": market.get("value") or "Unavailable",
                "tone": market.get("tone") or "crit",
                "note": market.get("note"),
            },
            {
                "label": "Paper execution",
                "value": paper_exec,
                "tone": oui.tone_for_status(paper_exec),
            },
            {
                "label": "Live execution",
                "value": live_value,
                "tone": oui.tone_for_live_execution(
                    "BLOCKED" if live.get("blocked") else "NOT BLOCKED"
                ),
                "note": (
                    "Required safety lock" if live.get("blocked") else "Investigate immediately"
                ),
            },
            {
                "label": "Continuous loop",
                "value": _human_loop_display(cont_card),
                "tone": cont_card["tone"],
                "note": cont_card.get("note"),
            },
            {
                "label": "Watchdog",
                "value": _human_loop_display(wd_card),
                "tone": wd_card["tone"],
                "note": wd_card.get("note"),
            },
            {
                "label": "BUY authority",
                "value": _buy_authority_display(status),
                "tone": oui.tone_for_status(_buy_authority_display(status)),
            },
            {
                "label": "SELL authority",
                "value": sell_auth["label"],
                "tone": sell_auth["tone"],
            },
            {
                "label": "Data freshness",
                "value": fresh["value"],
                "tone": fresh["tone"],
                "note": fresh.get("note"),
            },
        ]
    )
    if not live.get("blocked"):
        st.error(
            "Live execution does not appear blocked. There is no enable control. Investigate now."
        )


def render_authority_and_runtime_controls(status: Mapping[str, Any]) -> None:
    """BUY/SELL then protected Start/Stop/Restart — same callbacks as production."""
    from dashboard.operator_console import _render_start_controls
    from dashboard.paper_buy_add_arm_controls import render_paper_buy_add_arm_controls
    from dashboard.paper_sell_arm_controls import render_paper_sell_arm_controls

    oui.render_page_intro(
        "Authority and runtime", "Arm paper BUY or SELL, then start or stop TRITON."
    )
    buy, sell = st.columns(2, gap="medium")
    with buy:
        with st.container(border=True):
            render_paper_buy_add_arm_controls(compact=True)
    with sell:
        with st.container(border=True):
            render_paper_sell_arm_controls(compact=True)
    _render_start_controls(status_snapshot=dict(status))


def _portfolio_metrics(
    summary: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    *,
    certified: Optional[Mapping[str, Any]] = None,
) -> None:
    from dashboard.performance_presentation import SECOND_CLOSE_NOTE, UNAVAILABLE_LABEL

    if certified and certified.get("available"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio value", certified.get("portfolio_value"))
        c2.metric("Period change", UNAVAILABLE_LABEL)
        c3.metric("Cash", certified.get("cash"))
        c4, c5 = st.columns(2)
        c4.metric("Invested value", certified.get("market_value"))
        c5.metric("Unrealized P/L", certified.get("unrealized_pl"))
        if certified.get("period_change_note"):
            st.caption(certified["period_change_note"])
        return
    change = period_change(history_rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio value", money_text(summary.get("total_portfolio_value")))
    delta = change.get("change")
    pct = change.get("change_pct")
    delta_label = signed_text(delta) if delta is not None else "Unavailable"
    if pct is not None and delta is not None:
        delta_label = f"{signed_text(delta)} ({pct:+.2f}%)"
    c2.metric("Period change", delta_label)
    c3.metric("Cash", money_text(summary.get("cash")))
    c4, c5 = st.columns(2)
    c4.metric("Invested value", money_text(summary.get("market_value")))
    c5.metric("Unrealized P/L", signed_text(summary.get("unrealized_pnl")))
    from dashboard.preview_data import isolation_active

    realized = summary.get("realized_pnl")
    if isolation_active(_root()) or realized is None:
        st.caption(
            "Realized P/L: **Unavailable** — sale proceeds are not a cost-basis realized P/L figure."
        )
    else:
        st.caption(f"Realized P/L: **{signed_text(realized)}**")


def _holdings_table(summary: Mapping[str, Any]) -> None:
    holdings = list(summary.get("holdings") or [])
    if not holdings:
        oui.render_empty_state("No holdings", "No evidenced position rows for this snapshot.")
        return
    actions = _load_lifecycle_actions()
    total_mv = 0.0
    for row in holdings:
        try:
            total_mv += float(row.get("market_value") or row.get("value") or 0)
        except (TypeError, ValueError):
            pass
    display = []
    for row in holdings:
        sym = str(row.get("symbol") or row.get("ticker") or "").upper()
        try:
            mv = float(row.get("market_value") or row.get("value") or 0)
        except (TypeError, ValueError):
            mv = None
        weight = f"{(mv / total_mv * 100):.1f}%" if mv is not None and total_mv else "Unavailable"
        pl = (
            row.get("unrealized_pl")
            if row.get("unrealized_pl") is not None
            else row.get("unrealized_pnl")
        )
        item = {
            "Symbol": sym,
            "Quantity": quantity_text(
                row.get("qty") if row.get("qty") is not None else row.get("quantity")
            ),
            "Average cost": money_text(row.get("avg_entry_price")),
            "Current price": money_text(row.get("current_price")),
            "Market value": money_text(mv),
            "Gain / loss": signed_text(pl),
            "Weight": weight,
            "Lifecycle action": display_status(actions.get(sym) or "Unavailable"),
        }
        if row.get("company_name") or row.get("name"):
            item["Company"] = row.get("company_name") or row.get("name")
        display.append(item)
    render_operator_table(
        display,
        signed_columns=("Gain / loss",),
        caption="Gain / loss uses + or − in addition to color. Company names are shown only when the snapshot includes them.",
    )


def render_overview_page() -> None:
    from services.operator_console_status import build_operator_status, build_portfolio_summary
    from services.operator_activity_feed import collect_activity_events
    from services.paper_order_ledger import merge_activity_payload

    oui.inject_operator_theme()
    _isolation_banner()
    oui.render_page_intro(
        "Overview",
        "Is TRITON running, is live blocked, and what needs attention?",
        kicker="Operator console",
    )
    status = build_operator_status(root=_root())
    summary = build_portfolio_summary(root=_root())
    render_status_bar(status)

    certified = _certified_totals()
    hist = normalize_history_rows(_history_dicts(summary))
    latest = hist[-2:] if len(hist) >= 2 else hist
    _portfolio_metrics(summary, latest, certified=certified)

    from dashboard.performance_page import render_overview_performance_summary

    render_overview_performance_summary()

    render_authority_and_runtime_controls(status)

    attention = list(status.get("attention") or [])
    st.markdown("#### Needs attention")
    if attention:
        for item in attention[:6]:
            oui.render_attention_item(item)
        if len(attention) > 6:
            st.caption(f"{len(attention) - 6} more items are on Operations.")
    else:
        st.info("Nothing requires action right now.")

    st.markdown("#### Latest cycle")
    cycle = status.get("cycle") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Last completed", oui.format_timestamp(cycle.get("last_completed_at")))
    next_expected = cycle.get("next_expected_at")
    from dashboard.time_presentation import parse_aware_datetime

    if parse_aware_datetime(next_expected):
        next_shown = oui.format_timestamp(next_expected)
    elif next_expected:
        next_shown = display_label(next_expected)
    else:
        next_shown = "Unavailable"
    c2.metric("Next expected", next_shown)
    c3.metric("Overdue", "Yes" if cycle.get("overdue") else "No")

    st.markdown("#### Recent activity")
    try:
        payload = merge_activity_payload(collect_activity_events(root=_root(), since_hours=24.0))
        events = list(payload.get("events") or [])[:8]
    except Exception:
        events = []
    if events:
        _render_activity_timeline(events, limit=8, show_evidence=False)
    else:
        st.caption("No evidenced events in the last 24 hours.")

    with st.expander("Technical details", expanded=False):
        st.json(
            {
                "overall_status": status.get("overall_status"),
                "generated_at": status.get("generated_at"),
                "evidence_paths": status.get("evidence_paths"),
                "cycle": cycle,
                "live_execution": status.get("live_execution"),
                "paper_arm": status.get("paper_arm"),
                "market_session": _market_session_card().get("technical"),
            }
        )


def _history_dicts(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    root = _root()
    path = root / "data" / "results" / "portfolio_history.csv"
    if not path.is_file():
        return []
    try:
        import pandas as pd

        df = pd.read_csv(path)
        return df.to_dict("records")
    except Exception:
        return []


def _certified_holdings_table(holdings: Sequence[Mapping[str, Any]]) -> None:
    if not holdings:
        oui.render_empty_state("No holdings", "No certified position rows for this close.")
        return
    display = [
        {
            "Symbol": row.get("symbol"),
            "Quantity": row.get("quantity"),
            "Market value": row.get("market_value"),
            "Average basis": row.get("average_basis"),
            "Unrealized gain/loss": row.get("unrealized"),
            "Weight": row.get("weight"),
        }
        for row in holdings
    ]
    render_operator_table(
        display,
        signed_columns=("Unrealized gain/loss",),
        caption="Holdings are from the latest certified accounting close.",
    )


def _legacy_portfolio_history_section(summary: Mapping[str, Any]) -> None:
    rows = normalize_history_rows(_history_dicts(summary))
    with st.expander("Legacy portfolio history", expanded=False):
        st.caption(
            "Historical operational records are not used for verified performance calculations."
        )
        periods = available_periods(rows)
        if not periods:
            st.caption("No dated operational observations are available.")
            return
        default_idx = periods.index("1M") if "1M" in periods else len(periods) - 1
        period = st.radio(
            "Period", periods, index=default_idx, horizontal=True, key="ui_a_port_period"
        )
        filtered = filter_period(rows, period)
        state = chart_state(filtered)
        try:
            fig = build_plotly_figure(filtered)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.caption("Interactive chart unavailable — values below remain evidenced.")
            st.line_chart(
                {str(p["date"]): p["total_value"] for p in filtered} if filtered else {},
                height=280,
            )
        tone = str(state.get("tone") or "unavailable")
        tone_phrase = {
            "positive": "up",
            "negative": "down",
            "neutral": "unchanged",
            "unavailable": "unavailable",
        }.get(tone, tone)
        st.caption(
            f"The selected operational period is {tone_phrase}. "
            "Gaps are missing observations, not interpolated values."
        )


def render_portfolio_page() -> None:
    from services.operator_console_status import build_portfolio_summary

    oui.inject_operator_theme()
    _isolation_banner()
    oui.render_page_intro(
        "Portfolio",
        "Current totals from the latest certified accounting close.",
    )
    summary = build_portfolio_summary(root=_root())
    certified = _certified_totals()
    if certified and certified.get("available"):
        _portfolio_metrics(summary, [], certified=certified)
        st.markdown("#### Holdings")
        _certified_holdings_table(certified.get("holdings") or [])
        _legacy_portfolio_history_section(summary)
    else:
        if not summary.get("available"):
            st.warning("Portfolio artifacts are unavailable or unreadable.")
        rows = normalize_history_rows(_history_dicts(summary))
        _portfolio_metrics(summary, rows[-2:] if len(rows) >= 2 else rows)
        st.markdown("#### Holdings")
        _holdings_table(summary)
        _legacy_portfolio_history_section(summary)
    if summary.get("concentration_warnings"):
        with st.expander("Concentration warnings", expanded=True):
            st.write(summary["concentration_warnings"])
    with st.expander("Data sources", expanded=False):
        st.write(summary.get("evidence_paths"))
        if certified:
            st.caption("Current totals source: certified accounting close.")
        st.caption(f"Last refresh: {oui.format_timestamp(summary.get('last_refresh_time'))}")


def _order_badge_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    term = row.get("terminal_status")
    proven = bool(row.get("terminal_status_proven"))
    submitted = row.get("submission_status")
    disposition = row.get("action_disposition")
    if proven and term:
        status = str(term)
    elif (
        disposition in {"explicitly_skipped", "SKIPPED_ZERO"}
        or str(disposition).upper() == "EXPLICITLY_SKIPPED"
    ):
        status = "Skipped"
    elif disposition == "failed":
        status = "Failed"
    elif submitted:
        status = str(submitted)
        if str(submitted).upper() == "FILLED" and not proven:
            status = "Submitted"
    else:
        status = "Unknown"
    price = row.get("average_fill_price")
    if price is None:
        price = row.get("submitted_notional")
    return {
        "Status": display_status(status),
        "status_raw": status,
        "Symbol": row.get("symbol"),
        "Side": _display_order_side(row),
        "Quantity": quantity_text(
            row.get("selected_qty")
            if row.get("selected_qty") is not None
            else row.get("filled_qty")
        ),
        "Price": money_text(price) if price is not None else "Unavailable",
        "Time": oui.format_timestamp(
            row.get("event_timestamp") or row.get("last_event_timestamp") or row.get("trading_date")
        ),
    }


def _display_order_side(row: Mapping[str, Any]) -> str:
    from dashboard.order_presentation import display_side

    return display_side(row.get("side"))


def render_orders_page() -> None:
    from dashboard.order_presentation import (
        AuthoritativeLedgerError,
        load_authoritative_presentation,
        plan_table_row,
    )
    from dashboard.preview_data import (
        assert_not_production_path,
        expected_ledger_sha256,
        isolation_active,
    )
    from services.operator_console_status import build_orders_summary
    from services.paper_order_ledger import LEDGER_RELATIVE, filter_unified_rows, ledger_path

    oui.inject_operator_theme()
    _isolation_banner()
    oui.render_page_intro("Orders", "Unified paper-order ledger. Submitted is not filled.")
    summary = build_orders_summary(root=_root())
    assert_not_production_path(ledger_path(_root()), root=_repo_root())
    isolated = isolation_active(_repo_root())
    expected = expected_ledger_sha256(_repo_root()) if isolated else None
    try:
        counts = load_authoritative_presentation(
            root=_root(),
            expected_sha256=expected,
            require_expected_hash=isolated,
        )
    except AuthoritativeLedgerError as exc:
        oui.render_empty_state(
            "Authoritative ledger unavailable",
            "Orders are not shown because the paper-order ledger is missing, unreadable, or failed validation.",
        )
        st.caption(str(exc))
        return
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Broker orders", counts.get("broker_orders") or 0)
    t2.metric("Filled", counts.get("filled") or 0)
    t3.metric("Open", counts.get("open") or 0)
    t4.metric("Skipped", counts.get("skipped") or 0)
    t5.metric(
        "Failed / rejected",
        int(counts.get("failed") or 0) + int(counts.get("rejected") or 0),
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        date_f = st.text_input("Trading date", key="ui_a_pol_date")
    with f2:
        symbol_f = st.text_input("Symbol", key="ui_a_pol_symbol")
    with f3:
        side_f = st.selectbox(
            "Side",
            ["ALL", "BUY", "SELL"],
            format_func=lambda v: "All" if v == "ALL" else display_status(v),
            key="ui_a_pol_side",
        )
    with f4:
        status_f = st.selectbox(
            "Status",
            ["ALL", "SUBMITTED", "OPEN", "PARTIAL", "FILLED", "REJECTED", "CANCELLED", "EXPIRED"],
            format_func=lambda v: "All" if v == "ALL" else display_status(v),
            key="ui_a_pol_status",
        )
    shown = filter_unified_rows(
        counts["broker_rows"],
        trading_date=date_f or None,
        symbol=symbol_f or None,
        side=None if side_f == "ALL" else side_f,
        status=None if status_f == "ALL" else status_f,
    )
    if not shown:
        oui.render_empty_state("No orders", "No orders match these filters.")
    else:
        display = []
        for row in shown[:200]:
            fields = _order_badge_fields(row)
            display.append(
                {
                    "Status": fields["Status"],
                    "Symbol": fields["Symbol"],
                    "Side": fields["Side"],
                    "Quantity": fields["Quantity"],
                    "Price": fields["Price"],
                    "Time": fields["Time"],
                }
            )
        render_operator_table(display, caption="Submitted is not the same as filled.")
        with st.expander("Row details", expanded=False):
            st.caption("Authority IDs, dispatch IDs, and broker references stay here.")
            pick = st.selectbox(
                "Row",
                options=list(range(min(len(shown), 50))),
                format_func=lambda i: (
                    f"{shown[i].get('symbol') or '—'} · "
                    f"{_display_order_side(shown[i])} · "
                    f"{_order_badge_fields(shown[i])['Status']}"
                ),
                key="ui_a_order_detail",
            )
            row = shown[int(pick)]
            st.write(_order_badge_fields(row))
            st.caption("Technical identifiers")
            st.json(
                technical_payload(
                    row,
                    (
                        "authority_id",
                        "dispatch_id",
                        "broker_order_id",
                        "client_order_id",
                        "source_pipeline",
                        "logical_key",
                        "source_artifact",
                        "side",
                    ),
                )
            )

    plan_rows = list(counts.get("plan_rows") or [])
    with st.expander(
        f"Historical plans ({len(plan_rows)})",
        expanded=False,
    ):
        st.caption(
            "Legacy planning records. These are not broker orders and are not an order side."
        )
        if plan_rows:
            render_operator_table([plan_table_row(r) for r in plan_rows[:200]])
            with st.expander("Technical details", expanded=False):
                st.json(
                    [
                        technical_payload(
                            r,
                            (
                                "logical_key",
                                "side",
                                "source_pipeline",
                                "action_disposition",
                                "source_artifact",
                                "kind",
                            ),
                        )
                        for r in plan_rows[:50]
                    ]
                )
        else:
            st.caption("No historical planning records in the current evidence.")

    current = summary.get("current_open_orders") or {}
    with st.expander("Current open-order snapshot", expanded=False):
        from dashboard.freshness_presentation import shared_freshness_view
        from dashboard.preview_data import isolation_active as _iso

        fresh = shared_freshness_view(
            {"data_freshness": {"class": current.get("freshness")}},
            isolated=_iso(_repo_root()),
        )
        st.caption(
            f"{fresh['value']}"
            + (f" · {fresh['note']}" if fresh.get("note") else "")
            + (
                f" · {current.get('evidence_timestamp')}"
                if current.get("evidence_timestamp") and not fresh.get("isolated")
                else ""
            )
        )
        open_rows = current.get("rows") or []
        if open_rows:
            render_operator_table(open_rows, format_headers=True, raw_label="Raw open-order rows")
        else:
            st.caption(current.get("message") or "No open-order rows in this snapshot.")
    with st.expander("Evidence", expanded=False):
        st.write({"ledger": LEDGER_RELATIVE, **(summary.get("evidence_paths") or {})})
        st.caption(
            f"Broker orders={counts.get('broker_orders')} · "
            f"historical plans={counts.get('historical_plans')}"
        )


def _activity_tone(event: Mapping[str, Any]) -> str:
    sev = str(event.get("severity") or "").upper()
    title = str(event.get("title") or "").lower()
    if sev in ("CRITICAL", "ERROR", "FAILED") or "fail" in title or "block" in title:
        return "crit"
    if "fill" in title or sev == "SUCCESS":
        return "ok"
    if "skip" in title or sev in ("WARN", "WARNING"):
        return "warn"
    if "submit" in title:
        return "info"
    return "muted"


def _render_activity_timeline(
    events: Sequence[Mapping[str, Any]],
    *,
    limit: int = 40,
    show_evidence: bool = True,
) -> None:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for ev in events[:limit]:
        key = str(
            ev.get("order_id") or ev.get("cycle_number") or ev.get("occurred_date") or "other"
        )
        grouped.setdefault(key, []).append(ev)
    st.markdown('<div class="triton-timeline">', unsafe_allow_html=True)
    for group_key, items in grouped.items():
        label = group_key
        if is_raw_identifier(group_key):
            label = f"Group {group_key[:12]}"
        st.caption(f"{label} · {len(items)}")
        for ev in items:
            tone = _activity_tone(ev)
            title = display_label(ev.get("title") or "Event")
            when = oui.format_timestamp(ev.get("occurred_at") or ev.get("occurred_date"))
            st.markdown(
                f'<div class="triton-timeline-item {tone}">'
                f'<div><strong>{oui.escape(title)}</strong> · {oui.escape(display_status(ev.get("severity")))}</div>'
                f'<div class="triton-attention-body">{oui.escape(ev.get("message") or "")}</div>'
                f'<div class="triton-attention-body">{oui.escape(when)}'
                f' · {oui.escape(display_label(ev.get("category") or ""))}'
                f' · {oui.escape(str(ev.get("symbol") or ""))}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        if show_evidence:
            with st.expander(f"Evidence · {label}", expanded=False):
                for ev in items:
                    st.caption(f"Source: {ev.get('source')}")
                    evidence_path = str(ev.get("evidence_path") or "")
                    if evidence_path and ":" not in evidence_path[:3] and "\\" not in evidence_path:
                        st.caption(f"Artifact: {evidence_path}")
                    st.code(str(ev.get("raw_reference") or ""), language="text")
    st.markdown("</div>", unsafe_allow_html=True)


def render_activity_page() -> None:
    from services.operator_activity_feed import (
        CATEGORIES,
        SEVERITIES,
        collect_activity_events,
        filter_activity_events,
    )
    from services.paper_order_ledger import merge_activity_payload

    oui.inject_operator_theme()
    _isolation_banner()
    oui.render_page_intro("Activity", "Chronological events from logs and the paper-order ledger.")
    cols = st.columns(4)
    with cols[0]:
        hours = st.selectbox("Hours", [6, 24, 72, 168, 720], index=1, key="ui_a_act_hours")
    with cols[1]:
        category = st.selectbox(
            "Category",
            ["ALL", *CATEGORIES],
            format_func=lambda v: "All" if v == "ALL" else display_label(v),
            key="ui_a_act_cat",
        )
    with cols[2]:
        severity = st.selectbox(
            "Severity",
            ["ALL", *SEVERITIES],
            format_func=lambda v: "All" if v == "ALL" else display_status(v),
            key="ui_a_act_sev",
        )
    with cols[3]:
        query = st.text_input("Search", key="ui_a_act_q")
    payload = merge_activity_payload(
        collect_activity_events(root=_root(), since_hours=float(hours))
    )
    events = filter_activity_events(
        payload, category=category, severity=severity, source="ALL", query=query
    )
    st.caption(f"{len(events)} of {payload.get('event_count', 0)} evidenced events.")
    if not events:
        oui.render_empty_state(
            "No events", "Nothing matched these filters. Sources were not invented."
        )
        with st.expander("Data sources", expanded=False):
            st.write(payload.get("sources_used") or [])
        return
    _render_activity_timeline(events, limit=60)
    with st.expander("Data sources", expanded=False):
        st.write(payload.get("sources_used") or [])


def render_operations_page() -> None:
    from services.operator_console_status import build_operator_status
    from services.operator_runtime_controller import start_eligibility

    oui.inject_operator_theme()
    _isolation_banner()
    oui.render_page_intro("Operations", "What is running, and what should you do next?")
    status = build_operator_status(root=_root())
    render_status_bar(status)
    render_authority_and_runtime_controls(status)

    attention = list(status.get("attention") or [])
    start_el = start_eligibility()
    st.markdown("#### Needs attention")
    if attention:
        for item in attention:
            oui.render_attention_item(
                item,
                start_decision=start_el.get("decision"),
                start_button_enabled=start_el.get("start_button_enabled"),
            )
    else:
        st.info("No evidence-backed attention items.")

    from dashboard.operator_console import _render_runtime_recovery_guidance

    with st.expander("Why this status?", expanded=False):
        _render_runtime_recovery_guidance()

    with st.expander("Technical details", expanded=False):
        st.json(
            {
                "overall_status": status.get("overall_status"),
                "continuous": status.get("continuous"),
                "watchdog": status.get("watchdog"),
                "cycle": status.get("cycle"),
                "broker": status.get("broker"),
                "live_execution": status.get("live_execution"),
                "paper_arm": status.get("paper_arm"),
                "market_session": _market_session_card().get("technical"),
                "evidence_paths": status.get("evidence_paths"),
            }
        )


def render_risk_alerts_page() -> None:
    from services.operator_console_status import build_risk_alerts_summary

    oui.inject_operator_theme()
    _isolation_banner()
    oui.render_page_intro(
        "Risk & Alerts", "Read-only risk posture. Alerts cannot be dismissed here."
    )
    summary = build_risk_alerts_summary(root=_root())
    st.metric("Capital preservation", display_status(summary.get("capital_preservation_posture")))
    if summary.get("recommended_operator_response"):
        st.caption(summary["recommended_operator_response"])
    active = summary.get("active_conditions") or []
    st.markdown("#### Active")
    if active:
        render_operator_table(active, format_headers=True)
    else:
        st.info("No active alerts evidenced.")
    with st.expander("Resolved and supporting evidence", expanded=False):
        st.write("Resolved", summary.get("resolved_conditions") or [])
        st.write("Risk-overlay actions", summary.get("risk_overlay_actions") or [])
        st.write("Concentration", summary.get("concentration_issues") or [])
        st.write("Drawdown", summary.get("drawdown_warnings") or [])
        st.write("Governance blocks", summary.get("governance_blocks"))
        st.write(summary.get("evidence_paths"))


def render_diagnostics_page(*, switch_to_legacy_callback=None) -> None:
    from dashboard.operator_console import render_diagnostics_page as _legacy

    oui.inject_operator_theme()
    _isolation_banner()
    oui.render_page_intro("Diagnostics", "Technical evidence. Secrets are not shown.")
    with st.expander("Advanced diagnostics", expanded=True):
        _legacy(switch_to_legacy_callback=switch_to_legacy_callback)


def render_reports_page() -> None:
    from dashboard.operator_reports import render_operator_reports_page

    oui.inject_operator_theme()
    _isolation_banner()
    render_operator_reports_page()


def render_performance_page() -> None:
    from dashboard.performance_page import render_performance_page as _render

    _render()


UI_A_RENDERERS = {
    "Overview": render_overview_page,
    "Portfolio": render_portfolio_page,
    "Performance": render_performance_page,
    "Orders": render_orders_page,
    "Activity": render_activity_page,
    "Reports": render_reports_page,
    "Operations": render_operations_page,
    "Risk & Alerts": render_risk_alerts_page,
    "Diagnostics": render_diagnostics_page,
}
