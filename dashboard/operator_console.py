"""
operator_console.py
Phase DO-002/DO-003 — Streamlit operator-console pages.

DO-003 adds protected START_TRITON only.
Stop/Restart/Arm/Approve/Execute remain unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

OPERATOR_PAGES = [
    "Operations",
    "Activity",
    "Portfolio",
    "Orders",
    "Risk & Alerts",
    "Diagnostics",
]

DEFAULT_OPERATOR_PAGE = "Operations"

SIMULATION_LABEL_FRAGMENTS = (
    "Simulation",
    "Sandbox",
    "Stress Lab",
    "Scenario Planning",
    "Scenario Forecasting",
    "Rehearsal",
    "Trials",
)


def is_simulation_nav_label(label: str) -> bool:
    return any(frag.lower() in str(label).lower() for frag in SIMULATION_LABEL_FRAGMENTS)


def operator_page_list() -> List[str]:
    return list(OPERATOR_PAGES)


def _status_color(status: str) -> str:
    s = str(status or "").upper()
    if s in ("RUNNING", "ARMED", "FRESH", "SUCCESS", "LIVE EXECUTION BLOCKED"):
        return "🟢"
    if s in ("DEGRADED", "EXPIRED", "STALE", "WARN"):
        return "🟡"
    if s in ("STOPPED", "MISSING"):
        return "⚪"
    if s in ("CRITICAL", "INVALID") or "NOT BLOCKED" in s:
        return "🔴"
    return "⚫"


def _render_start_controls() -> None:
    """Protected START_TRITON UI (DO-003). Stop/Restart remain unavailable."""
    from services.operator_runtime_controller import (
        ACTION_START,
        execute_start_triton,
        start_eligibility,
    )

    eligibility = start_eligibility()
    st.subheader("Runtime controls")
    st.caption(
        f"Start eligibility: **{eligibility.get('decision')}** · "
        f"Paper arm (not modified by Start): **{eligibility.get('paper_arm_status')}**"
    )
    st.write(eligibility.get("explanation") or "")

    if "start_idempotency_key" not in st.session_state:
        st.session_state["start_idempotency_key"] = None
    if "start_confirm_pending" not in st.session_state:
        st.session_state["start_confirm_pending"] = False
    if "start_last_result" not in st.session_state:
        st.session_state["start_last_result"] = None
    if "start_in_progress" not in st.session_state:
        st.session_state["start_in_progress"] = False

    last = st.session_state.get("start_last_result")
    if last:
        res = str(last.get("result") or "")
        msg = last.get("safe_message") or res
        if res in ("STARTED", "ALREADY_RUNNING"):
            st.success(msg)
        elif res in ("PARTIAL_START", "START_IN_PROGRESS"):
            st.warning(msg) if res == "PARTIAL_START" else st.info(msg)
        else:
            st.error(msg)
        with st.expander("Start technical evidence", expanded=False):
            st.json(
                {
                    "result": last.get("result"),
                    "action_id": last.get("action_id"),
                    "launched_components": last.get("launched_components"),
                    "preserved_components": last.get("preserved_components"),
                    "live_execution_blocked_after": last.get("live_execution_blocked_after"),
                    "paper_arm_status": last.get("paper_arm_status"),
                    "paper_arm_mutated": last.get("paper_arm_mutated"),
                    "error_code": last.get("error_code"),
                }
            )

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        if eligibility.get("already_running"):
            st.info("TRITON is already running")
        elif st.session_state.get("start_in_progress") or eligibility.get("guard_active"):
            st.button("Start TRITON", key="start_triton_disabled_busy", disabled=True)
            st.caption("Start action in progress…")
        elif not eligibility.get("start_button_enabled"):
            st.button("Start TRITON", key="start_triton_disabled_unsafe", disabled=True)
            st.caption("Start disabled: unsafe, ambiguous, or not eligible.")
        else:
            if not st.session_state.get("start_confirm_pending"):
                if st.button("Start TRITON", key="start_triton_request", type="primary"):
                    import uuid

                    st.session_state["start_confirm_pending"] = True
                    st.session_state["start_idempotency_key"] = str(uuid.uuid4())
                    st.rerun()
            else:
                st.warning(
                    "This starts TRITON’s local watchdog and continuous runtime in its existing "
                    "paper-only configuration. Live execution will remain blocked. "
                    "This does **not** create or extend paper-arm authorization."
                )
                conf_a, conf_b = st.columns(2)
                with conf_a:
                    if st.button("Confirm Start", key="start_triton_confirm", type="primary"):
                        st.session_state["start_in_progress"] = True
                        st.session_state["start_confirm_pending"] = False
                        with st.spinner("Starting and validating TRITON…"):
                            result = execute_start_triton(
                                action=ACTION_START,
                                idempotency_key=st.session_state.get("start_idempotency_key"),
                                source="operator_console",
                                operator_context={"ui": "operations"},
                            )
                        st.session_state["start_last_result"] = result
                        st.session_state["start_in_progress"] = False
                        st.session_state["start_idempotency_key"] = None
                        st.rerun()
                with conf_b:
                    if st.button("Cancel", key="start_triton_cancel"):
                        st.session_state["start_confirm_pending"] = False
                        st.session_state["start_idempotency_key"] = None
                        st.rerun()
    with ctrl2:
        st.button("Stop TRITON — unavailable", key="stop_triton_placeholder", disabled=True)
        st.caption(
            "Stop and Restart will become available only after cooperative shutdown "
            "is implemented and validated."
        )
    with ctrl3:
        st.button("Restart TRITON — unavailable", key="restart_triton_placeholder", disabled=True)
        st.caption("No process termination or lock deletion is available from this console.")

    st.caption(
        f"allowed_action={eligibility.get('allowed_action')} · "
        f"stop_available={eligibility.get('stop_available')} · "
        f"restart_available={eligibility.get('restart_available')}"
    )


def render_operations_page() -> None:
    from services.operator_console_status import build_operator_status

    st.title("Operations")
    st.caption(
        "Operator console · Protected Start available · Live execution remains blocked · "
        "Stop/Restart unavailable until cooperative shutdown exists."
    )

    status = build_operator_status()
    overall = str(status.get("overall_status") or "UNKNOWN")
    st.markdown(f"### {_status_color(overall)} TRITON overall status: **{overall}**")

    live = status.get("live_execution") or {}
    if live.get("blocked"):
        st.success("LIVE EXECUTION BLOCKED")
    else:
        st.error(
            "CRITICAL SAFETY WARNING: live execution does not appear blocked. "
            "No enable control is provided. Investigate immediately."
        )

    c1, c2, c3, c4 = st.columns(4)
    cont = status.get("continuous") or {}
    wd = status.get("watchdog") or {}
    paper = status.get("paper_arm") or {}
    with c1:
        st.metric("Continuous loop", "RUNNING" if cont.get("process_running") else "NOT RUNNING")
        st.caption(f"Lock: {cont.get('lock_classification')}")
    with c2:
        st.metric("Watchdog", "RUNNING" if wd.get("process_running") else "NOT RUNNING")
        st.caption(f"Lock: {wd.get('lock_classification')}")
    with c3:
        st.metric("Paper arm", paper.get("display_status") or "UNKNOWN")
        st.caption(paper.get("evidence_path") or "")
    with c4:
        st.metric("Live execution", "BLOCKED" if live.get("blocked") else "NOT BLOCKED")

    c5, c6, c7, c8 = st.columns(4)
    broker = status.get("broker") or {}
    cycle = status.get("cycle") or {}
    op = status.get("orders_positions") or {}
    alerts = status.get("alerts") or {}
    freshness = status.get("data_freshness") or {}
    with c5:
        conn = broker.get("connected")
        st.metric(
            "Broker connection",
            "CONNECTED" if conn is True else ("DISCONNECTED" if conn is False else "UNKNOWN"),
        )
    with c6:
        st.metric("Market status", (status.get("market") or {}).get("status") or "UNKNOWN")
    with c7:
        st.metric(
            "Last cycle",
            str(
                cycle.get("last_cycle_number")
                if cycle.get("last_cycle_number") is not None
                else "UNKNOWN"
            ),
        )
        st.caption(cycle.get("last_completed_at") or "No cycle timestamp evidenced")
    with c8:
        st.metric("Next expected cycle", "See caption")
        st.caption(cycle.get("next_expected_at") or "Unavailable")

    c9, c10, c11, c12 = st.columns(4)
    with c9:
        oo = op.get("open_orders")
        st.metric("Open orders", oo if oo is not None else "UNKNOWN")
    with c10:
        pc = op.get("position_count")
        st.metric("Position count", pc if pc is not None else "UNKNOWN")
    with c11:
        st.metric(
            "Active alerts",
            alerts.get("active_count") if alerts.get("active_count") is not None else "UNKNOWN",
        )
    with c12:
        st.metric("Data freshness", freshness.get("class") or "UNKNOWN")

    st.markdown("---")
    st.subheader("Operator attention")
    attention = status.get("attention") or []
    if not attention:
        st.info("No evidence-backed attention items right now.")
    else:
        for item in attention:
            sev = str(item.get("severity") or "INFO")
            header = f"[{sev}] {item.get('title')}"
            with st.expander(header, expanded=sev in ("CRITICAL", "HIGH")):
                st.write(item.get("explanation"))
                st.caption(f"Evidence: {item.get('evidence_source')}")
                st.write(f"Recommended safe action: {item.get('recommended_action')}")

    st.markdown("---")
    _render_start_controls()


def render_activity_page() -> None:
    from services.operator_activity_feed import (
        CATEGORIES,
        SEVERITIES,
        collect_activity_events,
        filter_activity_events,
    )

    st.title("Activity")
    st.caption(
        "Plain-language events derived only from real logs and artifacts. Nothing is invented."
    )

    # Auto-refresh interval (seconds) via fragment-like rerun hint
    if "activity_auto_refresh_sec" not in st.session_state:
        st.session_state["activity_auto_refresh_sec"] = 30

    cols = st.columns([1, 1, 1, 1, 2])
    with cols[0]:
        if st.button("Refresh", key="activity_refresh_btn"):
            st.session_state.pop("activity_feed_cache", None)
            st.rerun()
    with cols[1]:
        hours = st.selectbox(
            "Time range (hours)", [6, 24, 72, 168, 720], index=2, key="activity_hours"
        )
    with cols[2]:
        category = st.selectbox("Category", ["ALL", *CATEGORIES], key="activity_category")
    with cols[3]:
        severity = st.selectbox("Severity", ["ALL", *SEVERITIES], key="activity_severity")
    with cols[4]:
        query = st.text_input("Search symbol/text", key="activity_query")

    source = st.selectbox(
        "Source",
        [
            "ALL",
            "continuous_loop_log",
            "watchdog_loop_log",
            "runtime_autostart",
            "paper_entry_stage_audit",
            "paper_execution_audit",
            "watchdog_alerts",
            "data_freshness_audit",
            "execution_readiness",
            "governance_authorization",
        ],
        key="activity_source",
    )

    payload = collect_activity_events(since_hours=float(hours))
    for warn in payload.get("stale_source_warnings") or []:
        st.warning(f"Stale source: {warn}")

    events = filter_activity_events(
        payload, category=category, severity=severity, source=source, query=query
    )
    st.caption(f"Showing {len(events)} of {payload.get('event_count', 0)} evidenced events.")

    if not events:
        st.info(
            "No evidenced events for the selected filters. Missing sources display as unavailable — nothing was fabricated."
        )
        with st.expander("Sources consulted"):
            st.write(payload.get("sources_used") or [])
            st.write(payload.get("evidence_rules") or [])
        return

    for ev in events:
        title = f"{ev.get('severity')} · {ev.get('category')} · {ev.get('title')}"
        with st.expander(title, expanded=False):
            st.write(ev.get("message"))
            meta = []
            if ev.get("occurred_at"):
                meta.append(f"When: {ev['occurred_at']}")
            if ev.get("cycle_number") is not None:
                meta.append(f"Cycle: {ev['cycle_number']}")
            if ev.get("symbol"):
                meta.append(f"Symbol: {ev['symbol']}")
            if ev.get("order_id"):
                meta.append(f"Order: {ev['order_id']}")
            meta.append(f"Source: {ev.get('source')}")
            st.caption(" · ".join(meta))
            st.code(str(ev.get("raw_reference") or ""), language="text")
            st.caption(f"Evidence path: {ev.get('evidence_path')}")

    # Lightweight auto-refresh using Streamlit meta refresh alternative
    refresh_sec = int(st.session_state.get("activity_auto_refresh_sec") or 0)
    if refresh_sec > 0:
        st.caption(
            f"Auto-refresh target interval: {refresh_sec}s (manual Refresh always available)."
        )
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore

            st_autorefresh(interval=refresh_sec * 1000, key="activity_autorefresh")
        except Exception:
            # Optional dependency; manual refresh remains available
            pass


def render_portfolio_page() -> None:
    from services.operator_console_status import build_portfolio_summary

    st.title("Portfolio")
    st.caption("Read-only portfolio view from trusted artifacts.")
    summary = build_portfolio_summary()
    if not summary.get("available"):
        st.warning("Portfolio artifacts unavailable or unreadable.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash", summary.get("cash") if summary.get("cash") is not None else "UNKNOWN")
    c2.metric(
        "Market value",
        summary.get("market_value") if summary.get("market_value") is not None else "UNKNOWN",
    )
    c3.metric(
        "Total portfolio value",
        (
            summary.get("total_portfolio_value")
            if summary.get("total_portfolio_value") is not None
            else "UNKNOWN"
        ),
    )
    c4.metric(
        "Position count",
        summary.get("position_count") if summary.get("position_count") is not None else "UNKNOWN",
    )
    c5, c6 = st.columns(2)
    c5.metric(
        "Realized P/L",
        summary.get("realized_pnl") if summary.get("realized_pnl") is not None else "UNKNOWN",
    )
    c6.metric(
        "Unrealized P/L",
        summary.get("unrealized_pnl") if summary.get("unrealized_pnl") is not None else "UNKNOWN",
    )
    st.caption(f"Last refresh: {summary.get('last_refresh_time') or 'UNKNOWN'}")

    if summary.get("concentration_warnings"):
        st.subheader("Concentration warnings")
        st.write(summary["concentration_warnings"])
    if summary.get("risk_flags"):
        st.subheader("Risk flags")
        st.dataframe(summary["risk_flags"], use_container_width=True)
    st.subheader("Current holdings")
    holdings = summary.get("holdings") or []
    if holdings:
        st.dataframe(holdings, use_container_width=True)
    else:
        st.info("No holdings rows evidenced.")
    with st.expander("Evidence paths"):
        st.write(summary.get("evidence_paths"))


def render_orders_page() -> None:
    from services.operator_console_status import build_orders_summary

    st.title("Orders")
    st.caption(
        "Read-only paper/order visibility. No Submit, Cancel, Approve, Retry, or Execute controls."
    )
    summary = build_orders_summary()
    st.write(summary.get("note"))
    st.write("Counts by normalized status:", summary.get("by_status"))

    def _table(title: str, rows: List[Dict[str, Any]]) -> None:
        st.subheader(title)
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info(f"No evidenced rows for {title}.")

    _table("Open / working orders", summary.get("open_orders") or [])
    _table("Filled orders", summary.get("filled_orders") or [])
    _table("Rejected orders", summary.get("rejected_orders") or [])
    _table("Cancelled orders", summary.get("cancelled_orders") or [])
    _table("All ingested order rows", summary.get("orders") or [])
    with st.expander("Evidence paths"):
        st.write(summary.get("evidence_paths"))


def render_risk_alerts_page() -> None:
    from services.operator_console_status import build_risk_alerts_summary

    st.title("Risk & Alerts")
    st.caption(
        "Read-only risk and watchdog posture. Alerts cannot be dismissed or overridden here."
    )
    summary = build_risk_alerts_summary()
    st.metric(
        "Capital-preservation posture", summary.get("capital_preservation_posture") or "UNKNOWN"
    )
    st.info(summary.get("recommended_operator_response") or "")

    st.subheader("Active conditions")
    active = summary.get("active_conditions") or []
    if active:
        st.dataframe(active, use_container_width=True)
    else:
        st.write("No active alerts evidenced.")

    st.subheader("Resolved conditions")
    resolved = summary.get("resolved_conditions") or []
    if resolved:
        st.dataframe(resolved, use_container_width=True)
    else:
        st.write("No resolved alerts evidenced.")

    st.subheader("Risk-overlay actions")
    st.dataframe(summary.get("risk_overlay_actions") or [], use_container_width=True)
    st.subheader("Concentration issues")
    st.write(summary.get("concentration_issues") or [])
    st.subheader("Drawdown warnings")
    st.write(summary.get("drawdown_warnings") or [])
    st.subheader("Governance blocks")
    st.write(summary.get("governance_blocks"))
    st.subheader("Execution readiness")
    st.write(summary.get("execution_readiness"))
    st.subheader("Informational notices")
    st.write(summary.get("informational_notices") or [])
    with st.expander("Evidence"):
        st.write(summary.get("evidence_timestamps"))
        st.write(summary.get("evidence_paths"))
        st.caption(f"can_dismiss_or_override={summary.get('can_dismiss_or_override')}")


def render_diagnostics_page(*, switch_to_legacy_callback=None) -> None:
    from services.operator_console_status import build_operator_status

    st.title("Diagnostics")
    st.caption("Technical evidence for operators. Secrets and credentials are not displayed.")

    status = build_operator_status()
    st.subheader("Process and lock evidence")
    st.json(
        {
            "overall_status": status.get("overall_status"),
            "continuous": status.get("continuous"),
            "watchdog": status.get("watchdog"),
        }
    )
    st.subheader("Cycle / watchdog / broker / market-data freshness")
    st.json(
        {
            "cycle": status.get("cycle"),
            "broker": status.get("broker"),
            "data_freshness": status.get("data_freshness"),
            "alerts": {
                "active_count": (status.get("alerts") or {}).get("active_count"),
                "critical_count": (status.get("alerts") or {}).get("critical_count"),
            },
        }
    )
    st.subheader("Paper-arm and live-block evidence")
    st.json({"paper_arm": status.get("paper_arm"), "live_execution": status.get("live_execution")})

    st.subheader("Start eligibility (DO-003)")
    from services.operator_runtime_controller import read_recent_runtime_actions, start_eligibility

    eligibility = start_eligibility()
    st.json(
        {
            "decision": eligibility.get("decision"),
            "explanation": eligibility.get("explanation"),
            "components_to_start": eligibility.get("components_to_start"),
            "preserved_components": eligibility.get("preserved_components"),
            "live_execution_blocked": eligibility.get("live_execution_blocked"),
            "paper_arm_status": eligibility.get("paper_arm_status"),
            "guard_active": eligibility.get("guard_active"),
            "start_button_enabled": eligibility.get("start_button_enabled"),
            "fixed_paths": eligibility.get("fixed_paths"),
            "stop_available": eligibility.get("stop_available"),
            "restart_available": eligibility.get("restart_available"),
            "allowed_action": eligibility.get("allowed_action"),
            "denied_actions": eligibility.get("denied_actions"),
        }
    )
    with st.expander("Loop classifications"):
        st.json(eligibility.get("classifications"))
    with st.expander("Recent sanitized runtime-action audit"):
        st.json(read_recent_runtime_actions(limit=15))
    st.caption(
        "Diagnostics is read-only. No repair, delete, kill, or override buttons are provided."
    )

    st.subheader("Runtime autostart audit")
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parents[1]
    for name in ("runtime_autostart_931.json", "runtime_supervisor_autostart.json"):
        path = root / "data" / "results" / name
        with st.expander(name):
            if not path.is_file():
                st.warning(f"Missing: {path}")
            else:
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                    # Redact potential secrets
                    if isinstance(doc, dict):
                        safe = {
                            k: v
                            for k, v in doc.items()
                            if not any(
                                s in str(k).lower()
                                for s in ("key", "token", "secret", "password", "credential")
                            )
                        }
                        # Strip env values that might contain secrets
                        if isinstance(safe.get("env"), dict):
                            safe["env"] = {
                                ek: (
                                    "<redacted>"
                                    if any(x in ek.lower() for x in ("key", "secret", "token"))
                                    else ev
                                )
                                for ek, ev in safe["env"].items()
                            }
                        st.json(safe)
                    else:
                        st.write("Unreadable audit shape")
                except Exception:
                    st.warning("Malformed audit JSON")

    st.subheader("Dashboard configuration")
    st.json(
        {
            "operator_pages": OPERATOR_PAGES,
            "default_page": DEFAULT_OPERATOR_PAGE,
            "runtime_controls_exposed": True,
            "allowed_runtime_action": "START_TRITON",
            "live_execution_control_exposed": False,
            "stop_restart_available": False,
            "localhost_recommended": "streamlit run view_results.py --server.address 127.0.0.1",
        }
    )

    missing = []
    for label, p in (status.get("evidence_paths") or {}).items():
        if not Path(p).is_file():
            missing.append({"artifact": label, "path": p})
    st.subheader("Missing-file warnings")
    if missing:
        st.dataframe(missing, use_container_width=True)
    else:
        st.write("All primary evidence paths currently exist (or none reported missing).")

    with st.expander("All evidence paths"):
        st.write(status.get("evidence_paths"))

    st.markdown("---")
    st.info(
        "Advanced/Legacy Dashboard is available from the sidebar mode switch. "
        "It is not required for routine operation."
    )
    if switch_to_legacy_callback is not None:
        if st.button("Switch to Advanced / Legacy Dashboard", key="diag_switch_legacy"):
            switch_to_legacy_callback()


OPERATOR_RENDERERS = {
    "Operations": render_operations_page,
    "Activity": render_activity_page,
    "Portfolio": render_portfolio_page,
    "Orders": render_orders_page,
    "Risk & Alerts": render_risk_alerts_page,
    "Diagnostics": render_diagnostics_page,
}
