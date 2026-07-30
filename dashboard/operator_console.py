"""
operator_console.py
Phase DO-002/DO-003/DO-004/DO-005 — Streamlit operator-console pages.

DO-005 adds confirmation-protected STOP_TRITON (maps to DO-004 cooperative shutdown).
Restart/Arm/Approve/Execute remain unavailable.
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
    """Protected START (DO-003), STOP (DO-005), and RESTART (DO-006) orchestration."""
    from services.operator_runtime_controller import (
        ACTION_START,
        execute_start_triton,
        start_eligibility,
    )
    from services.operator_stop_action import stop_eligibility
    from services.operator_restart_action import restart_eligibility

    eligibility = start_eligibility()
    stop_el = stop_eligibility()
    restart_el = restart_eligibility()
    st.subheader("Runtime controls")
    st.caption(
        f"Start eligibility: **{eligibility.get('decision')}** · "
        f"Stop eligibility: **{stop_el.get('decision')}** · "
        f"Restart eligibility: **{restart_el.get('decision')}** · "
        f"Paper arm (not modified by Start/Stop/Restart): **{eligibility.get('paper_arm_status')}**"
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
    if "stop_confirm_pending" not in st.session_state:
        st.session_state["stop_confirm_pending"] = False
    if "stop_preflight_token" not in st.session_state:
        st.session_state["stop_preflight_token"] = None
    if "stop_request_nonce" not in st.session_state:
        st.session_state["stop_request_nonce"] = None
    if "stop_preflight_expires_at" not in st.session_state:
        st.session_state["stop_preflight_expires_at"] = None
    if "stop_last_result" not in st.session_state:
        st.session_state["stop_last_result"] = None
    if "stop_in_progress" not in st.session_state:
        st.session_state["stop_in_progress"] = False
    if "restart_confirm_pending" not in st.session_state:
        st.session_state["restart_confirm_pending"] = False
    if "restart_preflight_token" not in st.session_state:
        st.session_state["restart_preflight_token"] = None
    if "restart_request_nonce" not in st.session_state:
        st.session_state["restart_request_nonce"] = None
    if "restart_preflight_expires_at" not in st.session_state:
        st.session_state["restart_preflight_expires_at"] = None
    if "restart_last_result" not in st.session_state:
        st.session_state["restart_last_result"] = None
    if "restart_in_progress" not in st.session_state:
        st.session_state["restart_in_progress"] = False

    restart_busy = bool(
        st.session_state.get("restart_in_progress")
        or st.session_state.get("restart_confirm_pending")
        or restart_el.get("restart_guard_active")
    )
    stop_busy = bool(
        st.session_state.get("stop_in_progress")
        or st.session_state.get("stop_confirm_pending")
        or stop_el.get("shutdown_guard_active")
        or (stop_el.get("shutdown_active") or {}).get("active")
        or eligibility.get("decision") == "START_REFUSED_SHUTDOWN_IN_PROGRESS"
        or restart_busy
    )
    start_busy = bool(
        st.session_state.get("start_in_progress")
        or st.session_state.get("start_confirm_pending")
        or eligibility.get("guard_active")
        or restart_busy
    )

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

    stop_last = st.session_state.get("stop_last_result")
    if stop_last:
        sres = str(stop_last.get("result") or "")
        smsg = stop_last.get("safe_message") or sres
        if sres in ("STOP_COMPLETED", "STOP_ALREADY_STOPPED"):
            st.success(smsg)
        elif sres in ("STOP_PARTIAL_COMPLETION", "STOP_TIMED_OUT", "STOP_IN_PROGRESS"):
            st.warning(smsg)
        elif sres == "STOP_AUDIT_FAILED":
            st.error(smsg)
        else:
            st.error(smsg)
        with st.expander("Stop technical evidence", expanded=False):
            st.json(
                {
                    "result": stop_last.get("result"),
                    "ui_state": stop_last.get("ui_state"),
                    "action_id": stop_last.get("action_id"),
                    "request_id": stop_last.get("request_id"),
                    "target_components": stop_last.get("target_components"),
                    "acknowledgement_results": stop_last.get("acknowledgement_results"),
                    "process_exit_results": stop_last.get("process_exit_results"),
                    "lock_release_results": stop_last.get("lock_release_results"),
                    "forced_termination_used": stop_last.get("forced_termination_used"),
                    "runtime_locks_deleted_by_controller": stop_last.get(
                        "runtime_locks_deleted_by_controller"
                    ),
                    "live_execution_blocked_after": stop_last.get("live_execution_blocked_after"),
                    "paper_arm_mutated": stop_last.get("paper_arm_mutated"),
                    "error_code": stop_last.get("error_code"),
                }
            )

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        if eligibility.get("already_running"):
            st.info("TRITON is already running")
        elif stop_busy:
            st.button("Start TRITON", key="start_triton_disabled_stop_busy", disabled=True)
            st.caption(
                "Start unavailable while Stop or Restart is active or unresolved."
                if restart_busy
                else "Start unavailable while Stop is active or unresolved."
            )
        elif start_busy:
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

    restart_last = st.session_state.get("restart_last_result")
    if restart_last:
        rres = str(restart_last.get("result") or "")
        rmsg = restart_last.get("safe_message") or rres
        if rres == "RESTART_COMPLETED":
            st.success(rmsg)
        elif rres in (
            "RESTART_PARTIAL_COMPLETION",
            "RESTART_TIMED_OUT",
            "RESTART_IN_PROGRESS",
            "RESTART_STOP_IN_PROGRESS",
        ):
            st.warning(rmsg)
        else:
            st.error(rmsg)
        stages = list(restart_last.get("stages") or [])
        stage_names = [str(s.get("stage") or s) if isinstance(s, dict) else str(s) for s in stages]
        stop_done = any(s in stage_names for s in ("STOP_COMPLETED", "STOP_NOT_REQUIRED")) or str(
            restart_last.get("stop_result") or ""
        ) in (
            "STOP_COMPLETED",
            "STOP_ALREADY_STOPPED",
        )
        start_eligible = bool(restart_last.get("fresh_start_classification")) or any(
            s in stage_names for s in ("START_ELIGIBILITY_CONFIRMED", "START_FRESH_CLASSIFICATION")
        )
        start_progress = any(s.startswith("START_") for s in stage_names) or bool(
            restart_last.get("start_result")
        )
        st.caption("Restart stages (Stop and Start are separate verified transactions)")
        st.caption(
            "Stop-stage progress: "
            + (
                ", ".join(s for s in stage_names if "STOP" in s)
                or str(restart_last.get("stop_result") or "not started")
            )
        )
        st.caption(
            "Verified Stop completion: "
            + ("yes" if stop_done else f"no ({restart_last.get('stop_result') or 'incomplete'})")
        )
        st.caption(
            "Fresh Start eligibility: "
            + (
                str(restart_last.get("fresh_start_classification") or "evaluated")
                if start_eligible
                else "not reached"
            )
        )
        st.caption(
            "Start-stage progress: "
            + (
                ", ".join(s for s in stage_names if "START" in s)
                or str(restart_last.get("start_result") or "not started")
            )
        )
        st.caption(
            "Final runtime verification: "
            + str(rres)
            + (
                " (replacement process/lock ownership verified)"
                if rres == "RESTART_COMPLETED"
                else " (not claimed complete until Start postconditions pass)"
            )
        )
        with st.expander("Restart technical evidence", expanded=False):
            st.json(
                {
                    "result": restart_last.get("result"),
                    "action_id": restart_last.get("action_id"),
                    "stop_required": restart_last.get("stop_required"),
                    "stop_result": restart_last.get("stop_result"),
                    "stop_action_id": restart_last.get("stop_action_id"),
                    "fresh_start_classification": restart_last.get("fresh_start_classification"),
                    "start_result": restart_last.get("start_result"),
                    "start_transaction_id": restart_last.get("start_transaction_id"),
                    "stages": restart_last.get("stages"),
                    "forced_termination_used": restart_last.get("forced_termination_used"),
                    "live_execution_blocked_after": restart_last.get(
                        "live_execution_blocked_after"
                    ),
                    "paper_arm_mutated": restart_last.get("paper_arm_mutated"),
                    "error_code": restart_last.get("error_code"),
                }
            )

    with ctrl2:
        _render_stop_control_column(
            stop_el, start_busy=start_busy or restart_busy, restart_busy=restart_busy
        )

    with ctrl3:
        _render_restart_control_column(
            restart_el, start_busy=start_busy, stop_busy=stop_busy and not restart_busy
        )

    st.caption(
        f"allowed_start={eligibility.get('allowed_action')} · "
        f"allowed_stop={stop_el.get('allowed_dashboard_action')} · "
        f"allowed_restart={restart_el.get('allowed_dashboard_action')} · "
        f"stop_enabled={stop_el.get('stop_button_enabled')} · "
        f"restart_enabled={restart_el.get('restart_button_enabled')}"
    )
    st.info(
        "Restart TRITON uses two separately verified transactions: a cooperative Stop followed "
        "by a fresh protected Start. Restart does not force-terminate processes, delete locks, "
        "bypass Start eligibility, arm paper execution, or enable live execution."
    )
    st.info(
        "Stop TRITON uses verified cooperative shutdown. Active bounded work is allowed to "
        "reach its documented safe boundary. TRITON is never force-terminated by this control."
    )

    _render_cooperative_shutdown_status()


def _render_stop_control_column(
    stop_el: Dict[str, Any], *, start_busy: bool, restart_busy: bool = False
) -> None:
    from services.operator_stop_action import (
        ACTION_STOP,
        CONFIRMATION_PHRASE,
        execute_stop_triton,
        issue_stop_preflight,
    )

    if start_busy or restart_busy:
        st.button("Stop TRITON Safely", key="stop_triton_disabled_start_busy", disabled=True)
        st.caption(
            "Stop unavailable while Restart is active."
            if restart_busy
            else "Stop unavailable while Start is active."
        )
        return

    if st.session_state.get("stop_in_progress"):
        st.button("Stop TRITON Safely", key="stop_triton_disabled_busy", disabled=True)
        st.caption("Stop submission in progress… cooperative shutdown does not force-terminate.")
        return

    if stop_el.get("decision") == "STOP_ALREADY_STOPPED":
        st.info("TRITON is already stopped")
        st.button("Stop TRITON Safely", key="stop_triton_disabled_already", disabled=True)
        return

    if not stop_el.get("stop_button_enabled") and not st.session_state.get("stop_confirm_pending"):
        st.button("Stop TRITON Safely", key="stop_triton_disabled_unsafe", disabled=True)
        st.caption(
            f"Stop disabled: {stop_el.get('decision')} — "
            f"{(stop_el.get('explanation') or '')[:120]}"
        )
        return

    if not st.session_state.get("stop_confirm_pending"):
        if st.button("Stop TRITON Safely", key="stop_triton_request", type="primary"):
            import uuid

            pre = issue_stop_preflight()
            if not pre.get("ok") or not pre.get("stop_button_enabled"):
                st.session_state["stop_last_result"] = {
                    "result": pre.get("decision") or "STOP_REFUSED",
                    "safe_message": pre.get("explanation")
                    or "Stop is not currently eligible. Confirmation was not opened.",
                }
                st.rerun()
                return
            st.session_state["stop_confirm_pending"] = True
            st.session_state["stop_preflight_token"] = pre.get("preflight_token")
            st.session_state["stop_request_nonce"] = pre.get("preflight_nonce") or str(uuid.uuid4())
            st.session_state["stop_preflight_expires_at"] = pre.get("expires_at")
            st.session_state["stop_preflight_decision"] = pre.get("decision")
            st.session_state["stop_preflight_targets"] = pre.get("target_components")
            st.rerun()
        st.caption(
            "Requests a cooperative shutdown. Active bounded work will finish at its "
            "documented safe boundary. No process will be force-terminated."
        )
        return

    st.warning(
        "Requests a cooperative shutdown. Active bounded work will finish at its documented "
        "safe boundary. No process will be force-terminated. Live execution remains blocked. "
        "This does **not** create or extend paper-arm authorization."
    )
    st.caption(
        f"Preflight expires: {st.session_state.get('stop_preflight_expires_at')} · "
        f"decision={st.session_state.get('stop_preflight_decision')} · "
        f"targets={st.session_state.get('stop_preflight_targets')}"
    )
    checked = st.checkbox(
        "I understand that TRITON will finish its current safe unit of work before exiting "
        "and that this action will not force-terminate any process.",
        key="stop_confirm_checkbox",
    )
    phrase = st.text_input(
        "Type exactly: STOP TRITON",
        key="stop_confirm_phrase",
    )
    can_submit = (
        checked
        and phrase == CONFIRMATION_PHRASE
        and bool(st.session_state.get("stop_preflight_token"))
        and not start_busy
    )
    conf_a, conf_b = st.columns(2)
    with conf_a:
        if st.button(
            "Confirm Stop",
            key="stop_triton_confirm",
            type="primary",
            disabled=not can_submit,
        ):
            st.session_state["stop_in_progress"] = True
            st.session_state["stop_confirm_pending"] = False
            with st.spinner("Requesting cooperative shutdown and validating…"):
                result = execute_stop_triton(
                    action=ACTION_STOP,
                    confirmation_phrase=phrase,
                    confirmation_checked=True,
                    preflight_token=st.session_state.get("stop_preflight_token"),
                    request_nonce=st.session_state.get("stop_request_nonce"),
                    source="operator_console",
                )
            st.session_state["stop_last_result"] = result
            st.session_state["stop_in_progress"] = False
            st.session_state["stop_preflight_token"] = None
            st.session_state["stop_request_nonce"] = None
            st.rerun()
    with conf_b:
        if st.button("Cancel", key="stop_triton_cancel"):
            st.session_state["stop_confirm_pending"] = False
            st.session_state["stop_preflight_token"] = None
            st.session_state["stop_request_nonce"] = None
            st.rerun()


def _render_restart_control_column(
    restart_el: Dict[str, Any], *, start_busy: bool, stop_busy: bool
) -> None:
    from services.operator_restart_action import (
        ACTION_RESTART,
        CONFIRMATION_PHRASE,
        execute_restart_triton,
        issue_restart_preflight,
    )

    if start_busy or stop_busy:
        st.button("Restart TRITON Safely", key="restart_triton_disabled_busy_peer", disabled=True)
        st.caption("Restart unavailable while Start or Stop is active.")
        return

    if st.session_state.get("restart_in_progress"):
        st.button("Restart TRITON Safely", key="restart_triton_disabled_busy", disabled=True)
        st.caption("Restart in progress… Stop must fully complete before a fresh Start evaluation.")
        return

    if not restart_el.get("restart_button_enabled") and not st.session_state.get(
        "restart_confirm_pending"
    ):
        st.button("Restart TRITON Safely", key="restart_triton_disabled_unsafe", disabled=True)
        st.caption(
            f"Restart disabled: {restart_el.get('decision')} — "
            f"{(restart_el.get('explanation') or '')[:120]}"
        )
        return

    if not st.session_state.get("restart_confirm_pending"):
        if st.button("Restart TRITON Safely", key="restart_triton_request", type="primary"):
            import uuid

            pre = issue_restart_preflight()
            if not pre.get("ok") or not pre.get("restart_button_enabled"):
                st.session_state["restart_last_result"] = {
                    "result": pre.get("decision") or "RESTART_REFUSED",
                    "safe_message": pre.get("explanation")
                    or "Restart is not currently eligible. Confirmation was not opened.",
                }
                st.rerun()
                return
            st.session_state["restart_confirm_pending"] = True
            st.session_state["restart_preflight_token"] = pre.get("preflight_token")
            st.session_state["restart_request_nonce"] = pre.get("preflight_nonce") or str(
                uuid.uuid4()
            )
            st.session_state["restart_preflight_expires_at"] = pre.get("expires_at")
            st.session_state["restart_preflight_decision"] = pre.get("decision")
            st.session_state["restart_preflight_targets"] = pre.get("target_components")
            st.session_state["restart_stop_required"] = pre.get("stop_required")
            st.rerun()
        st.caption(
            "Verified cooperative Stop, then a fresh protected Start. "
            "Does not force-terminate or bypass Start eligibility."
        )
        if restart_el.get("unhealthy_verified"):
            st.caption("Verified unhealthy process: cooperative Stop may take longer.")
        return

    st.warning(
        "Restart TRITON performs a verified cooperative Stop followed by a fresh protected Start. "
        "Active bounded work may first reach its documented safe boundary. Restart will not "
        "continue unless Stop fully completes and a new Start evaluation passes."
    )
    st.caption(
        f"Preflight expires: {st.session_state.get('restart_preflight_expires_at')} · "
        f"decision={st.session_state.get('restart_preflight_decision')} · "
        f"stop_required={st.session_state.get('restart_stop_required')} · "
        f"targets={st.session_state.get('restart_preflight_targets')}"
    )
    checked = st.checkbox(
        "I understand Restart uses two separately verified transactions and will not "
        "force-terminate processes, delete locks, or enable live execution.",
        key="restart_confirm_checkbox",
    )
    phrase = st.text_input(
        "Type exactly: RESTART TRITON",
        key="restart_confirm_phrase",
    )
    can_submit = (
        checked
        and phrase == CONFIRMATION_PHRASE
        and bool(st.session_state.get("restart_preflight_token"))
        and not start_busy
        and not stop_busy
    )
    conf_a, conf_b = st.columns(2)
    with conf_a:
        if st.button(
            "Confirm Restart",
            key="restart_triton_confirm",
            type="primary",
            disabled=not can_submit,
        ):
            st.session_state["restart_in_progress"] = True
            st.session_state["restart_confirm_pending"] = False
            with st.spinner("Orchestrating cooperative Stop then fresh Start…"):
                result = execute_restart_triton(
                    action=ACTION_RESTART,
                    confirmation_phrase=phrase,
                    confirmation_checked=True,
                    preflight_token=st.session_state.get("restart_preflight_token"),
                    request_nonce=st.session_state.get("restart_request_nonce"),
                    source="operator_console",
                )
            st.session_state["restart_last_result"] = result
            st.session_state["restart_in_progress"] = False
            st.session_state["restart_preflight_token"] = None
            st.session_state["restart_request_nonce"] = None
            st.rerun()
    with conf_b:
        if st.button("Cancel", key="restart_triton_cancel"):
            st.session_state["restart_confirm_pending"] = False
            st.session_state["restart_preflight_token"] = None
            st.session_state["restart_request_nonce"] = None
            st.rerun()


def _render_cooperative_shutdown_status() -> None:
    """Read-only DO-004/DO-005 cooperative shutdown visibility."""
    from services.operator_shutdown_controller import shutdown_eligibility
    from services.operator_stop_action import stop_eligibility

    status = shutdown_eligibility()
    stop_el = stop_eligibility()
    st.subheader("Cooperative Shutdown / Protected Stop")
    st.info(stop_el.get("safe_message") or status.get("safe_message") or "")
    st.caption(
        f"Stop decision: **{stop_el.get('decision')}** · "
        f"shutdown decision: **{status.get('decision')}** · "
        f"targets={stop_el.get('target_components')} · "
        f"protected_stop={stop_el.get('protected_stop_exposed')} · "
        f"restart={stop_el.get('restart_exposed')}"
    )
    st.write(stop_el.get("explanation") or status.get("explanation") or "")
    st.write(stop_el.get("recommended_action") or "")

    req = status.get("active_request") or {}
    active = status.get("shutdown_active") or {}
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Request / guard**")
        st.json(
            {
                "shutdown_active": active.get("active"),
                "reasons": active.get("reasons"),
                "request_id": req.get("request_id"),
                "request_state": req.get("state"),
                "requested_at": req.get("requested_at"),
                "expires_at": req.get("expires_at"),
                "target_components": req.get("target_components"),
                "start_guard_active": status.get("start_guard_active"),
                "shutdown_guard_active": status.get("shutdown_guard_active"),
            }
        )
    with col_b:
        st.markdown("**Acknowledgements / progress**")
        st.json(
            {
                "continuous": status.get("ack_continuous") or {},
                "watchdog": status.get("ack_watchdog") or {},
                "last_stop_ui_state": (st.session_state.get("stop_last_result") or {}).get(
                    "ui_state"
                ),
            }
        )

    with st.expander("Shutdown / Stop decision evidence", expanded=False):
        st.json(
            {
                "stop_decision": stop_el.get("decision"),
                "shutdown_decision": status.get("decision"),
                "target_components": stop_el.get("target_components"),
                "live_execution_blocked": stop_el.get("live_execution_blocked"),
                "allowed_dashboard_action": stop_el.get("allowed_dashboard_action"),
                "backend_operation": stop_el.get("backend_operation"),
                "dashboard_target_policy": stop_el.get("dashboard_target_policy"),
                "classifications": {
                    "continuous": (stop_el.get("classifications") or {}).get("continuous"),
                    "watchdog": (stop_el.get("classifications") or {}).get("watchdog"),
                },
            }
        )


def render_operations_page() -> None:
    from services.operator_console_status import build_operator_status

    st.title("Operations")
    st.caption(
        "Operator console · Protected Start · Protected Stop (cooperative) · "
        "Live execution remains blocked · Restart uses verified Stop then fresh Start."
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

    st.subheader("Cooperative shutdown evidence (DO-004)")
    from services.operator_shutdown_controller import (
        read_recent_shutdown_actions,
        shutdown_eligibility,
    )

    shut = shutdown_eligibility()
    st.json(
        {
            "decision": shut.get("decision"),
            "explanation": shut.get("explanation"),
            "target_components": shut.get("target_components"),
            "protocol_implemented": shut.get("protocol_implemented"),
            "clickable_stop_exposed": shut.get("clickable_stop_exposed"),
            "clickable_restart_exposed": shut.get("clickable_restart_exposed"),
            "live_execution_blocked": shut.get("live_execution_blocked"),
            "start_guard_active": shut.get("start_guard_active"),
            "shutdown_guard_active": shut.get("shutdown_guard_active"),
            "shutdown_active": shut.get("shutdown_active"),
            "active_request": shut.get("active_request"),
            "ack_continuous": shut.get("ack_continuous"),
            "ack_watchdog": shut.get("ack_watchdog"),
            "recommended_action": shut.get("recommended_action"),
            "allowed_backend_action": shut.get("allowed_backend_action"),
            "denied_actions": shut.get("denied_actions"),
        }
    )
    with st.expander("Continuous / watchdog identity for shutdown"):
        st.json(
            {
                "continuous": (shut.get("classifications") or {}).get("continuous"),
                "watchdog": (shut.get("classifications") or {}).get("watchdog"),
            }
        )
    with st.expander("Recent sanitized shutdown-action audit"):
        st.json(read_recent_shutdown_actions(limit=15))
    st.subheader("Protected Stop evidence (DO-005)")
    from services.operator_stop_action import read_recent_stop_actions, stop_eligibility

    stop_el = stop_eligibility()
    st.json(
        {
            "decision": stop_el.get("decision"),
            "explanation": stop_el.get("explanation"),
            "target_components": stop_el.get("target_components"),
            "stop_button_enabled": stop_el.get("stop_button_enabled"),
            "protected_stop_exposed": stop_el.get("protected_stop_exposed"),
            "restart_exposed": stop_el.get("restart_exposed"),
            "confirmation_phrase_required": stop_el.get("confirmation_phrase_required"),
            "preflight_token_required": stop_el.get("preflight_token_required"),
            "dashboard_target_policy": stop_el.get("dashboard_target_policy"),
            "backend_operation": stop_el.get("backend_operation"),
            "live_execution_blocked": stop_el.get("live_execution_blocked"),
            "start_guard_active": stop_el.get("start_guard_active"),
            "shutdown_guard_active": stop_el.get("shutdown_guard_active"),
            "shutdown_active": stop_el.get("shutdown_active"),
            "recommended_action": stop_el.get("recommended_action"),
            "token_presented": False,
        }
    )
    with st.expander("Recent sanitized stop-action audit"):
        st.json(read_recent_stop_actions(limit=15))
    st.subheader("Protected Restart evidence (DO-006)")
    from services.operator_restart_action import (
        read_recent_restart_actions,
        read_restart_transaction,
        restart_eligibility,
    )

    restart_el = restart_eligibility()
    st.json(
        {
            "decision": restart_el.get("decision"),
            "explanation": restart_el.get("explanation"),
            "restart_button_enabled": restart_el.get("restart_button_enabled"),
            "stop_required": restart_el.get("stop_required"),
            "target_components": restart_el.get("target_components"),
            "confirmation_phrase_required": restart_el.get("confirmation_phrase_required"),
            "preflight_token_required": restart_el.get("preflight_token_required"),
            "allowed_dashboard_action": restart_el.get("allowed_dashboard_action"),
            "stop_backend_operation": restart_el.get("stop_backend_operation"),
            "start_backend_operation": restart_el.get("start_backend_operation"),
            "live_execution_blocked": restart_el.get("live_execution_blocked"),
            "start_guard_active": restart_el.get("start_guard_active"),
            "shutdown_guard_active": restart_el.get("shutdown_guard_active"),
            "restart_guard_active": restart_el.get("restart_guard_active"),
            "recommended_action": restart_el.get("recommended_action"),
            "token_presented": False,
            "active_transaction": {
                k: v for k, v in (read_restart_transaction() or {}).items() if k not in ("final",)
            },
        }
    )
    with st.expander("Recent sanitized restart-action audit"):
        st.json(read_recent_restart_actions(limit=15))
    st.caption(
        "Diagnostics is read-only. No repair, delete, kill, Stop submit, Restart submit, or override buttons are provided."
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
            "allowed_runtime_actions": ["START_TRITON", "STOP_TRITON", "RESTART_TRITON"],
            "live_execution_control_exposed": False,
            "restart_available": True,
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
