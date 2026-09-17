"""
operator_console.py
Phase DO-002/DO-003/DO-004/DO-005/DO-006/DO-007 + UIR-001 — Streamlit operator-console pages.

DO-005 adds confirmation-protected STOP_TRITON (maps to DO-004 cooperative shutdown).
UIR-001 presentation foundation + UIR-002 Reports (read-only reconciled reporting).
Runtime-control callbacks and eligibility decisions are unchanged.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

_OPERATOR_CONSOLE_ROOT = Path(__file__).resolve().parents[1]

OPERATOR_PAGES = [
    "Operations",
    "Activity",
    "Portfolio",
    "Reports",
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


_HELD_SKIP_REASONS = frozenset(
    {
        "INVALID_PROCESSED_SESSION_BAR",
        "STALE_PROCESSED_SESSION_BAR",
        "MISSING_PROCESSED_SESSION_BAR",
    }
)
_INDEPENDENT_RISK_ACTIONS = frozenset({"FORCE_EXIT", "TRIM_PRIORITY"})
_PLAN_BLOCK_CODES = frozenset({"STALE_LIFECYCLE_NO_POSITION", "LIFECYCLE_SKIPPED_HELD"})


def collect_held_risk_plan_blocked_alerts(*, root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read dated skip + plan evidence for held names whose risk cannot reach F.

    Rendering only. Does not invoke Stop, Start, Restart, or any trading control.
    Does not imply a SELL was placed.
    """
    base = Path(root) if root is not None else _OPERATOR_CONSOLE_ROOT
    results = base / "data" / "results"
    diag_path = results / "signals_asof_diagnostics.json"
    pos_path = results / "positions_snapshot.csv"
    overlay_path = results / "performance_risk_overlay.csv"
    plan_path = results / "manage_positions_plan.csv"
    if not diag_path.is_file() or not pos_path.is_file():
        return []

    try:
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    session = str(diag.get("session_asof_date") or diag.get("effective_asof_date") or "")
    skips: Dict[str, Dict[str, Any]] = {}
    for rec in diag.get("per_symbol_skips") or []:
        if not isinstance(rec, dict):
            continue
        reason = str(rec.get("reason") or "")
        if reason not in _HELD_SKIP_REASONS:
            continue
        sym = str(rec.get("symbol") or rec.get("ticker") or "").strip().upper()
        if not sym or sym.startswith("^"):
            continue
        skips[sym] = rec

    held: Dict[str, Dict[str, str]] = {}
    with pos_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            tick = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
            try:
                qty = float(row.get("qty") or row.get("quantity") or 0)
            except Exception:
                qty = 0.0
            if tick and qty > 1e-9:
                held[tick] = row

    overlay: Dict[str, str] = {}
    if overlay_path.is_file():
        with overlay_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                tick = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
                overlay[tick] = str(row.get("risk_flag") or "")

    plans: Dict[str, Dict[str, str]] = {}
    if plan_path.is_file():
        with plan_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                tick = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
                if tick:
                    plans[tick] = row

    alerts: List[Dict[str, Any]] = []
    for sym, skip in skips.items():
        if sym not in held:
            continue
        flag = overlay.get(sym, "")
        risk_hits = [a for a in _INDEPENDENT_RISK_ACTIONS if a in flag.upper()]
        if not risk_hits:
            continue
        blocked_action = risk_hits[0]
        plan = plans.get(sym) or {}
        plan_code = str(plan.get("reason_code") or "")
        selected = str(plan.get("selected_for_execution") or "").lower() in {"true", "1"}
        final_action = str(plan.get("final_action") or "").upper()
        reached = selected and final_action in {"EXIT", "TRIM", "SELL"}
        if reached:
            continue
        if plan and plan_code not in _PLAN_BLOCK_CODES and final_action not in {"", "NONE"}:
            # Plan exists with some other non-exit action; still blocked if not selected.
            if selected:
                continue
        evidence_date = str(
            skip.get("expected_session_date") or skip.get("actual_latest_bar_date") or session
        )
        skip_reason = str(skip.get("reason") or "")
        plan_reason = plan_code or "NO_PROTECTED_PLAN_ROW"
        alerts.append(
            {
                "symbol": sym,
                "evidence_date": evidence_date,
                "skip_reason": skip_reason,
                "blocked_action": blocked_action,
                "plan_reason": plan_reason,
                "operator_review_required": True,
                "sell_placed": False,
                "message": (
                    f"{sym}: processed evidence dated {evidence_date} is "
                    f"{'stale' if 'STALE' in skip_reason else 'invalid'} "
                    f"({skip_reason}). Independent {blocked_action} cannot reach "
                    f"the protected plan ({plan_reason}). Operator review required. "
                    f"No SELL was placed."
                ),
            }
        )
    return alerts


def _render_held_risk_plan_blocked_alerts(*, root: Optional[Path] = None) -> None:
    alerts = collect_held_risk_plan_blocked_alerts(root=root)
    if not alerts:
        return
    st.subheader("Held-risk plan blocked")
    for alert in alerts:
        st.warning(alert["message"])
    st.caption(
        "These names are currently held, excluded for dated processed-bar evidence, "
        "and an independent risk action cannot reach the protected plan. "
        "This is not a SELL confirmation."
    )


def operator_page_list() -> List[str]:
    return list(OPERATOR_PAGES)


def _status_color(status: str) -> str:
    s = str(status or "").upper()
    if s in ("RUNNING", "ARMED", "FRESH", "SUCCESS"):
        return "🟢"
    if s in ("DEGRADED", "EXPIRED", "STALE", "WARN"):
        return "🟡"
    if s in ("STOPPED", "MISSING"):
        return "⚪"
    # Live-execution BLOCKED is a required safety lock — never green.
    if s in ("CRITICAL", "INVALID", "BLOCKED", "LIVE EXECUTION BLOCKED") or "NOT BLOCKED" in s:
        return "🔴"
    return "⚫"


def _render_start_controls(status_snapshot: dict | None = None) -> None:
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
    from dashboard.operator_ui import section_header

    section_header(
        "Protected runtime controls",
        "Confirmation required · eligibility-gated · existing callbacks unchanged",
    )
    st.markdown(
        '<div class="triton-ctrl-legend">'
        '<span class="start">Start</span>'
        '<span class="stop">Stop</span>'
        '<span class="restart">Restart</span>'
        '<span class="unavailable">Unavailable / muted when ineligible</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Start: **{eligibility.get('decision')}** · "
        f"Stop: **{stop_el.get('decision')}** · "
        f"Restart: **{restart_el.get('decision')}**"
    )
    if eligibility.get("explanation"):
        st.caption(str(eligibility.get("explanation")))

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

    # Durable controller guards/transactions are authoritative; session flags are presentation.
    # Always attempt stale-guard reconciliation (no-op when no expired guard exists).
    try:
        from services.operator_runtime_controller import reconcile_stale_start_transaction

        recon = reconcile_stale_start_transaction()
        if recon.get("reconciled"):
            st.session_state["start_in_progress"] = False
            st.session_state["start_confirm_pending"] = False
            eligibility = start_eligibility()
    except Exception:
        pass
    if st.session_state.get("start_in_progress") and not eligibility.get("guard_active"):
        # Presentation latch without an active authoritative transaction.
        st.session_state["start_in_progress"] = False

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
    # Confirmation-pending must NOT be treated as "in progress" for Start column rendering,
    # or Confirm Start becomes unreachable after the first click.
    start_confirm_open = bool(st.session_state.get("start_confirm_pending"))
    start_executing = bool(
        st.session_state.get("start_in_progress") or eligibility.get("guard_active")
    )
    start_busy = bool(start_executing or start_confirm_open or restart_busy)

    # UIR-003C: action banners reconciled against current process evidence.
    # Never present a Start "both are running" latch as current truth after Stop.
    from dashboard import operator_ui as _oui_actions
    from services.operator_console_status import build_operator_status as _build_ops_status

    _status_for_actions = (
        status_snapshot if isinstance(status_snapshot, dict) else _build_ops_status()
    )
    _cont_a = _status_for_actions.get("continuous") or {}
    _wd_a = _status_for_actions.get("watchdog") or {}
    _live_a = _status_for_actions.get("live_execution") or {}
    action_view = _oui_actions.present_current_runtime_action_message(
        overall_status=_status_for_actions.get("overall_status"),
        continuous_process_count=_cont_a.get("process_count"),
        watchdog_process_count=_wd_a.get("process_count"),
        continuous_lock_exists=bool(_cont_a.get("lock_exists")),
        watchdog_lock_exists=bool(_wd_a.get("lock_exists")),
        live_execution_blocked=bool(_live_a.get("blocked")),
        start_last_result=st.session_state.get("start_last_result"),
        stop_last_result=st.session_state.get("stop_last_result"),
        restart_last_result=st.session_state.get("restart_last_result"),
    )
    # Drop contradicted Start latch from session so refresh cannot resurrect it.
    if action_view.get("start_suppressed") and st.session_state.get("start_last_result"):
        st.session_state["start_result_history"] = st.session_state.get("start_last_result")
        st.session_state["start_last_result"] = None

    current_action = action_view.get("current")
    if current_action:
        level = str(current_action.get("level") or "info")
        msg = current_action.get("message") or current_action.get("result") or ""
        if level == "success":
            st.success(msg)
        elif level == "warning":
            st.warning(msg)
        elif level == "error":
            st.error(msg)
        else:
            st.info(msg)

    last = st.session_state.get("start_last_result") or st.session_state.get("start_result_history")
    if last and (
        not action_view.get("start_suppressed")
        or st.session_state.get("start_result_history") is last
    ):
        with st.expander(
            "Start technical evidence"
            + (
                " (historical — not current runtime status)"
                if action_view.get("start_suppressed")
                else ""
            ),
            expanded=False,
        ):
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
                    "presentation_note": (
                        "Suppressed as current status because process evidence disagrees"
                        if action_view.get("start_suppressed")
                        else None
                    ),
                }
            )

    stop_last = st.session_state.get("stop_last_result")
    if stop_last:
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

    hist = list(action_view.get("historical") or [])
    if hist:
        with st.expander("Prior runtime action messages (historical)", expanded=False):
            st.caption("Historical only — not current runtime status.")
            st.json(hist)

    st.caption(
        "Confirmation required before Start/Stop/Restart submits. "
        "Blocking reasons appear under each action when unavailable."
    )

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        with st.container(key="triton_ctrl_start"):
            st.caption("Action: **Start** · confirmation required")
            if eligibility.get("already_running"):
                st.info("TRITON is already running")
            elif stop_busy:
                st.button("Start TRITON", key="start_triton_disabled_stop_busy", disabled=True)
                st.caption(
                    "Start unavailable while Stop or Restart is active or unresolved."
                    if restart_busy
                    else "Start unavailable while Stop is active or unresolved."
                )
            elif start_executing:
                st.button("Start TRITON", key="start_triton_disabled_busy", disabled=True)
                st.caption("Start action in progress…")
            elif not eligibility.get("start_button_enabled") and not start_confirm_open:
                st.button("Start TRITON", key="start_triton_disabled_unsafe", disabled=True)
                st.caption("Start disabled: unsafe, ambiguous, or not eligible.")
            else:
                if not start_confirm_open:
                    if st.button("Start TRITON", key="start_triton_request", type="primary"):
                        import uuid

                        st.session_state["start_confirm_pending"] = True
                        st.session_state["start_idempotency_key"] = str(uuid.uuid4())
                        st.rerun()
                else:
                    targets = eligibility.get("components_to_start") or []
                    preserved = eligibility.get("preserved_components") or []
                    st.warning(
                        "This starts TRITON’s local watchdog and continuous runtime in its existing "
                        "paper-only configuration. Live execution will remain blocked. "
                        "This does **not** create or extend paper-arm authorization."
                    )
                    if targets or preserved:
                        st.caption(
                            f"Will start: {targets or 'none'} · Will preserve: {preserved or 'none'}"
                        )
                    conf_a, conf_b = st.columns(2)
                    with conf_a:
                        if st.button("Confirm Start", key="start_triton_confirm", type="primary"):
                            st.session_state["start_in_progress"] = True
                            st.session_state["start_confirm_pending"] = False
                            result = {
                                "result": "START_FAILED",
                                "safe_message": "Start did not complete a terminal controller result.",
                            }
                            try:
                                with st.spinner("Starting and validating TRITON…"):
                                    result = execute_start_triton(
                                        action=ACTION_START,
                                        idempotency_key=st.session_state.get(
                                            "start_idempotency_key"
                                        ),
                                        source="operator_console",
                                        operator_context={"ui": "operations"},
                                    )
                            except Exception as exc:
                                result = {
                                    "result": "START_FAILED",
                                    "error_code": "UI_START_EXCEPTION",
                                    "safe_message": (
                                        "Start failed in the operator console before a durable "
                                        f"controller result was recorded ({type(exc).__name__})."
                                    ),
                                }
                            finally:
                                st.session_state["start_last_result"] = result
                                st.session_state["start_in_progress"] = False
                                st.session_state["start_idempotency_key"] = None
                                # Start becomes current; demote prior Stop/Restart banners.
                                if str(result.get("result") or "") in (
                                    "STARTED",
                                    "ALREADY_RUNNING",
                                ):
                                    if st.session_state.get("stop_last_result"):
                                        st.session_state["stop_result_history"] = (
                                            st.session_state.get("stop_last_result")
                                        )
                                    st.session_state["stop_last_result"] = None
                                    st.session_state["restart_last_result"] = None
                            st.rerun()
                    with conf_b:
                        if st.button("Cancel", key="start_triton_cancel"):
                            st.session_state["start_confirm_pending"] = False
                            st.session_state["start_idempotency_key"] = None
                            st.rerun()

    restart_last = st.session_state.get("restart_last_result")
    # Banner text is owned by present_current_runtime_action_message; keep stage captions/evidence.
    if restart_last:
        rres = str(restart_last.get("result") or "")
        rmsg = restart_last.get("safe_message") or rres
        current_src = str((action_view.get("current") or {}).get("source") or "")
        if current_src != "restart":
            # Non-current restart outcomes still need operator visibility when failed/partial.
            if rres == "RESTART_COMPLETED":
                pass
            elif rres in (
                "RESTART_PARTIAL_COMPLETION",
                "RESTART_TIMED_OUT",
                "RESTART_IN_PROGRESS",
                "RESTART_STOP_IN_PROGRESS",
            ):
                st.warning(rmsg)
            elif rres:
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
        with st.container(key="triton_ctrl_stop"):
            st.caption("Action: **Stop** · confirmation phrase required")
            _render_stop_control_column(
                stop_el, start_busy=start_busy or restart_busy, restart_busy=restart_busy
            )

    with ctrl3:
        with st.container(key="triton_ctrl_restart"):
            st.caption("Action: **Restart** · confirmation phrase required")
            _render_restart_control_column(
                restart_el, start_busy=start_busy, stop_busy=stop_busy and not restart_busy
            )

    with st.expander("Control rules and eligibility evidence", expanded=False):
        st.caption(
            f"allowed_start={eligibility.get('allowed_action')} · "
            f"allowed_stop={stop_el.get('allowed_dashboard_action')} · "
            f"allowed_restart={restart_el.get('allowed_dashboard_action')} · "
            f"stop_enabled={stop_el.get('stop_button_enabled')} · "
            f"restart_enabled={restart_el.get('restart_button_enabled')} · "
            f"paper_arm={eligibility.get('paper_arm_status')}"
        )
        st.caption(
            "Stop uses verified cooperative shutdown: active bounded work may finish at its "
            "documented safe boundary; TRITON is never force-terminated by this control. "
            "Restart is a verified cooperative Stop followed by a fresh protected Start; "
            "it does not delete locks, bypass Start eligibility, arm paper, or enable live execution."
        )

    with st.expander("Cooperative shutdown details", expanded=False):
        _render_cooperative_shutdown_status()

    with st.expander("Control acknowledgement evidence", expanded=False):
        stop_ack = st.session_state.get("stop_last_result") or {}
        restart_ack = st.session_state.get("restart_last_result") or {}
        start_ack = st.session_state.get("start_last_result") or {}
        st.caption(
            f"Latest Start result: **{start_ack.get('result') or 'none'}** · "
            f"Stop: **{stop_ack.get('result') or 'none'}** · "
            f"Restart: **{restart_ack.get('result') or 'none'}**"
        )
        st.caption(
            f"Stop ui_state: {stop_ack.get('ui_state') or 'n/a'} · "
            f"acknowledgement_results present: "
            f"{'yes' if stop_ack.get('acknowledgement_results') else 'no'}"
        )
        if stop_ack.get("acknowledgement_results") or restart_ack.get("stages"):
            st.json(
                {
                    "stop_acknowledgement_results": stop_ack.get("acknowledgement_results"),
                    "restart_stages": restart_ack.get("stages"),
                }
            )


def _render_stop_control_column(
    stop_el: Dict[str, Any], *, start_busy: bool, restart_busy: bool = False
) -> None:
    from dashboard import operator_ui as oui
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
        # secondary + keyed-container CSS (red); avoid Streamlit primary red colliding with Start
        if st.button("Stop TRITON Safely", key="stop_triton_request", type="secondary"):
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
        f"Preflight expires: {oui.format_timestamp(st.session_state.get('stop_preflight_expires_at'))} · "
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
            type="secondary",
            disabled=not can_submit,
        ):
            st.session_state["stop_in_progress"] = True
            st.session_state["stop_confirm_pending"] = False
            result = {
                "result": "STOP_FAILED",
                "safe_message": "Stop did not complete a terminal controller result.",
            }
            try:
                with st.spinner("Requesting cooperative shutdown and validating…"):
                    result = execute_stop_triton(
                        action=ACTION_STOP,
                        confirmation_phrase=phrase,
                        confirmation_checked=True,
                        preflight_token=st.session_state.get("stop_preflight_token"),
                        request_nonce=st.session_state.get("stop_request_nonce"),
                        source="operator_console",
                    )
            except Exception as exc:
                result = {
                    "result": "STOP_FAILED",
                    "error_code": "UI_STOP_EXCEPTION",
                    "safe_message": (
                        "Stop failed in the operator console before a durable "
                        f"controller result was recorded ({type(exc).__name__})."
                    ),
                }
            finally:
                st.session_state["stop_last_result"] = result
                st.session_state["stop_in_progress"] = False
                st.session_state["stop_preflight_token"] = None
                st.session_state["stop_request_nonce"] = None
                # Stop supersedes Start success so refresh cannot resurrect "both running".
                if str(result.get("result") or "") in (
                    "STOP_COMPLETED",
                    "STOP_ALREADY_STOPPED",
                ):
                    if st.session_state.get("start_last_result"):
                        st.session_state["start_result_history"] = st.session_state.get(
                            "start_last_result"
                        )
                    st.session_state["start_last_result"] = None
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
    from dashboard import operator_ui as oui
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
        # secondary + keyed-container CSS (amber); Start keeps primary (blue)
        if st.button("Restart TRITON Safely", key="restart_triton_request", type="secondary"):
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
        f"Preflight expires: {oui.format_timestamp(st.session_state.get('restart_preflight_expires_at'))} · "
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
            type="secondary",
            disabled=not can_submit,
        ):
            st.session_state["restart_in_progress"] = True
            st.session_state["restart_confirm_pending"] = False
            result = {
                "result": "RESTART_FAILED",
                "safe_message": "Restart did not complete a terminal controller result.",
            }
            try:
                with st.spinner("Orchestrating cooperative Stop then fresh Start…"):
                    result = execute_restart_triton(
                        action=ACTION_RESTART,
                        confirmation_phrase=phrase,
                        confirmation_checked=True,
                        preflight_token=st.session_state.get("restart_preflight_token"),
                        request_nonce=st.session_state.get("restart_request_nonce"),
                        source="operator_console",
                    )
            except Exception as exc:
                result = {
                    "result": "RESTART_FAILED",
                    "error_code": "UI_RESTART_EXCEPTION",
                    "safe_message": (
                        "Restart failed in the operator console before a durable "
                        f"controller result was recorded ({type(exc).__name__})."
                    ),
                }
            finally:
                st.session_state["restart_last_result"] = result
                st.session_state["restart_in_progress"] = False
                st.session_state["restart_preflight_token"] = None
                st.session_state["restart_request_nonce"] = None
                if str(result.get("result") or "") == "RESTART_COMPLETED":
                    # Fresh runtime ownership; prior Start/Stop latches are historical.
                    if st.session_state.get("start_last_result"):
                        st.session_state["start_result_history"] = st.session_state.get(
                            "start_last_result"
                        )
                    st.session_state["start_last_result"] = None
                    if st.session_state.get("stop_last_result"):
                        st.session_state["stop_result_history"] = st.session_state.get(
                            "stop_last_result"
                        )
                    st.session_state["stop_last_result"] = None
            st.rerun()
    with conf_b:
        if st.button("Cancel", key="restart_triton_cancel"):
            st.session_state["restart_confirm_pending"] = False
            st.session_state["restart_preflight_token"] = None
            st.session_state["restart_request_nonce"] = None
            st.rerun()


def _render_cooperative_shutdown_status() -> None:
    """Read-only DO-004/DO-005 cooperative shutdown visibility (collapsed by caller)."""
    from services.operator_shutdown_controller import shutdown_eligibility
    from services.operator_stop_action import stop_eligibility

    status = shutdown_eligibility()
    stop_el = stop_eligibility()
    st.markdown("**Cooperative Shutdown / Protected Stop evidence**")
    st.caption(stop_el.get("safe_message") or status.get("safe_message") or "")
    st.caption(
        f"Stop decision: **{stop_el.get('decision')}** · "
        f"shutdown decision: **{status.get('decision')}** · "
        f"targets={stop_el.get('target_components')} · "
        f"protected_stop={stop_el.get('protected_stop_exposed')} · "
        f"restart={stop_el.get('restart_exposed')}"
    )
    with st.expander("Operator-facing explanation", expanded=False):
        st.write(stop_el.get("explanation") or status.get("explanation") or "")
        st.write(stop_el.get("recommended_action") or "")

    req = status.get("active_request") or {}
    active = status.get("shutdown_active") or {}
    with st.expander("Request / guard JSON", expanded=False):
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
    with st.expander("Acknowledgements / progress JSON", expanded=False):
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
    from dashboard import operator_ui as oui
    from services.operator_console_status import DATA_FRESH_MINUTES, build_operator_status
    from services.operator_console_status import build_portfolio_summary
    from services.operator_runtime_controller import start_eligibility

    oui.inject_operator_theme()
    status = build_operator_status()
    start_el = start_eligibility()
    portfolio = build_portfolio_summary()
    live = status.get("live_execution") or {}
    paper = status.get("paper_arm") or {}
    cont = status.get("continuous") or {}
    wd = status.get("watchdog") or {}
    broker = status.get("broker") or {}
    cycle = status.get("cycle") or {}
    op = status.get("orders_positions") or {}
    alerts = status.get("alerts") or {}
    freshness = status.get("data_freshness") or {}
    market = status.get("market") or {}
    paper_cert = status.get("paper_certification") or {}
    live_perm = status.get("live_permission") or {}
    shutdown = status.get("shutdown_presentation") or {}
    overall = str(status.get("overall_status") or "UNKNOWN")
    attention = list(status.get("attention") or [])

    # Authoritative evidence timestamp for the header (prefer cycle lock refresh, then heartbeat).
    evidence_ts = oui.format_timestamp(
        cycle.get("last_completed_at")
        or freshness.get("heartbeat_timestamp")
        or status.get("generated_at")
    )
    market_state = oui.display_or_unknown(market.get("status"))
    runtime_expectation = status.get("runtime_expectation") or {}
    expected_after_hours_stop = bool(runtime_expectation.get("expected"))
    # Pipeline heartbeat = heartbeat.json class from operator status (not watchdog).
    pipeline_hb = oui.present_pipeline_heartbeat_label(
        freshness_class=freshness.get("class"),
        process_count=cont.get("process_count"),
        discovery_available=cont.get("process_discovery_available", True),
        lock_held=bool(cont.get("lock_exists")),
        expected_after_hours_stop=expected_after_hours_stop,
        age_minutes=freshness.get("heartbeat_age_minutes"),
        evidence_path=freshness.get("evidence_path") or "heartbeat.json",
    )
    # Watchdog heartbeat = watchdog_status.json via existing read-only loader (banner source).
    from services.dashboard_status_sync import load_runtime_heartbeat

    watchdog_hb = oui.present_watchdog_heartbeat_label(
        load_runtime_heartbeat(),
        process_count=wd.get("process_count"),
        discovery_available=wd.get("process_discovery_available", True),
        lock_held=bool(wd.get("lock_exists")),
        expected_after_hours_stop=expected_after_hours_stop,
    )
    portfolio_freshness = oui.classify_portfolio_freshness_label(
        last_refresh_time=portfolio.get("last_refresh_time"),
        fresh_minutes=float(DATA_FRESH_MINUTES),
    )
    next_expected = cycle.get("next_expected_at") or "Unavailable"
    watchdog_alert_count = (
        alerts.get("active_count") if alerts.get("active_count") is not None else "UNKNOWN"
    )
    attention_count = len(attention)

    hdr_l, hdr_r = st.columns([6.2, 1.0])
    with hdr_l:
        oui.render_command_header(
            title="Operations",
            context=(
                "Daily operator workspace · Protected Start/Stop/Restart · "
                "Live execution remains blocked unless evidence shows otherwise"
            ),
            market_state=market_state,
            evidence_ts=evidence_ts,
        )
    with hdr_r:
        st.write("")
        if st.button("Refresh", key="cc_ops_refresh", use_container_width=True):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.session_state.pop("contract_results", None)
            st.session_state.pop("contract_summary", None)
            st.session_state.pop("pipeline_health", None)
            st.session_state.pop("activity_feed_cache", None)
            st.rerun()

    _render_held_risk_plan_blocked_alerts()

    live_value = "BLOCKED" if live.get("blocked") else "NOT BLOCKED"
    paper_value = oui.display_or_unknown(paper.get("display_status"))
    # Continuous stale warning uses pipeline heartbeat only (heartbeat.json).
    # Watchdog stale warning uses watchdog heartbeat only (watchdog_status.json).
    cont_card = oui.loop_process_card_state(
        process_running=cont.get("process_running"),
        process_count=cont.get("process_count"),
        discovery_available=cont.get("process_discovery_available", True),
        heartbeat_class=pipeline_hb["label"],
        heartbeat_age_minutes=freshness.get("heartbeat_age_minutes"),
    )
    wd_card = oui.loop_process_card_state(
        process_running=wd.get("process_running"),
        process_count=wd.get("process_count"),
        discovery_available=wd.get("process_discovery_available", True),
        heartbeat_class=watchdog_hb["label"],
        heartbeat_age_minutes=watchdog_hb.get("age_minutes"),
    )
    # Permanent six-card strip (Live remains the safety lock surface; no full-width banner).
    # Continuous/Watchdog always show process-derived instance counts (UIR-003C).
    oui.render_safety_strip(
        [
            {
                "label": "Live execution",
                "value": live_value,
                "tone": oui.tone_for_live_execution(live_value),
                "note": "Required safety lock" if live_value == "BLOCKED" else None,
            },
            {
                "label": "Paper authorization",
                "value": paper_value,
                "tone": oui.tone_for_status(paper_value),
            },
            {
                "label": "Market",
                "value": market_state,
                "tone": oui.tone_for_market(market_state),
            },
            {
                "label": "Continuous loop",
                "value": cont_card.get("display") or cont_card["label"],
                "tone": cont_card["tone"],
                "note": cont_card.get("note") or cont_card.get("stale_warning"),
            },
            {
                "label": "Watchdog",
                "value": wd_card.get("display") or wd_card["label"],
                "tone": wd_card["tone"],
                "note": wd_card.get("note") or wd_card.get("stale_warning"),
            },
            {
                "label": "Portfolio data freshness",
                "value": portfolio_freshness,
                "tone": oui.tone_for_status(portfolio_freshness),
            },
        ]
    )

    # Only surface a critical alert when the safety lock appears missing.
    if not live.get("blocked"):
        st.error(
            "CRITICAL SAFETY WARNING: live execution does not appear blocked. "
            "No enable control is provided. Investigate immediately."
        )

    # Compact supporting summary — stacked label/value rows (no inline collision).
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        oui.render_compact_panel(
            "Runtime",
            [
                {"label": "Overall runtime", "value": overall},
                {
                    "label": "Trading cycle",
                    "value": oui.display_or_unknown(
                        (status.get("trading_pipeline") or {}).get("status")
                    ),
                },
                {
                    "label": "Session context",
                    "value": oui.display_or_unknown((market.get("session_phase") or market_state)),
                },
                {"label": "Pipeline heartbeat", "value": pipeline_hb["display"]},
                {"label": "Next expected cycle", "value": next_expected},
            ],
        )
    with s2:
        oui.render_compact_panel(
            "Readiness",
            [
                {
                    "label": "Certification / readiness",
                    "value": oui.display_or_unknown(
                        paper_cert.get("display_status") or "PAPER:UNKNOWN"
                    ),
                },
                {"label": "Watchdog heartbeat", "value": watchdog_hb["display"]},
                {
                    "label": "Cycle overdue",
                    "value": "yes" if cycle.get("overdue") else "no",
                },
            ],
        )
    with s3:
        broker_state = oui.display_or_unknown(broker.get("state"))
        oui.render_compact_panel(
            "Evidence",
            [
                {"label": "Broker evidence state", "value": broker_state},
                {
                    "label": "Start eligibility",
                    "value": start_el.get("decision") or "UNKNOWN",
                },
                {
                    "label": "Attention items",
                    "value": attention_count,
                },
            ],
        )
    with s4:
        oui.render_compact_panel(
            "Portfolio counts",
            [
                {
                    "label": "Position count",
                    "value": (
                        op.get("position_count")
                        if op.get("position_count") is not None
                        else "UNKNOWN"
                    ),
                },
                {
                    "label": "Open orders",
                    "value": (
                        op.get("open_orders") if op.get("open_orders") is not None else "UNKNOWN"
                    ),
                },
                {
                    "label": "Watchdog active alerts",
                    "value": watchdog_alert_count,
                },
            ],
        )

    # Protected controls immediately after summary — exact existing callbacks reused.
    _render_start_controls(status_snapshot=status)

    attention_summary = oui.summarize_attention_items(attention)
    top_issue = attention_summary.get("top_item")
    attention_groups = status.get("attention_groups") or attention_summary.get("groups") or {}
    action_now_n = int(
        attention_groups.get("action_required_now_count")
        or attention_summary.get("action_required_now_count")
        or 0
    )
    before_open_n = int(
        attention_groups.get("before_next_market_open_count")
        or attention_summary.get("before_next_market_open_count")
        or 0
    )
    info_n = int(
        attention_groups.get("informational_count")
        or attention_summary.get("informational_count")
        or 0
    )

    center, right = st.columns([3.35, 1.15], gap="medium")
    with center:
        _render_runtime_recovery_guidance()

        with st.expander(
            f"Operator Attention details — {attention_count} items "
            f"(Now {action_now_n} · Before open {before_open_n} · Info {info_n})",
            expanded=bool(action_now_n),
        ):
            if not attention:
                st.info("No evidence-backed attention items right now.")
            else:
                sections = (
                    ("Action required now", attention_groups.get("action_required_now") or []),
                    (
                        "Before next market open",
                        attention_groups.get("before_next_market_open") or [],
                    ),
                    ("Informational / expected", attention_groups.get("informational") or []),
                )
                for section_title, section_items in sections:
                    st.markdown(f"**{section_title}** · {len(section_items)}")
                    if not section_items:
                        st.caption("None")
                        continue
                    for item in section_items:
                        oui.render_attention_item(
                            item,
                            start_decision=start_el.get("decision"),
                            start_button_enabled=start_el.get("start_button_enabled"),
                        )

        with st.expander("Technical evidence", expanded=False):
            st.caption("Raw classifier / lock / cycle / broker evidence (read-only).")
            st.json(
                {
                    "overall_status": overall,
                    "continuous": cont,
                    "watchdog": wd,
                    "paper_arm": paper,
                    "live_execution": live,
                    "broker": broker,
                    "market": market,
                    "cycle": cycle,
                    "runtime_expectation": runtime_expectation,
                    "pipeline_heartbeat": {
                        "channel": pipeline_hb.get("channel"),
                        "source": pipeline_hb.get("source"),
                        "label": pipeline_hb.get("label"),
                        "from_status_data_freshness_class": freshness.get("class"),
                        "heartbeat_timestamp": freshness.get("heartbeat_timestamp"),
                        "age_minutes": pipeline_hb.get("age_minutes"),
                        "display": pipeline_hb.get("display"),
                    },
                    "watchdog_heartbeat": {
                        "channel": watchdog_hb.get("channel"),
                        "source": watchdog_hb.get("source"),
                        "label": watchdog_hb.get("label"),
                        "age_minutes": watchdog_hb.get("age_minutes"),
                        "display": watchdog_hb.get("display"),
                        "actionable_stale_banner": watchdog_hb.get("actionable_stale_banner"),
                    },
                    "portfolio_data_freshness": {
                        "class": portfolio_freshness,
                        "last_refresh_time": portfolio.get("last_refresh_time"),
                    },
                    "paper_certification": paper_cert,
                    "live_permission": live_perm,
                    "shutdown_presentation": shutdown,
                    "orders_positions": op,
                    "alerts": {
                        "active_count": alerts.get("active_count"),
                        "critical_count": alerts.get("critical_count"),
                        "note": "watchdog artifact active-alert count",
                    },
                    "attention_item_count": attention_count,
                    "evidence_paths": status.get("evidence_paths"),
                    "generated_at": status.get("generated_at"),
                    "start_eligibility_decision": start_el.get("decision"),
                    "start_button_enabled": start_el.get("start_button_enabled"),
                }
            )

    with right:
        counts = attention_summary.get("counts") or {}
        oui.render_compact_panel(
            "Operator Attention",
            [
                {"label": "Action required now", "value": action_now_n},
                {"label": "Before next market open", "value": before_open_n},
                {"label": "Informational / expected", "value": info_n},
                {"label": "Total items", "value": attention_count},
                {
                    "label": "Severity counts",
                    "value": (
                        f"C{counts.get('CRITICAL', 0)}/"
                        f"H{counts.get('HIGH', 0)}/"
                        f"M{counts.get('MEDIUM', 0) + counts.get('WARN', 0)}/"
                        f"I{counts.get('INFO', 0)}"
                    ),
                },
                {
                    "label": "Top issue",
                    "value": (
                        f"[{top_issue.get('severity')}] {top_issue.get('title')}"
                        if top_issue
                        else "None"
                    ),
                },
            ],
        )
        if top_issue:
            presented = oui.present_attention_recommendation(
                top_issue,
                start_decision=start_el.get("decision"),
                start_button_enabled=start_el.get("start_button_enabled"),
            )
            st.caption(
                f"Recommended safe action: {presented.get('current_action') or 'Review evidence.'}"
            )
        else:
            st.caption("No evidence-backed attention items.")
        st.caption("Full issue explanations are in Operator Attention details below-left.")

        with st.expander("Current action status", expanded=False):
            # Unique fields only — Live / Paper / Market / loops already on the strip.
            oui.render_compact_panel(
                "Action status",
                [
                    {
                        "label": "Shutdown request",
                        "value": shutdown.get("display_status") or "NONE",
                    },
                    {
                        "label": "Shutdown historical",
                        "value": "yes" if shutdown.get("historical") else "no",
                    },
                    {
                        "label": "Start eligibility",
                        "value": start_el.get("decision") or "UNKNOWN",
                    },
                ],
            )
            if shutdown.get("historical") and not shutdown.get("active"):
                st.caption("Historical/expired shutdown — does not affect current runtime health")
            st.caption(
                f"Market evidence: source={market.get('source') or 'n/a'} · "
                f"as_of={market.get('as_of') or 'n/a'}"
            )
            if watchdog_hb.get("age_note"):
                st.caption(
                    f"Watchdog heartbeat age: {watchdog_hb['age_note']} · "
                    f"source={watchdog_hb.get('source')}"
                )
            label = broker.get("display_label") or "Paper broker REST: Unknown"
            st.caption(label)
            st.caption(str(broker.get("meaning") or ""))
            st.write(status.get("evidence_paths") or {})
            st.caption("Market CLOSED is a neutral market state, not a critical runtime failure.")


def _render_runtime_recovery_guidance() -> None:
    """DO-007 read-only advisory section. Never submits Start/Stop/Restart."""
    from services.operator_runtime_recovery_guidance import build_runtime_recovery_guidance

    # Fresh read-only evaluation on each render; no auto-submit, no preflight, no tokens.
    guidance = build_runtime_recovery_guidance()
    sev = str(guidance.get("severity") or "INFO")
    overall = str(guidance.get("runtime_overall_status") or "UNKNOWN")
    action = guidance.get("safe_next_action") or "none"

    st.markdown("**Runtime Recovery Guidance**")
    if sev == "CRITICAL":
        st.error(f"**{overall}** · severity={sev}")
    elif sev == "HIGH":
        st.warning(f"**{overall}** · severity={sev}")
    else:
        st.caption(f"**{overall}** · severity={sev}")
    st.caption(f"Recommended safe action: **{action}**")
    st.caption(
        "Recovery guidance is advisory only. It does not start, stop, restart, terminate, "
        "delete locks, create paper authorization, or enable live execution."
    )

    # Keep active blockers visible when present; details otherwise collapsed.
    blockers = guidance.get("blockers") or []
    if blockers:
        st.caption("Blockers: " + ", ".join(str(b) for b in blockers[:6]))

    with st.expander("Runtime recovery details", expanded=False):
        st.write(guidance.get("operator_message") or "")
        targets = guidance.get("affected_targets") or []
        st.caption(f"Affected targets: {', '.join(targets) if targets else 'none'}")
        st.caption(
            f"Eligible protected control (advisory): "
            f"**{guidance.get('eligible_protected_control') or 'none'}**"
        )
        st.caption(
            f"Manual investigation required: **{guidance.get('manual_investigation_required')}** · "
            f"Escalation required: **{guidance.get('escalation_required')}** · "
            f"Confidence: {guidance.get('confidence')}"
        )
        st.caption(
            f"Live execution blocked: **{guidance.get('live_execution_blocked')}** "
            "(required safe state; never enable live from this page)"
        )
        st.caption(
            f"Paper authorization: **{guidance.get('paper_arm_status')}** "
            "(separate prerequisite; guidance does not create or extend authorization)"
        )
        warnings = [
            w
            for w in (guidance.get("warnings") or [])
            if w not in ("LIVE_EXECUTION_BLOCKED (required safe state)",)
        ]
        st.caption("Warnings: " + (", ".join(str(w) for w in warnings[:8]) if warnings else "none"))

    evidence = guidance.get("evidence_summary") or []
    with st.expander("Recovery evidence summary", expanded=False):
        for line in evidence:
            st.write(f"- {line}")
        if guidance.get("restart_stage"):
            st.caption(
                f"Restart stage (Stop and Start are separate): stop_side="
                f"{guidance.get('restart_stop_stage')} · start_side="
                f"{guidance.get('restart_start_stage')} · stage="
                f"{guidance.get('restart_stage')}"
            )
        st.caption(
            "Guidance does not promise successful recovery, immediate restart, "
            "forced termination, lock repair, paper authorization, or live enablement."
        )

    with st.expander("Recovery technical evidence", expanded=False):
        st.json(
            {
                "guidance_id": guidance.get("guidance_id"),
                "guidance_id_is_authority": False,
                "advisory_only": True,
                "execution_attempted": False,
                "fresh_start_eligibility_decision": guidance.get(
                    "fresh_start_eligibility_decision"
                ),
                "fresh_stop_eligibility_decision": guidance.get("fresh_stop_eligibility_decision"),
                "fresh_restart_eligibility_decision": guidance.get(
                    "fresh_restart_eligibility_decision"
                ),
            }
        )


def render_reports_page() -> None:
    """UIR-002 Reports — read-only reconciled portfolio reporting engine presentation."""
    from dashboard.operator_reports import render_operator_reports_page

    render_operator_reports_page()


def render_activity_page() -> None:
    from services.operator_activity_feed import (
        CATEGORIES,
        SEVERITIES,
        collect_activity_events,
        filter_activity_events,
    )
    from dashboard import operator_ui as oui

    oui.inject_operator_theme()
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
        date_label = ev.get("occurred_date") or (
            str(ev.get("occurred_at") or "")[:10] if ev.get("occurred_at") else ""
        )
        title = f"{ev.get('severity')} · {ev.get('category')} · {ev.get('title')}"
        if ev.get("category") == "ORDER" and date_label:
            title = f"{title} · {date_label}"
        with st.expander(title, expanded=False):
            st.write(ev.get("message"))
            meta = []
            if date_label and ev.get("category") == "ORDER":
                meta.append(f"Date: {date_label}")
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
    from dashboard import operator_ui as oui
    from services.operator_console_status import build_portfolio_summary

    oui.inject_operator_theme()
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
    from dashboard import operator_ui as oui
    from services.operator_console_status import build_orders_summary

    oui.inject_operator_theme()
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
    from dashboard import operator_ui as oui
    from services.operator_console_status import build_risk_alerts_summary

    oui.inject_operator_theme()
    st.title("Risk & Alerts")
    st.caption(
        "Read-only risk and watchdog posture. Alerts cannot be dismissed or overridden here."
    )
    _render_held_risk_plan_blocked_alerts()
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
    from dashboard import operator_ui as oui
    from services.operator_console_status import build_operator_status

    oui.inject_operator_theme()
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
    "Reports": render_reports_page,
    "Orders": render_orders_page,
    "Risk & Alerts": render_risk_alerts_page,
    "Diagnostics": render_diagnostics_page,
}
