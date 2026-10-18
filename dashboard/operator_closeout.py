"""Operator review screen for Close out & stop TRITON."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from services.operator_closeout_control import (
    COMPLETED_STATES,
    CONFIRMATION_PHRASE,
    STATE_PUBLISHING,
    active_closeout,
    evaluate_eligibility,
    execute_closeout,
    exit_current_dashboard,
    launch_detached_worker,
    list_closeout_receipts,
    load_receipt,
    receipt_path,
)

ROOT = Path(__file__).resolve().parents[1]


def _session_now() -> Dict[str, Any]:
    try:
        from services.market_calendar import regular_session_state

        return dict(regular_session_state())
    except Exception as exc:
        return {
            "session_state": "UNKNOWN",
            "reason": "calendar_unavailable",
            "calendar": "XNYS",
            "calendar_error": type(exc).__name__,
        }


def _paper_broker() -> Any:
    from services.broker_alpaca import AlpacaBroker

    return AlpacaBroker(mode="paper")


def _runtime_inputs(root: Path) -> Dict[str, Any]:
    from services.operator_cooperative_shutdown import is_guard_active
    from services.operator_runtime_classifier import classify_runtime_loops
    from services.operator_runtime_controller import live_execution_appears_enabled
    from services.performance_publication_authority import (
        inspect_git_identity,
        inspect_runtime_state,
    )

    now = datetime.now(timezone.utc)
    live = root / "data" / "live"
    runtime = inspect_runtime_state(root)
    try:
        git_identity = dict(inspect_git_identity(root))
    except Exception as exc:
        git_identity = {"git_identity_valid": False, "failure_reason": type(exc).__name__}
    return {
        "classifications": classify_runtime_loops(root=root),
        "live_execution_blocked": not live_execution_appears_enabled(root=root),
        "start_guard_active": is_guard_active(live / "operator_start_action.guard.json", now=now),
        "shutdown_guard_active": is_guard_active(
            live / "operator_shutdown_action.guard.json", now=now
        ),
        "git_identity": git_identity,
        "freshness_idle": runtime.get("freshness_writer_idle") is True,
        "freshness_lock_present": runtime.get("freshness_lock_present") is True,
        "refresh_running": False,
        "now": now,
    }


def _benchmarks(root: Path, session_date: str) -> Dict[str, Dict[str, Any]]:
    bars: Dict[str, Dict[str, Any]] = {}
    for symbol in ("SPY", "QQQ"):
        path = root / "data" / "raw" / f"{symbol}.csv"
        row = {"symbol": symbol, "session_date": session_date, "close": None, "adj_close": None}
        if path.is_file():
            import csv

            with path.open("r", encoding="utf-8", newline="") as handle:
                for item in csv.DictReader(handle):
                    if str(item.get("date") or "")[:10] == session_date:
                        row["close"] = item.get("close")
                        row["adj_close"] = item.get("adj_close") or item.get("adj close")
        bars[symbol] = row
    return bars


def _show_receipt(receipt: Dict[str, Any]) -> None:
    state = str(receipt.get("state") or "UNKNOWN")
    if state in COMPLETED_STATES:
        st.success(f"Closeout {state} for {receipt.get('session_date')}.")
    elif state == STATE_PUBLISHING:
        st.warning(
            "Publication is in progress. This dashboard must be stopped before the close is written."
        )
    else:
        st.error(f"Closeout {state}: {receipt.get('reason') or 'blocked'}")
    st.caption(
        f"Session {receipt.get('session_date') or 'Unavailable'} · "
        f"close {receipt.get('session_close_display') or 'Unavailable'} · "
        f"cash {receipt.get('cash') or 'Unavailable'} · "
        f"market value {receipt.get('market_value') or 'Unavailable'} · "
        f"equity {receipt.get('equity') or 'Unavailable'}"
    )
    st.caption(
        "orders_submitted=false · orders_cancelled=false · "
        "orders_replaced=false · orders_modified=false"
    )
    with st.expander("Closeout receipt", expanded=False):
        visible = {key: value for key, value in receipt.items() if key != "sealed_snapshot"}
        st.json(visible)


def render_closeout_control(
    *,
    start_busy: bool,
    stop_busy: bool,
    restart_busy: bool,
    root: Optional[Path] = None,
) -> None:
    """Review screen and confirmation for Close out & stop TRITON."""
    root = Path(root) if root is not None else ROOT
    st.subheader("Close out & stop TRITON")
    st.caption(
        "PAPER_ONLY post-close control. It does not submit, cancel, replace, or modify broker orders."
    )
    receipts = list_closeout_receipts(root)
    if receipts:
        _show_receipt(receipts[-1])
    if st.session_state.get("closeout_exit_pending"):
        st.warning(
            "The closeout worker is detached. This dashboard is stopping so publication can recheck "
            "ports 8501, 8502, and 8765. Reopen the dashboard to read the receipt."
        )
        exit_current_dashboard()
        return
    publishing = next(
        (item for item in receipts if str(item.get("state")) == STATE_PUBLISHING), None
    )
    if publishing is not None:
        st.caption(
            "Publication is waiting for this dashboard to stop. "
            "Finish closeout launches the detached worker and stops this dashboard last."
        )
        if st.button("Finish closeout", key="closeout_resume", type="secondary"):
            launch_detached_worker(
                root, receipt_path(root, str(publishing.get("session_date") or ""))
            )
            st.session_state["closeout_exit_pending"] = True
            st.rerun()
        return
    busy = start_busy or stop_busy or restart_busy or active_closeout(root) is not None
    if busy and not st.session_state.get("closeout_review_pending"):
        st.button("Close out & stop TRITON", key="closeout_disabled_busy", disabled=True)
        st.caption(
            "Closeout is unavailable while Start, Stop, Restart, or another closeout is active."
        )
        return
    if not st.session_state.get("closeout_review_pending"):
        if st.button("Close out & stop TRITON", key="closeout_open_review", type="secondary"):
            st.session_state["closeout_review_pending"] = True
            st.session_state.pop("closeout_review", None)
            st.rerun()
        return

    review = st.session_state.get("closeout_review")
    if not isinstance(review, dict):
        session = _session_now()
        try:
            broker = _paper_broker()
        except Exception as exc:
            broker = _BrokenBroker(exc)
        inputs = _runtime_inputs(root)
        session_date = str(session.get("session_date") or "")
        review = evaluate_eligibility(
            root=root,
            session=session,
            broker=broker,
            benchmarks=_benchmarks(root, session_date),
            **inputs,
        )
        st.session_state["closeout_review"] = review
        st.session_state["closeout_session"] = session

    st.markdown(
        f"**Session date:** {review.get('session_date') or 'Unavailable'}  \n"
        f"**XNYS close ({review.get('close_kind') or 'unavailable'}):** "
        f"{review.get('session_close_display') or 'Unavailable'}"
    )
    for row in review.get("conditions") or []:
        mark = "ready" if row.get("ok") else "blocked"
        st.caption(f"{mark} · {row.get('code')} · {row.get('detail')}")
    if review.get("blockers"):
        st.error("Closeout is blocked. No stop has been requested.")
    phrase = st.text_input(
        "Type exactly: CLOSE OUT AND STOP TRITON",
        key="closeout_phrase",
    )
    can_submit = phrase == CONFIRMATION_PHRASE and review.get("ok") is True and not busy
    confirm, cancel = st.columns(2)
    with confirm:
        if st.button(
            "Confirm close out & stop",
            key="closeout_confirm",
            type="secondary",
            disabled=not can_submit,
        ):
            session = st.session_state.get("closeout_session") or _session_now()
            inputs = _runtime_inputs(root)
            try:
                broker = _paper_broker()
            except Exception as exc:
                broker = _BrokenBroker(exc)
            result = execute_closeout(
                confirmation_phrase=phrase,
                root=root,
                session=session,
                broker=broker,
                benchmarks=_benchmarks(root, str(session.get("session_date") or "")),
                **inputs,
            )
            st.session_state["closeout_last_result"] = result
            st.session_state["closeout_review_pending"] = False
            if result.get("dashboard_exit_required"):
                st.session_state["closeout_exit_pending"] = True
            st.rerun()
    with cancel:
        if st.button("Cancel", key="closeout_cancel"):
            st.session_state["closeout_review_pending"] = False
            st.session_state.pop("closeout_review", None)
            st.rerun()
    stored = st.session_state.get("closeout_last_result")
    if isinstance(stored, dict):
        _show_receipt(stored)
    session_date = str((st.session_state.get("closeout_session") or {}).get("session_date") or "")
    if session_date:
        durable = load_receipt(receipt_path(root, session_date))
        if durable:
            _show_receipt(durable)


class _BrokenBroker:
    def __init__(self, exc: Exception) -> None:
        self.mode = "paper"
        self.base = "https://paper-api.alpaca.markets"
        self._exc = exc

    def get_account(self) -> Dict[str, Any]:
        raise self._exc

    def get_positions(self) -> list:
        raise self._exc

    def get_open_orders(self) -> list:
        raise self._exc
