"""Operations-page paper BUY/ADD arm control.

Presentation only. Writes go through services.paper_buy_add_arm_console, which
calls the same write_confirm used by the CLI. Does not start TRITON or submit orders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import streamlit as st

from services.paper_buy_add_arm_console import (
    ARM_PHRASE,
    REVOKE_PHRASE,
    build_presentation,
    confirm_paper_buy_add_arm,
    revoke_paper_buy_add_arm,
)

SESSION_PHRASE_KEY = "paper_buy_add_arm_phrase"
SESSION_REVOKE_KEY = "paper_buy_add_revoke_phrase"
SESSION_FLASH_KEY = "paper_buy_add_arm_flash"
SESSION_IDEM_KEY = "paper_buy_add_arm_idempotency"
ARM_BUTTON_LABEL = "Arm Paper BUY/ADD"
RENEW_BUTTON_LABEL = "Renew Paper BUY/ADD Arm"
REVOKE_BUTTON_LABEL = "Revoke Paper BUY/ADD Arm"


def render_paper_buy_add_arm_controls(
    *,
    confirm_path: Optional[Path] = None,
    paper_mode: Optional[bool] = None,
    live_blocked: Optional[bool] = None,
    allow_path_override: bool = False,
    session_state_fn=None,
) -> None:
    from dashboard.operator_ui import section_header

    section_header(
        "Paper BUY/ADD authorization",
        "Creates or renews the existing paper_arm_confirm latch · does not start TRITON",
    )
    pres = build_presentation(
        paper_mode=paper_mode,
        live_blocked=live_blocked,
        confirm_path=confirm_path,
        allow_path_override=allow_path_override,
        session_state_fn=session_state_fn,
    )
    st.caption(pres["no_start_caption"])
    st.caption(
        "Use this control to arm paper BUY/ADD entries. The PowerShell command remains an operator fallback."
    )
    st.write(f"Status: {pres['status']}")
    st.write(f"Authorized session date: {pres.get('authorized_session_date') or 'unavailable'}")
    st.write(f"Created: {pres.get('created_at') or 'unavailable'}")
    st.write(f"Expiry: {pres.get('expires_at') or 'unavailable'}")
    remaining = pres.get("remaining_minutes")
    if remaining is None:
        st.write("Remaining duration: unavailable")
    else:
        st.write(f"Remaining duration: {remaining:.1f} minutes")
    st.write(f"Live execution: {pres.get('live_execution')}")

    flash = st.session_state.get(SESSION_FLASH_KEY)
    if flash:
        if flash.get("kind") == "success":
            st.success(flash.get("text"))
        else:
            st.error(flash.get("text"))

    if pres.get("arm_deny_reason") and not pres.get("arm_enabled"):
        from services.paper_buy_add_arm_console import _operator_message

        st.warning(_operator_message(str(pres["arm_deny_reason"])))

    if st.session_state.pop("_paper_buy_add_clear_phrase", False):
        st.session_state[SESSION_PHRASE_KEY] = ""
    if st.session_state.pop("_paper_buy_add_clear_revoke", False):
        st.session_state[SESSION_REVOKE_KEY] = ""

    st.text_input("Type exactly: ARM PAPER ENTRIES", key=SESSION_PHRASE_KEY)
    arm_label = (
        RENEW_BUTTON_LABEL if pres.get("button_label") == RENEW_BUTTON_LABEL else ARM_BUTTON_LABEL
    )
    if st.button(
        arm_label,
        key="paper_buy_add_arm_submit",
        disabled=not pres.get("arm_enabled"),
        type="primary",
    ):
        import uuid

        click_id = str(uuid.uuid4())
        result = confirm_paper_buy_add_arm(
            confirmation_phrase=str(st.session_state.get(SESSION_PHRASE_KEY) or ""),
            paper_mode=paper_mode,
            live_blocked=live_blocked,
            confirm_path=confirm_path,
            allow_path_override=allow_path_override,
            session_state_fn=session_state_fn,
            idempotency_key=click_id,
            last_idempotency_key=st.session_state.get(SESSION_IDEM_KEY),
        )
        if result.get("ok"):
            st.session_state[SESSION_IDEM_KEY] = click_id
        st.session_state["_paper_buy_add_clear_phrase"] = True
        _flash(result)

    if pres.get("revoke_enabled"):
        st.text_input("Type exactly: REVOKE PAPER ENTRIES", key=SESSION_REVOKE_KEY)
        if st.button(
            REVOKE_BUTTON_LABEL,
            key="paper_buy_add_arm_revoke",
            type="secondary",
        ):
            result = revoke_paper_buy_add_arm(
                confirmation_phrase=str(st.session_state.get(SESSION_REVOKE_KEY) or ""),
                paper_mode=paper_mode,
                live_blocked=live_blocked,
                confirm_path=confirm_path,
                allow_path_override=allow_path_override,
            )
            st.session_state["_paper_buy_add_clear_revoke"] = True
            _flash(result)


def _flash(result: Dict[str, Any]) -> None:
    st.session_state[SESSION_FLASH_KEY] = {
        "kind": "success" if result.get("ok") else "error",
        "text": result.get("message"),
    }
    st.rerun()
