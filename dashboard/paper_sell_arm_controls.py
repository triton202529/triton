"""Operations-page paper SELL arm controls.

Presentation only. Authority lives in SQLite via services.paper_sell_arm_console.
Does not dispatch, create permits, or contact a broker.
Production calls use the canonical data/live SQLite path (no env/session redirects).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import streamlit as st

from services.paper_sell_arm_console import (
    arm_reviewed_sells,
    build_presentation,
    confirm_and_arm_paper_sells,
    operator_fail_message,
)

SESSION_CHALLENGE_KEY = "paper_sell_arm_challenge_id"
SESSION_PHRASE_KEY = "paper_sell_confirm_phrase"
SESSION_FLASH_KEY = "paper_sell_arm_flash"


def render_paper_sell_arm_controls(
    *,
    db_path: Optional[Path] = None,
    results_root: Optional[Path] = None,
    audit_root: Optional[Path] = None,
    paper_mode: Optional[bool] = None,
    live_blocked: Optional[bool] = None,
) -> None:
    """Render SELL-arm buttons. Omit path args in production (canonical live DB)."""
    st.subheader("Reviewed paper SELLs")
    phrase = st.session_state.get(SESSION_PHRASE_KEY, "")
    pres = build_presentation(
        store_root=audit_root,
        results_root=results_root,
        paper_mode=paper_mode,
        live_blocked=live_blocked,
        confirmation_phrase=phrase,
        session_challenge_id=st.session_state.get(SESSION_CHALLENGE_KEY),
        db_path=db_path,
    )

    st.caption(pres["no_submit_caption"])
    st.write(f"Packet freshness: {pres['freshness']}")
    st.write(f"Packet generated: {pres['generated_at'] or 'unavailable'}")
    st.write(f"Reviewed SELL actions: {pres['action_count']}")
    for row in pres.get("actions") or []:
        st.write(f"- {row.get('symbol')} {row.get('action')} qty={row.get('quantity')}")
    st.write(f"Challenge status: {pres.get('challenge_state') or 'ABSENT'}")
    if pres.get("challenge_expires_at"):
        st.write(f"Challenge expiry: {pres['challenge_expires_at']}")
    st.write(f"Authority status: {pres.get('authority_state') or 'ABSENT'}")
    if pres.get("authority_expires_at"):
        st.write(f"Authority expiry: {pres['authority_expires_at']}")

    flash = st.session_state.get(SESSION_FLASH_KEY)
    if flash:
        text = flash.get("text")
        if flash.get("kind") == "success":
            st.success(text)
        else:
            st.error(text)
        for line in flash.get("extra") or []:
            if line:
                st.info(line)

    if pres.get("arm_deny_reason") and not pres.get("arm_enabled"):
        st.warning(operator_fail_message(str(pres["arm_deny_reason"])))

    if st.button(
        "Arm Reviewed Sells",
        key="paper_sell_arm_reviewed",
        disabled=not pres.get("arm_enabled"),
        type="primary",
    ):
        result = arm_reviewed_sells(
            store_root=audit_root,
            results_root=results_root,
            paper_mode=paper_mode,
            live_blocked=live_blocked,
            db_path=db_path,
        )
        _apply_result(result)

    st.text_input(
        "Type exactly: CONFIRM PAPER SELLS",
        key=SESSION_PHRASE_KEY,
    )
    phrase_now = str(st.session_state.get(SESSION_PHRASE_KEY) or "")
    pres2 = build_presentation(
        store_root=audit_root,
        results_root=results_root,
        paper_mode=paper_mode,
        live_blocked=live_blocked,
        confirmation_phrase=phrase_now,
        session_challenge_id=st.session_state.get(SESSION_CHALLENGE_KEY),
        db_path=db_path,
    )
    if st.button(
        "Confirm & Arm Paper Sells",
        key="paper_sell_arm_confirm",
        disabled=not pres2.get("confirm_enabled"),
        type="secondary",
    ):
        result = confirm_and_arm_paper_sells(
            store_root=audit_root,
            results_root=results_root,
            confirmation_phrase=str(st.session_state.get(SESSION_PHRASE_KEY) or ""),
            challenge_id=st.session_state.get(SESSION_CHALLENGE_KEY),
            paper_mode=paper_mode,
            live_blocked=live_blocked,
            db_path=db_path,
        )
        if result.get("ok"):
            st.session_state[SESSION_PHRASE_KEY] = ""
        _apply_result(result)


def _apply_result(result: Dict[str, Any]) -> None:
    extra = []
    if result.get("ok") and result.get("button") == "arm":
        st.session_state[SESSION_CHALLENGE_KEY] = result.get("challenge_id")
        extra = [
            f"Challenge expiry: {result.get('challenge_expires_at')}",
            f"Exact reviewed action count: {result.get('action_count')}",
            str(result.get("instruction") or ""),
        ]
    elif result.get("ok"):
        extra = [
            f"Authority expiry: {result.get('authority_expires_at')}",
            f"Exact action count: {result.get('action_count')}",
            "single-use: True",
            str(result.get("warning") or ""),
        ]
    st.session_state[SESSION_FLASH_KEY] = {
        "kind": "success" if result.get("ok") else "error",
        "text": result.get("message"),
        "extra": extra,
    }
    st.rerun()
