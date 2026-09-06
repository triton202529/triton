"""Operations-page paper SELL arm controls.

Presentation only. Authority lives in SQLite via services.paper_sell_arm_console.
Does not dispatch, create permits, or contact a broker.
Production calls use the canonical data/live SQLite path (no env/session redirects).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from services.paper_sell_arm_console import (
    arm_reviewed_sells,
    build_presentation,
    confirm_and_arm_paper_sells,
    operator_fail_message,
)

SESSION_CHALLENGE_KEY = "paper_sell_arm_challenge_id"
SESSION_PHRASE_KEY = "paper_sell_confirm_phrase"
SESSION_PHRASE_CLEAR_PENDING_KEY = "paper_sell_confirm_phrase_clear_pending"
SESSION_FLASH_KEY = "paper_sell_arm_flash"
SESSION_TOKEN_KEY = "paper_sell_arm_presentation_token"
ARM_BUTTON_LABEL = "Arm Reviewed Sells"
CONFIRM_BUTTON_LABEL = "Confirm & Arm Paper Sells"

# Visual status labels (presentation only; controller remains authoritative).
LABEL_ARMED = "SELL ARM — ARMED"
LABEL_READY = "SELL ARM — READY TO ARM"
LABEL_AWAITING = "SELL ARM — AWAITING CONFIRMATION"
LABEL_UNAVAILABLE = "SELL ARM — UNAVAILABLE"
LABEL_NOT_REQUIRED = "SELL ARM — NOT REQUIRED"
LABEL_COMPLETED = "SELL ARM — COMPLETED"
LABEL_ATTENTION = "SELL ARM — ATTENTION REQUIRED"

COLOR_GREEN = "GREEN"
COLOR_AMBER = "AMBER"
COLOR_BLUE = "BLUE"
COLOR_RED = "RED"

_STATUS_STYLES = {
    COLOR_GREEN: {"bg": "#d1e7dd", "fg": "#0f5132", "border": "#badbcc"},
    COLOR_AMBER: {"bg": "#fff3cd", "fg": "#664d03", "border": "#ffecb5"},
    COLOR_BLUE: {"bg": "#cfe2ff", "fg": "#084298", "border": "#b6d4fe"},
    COLOR_RED: {"bg": "#f8d7da", "fg": "#842029", "border": "#f5c2c7"},
}


def _authority_matches_batch(pres: Dict[str, Any]) -> bool:
    """True only when latest authority binds the current reviewed plan+batch."""
    plan_hash = str(pres.get("plan_hash") or "")
    batch_hash = str(pres.get("batch_hash") or "")
    if not plan_hash or not batch_hash:
        return False
    auth = ((pres.get("snapshot") or {}).get("latest_authority")) or {}
    payload = auth.get("payload") if isinstance(auth.get("payload"), dict) else {}
    auth_plan = str(auth.get("plan_hash") or payload.get("plan_hash") or "")
    auth_batch = str(auth.get("batch_hash") or payload.get("batch_hash") or "")
    return auth_plan == plan_hash and auth_batch == batch_hash


def map_sell_arm_visual_status(pres: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only mapper from controller presentation → colored status box.

    False-positive rules: OPEN challenge, expired/stale/mismatched/consumed
    authority, absent SQLite, and malformed packets never render ARMED.
    Only valid, unexpired, UNUSED exact-batch authority renders ARMED.
    """
    freshness = str(pres.get("freshness") or "")
    challenge = str(pres.get("challenge_state") or "ABSENT")
    authority = str(pres.get("authority_state") or "ABSENT")
    deny = str(pres.get("arm_deny_reason") or "")
    reason = ""
    if deny:
        reason = operator_fail_message(deny)
    elif freshness == "OUTCOME_UNKNOWN":
        reason = operator_fail_message("sell_outcome_unknown")

    # Attention-first gates (never ARMED).
    if freshness == "OUTCOME_UNKNOWN" or deny == "sell_outcome_unknown":
        return {
            "label": LABEL_ATTENTION,
            "color": COLOR_RED,
            "reason": reason or operator_fail_message("sell_outcome_unknown"),
            "code": "ATTENTION_REQUIRED",
        }

    # OPEN challenge is never ARMED.
    if challenge == "OPEN":
        return {
            "label": LABEL_AWAITING,
            "color": COLOR_AMBER,
            "reason": reason or "Confirmation challenge is OPEN.",
            "code": "AWAITING_CONFIRMATION",
        }

    # Only UNUSED + exact-batch match + fresh packet context → ARMED.
    if authority == "UNUSED" and _authority_matches_batch(pres):
        return {
            "label": LABEL_ARMED,
            "color": COLOR_GREEN,
            "reason": "Valid unused exact-batch authority.",
            "code": "ARMED",
        }

    # Consumed matching authority → COMPLETED (not ARMED).
    if authority == "CONSUMED" and _authority_matches_batch(pres):
        return {
            "label": LABEL_COMPLETED,
            "color": COLOR_BLUE,
            "reason": pres.get("last_dispatch_message")
            or reason
            or "Authority consumed for this reviewed batch.",
            "code": "COMPLETED",
        }

    # Empty reviewed set → NOT REQUIRED.
    if freshness == "EMPTY" or deny == "no_reviewed_sell_actions":
        return {
            "label": LABEL_NOT_REQUIRED,
            "color": COLOR_BLUE,
            "reason": reason or operator_fail_message("no_reviewed_sell_actions"),
            "code": "NOT_REQUIRED",
        }

    # Ready to arm (controller arm_enabled with fresh packet).
    if freshness == "FRESH" and bool(pres.get("arm_enabled")):
        return {
            "label": LABEL_READY,
            "color": COLOR_AMBER,
            "reason": reason or "Cap-compliant reviewed SELL slice is ready.",
            "code": "READY_TO_ARM",
        }

    # Mismatched / expired / restricted authority while a packet exists.
    if authority in {"UNUSED", "EXPIRED", "RESTRICTED", "REVOKED"} and not _authority_matches_batch(
        pres
    ):
        # UNUSED without match is a false-positive trap → ATTENTION, not ARMED.
        if authority == "UNUSED":
            return {
                "label": LABEL_ATTENTION,
                "color": COLOR_RED,
                "reason": reason or "Authority does not match the current reviewed batch.",
                "code": "ATTENTION_REQUIRED",
            }

    # Blocked / unknown outcomes that need operator attention.
    if freshness in {"BLOCKED_OTHER_GATE", "SOURCE_UNBOUND"} or deny in {
        "blocked_other_gate",
        "source_unbound",
        "sqlite_unavailable",
        "sqlite_pragma_or_schema_failure",
        "sqlite_corrupt",
        "live_execution_not_blocked",
        "non_paper_mode",
        "configuration_blocked",
    }:
        return {
            "label": LABEL_ATTENTION,
            "color": COLOR_RED,
            "reason": reason or LABEL_ATTENTION,
            "code": "ATTENTION_REQUIRED",
        }

    # Malformed / stale / unavailable packets → UNAVAILABLE (AMBER).
    if freshness in {"MALFORMED", "STALE", "UNAVAILABLE", "unavailable", "EMPTY"} or deny:
        return {
            "label": LABEL_UNAVAILABLE,
            "color": COLOR_AMBER,
            "reason": reason or "Reviewed SELL arm is unavailable.",
            "code": "UNAVAILABLE",
        }

    return {
        "label": LABEL_ATTENTION,
        "color": COLOR_RED,
        "reason": reason or LABEL_ATTENTION,
        "code": "ATTENTION_REQUIRED",
    }


def _fmt_money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _fmt_qty(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value or "")
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _short_fingerprint(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "n/a"
    return text[:12]


def _render_slice_details(pres: Dict[str, Any]) -> None:
    cap = pres.get("applied_cap")
    if cap is not None:
        st.write(f"**Applied authority cap:** {_fmt_money(cap)}")
    full_count = int(pres.get("action_count") or 0)
    full_total = pres.get("full_total_notional")
    if full_count or full_total:
        st.write(
            f"**Full reviewed requirement:** {full_count} action(s), {_fmt_money(full_total or 0)}"
        )
    slice_actions = list(pres.get("slice_actions") or [])
    st.write(
        f"**Full source packet:** `{_short_fingerprint(pres.get('source_packet_fingerprint'))}`"
    )
    st.write(f"**Current authorized slice:** `{_short_fingerprint(pres.get('slice_hash'))}`")
    if slice_actions:
        st.write(
            f"**Current authorization slice:** {int(pres.get('slice_action_count') or 0)} action(s), "
            f"{_fmt_money(pres.get('slice_total_notional') or 0)}"
        )
        for row in slice_actions:
            st.write(
                f"- {row.get('symbol')} {row.get('action')} qty={_fmt_qty(row.get('qty'))} "
                f"({_fmt_money(row.get('est_notional') or 0)})"
            )
    deferred = list(pres.get("deferred_actions") or [])
    if deferred:
        st.write(
            f"**Deferred remainder:** {int(pres.get('deferred_action_count') or 0)} action(s), "
            f"estimated {_fmt_money(pres.get('deferred_total_notional') or 0)}"
        )
        for row in deferred:
            qty = row.get("deferred_qty")
            if qty is None:
                qty = row.get("qty")
            st.write(
                f"- {row.get('symbol')} {row.get('action')} qty={_fmt_qty(qty)} "
                f"(est. {_fmt_money(row.get('est_notional') or 0)})"
            )
    if pres.get("estimated_remaining_slice_count_is_estimate") and pres.get("slice_action_count"):
        remaining = int(pres.get("estimated_remaining_slice_count") or 0)
        st.caption(
            f"Estimated remaining slices after this one: {remaining}. "
            "Every slice requires separate operator review and confirmation."
        )
    elif pres.get("slice_action_count"):
        st.caption("Every slice requires separate operator review and confirmation.")


def _render_status_box(visual: Dict[str, Any]) -> None:
    style = _STATUS_STYLES.get(str(visual.get("color") or COLOR_AMBER), _STATUS_STYLES[COLOR_AMBER])
    label = str(visual.get("label") or LABEL_ATTENTION)
    reason = str(visual.get("reason") or "").strip()
    reason_html = (
        f'<div style="margin-top:0.25rem;font-size:0.85rem;opacity:0.95;">{reason}</div>'
        if reason
        else ""
    )
    st.markdown(
        f'<div data-testid="sell-arm-status-box" style="'
        f"background:{style['bg']};color:{style['fg']};border:1px solid {style['border']};"
        f"border-radius:0.4rem;padding:0.65rem 0.85rem;margin:0.35rem 0 0.75rem 0;"
        f'font-weight:650;line-height:1.35;">'
        f"{label}{reason_html}</div>",
        unsafe_allow_html=True,
    )


def render_paper_sell_arm_controls(
    *,
    db_path: Optional[Path] = None,
    results_root: Optional[Path] = None,
    audit_root: Optional[Path] = None,
    paper_mode: Optional[bool] = None,
    live_blocked: Optional[bool] = None,
    compact: bool = False,
) -> None:
    """Render SELL-arm buttons. Omit path args in production (canonical live DB)."""
    if compact:
        st.markdown("##### Reviewed paper SELL authorization")
    else:
        st.subheader("Reviewed paper SELLs")

    # Streamlit cannot modify the session_state key after a widget with the same
    # key has been instantiated in this run. We therefore clear the phrase via
    # a pending flag and perform the actual clear before the text_input widget
    # is created on the next render.
    if st.session_state.get(SESSION_PHRASE_CLEAR_PENDING_KEY):
        st.session_state[SESSION_PHRASE_KEY] = ""
        st.session_state[SESSION_PHRASE_CLEAR_PENDING_KEY] = False

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

    visual = map_sell_arm_visual_status(pres)
    _render_status_box(visual)

    st.caption(pres["no_submit_caption"])
    st.write(f"**Packet freshness:** {pres['freshness']}")
    st.write(f"Packet generated: {pres['generated_at'] or 'unavailable'}")
    _render_slice_details(pres)
    if not pres.get("slice_actions") and pres.get("actions"):
        st.write(f"Reviewed SELL actions: {pres['action_count']}")
        for row in pres.get("actions") or []:
            st.write(f"- {row.get('symbol')} {row.get('action')} qty={row.get('quantity')}")

    challenge_state = str(pres.get("challenge_state") or "ABSENT")
    authority_state = str(pres.get("authority_state") or "ABSENT")
    st.write(f"**Challenge status:** {challenge_state}")
    if pres.get("challenge_expires_at"):
        st.write(f"Challenge expiry: {pres['challenge_expires_at']}")
    st.write(f"**Authority status:** {authority_state}")
    if pres.get("authority_expires_at"):
        st.write(f"Authority expiry: {pres['authority_expires_at']}")

    st.write(f"**Current readiness packet:** {pres['freshness']}")
    st.write(f"**Last authority:** {authority_state}")
    if pres.get("last_dispatch_message"):
        st.write(f"**Last dispatch:** {pres.get('last_dispatch_message')}")
        vis = pres.get("last_dispatch_visibility") or {}
        if vis.get("dispatch_ids"):
            st.caption("Last broker reconciliation is available on Orders and Activity.")

    flash = st.session_state.get(SESSION_FLASH_KEY)
    if flash:
        text = str(flash.get("text") or "")
        extra_lines = list(flash.get("extra") or [])
        if pres.get("suppress_no_orders_submitted") or authority_state == "CONSUMED":
            if "No orders were submitted" in text:
                text = pres.get("last_dispatch_message") or (
                    "Paper SELL authority consumed. Open Orders or Activity for execution status."
                )
            extra_lines = [ln for ln in extra_lines if "No orders were submitted" not in str(ln)]
        if flash.get("kind") == "success":
            st.success(text)
        else:
            st.error(text)
        for line in extra_lines:
            if line:
                st.info(line)

    # One concise readiness/refusal reason (avoid repeating the same warning).
    if (
        pres.get("arm_deny_reason")
        and not pres.get("arm_enabled")
        and not pres.get("confirm_enabled")
    ):
        # Status box already shows the concise reason; skip duplicate warning when identical.
        msg = operator_fail_message(str(pres["arm_deny_reason"]))
        if msg != str(visual.get("reason") or ""):
            st.warning(msg)
    elif authority_state == "UNUSED":
        st.info(
            "Reviewed batch is armed (unused valid authority). No additional Arm action for this batch."
        )

    open_challenge = challenge_state == "OPEN"
    # When an open challenge exists, Confirm is the primary action; Arm is not primary.
    arm_is_primary = (
        bool(pres.get("arm_enabled")) and not open_challenge and authority_state != "UNUSED"
    )
    confirm_is_primary = open_challenge or bool(pres.get("confirm_enabled"))
    arm_disabled = (not pres.get("arm_enabled")) or open_challenge or authority_state == "UNUSED"

    if st.button(
        ARM_BUTTON_LABEL,
        key="paper_sell_arm_reviewed",
        disabled=arm_disabled,
        type="primary" if arm_is_primary else "secondary",
        use_container_width=True,
    ):
        result = arm_reviewed_sells(
            store_root=audit_root,
            results_root=results_root,
            paper_mode=paper_mode,
            live_blocked=live_blocked,
            db_path=db_path,
            expected_presentation_token=st.session_state.get(SESSION_TOKEN_KEY),
        )
        _apply_result(result)

    st.session_state[SESSION_TOKEN_KEY] = pres.get("presentation_token")

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
        CONFIRM_BUTTON_LABEL,
        key="paper_sell_arm_confirm",
        disabled=not pres2.get("confirm_enabled"),
        type="primary" if (confirm_is_primary or pres2.get("confirm_enabled")) else "secondary",
        use_container_width=True,
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
            # Defer clearing until the next run (see pending-clear at top).
            st.session_state[SESSION_PHRASE_CLEAR_PENDING_KEY] = True
        _apply_result(result)

    with st.expander("SELL readiness and authority details", expanded=False):
        st.write(f"Plan hash: {pres.get('plan_hash') or 'n/a'}")
        st.write(f"Batch hash: {pres.get('batch_hash') or 'n/a'}")
        st.write(f"Resolved SQLite: {pres.get('resolved_db_path') or 'n/a'}")
        st.write(f"Store root: {pres.get('store_root') or 'n/a'}")
        st.write(f"Results root: {pres.get('results_root') or 'n/a'}")
        st.write(f"Arm duplicate: {pres.get('arm_duplicate')}")
        st.write(f"Labels: {ARM_BUTTON_LABEL} / {CONFIRM_BUTTON_LABEL}")
        st.write(f"Visual status: {visual.get('label')} ({visual.get('color')})")
        st.write(f"Slice hash: {pres.get('slice_hash') or 'n/a'}")
        st.write(f"Source packet fingerprint: {pres.get('source_packet_fingerprint') or 'n/a'}")


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
