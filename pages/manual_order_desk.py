"""
manual_order_desk.py — TRITON Manual Order Desk (Streamlit UI wrapper)
---------------------------------------------------------------------

READ/WRITE capabilities (broker actions):
- Place market / limit / bracket orders
- Cancel by order_id
- Cancel all open orders
- View open orders (nested=true)
- View positions
- Account summary

Safety:
- Defaults to paper if env says paper
- Requires explicit toggles for destructive actions
- Extra confirmation for LIVE mode
- No background loops

This file is imported optionally by view_results.py:
    from pages.manual_order_desk import render_manual_order_desk
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


# ──────────────────────────────
# Project root + imports
# ──────────────────────────────
def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (
            (cur / ".git").exists()
            or (cur / "pyproject.toml").exists()
            or (cur / "requirements.txt").exists()
        ):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Broker wrapper (must exist in your repo)
from services.broker_alpaca import AlpacaBroker  # type: ignore


# ──────────────────────────────
# Helpers
# ──────────────────────────────
_TICKER_RE = re.compile(r"^[A-Z0-9.\-_]{1,15}$")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _env_mode_default() -> str:
    base = os.getenv("APCA_API_BASE_URL", "").lower().strip()
    if "paper-api" in base:
        return "paper"
    if base:
        return "live"
    return "paper"  # safest default


def _df_from_list(rows: Any) -> pd.DataFrame:
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, pd.DataFrame):
        return rows
    if isinstance(rows, list):
        return pd.DataFrame(rows)
    if isinstance(rows, dict):
        return pd.DataFrame([rows])
    return pd.DataFrame()


def _clean_symbol(s: str) -> str:
    return (s or "").strip().upper()


def _valid_symbol(s: str) -> bool:
    return bool(_TICKER_RE.match(s or ""))


@dataclass
class DeskState:
    mode: str = "paper"


# ──────────────────────────────
# Main renderer
# ──────────────────────────────
def render_manual_order_desk() -> None:
    st.markdown("### 🧾 Manual Order Desk")
    st.caption("Manual broker controls. Use carefully. Recommended: Paper mode while testing.")

    # Mode selector
    default_mode = _env_mode_default()
    mode = st.selectbox(
        "Broker mode",
        options=["paper", "live"],
        index=0 if default_mode == "paper" else 1,
        help="Paper is safest. Live sends real orders.",
    )

    # Hard gate for LIVE
    live_ack = True
    if mode == "live":
        st.warning("LIVE mode selected. This can place real orders.", icon="⚠️")
        live_ack = st.checkbox("I understand LIVE mode can place real orders.", value=False)

    st.info(f"Selected mode: **{mode.upper()}**  •  Time: `{_now_str()}`", icon="🧭")

    # Create broker
    try:
        broker = AlpacaBroker(mode=mode)
    except Exception as e:
        st.error("Failed to initialize broker. Check env vars (.env) and Alpaca keys.")
        st.exception(e)
        return

    st.markdown("---")

    # Keep the UI linear with expanders (Phase 1.5 spirit).
    with st.expander("📌 Account Summary", expanded=True):
        try:
            acct = broker.get_account()
            st.json(acct)
        except Exception as e:
            st.error("Failed to fetch account.")
            st.exception(e)

    with st.expander("📍 Positions", expanded=False):
        try:
            pos = broker.list_positions()
            dfp = _df_from_list(pos)
            if dfp.empty:
                st.caption("No positions.")
            else:
                st.dataframe(dfp, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error("Failed to list positions.")
            st.exception(e)

    with st.expander("🧾 Open Orders (nested=true)", expanded=False):
        try:
            oo = broker.list_orders(status="open", nested=True, limit=200)
            dfo = _df_from_list(oo)
            if dfo.empty:
                st.caption("No open orders.")
            else:
                preferred = [
                    "id",
                    "symbol",
                    "side",
                    "type",
                    "order_class",
                    "qty",
                    "filled_qty",
                    "limit_price",
                    "stop_price",
                    "status",
                    "time_in_force",
                    "created_at",
                    "submitted_at",
                    "updated_at",
                    "client_order_id",
                ]
                cols = [c for c in preferred if c in dfo.columns] + [
                    c for c in dfo.columns if c not in preferred
                ]
                st.dataframe(dfo[cols], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error("Failed to list open orders.")
            st.exception(e)

    st.markdown("---")
    st.markdown("### ✍️ Place Order (Manual)")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        symbol = _clean_symbol(c1.text_input("Symbol", value="AAPL"))
        side = c2.selectbox("Side", options=["buy", "sell"], index=0)
        order_type = c3.selectbox("Type", options=["market", "limit"], index=0)
        tif = c4.selectbox("TIF", options=["day", "gtc"], index=0)

        c5, c6, c7, c8 = st.columns([1, 1, 1, 1])
        qty = c5.number_input("Qty (integer)", min_value=1.0, value=1.0, step=1.0)
        limit_price_raw = c6.text_input("Limit Price (if limit)", value="")
        bracket = c7.toggle("Bracket order", value=False)

        # Execution toggle (two-stage safety: user toggle + live ack)
        really = c8.toggle("Really place (SEND order)", value=False)

        c9, c10 = st.columns(2)
        tp_raw = c9.text_input("Take Profit (tp_limit)", value="")
        sl_raw = c10.text_input("Stop Loss (sl_stop)", value="")

        st.caption(
            "Safety: No order is sent unless **Really place** is enabled (and LIVE ack is checked when in LIVE)."
        )

        if st.button("Submit Order", use_container_width=True):
            # Validate symbol
            if not symbol:
                st.error("Symbol is required.")
                return
            if not _valid_symbol(symbol):
                st.error(
                    "Invalid symbol format. Use letters/numbers and . - _ only (max 15 chars)."
                )
                return

            # Validate qty (must be integer-like)
            if float(qty) <= 0:
                st.error("Qty must be > 0.")
                return
            if abs(float(qty) - round(float(qty))) > 1e-9:
                st.error("Qty must be a whole number (no decimals).")
                return
            q = int(round(float(qty)))

            # Parse numerics
            lp = _safe_float(limit_price_raw)
            tp_f = _safe_float(tp_raw)
            sl_f = _safe_float(sl_raw)

            # Validate order type requirements
            if order_type == "limit" and lp is None:
                st.error("Limit order requires a valid limit price.")
                return

            # Validate bracket requirements
            if bracket:
                if tp_f is None or sl_f is None:
                    st.error(
                        "Bracket order requires BOTH take profit (tp_limit) and stop loss (sl_stop)."
                    )
                    return
                # If user selected limit bracket, ensure limit price exists.
                if order_type == "limit" and lp is None:
                    st.error("Limit bracket requires a valid limit price.")
                    return

            payload: Dict[str, Any] = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "qty": q,
                "limit_price": lp,
                "tif": tif,
                "bracket": bracket,
                "tp_limit": tp_f,
                "sl_stop": sl_f,
                "mode": mode,
            }

            # DRY RUN branch
            if not really:
                st.warning("DRY RUN (not sent). Enable 'Really place' to execute.", icon="🧪")
                st.write(payload)
                return

            # LIVE acknowledgement gate
            if mode == "live" and not live_ack:
                st.warning(
                    "LIVE mode requires the acknowledgement checkbox before sending orders.",
                    icon="⚠️",
                )
                st.write(payload)
                return

            # Execute
            try:
                st.write("Sending payload:")
                st.code(payload)

                if bracket:
                    resp = broker.submit_bracket_order(
                        symbol=symbol,
                        side=side,
                        qty=q,
                        limit_price=lp,  # None allowed for market-bracket if your wrapper supports it
                        tp_limit=tp_f,
                        sl_stop=sl_f,
                        time_in_force=tif,
                    )
                else:
                    resp = broker.submit_order(
                        symbol=symbol,
                        side=side,
                        qty=q,
                        order_type=order_type,
                        time_in_force=tif,
                        limit_price=lp,
                    )

                st.success("Order submitted.")
                st.json(resp)
            except Exception as e:
                st.error("Order submit failed.")
                st.exception(e)

    st.markdown("---")
    st.markdown("### 🧨 Cancel Controls")

    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        cancel_id = c1.text_input("Cancel by Order ID", value="").strip()
        confirm_cancel = c2.toggle("Confirm cancel", value=False)

        if st.button("Cancel Order ID", use_container_width=True):
            if not cancel_id:
                st.error("Enter an order id.")
                return
            if not confirm_cancel:
                st.warning("Enable 'Confirm cancel' before canceling.", icon="⚠️")
                return
            if mode == "live" and not live_ack:
                st.warning(
                    "LIVE mode requires acknowledgement checkbox before destructive actions.",
                    icon="⚠️",
                )
                return
            try:
                resp = broker.cancel_order(cancel_id)
                st.success("Cancel request sent.")
                st.json(
                    resp
                    if resp is not None
                    else {"status": "cancel_requested", "order_id": cancel_id}
                )
            except Exception as e:
                st.error("Cancel failed.")
                st.exception(e)

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        confirm_cancel_all = c1.toggle("Confirm CANCEL ALL", value=False)
        include_oco = c2.toggle("Include bracket/OCO legs", value=True)
        st.caption("Canceling all open orders is destructive. Use sparingly.")

        if st.button("Cancel ALL Open Orders", use_container_width=True):
            if not confirm_cancel_all:
                st.warning("Enable 'Confirm CANCEL ALL' first.", icon="⚠️")
                return
            if mode == "live" and not live_ack:
                st.warning(
                    "LIVE mode requires acknowledgement checkbox before destructive actions.",
                    icon="⚠️",
                )
                return
            try:
                resp = broker.cancel_all_orders(include_nested=include_oco)
                st.success("Cancel-all request sent.")
                st.json(resp if resp is not None else {"status": "cancel_all_requested"})
            except Exception as e:
                st.error("Cancel-all failed.")
                st.exception(e)

    st.markdown("---")
    st.caption("Manual Order Desk loaded successfully.")
