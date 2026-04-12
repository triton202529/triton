# pages/manual_order_desk.py
"""
TRITON — Manual Order Desk (Phase 2.2)
UPDATED: Phase 2.3 — Ledger + Reconciliation integrated
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import re
import streamlit as st

from services.broker_alpaca import AlpacaBroker, AlpacaError
from services.ledger import load_ledger, LEDGER_PATH
from services.reconcile_state import (
    reconcile_state,
)  # read-only here; placement uses its own call path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_ROOT / "results"
ORDERS_DIR = DATA_ROOT / "orders"

GUARD_SNAPSHOT_PATH = RESULTS_DIR / "guard_snapshot.json"
LIVE_ORDERS_PATH = RESULTS_DIR / "live_orders.csv"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_guard() -> Dict[str, Any]:
    if not GUARD_SNAPSHOT_PATH.exists() or GUARD_SNAPSHOT_PATH.stat().st_size == 0:
        return {}
    try:
        import json

        return json.loads(GUARD_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _kill_switch_state(guard: Dict[str, Any]) -> Tuple[bool, str]:
    if not guard:
        return False, ""
    reason = str(guard.get("reason") or guard.get("message") or "").strip()

    ks = guard.get("kill_switch")
    if isinstance(ks, bool) and ks:
        return True, reason or "Kill switch enabled."

    mode = str(guard.get("mode") or "").upper().strip()
    if mode in {"KILL_SWITCH", "KILLSWITCH", "FROZEN", "FREEZE", "MAINTENANCE"}:
        return True, reason or "Trading is frozen by guard mode."

    blocked = guard.get("blocked")
    if isinstance(blocked, bool) and blocked:
        return True, reason or "Trading blocked by guard."

    return False, ""


def _append_live_order(row: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not LIVE_ORDERS_PATH.exists()

    cols = [
        "ts_utc",
        "mode",
        "symbol",
        "side",
        "order_type",
        "tif",
        "sizing",
        "qty",
        "notional",
        "limit_price",
        "stop_price",
        "order_class",
        "tp_price",
        "sl_price",
        "client_order_id",
        "alpaca_order_id",
        "status",
        "raw",
    ]
    out = {c: row.get(c, "") for c in cols}

    with open(LIVE_ORDERS_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerow(out)


def _is_valid_symbol(sym: str) -> bool:
    sym = (sym or "").strip().upper()
    return bool(sym) and bool(re.match(r"^[A-Z0-9\.\-_]{1,15}$", sym))


def _parse_float(x: Any) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def _validate_inputs(
    symbol: str,
    sizing: str,
    qty: Optional[float],
    notional: Optional[float],
    order_type: str,
    limit_price: Optional[float],
    stop_price: Optional[float],
    order_class: str,
    tp_price: Optional[float],
    sl_price: Optional[float],
) -> Tuple[bool, str]:
    if not _is_valid_symbol(symbol):
        return False, "Invalid symbol. Use letters/numbers only (e.g., AAPL, TSLA)."

    if sizing == "qty":
        if qty is None or qty <= 0:
            return False, "Quantity must be > 0."
    else:
        if notional is None or notional <= 0:
            return False, "Notional (USD) must be > 0."

    if order_type == "limit":
        if limit_price is None or limit_price <= 0:
            return False, "Limit price must be > 0 for limit orders."

    if order_type == "stop":
        if stop_price is None or stop_price <= 0:
            return False, "Stop price must be > 0 for stop orders."

    if order_class == "bracket":
        if tp_price is None or tp_price <= 0:
            return False, "Take-profit price must be > 0 for bracket orders."
        if sl_price is None or sl_price <= 0:
            return False, "Stop-loss price must be > 0 for bracket orders."

    return True, ""


def render_manual_order_desk() -> None:
    st.markdown("### 🧾 Manual Order Desk")
    st.caption("Validate-only by default. Real placement is blocked by Guard + Reconcile Freeze.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)

    # Always show ledger snapshot (authoritative)
    with st.expander("📒 Ledger (authoritative state)", expanded=False):
        df = load_ledger(LEDGER_PATH)
        st.caption(f"Ledger file: {LEDGER_PATH}")
        st.dataframe(df, use_container_width=True)

    guard = _load_guard()
    blocked, reason = _kill_switch_state(guard)

    # Header banner
    if blocked:
        st.error(f"⛔ **BLOCKED**: {reason or 'Trading is frozen.'}", icon="⛔")
        with st.expander("Guard snapshot", expanded=False):
            st.json(guard)
    else:
        st.info("Guard is OK. (Reconcile still runs on placement.)", icon="✅")

    # Broker mode
    top = st.columns([1, 2, 2, 2])
    with top[0]:
        mode = st.selectbox("Mode", ["paper", "live"], index=0)
    with top[1]:
        symbol = st.text_input("Symbol", value="TSLA").strip().upper()
    with top[2]:
        side = st.selectbox("Side", ["buy", "sell"], index=0)
    with top[3]:
        order_type = st.selectbox("Type", ["market", "limit", "stop"], index=1)

    row2 = st.columns([1, 2, 2, 2])
    with row2[0]:
        tif = st.selectbox("TIF", ["day", "gtc"], index=0)
    with row2[1]:
        sizing = st.selectbox("Sizing", ["notional", "qty"], index=0)
    with row2[2]:
        order_class = st.selectbox("Order Class", ["simple", "bracket"], index=0)
    with row2[3]:
        client_order_id = st.text_input("Client Order ID (optional)", value="").strip() or None

    # Sizing inputs
    qty_val: Optional[float] = None
    notional_val: Optional[float] = None
    if sizing == "qty":
        qty_val = _parse_float(st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0))
    else:
        notional_val = _parse_float(
            st.number_input("Notional (USD)", min_value=0.0, value=100.0, step=10.0)
        )

    # Price inputs
    limit_price = None
    stop_price = None
    if order_type == "limit":
        limit_price = _parse_float(
            st.number_input("Limit Price", min_value=0.0, value=0.0, step=0.01, format="%.4f")
        )
    if order_type == "stop":
        stop_price = _parse_float(
            st.number_input("Stop Price", min_value=0.0, value=0.0, step=0.01, format="%.4f")
        )

    # Bracket inputs
    tp_price = None
    sl_price = None
    if order_class == "bracket":
        ctp, csl = st.columns(2)
        with ctp:
            tp_price = _parse_float(
                st.number_input(
                    "Take Profit Price", min_value=0.0, value=0.0, step=0.01, format="%.4f"
                )
            )
        with csl:
            sl_price = _parse_float(
                st.number_input(
                    "Stop Loss Price", min_value=0.0, value=0.0, step=0.01, format="%.4f"
                )
            )

    dry_run = st.toggle("Dry run (validate only)", value=True)

    valid, msg = _validate_inputs(
        symbol=symbol,
        sizing=("qty" if sizing == "qty" else "notional"),
        qty=qty_val,
        notional=notional_val,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        order_class=order_class,
        tp_price=tp_price,
        sl_price=sl_price,
    )

    actions = st.columns([1, 1, 4])
    with actions[0]:
        do_validate = st.button("Validate", use_container_width=True)
    with actions[1]:
        do_place = st.button("🚀 Place Order", use_container_width=True, disabled=(not valid))

    if do_validate:
        if not valid:
            st.error(msg)

        payload_preview: Dict[str, Any] = {
            "mode": mode,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": tif,
            "sizing": sizing,
            "qty": qty_val,
            "notional": notional_val,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "order_class": order_class,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "client_order_id": client_order_id,
            "dry_run": dry_run,
            "blocked": blocked,
        }
        st.json(payload_preview)

        if not valid:
            st.warning("Fix validation errors above before placing.")
        elif blocked and not dry_run:
            st.error("Execution is BLOCKED — reconcile/guard freeze active.")
        else:
            st.success("Validation looks OK.")

    if do_place:
        if not valid:
            st.error(msg)
            return

        if blocked and not dry_run:
            st.error("Execution is BLOCKED — reconcile/guard freeze active.")
            with st.expander("Guard snapshot", expanded=False):
                st.json(guard)
            return

        try:
            broker = AlpacaBroker(mode=mode)
        except Exception as e:
            st.error(f"Broker init failed: {e}")
            return

        # Reconcile pre (even if dry_run, we allow; but do not block)
        try:
            rr = reconcile_state(broker, phase="manual_pre", source="manual_order_desk")
            if not rr.ok and not dry_run:
                st.error(rr.message)
                return
        except Exception as e:
            if not dry_run:
                st.error(f"Reconcile failed: {e}")
                return

        ts = _utc_now_iso()

        try:
            if dry_run:
                st.info("Dry run: no order sent to Alpaca.")
                _append_live_order(
                    {
                        "ts_utc": ts,
                        "mode": mode,
                        "symbol": symbol,
                        "side": side,
                        "order_type": order_type,
                        "tif": tif,
                        "sizing": sizing,
                        "qty": qty_val,
                        "notional": notional_val,
                        "limit_price": limit_price,
                        "stop_price": stop_price,
                        "order_class": order_class,
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                        "client_order_id": client_order_id or "",
                        "alpaca_order_id": "",
                        "status": "DRY_RUN",
                        "raw": "",
                    }
                )
                st.success("Dry run logged to live_orders.csv")
                return

            # Real placement
            if sizing == "qty":
                if order_class == "bracket":
                    resp = broker.place_bracket_market(
                        symbol=symbol,
                        qty=float(qty_val or 0.0),
                        side=side,
                        take_profit_price=float(tp_price or 0.0),
                        stop_loss_price=float(sl_price or 0.0),
                        tif=tif,
                        client_order_id=client_order_id,
                    )
                else:
                    resp = broker.submit_order(
                        symbol=symbol,
                        qty=float(qty_val or 0.0),
                        side=side,
                        order_type=order_type,
                        time_in_force=tif,
                        limit_price=limit_price,
                        stop_price=stop_price,
                        client_order_id=client_order_id,
                    )
            else:
                if order_type != "market":
                    st.error(
                        "Notional sizing supports market orders only. Switch Type to market or use qty sizing."
                    )
                    return
                resp = broker.submit_order_notional(
                    symbol=symbol,
                    notional=float(notional_val or 0.0),
                    side=side,
                    time_in_force=tif,
                    client_order_id=client_order_id,
                )

            st.success("Order submitted.")
            st.json(resp)

            _append_live_order(
                {
                    "ts_utc": ts,
                    "mode": mode,
                    "symbol": symbol,
                    "side": side,
                    "order_type": order_type,
                    "tif": tif,
                    "sizing": sizing,
                    "qty": qty_val,
                    "notional": notional_val,
                    "limit_price": limit_price,
                    "stop_price": stop_price,
                    "order_class": order_class,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "client_order_id": client_order_id or "",
                    "alpaca_order_id": (resp or {}).get("id", ""),
                    "status": (resp or {}).get("status", ""),
                    "raw": str(resp)[:2000],
                }
            )

            # Reconcile post
            try:
                rr2 = reconcile_state(broker, phase="manual_post", source="manual_order_desk")
                if not rr2.ok:
                    st.warning(rr2.message)
            except Exception:
                pass

        except AlpacaError as e:
            st.error(f"Alpaca error: {e}")
        except Exception as e:
            st.error(f"Order failed: {e}")


if __name__ == "__main__":
    render_manual_order_desk()
