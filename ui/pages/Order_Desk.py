#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pages/Order_Desk.py — TRITON Manual Order Desk (Capital Preservation First)

Features:
- Loads .env automatically (Streamlit-safe)
- Market clock panel
- Open orders panel (with pending_cancel detection)
- Cancel all orders (with confirmation)
- Load CSV orders + preview
- DRY RUN placement preview
- REAL placement button (guarded by Order Hygiene Gate)
- Audit log: data/results/live_orders.csv

CSV columns expected:
  ticker|sym|symbol, side, qty, close
Optional:
  limit_price, order_type, tif, extended_hours, client_order_id
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


# ──────────────────────────────
# Root + .env loading
# ──────────────────────────────
def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
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
DOTENV_PATH = PROJECT_ROOT / ".env"

DATA_DIR = PROJECT_ROOT / "data"
ORDERS_DIR = DATA_DIR / "orders"
RESULTS_DIR = DATA_DIR / "results"
AUDIT_LOG_PATH = RESULTS_DIR / "live_orders.csv"


def load_env_silent() -> None:
    """Load .env if python-dotenv is installed. Not fatal if missing."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    if DOTENV_PATH.exists():
        load_dotenv(DOTENV_PATH, override=True)


# ──────────────────────────────
# Alpaca REST helpers
# ──────────────────────────────
def _first_env(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v and str(v).strip():
            return str(v).strip()
    return ""


def alpaca_base_url() -> str:
    base = _first_env("APCA_API_BASE_URL", "ALPACA_BASE_URL", "ALPACA_ENDPOINT")
    if not base:
        base = "https://paper-api.alpaca.markets"
    return base.rstrip("/")


def alpaca_headers() -> Dict[str, str]:
    key = _first_env("APCA_API_KEY_ID", "ALPACA_API_KEY", "ALPACA_KEY_ID")
    sec = _first_env("APCA_API_SECRET_KEY", "ALPACA_API_SECRET", "ALPACA_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError(
            "Missing Alpaca credentials.\n"
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (recommended via .env at project root)."
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def mask(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    return s[:4] + "…" + s[-2:] if len(s) > 6 else (s[:2] + "…" if len(s) > 2 else "****")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mode_from_base(base: str) -> str:
    return "paper" if "paper-api" in base else "live"


def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    base = alpaca_base_url()
    H = alpaca_headers()
    r = requests.get(f"{base}{path}", params=params or {}, headers=H, timeout=30)
    if r.status_code == 401:
        raise RuntimeError(
            "401 Unauthorized from Alpaca.\n"
            "This usually means your Streamlit process didn’t load .env.\n"
            f"base={base} key={mask(H.get('APCA-API-KEY-ID',''))}"
        )
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: Dict[str, Any]) -> Tuple[int, Any]:
    base = alpaca_base_url()
    H = alpaca_headers()
    r = requests.post(f"{base}{path}", json=payload, headers=H, timeout=30)
    try:
        data = r.json()
    except Exception:
        data = r.text
    return r.status_code, data


def api_delete(path: str) -> Tuple[int, Any]:
    base = alpaca_base_url()
    H = alpaca_headers()
    r = requests.delete(f"{base}{path}", headers=H, timeout=30)
    try:
        data = r.json()
    except Exception:
        data = r.text
    return r.status_code, data


# ──────────────────────────────
# CSV Orders
# ──────────────────────────────
@dataclass
class OrderRow:
    symbol: str
    side: str  # BUY/SELL
    qty: float
    close: float
    order_type: str  # market/limit
    tif: str  # day/gtc/...
    limit_price: Optional[float] = None
    extended_hours: bool = False
    client_order_id: Optional[str] = None


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _as_bool(x: Any) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _pick_symbol(row: Dict[str, Any], fieldnames: List[str]) -> str:
    first_key = fieldnames[0] if fieldnames else ""
    raw = (
        row.get("ticker")
        or row.get("sym")
        or row.get("symbol")
        or row.get("Symbol")
        or (row.get(first_key) if first_key else "")
        or ""
    )
    return str(raw).strip().upper()


def load_orders(csv_path: Path, default_order_type: str, default_tif: str) -> List[OrderRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if csv_path.stat().st_size == 0:
        raise RuntimeError(f"CSV is empty: {csv_path}")

    out: List[OrderRow] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("CSV has no header row / fieldnames.")

        fieldnames = list(reader.fieldnames)

        for i, row in enumerate(reader, start=1):
            symbol = _pick_symbol(row, fieldnames)
            side = str(row.get("side") or "BUY").strip().upper()

            qty = _as_float(row.get("qty"))
            close = _as_float(row.get("close"))

            if not symbol or side not in ("BUY", "SELL"):
                continue
            if qty is None or close is None or qty <= 0 or close <= 0:
                continue

            order_type = str(row.get("order_type") or default_order_type).strip().lower()
            tif = str(row.get("tif") or default_tif).strip().lower()

            limit_price = _as_float(row.get("limit_price"))
            extended_hours = _as_bool(row.get("extended_hours"))
            client_order_id = (row.get("client_order_id") or "").strip() or None

            if limit_price is not None and order_type == "market":
                order_type = "limit"

            out.append(
                OrderRow(
                    symbol=symbol,
                    side=side,
                    qty=float(qty),
                    close=float(close),
                    order_type=order_type,
                    tif=tif,
                    limit_price=limit_price,
                    extended_hours=extended_hours,
                    client_order_id=client_order_id,
                )
            )

    return out


def _format_qty(qty: float) -> str:
    return str(int(qty)) if float(qty).is_integer() else str(qty)


# ──────────────────────────────
# Audit log
# ──────────────────────────────
def ensure_audit_header() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not AUDIT_LOG_PATH.exists():
        with AUDIT_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "ts_utc",
                    "mode",
                    "action",
                    "symbol",
                    "side",
                    "qty",
                    "order_type",
                    "tif",
                    "limit_price",
                    "extended_hours",
                    "client_order_id",
                    "status",
                    "order_id",
                    "raw_response",
                ]
            )


def append_audit(
    mode: str,
    action: str,
    o: OrderRow,
    status: str,
    order_id: str = "",
    raw_response: Any = "",
) -> None:
    ensure_audit_header()
    with AUDIT_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                utc_iso(),
                mode,
                action,
                o.symbol,
                o.side,
                f"{o.qty:.6f}",
                o.order_type,
                o.tif,
                "" if o.limit_price is None else f"{o.limit_price:.6f}",
                str(bool(o.extended_hours)),
                o.client_order_id or "",
                status,
                order_id,
                (
                    json.dumps(raw_response)
                    if isinstance(raw_response, (dict, list))
                    else str(raw_response)
                ),
            ]
        )


# ──────────────────────────────
# Order Hygiene Gate
# ──────────────────────────────
def order_signature_from_open(o: Dict[str, Any]) -> str:
    # Normalize to detect duplicates
    sym = str(o.get("symbol", "")).upper()
    side = str(o.get("side", "")).lower()
    typ = str(o.get("type", "")).lower()  # market/limit
    tif = str(o.get("time_in_force", "")).lower()
    lim = o.get("limit_price")
    lim_s = "" if lim is None else str(lim)
    return f"{sym}|{side}|{typ}|{tif}|{lim_s}"


def order_signature_from_csv(o: OrderRow) -> str:
    sym = o.symbol.upper()
    side = o.side.lower()
    typ = o.order_type.lower()
    tif = o.tif.lower()
    lim_s = "" if o.limit_price is None else str(o.limit_price)
    # if limit order but no limit price, we’ll use close as implicit limit for signature
    if typ == "limit" and (o.limit_price is None):
        lim_s = str(o.close)
    return f"{sym}|{side}|{typ}|{tif}|{lim_s}"


def gate_check(
    clock: Dict[str, Any], open_orders: List[Dict[str, Any]], csv_orders: List[OrderRow]
) -> Tuple[bool, List[str], List[str]]:
    """
    Returns: (allowed, blockers, warnings)
    """
    blockers: List[str] = []
    warnings: List[str] = []

    is_open = bool(clock.get("is_open"))
    pending_cancel = [
        o for o in open_orders if str(o.get("status", "")).lower() == "pending_cancel"
    ]
    if pending_cancel:
        blockers.append(
            f"{len(pending_cancel)} open orders are pending_cancel. Wait until Alpaca finalizes cancels (usually after market opens)."
        )

    # block market orders when market is closed
    if not is_open:
        market_in_csv = [o for o in csv_orders if o.order_type.lower() == "market"]
        if market_in_csv:
            blockers.append(
                f"Market is closed and you have {len(market_in_csv)} market orders in the CSV. Convert to limit or wait for open."
            )

    # duplicates: CSV vs currently open
    open_sigs = {order_signature_from_open(o) for o in open_orders}
    dupes = [o for o in csv_orders if order_signature_from_csv(o) in open_sigs]
    if dupes:
        blockers.append(
            f"Duplicate protection: {len(dupes)} CSV orders match existing open orders (same symbol/side/type/tif/limit)."
        )

    # warnings (non-blockers)
    if not csv_orders:
        blockers.append("No valid orders loaded from CSV.")

    return (len(blockers) == 0), blockers, warnings


# ──────────────────────────────
# Placement
# ──────────────────────────────
def build_payload(o: OrderRow) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "symbol": o.symbol,
        "qty": _format_qty(o.qty),
        "side": o.side.lower(),
        "type": o.order_type.lower(),
        "time_in_force": o.tif.lower(),
    }
    if payload["type"] == "limit":
        payload["limit_price"] = str(o.limit_price if o.limit_price is not None else o.close)
    if o.extended_hours:
        payload["extended_hours"] = True
    if o.client_order_id:
        payload["client_order_id"] = o.client_order_id
    return payload


def place_one(o: OrderRow, dry_run: bool) -> Tuple[str, str, Any, Dict[str, Any]]:
    payload = build_payload(o)
    if dry_run:
        return "DRY_RUN", "", payload, payload
    code, data = api_post("/v2/orders", payload)
    if 200 <= code < 300 and isinstance(data, dict):
        return "PLACED", str(data.get("id", "")), data, payload
    return f"ERROR_{code}", "", data, payload


# ──────────────────────────────
# UI
# ──────────────────────────────
st.set_page_config(page_title="TRITON — Order Desk", layout="wide")

load_env_silent()

st.title("🧾 TRITON — Manual Order Desk")
st.caption("Capital Preservation First • Order Hygiene Gate enabled")

# Sidebar settings
st.sidebar.header("Settings")
default_csv = str((ORDERS_DIR / "orders_today.csv").resolve())
csv_path_str = st.sidebar.text_input("CSV path", value=default_csv)
default_order_type = st.sidebar.selectbox("Default order type", ["market", "limit"], index=0)
default_tif = st.sidebar.selectbox(
    "Default TIF", ["day", "gtc", "opg", "cls", "ioc", "fok"], index=0
)
max_orders = st.sidebar.number_input("Max rows", min_value=1, max_value=5000, value=500, step=1)

base = alpaca_base_url()
mode = mode_from_base(base)

# Top: account + clock
colA, colB, colC = st.columns([1.2, 1.2, 1.6])

with colA:
    st.subheader("Broker Mode")
    st.code(f"{mode.upper()}  •  {base}", language="text")

with colB:
    st.subheader("Market Clock")
    try:
        clock = api_get("/v2/clock")
        is_open = bool(clock.get("is_open"))
        st.metric("Market Open", "YES" if is_open else "NO")
        st.write(f"**Next Open:** {clock.get('next_open')}")
        st.write(f"**Next Close:** {clock.get('next_close')}")
        st.write(f"**Timestamp:** {clock.get('timestamp')}")
    except Exception as e:
        st.error(f"Clock error: {e}")
        clock = {"is_open": False}

with colC:
    st.subheader("Account Snapshot")
    try:
        acct = api_get("/v2/account")
        bp = acct.get("buying_power") or acct.get("cash")
        pv = acct.get("portfolio_value") or acct.get("equity")
        st.write(f"**Status:** {acct.get('status')}")
        st.write(f"**Buying Power:** {bp}")
        st.write(f"**Portfolio Value:** {pv}")
        st.write(f"**Account #:** {acct.get('account_number')}")
    except Exception as e:
        st.error(f"Account error: {e}")

st.divider()

# Open Orders + Cancel All
st.subheader("📌 Open Orders")

open_orders: List[Dict[str, Any]] = []
try:
    open_orders = api_get("/v2/orders", params={"status": "open", "limit": 500, "nested": "true"})
except Exception as e:
    st.error(f"Failed to fetch open orders: {e}")

pending_cancel = [o for o in open_orders if str(o.get("status", "")).lower() == "pending_cancel"]
accepted = [o for o in open_orders if str(o.get("status", "")).lower() == "accepted"]

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
c1.metric("Open Orders", str(len(open_orders)))
c2.metric("pending_cancel", str(len(pending_cancel)))
c3.metric("accepted", str(len(accepted)))

with c4:
    with st.expander("Cancel All Orders (Guarded)", expanded=False):
        st.warning(
            "This sends DELETE /v2/orders. Alpaca may return 207 if some are already pending_cancel."
        )
        confirm_cancel = st.checkbox("I understand. Cancel all open orders.", value=False)
        if st.button("🛑 CANCEL ALL", disabled=not confirm_cancel):
            code, data = api_delete("/v2/orders")
            st.write(f"Result: **{code}**")
            st.code(
                json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data),
                language="json",
            )
            st.success("Cancel request sent. Pending cancels may persist until market opens.")

if open_orders:
    # Render small table
    rows = []
    for o in open_orders[:200]:
        rows.append(
            {
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "type": o.get("type"),
                "tif": o.get("time_in_force"),
                "status": o.get("status"),
                "limit_price": o.get("limit_price"),
                "qty": o.get("qty"),
                "filled_qty": o.get("filled_qty"),
                "id": o.get("id"),
            }
        )
    st.dataframe(rows, use_container_width=True, height=280)
else:
    st.info("No open orders found.")

st.divider()

# Load CSV orders
st.subheader("📄 CSV Orders")

csv_path = Path(csv_path_str).expanduser()
csv_orders: List[OrderRow] = []
csv_error = None

try:
    csv_orders = load_orders(
        csv_path, default_order_type=default_order_type, default_tif=default_tif
    )[: int(max_orders)]
except Exception as e:
    csv_error = str(e)

if csv_error:
    st.error(csv_error)
else:
    st.success(f"Loaded **{len(csv_orders)}** valid orders from: {csv_path.resolve()}")

    preview = []
    for o in csv_orders:
        preview.append(
            {
                "symbol": o.symbol,
                "side": o.side,
                "qty": o.qty,
                "close": o.close,
                "order_type": o.order_type,
                "tif": o.tif,
                "limit_price": o.limit_price,
                "extended_hours": o.extended_hours,
                "client_order_id": o.client_order_id,
                "notional_est": round(o.qty * o.close, 2),
            }
        )
    st.dataframe(preview, use_container_width=True, height=260)

# Gate check
allowed, blockers, warnings = gate_check(
    clock=clock, open_orders=open_orders, csv_orders=csv_orders
)

st.subheader("🧱 Order Hygiene Gate")
if blockers:
    for b in blockers:
        st.error(b)
else:
    st.success("Gate PASSED — placement allowed.")

for w in warnings:
    st.warning(w)

st.divider()

# Actions: DRY RUN / REAL PLACE
left, right = st.columns([1, 1])

with left:
    st.subheader("✅ DRY RUN")
    if st.button("Run DRY RUN", disabled=not bool(csv_orders)):
        outputs = []
        for o in csv_orders:
            status, oid, raw, payload = place_one(o, dry_run=True)
            outputs.append(
                {"symbol": o.symbol, "side": o.side, "status": status, "payload": payload}
            )
            append_audit(
                mode=mode,
                action="dry-run",
                o=o,
                status=status,
                order_id=oid,
                raw_response={"payload": payload},
            )
        st.success("Dry run complete.")
        st.code(json.dumps(outputs[:50], indent=2), language="json")

with right:
    st.subheader("🚀 REAL PLACE (Guarded)")
    st.warning("This submits live orders to Alpaca (paper/live depending on APCA_API_BASE_URL).")

    confirm_phrase = st.text_input("Type EXACTLY: PLACE ORDERS", value="")
    ok_phrase = confirm_phrase.strip() == "PLACE ORDERS"

    can_place = allowed and ok_phrase and bool(csv_orders)

    if st.button("PLACE NOW", disabled=not can_place):
        results = []
        for o in csv_orders:
            status, oid, raw, payload = place_one(o, dry_run=False)
            results.append(
                {"symbol": o.symbol, "side": o.side, "status": status, "id": oid, "response": raw}
            )
            append_audit(
                mode=mode,
                action="submit" if status == "PLACED" else "error",
                o=o,
                status=status,
                order_id=oid,
                raw_response={"payload": payload, "response": raw},
            )

        placed = [r for r in results if r["status"] == "PLACED"]
        errors = [r for r in results if r["status"] != "PLACED"]

        if placed:
            st.success(f"Placed {len(placed)} orders.")
        if errors:
            st.error(f"{len(errors)} orders failed.")
        st.code(json.dumps(results, indent=2), language="json")

st.caption(f"Audit log: {AUDIT_LOG_PATH.resolve()}")
