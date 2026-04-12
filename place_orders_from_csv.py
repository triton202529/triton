#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
place_orders_from_csv.py — TRITON CSV Executor (Clean + Loud)
UPDATED: Phase 2.3 — ExecutionGuard + Reconciliation enforced

Pipeline (REAL placement):
  reconcile(pre) -> guard.validate_and_audit -> submit -> live_orders log -> reconcile(post)

Notes:
- Keeps the existing Alpaca REST placement approach (requests), but uses AlpacaBroker for:
  - ExecutionGuard validation (latest price / BP checks / cooldown / bracket sanity)
  - Reconciliation (reconcile_or_freeze)
- Writes to unified data/results/live_orders.csv schema (same as place_live_orders.py)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# ──────────────────────────────
# Project root + .env loading
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
DOTENV_PATH = PROJECT_ROOT / ".env"

# Make "services.*" importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_env_loud() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        if DOTENV_PATH.exists():
            print(
                f"[WARN] .env exists at {DOTENV_PATH} but python-dotenv is not installed. "
                "Install it: pip install python-dotenv",
                flush=True,
            )
        else:
            print("[env] python-dotenv not installed; using system environment vars", flush=True)
        return

    if DOTENV_PATH.exists():
        load_dotenv(DOTENV_PATH, override=True)
        print(f"[env] loaded .env (override=True) -> {DOTENV_PATH}", flush=True)
    else:
        print(f"[env] no .env found at {DOTENV_PATH} (using system environment vars)", flush=True)


# ──────────────────────────────
# Paths / Unified log schema (matches services/place_live_orders.py)
# ──────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LIVE_ORDERS_LOG = RESULTS_DIR / "live_orders.csv"

EXPECTED_COLS = [
    "timestamp",
    "session",
    "action",
    "symbol",
    "side",
    "qty",
    "type",
    "limit_price",
    "order_id",
    "status",
    "filled_qty",
    "filled_avg_price",
    "client_order_id",
    "tp_limit",
    "sl_stop",
]


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_live_orders_schema() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not LIVE_ORDERS_LOG.exists():
        with LIVE_ORDERS_LOG.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(EXPECTED_COLS)
        return

    try:
        with LIVE_ORDERS_LOG.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            with LIVE_ORDERS_LOG.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(EXPECTED_COLS)
            return
        if rows[0] == EXPECTED_COLS:
            return

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup = RESULTS_DIR / f"live_orders.backup.{ts}.csv"
        backup.write_text(
            LIVE_ORDERS_LOG.read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
        )

        with LIVE_ORDERS_LOG.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(EXPECTED_COLS)
            for r in rows[1:]:
                new_row = (r + [""] * len(EXPECTED_COLS))[: len(EXPECTED_COLS)]
                w.writerow(new_row)

        print(f"[log] Upgraded live_orders.csv schema. Backup -> {backup}", flush=True)

    except Exception as e:
        print(f"[log] Schema check/upgrade skipped: {e}", flush=True)


def _append_live_orders_row(row: List[Any]) -> None:
    _ensure_live_orders_schema()
    with LIVE_ORDERS_LOG.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def _append_event(session: str, action: str, status: str, message: str) -> None:
    # Uses sl_stop column as a generic message bucket for non-order events
    _append_live_orders_row(
        [
            _utc_now_iso_z(),
            session,
            action,
            "",
            "",
            0,
            "",
            "",
            "",
            status,
            0,
            "",
            "",
            "",
            message,
        ]
    )


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
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (recommended via .env at project root).\n"
            f"Expected .env path: {DOTENV_PATH}"
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def mask(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    if len(s) <= 4:
        return "****"
    if len(s) <= 6:
        return s[:2] + "…"
    return s[:4] + "…" + s[-2:]


def mode_from_base(base: str) -> str:
    return "paper" if "paper-api" in (base or "") else "live"


def _req_json(
    method: str, url: str, headers: Dict[str, str], *, json_body: Any = None, timeout: int = 30
) -> Any:
    if method.upper() == "GET":
        r = requests.get(url, headers=headers, timeout=timeout)
    elif method.upper() == "POST":
        r = requests.post(url, headers=headers, json=json_body, timeout=timeout)
    elif method.upper() == "DELETE":
        r = requests.delete(url, headers=headers, timeout=timeout)
    else:
        raise ValueError(f"Unsupported method: {method}")

    try:
        payload = r.json()
    except Exception:
        payload = r.text

    if r.status_code == 401:
        raise RuntimeError(
            "401 Unauthorized from Alpaca.\n"
            "Missing/invalid API keys in THIS process.\n"
            f"base={alpaca_base_url()} key={mask(headers.get('APCA-API-KEY-ID',''))}\n"
            f"response={payload}"
        )

    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"Alpaca error {r.status_code}: {payload}")

    return payload


def probe_account(verbose: bool = True) -> Dict[str, Any]:
    base = alpaca_base_url()
    H = alpaca_headers()
    acct = _req_json("GET", f"{base}/v2/account", H, timeout=20)

    if verbose:
        bp = acct.get("buying_power") or acct.get("cash")
        pv = acct.get("portfolio_value") or acct.get("equity")
        print(
            f"[acct] mode={mode_from_base(base)} base={base} key={mask(H.get('APCA-API-KEY-ID',''))} "
            f"status={acct.get('status')} buying_power={bp} portfolio_value={pv}",
            flush=True,
        )
    return acct


def fetch_positions() -> List[Dict[str, Any]]:
    base = alpaca_base_url()
    H = alpaca_headers()
    data = _req_json("GET", f"{base}/v2/positions", H, timeout=20)
    return data if isinstance(data, list) else []


def list_open_orders(nested: bool = True, limit: int = 500) -> List[Dict[str, Any]]:
    base = alpaca_base_url()
    H = alpaca_headers()
    url = f"{base}/v2/orders?status=open&limit={int(limit)}"
    if nested:
        url += "&nested=true"
    data = _req_json("GET", url, H, timeout=20)
    return data if isinstance(data, list) else []


def cancel_order(order_id: str) -> None:
    base = alpaca_base_url()
    H = alpaca_headers()
    _req_json("DELETE", f"{base}/v2/orders/{order_id}", H, timeout=20)


def get_clock() -> Optional[Dict[str, Any]]:
    base = alpaca_base_url()
    H = alpaca_headers()
    try:
        data = _req_json("GET", f"{base}/v2/clock", H, timeout=15)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def clock_is_open(clock: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not clock:
        return None
    v = clock.get("is_open", None)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y")
    return None


# ──────────────────────────────
# Closed-market forced limit helpers
# ──────────────────────────────
def _forced_limit_price(px: float, side: str, offset_bps: float) -> float:
    px = max(float(px), 1e-9)
    bps = max(0.0, float(offset_bps)) / 10000.0
    if side.upper() == "BUY":
        return px * (1.0 - bps)
    return px * (1.0 + bps)


# ──────────────────────────────
# CSV Orders
# ──────────────────────────────
@dataclass
class OrderRow:
    symbol: str
    side: str
    qty: float
    close: float
    order_type: str
    tif: str
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
                print(f"[skip] row {i}: bad symbol/side -> {row}", flush=True)
                continue
            if qty is None or close is None or qty <= 0 or close <= 0:
                print(f"[skip] row {i}: bad qty/close -> {row}", flush=True)
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


# ──────────────────────────────
# Risk Gate helpers (kept)
# ──────────────────────────────
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def build_position_maps(positions: List[Dict[str, Any]]) -> Tuple[Dict[str, float], float]:
    mv_by_sym: Dict[str, float] = {}
    gross = 0.0
    for p in positions:
        sym = str(p.get("symbol") or "").upper().strip()
        mv = abs(_safe_float(p.get("market_value"), 0.0))
        if sym:
            mv_by_sym[sym] = mv
        gross += mv
    return mv_by_sym, float(gross)


def clamp_order_by_risk_caps(
    o: OrderRow,
    *,
    equity: float,
    buying_power: float,
    reserve_pct: float,
    max_position_weight: float,
    max_gross_exposure: float,
    current_gross_mv: float,
    current_sym_mv: float,
) -> Tuple[OrderRow, str]:
    if o.side != "BUY" or equity <= 0:
        return o, ""

    reserve_pct = max(0.0, min(1.0, float(reserve_pct)))
    effective_bp = max(0.0, buying_power * (1.0 - reserve_pct))

    px = float(
        o.limit_price if (o.order_type == "limit" and o.limit_price is not None) else o.close
    )
    px = max(px, 1e-9)

    cap_pos_dollars = equity * max(0.0, float(max_position_weight))
    room_pos = max(0.0, cap_pos_dollars - max(0.0, current_sym_mv))
    if room_pos <= 0:
        return o, "risk_cap_position:room=0"

    cap_gross_dollars = equity * max(0.0, float(max_gross_exposure))
    room_gross = max(0.0, cap_gross_dollars - max(0.0, current_gross_mv))
    if room_gross <= 0:
        return o, "risk_cap_gross:room=0"

    room_bp = max(0.0, effective_bp)
    room = min(room_pos, room_gross, room_bp)

    desired_notional = float(o.qty) * px
    if desired_notional <= room:
        return o, ""

    max_qty = int(room // px)
    if max_qty <= 0:
        return o, f"risk_cap_qty_to_zero (room={room:.2f} px={px:.2f})"

    if max_qty < int(o.qty):
        note = f"reduced_qty {int(o.qty)}->{max_qty} (room={room:.2f})"
        o2 = OrderRow(
            symbol=o.symbol,
            side=o.side,
            qty=float(max_qty),
            close=o.close,
            order_type=o.order_type,
            tif=o.tif,
            limit_price=o.limit_price,
            extended_hours=o.extended_hours,
            client_order_id=o.client_order_id,
        )
        return o2, note

    return o, ""


# ──────────────────────────────
# Duplicate detection helpers (tolerance-aware)
# ──────────────────────────────
def _norm_price(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _int_qty(v: Any) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


def intended_symbol_side_key(o: OrderRow) -> Tuple[str, str]:
    return (o.symbol.upper(), o.side.lower())


def open_symbol_side_key(o: Dict[str, Any]) -> Tuple[str, str]:
    return (str(o.get("symbol") or "").upper(), str(o.get("side") or "").lower())


def intended_dupe_signature(o: OrderRow) -> Tuple[str, str, int, str, Optional[float]]:
    otype = str(o.order_type or "market").lower()
    lim = None
    if otype == "limit":
        lim = float(o.limit_price if o.limit_price is not None else o.close)
    return (o.symbol.upper(), o.side.lower(), int(float(o.qty)), otype, lim)


def strict_dupe_match_with_tol(
    open_order: Dict[str, Any], intended: OrderRow, *, price_tol: float
) -> bool:
    sym = str(open_order.get("symbol") or "").upper()
    side = str(open_order.get("side") or "").lower()
    qty = _int_qty(open_order.get("qty", 0))
    otype = str(open_order.get("type") or open_order.get("order_type") or "").lower()

    i_sym, i_side, i_qty, i_type, i_lim = intended_dupe_signature(intended)
    if sym != i_sym or side != i_side or qty != i_qty or otype != i_type:
        return False

    if i_type == "limit":
        o_lim = _norm_price(open_order.get("limit_price"))
        if o_lim is None or i_lim is None:
            return False
        return abs(float(o_lim) - float(i_lim)) <= float(price_tol)

    return True


def find_dupes(
    open_orders: List[Dict[str, Any]], intended: OrderRow, *, dupe_mode: str, price_tol: float
) -> List[Dict[str, Any]]:
    if dupe_mode == "symbol":
        k = intended_symbol_side_key(intended)
        return [oo for oo in open_orders if open_symbol_side_key(oo) == k]
    return [
        oo for oo in open_orders if strict_dupe_match_with_tol(oo, intended, price_tol=price_tol)
    ]


def describe_order_short(o: Dict[str, Any]) -> str:
    sym = str(o.get("symbol") or "").upper()
    side = str(o.get("side") or "").lower()
    qty = _int_qty(o.get("qty", 0))
    otype = str(o.get("type") or "").lower()
    lim = o.get("limit_price")
    status = str(o.get("status") or "").lower()
    oid = str(o.get("id") or "")
    created = str(o.get("created_at") or "")
    extra = f" limit={lim}" if lim not in (None, "", "0") else ""
    return (
        f"id={oid} {sym} {side} qty={qty} type={otype}{extra} status={status} created_at={created}"
    )


# ──────────────────────────────
# Guard + Reconciliation wrappers
# ──────────────────────────────
def _require_reconcile() -> Any:
    try:
        from services.reconciliation import reconcile_or_freeze  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing services.reconciliation.reconcile_or_freeze. "
            "Ledger/Reconciliation engine must be present for real execution."
        ) from e
    return reconcile_or_freeze


def _run_reconcile(session: str, *, phase: str, hard_stop: bool, source: str, mode: str) -> None:
    reconcile_or_freeze = _require_reconcile()
    try:
        reconcile_or_freeze(mode=mode, source=source, phase=phase, hard_stop=hard_stop)
        _append_event(session, f"reconcile-{phase}", "OK", f"{source} {mode}")
    except Exception as e:
        msg = f"Reconcile {phase} FAILED: {e}"
        _append_event(session, f"reconcile-{phase}", "FAIL", msg)
        if hard_stop:
            raise


# ──────────────────────────────
# Place Orders (REST submit, but guard validates via AlpacaBroker)
# ──────────────────────────────
def _format_qty(qty: float) -> str:
    return str(int(qty)) if float(qty).is_integer() else str(qty)


def build_guard_payload(o: OrderRow) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "symbol": o.symbol,
        "side": o.side.lower(),
        "type": o.order_type,
        "time_in_force": o.tif,
        "qty": float(o.qty),
    }
    if o.order_type == "limit":
        payload["limit_price"] = float(o.limit_price if o.limit_price is not None else o.close)
    if o.client_order_id:
        payload["client_order_id"] = o.client_order_id
    return payload


def place_order(o: OrderRow, dry_run: bool) -> Tuple[str, str, Any, Dict[str, Any]]:
    base = alpaca_base_url()
    H = alpaca_headers()

    payload: Dict[str, Any] = {
        "symbol": o.symbol,
        "qty": _format_qty(o.qty),
        "side": o.side.lower(),
        "type": o.order_type,
        "time_in_force": o.tif,
    }

    if o.order_type == "limit":
        payload["limit_price"] = str(o.limit_price if o.limit_price is not None else o.close)

    if o.extended_hours:
        payload["extended_hours"] = True

    if o.client_order_id:
        payload["client_order_id"] = o.client_order_id

    if dry_run:
        return "DRY_RUN", "", payload, payload

    raw = _req_json("POST", f"{base}/v2/orders", H, json_body=payload, timeout=30)
    oid = str(raw.get("id", "")) if isinstance(raw, dict) else ""
    status = str(raw.get("status", "")) if isinstance(raw, dict) else "unknown"
    return status or "PLACED", oid, raw, payload


def log_order_row(
    session: str,
    action: str,
    o: OrderRow,
    *,
    status: str,
    order_id: str = "",
    filled_qty: Any = "",
    filled_avg_price: Any = "",
    tp_limit: Any = "",
    sl_stop: Any = "",
) -> None:
    _append_live_orders_row(
        [
            _utc_now_iso_z(),
            session,
            action,
            o.symbol,
            o.side.lower(),
            int(float(o.qty)),
            o.order_type,
            (
                float(o.limit_price)
                if (o.order_type == "limit" and o.limit_price is not None)
                else ""
            ),
            order_id,
            status,
            filled_qty if filled_qty not in (None, "") else 0,
            filled_avg_price if filled_avg_price not in (None, "") else "",
            o.client_order_id or "",
            tp_limit or "",
            sl_stop or "",
        ]
    )


# ──────────────────────────────
# Main
# ──────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        prog="place_orders_from_csv.py",
        description="TRITON CSV Executor (DRY RUN default; use --really-place to submit).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="Path to CSV orders file.")
    parser.add_argument(
        "--order-type", default="market", choices=["market", "limit"], help="Default order type."
    )
    parser.add_argument(
        "--tif",
        default="day",
        choices=["day", "gtc", "opg", "cls", "ioc", "fok"],
        help="Default TIF.",
    )
    parser.add_argument(
        "--really-place", action="store_true", help="Actually submit orders (otherwise DRY RUN)."
    )
    parser.add_argument(
        "--max-orders", type=int, default=9999, help="Limit number of rows processed."
    )

    # Safety parity flags
    parser.add_argument(
        "--require-market-open",
        action="store_true",
        help="Block REAL placement if market is closed.",
    )
    parser.add_argument(
        "--allow-market-closed", action="store_true", help="Override market-open gate."
    )
    parser.add_argument(
        "--ignore-pending-cancel", action="store_true", help="Override pending_cancel safety gate."
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow placing even if a duplicate open order exists.",
    )
    parser.add_argument(
        "--cancel-open-for-symbols",
        action="store_true",
        help="Cancel any open orders for symbols being placed.",
    )

    # Duplicate mode
    parser.add_argument(
        "--dupe-mode",
        default="strict",
        choices=["strict", "symbol"],
        help="Duplicate detection: strict=(symbol,side,qty,type,limit±tol) vs symbol=(symbol,side) blocks any extra open order.",
    )
    parser.add_argument(
        "--dupe-price-tol",
        type=float,
        default=0.01,
        help="Strict dupe mode: abs tol for limit_price (0.01 = 1c).",
    )
    parser.add_argument(
        "--on-dupe",
        default="skip",
        choices=["skip", "cancel", "replace"],
        help="skip | cancel dupes only | replace dupes then place",
    )
    parser.add_argument(
        "--force-replace",
        action="store_true",
        help="In replace mode: replace even if identical already exists.",
    )
    parser.add_argument(
        "--cancel-settle-retries",
        type=int,
        default=8,
        help="After canceling: refresh attempts to ensure dupes cleared.",
    )
    parser.add_argument(
        "--cancel-settle-sleep",
        type=float,
        default=0.25,
        help="Sleep between cancel settle refresh attempts.",
    )

    # Closed-market forced limit behavior
    parser.add_argument(
        "--force-limit-when-closed",
        action="store_true",
        help="If market is closed and allowed, convert market->limit resting.",
    )
    parser.add_argument(
        "--limit-offset-bps",
        type=float,
        default=10.0,
        help="Forced limit offset bps (10=0.10%). BUY below px, SELL above px.",
    )

    args = parser.parse_args()

    session = datetime.now().strftime("%Y%m%d-%H%M%S")
    source = "csv_executor"

    print(">>> place_orders_from_csv.py STARTED <<<", flush=True)
    print(f"[root] {PROJECT_ROOT}", flush=True)

    load_env_loud()

    csv_path = Path(args.csv).expanduser().resolve()
    dry_run = not bool(args.really_place)

    base = alpaca_base_url()
    mode = mode_from_base(base)
    print(f"[mode] {mode} base={base}", flush=True)

    _ensure_live_orders_schema()

    # Broker (for guard + reconcile)
    broker = None
    guard = None
    if not dry_run:
        try:
            from services.broker_alpaca import AlpacaBroker  # type: ignore
            from services.execution_guard import ExecutionGuard  # type: ignore

            broker = AlpacaBroker(mode=mode)
            guard = ExecutionGuard(broker)
        except Exception as e:
            msg = f"Unable to init AlpacaBroker/ExecutionGuard: {e}"
            print(f"[FATAL] {msg}", flush=True)
            _append_event(session, "guard-init-fail", "FAIL", msg)
            return 99

        try:
            from services.master_execution_gate import (
                MasterExecutionGate,
                append_gate_log_csv,
                write_snapshot,
            )

            _mgd = MasterExecutionGate(project_root=PROJECT_ROOT).evaluate(
                mode=mode,
                broker=broker,
                verbose=False,
                require_market_open=(
                    False if (mode == "live" and args.allow_market_closed) else None
                ),
            )
            write_snapshot(_mgd)
            append_gate_log_csv(_mgd)
            if not _mgd.ok:
                print(f"[MASTER_GATE_BLOCK] {_mgd.summary}", flush=True)
                for _r in _mgd.reasons:
                    print(f"  reason={_r}", flush=True)
                _append_event(session, "master-gate", "BLOCKED", _mgd.summary)
                return 2
        except Exception as e:
            print(f"[MASTER_GATE] evaluation error: {e}", flush=True)
            _append_event(session, "master-gate", "ERROR", str(e))
            return 1

    # Risk Gate (REAL placement only) — kept as an additional layer
    decision = None
    reserve_pct = 0.0
    if not dry_run:
        try:
            from services.risk_gate import assert_can_place_orders  # type: ignore

            decision = assert_can_place_orders()
            try:
                reserve_pct = float(
                    (decision.raw or {}).get("broker", {}).get("reserve_pct") or 0.0
                )
            except Exception:
                reserve_pct = 0.0

            print(
                f"[RiskGate] OK | regime={decision.regime} mode={decision.mode} "
                f"max_gross={decision.max_gross_exposure} max_pos={decision.max_position_weight} reserve_pct={reserve_pct}",
                flush=True,
            )
        except Exception as e:
            msg = str(e)
            print(f"[FATAL] {msg}", flush=True)
            _append_event(session, "risk-gate-block", "BLOCKED", msg)
            return 10

    # Preflight reconcile (REAL only)
    if not dry_run:
        try:
            _run_reconcile(session, phase="pre", hard_stop=True, source=source, mode=mode)
        except Exception as e:
            print(f"[FATAL] {e}", flush=True)
            return 98

    # Probe account (REST)
    try:
        acct = probe_account(verbose=True)
    except Exception as e:
        print(f"[FATAL] account probe failed:\n{e}", flush=True)
        _append_event(session, "account-probe-fail", "FAIL", str(e))
        return 2

    equity = _safe_float(acct.get("equity") or acct.get("portfolio_value") or 0.0, 0.0)
    buying_power = _safe_float(acct.get("buying_power") or acct.get("cash") or 0.0, 0.0)

    # Load + parse CSV
    try:
        orders = load_orders(csv_path, default_order_type=args.order_type, default_tif=args.tif)
    except Exception as e:
        print(f"[FATAL] failed loading CSV:\n{e}", flush=True)
        _append_event(session, "csv-load-fail", "FAIL", str(e))
        return 3

    if not orders:
        print("[INFO] 0 valid orders after parsing. Nothing to do.", flush=True)
        return 0

    print(f"[INFO] loaded {len(orders)} valid orders from {csv_path}", flush=True)
    print(f"[MODE] {'DRY RUN' if dry_run else 'REAL PLACEMENT'}", flush=True)
    if not dry_run:
        print(
            f"[dupe] mode={args.dupe_mode} on_dupe={args.on_dupe} tol={args.dupe_price_tol}",
            flush=True,
        )

    # Market-open gate (REAL placement)
    market_closed = False
    if not dry_run:
        c = get_clock()
        is_open = clock_is_open(c)
        market_closed = is_open is False

        require_open = bool(args.require_market_open)
        if (not args.allow_market_closed) and (not args.require_market_open):
            require_open = True

        if require_open and market_closed:
            msg = (
                "Market is CLOSED. Blocking real placement (use --allow-market-closed to override)."
            )
            print(f"[FATAL] {msg}", flush=True)
            _append_event(session, "market-closed-block", "BLOCKED", msg)
            return 12

        if market_closed and args.allow_market_closed:
            print(
                "[gate] market is CLOSED, but --allow-market-closed set (will place as resting orders).",
                flush=True,
            )

    # Open orders + pending_cancel gate (REAL placement)
    open_orders: List[Dict[str, Any]] = []
    if not dry_run:
        try:
            open_orders = list_open_orders(nested=True, limit=500)
        except Exception as e:
            msg = f"Unable to list open orders: {e}"
            print(f"[FATAL] {msg}", flush=True)
            _append_event(session, "open-orders-fail", "FAIL", msg)
            return 13

        pending_cancel = [
            o for o in open_orders if str(o.get("status") or "").lower() == "pending_cancel"
        ]
        if pending_cancel and (not args.ignore_pending_cancel):
            msg = (
                f"Safety block: {len(pending_cancel)} open orders are in PENDING_CANCEL. "
                "Refusing new real placement to prevent pileups. Use --ignore-pending-cancel to override."
            )
            print(f"[FATAL] {msg}", flush=True)
            _append_event(session, "pending-cancel-block", "BLOCKED", msg)
            return 14

    # Optional: cancel open orders for symbols being placed
    if not dry_run and args.cancel_open_for_symbols:
        syms_in_run = sorted(set(o.symbol for o in orders[: args.max_orders]))
        to_cancel = [oo for oo in open_orders if str(oo.get("symbol") or "").upper() in syms_in_run]
        if to_cancel:
            print(
                f"[cancel] Canceling {len(to_cancel)} open orders for symbols in this run...",
                flush=True,
            )
            for oo in to_cancel:
                oid = str(oo.get("id") or "")
                if not oid:
                    continue
                st = str(oo.get("status") or "").lower()
                if st == "pending_cancel" and not args.ignore_pending_cancel:
                    continue
                try:
                    cancel_order(oid)
                except Exception as e:
                    print(f"[cancel] failed id={oid} sym={oo.get('symbol')} err={e}", flush=True)
            open_orders = list_open_orders(nested=True, limit=500)

    # Positions / exposure maps for caps (REAL placement only)
    mv_by_sym: Dict[str, float] = {}
    gross_mv = 0.0
    if (not dry_run) and decision is not None:
        try:
            positions = fetch_positions()
            mv_by_sym, gross_mv = build_position_maps(positions)
            print(f"[RiskGate] positions loaded: gross_mv≈${gross_mv:,.2f}", flush=True)
        except Exception as e:
            msg = f"Unable to fetch positions for RiskGate caps: {e}"
            print(f"[FATAL] {msg}", flush=True)
            _append_event(session, "risk-caps-block", "BLOCKED", msg)
            return 11

    submitted = 0
    processed = 0
    dupe_cancels = 0

    for idx, o0 in enumerate(orders[: args.max_orders], start=1):
        processed += 1
        o = o0

        # Closed-market behavior: force market -> limit BEFORE dupe detection
        if (not dry_run) and market_closed and args.allow_market_closed:
            force = bool(args.force_limit_when_closed) or True
            if force and str(o.order_type).lower() == "market":
                px = float(o.close if o.close else 0.0)
                lim = _forced_limit_price(px, o.side, args.limit_offset_bps)
                lim_rounded = round(lim, 2)
                print(
                    f"[{idx}] {o.symbol} {o.side} -> forced_limit_closed_market px={px:.2f} "
                    f"off={args.limit_offset_bps/10000.0:.4f} lim={lim_rounded:.2f}",
                    flush=True,
                )
                o = OrderRow(
                    symbol=o.symbol,
                    side=o.side,
                    qty=o.qty,
                    close=o.close,
                    order_type="limit",
                    tif=o.tif,
                    limit_price=lim_rounded,
                    extended_hours=o.extended_hours,
                    client_order_id=o.client_order_id,
                )

        # Duplicate logic (REAL only)
        if not dry_run and (not args.allow_duplicates):
            dupes = find_dupes(
                open_orders, o, dupe_mode=args.dupe_mode, price_tol=args.dupe_price_tol
            )
            if dupes:
                print(f"[{idx}] DUPE FOUND ({len(dupes)} match):", flush=True)
                for d in dupes[:5]:
                    print(f"    - {describe_order_short(d)}", flush=True)

                if args.on_dupe == "cancel":
                    print(
                        f"[{idx}] on_dupe=cancel: canceling dupes (cleanup only, no new order)...",
                        flush=True,
                    )
                    for d in dupes:
                        oid = str(d.get("id") or "")
                        if not oid:
                            continue
                        try:
                            cancel_order(oid)
                            dupe_cancels += 1
                        except Exception as e:
                            print(f"[{idx}] cancel failed id={oid} err={e}", flush=True)

                    for _ in range(max(1, int(args.cancel_settle_retries))):
                        time.sleep(max(0.0, float(args.cancel_settle_sleep)))
                        try:
                            open_orders = list_open_orders(nested=True, limit=500)
                        except Exception:
                            break
                        still = find_dupes(
                            open_orders, o, dupe_mode=args.dupe_mode, price_tol=args.dupe_price_tol
                        )
                        if not still:
                            break

                    log_order_row(session, "dupe-cancel", o, status="OK")
                    continue

                if args.on_dupe == "replace":
                    already_same = False
                    if args.dupe_mode == "strict":
                        already_same = any(
                            strict_dupe_match_with_tol(d, o, price_tol=args.dupe_price_tol)
                            for d in dupes
                        )
                    else:
                        already_same = True

                    if already_same and (not args.force_replace):
                        msg = f"Intended order already exists (tol={args.dupe_price_tol}). NOOP (use --force-replace to churn)."
                        print(f"[{idx}] {msg}", flush=True)
                        log_order_row(session, "dupe-noop", o, status="NOOP")
                        continue

                    print(
                        f"[{idx}] on_dupe=replace: canceling dupes then placing intended order...",
                        flush=True,
                    )
                    for d in dupes:
                        oid = str(d.get("id") or "")
                        if not oid:
                            continue
                        try:
                            cancel_order(oid)
                            dupe_cancels += 1
                        except Exception as e:
                            print(f"[{idx}] cancel failed id={oid} err={e}", flush=True)

                    cleared = False
                    for _ in range(max(1, int(args.cancel_settle_retries))):
                        time.sleep(max(0.0, float(args.cancel_settle_sleep)))
                        open_orders = list_open_orders(nested=True, limit=500)
                        still = find_dupes(
                            open_orders, o, dupe_mode=args.dupe_mode, price_tol=args.dupe_price_tol
                        )
                        if not still:
                            cleared = True
                            break

                    if not cleared:
                        msg = "Canceled dupes but they did not clear in time (still open/pending_cancel). Blocking to prevent pileups."
                        print(f"[{idx}] [FATAL] {msg}", flush=True)
                        _append_event(session, "dupe-cancel-not-cleared", "BLOCKED", msg)
                        return 15

                    log_order_row(session, "dupe-replace", o, status="OK")

                else:
                    msg = (
                        f"Duplicate open order exists for {o.symbol} {o.side} "
                        f"qty={int(o.qty)} type={o.order_type} (tol={args.dupe_price_tol}); skipping."
                    )
                    print(f"[{idx}] {msg}", flush=True)
                    log_order_row(session, "dupe-skip", o, status="SKIP")
                    continue

        # Apply RiskGate caps (REAL placement only)
        cap_note = ""
        if (not dry_run) and decision is not None:
            o, cap_note = clamp_order_by_risk_caps(
                o,
                equity=float(equity),
                buying_power=float(buying_power),
                reserve_pct=float(reserve_pct),
                max_position_weight=float(decision.max_position_weight),
                max_gross_exposure=float(decision.max_gross_exposure),
                current_gross_mv=float(gross_mv),
                current_sym_mv=float(mv_by_sym.get(o.symbol, 0.0)),
            )
            if "room=0" in cap_note or "qty_to_zero" in cap_note:
                print(f"[{idx}] {o0.symbol} {o0.side} -> SKIP ({cap_note})", flush=True)
                log_order_row(session, "risk-skip", o0, status="SKIP")
                continue

        px_for_print = float(
            o.limit_price if (o.order_type == "limit" and o.limit_price is not None) else o.close
        )
        notional = float(o.qty) * max(px_for_print, 0.0)

        print(
            f"[{idx}] {o.symbol} {o.side} qty={o.qty} type={o.order_type} tif={o.tif} "
            f"limit={o.limit_price} notional~{notional:.2f} {('['+cap_note+']') if cap_note else ''}",
            flush=True,
        )

        # ExecutionGuard (REAL placement only)
        if not dry_run:
            assert guard is not None
            guard_payload = build_guard_payload(o)
            decision_g = guard.validate_and_audit(guard_payload, source=source)
            if not decision_g.ok:
                msg = f"GUARD BLOCK [{decision_g.code}] {decision_g.message}"
                print(f"[{idx}] {msg}", flush=True)
                log_order_row(session, "guard-block", o, status="SKIP")
                continue

        # Submit
        try:
            status, oid, raw, _payload = place_order(o, dry_run=dry_run)
        except Exception as e:
            status, oid, raw, _payload = "ERROR_EXCEPTION", "", str(e), {}

        if dry_run:
            print("    -> DRY_RUN", flush=True)
            log_order_row(session, "dry-run", o, status="DRY")
            continue

        # REAL placement logging
        if isinstance(raw, dict):
            filled_qty = raw.get("filled_qty", 0)
            filled_avg_price = raw.get("filled_avg_price", "")
        else:
            filled_qty = 0
            filled_avg_price = ""

        print(f"    -> PLACED status={status} id={oid}", flush=True)
        log_order_row(
            session,
            "submit",
            o,
            status=str(status),
            order_id=str(oid),
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
        )
        submitted += 1

        # Update local exposure estimates after a successful BUY
        if decision is not None and o.side == "BUY":
            est = float(o.qty) * max(px_for_print, 0.0)
            gross_mv += est
            mv_by_sym[o.symbol] = float(mv_by_sym.get(o.symbol, 0.0) + est)
            buying_power = max(0.0, buying_power - est)

        # Refresh open_orders after placement
        if not args.allow_duplicates:
            try:
                open_orders = list_open_orders(nested=True, limit=500)
            except Exception:
                pass

    # Post reconcile (REAL only)
    if not dry_run:
        try:
            _run_reconcile(session, phase="post", hard_stop=True, source=source, mode=mode)
        except Exception as e:
            print(f"[FATAL] {e}", flush=True)
            return 97

    print(
        f"[DONE] submitted={submitted} processed={processed} dupe_cancels={dupe_cancels}",
        flush=True,
    )
    print(f"[AUDIT] {LIVE_ORDERS_LOG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
