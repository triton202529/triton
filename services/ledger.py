# services/ledger.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
LEDGER_DIR = DATA_ROOT / "ledger"
LEDGER_DIR.mkdir(parents=True, exist_ok=True)

LEDGER_PATH = LEDGER_DIR / "ledger.parquet"
LEDGER_META_PATH = LEDGER_DIR / "ledger_meta.json"


LEDGER_COLS = [
    "symbol",
    "qty",  # signed (long positive, short negative)
    "avg_entry_price",
    "market_price",
    "market_value",
    "unrealized_pl",
    "side",  # long/short/flat
    "broker_account",
    "asof_ts_utc",
    "source",
    "open_orders_count",
]


def utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any) -> float:
    try:
        if x is None or x == "" or str(x).lower() in ("nan", "none", "null"):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _safe_int(x: Any) -> int:
    try:
        if x is None or x == "" or str(x).lower() in ("nan", "none", "null"):
            return 0
        return int(float(x))
    except Exception:
        return 0


def _pos_side(qty: int) -> str:
    if qty > 0:
        return "long"
    if qty < 0:
        return "short"
    return "flat"


def ensure_ledger_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in LEDGER_COLS:
        if c not in df.columns:
            df[c] = "" if c in ("symbol", "side", "broker_account", "asof_ts_utc", "source") else 0
    df = df[LEDGER_COLS].copy()

    # normalize types
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    for c in ["avg_entry_price", "market_price", "market_value", "unrealized_pl"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    df["open_orders_count"] = (
        pd.to_numeric(df["open_orders_count"], errors="coerce").fillna(0).astype(int)
    )

    df["side"] = df["qty"].apply(lambda q: _pos_side(int(q)))
    return df


def load_ledger(path: Path = LEDGER_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return ensure_ledger_schema(pd.DataFrame(columns=LEDGER_COLS))
    try:
        df = pd.read_parquet(path)
        return ensure_ledger_schema(df)
    except Exception:
        # fall back to empty if corrupted
        return ensure_ledger_schema(pd.DataFrame(columns=LEDGER_COLS))


def save_ledger(df: pd.DataFrame, path: Path = LEDGER_PATH) -> None:
    df2 = ensure_ledger_schema(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df2.to_parquet(path, index=False)

    meta = {
        "updated_at": utc_now_iso_z(),
        "rows": int(df2.shape[0]),
        "path": str(path),
    }
    try:
        LEDGER_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass


def ledger_from_broker_positions(
    positions: List[Dict[str, Any]],
    *,
    broker_account: str = "",
    open_orders_by_symbol: Optional[Dict[str, int]] = None,
    source: str = "broker",
) -> pd.DataFrame:
    """
    Build a fresh ledger snapshot from broker positions (authoritative).
    Expects Alpaca-like position dicts:
      symbol, qty, avg_entry_price, current_price, market_value, unrealized_pl, side
    """
    open_orders_by_symbol = open_orders_by_symbol or {}

    rows: List[Dict[str, Any]] = []
    ts = utc_now_iso_z()

    for p in positions or []:
        sym = str(p.get("symbol") or "").upper().strip()
        if not sym:
            continue

        qty = _safe_int(p.get("qty", 0))
        # Alpaca includes explicit "side" but we always compute from signed qty for consistency
        # If p["side"] is "short", qty is often negative or positive? Alpaca positions qty is string with sign? sometimes positive.
        # We normalize using reported side if qty sign is missing.
        side_raw = str(p.get("side") or "").lower().strip()
        if qty >= 0 and side_raw == "short":
            qty = -abs(qty)

        avg_entry = _safe_float(p.get("avg_entry_price", 0))
        cur_px = _safe_float(p.get("current_price", 0))
        mv = _safe_float(p.get("market_value", 0))
        upl = _safe_float(p.get("unrealized_pl", 0))

        rows.append(
            {
                "symbol": sym,
                "qty": int(qty),
                "avg_entry_price": float(avg_entry),
                "market_price": float(cur_px),
                "market_value": float(mv),
                "unrealized_pl": float(upl),
                "side": _pos_side(int(qty)),
                "broker_account": broker_account,
                "asof_ts_utc": ts,
                "source": source,
                "open_orders_count": int(open_orders_by_symbol.get(sym, 0)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=LEDGER_COLS)

    return ensure_ledger_schema(df)


def index_by_symbol(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    df2 = ensure_ledger_schema(df)
    out: Dict[str, Dict[str, Any]] = {}
    for _, r in df2.iterrows():
        out[str(r["symbol"])] = {
            "symbol": str(r["symbol"]),
            "qty": int(r["qty"]),
            "avg_entry_price": float(r["avg_entry_price"]),
            "market_price": float(r["market_price"]),
            "market_value": float(r["market_value"]),
            "unrealized_pl": float(r["unrealized_pl"]),
            "open_orders_count": int(r["open_orders_count"]),
        }
    return out
