# services/snapshot_live_orders.py
# ------------------------------------------------------------
# TRITON — Snapshot Live Orders / Recent Orders / Positions
#
# Writes:
#   data/results/open_orders_snapshot.csv (open orders snapshot)   ✅ FIXED
#   data/results/recent_orders.csv        (recent/all orders)
#   data/results/positions_snapshot.csv   (positions)
#
# execute_trades may invoke this module once before placement when CSV snapshots are stale
# (see snapshot_hygiene_* in execute_trades config) — does not replace MasterExecutionGate checks.
#
# IMPORTANT:
# - live_orders.csv is the append-only EVENT LOG used by:
#     - place_live_orders.py  (action=submit)
#     - poll_order_status.py  (action=poll)
# - This snapshot module must NEVER write to live_orders.csv.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# ------------------------------------------------------------
# Ensure project root is importable when running directly
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ✅ FIX: do NOT write snapshots into live_orders.csv
OPEN_ORDERS_PATH = RESULTS_DIR / "open_orders_snapshot.csv"

RECENT_ORDERS_PATH = RESULTS_DIR / "recent_orders.csv"
POSITIONS_PATH = RESULTS_DIR / "positions_snapshot.csv"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _to_df(rows: Any, ts: str) -> pd.DataFrame:
    """
    Normalize list[dict] (or DataFrame) to DataFrame and ensure snapshot_ts exists.
    """
    if rows is None:
        df = pd.DataFrame()
    elif isinstance(rows, pd.DataFrame):
        df = rows.copy()
    elif isinstance(rows, list):
        df = pd.DataFrame(rows)
    else:
        try:
            df = pd.DataFrame(list(rows))
        except Exception:
            df = pd.DataFrame()

    if "snapshot_ts" not in df.columns:
        df.insert(0, "snapshot_ts", ts)
    else:
        df["snapshot_ts"] = ts

    return df


def _ensure_symbol_and_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contracts / compatibility:
      - Validator requires: `symbol`
      - Some dashboard/services expect: `ticker`
    """
    if df.empty:
        if "symbol" not in df.columns:
            df["symbol"] = pd.Series(dtype="object")
        if "ticker" not in df.columns:
            df["ticker"] = pd.Series(dtype="object")
        return df

    if "symbol" in df.columns and "ticker" not in df.columns:
        df["ticker"] = df["symbol"]
        return df

    if "ticker" in df.columns and "symbol" not in df.columns:
        df["symbol"] = df["ticker"]
        return df

    if "symbol" in df.columns and "ticker" in df.columns:
        return df

    for alt in ("sym", "instrument", "asset"):
        if alt in df.columns:
            df["symbol"] = df[alt]
            df["ticker"] = df[alt]
            return df

    df["symbol"] = pd.NA
    df["ticker"] = pd.NA
    return df


def _ensure_positions_contract(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure positions_snapshot.csv has:
      snapshot_ts, date, ticker, symbol, market_value, value

    - date is derived from snapshot_ts
    - value is alias of market_value
    """
    if "snapshot_ts" not in df.columns:
        df.insert(0, "snapshot_ts", pd.Series(dtype="object"))

    df = _ensure_symbol_and_ticker(df)

    if not df.empty:
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df = df[(df["ticker"] != "") & (df["ticker"] != "NAN")].copy()

    if "date" not in df.columns:
        df["date"] = pd.Series(dtype="object")

    ts_parsed = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)
    df["date"] = ts_parsed.dt.date.astype(str)

    if "market_value" not in df.columns:
        df["market_value"] = pd.Series(dtype="float")
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")

    if df["market_value"].isna().all():
        qty = pd.to_numeric(df.get("qty", pd.Series([pd.NA] * len(df))), errors="coerce")
        px = pd.to_numeric(
            df.get(
                "current_price",
                df.get("last_price", df.get("lastday_price", pd.Series([pd.NA] * len(df)))),
            ),
            errors="coerce",
        )
        mv = qty * px
        if mv.notna().any():
            df["market_value"] = mv

    if "value" not in df.columns:
        df["value"] = pd.Series(dtype="float")
    df["value"] = pd.to_numeric(df["market_value"], errors="coerce")

    preferred = ["snapshot_ts", "date", "ticker", "symbol", "market_value", "value"]
    rest = [c for c in df.columns if c not in preferred]
    return df[preferred + rest].copy()


def _auto_heal_contracts() -> None:
    """
    Auto-heal contracts (prevents regressions from other writers).
    Optional; never fail snapshot because normalizer isn't available.
    """
    try:
        from services.normalize_results_contracts import main as normalize_results_contracts  # type: ignore

        normalize_results_contracts()
    except Exception:
        pass


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="TRITON: snapshot orders + positions to CSV for dashboard"
    )
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument("--recent-limit", type=int, default=200)
    args = parser.parse_args()

    ts = _iso_utc()

    from services.broker_alpaca import AlpacaBroker  # type: ignore

    broker = AlpacaBroker(mode=args.mode)

    # Open orders snapshot ✅ to open_orders_snapshot.csv
    open_orders = broker.list_orders(status="open", nested=True, limit=500) or []
    df_open = _to_df(open_orders, ts)
    _write_csv(OPEN_ORDERS_PATH, df_open)

    # Recent/all orders
    recent_orders = (
        broker.list_orders(status="all", nested=True, limit=int(args.recent_limit)) or []
    )
    df_recent = _to_df(recent_orders, ts)
    _write_csv(RECENT_ORDERS_PATH, df_recent)

    # Positions (contract-correct snapshot)
    positions = broker.get_positions() or []
    df_pos = _to_df(positions, ts)
    df_pos = _ensure_positions_contract(df_pos)
    _write_csv(POSITIONS_PATH, df_pos)

    _auto_heal_contracts()

    return {
        "ok": True,
        "mode": args.mode,
        "ts": ts,
        "open_orders_written": int(len(open_orders)),
        "recent_orders_written": int(len(recent_orders)),
        "positions_written": int(len(df_pos)) if df_pos is not None else 0,
        "paths": {
            "open_orders": str(OPEN_ORDERS_PATH),
            "recent_orders": str(RECENT_ORDERS_PATH),
            "positions": str(POSITIONS_PATH),
        },
        "errors": [],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
