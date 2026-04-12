# services/snapshot_positions.py
# ------------------------------------------------------------
# TRITON — Positions Snapshot (Read-only artifact for dashboard)
#
# Purpose:
#   Write a simple positions_snapshot.csv that the dashboard can read,
#   containing valid (date,ticker,value) rows for Positions & Exposure.
#
# Output:
#   data/results/positions_snapshot.csv
#
# Required dashboard columns:
#   - date
#   - ticker
#   - value   (market_value alias)
#
# Validator contract (added):
#   - symbol  (alias of ticker)
#
# Notes:
#   - We intentionally map `value = market_value` so dashboards can plot
#     exposure without extra logic.
#   - This script is read-only against the broker (no cancels/placements).
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ------------------------------------------------------------
# Ensure project root is importable when running directly
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_OUT = RESULTS_DIR / "positions_snapshot.csv"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def load_positions(mode: str) -> List[Dict[str, Any]]:
    """
    Uses your AlpacaBroker wrapper. Assumes:
      - AlpacaBroker(mode="paper"|"live")
      - get_positions() -> list[dict]
    """
    try:
        from services.broker_alpaca import AlpacaBroker  # type: ignore
    except Exception as e:
        raise SystemExit(
            "ERROR: Could not import services.broker_alpaca.AlpacaBroker.\n"
            f"Import error: {e}\n"
            "Fix your broker wrapper import or run from project root."
        )

    broker = AlpacaBroker(mode=mode)
    positions = broker.get_positions() or []

    if isinstance(positions, list):
        return positions

    # Some wrappers return iterables / dict-like objects; normalize to list
    try:
        return list(positions)  # type: ignore[arg-type]
    except Exception:
        raise SystemExit("ERROR: get_positions() did not return a list-like structure.")


def positions_to_df(positions: List[Dict[str, Any]], snapshot_ts_utc: datetime) -> pd.DataFrame:
    """
    Normalize positions into a dashboard-friendly CSV.

    Produces:
      - date (UTC date)
      - snapshot_ts (UTC timestamp)
      - ticker
      - symbol (alias of ticker)  <-- validator expects this
      - qty
      - current_price
      - market_value
      - value (alias of market_value)  <-- dashboard uses this
      - plus extras for debugging
    """
    rows: List[Dict[str, Any]] = []

    snap_iso = snapshot_ts_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    snap_date = snapshot_ts_utc.date().isoformat()

    for p in positions:
        # Accept multiple possible keys from broker wrapper
        sym_raw = p.get("symbol") or p.get("ticker") or p.get("sym") or p.get("asset")
        ticker = _safe_str(sym_raw).strip().upper()

        qty = _safe_float(p.get("qty"))
        current_price = (
            _safe_float(p.get("current_price"))
            or _safe_float(p.get("last_price"))
            or _safe_float(p.get("lastday_price"))
        )
        market_value = _safe_float(p.get("market_value"))

        # If market_value isn't present, approximate if price + qty exist
        if market_value is None and qty is not None and current_price is not None:
            market_value = qty * current_price

        value = market_value  # dashboard exposure plots expect `value`

        rows.append(
            {
                "snapshot_ts": snap_iso,
                "date": snap_date,
                "ticker": ticker,
                "symbol": ticker,  # contract alias (critical)
                "qty": qty,
                "current_price": current_price,
                "market_value": market_value,
                "value": value,
                # extras
                "avg_entry_price": _safe_float(p.get("avg_entry_price")),
                "cost_basis": _safe_float(p.get("cost_basis")),
                "unrealized_pl": _safe_float(p.get("unrealized_pl")),
                "unrealized_plpc": _safe_float(p.get("unrealized_plpc")),
                "side": _safe_str(p.get("side")).strip(),
                "asset_id": _safe_str(p.get("asset_id")).strip(),
                "exchange": _safe_str(p.get("exchange")).strip(),
                "asset_class": _safe_str(p.get("asset_class")).strip(),
                "asset_marginable": p.get("asset_marginable"),
            }
        )

    df = pd.DataFrame(rows)

    # Clean up obvious junk rows
    if not df.empty:
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df = df[(df["ticker"] != "") & (df["ticker"] != "NAN")].copy()

    return df


def write_snapshot(df: pd.DataFrame, out_path: Path, append: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if append and out_path.exists():
        df.to_csv(out_path, mode="a", index=False, header=False)
    else:
        df.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="TRITON: snapshot positions to CSV for dashboard")
    ap.add_argument("--mode", choices=["paper", "live"], default="paper", help="Broker mode")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path")
    ap.add_argument("--append", action="store_true", help="Append rows instead of overwrite")
    ap.add_argument("--empty-ok", action="store_true", help="Do not fail if there are no positions")
    args = ap.parse_args()

    out_path = Path(args.out)
    snapshot_ts = _utc_now()

    positions = load_positions(args.mode)

    if not positions and not args.empty_ok:
        raise SystemExit(
            "No positions returned (empty). Use --empty-ok to still write an empty snapshot."
        )

    df = positions_to_df(positions, snapshot_ts)

    # Ensure dashboard/validator-critical columns exist even if empty
    for col in ("date", "ticker", "symbol", "value"):
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    write_snapshot(df, out_path, append=args.append)

    print(f"[snapshot_positions] Wrote {len(df)} rows -> {out_path.as_posix()}")
    if len(df) > 0:
        preview_cols = [
            c
            for c in ("date", "ticker", "symbol", "qty", "current_price", "market_value", "value")
            if c in df.columns
        ]
        print(df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
