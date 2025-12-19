# services/simulate_portfolio.py

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np

# ── Defaults
DEFAULT_INITIAL_BALANCE = 100_000.0
DEFAULT_POSITION_SIZE   = 0.10  # 10% of CASH per new BUY
PROJECT_ROOT            = Path(__file__).resolve().parents[1]
DATA_DIR                = PROJECT_ROOT / "data"
RESULTS_DIR             = DATA_DIR / "results"
PREDICTIONS_DIR         = DATA_DIR / "predictions"

# Preferred signal locations (searched in order if --signals not provided)
DEFAULT_SIGNAL_CANDIDATES = [
    RESULTS_DIR / "signals_with_rationale.csv",
    RESULTS_DIR / "signals.csv",
    PREDICTIONS_DIR / "signals.csv",
]

DEFAULT_PORTFOLIO_HISTORY_FILE = RESULTS_DIR / "portfolio_history.csv"
DEFAULT_TRADE_LOG_FILE         = RESULTS_DIR / "trade_log.csv"


def _exists(p: Path | str) -> bool:
    p = Path(p)
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False


def _normalize_signals_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has: date (datetime), ticker (UPPER), signal (UPPER), price (float)
    Accepts 'close' instead of 'price'; falls back from 'timestamp' -> 'date'.
    Drops rows with missing essentials.
    """
    df = df.copy()

    # Date
    if "date" not in df.columns:
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
        else:
            # synthesize today's date to avoid hard crash (keeps dashboard alive)
            df["date"] = pd.Timestamp.today().normalize()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)

    # Ticker
    if "ticker" not in df.columns:
        raise ValueError("signals file has no 'ticker' column")
    df["ticker"] = df["ticker"].astype(str).str.upper()

    # Signal
    if "signal" not in df.columns:
        df["signal"] = "HOLD"
    df["signal"] = df["signal"].astype(str).str.upper()

    # Price
    if "price" not in df.columns:
        if "close" in df.columns:
            df = df.rename(columns={"close": "price"})
        else:
            # last resort: create a dummy price so the sim can still output a flat curve
            df["price"] = np.nan

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Keep only useful columns
    keep = [c for c in ["date", "ticker", "signal", "price"] if c in df.columns]
    df = df[keep]

    # Drop NAs on essentials (allow NaN price rows; they just won't transact)
    df = df.dropna(subset=["date", "ticker", "signal"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    return df


def load_signals(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load signals from an explicit path or from default candidates.
    """
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    candidates.extend(DEFAULT_SIGNAL_CANDIDATES)

    last_err = None
    for p in candidates:
        if _exists(p):
            try:
                df = pd.read_csv(p)
                df = _normalize_signals_schema(df)
                if not df.empty:
                    return df
            except Exception as e:
                last_err = e
                continue

    # If we got here, nothing worked
    raise FileNotFoundError(
        f"No readable signals CSV found. Tried: {', '.join(str(c) for c in candidates)}"
        + (f" (last error: {last_err})" if last_err else "")
    )


def _mark_to_market(positions: Dict[str, Dict[str, float]], prices: Dict[str, float]) -> float:
    mv = 0.0
    for tkr, pos in positions.items():
        px = prices.get(tkr)
        if px is not None and np.isfinite(px):
            mv += float(pos["shares"]) * float(px)
    return mv


def simulate_portfolio(
    trades_df: pd.DataFrame,
    starting_cash: float = DEFAULT_INITIAL_BALANCE,
    position_size: float = DEFAULT_POSITION_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple paper-trading simulator.

    Inputs:
      trades_df columns required:
        - date (datetime-like)
        - ticker (str)
        - signal (str: BUY/SELL/HOLD)
        - price (float)  <-- we auto-rename 'close' to 'price' during load

      starting_cash: initial cash balance
      position_size: fraction of *current cash* to allocate to a NEW BUY (0..1)

    Behavior:
      - Processes day by day (SELLs first, then BUYs).
      - SELL fully closes any existing position in that ticker.
      - BUY only if no existing long; buys whole shares using (cash * position_size).
      - Marks to market each day using the last price observed that day per ticker.
      - No slippage/fees; this is a dashboard curve, not an accounting engine.

    Returns:
      portfolio_history_df: [date, cash, market_value, total_value]
      trade_log_df:         [date, action, ticker, price, quantity, cash_after, total_value]
    """
    if trades_df is None or trades_df.empty:
        ph_cols = ["date", "cash", "market_value", "total_value"]
        tl_cols = ["date", "action", "ticker", "price", "quantity", "cash_after", "total_value"]
        return pd.DataFrame(columns=ph_cols), pd.DataFrame(columns=tl_cols)

    df = trades_df.copy()

    # Validate essentials
    for col in ("date", "ticker", "signal", "price"):
        if col not in df.columns:
            raise ValueError(f"simulate_portfolio: required column '{col}' missing after normalization")

    # Coerce types
    df["date"]   = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["signal"] = df["signal"].astype(str).str.upper()
    df["price"]  = pd.to_numeric(df["price"], errors="coerce")

    # Drop hopeless rows (no date/ticker/signal)
    df = df.dropna(subset=["date", "ticker", "signal"]).reset_index(drop=True)
    if df.empty:
        ph_cols = ["date", "cash", "market_value", "total_value"]
        tl_cols = ["date", "action", "ticker", "price", "quantity", "cash_after", "total_value"]
        return pd.DataFrame(columns=ph_cols), pd.DataFrame(columns=tl_cols)

    # Sort
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # State
    cash: float = float(starting_cash)
    positions: Dict[str, Dict[str, float]] = {}  # ticker -> {"shares": int, "avg_price": float}

    portfolio_history_rows: List[Dict] = []
    trade_log_rows: List[Dict] = []

    # Process day by day
    for day, day_df in df.groupby(df["date"].dt.normalize(), sort=True):
        # Build last price map for the day
        last_price_for_day: Dict[str, float] = {}
        for r in day_df.itertuples():
            if np.isfinite(r.price):
                last_price_for_day[r.ticker] = float(r.price)

        # 1) Sells first
        for row in day_df[day_df["signal"] == "SELL"].itertuples():
            tkr = row.ticker
            px  = float(row.price) if np.isfinite(row.price) else last_price_for_day.get(tkr)
            if px is None or not np.isfinite(px):
                continue
            pos = positions.get(tkr)
            if pos and pos["shares"] > 0:
                qty = int(pos["shares"])
                proceeds = qty * px
                cash += proceeds
                positions.pop(tkr, None)
                total_value = cash + _mark_to_market(positions, last_price_for_day)
                trade_log_rows.append(
                    {
                        "date": day,
                        "action": "SELL",
                        "ticker": tkr,
                        "price": round(px, 6),
                        "quantity": -qty,
                        "cash_after": round(cash, 2),
                        "total_value": round(total_value, 2),
                    }
                )

        # 2) Buys (open only if not already long)
        for row in day_df[day_df["signal"] == "BUY"].itertuples():
            tkr = row.ticker
            if tkr in positions and positions[tkr]["shares"] > 0:
                continue
            px = float(row.price) if np.isfinite(row.price) else last_price_for_day.get(tkr)
            if px is None or not np.isfinite(px) or px <= 0:
                continue
            budget = cash * float(position_size)
            qty = int(budget // px)
            if qty <= 0:
                continue
            cost = qty * px
            if cost > cash:
                continue
            cash -= cost
            positions[tkr] = {"shares": qty, "avg_price": px}
            total_value = cash + _mark_to_market(positions, last_price_for_day)
            trade_log_rows.append(
                {
                    "date": day,
                    "action": "BUY",
                    "ticker": tkr,
                    "price": round(px, 6),
                    "quantity": qty,
                    "cash_after": round(cash, 2),
                    "total_value": round(total_value, 2),
                }
            )

        # 3) End-of-day snapshot
        mv = _mark_to_market(positions, last_price_for_day)
        tv = cash + mv
        portfolio_history_rows.append(
            {
                "date": day,
                "cash": round(cash, 2),
                "market_value": round(mv, 2),
                "total_value": round(tv, 2),
            }
        )

    portfolio_history = (
        pd.DataFrame(portfolio_history_rows).sort_values("date").reset_index(drop=True)
    )
    trade_log = pd.DataFrame(trade_log_rows).sort_values("date").reset_index(drop=True)
    return portfolio_history, trade_log


def main():
    ap = argparse.ArgumentParser(description="Simulate a simple portfolio curve from signals.")
    ap.add_argument("--signals", type=str, default=None, help="Path to signals CSV. If omitted, tries results/predictions defaults.")
    ap.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_BALANCE, help="Starting cash balance.")
    ap.add_argument("--position-size", type=float, default=DEFAULT_POSITION_SIZE, help="Fraction of cash per new BUY (0..1).")
    ap.add_argument("--out-history", type=str, default=str(DEFAULT_PORTFOLIO_HISTORY_FILE), help="Where to write portfolio_history.csv")
    ap.add_argument("--out-trades", type=str, default=str(DEFAULT_TRADE_LOG_FILE), help="Where to write trade_log.csv")
    ap.add_argument("--no-write", action="store_true", help="If set, prints tail instead of writing files.")
    args = ap.parse_args()

    # Load + normalize signals
    signals_df = load_signals(args.signals)

    # Run sim
    ph, tl = simulate_portfolio(
        signals_df,
        starting_cash=args.initial_cash,
        position_size=args.position_size,
    )

    # Output
    if args.no_write:
        print("---- portfolio_history (tail) ----")
        print(ph.tail(5).to_string(index=False))
        print("\n---- trade_log (tail) ----")
        print(tl.tail(5).to_string(index=False))
    else:
        out_hist = Path(args.out_history)
        out_tr   = Path(args.out_trades)
        out_hist.parent.mkdir(parents=True, exist_ok=True)
        out_tr.parent.mkdir(parents=True, exist_ok=True)
        ph.to_csv(out_hist, index=False)
        tl.to_csv(out_tr, index=False)
        print(f"✅ Wrote {out_hist} ({len(ph)} rows)")
        print(f"✅ Wrote {out_tr} ({len(tl)} rows)")


if __name__ == "__main__":
    main()
