# services/simulate_portfolio.py

from __future__ import annotations
import os
from typing import Tuple, Dict, List
import pandas as pd

# Defaults (used when running this file directly)
DEFAULT_INITIAL_BALANCE = 100_000.0
DEFAULT_POSITION_SIZE = 0.10  # 10% of cash per new position

# Back-compat paths for standalone run
DEFAULT_SIGNALS_FILE = "data/predictions/signals.csv"
DEFAULT_RESULTS_DIR = "data/results"
DEFAULT_PORTFOLIO_HISTORY_FILE = os.path.join(DEFAULT_RESULTS_DIR, "portfolio_history.csv")
DEFAULT_TRADE_LOG_FILE = os.path.join(DEFAULT_RESULTS_DIR, "trade_log.csv")


def simulate_portfolio(
    trades_df: pd.DataFrame,
    starting_cash: float = DEFAULT_INITIAL_BALANCE,
    position_size: float = DEFAULT_POSITION_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple paper-trading simulator.

    Inputs:
      trades_df columns required:
        - date (datetime-like or string)
        - ticker (str)
        - signal (str: BUY/SELL/HOLD)
        - price (float)  <-- if you pass 'close', rename to 'price' before calling

      starting_cash: initial cash balance
      position_size: fraction of *current cash* to allocate to a NEW BUY (0..1)

    Behavior:
      - Processes trades day by day (sorted by date).
      - SELL: fully closes any existing position in that ticker.
      - BUY: if no existing position, buys whole shares using (cash * position_size).
      - HOLD: ignored.
      - Uses the provided 'price' for that day (no slippage/fees).
      - Marks to market at the end of each day using the last 'price' seen for each open ticker that day.

    Returns:
      portfolio_history_df with columns: [date, cash, market_value, total_value]
      trade_log_df with columns: [date, action, ticker, price, quantity, cash_after, total_value]
    """
    if trades_df is None or trades_df.empty:
        ph_cols = ["date", "cash", "market_value", "total_value"]
        tl_cols = ["date", "action", "ticker", "price", "quantity", "cash_after", "total_value"]
        return pd.DataFrame(columns=ph_cols), pd.DataFrame(columns=tl_cols)

    df = trades_df.copy()

    # Normalize schema
    if "date" not in df.columns:
        raise ValueError("simulate_portfolio: required column 'date' missing")
    if "ticker" not in df.columns:
        raise ValueError("simulate_portfolio: required column 'ticker' missing")
    if "signal" not in df.columns:
        raise ValueError("simulate_portfolio: required column 'signal' missing")
    if "price" not in df.columns:
        # Allow 'close' fallback
        if "close" in df.columns:
            df = df.rename(columns={"close": "price"})
        else:
            raise ValueError("simulate_portfolio: required column 'price' (or 'close') missing")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["signal"] = df["signal"].astype(str).str.upper()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "signal", "price"])
    if df.empty:
        ph_cols = ["date", "cash", "market_value", "total_value"]
        tl_cols = ["date", "action", "ticker", "price", "quantity", "cash_after", "total_value"]
        return pd.DataFrame(columns=ph_cols), pd.DataFrame(columns=tl_cols)

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # State
    cash: float = float(starting_cash)
    # positions: ticker -> {"shares": int, "avg_price": float}
    positions: Dict[str, Dict[str, float]] = {}

    portfolio_history_rows: List[Dict] = []
    trade_log_rows: List[Dict] = []

    # Process day by day
    for day, day_df in df.groupby(df["date"].dt.normalize(), sort=True):
        # Last price per ticker for marking to market
        last_price_for_day: Dict[str, float] = {
            r.ticker: float(r.price) for r in day_df.itertuples()
        }

        # 1) Sells first — close positions fully
        for row in day_df[day_df["signal"] == "SELL"].itertuples():
            tkr = row.ticker
            px = float(row.price)
            pos = positions.get(tkr)
            if pos and pos["shares"] > 0:
                qty = int(pos["shares"])
                proceeds = qty * px
                cash += proceeds
                positions.pop(tkr, None)
                total_value = cash + _mark_to_market(positions, last_price_for_day)
                trade_log_rows.append({
                    "date": day,
                    "action": "SELL",
                    "ticker": tkr,
                    "price": px,
                    "quantity": -qty,
                    "cash_after": round(cash, 2),
                    "total_value": round(total_value, 2),
                })

        # 2) Buys — only if no existing position
        for row in day_df[day_df["signal"] == "BUY"].itertuples():
            tkr = row.ticker
            if tkr in positions and positions[tkr]["shares"] > 0:
                continue  # already long; skip in this simple model
            px = float(row.price)
            if px <= 0:
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
            trade_log_rows.append({
                "date": day,
                "action": "BUY",
                "ticker": tkr,
                "price": px,
                "quantity": qty,
                "cash_after": round(cash, 2),
                "total_value": round(total_value, 2),
            })

        # 3) End-of-day snapshot
        mv = _mark_to_market(positions, last_price_for_day)
        tv = cash + mv
        portfolio_history_rows.append({
            "date": day,
            "cash": round(cash, 2),
            "market_value": round(mv, 2),
            "total_value": round(tv, 2),
        })

    portfolio_history = pd.DataFrame(portfolio_history_rows).sort_values("date").reset_index(drop=True)
    trade_log = pd.DataFrame(trade_log_rows).sort_values("date").reset_index(drop=True)
    return portfolio_history, trade_log


def _mark_to_market(positions: Dict[str, Dict[str, float]], prices: Dict[str, float]) -> float:
    mv = 0.0
    for tkr, pos in positions.items():
        px = prices.get(tkr)
        if px is not None:
            mv += float(pos["shares"]) * float(px)
    return mv


# -------- Backwards-compatible CLI (optional) --------
if __name__ == "__main__":
    # This preserves your original behavior: read signals.csv and write results.
    if not os.path.exists(DEFAULT_SIGNALS_FILE):
        raise FileNotFoundError(f"❌ No signals file at {DEFAULT_SIGNALS_FILE}")

    signals_df = pd.read_csv(DEFAULT_SIGNALS_FILE)
    # Expect original schema: date,ticker,signal,close
    signals_df["date"] = pd.to_datetime(signals_df["date"], errors="coerce")
    signals_df = signals_df.rename(columns={"close": "price"})

    ph, tl = simulate_portfolio(signals_df, starting_cash=DEFAULT_INITIAL_BALANCE, position_size=DEFAULT_POSITION_SIZE)

    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
    ph.to_csv(DEFAULT_PORTFOLIO_HISTORY_FILE, index=False)
    tl.to_csv(DEFAULT_TRADE_LOG_FILE, index=False)
    print("✅ Portfolio simulation complete (standalone).")
