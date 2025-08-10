# services/run_paper_trading.py

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# --- make sure we can import from the project root when run directly ---
try:
    from services.simulate_portfolio import simulate_portfolio
except Exception:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from services.simulate_portfolio import simulate_portfolio

RESULTS_DIR_DEFAULT = "data/results"
SIGNALS_FILE_DEFAULT = os.path.join(RESULTS_DIR_DEFAULT, "signals_with_rationale.csv")

def load_signals(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No signals file found at {path}")

    df = pd.read_csv(path)

    # Basic normalization / validation
    if "date" not in df.columns:
        raise ValueError("signals file missing required column 'date'")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # tolerate either 'price' or 'close'
    if "price" not in df.columns:
        if "close" in df.columns:
            df = df.rename(columns={"close": "price"})
        else:
            raise ValueError("signals file must include 'price' or 'close'")

    need = {"ticker", "signal", "price"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"signals file missing columns: {sorted(missing)}")

    # tidy
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["signal"] = df["signal"].astype(str).str.upper()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "signal", "price"])

    return df

def choose_latest_day(df: pd.DataFrame) -> pd.Timestamp:
    # Use normalized (date-only) max to avoid tz/naive mixups
    dates = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return dates.max()

def filter_actionable(
    df: pd.DataFrame,
    latest_day: pd.Timestamp,
    min_confidence: float | None = None,
    require_non_hold: bool = True
) -> pd.DataFrame:
    sub = df[df["date"].dt.normalize() == latest_day].copy()

    if require_non_hold:
        sub = sub[sub["signal"].isin(["BUY", "SELL"])]

    # Optional confidence filtering if column exists
    if min_confidence is not None and "confidence" in sub.columns:
        # Some pipelines store confidence as delta pct; we still allow numeric filter
        sub["confidence"] = pd.to_numeric(sub["confidence"], errors="coerce")
        sub = sub[sub["confidence"].abs() >= float(min_confidence)]

    return sub

def main():
    p = argparse.ArgumentParser(description="Run paper-trading on latest signals day.")
    p.add_argument("--signals-file", default=SIGNALS_FILE_DEFAULT, help="Path to signals CSV")
    p.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT, help="Where to write results CSVs")
    p.add_argument("--out-prefix", default="paper_", help="Prefix for output files (portfolio_history/trade_log)")
    p.add_argument("--starting-cash", type=float, default=100000.0, help="Initial cash balance")
    p.add_argument("--position-size", type=float, default=0.10, help="Fraction of cash to allocate per new BUY")
    p.add_argument("--min-confidence", type=float, default=None, help="Min |confidence| to trade (if column exists)")
    p.add_argument("--allow-hold", action="store_true", help="Include HOLD rows (generally not useful)")
    args = p.parse_args()

    signals = load_signals(args.signals_file)
    if signals.empty:
        print("⚠️ Signals file is empty after cleaning.")
        return

    latest_day = choose_latest_day(signals)
    if pd.isna(latest_day):
        print("⚠️ Could not determine a valid latest day in signals.")
        return

    actionable = filter_actionable(
        signals,
        latest_day,
        min_confidence=args.min_confidence,
        require_non_hold=not args.allow_hold
    )

    print(f"📅 Running paper trades for {latest_day.date()} (rows: {len(actionable)})")
    if actionable.empty:
        print("⚠️ No actionable signals for latest day.")
        return

    # simulate_portfolio expects columns: date, ticker, signal, price
    ph, tl = simulate_portfolio(
        actionable[["date", "ticker", "signal", "price"]].copy(),
        starting_cash=args.starting_cash,
        position_size=args.position_size,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    out_port = os.path.join(args.results_dir, f"{args.out_prefix}portfolio_history.csv")
    out_trades = os.path.join(args.results_dir, f"{args.out_prefix}trade_log.csv")
    ph.to_csv(out_port, index=False)
    tl.to_csv(out_trades, index=False)

    print("✅ Portfolio simulation complete.")
    print(f"💾 Saved → {out_port}")
    print(f"💾 Saved → {out_trades}")

if __name__ == "__main__":
    main()
