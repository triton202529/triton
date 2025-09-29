# pipeline_real_data.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = DATA_DIR / "results"
ORDERS_DIR   = DATA_DIR / "orders"
for p in (RESULTS_DIR, ORDERS_DIR):
    p.mkdir(parents=True, exist_ok=True)

def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Make sure columns are single-level and canonical: open, high, low, close, adj_close, volume."""
    if isinstance(df.columns, pd.MultiIndex):
        # If it's like ('Open','AAPL'), ('Close','AAPL'), keep level 0 only.
        if df.columns.nlevels == 2 and len(set(df.columns.get_level_values(1))) == 1:
            df.columns = [str(l0) for l0, _ in df.columns]
        else:
            df.columns = [
                "_".join([str(x) for x in tup if x not in (None, "", "nan")])
                for tup in df.columns
            ]

    # lower + underscores
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))

    # remove trailing _<ticker> if present (e.g., close_aapl -> close)
    suff = f"_{ticker.lower()}"
    ren = {}
    for c in df.columns:
        if c.endswith(suff):
            ren[c] = c[: -len(suff)]
    if ren:
        df = df.rename(columns=ren)

    # map common variants
    alias_map = {
        "adj_close": "adj_close",
        "adjclose": "adj_close",
        "close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
    }
    ren = {}
    for c in list(df.columns):
        base = c
        if base in alias_map:
            ren[c] = alias_map[base]
        # keep unknowns untouched
    if ren:
        df = df.rename(columns=ren)

    # If adj_close missing, create from close
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    return df

def fetch_prices(ticker: str, days: int) -> pd.DataFrame:
    """Fetch daily OHLCV and save to parquet for candlesticks."""
    df = yf.download(ticker, period=f"{days}d", interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index().rename(columns={"Date": "date"})
    # normalize date to naive (dashboard expects naive)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

    df = _flatten_columns(df, ticker)

    # ensure we have core columns
    needed = {"date", "open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        # Try to salvage: sometimes yfinance returns 'adj_close' but not 'close'
        if "close" not in df.columns and "adj_close" in df.columns:
            df["close"] = df["adj_close"]
    # Final check
    if "close" not in df.columns:
        return pd.DataFrame()

    # write OHLC parquet for Streamlit candlesticks
    keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df[keep_cols].to_parquet(RESULTS_DIR / f"{ticker}.parquet", index=False)
    return df

def compute_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """Toy signals + predictions for every row (deterministic, robust)."""
    if prices.empty or "close" not in prices.columns:
        return pd.DataFrame()

    out = prices.loc[:, ["date", "close"]].copy()

    # Ensure numeric close
    close_series = pd.to_numeric(out["close"], errors="coerce")

    # SMAs
    out["sma20"] = close_series.rolling(20, min_periods=1).mean()
    out["sma50"] = close_series.rolling(50, min_periods=1).mean()

    # RSI(14)
    delta = close_series.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs    = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))

    # Predicted close: EMA(10) shifted 1 forward, then safely filled
    ema = close_series.ewm(span=10, adjust=False).mean()
    pred = ema.shift(1).fillna(out["sma20"]).fillna(close_series).astype(float)
    out["predicted_close"] = pred

    # SAFE 1-D arrays for edge
    close_np = close_series.to_numpy(dtype=float)
    pred_np  = pd.to_numeric(out["predicted_close"], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(close_np) & np.isfinite(pred_np) & (close_np != 0.0)
    edge = np.full_like(close_np, np.nan, dtype=float)
    np.divide(pred_np - close_np, close_np, out=edge, where=mask)
    out["edge_pct"] = edge

    # confidence: |edge| capped at 5% -> 0..1
    conf = np.clip(np.abs(edge), 0.0, 0.05) / 0.05
    conf = np.where(np.isfinite(conf), conf, 0.0)
    out["confidence"] = conf

    # signal + rationale
    def classify(e):
        if not np.isfinite(e): return "HOLD", "No strong edge"
        if e > 0.01:          return "BUY",  "Positive price edge vs EMA"
        if e < -0.01:         return "SELL", "Negative price edge vs EMA"
        return "HOLD", "No strong edge"

    sigs, rats = zip(*[classify(v) for v in edge])
    out["signal"]    = list(sigs)
    out["rationale"] = list(rats)

    return out[[
        "date", "close", "predicted_close", "signal", "confidence",
        "rsi14", "sma20", "sma50", "edge_pct", "rationale"
    ]]

def main(tickers: list[str], days: int):
    all_rows = []
    for t in tickers:
        print(f"Fetching {t}...")
        px = fetch_prices(t, days)
        if px.empty:
            print(f"Warning: no usable data for {t}")
            continue
        sig = compute_signals(px)
        if sig.empty:
            print(f"Warning: no signals for {t}")
            continue
        sig.insert(1, "ticker", t)  # after 'date'
        all_rows.append(sig)

    if not all_rows:
        print("Nothing to write.")
        return

    signals = pd.concat(all_rows, ignore_index=True)
    # ISO date text for CSV (Streamlit will reparse)
    signals["date"] = pd.to_datetime(signals["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    out_csv = RESULTS_DIR / "signals_with_rationale.csv"
    signals.to_csv(out_csv, index=False)
    signals.drop(columns=["rationale"]).to_csv(RESULTS_DIR / "signals.csv", index=False)
    print(f"Wrote {len(signals):,} rows -> {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tickers", nargs="+", required=True, help="Tickers e.g. AAPL MSFT")
    ap.add_argument("-d", "--days", type=int, default=365, help="Lookback days (default 365)")
    args = ap.parse_args()
    main(args.tickers, args.days)
