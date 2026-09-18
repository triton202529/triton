# services/feature_generator.py

import pandas as pd

# Yahoo event columns are 0 on most days. Concatenating ETFs (which include
# capital_gains) with stocks (which do not) fills those stock rows with NaN.
EVENT_COLUMNS = ("dividends", "stock_splits", "capital_gains")
INDICATOR_COLUMNS = (
    "ma7",
    "ma21",
    "ema12",
    "ema26",
    "rsi",
    "macd",
    "volatility",
    "returns",
)


def fill_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace concat-induced NaNs in sparse Yahoo event columns with 0."""
    df = df.copy()
    for col in EVENT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def combine_ticker_frames(frames) -> pd.DataFrame:
    """Concat ticker frames without wiping stocks that lack ETF-only columns."""
    combined = pd.concat(frames, ignore_index=True)
    combined = fill_event_columns(combined)
    return combined.dropna(subset=["close"])


def add_technical_indicators(df: pd.DataFrame, spy_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Adds technical indicators to the stock DataFrame.
    Optionally includes SPY data for comparison features.
    """

    df = df.copy()

    # === Basic moving averages ===
    df["ma7"] = df["close"].rolling(window=7).mean()
    df["ma21"] = df["close"].rolling(window=21).mean()

    # === Exponential moving averages ===
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

    # === Relative Strength Index (RSI) ===
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # === MACD (Moving Average Convergence Divergence) ===
    df["macd"] = df["ema12"] - df["ema26"]

    # === Volatility ===
    df["volatility"] = df["close"].rolling(window=10).std()

    # === Returns ===
    df["returns"] = df["close"].pct_change()

    # === Spy comparison (optional) ===
    if spy_df is not None and not spy_df.empty:
        spy_df = spy_df[["date", "close"]].rename(columns={"close": "spy_close"})
        df = df.merge(spy_df, on="date", how="left")
        df["spy_returns"] = df["spy_close"].pct_change()

    # Drop rolling-window warmup rows only. A bare dropna() also drops every
    # stock row when concat aligned an all-NaN capital_gains column onto them.
    df = fill_event_columns(df)
    subset = [c for c in INDICATOR_COLUMNS if c in df.columns]
    if "spy_returns" in df.columns and df["spy_returns"].notna().any():
        subset.append("spy_returns")
    df.dropna(subset=subset, inplace=True)
    return df
