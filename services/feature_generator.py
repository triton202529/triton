# services/feature_generator.py

import pandas as pd

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

    # Final cleanup
    df.dropna(inplace=True)
    return df
