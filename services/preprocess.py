import pandas as pd
import os

def preprocess_stock_csv(file_path: str) -> pd.DataFrame:
    print(f"\n🔍 Preprocessing {file_path}...")

    try:
        df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return pd.DataFrame()

    # Flatten multi-level columns: ("Close", "AAPL") => "close_aapl"
    df.columns = ['_'.join(col).lower().strip() for col in df.columns]

    # Attempt to find the close column automatically
    close_cols = [col for col in df.columns if col.startswith("close_")]

    if not close_cols:
        print(f"❌ No 'close_*' column found in {file_path}")
        return pd.DataFrame()

    close_col = close_cols[0]  # Pick the first matching column
    df = df[[close_col]].copy()  # Only keep the close column
    df.rename(columns={close_col: "close"}, inplace=True)

    # Drop missing data
    df.dropna(inplace=True)

    # Convert to numeric (just in case)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(inplace=True)

    # Compute returns and target
    df["daily_return"] = df["close"].pct_change()
    df["target"] = (df["daily_return"].shift(-1) > 0).astype(int)

    print(f"✅ Cleaned {file_path} — {len(df)} rows")
    return df
