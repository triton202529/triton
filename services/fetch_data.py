import yfinance as yf
import pandas as pd
import os

TICKERS = [
    "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "NFLX", "AMD", "INTC",
    "SPY", "QQQ", "DIA", "VTI", "ARKK",
    "XLF", "XLE", "XLY", "XLV", "XLI", "XLK", "XLP", "XLU", "XLRE", "XLB",
    "GLD", "SLV", "BITO", "GBTC", "USO", "UNG", "DBA",
    "^GSPC", "^IXIC", "^DJI", "^VIX",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW",
    "BLK", "BRK-B", "GE", "UNH", "JNJ", "PG", "V", "MA", "PEP", "KO",
    "CVX", "XOM", "WMT", "HD", "DIS", "T", "PFE", "ABBV", "MRK"
]

START_DATE = "2023-01-01"
END_DATE = "2023-12-31"
OUTPUT_DIR = "data"

def fetch_data():
    all_data = []

    for ticker in TICKERS:
        print(f"\n📥 Fetching {ticker}...")
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False
        )

        if df.empty or "Close" not in df.columns:
            print(f"❌ Skipping {ticker} — no valid data.")
            continue

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        df["symbol"] = ticker
        all_data.append(df)

        print(f"✅ Got {len(df)} rows for {ticker}")

    if not all_data:
        raise RuntimeError("❌ No valid data fetched. Check internet or ticker list.")

    return pd.concat(all_data, ignore_index=True)

def add_features(df):
    print("✨ Adding features...")

    # Ensure 'close' column is valid and numeric
    if "close" not in df.columns:
        raise ValueError("❌ 'close' column is missing after fetching data.")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    df['ma7'] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(window=7).mean())
    df['ma21'] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(window=21).mean())
    df['returns'] = df.groupby("symbol")["close"].pct_change()

    print("🧠 Feature columns added: ma7, ma21, returns")
    return df

def save_to_csv(df, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"✅ Data saved to {output_path}")

def main():
    df = fetch_data()
    print(f"📊 Combined shape before features: {df.shape}")
    print(df.head())

    df = add_features(df)

    print(f"📈 Final shape after features: {df.shape}")
    save_to_csv(df, "cleaned_data.csv")

if __name__ == "__main__":
    main()
