import yfinance as yf
import pandas as pd
import os

RESULTS_DIR = "data/results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "fundamentals.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Get tickers from all .parquet files in results dir
tickers = [f.replace(".parquet", "") for f in os.listdir(RESULTS_DIR) if f.endswith(".parquet")]

fundamentals = []
print(f"📡 Fetching fundamentals for {len(tickers)} tickers...")

for ticker in tickers:
    print(f"🔍 {ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        pe_ratio = info.get("trailingPE", 15)
        eps = info.get("trailingEps", 5)
        revenue = info.get("totalRevenue", 1e9)
        market_cap = info.get("marketCap", 1e10)
        pb_ratio = info.get("priceToBook", 1.5)
        dividend_yield = info.get("dividendYield", 0)

        fundamentals.append({
            "ticker": ticker.upper(),
            "pe_ratio": pe_ratio,
            "eps": eps,
            "revenue": revenue,
            "market_cap": market_cap,
            "pb_ratio": pb_ratio,
            "dividend_yield": dividend_yield
        })

    except Exception as e:
        print(f"⚠️ Error fetching {ticker}, using defaults: {e}")
        fundamentals.append({
            "ticker": ticker.upper(),
            "pe_ratio": 15,
            "eps": 5,
            "revenue": 1e9,
            "market_cap": 1e10,
            "pb_ratio": 1.5,
            "dividend_yield": 0
        })

df = pd.DataFrame(fundamentals)
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Fundamentals saved to {OUTPUT_PATH}")
