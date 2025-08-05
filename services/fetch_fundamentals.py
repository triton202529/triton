import yfinance as yf
import pandas as pd
import numpy as np
import os

processed_path = "data/results"  # now reading tickers from results dir
output_path = "data/results/fundamentals.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Get tickers from per-ticker parquet files in results dir
tickers = [f.replace(".parquet", "") for f in os.listdir(processed_path) if f.endswith(".parquet")]

fundamentals = []
print(f"📡 Fetching fundamentals for {len(tickers)} tickers...")

for ticker in tickers:
    print(f"🔍 {ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        pe_ratio = info.get("trailingPE")
        eps = info.get("trailingEps")
        revenue = info.get("totalRevenue")
        market_cap = info.get("marketCap")
        pb_ratio = info.get("priceToBook")
        dividend_yield = info.get("dividendYield")

        # Fill defaults if missing
        if any(v is None for v in [pe_ratio, eps, revenue, market_cap, pb_ratio, dividend_yield]):
            print(f"⚠️ Missing fundamentals for {ticker} — using defaults")
            pe_ratio = pe_ratio if pe_ratio is not None else 15
            eps = eps if eps is not None else 5
            revenue = revenue if revenue is not None else 1e9
            market_cap = market_cap if market_cap is not None else 1e10
            pb_ratio = pb_ratio if pb_ratio is not None else 1.5
            dividend_yield = dividend_yield if dividend_yield is not None else 0

        score = (
            (1 / pe_ratio) * 0.25 +
            eps * 0.25 +
            (revenue / 1e9) * 0.25 +
            dividend_yield * 0.25
        )

        fundamentals.append({
            "ticker": ticker.upper(),
            "pe_ratio": pe_ratio,
            "eps": eps,
            "revenue": revenue,
            "market_cap": market_cap,
            "pb_ratio": pb_ratio,
            "dividend_yield": dividend_yield,
            "score": round(score, 4)
        })

    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")

df_out = pd.DataFrame(fundamentals)
df_out.to_csv(output_path, index=False)
print(f"✅ Fundamentals saved to {output_path}")
