import pandas as pd
import os

DATA_DIR = "data/results"
FUND_FILE = os.path.join(DATA_DIR, "fundamentals.csv")
SCORED_FILE = os.path.join(DATA_DIR, "stock_scores.csv")

print("📊 Scoring stocks using individual ticker files...")

fund = pd.read_csv(FUND_FILE)
fund["ticker"] = fund["ticker"].str.upper()

# Ensure all tickers in results dir exist in fundamentals
all_tickers = [f.replace(".parquet", "").upper() for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
for t in all_tickers:
    if t not in fund["ticker"].values:
        fund = pd.concat([fund, pd.DataFrame([{
            "ticker": t,
            "pe_ratio": 15,
            "eps": 5,
            "revenue": 1e9,
            "market_cap": 1e10,
            "pb_ratio": 1.5,
            "dividend_yield": 0
        }])], ignore_index=True)

# Calculate momentum
momentum = {}
for file in os.listdir(DATA_DIR):
    if file.endswith(".parquet"):
        ticker = file.replace(".parquet", "").upper()
        path = os.path.join(DATA_DIR, file)
        try:
            df = pd.read_parquet(path).sort_values("date")
            if len(df) >= 2:
                mom = (df.iloc[-1]["close"] - df.iloc[0]["close"]) / df.iloc[0]["close"]
            else:
                mom = 0
            momentum[ticker] = mom
        except:
            momentum[ticker] = 0

momentum_df = pd.DataFrame(list(momentum.items()), columns=["ticker", "momentum"])
merged = pd.merge(fund, momentum_df, on="ticker", how="left").fillna({"momentum": 0})

# Normalize and score
def normalize(series, inverse=False):
    if series.max() == series.min():
        return pd.Series([50] * len(series), index=series.index)
    s = series.max() - series if inverse else series
    return 100 * (s - s.min()) / (s.max() - s.min())

merged["score_pe"] = normalize(merged["pe_ratio"], inverse=True)
merged["score_eps"] = normalize(merged["eps"])
merged["score_rev"] = normalize(merged["revenue"])
merged["score_cap"] = normalize(merged["market_cap"])
merged["score_mom"] = normalize(merged["momentum"])

merged["total_score"] = merged[["score_pe", "score_eps", "score_rev", "score_cap", "score_mom"]].mean(axis=1)

merged.to_csv(SCORED_FILE, index=False)
print(f"✅ Stock scores saved to {SCORED_FILE}")
