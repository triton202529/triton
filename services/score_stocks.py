import pandas as pd
import os
import shutil

DATA_DIR = "data/processed"
FUND_FILE = os.path.join(DATA_DIR, "fundamentals.csv")
SCORED_FILE = "data/results/stock_scores.csv"
FUND_OUT = "data/results/fundamentals.csv"

print("📊 Scoring stocks using individual ticker files...")

# ✅ Load fundamentals (no index set)
fund = pd.read_csv(FUND_FILE)

momentum = {}

# Loop through all individual .parquet files (excluding the merged one)
for file in os.listdir(DATA_DIR):
    if file.endswith(".parquet") and file != "stock_data.parquet":
        ticker = file.replace(".parquet", "")
        path = os.path.join(DATA_DIR, file)

        try:
            df = pd.read_parquet(path)
            df = df.sort_values("date")

            first_close = df.iloc[0]["close"]
            last_close = df.iloc[-1]["close"]
            mom = (last_close - first_close) / first_close
            momentum[ticker] = mom

        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

# Merge into fundamentals
momentum_df = pd.DataFrame.from_dict(momentum, orient="index", columns=["momentum"])
momentum_df.reset_index(inplace=True)
momentum_df.rename(columns={"index": "ticker"}, inplace=True)

merged = pd.merge(fund, momentum_df, on="ticker", how="inner")

# Normalize and score
def normalize(series, inverse=False):
    s = series.copy()
    if inverse:
        s = s.max() - s
    return 100 * (s - s.min()) / (s.max() - s.min())

merged["score_pe"] = normalize(merged["pe_ratio"], inverse=True)
merged["score_eps"] = normalize(merged["eps"])
merged["score_rev"] = normalize(merged["revenue"])
merged["score_cap"] = normalize(merged["market_cap"])
merged["score_mom"] = normalize(merged["momentum"])

merged["total_score"] = (
    merged["score_pe"] * 0.2 +
    merged["score_eps"] * 0.2 +
    merged["score_rev"] * 0.2 +
    merged["score_cap"] * 0.2 +
    merged["score_mom"] * 0.2
)

final = merged[[
    "ticker",
    "pe_ratio", "eps", "revenue", "market_cap", "momentum",
    "score_pe", "score_eps", "score_rev", "score_cap", "score_mom",
    "total_score"
]].sort_values("total_score", ascending=False)

# Save outputs
os.makedirs("data/results", exist_ok=True)
final.to_csv(SCORED_FILE, index=False)
shutil.copy(FUND_FILE, FUND_OUT)

print(f"✅ Stock scores saved to {SCORED_FILE}")
print(f"📘 Fundamentals copied to {FUND_OUT}")
