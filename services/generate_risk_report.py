# services/generate_risk_report.py

import pandas as pd
import numpy as np
import os

DATA_DIR = "data/processed"
RESULT_FILE = "data/results/risk_report.csv"

print("📉 Generating risk report...")

records = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".parquet") and file != "stock_data.parquet":
        ticker = file.replace(".parquet", "")
        path = os.path.join(DATA_DIR, file)

        try:
            df = pd.read_parquet(path).sort_values("date")
            df["returns"] = df["close"].pct_change()

            # Volatility: standard deviation of daily returns
            volatility = df["returns"].std()

            # Max Drawdown
            df["cumulative"] = (1 + df["returns"]).cumprod()
            peak = df["cumulative"].cummax()
            drawdown = (df["cumulative"] - peak) / peak
            max_drawdown = drawdown.min()

            records.append({
                "ticker": ticker,
                "volatility": volatility,
                "max_drawdown": max_drawdown
            })

        except Exception as e:
            print(f"⚠️ Failed to process {ticker}: {e}")

# Create DataFrame
risk_df = pd.DataFrame(records)

# Normalize for risk score (lower vol & drawdown = higher score)
risk_df["risk_score"] = 100 - (
    50 * (risk_df["volatility"] - risk_df["volatility"].min()) / (risk_df["volatility"].max() - risk_df["volatility"].min()) +
    50 * (risk_df["max_drawdown"].abs() - risk_df["max_drawdown"].abs().min()) / (risk_df["max_drawdown"].abs().max() - risk_df["max_drawdown"].abs().min())
)

risk_df["risk_score"] = risk_df["risk_score"].round(2)
risk_df = risk_df.sort_values("risk_score", ascending=False)

# Save
os.makedirs("data/results", exist_ok=True)
risk_df.to_csv(RESULT_FILE, index=False)
print(f"✅ Risk report saved to {RESULT_FILE}")
