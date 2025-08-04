# services/generate_trade_summary.py

import pandas as pd
import os

TRADE_LOG_FILE = "data/results/trade_log.csv"
OUTPUT_FILE = "data/results/trades_detailed.csv"

if not os.path.exists(TRADE_LOG_FILE):
    raise FileNotFoundError(f"❌ Trade log not found at: {TRADE_LOG_FILE}")

print("🔍 Loading trade log...")
df = pd.read_csv(TRADE_LOG_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["ticker", "date"])

detailed_trades = []

# Group trades by ticker
for ticker, group in df.groupby("ticker"):
    group = group.sort_values("date").reset_index(drop=True)
    holding = None

    for _, row in group.iterrows():
        action = row["action"]
        price = row["price"]
        date = row["date"]

        if action == "BUY":
            holding = {
                "ticker": ticker,
                "entry_date": date,
                "entry_price": price,
            }
        elif action == "SELL" and holding:
            exit_date = date
            exit_price = price
            profit = (exit_price - holding["entry_price"])
            return_pct = (profit / holding["entry_price"]) * 100
            days_held = (exit_date - holding["entry_date"]).days

            detailed_trades.append({
                "ticker": ticker,
                "entry_date": holding["entry_date"].date(),
                "exit_date": exit_date.date(),
                "entry_price": round(holding["entry_price"], 2),
                "exit_price": round(exit_price, 2),
                "profit": round(profit, 2),
                "return_pct": round(return_pct, 2),
                "days_held": days_held
            })

            holding = None  # Reset

# Save to CSV
pd.DataFrame(detailed_trades).to_csv(OUTPUT_FILE, index=False)
print(f"✅ Trade outcome summary saved to: {OUTPUT_FILE}")
