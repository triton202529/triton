# services/evaluate_strategy.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
RESULTS_DIR = "../data/results"
BACKTEST_FILE = os.path.join(RESULTS_DIR, "signal_backtest.csv")

# Check if file exists
if not os.path.exists(BACKTEST_FILE):
    print(f"❌ Backtest file not found: {BACKTEST_FILE}")
    exit(1)

# Load backtest data
df = pd.read_csv(BACKTEST_FILE)

# Ensure required columns exist
required_cols = ["date", "symbol", "action", "actual_price", "predicted_price", "pnl"]
if not all(col in df.columns for col in required_cols):
    print("❌ Missing required columns in backtest file.")
    exit(1)

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Sort by date
df.sort_values("date", inplace=True)

# Compute cumulative PnL
df["cumulative_pnl"] = df["pnl"].cumsum()

# Print summary
total_trades = len(df)
wins = (df["pnl"] > 0).sum()
losses = (df["pnl"] <= 0).sum()
avg_pnl = df["pnl"].mean()
win_rate = wins / total_trades * 100 if total_trades > 0 else 0

print("\n📈 Strategy Evaluation Summary")
print("-----------------------------------")
print(f"Total Trades      : {total_trades}")
print(f"Winning Trades    : {wins}")
print(f"Losing Trades     : {losses}")
print(f"Win Rate (%)      : {win_rate:.2f}")
print(f"Average PnL ($)   : {avg_pnl:.2f}")
print(f"Total Cumulative PnL ($): {df['cumulative_pnl'].iloc[-1]:.2f}")

# Plot cumulative PnL over time
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["cumulative_pnl"], label="Cumulative PnL", color="green")
plt.title("📊 Cumulative PnL Over Time")
plt.xlabel("Date")
plt.ylabel("PnL ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
