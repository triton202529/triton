# services/backtest_strategy.py

import pandas as pd
import os

ticker = "AAPL"  # You can later loop over multiple tickers
predictions_path = f"../data/predictions/{ticker}_predictions.parquet"

if not os.path.exists(predictions_path):
    print(f"❌ Predictions file not found: {predictions_path}")
    exit(1)

print(f"📈 Running backtest for {ticker}...")

df = pd.read_parquet(predictions_path)

# Simple Strategy: Buy if tomorrow's predicted close > today's close
df["signal"] = (df["prediction"] > df["close"]).astype(int)
df["daily_return"] = df["close"].pct_change()
df["strategy_return"] = df["daily_return"] * df["signal"].shift(1)  # Use signal from previous day

df.dropna(inplace=True)

# 🔧 Clip extreme strategy returns to prevent unrealistic spikes
df["strategy_return"] = df["strategy_return"].clip(lower=-1, upper=1)

# Cumulative returns
df["cumulative_market"] = (1 + df["daily_return"]).cumprod()
df["cumulative_strategy"] = (1 + df["strategy_return"]).cumprod()

# Save results
results_path = f"../data/results/{ticker}_backtest.csv"
os.makedirs(os.path.dirname(results_path), exist_ok=True)
df.to_csv(results_path, index=False)

print(f"✅ Backtest completed and saved to {results_path}")
