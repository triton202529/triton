# services/performance_metrics.py

import pandas as pd
import numpy as np
import os

portfolio_file = "data/results/portfolio_history.csv"
trades_file = "data/results/trade_log.csv"
output_file = "data/results/backtest_summary.csv"

# Load data
portfolio = pd.read_csv(portfolio_file)
trades = pd.read_csv(trades_file)

# Convert dates
portfolio["date"] = pd.to_datetime(portfolio["date"])
trades["date"] = pd.to_datetime(trades["date"])

# Total Return
initial_value = portfolio["total_value"].iloc[0]
final_value = portfolio["total_value"].iloc[-1]
total_return = (final_value - initial_value) / initial_value

# Annualized Return
days = (portfolio["date"].iloc[-1] - portfolio["date"].iloc[0]).days
annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

# Daily Returns
portfolio["daily_return"] = portfolio["total_value"].pct_change().fillna(0)
sharpe_ratio = (portfolio["daily_return"].mean() / portfolio["daily_return"].std()) * np.sqrt(252)

# Max Drawdown
cumulative = portfolio["total_value"].cummax()
drawdown = (portfolio["total_value"] - cumulative) / cumulative
max_drawdown = drawdown.min()

# Trade win rate
trades["return"] = trades.groupby("ticker")["price"].pct_change().fillna(0)
win_rate = (trades["return"] > 0).mean()

# Average trade return
avg_trade_return = trades["return"].mean()

# Save metrics
summary = {
    "Total Return": [f"{total_return:.2%}"],
    "Annualized Return": [f"{annualized_return:.2%}"],
    "Sharpe Ratio": [f"{sharpe_ratio:.2f}"],
    "Max Drawdown": [f"{max_drawdown:.2%}"],
    "Win Rate": [f"{win_rate:.2%}"],
    "Average Trade Return": [f"{avg_trade_return:.2%}"]
}

summary_df = pd.DataFrame(summary)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
summary_df.to_csv(output_file, index=False)

print("✅ Performance metrics saved to backtest_summary.csv")
