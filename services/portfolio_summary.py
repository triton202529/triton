import pandas as pd
import os

# === Config ===
PORTFOLIO_FILE = "data/results/portfolio_history.csv"
SUMMARY_FILE = "data/results/portfolio_performance_summary.csv"

# === Load Data ===
if not os.path.exists(PORTFOLIO_FILE):
    raise FileNotFoundError(f"❌ Portfolio file not found at: {PORTFOLIO_FILE}")

portfolio = pd.read_csv(PORTFOLIO_FILE)

# Handle missing or malformed data
if "strategy_value" not in portfolio.columns:
    raise ValueError("❌ Column 'strategy_value' not found in portfolio history file.")

portfolio["date"] = pd.to_datetime(portfolio["date"])
portfolio = portfolio.sort_values("date").reset_index(drop=True)

# === Calculate Performance ===
initial_value = portfolio.iloc[0]["strategy_value"]
final_value = portfolio.iloc[-1]["strategy_value"]
total_return = ((final_value - initial_value) / initial_value) * 100
duration = (portfolio.iloc[-1]["date"] - portfolio.iloc[0]["date"]).days
average_daily_return = (portfolio["strategy_value"].pct_change().mean()) * 100

# === Create Summary ===
summary = {
    "Initial Value": round(initial_value, 2),
    "Final Value": round(final_value, 2),
    "Total Return (%)": round(total_return, 2),
    "Average Daily Return (%)": round(average_daily_return, 4),
    "Backtest Duration (Days)": duration,
}

# === Save Output ===
os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)

# === Display ===
print("✅ Portfolio performance summary generated:")
for key, value in summary.items():
    print(f"{key}: {value}")
