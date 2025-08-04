# services/simulate_trades.py

import pandas as pd
import os

print("🚀 Simulating trades based on signals...")

# File paths
signals_file = "data/predictions/signals.csv"
trade_log_file = "data/results/trade_log.csv"
portfolio_file = "data/results/portfolio_history.csv"

# Parameters
initial_cash = 100_000
trade_amount = 5_000

# Load signals
df = pd.read_csv(signals_file)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

# Track portfolio state
cash = initial_cash
holdings = {}  # ticker -> (shares, avg_price)
trade_log = []
portfolio_history = []

for _, row in df.iterrows():
    date = row["date"]
    ticker = row["ticker"]
    price = row["close"]
    signal = row["signal"]

    shares_held, avg_price = holdings.get(ticker, (0, 0))

    if signal == "BUY" and cash >= trade_amount:
        shares_to_buy = trade_amount // price
        cost = shares_to_buy * price
        if shares_to_buy > 0:
            cash -= cost
            new_total_shares = shares_held + shares_to_buy
            avg_price = (shares_held * avg_price + shares_to_buy * price) / new_total_shares
            holdings[ticker] = (new_total_shares, avg_price)
            trade_log.append({
                "date": date, "ticker": ticker, "action": "BUY",
                "shares": shares_to_buy, "price": price, "value": cost
            })

    elif signal == "SELL" and shares_held > 0:
        proceeds = shares_held * price
        cash += proceeds
        trade_log.append({
            "date": date, "ticker": ticker, "action": "SELL",
            "shares": shares_held, "price": price, "value": proceeds
        })
        holdings.pop(ticker)

    # Portfolio snapshot
    market_value = sum(sh * price for tkr, (sh, _) in holdings.items() if tkr == ticker)
    total_value = cash + market_value
    portfolio_history.append({
        "date": date, "cash": cash, "market_value": market_value, "total_value": total_value
    })

# Save results
os.makedirs("data/results", exist_ok=True)
pd.DataFrame(trade_log).to_csv(trade_log_file, index=False)
pd.DataFrame(portfolio_history).to_csv(portfolio_file, index=False)

print(f"✅ Trade log saved to: {trade_log_file}")
print(f"📈 Portfolio history saved to: {portfolio_file}")
