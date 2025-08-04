import pandas as pd
import os

# === Configuration ===
INITIAL_BALANCE = 100000
POSITION_SIZE = 0.10  # 10% of available cash
SIGNALS_FILE = "C:/Users/Akimw/triton/data/predictions/signals.csv"
PORTFOLIO_HISTORY_FILE = "data/results/portfolio_history.csv"
TRADE_LOG_FILE = "data/results/trade_log.csv"

# === Load Signals ===
if not os.path.exists(SIGNALS_FILE):
    raise FileNotFoundError(f"❌ Signals file not found at: {SIGNALS_FILE}")

signals_df = pd.read_csv(SIGNALS_FILE)
signals_df["date"] = pd.to_datetime(signals_df["date"])
signals_df = signals_df.sort_values("date")

# === Initialize Portfolio ===
cash = INITIAL_BALANCE
positions = {}
portfolio_history = []
trade_log = []

# === Run Simulation ===
unique_dates = signals_df["date"].drop_duplicates().sort_values()

for current_date in unique_dates:
    daily_signals = signals_df[signals_df["date"] == current_date]

    for _, row in daily_signals.iterrows():
        ticker = row["ticker"]
        signal = row["signal"]
        price = row["close"]

        # SELL logic
        if signal == "SELL" and ticker in positions:
            shares = positions[ticker]["shares"]
            proceeds = shares * price
            cash += proceeds
            trade_log.append({
                "date": current_date.date(),
                "ticker": ticker,
                "action": "SELL",
                "shares": shares,
                "price": price,
                "value": proceeds
            })
            del positions[ticker]

        # BUY logic
        elif signal == "BUY" and ticker not in positions:
            budget = cash * POSITION_SIZE
            shares = int(budget // price)
            if shares > 0:
                cost = shares * price
                cash -= cost
                positions[ticker] = {
                    "entry_price": price,
                    "shares": shares,
                    "entry_date": current_date.date()
                }
                trade_log.append({
                    "date": current_date.date(),
                    "ticker": ticker,
                    "action": "BUY",
                    "shares": shares,
                    "price": price,
                    "value": cost
                })

    # Calculate daily portfolio value
    market_value = 0
    for ticker, pos in positions.items():
        latest_price_row = daily_signals[daily_signals["ticker"] == ticker]
        if not latest_price_row.empty:
            current_price = latest_price_row["close"].values[0]
            market_value += pos["shares"] * current_price

    total_value = cash + market_value
    portfolio_history.append({
        "date": current_date.date(),
        "cash": round(cash, 2),
        "market_value": round(market_value, 2),
        "total_value": round(total_value, 2)
    })

# === Save Results ===
os.makedirs("data/results", exist_ok=True)

pd.DataFrame(portfolio_history).to_csv(PORTFOLIO_HISTORY_FILE, index=False)
pd.DataFrame(trade_log).to_csv(TRADE_LOG_FILE, index=False)

print("✅ Portfolio simulation complete.")
print(f"📈 Results saved to: {PORTFOLIO_HISTORY_FILE}")
print(f"📄 Trade log saved to: {TRADE_LOG_FILE}")
