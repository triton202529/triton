# services/simulate_portfolio.py
import pandas as pd
import os

INITIAL_BALANCE = 100000
POSITION_SIZE = 0.10
SIGNALS_FILE = "data/predictions/signals.csv"
PORTFOLIO_HISTORY_FILE = "data/results/portfolio_history.csv"
TRADE_LOG_FILE = "data/results/trade_log.csv"

signals_df = pd.read_csv(SIGNALS_FILE)
signals_df["date"] = pd.to_datetime(signals_df["date"])
signals_df = signals_df.sort_values("date")

cash = INITIAL_BALANCE
positions = {}
portfolio_history = []
trade_log = []

unique_dates = signals_df["date"].drop_duplicates().sort_values()

for current_date in unique_dates:
    daily_signals = signals_df[signals_df["date"] == current_date]

    for _, row in daily_signals.iterrows():
        ticker = row["ticker"]
        signal = row["signal"]
        price = row["close"]

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

os.makedirs("data/results", exist_ok=True)
pd.DataFrame(portfolio_history).to_csv(PORTFOLIO_HISTORY_FILE, index=False)
pd.DataFrame(trade_log).to_csv(TRADE_LOG_FILE, index=False)

print("✅ Portfolio simulation complete.")
