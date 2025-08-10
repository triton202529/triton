# services/backtest_signals.py
import pandas as pd
import os

# ✅ Use correct signal file
signals_file = "data/results/signals_with_rationale.csv"
output_dir = "data/results"
os.makedirs(output_dir, exist_ok=True)

print("🔁 Running signal-based strategy backtest with SL/TP...")

# Load signals
df = pd.read_csv(signals_file)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

# Setup
initial_balance = 100_000
trade_log = []
portfolio_history = []
summary = []

tickers = df["ticker"].unique()

# Strategy vs Market DataFrame
df["returns"] = df.groupby("ticker")["close"].pct_change()
signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
df["signal_numeric"] = df["signal"].map(signal_map)
df["position"] = df.groupby("ticker")["signal_numeric"].shift(1).fillna(0)
df["strategy_return"] = df["returns"] * df["position"]
df["cumulative_market"] = df.groupby("ticker")["returns"].transform(lambda x: (1 + x).cumprod())
df["cumulative_strategy"] = df.groupby("ticker")["strategy_return"].transform(lambda x: (1 + x).cumprod())

# Backtest loop
for ticker in tickers:
    data = df[df["ticker"] == ticker].copy()
    balance = initial_balance
    position = 0
    entry_price = 0
    trades = 0
    profit = 0

    for _, row in data.iterrows():
        date = row["date"]
        signal = row["signal"]
        price = row["close"]

        stop_loss = entry_price * 0.95 if entry_price > 0 else None
        take_profit = entry_price * 1.05 if entry_price > 0 else None

        if signal == "BUY" and position == 0:
            quantity = int(balance / price)
            if quantity > 0:
                position = quantity
                entry_price = price
                balance -= quantity * price
                trades += 1
                trade_log.append({
                    "date": date,
                    "ticker": ticker,
                    "action": "BUY",
                    "price": price,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "exit_price": None,
                    "signal": "BUY",
                    "profit": None,
                    "stop_loss": price * 0.95,
                    "take_profit": price * 1.05
                })

        elif signal == "SELL" and position > 0:
            exit_price = price
            trade_profit = (exit_price - entry_price) * position
            balance += position * exit_price
            profit += trade_profit
            trade_log.append({
                "date": date,
                "ticker": ticker,
                "action": "SELL",
                "price": price,
                "quantity": position,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "signal": "SELL",
                "profit": trade_profit,
                "stop_loss": entry_price * 0.95,
                "take_profit": entry_price * 1.05
            })
            position = 0

        # Save portfolio snapshot
        market_value = position * price
        total_value = balance + market_value
        portfolio_history.append({
            "date": date,
            "cash": balance,
            "market_value": market_value,
            "total_value": total_value
        })

    # Final portfolio summary
    final_price = data.iloc[-1]["close"]
    ending_value = balance + position * final_price
    total_return = (ending_value - initial_balance) / initial_balance * 100

    summary.append({
        "ticker": ticker,
        "trades": trades,
        "profit": round(profit, 2),
        "final_value": round(ending_value, 2),
        "return_pct": round(total_return, 2)
    })

# Convert to DataFrames
trade_log_df = pd.DataFrame(trade_log)
portfolio_history_df = pd.DataFrame(portfolio_history)
summary_df = pd.DataFrame(summary)

# Save results with both backtest_ prefix and default dashboard files
trade_log_df.to_csv(f"{output_dir}/backtest_trade_log.csv", index=False)
portfolio_history_df.to_csv(f"{output_dir}/backtest_portfolio_history.csv", index=False)
summary_df.to_csv(f"{output_dir}/backtest_summary.csv", index=False)
df.to_csv(f"{output_dir}/strategy_vs_market.csv", index=False)

# ✅ Also overwrite main dashboard files
trade_log_df.to_csv(f"{output_dir}/trade_log.csv", index=False)
portfolio_history_df.to_csv(f"{output_dir}/portfolio_history.csv", index=False)

print("✅ Backtest completed. Saved:")
print("   📄 backtest_trade_log.csv")
print("   📄 backtest_portfolio_history.csv")
print("   📄 backtest_summary.csv")
print("   📄 strategy_vs_market.csv")
print("   📄 trade_log.csv (for dashboard)")
print("   📄 portfolio_history.csv (for dashboard)")
