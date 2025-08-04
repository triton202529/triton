import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

SIGNALS_PATH = "../predictions/signals.csv" 
PRICES_PATH = "data/combined_prices.csv"
TRADE_LOG_PATH = "data/trade_log.csv"
PORTFOLIO_VALUE_PATH = "data/portfolio_value.csv"

# Load predicted signals
signals_df = pd.read_csv(SIGNALS_PATH)
signals_df['date'] = pd.to_datetime(signals_df['date'])

# Load historical prices
prices_df = pd.read_csv(PRICES_PATH)
prices_df['date'] = pd.to_datetime(prices_df['date'])

# Initialize portfolio
initial_cash = 10000
cash = initial_cash
portfolio = {}
portfolio_values = []
trade_log = []

for date in signals_df['date'].unique():
    date = pd.to_datetime(date)
    daily_signals = signals_df[signals_df['date'] == date]

    for _, row in daily_signals.iterrows():
        ticker = row['ticker']
        signal = row['signal']
        price_row = prices_df[(prices_df['date'] == date) & (prices_df['ticker'] == ticker)]

        if price_row.empty:
            continue

        price = price_row.iloc[0]['close']

        if signal == 'BUY' and cash >= price:
            quantity = int(cash // price)
            if quantity > 0:
                cash -= quantity * price
                portfolio[ticker] = portfolio.get(ticker, 0) + quantity
                trade_log.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'BUY',
                    'price': price,
                    'quantity': quantity,
                    'remaining_cash': cash
                })
                print(f"\033[92m● Buying {ticker} at ${price:.2f}\033[0m")

        elif signal == 'SELL' and ticker in portfolio:
            quantity = portfolio.get(ticker, 0)
            if quantity > 0:
                cash += quantity * price
                trade_log.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'price': price,
                    'quantity': quantity,
                    'remaining_cash': cash
                })
                del portfolio[ticker]
                print(f"\033[91m● Selling {ticker} at ${price:.2f}\033[0m")

    # Calculate portfolio value
    portfolio_value = cash
    for ticker, quantity in portfolio.items():
        price_row = prices_df[(prices_df['date'] == date) & (prices_df['ticker'] == ticker)]
        if not price_row.empty:
            current_price = price_row.iloc[0]['close']
            portfolio_value += quantity * current_price
    portfolio_values.append({'date': date, 'value': portfolio_value})

# Save trade log
os.makedirs('data', exist_ok=True)
pd.DataFrame(trade_log).to_csv(TRADE_LOG_PATH, index=False)

# Save and plot portfolio value over time
portfolio_df = pd.DataFrame(portfolio_values)
portfolio_df.dropna(subset=['date', 'value'], inplace=True)
portfolio_df.to_csv(PORTFOLIO_VALUE_PATH, index=False)

# Metrics
final_value = portfolio_df['value'].iloc[-1]
returns = ((final_value - initial_cash) / initial_cash) * 100
drawdowns = portfolio_df['value'].cummax() - portfolio_df['value']
max_drawdown = drawdowns.max()

# Win rate
num_trades = len(trade_log)
num_wins = 0
for i, trade in enumerate(trade_log):
    if trade['action'] == 'SELL':
        buy_trade = next(
            (t for t in reversed(trade_log[:i]) if t['ticker'] == trade['ticker'] and t['action'] == 'BUY'), 
            None
        )
        if buy_trade and trade['price'] > buy_trade['price']:
            num_wins += 1
win_rate = (num_wins / (num_trades / 2)) * 100 if num_trades > 0 else 0

# Print results
print(f"\nInitial Cash: ${initial_cash:,.2f}")
print(f"Final Portfolio Value: ${final_value:,.2f}")
print(f"Return: {returns:.2f}%")
print(f"Max Drawdown: ${max_drawdown:.2f}")
print(f"Total Trades: {num_trades}")
print(f"Winning Trades: {num_wins}")
print(f"Win Rate: {win_rate:.2f}%")

# Plot portfolio value
plt.figure(figsize=(12, 6))
plt.plot(portfolio_df['date'], portfolio_df['value'], label='Portfolio Value', color='blue')
plt.title('Backtest Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

