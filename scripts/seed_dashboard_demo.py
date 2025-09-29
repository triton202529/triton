# scripts/seed_dashboard_demo.py
from pathlib import Path
import pandas as pd, numpy as np
from pandas.tseries.offsets import BDay

RESULTS = Path("data/results"); ORDERS = Path("data/orders")
RESULTS.mkdir(parents=True, exist_ok=True); ORDERS.mkdir(parents=True, exist_ok=True)

today = pd.Timestamp.today().normalize()
dates = pd.date_range(end=today, periods=60, freq="B")

# 1) portfolio_history.csv
ph = pd.DataFrame({
    "date": dates,
    "total_value": np.linspace(95000, 102000, len(dates)).round(2),
    "cash": np.linspace(15000, 12000, len(dates)).round(2),
    "market_value": np.linspace(80000, 90000, len(dates)).round(2),
})
ph.to_csv(RESULTS / "portfolio_history.csv", index=False)

# 2) executed_trades.csv (dashboard auto-detects this in place of trade_log.csv)
trades = pd.DataFrame([
    {"date": dates[-5], "ticker": "AAPL", "action": "BUY",  "price": 185.50, "quantity": 10, "profit": 0},
    {"date": dates[-4], "ticker": "AAPL", "action": "SELL", "price": 188.00, "quantity": 10, "profit": 25.0},
    {"date": dates[-3], "ticker": "MSFT", "action": "BUY",  "price": 405.00, "quantity": 5,  "profit": 0},
    {"date": dates[-2], "ticker": "MSFT", "action": "SELL", "price": 410.00, "quantity": 5,  "profit": 25.0},
])
trades.to_csv(RESULTS / "executed_trades.csv", index=False)

# 3) strategy_vs_market.csv
svm = pd.DataFrame({
    "date": dates,
    "ticker": "AAPL",
    "cumulative_strategy": (np.cumprod(1 + np.random.uniform(-0.01, 0.01, len(dates))) - 1),
    "cumulative_market":   (np.cumprod(1 + np.random.uniform(-0.008, 0.008, len(dates))) - 1),
})
svm.to_csv(RESULTS / "strategy_vs_market.csv", index=False)

# 4) signals_with_rationale.csv
sig_dates = pd.date_range(end=today, periods=30, freq="B")
signals = (["BUY","HOLD","SELL","BUY"] * 8)[:len(sig_dates)]
sig = pd.DataFrame({
    "date": sig_dates,
    "ticker": "AAPL",
    "close": np.linspace(175, 190, len(sig_dates)).round(2),
    "predicted_close": np.linspace(176, 192, len(sig_dates)).round(2),
    "signal": signals,
    "confidence": np.linspace(0.3, 0.9, len(sig_dates)).round(3),
    "rationale": ["Example rationale"] * len(sig_dates),
})
sig.to_csv(RESULTS / "signals_with_rationale.csv", index=False)

# 5) news_sentiment.csv
news = pd.DataFrame({
    "date": [dates[-6], dates[-3]],
    "ticker": ["AAPL", "MSFT"],
    "title": ["Apple launches new device", "Microsoft announces earnings"],
    "url": ["https://example.com/a", "https://example.com/m"],
    "sentiment": [0.4, 0.2],
})
news.to_csv(RESULTS / "news_sentiment.csv", index=False)

# 6) orders_today.csv
orders_today = pd.DataFrame({
    "ticker": ["AAPL", "MSFT"],
    "action": ["BUY", "SELL"],
    "target_weight": [0.10, -0.05],
})
orders_today.to_csv(ORDERS / "orders_today.csv", index=False)

# 7) buffett_orders.csv
bo = pd.DataFrame({
    "ticker": ["AAPL", "MSFT"],
    "action": ["BUY", "SELL"],
    "delta_notional": [20000, -15000],
    "target_weight": [0.12, -0.07],
    "buffett_score": [0.82, 0.45],
})
bo.to_csv(ORDERS / "buffett_orders.csv", index=False)

print("Seeded demo files in:", RESULTS.resolve())
