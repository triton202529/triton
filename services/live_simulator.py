# services/live_simulator.py

import os
import pandas as pd
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL")

alpaca = REST(API_KEY, API_SECRET, BASE_URL)

print("⚙️ Loading signals...")
signals_df = pd.read_csv("data/predictions/signals.csv")

# ✅ FIXED: Use 'ticker' instead of 'symbol'
unique_symbols = signals_df["ticker"].unique()

print(f"📈 Running simulation for {len(unique_symbols)} tickers...")

for symbol in unique_symbols:
    symbol_df = signals_df[signals_df["ticker"] == symbol]

    latest_signal = symbol_df.sort_values("date", ascending=False).iloc[0]["signal"]
    latest_price = symbol_df.sort_values("date", ascending=False).iloc[0]["close"]

    print(f"\n📊 {symbol}: Last Signal = {latest_signal} @ ${latest_price:.2f}")

    if latest_signal == "BUY":
        print(f"🟢 Would BUY {symbol} at ${latest_price}")
    elif latest_signal == "SELL":
        print(f"🔴 Would SELL {symbol} at ${latest_price}")
    else:
        print(f"⏸️ HOLDING {symbol} - No action")
