import os
import pandas as pd
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv
from datetime import datetime, timezone
from risk_control import risk_check  # ✅ Local import (no scripts. prefix)

# Load environment variables
load_dotenv()

# Alpaca credentials
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

# Connect to Alpaca
api = REST(API_KEY, API_SECRET, BASE_URL)

# Config
SIGNALS_PATH = "data/predictions/signals.csv"
LOG_PATH = "data/results/executed_trades.csv"
TRADE_PCT = 0.05  # Max 5% of buying power per trade
AUTO_TRADE = True

print("🚀 Starting trade execution..." if AUTO_TRADE else "🚀 Starting trade simulation...")

# Load signals
df = pd.read_csv(SIGNALS_PATH)
if "ticker" not in df.columns or "signal" not in df.columns:
    raise ValueError("❌ 'ticker' and 'signal' columns are required in the signals file.")

latest_signals = df.groupby("ticker").last().reset_index()

# Ensure trade log file exists
if not os.path.exists(LOG_PATH):
    pd.DataFrame(columns=["timestamp", "ticker", "action", "quantity", "price", "status", "note"]).to_csv(LOG_PATH, index=False)

logs = []

# Get account balance
account = api.get_account()
buying_power = float(account.buying_power)
print(f"💰 Buying power: ${buying_power:,.2f}")

for _, row in latest_signals.iterrows():
    ticker = row["ticker"]
    signal = row["signal"].upper()
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"📊 {ticker}: Signal = {signal}")

    try:
        if signal not in ["BUY", "SELL"]:
            print(f"⏸️ HOLD for {ticker} — no action")
            logs.append([timestamp, ticker, "HOLD", 0, None, "SKIPPED", "No action"])
            continue

        if not risk_check(ticker, signal, api):
            print(f"⚠️ Trade blocked by risk control: {ticker} {signal}")
            logs.append([timestamp, ticker, signal, 0, None, "BLOCKED", "Blocked by risk control"])
            continue

        if signal == "BUY":
            last_price = float(api.get_latest_trade(ticker).price)
            max_allocation = TRADE_PCT * buying_power
            qty = int(max_allocation // last_price)

            if qty <= 0:
                logs.append([timestamp, ticker, "BUY", 0, last_price, "SKIPPED", "Insufficient buying power"])
                print(f"❌ Not enough to buy even 1 share of {ticker}")
                continue

            if AUTO_TRADE:
                order = api.submit_order(symbol=ticker, qty=qty, side="buy", type="market", time_in_force="gtc")
                logs.append([timestamp, ticker, "BUY", qty, last_price, "EXECUTED", f"Order ID: {order.id}"])
                print(f"✅ Bought {qty} x {ticker} at ~${last_price}")
            else:
                logs.append([timestamp, ticker, "BUY", qty, last_price, "SIMULATED", "Would BUY"])
                print(f"📝 Simulated BUY: {qty} x {ticker} at ~${last_price}")

        elif signal == "SELL":
            try:
                position = api.get_position(ticker)
                qty = int(float(position.qty))
            except:
                qty = 0

            if qty > 0:
                if AUTO_TRADE:
                    order = api.submit_order(symbol=ticker, qty=qty, side="sell", type="market", time_in_force="gtc")
                    logs.append([timestamp, ticker, "SELL", qty, None, "EXECUTED", f"Order ID: {order.id}"])
                    print(f"✅ Sold {qty} x {ticker}")
                else:
                    logs.append([timestamp, ticker, "SELL", qty, None, "SIMULATED", "Would SELL"])
                    print(f"📝 Simulated SELL: {qty} x {ticker}")
            else:
                logs.append([timestamp, ticker, "SELL", 0, None, "SKIPPED", "No position"])
                print(f"⏸️ No position to SELL in {ticker}")

    except Exception as e:
        logs.append([timestamp, ticker, signal, 0, None, "FAILED", str(e)])
        print(f"❌ Error processing {ticker}: {str(e)}")

# Save trade log
log_df = pd.DataFrame(logs, columns=["timestamp", "ticker", "action", "quantity", "price", "status", "note"])
log_df.to_csv(LOG_PATH, mode="a", header=False, index=False)
print("📄 Trade log saved.")
