thats my version       import os
import pandas as pd
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

# Load Alpaca credentials
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL")

# Connect to Alpaca
api = REST(API_KEY, API_SECRET, BASE_URL)

# Configuration
SIGNALS_PATH = "data/predictions/signals.csv"
LOG_PATH = "data/results/executed_trades.csv"
AUTO_TRADE = False  # 🚫 Simulation mode by default

print("🚀 Starting trade simulation...")

# Load signals
df = pd.read_csv(SIGNALS_PATH)

if 'ticker' not in df.columns or 'signal' not in df.columns:
    raise ValueError("❌ 'ticker' and 'signal' columns are required in the signals file.")

# Filter latest signal per ticker
latest_signals = df.groupby("ticker").last().reset_index()

# Prepare log file if it doesn't exist
if not os.path.exists(LOG_PATH):
    pd.DataFrame(columns=["timestamp", "ticker", "action", "quantity", "price", "status", "note"]).to_csv(LOG_PATH, index=False)

logs = []

# Process each signal
for _, row in latest_signals.iterrows():
    ticker = row["ticker"]
    signal = row["signal"].upper()
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"📊 {ticker}: Signal = {signal}")

    try:
        if signal == "BUY":
            logs.append([timestamp, ticker, "BUY", 1, None, "SIMULATED", "Would BUY"])
            print(f"📝 Would BUY {ticker}")

        elif signal == "SELL":
            try:
                position = api.get_position(ticker)
                qty = int(float(position.qty))
            except:
                qty = 0

            if qty > 0:
                logs.append([timestamp, ticker, "SELL", qty, None, "SIMULATED", "Would SELL"])
                print(f"📝 Would SELL {ticker}")
            else:
                logs.append([timestamp, ticker, "SELL", 0, None, "SKIPPED", "No position"])
                print(f"⏸️ No position to SELL in {ticker}")

        else:
            logs.append([timestamp, ticker, "HOLD", 0, None, "SKIPPED", "No action"])
            print(f"⏸️ No action for {ticker} (Signal: {signal})")

    except Exception as e:
        logs.append([timestamp, ticker, signal, 0, None, "FAILED", str(e)])
        print(f"❌ Error processing {ticker}: {str(e)}")

# Save to log
log_df = pd.DataFrame(logs, columns=["timestamp", "ticker", "action", "quantity", "price", "status", "note"])
log_df.to_csv(LOG_PATH, mode='a', header=False, index=False)

print("📄 Simulated trade log saved.")
