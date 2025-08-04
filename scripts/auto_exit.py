import os
import pandas as pd
from datetime import datetime
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv

load_dotenv()

# 🔐 API credentials
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://api.alpaca.markets"

SIGNALS_FILE = "data/results/signals_with_rationale.csv"
EXIT_LOG_FILE = "data/results/live_exit_log.csv"
CONFIDENCE_THRESHOLD = 0.5  # Minimum to trust signal

api = REST(API_KEY, API_SECRET, BASE_URL)

def load_sell_signals():
    print("📉 Loading SELL signals...")
    df = pd.read_csv(SIGNALS_FILE, parse_dates=["date"])
    
    # Fallback if 'confidence' is missing
    if "confidence" not in df.columns:
        print("⚠️ No 'confidence' column found. Generating from price delta...")
        df["confidence"] = abs(df["predicted_close"] - df["close"]) / df["close"]

    today = pd.to_datetime(datetime.utcnow().date())
    df_today = df[df["date"] == today]

    sell_signals = df_today[
        (df_today["signal"] == "SELL") &
        (df_today["confidence"] >= CONFIDENCE_THRESHOLD)
    ]
    return sell_signals.set_index("ticker")

def get_open_positions():
    positions = api.list_positions()
    return {p.symbol: int(float(p.qty)) for p in positions if float(p.qty) > 0}

def submit_exit(ticker, qty, rationale, confidence):
    try:
        print(f"💼 Closing {qty} shares of {ticker} | 🧠 {rationale} (conf: {confidence})")
        order = api.submit_order(
            symbol=ticker,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="gtc"
        )
        return "submitted"
    except Exception as e:
        print(f"❌ Exit failed for {ticker}: {e}")
        return "failed"

def log_exit(ticker, qty, rationale, confidence, status):
    timestamp = datetime.utcnow().isoformat()
    row = {
        "timestamp": timestamp,
        "ticker": ticker,
        "qty": qty,
        "rationale": rationale,
        "confidence": confidence,
        "status": status
    }
    df_log = pd.DataFrame([row])
    if not os.path.exists(EXIT_LOG_FILE):
        df_log.to_csv(EXIT_LOG_FILE, index=False)
    else:
        df_log.to_csv(EXIT_LOG_FILE, mode="a", header=False, index=False)

def main():
    sell_signals = load_sell_signals()
    open_positions = get_open_positions()

    if not open_positions:
        print("📭 No open positions to monitor.")
        return

    for ticker, qty in open_positions.items():
        if ticker in sell_signals.index:
            row = sell_signals.loc[ticker]
            rationale = row.get("rationale", "AI SELL signal")
            confidence = row.get("confidence", 1.0)

            status = submit_exit(ticker, qty, rationale, confidence)
            log_exit(ticker, qty, rationale, confidence, status)
        else:
            print(f"⏳ No SELL signal for {ticker}, holding position...")

if __name__ == "__main__":
    main()
