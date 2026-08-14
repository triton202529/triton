import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame
from dotenv import load_dotenv

load_dotenv()


def _deprecation_guard() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from services.execution_path_guard import exit_if_deprecated

    exit_if_deprecated(__file__)


# 🔐 API keys (live)
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://api.alpaca.markets"

SIGNALS_FILE = "data/results/signals_with_rationale.csv"
CONFIDENCE_THRESHOLD = 0.7
TRADE_LOG = "data/results/live_trade_log.csv"

api = REST(API_KEY, API_SECRET, BASE_URL)


def load_latest_signals():
    print("📈 Loading AI signals...")
    df = pd.read_csv(SIGNALS_FILE, parse_dates=["date"])

    # Add confidence if missing
    if "confidence" not in df.columns:
        print("⚠️ No 'confidence' column found. Generating it from price delta...")
        df["confidence"] = abs(df["predicted_close"] - df["close"]) / df["close"]
        df["confidence"] = df["confidence"].clip(0, 1)

    today = pd.to_datetime(datetime.utcnow().date())
    df_today = df[df["date"] == today]

    df_today = df_today[
        (df_today["signal"].isin(["BUY", "SELL"]))
        & (df_today["confidence"] >= CONFIDENCE_THRESHOLD)
    ]
    return df_today


def submit_order(ticker, side, rationale, confidence):
    try:
        qty = 1
        print(f"🛒 {side} {qty} shares of {ticker} | 🧠 {rationale} (conf: {confidence:.2f})")
        from services.paper_sell_raw_path_gate import gate_raw_paper_sell

        gate_raw_paper_sell(
            side=side.lower(),
            symbol=ticker,
            qty=qty,
            source="scripts.auto_trade.submit_order",
        )
        order = api.submit_order(
            symbol=ticker,
            qty=qty,
            side=side.lower(),
            type="market",
            time_in_force="gtc",
        )
        return order
    except Exception as e:
        print(f"❌ Trade failed for {ticker}: {e}")
        return None


def log_trade(ticker, side, rationale, confidence, status):
    timestamp = datetime.utcnow().isoformat()
    row = {
        "timestamp": timestamp,
        "ticker": ticker,
        "action": side,
        "rationale": rationale,
        "confidence": confidence,
        "status": status,
    }
    df_log = pd.DataFrame([row])
    if not os.path.exists(TRADE_LOG):
        df_log.to_csv(TRADE_LOG, index=False)
    else:
        df_log.to_csv(TRADE_LOG, mode="a", header=False, index=False)


def main():
    signals = load_latest_signals()
    if signals.empty:
        print("🚫 No qualifying signals for today.")
        return

    for _, row in signals.iterrows():
        ticker = row["ticker"]
        action = row["signal"]
        rationale = row.get("rationale", "No rationale provided")
        confidence = row.get("confidence", 1.0)

        order = submit_order(ticker, action, rationale, confidence)
        status = "submitted" if order else "failed"
        log_trade(ticker, action, rationale, confidence, status)


if __name__ == "__main__":
    _deprecation_guard()
    main()
