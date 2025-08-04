# view_live_trades.py

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from alpaca_trade_api.rest import REST
from datetime import datetime

load_dotenv()

st.set_page_config(page_title="TRITON Live Trades", layout="wide")

# 🔐 Alpaca keys
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://api.alpaca.markets"
api = REST(API_KEY, API_SECRET, BASE_URL)

# Files
TRADE_LOG_FILE = "data/results/live_trade_log.csv"
SIGNALS_FILE = "data/results/signals_with_rationale.csv"

st.title("🚀 TRITON Live Trading Dashboard")

# --- Live Trades Table ---
st.subheader("📋 Today's Executed Trades")

if os.path.exists(TRADE_LOG_FILE):
    df_trades = pd.read_csv(TRADE_LOG_FILE, parse_dates=["timestamp"])
    today = pd.to_datetime(datetime.utcnow().date())
    today_trades = df_trades[df_trades["timestamp"].dt.date == today.date()]
    st.dataframe(today_trades.sort_values(by="timestamp", ascending=False), use_container_width=True)
else:
    st.warning("No live trades found.")

# --- Alpaca Account Snapshot ---
st.subheader("💼 Alpaca Account Overview")
try:
    acct = api.get_account()
    st.metric("🪙 Equity", f"${acct.equity}")
    st.metric("💵 Cash", f"${acct.cash}")
    st.metric("📈 Portfolio Value", f"${acct.portfolio_value}")
except Exception as e:
    st.error(f"Failed to load Alpaca account: {e}")

# --- Current Positions ---
st.subheader("📊 Current Open Positions")
try:
    positions = api.list_positions()
    if positions:
        df_pos = pd.DataFrame([{
            "Ticker": p.symbol,
            "Qty": p.qty,
            "Market Value": p.market_value,
            "Avg Entry": p.avg_entry_price,
            "Unrealized PnL": p.unrealized_pl,
            "Side": "LONG" if float(p.qty) > 0 else "SHORT"
        } for p in positions])
        st.dataframe(df_pos, use_container_width=True)
    else:
        st.info("No open positions.")
except Exception as e:
    st.error(f"Error loading positions: {e}")

# --- Signal Summary ---
st.subheader("🧠 Today's AI Signals Summary")
try:
    df_signals = pd.read_csv(SIGNALS_FILE, parse_dates=["date"], dtype={"ticker": str})
    df_signals["rationale"] = df_signals["rationale"].fillna("N/A").astype(str)
    today = pd.to_datetime(datetime.utcnow().date())
    df_signals_today = df_signals[df_signals["date"] == today]
    df_valid = df_signals_today[df_signals_today["signal"].isin(["BUY", "SELL"])]

    if df_valid.empty:
        st.warning("No signals for today. Showing most recent valid signals instead.")
        df_valid = df_signals[df_signals["signal"].isin(["BUY", "SELL"])].sort_values("date", ascending=False).head(10)

    st.write(f"Total Signals Today: {len(df_signals_today)}")
    st.write(f"Qualifying Trades (Buy/Sell): {len(df_valid)}")
    st.dataframe(df_valid[["ticker", "signal", "confidence", "rationale"]], use_container_width=True)
except Exception as e:
    st.warning(f"Unable to load today's signal summary: {e}")
