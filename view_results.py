# view_results.py

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Triton AI Unified Dashboard", layout="wide")
st.title("📊 Triton AI Unified Dashboard")

RESULTS_DIR = "data/results"

def load_csv(filename):
    try:
        path = os.path.join(RESULTS_DIR, filename)
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"❌ Could not load {filename}: {e}")
        return pd.DataFrame()

tabs = st.tabs([
    "📈 Portfolio History", "📋 Trade Log", "📊 Strategy vs Market", "🧠 AI Signals",
    "📁 Raw CSV", "📋 Backtest Summary", "📉 Risk Report", "📊 Strategy Diagnostics",
    "🏦 Portfolio Allocations", "📽️ Trade Replay", "📘 Fundamentals", "📈 Stock Scores",
    "🎯 Top Picks", "📰 News Sentiment", "🚨 Smart Alerts", "📆 Economic Calendar",
    "🔬 Feature Importance", "🎯 SL/TP Performance", "📈 Model Comparison", "💬 Sentiment + Signal Fusion",
    "🧠 AI Learning Lab"
])

# --- Tabs 0 to 19 omitted here for brevity; you've already got them and they haven't changed --- #

# Tab 21 — AI Learning Lab (Strategy Selector Included)
with tabs[20]:
    st.subheader("🧠 AI Learning Lab")

    st.markdown("""
    Welcome to TRITON's experimental sandbox.

    - 📁 Upload custom stock data (CSV)
    - 🧠 Prototype new strategies:
        - Moving Average Crossover
        - RSI Oversold/Overbought
        - Bollinger Band Breakout
    - 📊 Run signal tests and visualize results
    - 🔬 Use this tab to build and refine TRITON’s intelligence
    """)

    uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["date"])
            st.success("✅ File loaded successfully!")
            st.dataframe(df.head())

            if 'close' not in df.columns or 'date' not in df.columns:
                st.error("❌ CSV must include 'date' and 'close' columns.")
            else:
                df = df.sort_values("date")
                strategy = st.selectbox("🧠 Choose a Strategy", ["Moving Average Crossover", "RSI Strategy", "Bollinger Bands"])

                if strategy == "Moving Average Crossover":
                    df["ma5"] = df["close"].rolling(window=5).mean()
                    df["ma20"] = df["close"].rolling(window=20).mean()
                    df["signal"] = (df["ma5"] > df["ma20"]).astype(int).diff().fillna(0)

                elif strategy == "RSI Strategy":
                    delta = df["close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    df["rsi"] = 100 - (100 / (1 + rs))
                    df["signal"] = 0
                    df.loc[df["rsi"] < 30, "signal"] = 1
                    df.loc[df["rsi"] > 70, "signal"] = -1
                    df["signal"] = df["signal"].diff().fillna(0)

                elif strategy == "Bollinger Bands":
                    ma20 = df["close"].rolling(window=20).mean()
                    std20 = df["close"].rolling(window=20).std()
                    df["upper"] = ma20 + (2 * std20)
                    df["lower"] = ma20 - (2 * std20)
                    df["signal"] = 0
                    df.loc[df["close"] < df["lower"], "signal"] = 1
                    df.loc[df["close"] > df["upper"], "signal"] = -1
                    df["signal"] = df["signal"].diff().fillna(0)

                df["strategy_return"] = df["close"].pct_change().fillna(0) * df["signal"].shift(1).fillna(0)
                df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()

                st.subheader(f"📈 Strategy Equity Curve ({strategy})")
                fig, ax = plt.subplots()
                ax.plot(df["date"], df["cumulative_return"], label="Strategy", linewidth=2)
                ax.set_xlabel("Date")
                ax.set_ylabel("Cumulative Return")
                ax.legend()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
