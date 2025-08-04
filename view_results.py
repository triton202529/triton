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
    "🔬 Feature Importance", "🎯 SL/TP Performance", "📈 Model Comparison", "💬 Sentiment + Signal Fusion"
])

# Tab 1
with tabs[0]:
    st.subheader("📈 Portfolio History")
    df = load_csv("portfolio_history.csv")
    if not df.empty and "date" in df and "total_value" in df:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        fig = px.line(df, x="date", y="total_value", title="Portfolio Value Over Time")
        fig.update_traces(line=dict(color="blue"))
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing 'date' or 'total_value' in portfolio_history.csv")

# Tab 2
with tabs[1]:
    st.subheader("📋 Trade Log")
    df = load_csv("trade_log.csv")
    if not df.empty:
        st.dataframe(df)

# Tab 3
with tabs[2]:
    st.subheader("📊 Strategy vs Market")
    df = load_csv("strategy_vs_market.csv")
    if not df.empty and {"date", "cumulative_strategy", "cumulative_market"}.issubset(df.columns):
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        st.line_chart(df.set_index("date")[["cumulative_strategy", "cumulative_market"]])
    else:
        st.warning("Required columns not found in strategy_vs_market.csv")

# Tab 4
with tabs[3]:
    st.subheader("🧠 AI Signals")
    df = load_csv("signals_with_rationale.csv")
    if not df.empty:
        st.dataframe(df)

# Tab 5
with tabs[4]:
    st.subheader("📁 Raw CSV")
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
    selected = st.selectbox("Select CSV to preview", files)
    df = load_csv(selected)
    if not df.empty:
        st.dataframe(df)

# Tab 6
with tabs[5]:
    st.subheader("📋 Backtest Summary")
    df = load_csv("backtest_summary.csv")
    if not df.empty:
        st.dataframe(df)

# Tab 7
with tabs[6]:
    st.subheader("📉 Risk Report")
    df = load_csv("risk_report.csv")
    if not df.empty:
        st.dataframe(df)

# Tab 8
with tabs[7]:
    st.subheader("📊 Strategy Diagnostics")
    df = load_csv("trade_log.csv")
    if not df.empty:
        required = {"quantity", "exit_price", "entry_price", "signal", "ticker"}
        missing = required - set(df.columns)

        if missing:
            st.warning(f"⚠️ Missing required columns: {missing}")
        else:
            df["profit"] = df["quantity"] * (df["exit_price"] - df["entry_price"])
            df["win"] = df["profit"] > 0

            st.metric("✅ Win Rate", f"{100 * df['win'].mean():.2f}%")
            st.metric("💰 Avg Profit", f"${df['profit'].mean():.2f}")
            st.metric("📈 Total Trades", len(df))

            fig, ax = plt.subplots()
            df["signal"].value_counts().plot(kind="bar", ax=ax, title="Signal Distribution")
            st.pyplot(fig)

            st.markdown("### 📊 Profit Distribution by Ticker")
            ticker_profit = df.groupby("ticker")["profit"].sum().sort_values(ascending=False)
            st.bar_chart(ticker_profit)

            st.markdown("### 📊 Win/Loss Count by Ticker")
            win_loss = df.groupby("ticker")["win"].value_counts().unstack(fill_value=0)
            st.dataframe(win_loss)

# Tab 9
with tabs[8]:
    st.subheader("🏦 Portfolio Allocations")
    df = load_csv("trade_log.csv")
    if not df.empty and "action" in df and "quantity" in df and "ticker" in df:
        latest = df[df["action"] == "BUY"].groupby("ticker")["quantity"].sum()
        fig = go.Figure(data=[go.Pie(labels=latest.index, values=latest.values)])
        st.plotly_chart(fig)
        st.dataframe(latest.reset_index())
    else:
        st.warning("Missing necessary fields in trade_log.csv")

# Tab 10
with tabs[9]:
    st.subheader("📽️ Trade Replay")
    df = load_csv("trade_log.csv")
    if not df.empty and "ticker" in df.columns:
        for ticker in df["ticker"].unique():
            st.markdown(f"### {ticker}")
            st.dataframe(df[df["ticker"] == ticker])

# Tab 11
with tabs[10]:
    st.subheader("📘 Fundamentals")
    df = load_csv("fundamentals.csv")
    if not df.empty:
        st.dataframe(df)

# Tab 12
with tabs[11]:
    st.subheader("📈 Stock Scores")
    df = load_csv("stock_scores.csv")
    if not df.empty and "total_score" in df.columns:
        st.dataframe(df.sort_values("total_score", ascending=False))

# Tab 13
with tabs[12]:
    st.subheader("🎯 Top Picks")
    scores = load_csv("stock_scores.csv")
    signals = load_csv("signals_with_rationale.csv")

    if not scores.empty:
        top = scores.sort_values("total_score", ascending=False).head(10)
        st.markdown("### 🔝 Top 10 by Score")
        st.dataframe(top)

        if not signals.empty and "date" in signals.columns and "ticker" in signals.columns:
            recent = signals[signals["date"] == signals["date"].max()]
            if "ticker" in top.columns and "ticker" in recent.columns:
                merged = pd.merge(top, recent, on="ticker", how="left")
                st.markdown("### 🎯 AI BUY Picks")
                st.dataframe(merged[merged["signal"] == "BUY"])

# Tab 14
with tabs[13]:
    st.subheader("📰 News Sentiment")
    df = load_csv("news_sentiment.csv")
    if not df.empty:
        st.dataframe(df)

# Tab 15
with tabs[14]:
    st.subheader("🚨 Smart Alerts")
    df = load_csv("alerts.csv")
    if not df.empty:
        st.dataframe(df)

# Tab 16
with tabs[15]:
    st.subheader("📆 Economic Calendar")
    df = load_csv("economic_calendar.csv")
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        st.dataframe(df)

        today = pd.Timestamp.today().normalize()
        upcoming = df[df["date"] >= today]
        st.markdown("### 🕒 Upcoming Events")
        st.dataframe(upcoming)

        countries = sorted(df["country"].dropna().unique())
        selected_countries = st.multiselect("🌍 Filter by Country", countries, default=countries)

        importance_levels = sorted(df["importance"].dropna().unique())
        selected_levels = st.multiselect("⚠️ Filter by Importance", importance_levels, default=importance_levels)

        filtered = df[df["country"].isin(selected_countries) & df["importance"].isin(selected_levels)]
        st.markdown(f"### 📌 Filtered Events ({len(filtered)} results)")
        st.dataframe(filtered)
        st.download_button("📥 Download Filtered Calendar", filtered.to_csv(index=False), file_name="filtered_calendar.csv")

# Tab 17
with tabs[16]:
    st.subheader("🔬 Feature Importance")
    df = load_csv("feature_importance.csv")
    if not df.empty and {"ticker", "feature", "importance"}.issubset(df.columns):
        tickers = df["ticker"].unique()
        selected_ticker = st.selectbox("Select Ticker", sorted(tickers))
        filtered = df[df["ticker"] == selected_ticker].sort_values("importance", ascending=False)
        st.bar_chart(filtered.set_index("feature")["importance"])
        st.dataframe(filtered)
    else:
        st.warning("Missing data or columns in feature_importance.csv")

# Tab 18 — SL/TP Performance
with tabs[17]:
    st.subheader("🎯 SL/TP Performance Analysis")

    df = load_csv("trade_log.csv")
    if not df.empty and {"stop_loss", "take_profit", "exit_price", "entry_price", "ticker"}.issubset(df.columns):
        df["hit_sl"] = df["exit_price"] <= df["stop_loss"]
        df["hit_tp"] = df["exit_price"] >= df["take_profit"]
        df["reached"] = df.apply(
            lambda row: "TP" if row["hit_tp"] else "SL" if row["hit_sl"] else "None", axis=1
        )

        st.metric("🎯 Total Trades with SL/TP", len(df))
        st.metric("🟥 Stop Loss Hits", df["hit_sl"].sum())
        st.metric("🟩 Take Profit Hits", df["hit_tp"].sum())

        st.markdown("### 📊 Reached Target Breakdown")
        reached_counts = df["reached"].value_counts()
        st.bar_chart(reached_counts)

        st.markdown("### 📈 SL/TP Hits by Ticker")
        ticker_hits = df.groupby("ticker")[["hit_sl", "hit_tp"]].sum()
        st.dataframe(ticker_hits)

        st.markdown("### 📉 SL/TP Hit Rate")
        ticker_hits["total"] = ticker_hits["hit_sl"] + ticker_hits["hit_tp"]
        ticker_hits["tp_rate"] = (ticker_hits["hit_tp"] / ticker_hits["total"]).fillna(0)
        ticker_hits["sl_rate"] = (ticker_hits["hit_sl"] / ticker_hits["total"]).fillna(0)
        st.dataframe(ticker_hits[["tp_rate", "sl_rate"]])

        fig = go.Figure()
        fig.add_trace(go.Bar(x=ticker_hits.index, y=ticker_hits["tp_rate"], name="TP Rate", marker_color="green"))
        fig.add_trace(go.Bar(x=ticker_hits.index, y=ticker_hits["sl_rate"], name="SL Rate", marker_color="red"))
        fig.update_layout(barmode="group", title="SL/TP Hit Rates by Ticker")
        st.plotly_chart(fig)

# Tab 19 - Model Comparison
with tabs[18]:
    st.subheader("📈 Model Comparison")

    df = load_csv("model_comparison.csv")
    if not df.empty:
        st.markdown("### 📋 Raw Model Comparison Table")
        st.dataframe(df)

        if {"ticker", "model", "rmse"}.issubset(df.columns):
            st.markdown("### 📊 RMSE by Model and Ticker")
            fig = px.bar(df, x="ticker", y="rmse", color="model", barmode="group", title="Model RMSE by Ticker")
            st.plotly_chart(fig, use_container_width=True)

            selected_ticker = st.selectbox("🔍 Select Ticker for Comparison", df["ticker"].unique())
            filtered = df[df["ticker"] == selected_ticker]
            st.dataframe(filtered)
        else:
            st.warning("Missing columns in model_comparison.csv: expected 'ticker', 'model', 'rmse'")
    else:
        st.warning("model_comparison.csv not found or empty.")

# Tab 20 - Sentiment + Signal Fusion
with tabs[19]:
    st.subheader("💬 Sentiment + Signal Fusion")

    signals = load_csv("signals_with_rationale.csv")
    sentiment = load_csv("news_sentiment.csv")

    if not signals.empty and not sentiment.empty:
        if "date" in signals.columns and "ticker" in signals.columns and \
           "ticker" in sentiment.columns and "publishedAt" in sentiment.columns and \
           "sentiment" in sentiment.columns:

            sentiment["publishedAt"] = pd.to_datetime(sentiment["publishedAt"])
            sentiment["date"] = sentiment["publishedAt"].dt.date
            signals["date"] = pd.to_datetime(signals["date"]).dt.date

            merged = pd.merge(signals, sentiment, on=["ticker", "date"], how="left")

            st.markdown("### 🧠 Signals Fused with Sentiment")
            st.dataframe(merged)

            st.markdown("### 📈 Avg Sentiment per Signal Type")
            if "signal" in merged.columns and "sentiment" in merged.columns:
                sentiment_by_signal = merged.groupby("signal")["sentiment"].mean().dropna()
                fig = px.bar(sentiment_by_signal, x=sentiment_by_signal.index, y=sentiment_by_signal.values,
                             labels={"x": "Signal", "y": "Avg Sentiment"},
                             title="Average Sentiment per Signal Type")
                st.plotly_chart(fig)
        else:
            st.warning("Required columns missing in input files.")
    else:
        st.warning("signals_with_rationale.csv or news_sentiment.csv not found or empty.")

# Tab 21 - AI Learning Lab (Strategy Selector Included)
with st.tabs(["🧠 AI Learning Lab"])[0]:
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