# view_results.py

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Triton AI Unified Dashboard", layout="wide")
st.title("📊 Triton AI Unified Dashboard")

RESULTS_DIR = "data/results"

# ---------- helpers ----------
def load_csv(filename):
    """Load a CSV from data/results with friendly errors."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"❌ Could not load {filename}: {e}")
        return pd.DataFrame()

def parse_dates_inplace(df, cols=("date",)):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def get_score_col(df):
    if "total_score" in df.columns:  # current pipeline
        return "total_score"
    if "score" in df.columns:        # legacy
        return "score"
    return None

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def r2_score(y_true, y_pred):
    # Safe R^2 without sklearn
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# --- news helpers ---
def strip_html(s):
    if pd.isna(s):
        return s
    return re.sub(r"<[^>]*>", "", str(s))

def extract_href(s):
    if pd.isna(s):
        return None
    m = re.search(r'href="([^"]+)"', str(s))
    return m.group(1) if m else None

def make_clickable(title, url):
    """Return HTML anchor if url is present; otherwise just title."""
    if pd.isna(url) or not str(url).strip():
        return str(title) if not pd.isna(title) else ""
    # If url is already an <a> tag, just return it
    if str(url).strip().startswith("<a "):
        return str(url)
    safe_title = str(title) if not pd.isna(title) and str(title).strip() else "Link"
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{safe_title}</a>'

# ---------- tabs ----------
tabs = st.tabs([
    "📈 Portfolio History", "📋 Trade Log", "📊 Strategy vs Market", "🧠 AI Signals",
    "📁 Raw CSV", "📋 Backtest Summary", "📉 Risk Report", "📊 Strategy Diagnostics",
    "🏦 Portfolio Allocations", "📽️ Trade Replay", "📘 Fundamentals", "📈 Stock Scores",
    "🎯 Top Picks", "📰 News Sentiment", "🚨 Smart Alerts", "📆 Economic Calendar",
    "🔬 Feature Importance", "🎯 SL/TP Performance", "💬 Sentiment + Signal Fusion",
    "📊 Model Comparison", "🧠 AI Learning Lab"
])
#                ^ index 18                      ^ index 19            ^ index 20

# Tab 0 — Portfolio History
with tabs[0]:
    st.subheader("📈 Portfolio Value Over Time")
    df = load_csv("portfolio_history.csv")
    if df.empty:
        st.info("No portfolio_history.csv yet.")
    else:
        df = parse_dates_inplace(df, ("date",))
        to_numeric(df, ["total_value", "cash", "market_value"])
        df = df.dropna(subset=["date", "total_value"]).sort_values("date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["total_value"], mode="lines", name="Total Value"))
        fig.update_layout(title="Portfolio Equity Curve", xaxis_title="Date", yaxis_title="Portfolio Value")
        st.plotly_chart(fig, use_container_width=True)

# Tab 1 — Trade Log
with tabs[1]:
    st.subheader("📋 Trade Log")
    df = load_csv("trade_log.csv")
    if df.empty:
        st.info("No trade_log.csv yet.")
    else:
        st.dataframe(df)

# Tab 2 — Strategy vs Market
with tabs[2]:
    st.subheader("📊 Strategy vs Market")
    df = load_csv("strategy_vs_market.csv")
    if df.empty:
        st.info("No strategy_vs_market.csv yet.")
    else:
        parse_dates_inplace(df, ("date",))
        to_numeric(df, ["cumulative_strategy", "cumulative_market"])
        tickers = sorted(df["ticker"].dropna().unique()) if "ticker" in df else []
        if not tickers:
            st.warning("Missing 'ticker' column.")
        else:
            selected = st.selectbox("Select a ticker", tickers)
            chart_df = df[df["ticker"] == selected].dropna(subset=["date"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart_df["date"], y=chart_df.get("cumulative_strategy"), name="Strategy"))
            fig.add_trace(go.Scatter(x=chart_df["date"], y=chart_df.get("cumulative_market"), name="Market"))
            fig.update_layout(title=f"{selected} Strategy vs Market",
                              xaxis_title="Date", yaxis_title="Cumulative Return")
            st.plotly_chart(fig, use_container_width=True)

# Tab 3 — AI Signals
with tabs[3]:
    st.subheader("🧠 AI Signals (latest 100)")
    df = load_csv("signals_with_rationale.csv")
    if df.empty:
        st.info("No signals_with_rationale.csv yet.")
    else:
        st.dataframe(df.tail(100))

# Tab 4 — Raw CSV Browser
with tabs[4]:
    st.subheader("📁 Browse Any CSV in data/results")
    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")])
    if not files:
        st.info("No CSV files found in data/results.")
    else:
        selected = st.selectbox("Select a file", files)
        df = load_csv(selected)
        st.dataframe(df)

# Tab 5 — Backtest Summary
with tabs[5]:
    st.subheader("📋 Backtest Summary")
    df = load_csv("backtest_summary.csv")
    if df.empty:
        st.info("No backtest_summary.csv yet.")
    else:
        st.dataframe(df)

# Tab 6 — Risk Report (Drawdown)
with tabs[6]:
    st.subheader("📉 Risk: Portfolio Drawdown")
    df = load_csv("portfolio_history.csv")
    if df.empty:
        st.info("No portfolio_history.csv yet.")
    else:
        parse_dates_inplace(df, ("date",))
        to_numeric(df, ["total_value"])
        df = df.dropna(subset=["date", "total_value"]).sort_values("date")
        df["peak"] = df["total_value"].cummax()
        df["drawdown"] = df["total_value"] / df["peak"] - 1
        fig = px.area(df, x="date", y="drawdown", title="Drawdown (relative to running peak)")
        st.plotly_chart(fig, use_container_width=True)

# Tab 7 — Strategy Diagnostics
with tabs[7]:
    st.subheader("📊 Strategy Diagnostics")
    df = load_csv("trade_log.csv")
    if df.empty:
        st.info("No trade_log.csv yet.")
    else:
        if "signal" in df.columns:
            counts = df["signal"].value_counts()
            fig = px.bar(x=counts.index, y=counts.values,
                         labels={"x": "Signal", "y": "Count"},
                         title="Signal Distribution")
            st.plotly_chart(fig, use_container_width=True)
        if "profit" in df.columns:
            st.write("Average Profit per Trade:",
                     round(pd.to_numeric(df["profit"], errors="coerce").mean(), 2))

# Tab 8 — Portfolio Allocations
with tabs[8]:
    st.subheader("🏦 Portfolio Allocations")
    df = load_csv("trade_log.csv")
    if df.empty:
        st.info("No trade_log.csv yet.")
    else:
        required = {"action", "quantity", "ticker"}
        if not required.issubset(df.columns):
            st.warning(f"Missing columns: {sorted(required - set(df.columns))}")
        else:
            latest = df[df["action"].str.upper() == "BUY"].groupby("ticker")["quantity"].sum()
            if latest.empty:
                st.info("No BUY records to visualize.")
            else:
                fig = px.pie(values=latest.values, names=latest.index, title="Holdings Allocation")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(latest.reset_index().rename(columns={"quantity": "shares"}))

# Tab 9 — Trade Replay
with tabs[9]:
    st.subheader("📽️ Trade Replay")
    df = load_csv("trade_log.csv")
    if df.empty:
        st.info("No trade_log.csv yet.")
    else:
        if "ticker" not in df.columns:
            st.warning("Missing 'ticker' column.")
        else:
            ticker = st.selectbox("Select ticker", sorted(df["ticker"].dropna().unique()))
            trades = df[df["ticker"] == ticker]
            cols = ["date", "action", "price", "quantity"]
            st.dataframe(trades[cols] if set(cols).issubset(trades.columns) else trades)

# Tab 10 — Fundamentals
with tabs[10]:
    st.subheader("📘 Fundamental Data")
    df = load_csv("fundamentals.csv")
    if df.empty:
        st.info("No fundamentals.csv yet.")
    else:
        st.dataframe(df)

# Tab 11 — Stock Scores
with tabs[11]:
    st.subheader("📈 Stock Scores")
    df = load_csv("stock_scores.csv")
    if df.empty:
        st.info("No stock_scores.csv yet.")
    else:
        score_col = get_score_col(df)
        if score_col:
            st.dataframe(df.sort_values(score_col, ascending=False))
        else:
            st.warning("No score column found (expected 'total_score' or 'score'). Showing raw data.")
            st.dataframe(df)

# Tab 12 — Top Picks
with tabs[12]:
    st.subheader("🎯 Top Fundamental Picks")
    df = load_csv("stock_scores.csv")
    if df.empty:
        st.info("No stock_scores.csv yet.")
    else:
        score_col = get_score_col(df)
        if score_col:
            top = df.sort_values(score_col, ascending=False).head(10)
            st.dataframe(top)
        else:
            st.warning("No score column found (expected 'total_score' or 'score'). Showing first 10 rows.")
            st.dataframe(df.head(10))

# Tab 13 — News Sentiment (clickable links)
with tabs[13]:
    st.subheader("📰 News Sentiment")
    df = load_csv("news_sentiment.csv")
    if df.empty:
        st.info("No news_sentiment.csv yet.")
    else:
        parse_dates_inplace(df, ("publishedAt", "date"))
        # Derive URL + clean description
        if "description" in df.columns and ("url" not in df.columns or df["url"].isna().all()):
            df["url"] = df["description"].apply(extract_href)
            df["description"] = df["description"].apply(strip_html)
        # Build clickable column
        title_col = "title" if "title" in df.columns else None
        url_col = "url" if "url" in df.columns else None
        if title_col or url_col:
            df["news"] = df.apply(
                lambda r: make_clickable(r.get(title_col, ""), r.get(url_col, "")),
                axis=1
            )
        show_cols = [c for c in ["date", "ticker", "sentiment", "news", "description"] if c in df.columns or c == "news"]
        disp = df[show_cols] if show_cols else df
        # Render as HTML so links are clickable
        st.markdown(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

# Tab 14 — Smart Alerts (auto-pull + filters + clickable links)
with tabs[14]:
    st.subheader("🚨 Smart Alerts")

    df = load_csv("alerts.csv")  # auto-pull from data/results/alerts.csv
    if df.empty:
        df = load_csv("smart_alerts.csv")  # legacy fallback

    if df.empty:
        st.info("No alerts CSV found.")
    else:
        parse_dates_inplace(df, ("date", "timestamp"))

        if "priority" in df.columns:
            pri_order = ["LOW", "MEDIUM", "HIGH"]
            df["priority"] = pd.Categorical(df["priority"], categories=pri_order, ordered=True)

        col_l, col_r = st.columns([3, 2])
        with col_l:
            min_pri = st.selectbox("Minimum priority", options=["LOW", "MEDIUM", "HIGH"], index=1)
            tickers = sorted(df["ticker"].dropna().unique()) if "ticker" in df.columns else []
            sel_tickers = st.multiselect("Tickers", tickers, default=[])
        with col_r:
            days_back = st.slider("Show last N days", min_value=3, max_value=60, value=30, step=1)

        # Apply filters
        f = df.copy()
        if "priority" in f.columns:
            pri_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            min_rank = pri_rank[min_pri]
            f = f[f["priority"].map(pri_rank).fillna(0) >= min_rank]

        if sel_tickers:
            f = f[f["ticker"].isin(sel_tickers)]

        if "date" in f.columns:
            cutoff = pd.Timestamp("now").normalize() - pd.Timedelta(days=days_back)
            f = f[pd.to_datetime(f["date"], errors="coerce") >= cutoff]

        # Sort: HIGH first, then score desc, then most recent
        sort_cols = [c for c in ["priority", "score", "date"] if c in f.columns]
        if sort_cols:
            f = f.sort_values(sort_cols, ascending=[False if c != "date" else False for c in sort_cols])

        # Build clickable news column
        title_col = "title" if "title" in f.columns else None
        url_col = "url" if "url" in f.columns else None
        if title_col or url_col:
            f["news"] = f.apply(
                lambda r: make_clickable(r.get(title_col, ""), r.get(url_col, "")),
                axis=1
            )

        # Choose display columns (use 'news' instead of separate title/url)
        show_cols = [c for c in ["date", "ticker", "type", "priority", "score", "news", "message"] if c in f.columns or c == "news"]
        disp = f[show_cols] if show_cols else f

        # Render as HTML for clickable links
        st.markdown(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Export filtered
        st.download_button(
            "⬇️ Download filtered alerts (CSV)",
            data=f.to_csv(index=False).encode("utf-8"),
            file_name="alerts_filtered.csv",
            mime="text/csv",
        )

# Tab 15 — Economic Calendar
with tabs[15]:
    st.subheader("📆 Economic Calendar")
    df = load_csv("economic_calendar.csv")
    if df.empty:
        st.info("No economic_calendar.csv yet.")
    else:
        parse_dates_inplace(df, ("date",))
        st.dataframe(df)

# Tab 16 — Feature Importance
with tabs[16]:
    st.subheader("🔬 Feature Importance")
    df = load_csv("feature_importance.csv")
    if df.empty:
        st.info("No feature_importance.csv yet.")
    else:
        if "ticker" not in df.columns or "feature" not in df.columns or "importance" not in df.columns:
            st.warning("feature_importance.csv missing expected columns.")
        else:
            ticker = st.selectbox("Select a ticker", sorted(df["ticker"].unique()))
            filtered = df[df["ticker"] == ticker].sort_values("importance", ascending=False)
            fig = px.bar(filtered, x="feature", y="importance", title=f"Feature Importance: {ticker}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(filtered)

# Tab 17 — SL/TP Performance
with tabs[17]:
    st.subheader("🎯 SL/TP Performance Analysis")
    df = load_csv("trade_log.csv")
    if df.empty:
        st.info("No trade_log.csv yet.")
    else:
        to_numeric(df, ["profit", "stop_loss", "take_profit", "exit_price", "entry_price"])
        st.metric("Total Trades", len(df))
        if "profit" in df.columns:
            tp_trades = df[df["profit"] > 0]
            sl_trades = df[df["profit"] <= 0]
            st.metric("Avg Profit (TP)", round(tp_trades["profit"].mean(), 2) if not tp_trades.empty else 0.0)
            st.metric("Avg Loss (SL)", round(sl_trades["profit"].mean(), 2) if not sl_trades.empty else 0.0)

# Tab 18 — Sentiment + Signal Fusion (tidy)
with tabs[18]:
    st.subheader("💬 Sentiment + Signal Fusion")
    sig = load_csv("signals_with_rationale.csv")
    sns = load_csv("news_sentiment.csv")
    if sig.empty or sns.empty:
        st.info("Need both signals_with_rationale.csv and news_sentiment.csv.")
    else:
        parse_dates_inplace(sig, ("date",))
        if "publishedAt" in sns.columns and "date" not in sns.columns:
            sns["date"] = pd.to_datetime(sns["publishedAt"], errors="coerce").dt.normalize()
        else:
            parse_dates_inplace(sns, ("date",))
        sig["date"] = pd.to_datetime(sig["date"], errors="coerce").dt.normalize()
        sns["date"] = pd.to_datetime(sns["date"], errors="coerce").dt.normalize()

        need = {"ticker", "date"}
        if not need.issubset(sig.columns) or not need.issubset(sns.columns):
            st.warning("Required columns missing to merge on ['ticker','date'].")
        else:
            merged = pd.merge(sig, sns, on=["ticker", "date"], how="left")
            tidy_cols = [c for c in [
                "date","ticker","close","predicted_close","delta_pct","signal","confidence","rationale",
                "title","sentiment","url"
            ] if c in merged.columns]
            # Build clickable link for fusion view if fields are present
            if "title" in merged.columns or "url" in merged.columns:
                merged["news"] = merged.apply(
                    lambda r: make_clickable(r.get("title", ""), r.get("url", "")),
                    axis=1
                )
                if "title" in tidy_cols: tidy_cols.remove("title")
                if "url" in tidy_cols:   tidy_cols.remove("url")
                tidy_cols.insert(3, "news")  # put near the front
            st.markdown(merged[tidy_cols].to_html(escape=False, index=False), unsafe_allow_html=True)

# Tab 19 — Model Comparison (metrics + chart + export)
with tabs[19]:
    st.subheader("📊 Model Comparison")
    mc = load_csv("model_comparison.csv")
    if mc.empty:
        st.info("No model_comparison.csv yet. Expected columns: ['ticker','date','model','close','predicted_close'].")
    else:
        parse_dates_inplace(mc, ("date",))
        to_numeric(mc, ["close", "predicted_close"])
        required = {"ticker", "date", "model", "close", "predicted_close"}
        missing = sorted(required - set(mc.columns))
        if missing:
            st.warning(f"model_comparison.csv is missing: {missing}")
        else:
            tickers = sorted(mc["ticker"].dropna().unique())
            sel_ticker = st.selectbox("Select ticker", tickers)

            sub = mc[mc["ticker"] == sel_ticker].dropna(subset=["date"]).sort_values("date")
            models = sorted(sub["model"].dropna().unique())
            sel_models = st.multiselect("Select models to compare", models, default=models)

            sub = sub[sub["model"].isin(sel_models)]
            if sub.empty:
                st.info("No data for the chosen filters.")
            else:
                rows = []
                for m in sel_models:
                    dfm = sub[sub["model"] == m]
                    rows.append({
                        "model": m,
                        "R2": r2_score(dfm["close"], dfm["predicted_close"]),
                        "MAE": mae(dfm["close"], dfm["predicted_close"]),
                        "RMSE": rmse(dfm["close"], dfm["predicted_close"]),
                    })
                metrics_df = pd.DataFrame(rows).sort_values("RMSE")
                st.subheader("📐 Performance Metrics")
                st.dataframe(metrics_df, use_container_width=True)

                st.subheader("📈 Actual vs Predicted")
                fig = go.Figure()
                base = sub[["date", "close"]].dropna().drop_duplicates(subset=["date"]).sort_values("date")
                fig.add_trace(go.Scatter(x=base["date"], y=base["close"], name="Actual Close", mode="lines"))
                for m in sel_models:
                    dfm = sub[sub["model"] == m]
                    fig.add_trace(go.Scatter(x=dfm["date"], y=dfm["predicted_close"], name=f"{m} Predicted", mode="lines"))
                fig.update_layout(title=f"{sel_ticker}: Actual vs Predicted (by Model)",
                                  xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    label="⬇️ Download filtered comparison (CSV)",
                    data=sub.to_csv(index=False).encode("utf-8"),
                    file_name=f"{sel_ticker}_model_comparison_filtered.csv",
                    mime="text/csv"
                )

# Tab 20 — AI Learning Lab
with tabs[20]:
    st.subheader("🧠 AI Learning Lab")
    st.markdown("""
    Upload custom OHLC CSV and prototype quick strategies:
    - Moving Average Crossover
    - RSI Oversold/Overbought
    - Bollinger Band Breakout
    """)
    uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["date"])
            if not {"date", "close"}.issubset(df.columns):
                st.error("CSV must include 'date' and 'close' columns.")
            else:
                df = df.sort_values("date")
                strategy = st.selectbox(
                    "🧠 Choose a Strategy",
                    ["Moving Average Crossover", "RSI Strategy", "Bollinger Bands"]
                )

                if strategy == "Moving Average Crossover":
                    df["ma5"] = df["close"].rolling(5).mean()
                    df["ma20"] = df["close"].rolling(20).mean()
                    df["signal"] = (df["ma5"] > df["ma20"]).astype(int).diff().fillna(0)

                elif strategy == "RSI Strategy":
                    delta = df["close"].diff()
                    gain = delta.clip(lower=0).rolling(14).mean()
                    loss = (-delta.clip(upper=0)).rolling(14).mean()
                    rs = gain / loss.replace(0, np.nan)
                    df["rsi"] = 100 - (100 / (1 + rs))
                    df["signal"] = 0
                    df.loc[df["rsi"] < 30, "signal"] = 1
                    df.loc[df["rsi"] > 70, "signal"] = -1
                    df["signal"] = df["signal"].diff().fillna(0)

                else:  # Bollinger Bands
                    ma20 = df["close"].rolling(20).mean()
                    std20 = df["close"].rolling(20).std()
                    df["upper"] = ma20 + (2 * std20)
                    df["lower"] = ma20 - (2 * std20)
                    df["signal"] = 0
                    df.loc[df["close"] < df["lower"], "signal"] = 1
                    df.loc[df["close"] > df["upper"], "signal"] = -1
                    df["signal"] = df["signal"].diff().fillna(0)

                df["strategy_return"] = df["close"].pct_change().fillna(0) * df["signal"].shift(1).fillna(0)
                df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()

                st.subheader(f"📈 Strategy Equity Curve — {strategy}")
                fig, ax = plt.subplots()
                ax.plot(df["date"], df["cumulative_return"], label="Strategy", linewidth=2)
                ax.set_xlabel("Date")
                ax.set_ylabel("Cumulative Return")
                ax.legend()
                st.pyplot(fig)

                st.dataframe(df.tail(20))
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
