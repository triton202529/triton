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
PRED_DIR = "data/predictions"

# ---------- helpers ----------
@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from data/results with friendly errors and caching."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"❌ Could not load {filename}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.warning(f"⚠️ Could not read {os.path.basename(path)}: {e}")
        return pd.DataFrame()

def parse_dates_inplace(df: pd.DataFrame, cols=("date",), normalize=False):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            if normalize:
                df[c] = df[c].dt.normalize()
    return df

def get_score_col(df: pd.DataFrame):
    if "total_score" in df.columns:  # current pipeline
        return "total_score"
    if "score" in df.columns:        # legacy
        return "score"
    return None

def to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]; y_pred = y_pred[mask]
    if y_true.size < 2: return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]; y_pred = y_pred[mask]
    if y_true.size == 0: return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]; y_pred = y_pred[mask]
    if y_true.size == 0: return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# --- news helpers ---
def strip_html(s):
    if pd.isna(s): return s
    return re.sub(r"<[^>]*>", "", str(s))

def extract_href(s):
    if pd.isna(s): return None
    m = re.search(r'href="([^"]+)"', str(s))
    return m.group(1) if m else None

def make_clickable(title, url):
    if pd.isna(url) or not str(url).strip():
        return str(title) if not pd.isna(title) else ""
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
        parse_dates_inplace(df, ("date",))
        to_numeric(df, ["total_value", "cash", "market_value"])
        df = df.dropna(subset=["date", "total_value"]).sort_values("date")
        df = df[df["total_value"] > 0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["total_value"], mode="lines", name="Total Value"))
        fig.update_layout(title="Portfolio Equity Curve", xaxis_title="Date", yaxis_title="Portfolio Value")
        st.plotly_chart(fig, use_container_width=True)

# Tab 1 — Trade Log
with tabs[1]:
    st.subheader("📋 Trade Log")
    df = load_csv("trade_log.csv")
    if df.empty: st.info("No trade_log.csv yet.")
    else: st.dataframe(df)

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

# Tab 3 — AI Signals (candlestick toggle + confidence sizing + hover rationale)
with tabs[3]:
    st.subheader("🧠 AI Signals + Rationale")

    df = load_csv("signals_with_rationale.csv")
    if df.empty:
        st.info("No signals_with_rationale.csv yet.")
    else:
        parse_dates_inplace(df, ("date",))
        df = df.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"])
        to_numeric(df, ["close","predicted_close","confidence","rsi14","sma20","sma50",
                        "atr14","sentiment","total_score","pe_ratio","dividend_yield"])

        if {"close","predicted_close"}.issubset(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                df["edge_pct"] = ((df["predicted_close"] - df["close"]) / df["close"])\
                                    .replace([np.inf, -np.inf], np.nan)

        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
        with c1:
            tickers = sorted(df["ticker"].dropna().unique().tolist())
            selected_ticker = st.selectbox("Ticker", tickers)
        with c2:
            sel_signals = st.multiselect("Signals", ["BUY","SELL","HOLD"], default=["BUY","SELL","HOLD"])
        with c3:
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.05, 0.01)
        with c4:
            chart_type = st.selectbox("Chart type", ["Line", "Candlestick"])
        with c5:
            size_min, size_max = st.slider("Marker size range", 4, 32, (6, 22))

        show_sma = st.checkbox("Overlay SMA(20)", value=False)

        f = df[(df["ticker"] == selected_ticker) &
               (df["signal"].isin(sel_signals)) &
               (df["confidence"].fillna(0) >= min_conf)].copy()

        if f.empty:
            st.info("No rows after filtering. Try different filters.")
        else:
            base = df[df["ticker"] == selected_ticker].copy().sort_values("date")
            base["sma20_calc"] = base["close"].rolling(20).mean()

            # Marker size mapping
            conf = f["confidence"].fillna(0.0)
            conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
            f["conf_size"] = conf_norm * (size_max - size_min) + size_min

            fig = go.Figure()

            # Overlay price (line or candle)
            added_price = False
            if chart_type == "Candlestick":
                ohlc_path = os.path.join(RESULTS_DIR, f"{selected_ticker}.parquet")
                ohlc = load_parquet(ohlc_path)
                if not ohlc.empty and {"date","open","high","low","close"}.issubset(ohlc.columns):
                    parse_dates_inplace(ohlc, ("date",))
                    ohlc = ohlc.dropna(subset=["date"]).sort_values("date")
                    fig.add_trace(go.Candlestick(
                        x=ohlc["date"], open=ohlc["open"], high=ohlc["high"],
                        low=ohlc["low"], close=ohlc["close"], name="Price"
                    ))
                    added_price = True
            if not added_price:
                fig.add_trace(go.Scatter(
                    x=base["date"], y=base["close"], mode="lines", name="Price", opacity=0.55
                ))

            if show_sma:
                fig.add_trace(go.Scatter(
                    x=base["date"], y=base["sma20_calc"], mode="lines",
                    name="SMA(20)", opacity=0.85
                ))

            # BUY/SELL markers with rationale hover (includes predicted price)
            for sig, dfg in f.groupby("signal"):
                fig.add_trace(go.Scatter(
                    x=dfg["date"], y=dfg["close"], mode="markers",
                    name=sig,
                    marker=dict(size=dfg["conf_size"]),
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        "Close: %{y:.2f}<br>"
                        "Predicted: %{customdata[2]:.2f}<br>"
                        f"Signal: {sig}<br>"
                        "Confidence: %{customdata[0]:.2f}<br>"
                        "Edge: %{customdata[1]:.2%}<br>"
                        "<br><i>%{customdata[3]}</i><extra></extra>"
                    ),
                    customdata=np.stack([
                        dfg["confidence"].fillna(0).values,
                        dfg["edge_pct"].fillna(0).values if "edge_pct" in dfg else np.zeros(len(dfg)),
                        dfg["predicted_close"].fillna(0).values if "predicted_close" in dfg else np.zeros(len(dfg)),
                        dfg["rationale"].fillna("").values
                    ], axis=-1)
                ))

            fig.update_layout(
                title=f"{selected_ticker} — Signals over time (hover for rationale)",
                xaxis_title="date", yaxis_title="close", xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show table"):
                cols = ["date","ticker","close","predicted_close","edge_pct","signal",
                        "confidence","rsi14","sma20","sma50","atr14","sentiment",
                        "total_score","pe_ratio","dividend_yield","rationale"]
                cols = [c for c in cols if c in f.columns]
                st.dataframe(f[cols], use_container_width=True)

# Tab 4 — Raw CSV Browser
with tabs[4]:
    st.subheader("📁 Browse Any CSV in data/results")
    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")])
    if not files: st.info("No CSV files found in data/results.")
    else:
        selected = st.selectbox("Select a file", files)
        df = load_csv(selected); st.dataframe(df)

# Tab 5 — Backtest Summary
with tabs[5]:
    st.subheader("📋 Backtest Summary")
    df = load_csv("backtest_summary.csv")
    if df.empty: st.info("No backtest_summary.csv yet.")
    else: st.dataframe(df)

# Tab 6 — Risk Report (Drawdown)
with tabs[6]:
    st.subheader("📉 Risk: Portfolio Drawdown")
    df = load_csv("portfolio_history.csv")
    if df.empty:
        st.info("No portfolio_history.csv yet.")
    else:
        parse_dates_inplace(df, ("date",))
        to_numeric(df, ["total_value"])
        df = df.dropna(subset=["date","total_value"]).sort_values("date")
        df = df[df["total_value"] > 0]
        if df.empty:
            st.info("No positive portfolio values to chart yet.")
        else:
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
                         labels={"x":"Signal","y":"Count"}, title="Signal Distribution")
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
        required = {"action","quantity","ticker"}
        if not required.issubset(df.columns):
            st.warning(f"Missing columns: {sorted(required - set(df.columns))}")
        else:
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
            latest = df[df["action"].str.upper() == "BUY"].groupby("ticker")["quantity"].sum()
            if latest.empty or latest.fillna(0).sum() == 0:
                st.info("No BUY records to visualize.")
            else:
                fig = px.pie(values=latest.values, names=latest.index, title="Holdings Allocation")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(latest.reset_index().rename(columns={"quantity":"shares"}))

# Tab 9 — Trade Replay
with tabs[9]:
    st.subheader("📽️ Trade Replay")
    df = load_csv("trade_log.csv")
    if df.empty: st.info("No trade_log.csv yet.")
    else:
        if "ticker" not in df.columns: st.warning("Missing 'ticker' column.")
        else:
            ticker = st.selectbox("Select ticker", sorted(df["ticker"].dropna().unique()))
            trades = df[df["ticker"] == ticker]
            cols = ["date","action","price","quantity"]
            st.dataframe(trades[cols] if set(cols).issubset(trades.columns) else trades)

# Tab 10 — Fundamentals
with tabs[10]:
    st.subheader("📘 Fundamental Data")
    df = load_csv("fundamentals.csv")
    if df.empty: st.info("No fundamentals.csv yet.")
    else: st.dataframe(df)

# Tab 11 — Stock Scores
with tabs[11]:
    st.subheader("📈 Stock Scores")
    df = load_csv("stock_scores.csv")
    if df.empty: st.info("No stock_scores.csv yet.")
    else:
        score_col = get_score_col(df)
        if score_col: st.dataframe(df.sort_values(score_col, ascending=False))
        else:
            st.warning("No score column found (expected 'total_score' or 'score'). Showing raw data.")
            st.dataframe(df)

# Tab 12 — Top Picks
with tabs[12]:
    st.subheader("🎯 Top Fundamental Picks")
    df = load_csv("stock_scores.csv")
    if df.empty: st.info("No stock_scores.csv yet.")
    else:
        score_col = get_score_col(df)
        if score_col:
            top = df.sort_values(score_col, ascending=False).head(10); st.dataframe(top)
        else:
            st.warning("No score column found (expected 'total_score' or 'score'). Showing first 10 rows.")
            st.dataframe(df.head(10))

# Tab 13 — News Sentiment (clickable links)
with tabs[13]:
    st.subheader("📰 News Sentiment")
    df = load_csv("news_sentiment.csv")
    if df.empty: st.info("No news_sentiment.csv yet.")
    else:
        parse_dates_inplace(df, ("publishedAt","date"))
        if "description" in df.columns and ("url" not in df.columns or df["url"].isna().all()):
            df["url"] = df["description"].apply(extract_href)
            df["description"] = df["description"].apply(strip_html)
        title_col = "title" if "title" in df.columns else None
        url_col = "url" if "url" in df.columns else None
        if title_col or url_col:
            df["news"] = df.apply(lambda r: make_clickable(r.get(title_col,""), r.get(url_col,"")), axis=1)
        show_cols = [c for c in ["date","ticker","sentiment","news","description"] if c in df.columns or c=="news"]
        disp = df[show_cols] if show_cols else df
        st.markdown(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

# Tab 14 — Smart Alerts
with tabs[14]:
    st.subheader("🚨 Smart Alerts")
    df = load_csv("alerts.csv")
    if df.empty: df = load_csv("smart_alerts.csv")
    if df.empty:
        st.info("No alerts CSV found.")
    else:
        parse_dates_inplace(df, ("date","timestamp"))
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        if "priority" in df.columns:
            pri_order = ["LOW","MEDIUM","HIGH"]
            df["priority"] = pd.Categorical(df["priority"], categories=pri_order, ordered=True)

        col_l, col_r = st.columns([3,2])
        with col_l:
            min_pri = st.selectbox("Minimum priority", options=["LOW","MEDIUM","HIGH"], index=1)
            tickers = sorted(df["ticker"].dropna().unique()) if "ticker" in df.columns else []
            sel_tickers = st.multiselect("Tickers", tickers, default=[])
        with col_r:
            days_back = st.slider("Show last N days", 3, 60, 30, 1)

        f = df.copy()
        if "priority" in f.columns:
            pri_rank = {"LOW":0,"MEDIUM":1,"HIGH":2}
            f = f[f["priority"].map(pri_rank).fillna(0) >= pri_rank[min_pri]]
        if sel_tickers:
            f = f[f["ticker"].isin(sel_tickers)]
        if "date" in f.columns:
            cutoff = pd.Timestamp("now").normalize() - pd.Timedelta(days=days_back)
            f = f[pd.to_datetime(f["date"], errors="coerce") >= cutoff]

        sort_cols = [c for c in ["priority","score","date"] if c in f.columns]
        if sort_cols: f = f.sort_values(sort_cols, ascending=[False, False, False][:len(sort_cols)])

        title_col = "title" if "title" in f.columns else None
        url_col = "url" if "url" in f.columns else None
        if title_col or url_col:
            f["news"] = f.apply(lambda r: make_clickable(r.get(title_col,""), r.get(url_col,"")), axis=1)

        show_cols = [c for c in ["date","ticker","type","priority","score","news","message"] if c in f.columns or c=="news"]
        disp = f[show_cols] if show_cols else f
        st.markdown(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Alerts shown", len(f))
        with c2:
            if "priority" in f.columns: st.metric("HIGH priority", int((f["priority"] == "HIGH").sum()))
        with c3:
            if "ticker" in f.columns: st.metric("Unique tickers", f["ticker"].nunique())

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
    if df.empty: st.info("No economic_calendar.csv yet.")
    else:
        parse_dates_inplace(df, ("date",)); st.dataframe(df)

# Tab 16 — Feature Importance
with tabs[16]:
    st.subheader("🔬 Feature Importance")
    df = load_csv("feature_importance.csv")
    if df.empty: st.info("No feature_importance.csv yet.")
    else:
        if not {"ticker","feature","importance"}.issubset(df.columns):
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
        to_numeric(df, ["profit","stop_loss","take_profit","exit_price","entry_price"])
        if "profit" in df.columns:
            df = df[df["profit"].between(-1e6, 1e6)]  # clamp absurd values
        st.metric("Total Trades", len(df))
        if "profit" in df.columns:
            tp_trades = df[df["profit"] > 0]
            sl_trades = df[df["profit"] <= 0]
            st.metric("Avg Profit (TP)", round(tp_trades["profit"].mean(), 2) if not tp_trades.empty else 0.0)
            st.metric("Avg Loss (SL)", round(sl_trades["profit"].mean(), 2) if not sl_trades.empty else 0.0)

# Tab 18 — Sentiment + Signal Fusion
with tabs[18]:
    st.subheader("💬 Sentiment + Signal Fusion")
    sig = load_csv("signals_with_rationale.csv")
    sns = load_csv("news_sentiment.csv")
    if sig.empty or sns.empty:
        st.info("Need both signals_with_rationale.csv and news_sentiment.csv.")
    else:
        parse_dates_inplace(sig, ("date",), normalize=True)
        if "delta_pct" not in sig.columns and {"predicted_close","close"}.issubset(sig.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                sig["delta_pct"] = (sig["predicted_close"] - sig["close"]) / sig["close"]
        if "publishedAt" in sns.columns and "date" not in sns.columns:
            sns["date"] = pd.to_datetime(sns["publishedAt"], errors="coerce").dt.normalize()
        else:
            parse_dates_inplace(sns, ("date",), normalize=True)

        need = {"ticker","date"}
        if not need.issubset(sig.columns) or not need.issubset(sns.columns):
            st.warning("Required columns missing to merge on ['ticker','date'].")
        else:
            merged = pd.merge(sig, sns, on=["ticker","date"], how="left")
            tidy_cols = [c for c in [
                "date","ticker","close","news","predicted_close","delta_pct","signal","confidence","rationale",
                "sentiment","url","title"
            ] if c in merged.columns]
            if "title" in merged.columns or "url" in merged.columns:
                merged["news"] = merged.apply(lambda r: make_clickable(r.get("title",""), r.get("url","")), axis=1)
            for c in ("title","url"):
                if c in tidy_cols: tidy_cols.remove(c)
            if "news" not in tidy_cols: tidy_cols.insert(2, "news")
            st.markdown(merged[tidy_cols].to_html(escape=False, index=False), unsafe_allow_html=True)

# Tab 19 — Model Comparison
with tabs[19]:
    st.subheader("📊 Model Comparison")
    mc = load_csv("model_comparison.csv")
    if mc.empty:
        st.info("No model_comparison.csv yet. Expected columns: ['ticker','date','model','close','predicted_close'].")
    else:
        parse_dates_inplace(mc, ("date",))
        to_numeric(mc, ["close","predicted_close"])
        required = {"ticker","date","model","close","predicted_close"}
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
                base = sub[["date","close"]].dropna().drop_duplicates(subset=["date"]).sort_values("date")
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
            if not {"date","close"}.issubset(df.columns):
                st.error("CSV must include 'date' and 'close' columns.")
            else:
                df = df.sort_values("date")
                strategy = st.selectbox("🧠 Choose a Strategy",
                                        ["Moving Average Crossover", "RSI Strategy", "Bollinger Bands"])

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
                ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return"); ax.legend()
                st.pyplot(fig)

                st.dataframe(df.tail(20))
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
