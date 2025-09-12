# view_results.py

import os
import re
import glob
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

st.set_page_config(page_title="Triton AI Unified Dashboard", layout="wide")
st.title("📊 Triton AI Unified Dashboard")

# ──────────────────────────────
# Paths (absolute, robust)
# ──────────────────────────────
THIS_FILE = Path(__file__).resolve()
# If this file lives inside /scripts, use parent.parent; otherwise parent.
# Adjust ONE of the two lines below to match your repo layout.
PROJECT_ROOT = THIS_FILE.parent       # <- if view_results.py is in repo root
# PROJECT_ROOT = THIS_FILE.parent.parent  # <- if view_results.py is in /scripts

DATA_ROOT   = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_ROOT / "results"
ORDERS_DIR  = DATA_ROOT / "orders"
PRED_DIR    = DATA_ROOT / "predictions"

for p in (RESULTS_DIR, ORDERS_DIR, PRED_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────
# Helpers
# ──────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(filename: str, folder: Path = RESULTS_DIR) -> pd.DataFrame:
    """Load a CSV with friendly errors and caching."""
    path = (folder / filename)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"❌ Could not load {path}: {e}")
        return pd.DataFrame()

def load_csv_from(folder: Path, filename: str) -> pd.DataFrame:
    return load_csv(filename, folder)

@st.cache_data(show_spinner=False)
def load_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.warning(f"⚠️ Could not read {path.name}: {e}")
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
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred); y_true = y_true[mask]; y_pred = y_pred[mask]
    if y_true.size == 0: return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred); y_true = y_true[mask]; y_pred = y_pred[mask]
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

def sum_safe(s, default=0.0):
    try:
        s = pd.to_numeric(pd.Series(s), errors="coerce")
        return float(s.fillna(0).sum())
    except Exception:
        return float(default)

@st.cache_data(show_spinner=False)
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

# ──────────────────────────────
# Diagnostics (super helpful!)
# ──────────────────────────────
with st.expander("🛠 Diagnostics (paths & files)"):
    st.write("**Project root**:", PROJECT_ROOT)
    st.write("**RESULTS_DIR**:", RESULTS_DIR)
    st.write("**ORDERS_DIR**:", ORDERS_DIR)
    st.write("**PRED_DIR**:", PRED_DIR)
    cols = st.columns(3)
    with cols[0]:
        st.caption("results/*.csv")
        st.write([p.name for p in sorted(RESULTS_DIR.glob("*.csv"))])
    with cols[1]:
        st.caption("orders/*.csv")
        st.write([p.name for p in sorted(ORDERS_DIR.glob("*.csv"))])
    with cols[2]:
        st.caption("predictions/*.csv")
        st.write([p.name for p in sorted(PRED_DIR.glob("*.csv"))])
    if st.button("↻ Clear cache & rescan"):
        st.cache_data.clear()
        st.rerun()

# ──────────────────────────────
# Tabs
# ──────────────────────────────
tabs = st.tabs([
    "🔍 Portfolio Drilldown",
    "📈 Portfolio History", "📋 Trade Log", "📊 Strategy vs Market", "🧠 AI Signals",
    "📁 Raw CSV", "📋 Backtest Summary", "📉 Risk Report", "📊 Strategy Diagnostics",
    "🏦 Portfolio Allocations", "📽️ Trade Replay", "📘 Fundamentals", "📈 Stock Scores",
    "🎯 Top Picks", "📰 News Sentiment", "🚨 Smart Alerts", "📆 Economic Calendar",
    "🔬 Feature Importance", "🎯 SL/TP Performance", "💬 Sentiment + Signal Fusion",
    "📊 Model Comparison", "🧠 AI Learning Lab",
    "🧾 Buffett Orders", "🗂️ Consolidated Orders", "🤖 AI Feedback"
])

# ──────────────────────────────
# Tab 0 — Portfolio Drilldown (new)
# ──────────────────────────────
with tabs[0]:
    st.subheader("🔍 Portfolio Drilldown")

    # Load sources
    tl = load_csv("trade_log.csv", RESULTS_DIR)                    # expected: date, ticker, action/side, price, quantity, profit
    sig = load_csv("signals_with_rationale.csv", RESULTS_DIR)      # expected: date, ticker, close, predicted_close, signal, confidence, rationale
    ns  = load_csv("news_sentiment.csv", RESULTS_DIR)              # expected: date or publishedAt, ticker, title/url/sentiment
    cur_orders = load_csv("orders_today.csv", ORDERS_DIR)          # optional
    bo = load_csv("buffett_orders.csv", ORDERS_DIR)                # optional

    # Build ticker universe from what we have
    tickers = set()
    for df in (tl, sig, ns, cur_orders, bo):
        if not df.empty and "ticker" in df.columns:
            tickers.update([t for t in df["ticker"].dropna().astype(str).unique()])
    tickers = sorted(tickers)

    if not tickers:
        st.info("No tickers found across trade logs / signals / news / orders.")
    else:
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            sel = st.selectbox("Ticker", tickers, index=0)
        with c2:
            lookback_days = st.slider("Lookback (days)", 30, 365, 180, 15)
        with c3:
            show_candles = st.selectbox("Price View", ["Line", "Candlestick"], index=0)

        # Slice data for selected ticker and window
        cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=lookback_days)

        # Trades
        tl_t = pd.DataFrame()
        if not tl.empty:
            tl_t = tl.copy()
            if "date" in tl_t.columns:
                parse_dates_inplace(tl_t, ("date",))
                tl_t = tl_t[tl_t["ticker"] == sel]
                tl_t = tl_t[tl_t["date"] >= cutoff]
            to_numeric(tl_t, ["price", "quantity", "profit"])
            # Harmonize action column name
            if "action" not in tl_t.columns and "side" in tl_t.columns:
                tl_t["action"] = tl_t["side"].str.upper()
            if "action" in tl_t.columns:
                tl_t["action"] = tl_t["action"].astype(str).str.upper()

        # Signals
        sig_t = pd.DataFrame()
        if not sig.empty:
            sig_t = sig.copy()
            parse_dates_inplace(sig_t, ("date",), normalize=False)
            sig_t = sig_t[sig_t["ticker"] == sel]
            sig_t = sig_t.dropna(subset=["date"]).sort_values("date")
            sig_t = sig_t[sig_t["date"] >= cutoff]
            to_numeric(sig_t, ["close","predicted_close","confidence","total_score"])
            if {"predicted_close","close"}.issubset(sig_t.columns):
                with np.errstate(divide="ignore", invalid="ignore"):
                    sig_t["edge_pct"] = (sig_t["predicted_close"] - sig_t["close"]) / sig_t["close"]

        # News
        ns_t = pd.DataFrame()
        if not ns.empty:
            ns_t = ns.copy()
            if "date" in ns_t.columns:
                parse_dates_inplace(ns_t, ("date",), normalize=True)
            elif "publishedAt" in ns_t.columns:
                ns_t["date"] = pd.to_datetime(ns_t["publishedAt"], errors="coerce").dt.normalize()
            ns_t = ns_t[(ns_t.get("ticker","") == sel) & ns_t["date"].notna()]
            ns_t = ns_t[ns_t["date"] >= cutoff].sort_values("date", ascending=False)
            # clickable title
            title_col = "title" if "title" in ns_t.columns else None
            url_col   = "url" if "url" in ns_t.columns else None
            if title_col or url_col:
                ns_t["news"] = ns_t.apply(lambda r: make_clickable(r.get(title_col,""), r.get(url_col,"")), axis=1)

        # Orders context (consolidated + buffett)
        ord_t = pd.DataFrame()
        if not cur_orders.empty:
            ord_t = cur_orders.copy()
            if "ticker" in ord_t.columns:
                ord_t = ord_t[ord_t["ticker"] == sel]
        bo_t = pd.DataFrame()
        if not bo.empty:
            bo_t = bo.copy()
            if "ticker" in bo_t.columns:
                bo_t = bo_t[bo_t["ticker"] == sel]

        # ── KPIs
        k1, k2, k3, k4, k5 = st.columns(5)

        # Trades KPIs
        total_trades = int(len(tl_t)) if not tl_t.empty else 0
        wins = int((tl_t.get("profit", pd.Series(dtype=float)) > 0).sum()) if not tl_t.empty and "profit" in tl_t.columns else 0
        win_rate = (wins / total_trades) if total_trades > 0 else 0.0
        avg_pnl = float(tl_t["profit"].mean()) if not tl_t.empty and "profit" in tl_t.columns else 0.0
        cum_pnl = float(tl_t["profit"].sum()) if not tl_t.empty and "profit" in tl_t.columns else 0.0

        # Signal KPI
        last_sig = None; last_conf = None
        if not sig_t.empty and {"signal","confidence","date"}.issubset(sig_t.columns):
            row = sig_t.sort_values("date").iloc[-1]
            last_sig = str(row.get("signal",""))
            last_conf = float(row.get("confidence", np.nan))

        with k1: st.metric("Trades", total_trades)
        with k2: st.metric("Win Rate", f"{win_rate:.0%}")
        with k3: st.metric("Avg P&L", f"{avg_pnl:,.2f}")
        with k4: st.metric("Cum P&L", f"{cum_pnl:,.2f}")
        with k5:
            if last_sig is not None:
                st.metric("Last Signal", f"{last_sig} ({(last_conf or 0):.2f})")
            else:
                st.metric("Last Signal", "—")

        # ── Chart: Price + signals + trade markers
        st.markdown("#### Price, Signals & Trades")
        base = sig_t.copy() if not sig_t.empty else pd.DataFrame()
        price_added = False
        fig = go.Figure()

        if show_candles == "Candlestick":
            ohlc_path = RESULTS_DIR / f"{sel}.parquet"
            ohlc = load_parquet(ohlc_path)
            if not ohlc.empty and {"date","open","high","low","close"}.issubset(ohlc.columns):
                parse_dates_inplace(ohlc, ("date",))
                ohlc = ohlc.dropna(subset=["date"])
                ohlc = ohlc[ohlc["date"] >= cutoff].sort_values("date")
                fig.add_trace(go.Candlestick(
                    x=ohlc["date"], open=ohlc["open"], high=ohlc["high"],
                    low=ohlc["low"], close=ohlc["close"], name="Price"
                ))
                price_added = True

        if not price_added and not base.empty and "close" in base.columns:
            fig.add_trace(go.Scatter(x=base["date"], y=base["close"], mode="lines", name="Close", opacity=0.6))

        # Overlay predictions if available
        if not base.empty and "predicted_close" in base.columns:
            fig.add_trace(go.Scatter(x=base["date"], y=base["predicted_close"], mode="lines", name="Predicted", opacity=0.8))

        # Signal markers sized by confidence
        if not sig_t.empty and {"signal","confidence","close"}.issubset(sig_t.columns):
            conf = sig_t["confidence"].fillna(0.0)
            if len(conf) > 0 and conf.max() > conf.min():
                conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
                sizes = conf_norm * (24 - 6) + 8
            else:
                sizes = np.full(len(sig_t), 10)

            for sig_name, dfg in sig_t.groupby("signal"):
                fig.add_trace(go.Scatter(
                    x=dfg["date"], y=dfg["close"], mode="markers", name=f"Sig: {sig_name}",
                    marker=dict(size=sizes[dfg.index]),
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        "Close: %{y:.2f}<br>"
                        "Pred: %{customdata[0]:.2f}<br>"
                        "Conf: %{customdata[1]:.2f}<br>"
                        "Edge: %{customdata[2]:.2%}<br>"
                        "<br><i>%{customdata[3]}</i><extra></extra>"
                    ),
                    customdata=np.stack([
                        dfg.get("predicted_close", pd.Series(np.nan, index=dfg.index)).fillna(0).values,
                        dfg["confidence"].fillna(0).values,
                        dfg.get("edge_pct", pd.Series(0, index=dfg.index)).fillna(0).values,
                        dfg.get("rationale", pd.Series("", index=dfg.index)).fillna("").values
                    ], axis=-1)
                ))

        # Trade markers
        if not tl_t.empty and {"date","action","price"}.issubset(tl_t.columns):
            buys  = tl_t[tl_t["action"] == "BUY"]
            sells = tl_t[tl_t["action"] == "SELL"]
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys["date"], y=buys["price"], mode="markers", name="BUY",
                    marker=dict(symbol="triangle-up", size=12),
                    hovertemplate="BUY @ %{y:.2f} (%{x|%Y-%m-%d})<extra></extra>"
                ))
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells["date"], y=sells["price"], mode="markers", name="SELL",
                    marker=dict(symbol="triangle-down", size=12),
                    hovertemplate="SELL @ %{y:.2f} (%{x|%Y-%m-%d})<extra></extra>"
                ))

        fig.update_layout(
            title=f"{sel} — Price, Signals & Trades",
            xaxis_title="Date", yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Rationale / signals table
        with st.expander("Signals & Rationale (latest 200 rows)"):
            if not sig_t.empty:
                cols_show = [c for c in [
                    "date","ticker","close","predicted_close","edge_pct",
                    "signal","confidence","total_score","rationale"
                ] if c in sig_t.columns]
                st.dataframe(sig_t.sort_values("date", ascending=False).head(200)[cols_show],
                             use_container_width=True)
            else:
                st.info("No signals for this ticker in the selected window.")

        # ── News table
        with st.expander("Related News"):
            if not ns_t.empty:
                show_cols = [c for c in ["date","ticker","sentiment","news","description"] if c in ns_t.columns or c=="news"]
                disp = ns_t[show_cols] if show_cols else ns_t
                st.markdown(disp.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("No recent news rows for this ticker.")

        # ── Orders context
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**Consolidated Orders (today) for this ticker**")
            if not ord_t.empty:
                st.dataframe(ord_t, use_container_width=True, height=240)
            else:
                st.caption("No rows in data/orders/orders_today.csv for this ticker.")
        with cR:
            st.markdown("**Buffett Orders (current) for this ticker**")
            if not bo_t.empty:
                st.dataframe(bo_t, use_container_width=True, height=240)
            else:
                st.caption("No rows in data/orders/buffett_orders.csv for this ticker.")

# ──────────────────────────────
# Tab 1 — Portfolio History
# ──────────────────────────────
with tabs[1]:
    st.subheader("📈 Portfolio Value Over Time")
    df = load_csv("portfolio_history.csv", RESULTS_DIR)
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

# Tab 2 — Trade Log
with tabs[2]:
    st.subheader("📋 Trade Log")
    df = load_csv("trade_log.csv", RESULTS_DIR)
    if df.empty: st.info("No trade_log.csv yet.")
    else: st.dataframe(df, use_container_width=True)

# Tab 3 — Strategy vs Market
with tabs[3]:
    st.subheader("📊 Strategy vs Market")
    df = load_csv("strategy_vs_market.csv", RESULTS_DIR)
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

# Tab 4 — AI Signals
with tabs[4]:
    st.subheader("🧠 AI Signals + Rationale")
    df = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    if df.empty:
        st.info("No signals_with_rationale.csv yet.")
    else:
        parse_dates_inplace(df, ("date",))
        df = df.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"])
        to_numeric(df, ["close","predicted_close","confidence","rsi14","sma20","sma50",
                        "atr14","sentiment","total_score","pe_ratio","dividend_yield"])
        if {"close","predicted_close"}.issubset(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                df["edge_pct"] = ((df["predicted_close"] - df["close"]) / df["close"]).replace([np.inf, -np.inf], np.nan)

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
            conf = f["confidence"].fillna(0.0)
            conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
            f["conf_size"] = conf_norm * (size_max - size_min) + size_min

            fig = go.Figure()
            added_price = False
            if chart_type == "Candlestick":
                ohlc_path = RESULTS_DIR / f"{selected_ticker}.parquet"
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
                fig.add_trace(go.Scatter(x=base["date"], y=base["close"], mode="lines", name="Price", opacity=0.55))

            if show_sma:
                fig.add_trace(go.Scatter(x=base["date"], y=base["sma20_calc"], mode="lines", name="SMA(20)", opacity=0.85))

            for sig_name, dfg in f.groupby("signal"):
                fig.add_trace(go.Scatter(
                    x=dfg["date"], y=dfg["close"], mode="markers", name=sig_name,
                    marker=dict(size=dfg["conf_size"]),
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        "Close: %{y:.2f}<br>"
                        "Predicted: %{customdata[2]:.2f}<br>"
                        f"Signal: {sig_name}<br>"
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
            fig.update_layout(title=f"{selected_ticker} — Signals over time (hover for rationale)",
                              xaxis_title="date", yaxis_title="close", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show table"):
                cols = ["date","ticker","close","predicted_close","edge_pct","signal",
                        "confidence","rsi14","sma20","sma50","atr14","sentiment",
                        "total_score","pe_ratio","dividend_yield","rationale"]
                cols = [c for c in cols if c in f.columns]
                st.dataframe(f[cols], use_container_width=True)

# Tab 5 — Raw CSV Browser (multi-folder)
with tabs[5]:
    st.subheader("📁 Browse Any CSV")
    def _list_csvs(root: Path):
        return sorted([p for p in root.glob("*.csv")], key=lambda p: p.name.lower())

    c1, c2, c3, _ = st.columns([2,2,2,1])
    with c1:
        which_dir = st.selectbox("Folder", [RESULTS_DIR, ORDERS_DIR, PRED_DIR], format_func=lambda p: str(p))
    with c2:
        if st.button("↻ Refresh list"):
            st.cache_data.clear()
            st.rerun()
    with c3:
        files = _list_csvs(which_dir)
        st.caption(f"Found {len(files)} CSVs")

    if not files:
        st.info(f"No CSV files found in {which_dir}.")
    else:
        names = [p.name for p in files]
        selected = st.selectbox("Select a file", names, key=f"csv_sel_{which_dir}")
        df = load_csv_from(which_dir, selected)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download this CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=selected,
            mime="text/csv"
        )

# Tab 6 — Backtest Summary
with tabs[6]:
    st.subheader("📋 Backtest Summary")
    df = load_csv("backtest_summary.csv", RESULTS_DIR)
    if df.empty: st.info("No backtest_summary.csv yet.")
    else: st.dataframe(df, use_container_width=True)

# Tab 7 — Risk Report (Drawdown)
with tabs[7]:
    st.subheader("📉 Risk: Portfolio Drawdown")
    df = load_csv("portfolio_history.csv", RESULTS_DIR)
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

# Tab 8 — Strategy Diagnostics
with tabs[8]:
    st.subheader("📊 Strategy Diagnostics")
    df = load_csv("trade_log.csv", RESULTS_DIR)
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

# Tab 9 — Portfolio Allocations
with tabs[9]:
    st.subheader("🏦 Portfolio Allocations")
    df = load_csv("trade_log.csv", RESULTS_DIR)
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

# Tab 10 — Trade Replay
with tabs[10]:
    st.subheader("📽️ Trade Replay")
    df = load_csv("trade_log.csv", RESULTS_DIR)
    if df.empty: st.info("No trade_log.csv yet.")
    else:
        if "ticker" not in df.columns: st.warning("Missing 'ticker' column.")
        else:
            ticker = st.selectbox("Select ticker", sorted(df["ticker"].dropna().unique()))
            trades = df[df["ticker"] == ticker]
            cols = ["date","action","price","quantity"]
            st.dataframe(trades[cols] if set(cols).issubset(trades.columns) else trades, use_container_width=True)

# Tab 11 — Fundamentals
with tabs[11]:
    st.subheader("📘 Fundamental Data")
    df = load_csv("fundamentals.csv", RESULTS_DIR)
    if df.empty: st.info("No fundamentals.csv yet.")
    else: st.dataframe(df, use_container_width=True)

# Tab 12 — Stock Scores
with tabs[12]:
    st.subheader("📈 Stock Scores")
    df = load_csv("stock_scores.csv", RESULTS_DIR)
    if df.empty: st.info("No stock_scores.csv yet.")
    else:
        score_col = get_score_col(df)
        if score_col: st.dataframe(df.sort_values(score_col, ascending=False), use_container_width=True)
        else:
            st.warning("No score column found (expected 'total_score' or 'score'). Showing raw data.")
            st.dataframe(df, use_container_width=True)

# Tab 13 — Top Picks
with tabs[13]:
    st.subheader("🎯 Top Fundamental Picks")
    df = load_csv("stock_scores.csv", RESULTS_DIR)
    if df.empty: st.info("No stock_scores.csv yet.")
    else:
        score_col = get_score_col(df)
        if score_col:
            top = df.sort_values(score_col, ascending=False).head(10); st.dataframe(top, use_container_width=True)
        else:
            st.warning("No score column found (expected 'total_score' or 'score'). Showing first 10 rows.")
            st.dataframe(df.head(10), use_container_width=True)

# Tab 14 — News Sentiment
with tabs[14]:
    st.subheader("📰 News Sentiment")
    df = load_csv("news_sentiment.csv", RESULTS_DIR)
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

# Tab 15 — Smart Alerts
with tabs[15]:
    st.subheader("🚨 Smart Alerts")
    df = load_csv("alerts.csv", RESULTS_DIR)
    if df.empty: df = load_csv("smart_alerts.csv", RESULTS_DIR)
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

# Tab 16 — Economic Calendar
with tabs[16]:
    st.subheader("📆 Economic Calendar")
    df = load_csv("economic_calendar.csv", RESULTS_DIR)
    if df.empty: st.info("No economic_calendar.csv yet.")
    else:
        parse_dates_inplace(df, ("date",))
        st.dataframe(df, use_container_width=True)

# Tab 17 — Feature Importance
with tabs[17]:
    st.subheader("🔬 Feature Importance")
    df = load_csv("feature_importance.csv", RESULTS_DIR)
    if df.empty: st.info("No feature_importance.csv yet.")
    else:
        if not {"ticker","feature","importance"}.issubset(df.columns):
            st.warning("feature_importance.csv missing expected columns.")
        else:
            ticker = st.selectbox("Select a ticker", sorted(df["ticker"].unique()))
            filtered = df[df["ticker"] == ticker].sort_values("importance", ascending=False)
            fig = px.bar(filtered, x="feature", y="importance", title=f"Feature Importance: {ticker}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(filtered, use_container_width=True)

# Tab 18 — SL/TP Performance
with tabs[18]:
    st.subheader("🎯 SL/TP Performance Analysis")
    df = load_csv("trade_log.csv", RESULTS_DIR)
    if df.empty:
        st.info("No trade_log.csv yet.")
    else:
        to_numeric(df, ["profit","stop_loss","take_profit","exit_price","entry_price"])
        if "profit" in df.columns:
            df = df[df["profit"].between(-1e9, 1e9)]  # clamp absurd values
        st.metric("Total Trades", len(df))
        if "profit" in df.columns:
            tp_trades = df[df["profit"] > 0]
            sl_trades = df[df["profit"] <= 0]
            st.metric("Avg Profit (TP)", round(tp_trades["profit"].mean(), 2) if not tp_trades.empty else 0.0)
            st.metric("Avg Loss (SL)", round(sl_trades["profit"].mean(), 2) if not sl_trades.empty else 0.0)

# Tab 19 — Sentiment + Signal Fusion
with tabs[19]:
    st.subheader("💬 Sentiment + Signal Fusion")
    sig = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    sns = load_csv("news_sentiment.csv", RESULTS_DIR)
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

# Tab 20 — Model Comparison
with tabs[20]:
    st.subheader("📊 Model Comparison")
    mc = load_csv("model_comparison.csv", RESULTS_DIR)
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

# Tab 21 — AI Learning Lab
with tabs[21]:
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

                st.dataframe(df.tail(20), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# Tab 22 — Buffett Orders
with tabs[22]:
    st.subheader("🧾 Buffett Orders (current)")
    cur = load_csv("buffett_orders.csv", ORDERS_DIR)
    if cur.empty:
        st.info("No current buffett_orders.csv in data/orders yet.")
    else:
        to_numeric(cur, ["target_weight","current_weight","current_value","target_value","delta_notional","buffett_score"])
        n_syms = cur["ticker"].nunique() if "ticker" in cur.columns else len(cur)
        tw_sum = float(cur["target_weight"].fillna(0).sum()) if "target_weight" in cur.columns else np.nan
        buys = cur[cur["action"] == "BUY"]
        sells = cur[cur["action"] == "SELL"]
        c1, c2, c3a, c3b = st.columns([1,1,1,1])
        with c1: st.metric("Symbols", n_syms)
        with c2: st.metric("Sum target weights", f"{tw_sum:0.3f}" if np.isfinite(tw_sum) else "—")
        with c3a: st.metric("Total BUY $", f"{sum_safe(buys['delta_notional']):,.0f}")
        with c3b: st.metric("Total SELL $", f"{sum_safe(sells['delta_notional']):,.0f}")

        st.write("**Top BUYS by notional**")
        st.dataframe(buys.sort_values("delta_notional", ascending=False).head(10), use_container_width=True)
        st.write("**Top SELLS by notional**")
        st.dataframe(sells.sort_values("delta_notional", ascending=True).head(10), use_container_width=True)

        with st.expander("All Orders"):
            st.dataframe(cur, use_container_width=True)

        hist = sorted((RESULTS_DIR).glob("buffett_orders_*.csv"))
        if hist:
            st.caption(f"Latest history file: {hist[-1].name} (in {RESULTS_DIR})")

# Tab 23 — Consolidated Orders
with tabs[23]:
    st.subheader("🗂️ Consolidated Orders (ML × Buffett blend)")
    con = load_csv("orders_today.csv", ORDERS_DIR)
    if con.empty:
        st.info("No orders_today.csv in data/orders yet.")
    else:
        to_numeric(con, ["target_weight"])
        n_syms = con["ticker"].nunique() if "ticker" in con.columns else len(con)
        tw_sum = float(con["target_weight"].fillna(0).sum()) if "target_weight" in con.columns else np.nan
        c1, c2 = st.columns(2)
        with c1: st.metric("Symbols", n_syms)
        with c2: st.metric("Sum target weights", f"{tw_sum:0.3f}" if np.isfinite(tw_sum) else "—")
        st.dataframe(con, use_container_width=True)

# Tab 24 — AI Feedback
with tabs[24]:
    st.subheader("🤖 AI Feedback (allocator runs)")
    rows = load_jsonl(RESULTS_DIR / "ai_feedback.jsonl")
    if not rows:
        st.info("No ai_feedback.jsonl yet.")
    else:
        last = rows[-1]
        runs = len(rows)
        total_buy = last.get("orders", {}).get("total_buy_notional", 0) or 0
        total_sell = last.get("orders", {}).get("total_sell_notional", 0) or 0
        uni_size = last.get("universe", {}).get("count", None)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Runs", runs)
        with c2: st.metric("Total BUY $ (last)", f"{total_buy:,.0f}")
        with c3: st.metric("Total SELL $ (last)", f"{total_sell:,.0f}")
        with c4: st.metric("Universe size (last)", uni_size if uni_size is not None else "—")

        with st.expander("Latest run — details"):
            st.json(last)

        def flatten(d, prefix=""):
            out = {}
            for k, v in d.items():
                kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                if isinstance(v, dict):
                    out.update(flatten(v, kk))
                else:
                    out[kk] = v
            return out
        table = pd.DataFrame([flatten(r) for r in rows])
        st.write("All feedback records")
        st.dataframe(table, use_container_width=True)

# footer
st.caption("Tip: If new files aren’t appearing, use the Diagnostics expander or click ↻ in Raw CSV to clear cache and rescan.")
