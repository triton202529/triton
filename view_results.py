# view_results.py — TOP HALF (cleaned)
# - Keeps ONE cache-clear button (in Diagnostics expander)
# - Removes duplicate/global cache-clear button blocks
# - Adds build caption only (no duplicate keys)

import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Optional libs (graceful fallbacks)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    px = None
    go = None
    PLOTLY_OK = False

try:
    import matplotlib.pyplot as plt  # used in Learning Lab
    MPL_OK = True
except Exception:
    plt = None
    MPL_OK = False

# App meta
st.set_page_config(page_title="Triton AI Unified Dashboard", layout="wide")
APP_VERSION = "r25-29_2025-09-27a"
st.title("📊 Triton AI Unified Dashboard")

# ──────────────────────────────
# Repo root detection + override
# ──────────────────────────────
THIS_FILE = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = THIS_FILE.parent  # if file is in repo root

with st.sidebar:
    st.subheader("⚙️ Advanced")
    root_mode = st.radio(
        "Project root",
        options=["Auto (this file's folder)", "Manual path"],
        index=0,
        key="root_mode_choice",
    )
    if root_mode == "Manual path":
        custom_root = st.text_input(
            "Enter absolute path to repo root",
            value=str(st.session_state.get("custom_root", DEFAULT_PROJECT_ROOT)),
            key="root_manual_input",
        )
        try:
            PROJECT_ROOT = Path(custom_root).expanduser().resolve()
            st.session_state["custom_root"] = str(PROJECT_ROOT)
            st.caption(f"Using custom root: {PROJECT_ROOT}")
        except Exception as e:
            st.error(f"Invalid custom root: {e}")
            PROJECT_ROOT = DEFAULT_PROJECT_ROOT
    else:
        PROJECT_ROOT = DEFAULT_PROJECT_ROOT
        st.caption(f"Using auto root: {PROJECT_ROOT}")

    # Build tag only (no buttons here to avoid duplicate keys)
    st.caption(f"Build: {APP_VERSION}")

DATA_ROOT   = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_ROOT / "results"
ORDERS_DIR  = DATA_ROOT / "orders"
PRED_DIR    = DATA_ROOT / "predictions"

# Ensure folders exist
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
    """
    Parse given columns as datetime in UTC, then strip tz to naive.
    Optionally normalize to midnight.
    """
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            # remove timezone (naive UTC)
            df[c] = s.dt.tz_localize(None)
            if normalize:
                df[c] = df[c].dt.normalize()
    return df

def ensure_date(df: pd.DataFrame, candidates=None, normalize=False) -> pd.DataFrame:
    """
    Ensure df has a 'date' column parsed in UTC then made tz-naive.
    Try candidate columns; if none exist, create 'date' as NaT.
    """
    if candidates is None:
        candidates = ["date", "as_of", "timestamp", "time", "datetime", "Date", "created_at", "updated_at"]
    chosen = next((c for c in candidates if c in df.columns), None)
    if chosen is not None:
        s = pd.to_datetime(df[chosen], errors="coerce", utc=True)
        df["date"] = s.dt.tz_localize(None)
        if normalize:
            df["date"] = df["date"].dt.normalize()
    else:
        df["date"] = pd.NaT
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

# [NEW HELPERS] — normalization + quick performance stats for equity curves
def normalize_to_one(series: pd.Series) -> pd.Series:
    """Normalize a positive cumulative series to start at 1.0, ignoring leading NaNs."""
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    first = s.dropna().iloc[0]
    if not np.isfinite(first) or first == 0:
        return s
    return s / first

def perf_stats_from_levels(levels: pd.Series, freq_per_year: int = 252) -> dict:
    """
    Compute total_return, CAGR, volatility, sharpe (rf=0), max_drawdown from an equity curve (levels).
    """
    s = pd.to_numeric(levels, errors="coerce").dropna()
    if s.size < 3:
        return {"total_return": np.nan, "cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    rets = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = float(s.iloc[-1] / s.iloc[0] - 1)
    years = max((s.index[-1] - s.index[0]).days, 1) / 365.25 if hasattr(s.index, "dtype") else len(s)/freq_per_year
    cagr = float((s.iloc[-1] / s.iloc[0]) ** (1/years) - 1) if years > 0 else np.nan
    vol = float(rets.std() * np.sqrt(freq_per_year)) if rets.size else np.nan
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(freq_per_year)) if rets.size else np.nan
    peak = s.cummax()
    dd = (s/peak - 1).min()
    return {"total_return": total_return, "cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": float(dd)}

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
    # fixed: use .startswith, and pass through existing <a ...> safely
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

def derive_total_value(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a total_value column exists (accept 'portfolio_value' or cash+market_value)."""
    if "total_value" in df.columns:
        df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
        return df
    if "portfolio_value" in df.columns:
        df["total_value"] = pd.to_numeric(df["portfolio_value"], errors="coerce")
        return df
    if {"cash","market_value"}.issubset(df.columns):
        df["total_value"] = pd.to_numeric(df["cash"], errors="coerce").fillna(0) + \
                            pd.to_numeric(df["market_value"], errors="coerce").fillna(0)
    return df

def backfill_close_from_parquet(sig_df: pd.DataFrame) -> pd.DataFrame:
    """If 'close' is missing, try to backfill from {ticker}.parquet last close."""
    out = sig_df.copy()
    if "close" not in out.columns:
        out["close"] = np.nan
    need_close = ~out["close"].notna()
    if not need_close.any():
        return out
    tickers = out.loc[need_close, "ticker"].dropna().astype(str).unique()
    rows = []
    for t in tickers:
        pq = RESULTS_DIR / f"{t}.parquet"
        ohlc = load_parquet(pq)
        last_close = np.nan
        if not ohlc.empty and "close" in ohlc.columns:
            parse_dates_inplace(ohlc, ("date",))
            ohlc = ohlc.dropna(subset=["date"]).sort_values("date")
            if not ohlc.empty:
                last_close = float(ohlc["close"].iloc[-1])
        rows.append((t, last_close))
    if rows:
        fill = pd.DataFrame(rows, columns=["ticker", "_last_close"])
        out = out.merge(fill, on="ticker", how="left")
        out["close"] = pd.to_numeric(out["close"], errors="coerce").fillna(out["_last_close"])
        out.drop(columns=["_last_close"], inplace=True, errors="ignore")
    return out

def stabilize_weights(raw_wide: pd.DataFrame, cap: float) -> pd.DataFrame:
    """
    Row-wise cap & normalize portfolio weights.

    Parameters
    ----------
    raw_wide : DataFrame
        Wide matrix indexed by date (rows) with tickers as columns.
        Values are raw (non-normalized) weights or scores.
        NaNs mean "no signal / unavailable".
    cap : float
        Per-ticker cap expressed as a FRACTION (e.g., 0.15 for 15%).
        Use 0 to disable capping.

    Returns
    -------
    DataFrame
        Same shape as raw_wide, with each row summing to 1 (if any
        positive weight exists on that row). If a row has no positive
        weights, we equal-weight across the non-NaN tickers on that row.
        Rows that are entirely NaN become all-zeros.
    """
    if raw_wide is None or raw_wide.empty:
        return pd.DataFrame(index=getattr(raw_wide, "index", None),
                            columns=getattr(raw_wide, "columns", None), dtype=float)

    # Keep an availability mask (which tickers exist that day)
    avail = raw_wide.notna()

    # Replace NaNs with 0 for math, then clip lower/upper
    W = raw_wide.copy().astype(float).fillna(0.0)
    W = W.clip(lower=0.0)
    if cap is not None and np.isfinite(cap) and cap > 0:
        W = W.clip(upper=float(cap))

    # Normalize rows that have positive mass
    row_sum = W.sum(axis=1)
    has_mass = row_sum > 0

    W_norm = pd.DataFrame(0.0, index=W.index, columns=W.columns)
    if has_mass.any():
        W_norm.loc[has_mass] = W.loc[has_mass].div(row_sum.loc[has_mass], axis=0)

    # For rows with zero mass, equal-weight across available tickers that day
    no_mass = ~has_mass
    if no_mass.any():
        counts = avail.loc[no_mass].sum(axis=1)  # how many tickers available that day
        # Build equal-weight rows only where at least one ticker is available
        valid = counts > 0
        if valid.any():
            eq_idx = counts[valid].index
            W_eq = avail.loc[eq_idx].div(counts.loc[eq_idx], axis=0).astype(float)
            W_norm.loc[eq_idx] = W_eq

    # Preserve original NaNs as 0 (weights are numeric); keep index/columns intact
    return W_norm.fillna(0.0)


# ──────────────────────────────
# Diagnostics
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

    # Single cache-clear button (unique key)
    if st.button("↻ Clear cache & rescan", key="diag_rescan"):
        st.cache_data.clear()
        st.rerun()

# ──────────────────────────────
# Tabs
# ──────────────────────────────
tabs = st.tabs([
    "🔍 Portfolio Drilldown",                 # 0
    "📈 Portfolio History",                   # 1
    "📋 Trade Log",                           # 2
    "📊 Strategy vs Market",                  # 3
    "🧠 AI Signals",                          # 4
    "📁 Raw CSV",                             # 5
    "📋 Backtest Summary",                    # 6
    "📉 Risk Report",                         # 7
    "📊 Strategy Diagnostics",                # 8
    "🏦 Portfolio Allocations",               # 9
    "📽️ Trade Replay",                        # 10
    "📘 Fundamentals",                         # 11
    "📈 Stock Scores",                         # 12
    "🎯 Top Picks",                            # 13
    "📰 News Sentiment",                       # 14
    "🚨 Smart Alerts",                         # 15
    "📆 Economic Calendar",                    # 16
    "🔬 Feature Importance",                   # 17
    "🎯 SL/TP Performance",                    # 18
    "💬 Sentiment + Signal Fusion",            # 19
    "📊 Model Comparison",                     # 20
    "🧠 AI Learning Lab",                      # 21
    "🧾 Buffett Orders",                       # 22
    "🗂️ Consolidated Orders",                  # 23
    "🤖 AI Feedback",                          # 24
    "📚 Equal-Weight vs Benchmark",            # 25
    "🧮 Smart-Weight Portfolio vs Benchmark",  # 26
    "🧪 Confidence Calibration",               # 27
    "🧪 Confidence-Filtered Portfolio vs Benchmark",  # 28
    "📊 Confidence × Sharpe Portfolio vs Benchmark"  # 29
])

# ──────────────────────────────
# Tab 0 — Portfolio Drilldown
# ──────────────────────────────
with tabs[0]:
    st.subheader("🔍 Portfolio Drilldown")

    # Load sources
    tl = load_csv("trade_log.csv", RESULTS_DIR)                    # expected: date, ticker, action/side, price, quantity, profit
    sig = load_csv("signals_with_rationale.csv", RESULTS_DIR)      # preferred
    if sig.empty:
        sig = load_csv("signals.csv", RESULTS_DIR)                 # fallback
    ns  = load_csv("news_sentiment.csv", RESULTS_DIR)              # expected: date/publishedAt, ticker, title/url/sentiment
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
            sel = st.selectbox("Ticker", tickers, index=0, key="t0_ticker")
        with c2:
            lookback_days = st.slider("Lookback (days)", 30, 365, 180, 15, key="t0_lookback")
        with c3:
            show_candles = st.selectbox("Price View", ["Line", "Candlestick"], index=0, key="t0_price_view")

        # Slice window — make cutoff tz-naive UTC to match parsed data
        cutoff_ts = (pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(days=lookback_days))
        cutoff_date = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)).date()

        # Trades
        tl_t = pd.DataFrame()
        if not tl.empty:
            tl_t = tl.copy()
            tl_t = ensure_date(tl_t, candidates=["date","timestamp","time","datetime","Date"])
            tl_t = tl_t[tl_t.get("ticker","") == sel]
            has_trade_dates = tl_t["date"].notna().any()
            if has_trade_dates:
                tl_t = tl_t.dropna(subset=["date"])
                tl_t = tl_t[tl_t["date"] >= cutoff_ts]
            to_numeric(tl_t, ["price", "quantity", "profit"])
            # Harmonize action column name
            if "action" not in tl_t.columns and "side" in tl_t.columns:
                tl_t["action"] = tl_t["side"].astype(str).str.upper()
            if "action" in tl_t.columns:
                tl_t["action"] = tl_t["action"].astype(str).str.upper()

        # Signals
        sig_t = pd.DataFrame()
        if not sig.empty:
            sig_t = sig.copy()
            sig_t = ensure_date(sig_t, candidates=["date","as_of","timestamp","time","datetime","Date"])
            sig_t = sig_t[sig_t.get("ticker","") == sel]
            has_sig_dates = sig_t["date"].notna().any()
            if has_sig_dates:
                sig_t = sig_t.dropna(subset=["date"]).sort_values("date")
                sig_t = sig_t[sig_t["date"] >= cutoff_ts]
            to_numeric(sig_t, ["close","predicted_close","confidence","total_score"])
            if {"predicted_close","close"}.issubset(sig_t.columns):
                with np.errstate(divide="ignore", invalid="ignore"):
                    sig_t["edge_pct"] = (sig_t["predicted_close"] - sig_t["close"]) / sig_t["close"]
            else:
                sig_t["edge_pct"] = np.nan

        # News (tz-safe, date-only comparison)
        ns_t = pd.DataFrame()
        if not ns.empty:
            ns_t = ns.copy()
            if "date" in ns_t.columns:
                ns_t["date"] = pd.to_datetime(ns_t["date"], errors="coerce", utc=True).dt.tz_localize(None)
            elif "publishedAt" in ns_t.columns:
                ns_t["date"] = pd.to_datetime(ns_t["publishedAt"], errors="coerce", utc=True).dt.tz_localize(None)
            else:
                ns_t["date"] = pd.NaT
            ns_t["date"] = ns_t["date"].dt.normalize()
            ns_t = ns_t[(ns_t.get("ticker","") == sel) & ns_t["date"].notna()]
            ns_t = ns_t[ns_t["date"].dt.date >= cutoff_date].sort_values("date", ascending=False)
            # clickable title
            title_col = "title" if "title" in ns_t.columns else None
            url_col   = "url"   if "url"   in ns_t.columns else None
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
        if not sig_t.empty and "date" in sig_t.columns and sig_t["date"].notna().any():
            row = sig_t.sort_values("date").iloc[-1]
            last_sig = str(row.get("signal",""))
            last_conf = float(row.get("confidence", np.nan)) if "confidence" in sig_t.columns else np.nan

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

        if not PLOTLY_OK:
            st.warning("Plotly not installed — `pip install plotly` for charts.")
        else:
            base = sig_t.copy() if not sig_t.empty else pd.DataFrame()
            price_added = False
            fig = go.Figure()

            # Determine whether we have dates
            has_base_dates = (not base.empty) and base["date"].notna().any()

            if show_candles == "Candlestick" and has_base_dates:
                ohlc_path = RESULTS_DIR / f"{sel}.parquet"
                ohlc = load_parquet(ohlc_path)
                if not ohlc.empty and {"date","open","high","low","close"}.issubset(ohlc.columns):
                    parse_dates_inplace(ohlc, ("date",))
                    ohlc = ohlc.dropna(subset=["date"])
                    ohlc = ohlc[ohlc["date"] >= cutoff_ts].sort_values("date")
                    fig.add_trace(go.Candlestick(
                        x=ohlc["date"], open=ohlc["open"], high=ohlc["high"],
                        low=ohlc["low"], close=ohlc["close"], name="Price"
                    ))
                    price_added = True

            # Price line if no candlestick
            if not price_added and not base.empty and "close" in base.columns:
                x_base = base["date"] if has_base_dates else np.arange(len(base))
                fig.add_trace(go.Scatter(x=x_base, y=base["close"], mode="lines", name="Close", opacity=0.6))

            # Overlay predictions if available
            if not base.empty and "predicted_close" in base.columns:
                x_base = base["date"] if has_base_dates else np.arange(len(base))
                fig.add_trace(go.Scatter(x=x_base, y=base["predicted_close"], mode="lines", name="Predicted", opacity=0.8))

            # Signal markers sized by confidence
            if not sig_t.empty and "signal" in sig_t.columns and "close" in sig_t.columns:
                if "confidence" not in sig_t.columns:
                    sig_t["confidence"] = np.nan
                conf = sig_t["confidence"].fillna(0.0)
                if len(conf) > 0 and conf.max() > conf.min():
                    conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
                    sizes = conf_norm * (24 - 6) + 8
                else:
                    sizes = pd.Series(10, index=sig_t.index)

                for sig_name, dfg in sig_t.groupby("signal"):
                    x_vals = dfg["date"] if has_base_dates else np.arange(len(dfg))
                    hover_x = "%{x|%Y-%m-%d}" if has_base_dates else "%{x}"
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=dfg["close"], mode="markers", name=f"Sig: {sig_name}",
                        marker=dict(size=sizes.loc[dfg.index]),
                        hovertemplate=(
                            f"<b>{hover_x}</b><br>"
                            "Close: %{y:.2f}<br>"
                            "Pred: %{customdata[0]:.2f}<br>"
                            "Conf: %{customdata[1]:.2f}<br>"
                            "Edge: %{customdata[2]:.2%}<br>"
                            "<br><i>%{customdata[3]}</i><extra></extra>"
                        ),
                        customdata=np.stack([
                            dfg.get("predicted_close", pd.Series(np.nan, index=dfg.index)).fillna(0).values,
                            dfg.get("confidence", pd.Series(0, index=dfg.index)).fillna(0).values,
                            dfg.get("edge_pct", pd.Series(0, index=dfg.index)).fillna(0).values,
                            dfg.get("rationale", pd.Series("", index=dfg.index)).fillna("").values
                        ], axis=-1)
                    ))

            fig.update_layout(
                title=f"{sel} — Price, Signals & Trades",
                xaxis_title="Date" if has_base_dates else "Index",
                yaxis_title="Price",
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
                st.dataframe(sig_t.sort_values("date", ascending=True).tail(200)[cols_show],
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
        df = derive_total_value(df)
        parse_dates_inplace(df, ("date",))
        to_numeric(df, ["total_value"])
        df = df.dropna(subset=["date", "total_value"]).sort_values("date")
        df = df[df["total_value"] > 0]
        if df.empty:
            st.info("No positive portfolio values to chart yet.")
        else:
            if not PLOTLY_OK:
                st.warning("Plotly not installed — `pip install plotly` for charts.")
            else:
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

# [UPDATED TAB 3] — Strategy vs Market
with tabs[3]:
    st.subheader("📊 Strategy vs Market")

    df = load_csv("strategy_vs_market.csv", RESULTS_DIR)
    if df.empty:
        st.info("No strategy_vs_market.csv yet.")
    else:
        parse_dates_inplace(df, ("date",))

        # Accept either daily returns OR cumulative levels
        has_returns = {"strategy_return","market_return"} <= set(df.columns)
        has_cum     = {"cumulative_strategy","cumulative_market"} <= set(df.columns)
        if not has_returns and not has_cum:
            st.warning("Expected either daily returns ('strategy_return','market_return') "
                       "or cumulative levels ('cumulative_strategy','cumulative_market').")
        else:
            tickers = sorted(df["ticker"].dropna().unique()) if "ticker" in df.columns else []
            if not tickers:
                st.warning("Missing 'ticker' column.")
            else:
                c1, c2, c3 = st.columns([1.2, 1, 1])
                with c1:
                    selected = st.selectbox("Select a ticker", tickers, key="t3_ticker")
                with c2:
                    normalize = st.checkbox("Normalize curves to 1.0 at start", value=True, key="t3_norm")
                with c3:
                    show_kpis = st.checkbox("Show KPI panel", value=True, key="t3_kpis")

                sub = df[df["ticker"] == selected].dropna(subset=["date"]).sort_values("date").set_index("date")

                # Build cumulative if only returns are present
                if has_returns and not has_cum:
                    sr = pd.to_numeric(sub["strategy_return"], errors="coerce").fillna(0)
                    mr = pd.to_numeric(sub["market_return"], errors="coerce").fillna(0)
                    sub["cumulative_strategy"] = (1 + sr).cumprod()
                    sub["cumulative_market"]   = (1 + mr).cumprod()

                cs = pd.to_numeric(sub.get("cumulative_strategy"), errors="coerce")
                cm = pd.to_numeric(sub.get("cumulative_market"), errors="coerce")
                if normalize:
                    cs = normalize_to_one(cs)
                    cm = normalize_to_one(cm)

                # KPIs
                if show_kpis and cs.notna().any() and cm.notna().any():
                    s_stats = perf_stats_from_levels(cs.dropna())
                    m_stats = perf_stats_from_levels(cm.dropna())
                    k1, k2, k3, k4, k5, k6 = st.columns(6)
                    with k1: st.metric("Strat Total", f"{s_stats['total_return']:.1%}" if np.isfinite(s_stats['total_return']) else "—")
                    with k2: st.metric("Strat CAGR", f"{s_stats['cagr']:.1%}" if np.isfinite(s_stats['cagr']) else "—")
                    with k3: st.metric("Strat Sharpe", f"{s_stats['sharpe']:.2f}" if np.isfinite(s_stats['sharpe']) else "—")
                    with k4: st.metric("Mkt Total", f"{m_stats['total_return']:.1%}" if np.isfinite(m_stats['total_return']) else "—")
                    with k5: st.metric("Mkt CAGR", f"{m_stats['cagr']:.1%}" if np.isfinite(m_stats['cagr']) else "—")
                    with k6: st.metric("Mkt MaxDD", f"{m_stats['max_dd']:.1%}" if np.isfinite(m_stats['max_dd']) else "—")

                # Chart
                if not PLOTLY_OK:
                    st.warning("Plotly not installed — `pip install plotly` for charts.")
                else:
                    plot_df = pd.DataFrame({"Strategy": cs, "Market": cm}).dropna(how="all")
                    fig = go.Figure()
                    if plot_df["Strategy"].notna().any():
                        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Strategy"], name="Strategy", mode="lines"))
                    if plot_df["Market"].notna().any():
                        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Market"], name="Market", mode="lines"))
                    ttl = f"{selected} Strategy vs Market" + (" (normalized)" if normalize else "")
                    fig.update_layout(title=ttl, xaxis_title="Date", yaxis_title="Cumulative Level")
                    st.plotly_chart(fig, use_container_width=True)

                # Leaderboard across all tickers (alpha rank)
                with st.expander("📋 Leaderboard across tickers"):
                    agg = []
                    for t in tickers:
                        sub_t = df[df["ticker"] == t].dropna(subset=["date"]).sort_values("date").set_index("date")
                        if has_returns and not has_cum:
                            sr = pd.to_numeric(sub_t.get("strategy_return"), errors="coerce").fillna(0)
                            mr = pd.to_numeric(sub_t.get("market_return"), errors="coerce").fillna(0)
                            sub_t["cumulative_strategy"] = (1 + sr).cumprod()
                            sub_t["cumulative_market"]   = (1 + mr).cumprod()
                        cs_t = pd.to_numeric(sub_t.get("cumulative_strategy"), errors="coerce").dropna()
                        cm_t = pd.to_numeric(sub_t.get("cumulative_market"), errors="coerce").dropna()
                        if cs_t.empty or cm_t.empty:
                            continue
                        if normalize:
                            cs_t = normalize_to_one(cs_t)
                            cm_t = normalize_to_one(cm_t)
                        s_stats = perf_stats_from_levels(cs_t)
                        m_stats = perf_stats_from_levels(cm_t)
                        agg.append({
                            "ticker": t,
                            "strategy_total": s_stats["total_return"],
                            "market_total": m_stats["total_return"],
                            "alpha_total": (s_stats["total_return"] - m_stats["total_return"]) if np.isfinite(s_stats["total_return"]) and np.isfinite(m_stats["total_return"]) else np.nan,
                            "strategy_cagr": s_stats["cagr"],
                            "market_cagr": m_stats["cagr"],
                            "strategy_sharpe": s_stats["sharpe"],
                            "market_sharpe": m_stats["sharpe"],
                        })
                    if agg:
                        agg_df = pd.DataFrame(agg).sort_values("alpha_total", ascending=False)
                        st.dataframe(agg_df, use_container_width=True)
                        st.caption("Sorted by total-return alpha (Strategy − Market).")
                    else:
                        st.caption("Not enough data to build leaderboard.")

# Tab 4 — AI Signals
with tabs[4]:
    st.subheader("🧠 AI Signals + Rationale")
    df = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    if df.empty:
        df = load_csv("signals.csv", RESULTS_DIR)
    if df.empty:
        st.info("No signals CSV yet.")
    else:
        # Normalize / ensure dates (tz-safe)
        df = ensure_date(df, candidates=["date","as_of","timestamp","time","datetime","Date"], normalize=False)
        df = df.dropna(subset=["ticker"]).sort_values(["ticker", "date"])
        # Ensure required numeric columns exist
        for c in ["close","predicted_close","confidence","rsi14","sma20","sma50",
                  "atr14","sentiment","total_score","pe_ratio","dividend_yield"]:
            if c not in df.columns:
                df[c] = np.nan
        to_numeric(df, ["close","predicted_close","confidence","rsi14","sma20","sma50",
                        "atr14","sentiment","total_score","pe_ratio","dividend_yield"])
        if {"close","predicted_close"}.issubset(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                df["edge_pct"] = ((df["predicted_close"] - df["close"]) / df["close"]).replace([np.inf, -np.inf], np.nan)

        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
        with c1:
            tickers = sorted(df["ticker"].dropna().unique().tolist())
            selected_ticker = st.selectbox("Ticker", tickers, key="t4_ticker")
        with c2:
            sel_signals = st.multiselect("Signals", ["BUY","SELL","HOLD"], default=["BUY","SELL","HOLD"], key="t4_signals")
        with c3:
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.00, 0.01, key="t4_minconf")
        with c4:
            chart_type = st.selectbox("Chart type", ["Line", "Candlestick"], key="t4_charttype")
        with c5:
            size_min, size_max = st.slider("Marker size range", 4, 32, (6, 22), key="t4_sizes")

        show_sma = st.checkbox("Overlay SMA(20)", value=False, key="t4_sma")

        # Safe filter
        f = df[(df["ticker"] == selected_ticker)].copy()
        if "signal" in f.columns:
            f = f[f["signal"].isin(sel_signals)]
        if "confidence" in f.columns:
            f = f[f["confidence"].fillna(0) >= min_conf]

        if f.empty:
            st.info("No rows after filtering. Try different filters.")
        else:
            if not PLOTLY_OK:
                st.warning("Plotly not installed — `pip install plotly` for charts.")
            else:
                base = df[df["ticker"] == selected_ticker].copy().sort_values("date")
                has_base_dates = base["date"].notna().any()
                if "close" in base.columns:
                    base["sma20_calc"] = base["close"].rolling(20).mean()
                else:
                    base["sma20_calc"] = np.nan

                conf = f["confidence"].fillna(0.0) if "confidence" in f.columns else pd.Series([0]*len(f), index=f.index)
                conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
                f["conf_size"] = conf_norm * (size_max - size_min) + size_min

                fig = go.Figure()
                added_price = False
                if chart_type == "Candlestick" and has_base_dates:
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
                if not added_price and "close" in base.columns:
                    x_base = base["date"] if has_base_dates else np.arange(len(base))
                    fig.add_trace(go.Scatter(x=x_base, y=base["close"], mode="lines", name="Price", opacity=0.55))

                if show_sma and "sma20_calc" in base.columns:
                    x_base = base["date"] if has_base_dates else np.arange(len(base))
                    fig.add_trace(go.Scatter(x=x_base, y=base["sma20_calc"], mode="lines", name="SMA(20)", opacity=0.85))

                if "signal" in f.columns and "close" in f.columns:
                    has_f_dates = f["date"].notna().any()
                    for sig_name, dfg in f.groupby("signal"):
                        x_vals = dfg["date"] if has_f_dates else np.arange(len(dfg))
                        hover_x = "%{x|%Y-%m-%d}" if has_f_dates else "%{x}"
                        fig.add_trace(go.Scatter(
                            x=x_vals, y=dfg["close"], mode="markers", name=sig_name,
                            marker=dict(size=dfg["conf_size"]),
                            hovertemplate=(
                                f"<b>{hover_x}</b><br>"
                                "Close: %{y:.2f}<br>"
                                "Predicted: %{customdata[2]:.2f}<br>"
                                f"Signal: {sig_name}<br>"
                                "Confidence: %{customdata[0]:.2f}<br>"
                                "Edge: %{customdata[1]:.2%}<br>"
                                "<br><i>%{customdata[3]}</i><extra></extra>"
                            ),
                            customdata=np.stack([
                                dfg.get("confidence", pd.Series(0, index=dfg.index)).fillna(0).values,
                                dfg.get("edge_pct", pd.Series(0, index=dfg.index)).fillna(0).values,
                                dfg.get("predicted_close", pd.Series(0, index=dfg.index)).fillna(0).values,
                                dfg.get("rationale", pd.Series("", index=dfg.index)).fillna("").values
                            ], axis=-1)
                        ))
                    fig.update_layout(title=f"{selected_ticker} — Signals over time (hover for rationale)",
                                      xaxis_title="date" if has_base_dates else "index", yaxis_title="close",
                                      xaxis_rangeslider_visible=False)
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
        which_dir = st.selectbox("Folder", [RESULTS_DIR, ORDERS_DIR, PRED_DIR], format_func=lambda p: str(p), key="t5_folder")
    with c2:
        if st.button("↻ Refresh list", key="t5_refresh"):
            st.cache_data.clear()
            st.rerun()
    with c3:
        files = _list_csvs(which_dir)
        st.caption(f"Found {len(files)} CSVs")

    if not files:
        st.info(f"No CSV files found in {which_dir}.")
    else:
        names = [p.name for p in files]
        selected = st.selectbox("Select a file", names, key=f"t5_file_{hash(str(which_dir))}")
        df = load_csv_from(which_dir, selected)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download this CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=selected,
            mime="text/csv",
            key="t5_dl",
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
        df = derive_total_value(df)
        parse_dates_inplace(df, ("date",))
        to_numeric(df, ["total_value"])
        df = df.dropna(subset=["date","total_value"]).sort_values("date")
        df = df[df["total_value"] > 0]
        if df.empty:
            st.info("No positive portfolio values to chart yet.")
        else:
            if not PLOTLY_OK:
                st.warning("Plotly not installed — `pip install plotly` for charts.")
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
        # Try 'signal' first, else fall back to 'action' or 'side'
        label_col = "signal" if "signal" in df.columns else ("action" if "action" in df.columns else ("side" if "side" in df.columns else None))
        if label_col and PLOTLY_OK:
            counts = df[label_col].astype(str).value_counts()
            fig = px.bar(x=counts.index, y=counts.values,
                         labels={"x":label_col.capitalize(),"y":"Count"},
                         title=f"{label_col.capitalize()} Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No 'signal'/'action'/'side' column to visualize.")

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
            latest = df[df["action"].astype(str).str.upper() == "BUY"].groupby("ticker")["quantity"].sum()
            if latest.empty or latest.fillna(0).sum() == 0:
                st.info("No BUY records to visualize.")
            else:
                if not PLOTLY_OK:
                    st.warning("Plotly not installed — `pip install plotly` for charts.")
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
            ticker = st.selectbox("Select ticker", sorted(df["ticker"].dropna().unique()), key="t10_ticker")
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

# Tab 14 — News Sentiment (with filters)
with tabs[14]:
    st.subheader("📰 News Sentiment")

    df = load_csv("news_sentiment.csv", RESULTS_DIR)
    if df.empty:
        st.info("No news_sentiment.csv yet.")
    else:
        # Parse/normalize date
        if "publishedAt" in df.columns and "date" not in df.columns:
            df["date"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        else:
            parse_dates_inplace(df, ("date",), normalize=True)

        # Extract link from HTML description if needed
        if "description" in df.columns and ("url" not in df.columns or df["url"].isna().all()):
            df["url"] = df["description"].apply(extract_href)
            df["description"] = df["description"].apply(strip_html)

        # Clickable headlines
        title_col = "title" if "title" in df.columns else None
        url_col   = "url"   if "url"   in df.columns else None
        if title_col or url_col:
            df["news"] = df.apply(lambda r: make_clickable(r.get(title_col,""), r.get(url_col,"")), axis=1)

        # Filters
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])
        with c1:
            tickers = sorted(df["ticker"].dropna().unique()) if "ticker" in df.columns else []
            sel_tickers = st.multiselect("Tickers", tickers, default=[], key="t14_tickers")
        with c2:
            days = st.slider("Last N days", 1, 60, 14, 1, key="t14_days")
            cutoff = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(days=days)
        with c3:
            smin = float(np.nanmin(pd.to_numeric(df.get("sentiment", pd.Series([np.nan])), errors="coerce")))
            smax = float(np.nanmax(pd.to_numeric(df.get("sentiment", pd.Series([np.nan])), errors="coerce")))
            if not np.isfinite(smin): smin = -1.0
            if not np.isfinite(smax): smax = 1.0
            sel_sent = st.slider("Sentiment range", smin, smax, (smin, smax), 0.01, key="t14_srange")
        with c4:
            kw = st.text_input("Keyword (title/desc)", "", key="t14_kw")

        f = df.copy()
        # Date filter
        if "date" in f.columns:
            f = f[f["date"] >= cutoff]
        # Ticker filter
        if sel_tickers:
            f = f[f["ticker"].isin(sel_tickers)]
        # Sentiment filter
        if "sentiment" in f.columns:
            s = pd.to_numeric(f["sentiment"], errors="coerce")
            f = f[(s >= sel_sent[0]) & (s <= sel_sent[1])]
        # Keyword filter
        if kw.strip():
            kw_l = kw.strip().lower()
            hay = []
            if "title" in f.columns: hay.append(f["title"].astype(str).str.lower())
            if "description" in f.columns: hay.append(f["description"].astype(str).str.lower())
            if hay:
                mask = False
                for h in hay:
                    mask = mask | h.str.contains(re.escape(kw_l), na=False)
                f = f[mask]

        # KPIs
        cL, cM, cR = st.columns(3)
        with cL: st.metric("Rows", len(f))
        with cM:
            if "sentiment" in f.columns:
                st.metric("Avg Sentiment", f"{pd.to_numeric(f['sentiment'], errors='coerce').mean():.2f}")
        with cR:
            if "ticker" in f.columns:
                st.metric("Unique tickers", f["ticker"].nunique())

        # Display table (clickable)
        show_cols = [c for c in ["date","ticker","sentiment","news","description","source","author"] if c in f.columns or c=="news"]
        disp = f[show_cols] if show_cols else f
        st.markdown(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download filtered (CSV)",
            data=f.to_csv(index=False).encode("utf-8"),
            file_name="news_sentiment_filtered.csv",
            mime="text/csv",
            key="t14_dl",
        )

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
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        if "priority" in df.columns:
            pri_order = ["LOW","MEDIUM","HIGH"]
            df["priority"] = pd.Categorical(df["priority"], categories=pri_order, ordered=True)

        col_l, col_r = st.columns([3,2])
        with col_l:
            min_pri = st.selectbox("Minimum priority", options=["LOW","MEDIUM","HIGH"], index=1, key="t15_minpri")
            tickers = sorted(df["ticker"].dropna().unique()) if "ticker" in df.columns else []
            sel_tickers = st.multiselect("Tickers", tickers, default=[], key="t15_tickers")
        with col_r:
            days_back = st.slider("Show last N days", 3, 60, 30, 1, key="t15_days")

        f = df.copy()
        if "priority" in f.columns:
            pri_rank = {"LOW":0,"MEDIUM":1,"HIGH":2}
            f = f[f["priority"].map(pri_rank).fillna(0) >= pri_rank[min_pri]]
        if sel_tickers:
            f = f[f["ticker"].isin(sel_tickers)]
        if "date" in f.columns:
            cutoff = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(days=days_back)
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
            key="t15_dl",
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
            ticker = st.selectbox("Select a ticker", sorted(df["ticker"].unique()), key="t17_ticker")
            filtered = df[df["ticker"] == ticker].sort_values("importance", ascending=False)
            if PLOTLY_OK:
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
        for c in ["profit","stop_loss","take_profit","exit_price","entry_price"]:
            if c not in df.columns:
                df[c] = np.nan
        to_numeric(df, ["profit","stop_loss","take_profit","exit_price","entry_price"])
        df = df[df["profit"].between(-1e12, 1e12)]  # clamp absurd values
        st.metric("Total Trades", len(df))
        if "profit" in df.columns:
            tp_trades = df[df["profit"] > 0]
            sl_trades = df[df["profit"] <= 0]
            st.metric("Avg Profit (TP)", round(tp_trades["profit"].mean(), 2) if not tp_trades.empty else 0.0)
            st.metric("Avg Loss (SL)", round(sl_trades["profit"].mean(), 2) if not sl_trades.empty else 0.0)

# Tab 19 — Sentiment + Signal Fusion (more robust)
with tabs[19]:
    st.subheader("💬 Sentiment + Signal Fusion")
    sig = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    if sig.empty:
        sig = load_csv("signals.csv", RESULTS_DIR)
    sns = load_csv("news_sentiment.csv", RESULTS_DIR)
    if sig.empty or sns.empty:
        st.info("Need both signals_with_rationale.csv (or signals.csv) and news_sentiment.csv.")
    else:
        # signals
        sig = ensure_date(sig, candidates=["date","as_of","timestamp","time","datetime","Date"], normalize=True)

        # if every date is missing, set to today (so we can join to latest news per ticker)
        if sig["date"].isna().all():
            sig["date"] = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

        # ensure numeric columns exist
        for c in ["close","predicted_close"]:
            if c not in sig.columns:
                sig[c] = np.nan

        # try to backfill close from {ticker}.parquet if missing
        sig = backfill_close_from_parquet(sig)

        with np.errstate(divide="ignore", invalid="ignore"):
            sig["delta_pct"] = np.where(
                sig["close"].notna() & sig["predicted_close"].notna(),
                (sig["predicted_close"] - sig["close"]) / sig["close"],
                np.nan
            )

        # news
        if "publishedAt" in sns.columns and "date" not in sns.columns:
            sns["date"] = pd.to_datetime(sns["publishedAt"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        else:
            sns["date"] = pd.to_datetime(sns.get("date"), errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()

        # If either side lacks dates, fall back to a ticker-only join using the latest news per ticker
        if sig["date"].isna().all() or sns["date"].isna().all():
            latest_news = sns.sort_values("date").groupby("ticker").tail(1)
            merged = pd.merge(sig, latest_news, on="ticker", how="left")
        else:
            merged = pd.merge(sig, sns, on=["ticker","date"], how="left")

        tidy_cols = [c for c in [
            "date","ticker","news","close","predicted_close","delta_pct","signal","confidence","rationale",
            "sentiment","url","title","description"
        ] if c in merged.columns]

        if {"title","url"}.issubset(merged.columns):
            merged["news"] = merged.apply(lambda r: make_clickable(r.get("title",""), r.get("url","")), axis=1)

        for c in ("title","url"):
            if c in tidy_cols:
                try: tidy_cols.remove(c)
                except Exception: pass
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
        for c in ["ticker","date","model","close","predicted_close"]:
            if c not in mc.columns:
                mc[c] = np.nan
        to_numeric(mc, ["close","predicted_close"])
        missing = sorted({"ticker","date","model","close","predicted_close"} - set(mc.columns))
        if missing:
            st.warning(f"model_comparison.csv is missing: {missing}")
        else:
            tickers = sorted(mc["ticker"].dropna().unique())
            sel_ticker = st.selectbox("Select ticker", tickers, key="t20_ticker")

            sub = mc[mc["ticker"] == sel_ticker].dropna(subset=["date"]).sort_values("date")
            models = sorted(sub["model"].dropna().unique())
            sel_models = st.multiselect("Select models to compare", models, default=models, key="t20_models")

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

                if PLOTLY_OK:
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
                    mime="text/csv",
                    key="t20_dl",
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
    uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"], key="t21_upl")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["date"])
            if not {"date","close"}.issubset(df.columns):
                st.error("CSV must include 'date' and 'close' columns.")
            else:
                df = df.sort_values("date")
                strategy = st.selectbox("🧠 Choose a Strategy",
                                        ["Moving Average Crossover", "RSI Strategy", "Bollinger Bands"],
                                        key="t21_strategy")

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
                if not MPL_OK:
                    st.info("Matplotlib not installed — `pip install matplotlib` to show chart.")
                else:
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
        buys = cur[cur.get("action","") == "BUY"] if "action" in cur.columns else pd.DataFrame()
        sells = cur[cur.get("action","") == "SELL"] if "action" in cur.columns else pd.DataFrame()
        c1, c2, c3a, c3b = st.columns([1,1,1,1])
        with c1: st.metric("Symbols", n_syms)
        with c2: st.metric("Sum target weights", f"{tw_sum:0.3f}" if np.isfinite(tw_sum) else "—")
        with c3a: st.metric("Total BUY $", f"{sum_safe(buys.get('delta_notional', [])):,.0f}")
        with c3b: st.metric("Total SELL $", f"{sum_safe(sells.get('delta_notional', [])):,.0f}")

        st.write("**Top BUYS by notional**")
        if not buys.empty:
            st.dataframe(buys.sort_values("delta_notional", ascending=False).head(10), use_container_width=True)
        else:
            st.caption("No BUY rows.")
        st.write("**Top SELLS by notional**")
        if not sells.empty:
            st.dataframe(sells.sort_values("delta_notional", ascending=True).head(10), use_container_width=True)
        else:
            st.caption("No SELL rows.")

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
        total_buy = (last.get("orders", {}) or {}).get("total_buy_notional", 0) or 0
        total_sell = (last.get("orders", {}) or {}).get("total_sell_notional", 0) or 0
        uni_size = (last.get("universe", {}) or {}).get("count", None)
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

# ─────────────────────────────────────────────────────────────────────────────
# Global cache-buster + version tag (safe if already defined elsewhere)
# Place this ONCE near the top of your app. Including here with a guard so
# pasting this block alone won't crash if not yet present.
# ─────────────────────────────────────────────────────────────────────────────
try:
    APP_VERSION  # type: ignore[name-defined]
except NameError:
    APP_VERSION = "r25-29_2025-09-26b"

try:
    _TRITON_GLOBAL_CACHE_BUSTER_ADDED  # avoid duplicating sidebar widgets
except NameError:
    _TRITON_GLOBAL_CACHE_BUSTER_ADDED = True
    _cb_col1, _cb_col2 = st.sidebar.columns([1, 1])
    if _cb_col1.button("↻ Rescan data / clear cache", use_container_width=True, key="global_rescan"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.experimental_rerun()
    st.sidebar.caption(f"Build: {APP_VERSION}")

# Helper to optionally bypass cache in these tabs
from pathlib import Path
def _read_csv_uncached(name: str, root) -> pd.DataFrame:
    p = (Path(root) / name) if isinstance(root, (str, Path)) else Path(name)
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# Global cache-buster + version tag (safe if already defined elsewhere)
# ─────────────────────────────────────────────────────────────────────────────
try:
    APP_VERSION  # type: ignore[name-defined]
except NameError:
    APP_VERSION = "r25-29_2025-09-26c"

try:
    _TRITON_GLOBAL_CACHE_BUSTER_ADDED  # avoid duplicating sidebar widgets
except NameError:
    _TRITON_GLOBAL_CACHE_BUSTER_ADDED = True
    _cb_col1, _cb_col2 = st.sidebar.columns([1, 1])
    if _cb_col1.button("↻ Rescan data / clear cache", use_container_width=True, key="global_rescan"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.experimental_rerun()
    st.sidebar.caption(f"Build: {APP_VERSION}")

from pathlib import Path
def _read_csv_uncached(name: str, root) -> pd.DataFrame:
    p = (Path(root) / name) if isinstance(root, (str, Path)) else Path(name)
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

# ──────────────────────────────
# Tab 25 — Equal-Weight Portfolio vs Benchmark (safe indexing)
# ──────────────────────────────
with tabs[25]:
    st.subheader("📚 Equal-Weight Portfolio vs Benchmark")

    # ── helpers
    def _normalize_to_one(levels: pd.Series) -> pd.Series:
        levels = pd.to_numeric(levels, errors="coerce")
        if levels.empty:
            return levels
        first = levels.iloc[0]
        if not np.isfinite(first) or first == 0:
            first = 1.0
        return levels / first

    def _perf_stats_from_levels(levels: pd.Series) -> Dict[str, float]:
        out = {"total_return": np.nan, "cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan}
        if levels is None or levels.empty:
            return out
        levels = pd.to_numeric(levels, errors="coerce").dropna()
        if levels.empty:
            return out

        # daily returns
        rets = levels.pct_change().dropna()
        if not rets.empty:
            avg = float(rets.mean()); vol = float(rets.std())
            sharpe = (avg / vol) * np.sqrt(252) if vol > 0 else np.nan
        else:
            sharpe = np.nan

        # robust total return (works whether or not series is normalized)
        start = float(levels.iloc[0])
        end   = float(levels.iloc[-1])
        total_return = (end / start - 1.0) if (np.isfinite(start) and start != 0) else np.nan

        # CAGR
        days = (levels.index[-1] - levels.index[0]).days if len(levels.index) > 1 else 0
        years = days / 365.25 if days > 0 else np.nan
        if np.isfinite(years) and years > 0 and np.isfinite(start) and start > 0:
            cagr = float((end / start) ** (1 / years) - 1)
        else:
            cagr = np.nan

        # Max drawdown
        roll_max = levels.cummax()
        dd = levels / roll_max - 1.0
        max_dd = float(dd.min()) if not dd.empty else np.nan

        out.update({"total_return": total_return, "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd})
        return out

    # ── load and prepare
    df = load_csv("strategy_vs_market.csv", RESULTS_DIR)
    if df.empty:
        st.info("No strategy_vs_market.csv yet.")
    else:
        parse_dates_inplace(df, ("date",))
        df = df.dropna(subset=["date"]).copy().sort_values(["ticker", "date"])

        has_ret = {"strategy_return", "market_return"}.issubset(df.columns)
        has_cum = {"cumulative_strategy", "cumulative_market"}.issubset(df.columns)

        if has_ret and not has_cum:
            df["strategy_return"] = pd.to_numeric(df["strategy_return"], errors="coerce").fillna(0)
            df["market_return"]   = pd.to_numeric(df["market_return"], errors="coerce").fillna(0)
            df["cumulative_strategy"] = df.groupby("ticker")["strategy_return"].apply(lambda s: (1 + s).cumprod())
            df["cumulative_market"]   = df.groupby("ticker")["market_return"].apply(lambda s: (1 + s).cumprod())
        else:
            to_numeric(df, ["cumulative_strategy", "cumulative_market"])

        sub = (
            df[["date", "ticker", "cumulative_strategy", "cumulative_market"]]
            .dropna(subset=["ticker"])
            .set_index(["date", "ticker"])
            .sort_index()
        )
        sub.index = sub.index.set_names(["date", "ticker"])

        # daily returns
        sub["strat_ret"] = sub.groupby(level=1)["cumulative_strategy"].pct_change().fillna(0)
        sub["mkt_ret"]   = sub.groupby(level=1)["cumulative_market"].pct_change().fillna(0)

        # Controls
        c1, c2, c3 = st.columns([1.4, 1, 1])
        with c1:
            bench_choice = st.selectbox("Benchmark", ["Avg Market (across tickers)", "SPY (ticker)"], key="t25_bench")
        with c2:
            normalize = st.checkbox("Normalize to 1.0 at start", value=True, key="t25_norm")
        with c3:
            show_kpis = st.checkbox("Show KPIs", value=True, key="t25_kpi")

        # Equal-weight portfolio = mean of strategy daily returns across tickers each day
        eq_strat_ret = sub["strat_ret"].groupby(level=0).mean()
        avg_mkt_ret  = sub["mkt_ret"].groupby(level=0).mean()
        dates_index  = eq_strat_ret.index

        # Choose benchmark series
        if bench_choice.startswith("SPY") and ("SPY" in df.get("ticker", pd.Series(dtype=str)).astype(str).unique()):
            spy = df[df["ticker"] == "SPY"].copy().sort_values("date")
            if has_ret and not has_cum:
                spy["bench_ret"] = pd.to_numeric(spy["market_return"], errors="coerce").fillna(0)
            else:
                to_numeric(spy, ["cumulative_market"])
                spy["bench_ret"] = spy["cumulative_market"].pct_change().fillna(0)
            bench_ret = spy.set_index("date")["bench_ret"].reindex(dates_index).fillna(0)
        else:
            bench_ret = avg_mkt_ret.reindex(dates_index).fillna(0)

        # Build cumulative curves
        eq_strat = (1 + eq_strat_ret.fillna(0)).cumprod()
        bench    = (1 + bench_ret.fillna(0)).cumprod()

        if normalize:
            eq_strat = _normalize_to_one(eq_strat)
            bench    = _normalize_to_one(bench)

        if show_kpis:
            s_stats = _perf_stats_from_levels(eq_strat.dropna())
            b_stats = _perf_stats_from_levels(bench.dropna())
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            with k1: st.metric("Portfolio Total", f"{s_stats['total_return']:.1%}" if np.isfinite(s_stats['total_return']) else "—")
            with k2: st.metric("Portfolio CAGR",  f"{s_stats['cagr']:.1%}"  if np.isfinite(s_stats['cagr'])  else "—")
            with k3: st.metric("Portfolio Sharpe",f"{s_stats['sharpe']:.2f}" if np.isfinite(s_stats['sharpe']) else "—")
            with k4: st.metric("Bench Total",     f"{b_stats['total_return']:.1%}" if np.isfinite(b_stats['total_return']) else "—")
            with k5: st.metric("Bench CAGR",      f"{b_stats['cagr']:.1%}"  if np.isfinite(b_stats['cagr'])  else "—")
            with k6: st.metric("Bench MaxDD",     f"{b_stats['max_dd']:.1%}" if np.isfinite(b_stats['max_dd']) else "—")

        if not PLOTLY_OK:
            st.warning("Plotly not installed — `pip install plotly` for charts.")
        else:
            plot_df = pd.DataFrame({"Equal-Weight Portfolio": eq_strat, "Benchmark": bench}).dropna(how="all")
            fig = go.Figure()
            if plot_df["Equal-Weight Portfolio"].notna().any():
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Equal-Weight Portfolio"], name="Equal-Weight Portfolio", mode="lines"))
            if plot_df["Benchmark"].notna().any():
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Benchmark"], name="Benchmark", mode="lines"))
            ttl = "Equal-Weight Portfolio vs Benchmark" + (" (normalized)" if normalize else "")
            fig.update_layout(title=ttl, xaxis_title="Date", yaxis_title="Cumulative Level")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📎 Ticker attribution (avg daily return over period)"):
            attr = sub["strat_ret"].groupby(level=1).mean().sort_values(ascending=False)
            st.dataframe(attr.reset_index().rename(columns={"strat_ret": "avg_daily_ret"}), use_container_width=True)


# ──────────────────────────────
# Tab 26 — Smart-Weight Portfolio vs Benchmark
# ──────────────────────────────
with tabs[26]:
    st.subheader("🧮 Smart-Weight Portfolio vs Benchmark")

    # ——— Local helpers (guarded so order never breaks) ———
    if "ema_smooth_wide" not in globals():
        def ema_smooth_wide(W: pd.DataFrame, span: int) -> pd.DataFrame:
            if W is None or W.empty or span <= 1:
                return W
            return W.ewm(span=span, adjust=False, min_periods=1).mean()

    if "stabilize_weights" not in globals():
        def stabilize_weights(scores_wide: pd.DataFrame, max_pct: float) -> pd.DataFrame:
            """Row-normalize nonnegative scores, cap each column at max_pct, renormalize."""
            W = scores_wide.copy()
            W = W.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            W[W < 0] = 0.0
            row_sum = W.sum(axis=1)
            eq = pd.Series(1.0 / max(len(W.columns), 1), index=W.columns)
            for i in W.index:
                s = float(row_sum.loc[i])
                W.loc[i] = eq if (s <= 0 or not np.isfinite(s)) else W.loc[i] / s
            cap = float(max_pct)
            for i in W.index:
                w = W.loc[i].clip(lower=0.0)
                over = w > cap
                if over.any():
                    capped = w.copy(); capped[over] = cap
                    rem = 1.0 - float(capped.sum())
                    if rem > 1e-12:
                        room = (cap - capped).clip(lower=0.0)
                        room_sum = float(room.sum())
                        capped += (rem * (room / room_sum)) if room_sum > 0 else (rem / len(capped))
                    total = float(capped.sum())
                    W.loc[i] = capped / (total if total > 0 else 1.0)
                else:
                    total = float(w.sum())
                    W.loc[i] = w / (total if total > 0 else 1.0)
            return W

    def _daily_turnover(W: pd.DataFrame) -> pd.Series:
        if W is None or W.empty:
            return pd.Series(dtype=float)
        dW = W.diff().abs()
        return 0.5 * dW.sum(axis=1)

    # --- Load signals we’ll use to derive raw weights
    sig_src = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    if sig_src.empty:
        sig_src = load_csv("signals.csv", RESULTS_DIR)

    if sig_src.empty or "ticker" not in sig_src.columns:
        st.info("No signals found for Smart-Weight tab.")
    else:
        # Clean dates & numeric score fields
        sig_src = ensure_date(
            sig_src,
            candidates=["date", "as_of", "timestamp", "time", "datetime", "Date"],
            normalize=True,
        )
        to_numeric(sig_src, ["confidence", "total_score"])

        # --- UI
        c1, c2, c3 = st.columns(3)
        with c1:
            scheme = st.selectbox(
                "Weighting scheme",
                ["Confidence (daily)", "Total Score (daily)"],
                index=0,
                key="t26_scheme",
                help="Which signal column to convert to weights each day."
            )
        with c2:
            benchmark_choice = st.selectbox(
                "Benchmark",
                ["Avg Market (across tickers)"],
                index=0,
                key="t26_bench",
                help="Reference curve for comparison."
            )
        with c3:
            normalize_curves = st.checkbox(
                "Normalize to 1.0 at start", value=True, key="t26_norm",
                help="Rescales both curves so they start at 1.0."
            )

        max_pct = st.slider("Max weight per ticker (%)", 1, 50, 15, 1, key="t26_maxw") / 100.0
        lookback = st.slider("Sharpe lookback (days)", 10, 120, 30, 5, key="t26_lookback")
        alpha = st.slider("Blend α (smart share)", 0.0, 1.0, 0.30, 0.05, key="t26_alpha")
        cost_bps = st.slider("Trading cost (bps per $ traded)", 0, 50, 5, 1, key="t26_cost") / 1e4
        ema_span = st.slider("Signal smoothing (EMA days)", 1, 10, 1, 1, key="t26_ema")
        show_kpis = st.checkbox("Show KPIs", value=True, key="t26_kpis")

        # Choose the signal column
        score_col = "confidence" if "Confidence" in scheme else "total_score"
        if score_col not in sig_src.columns:
            st.warning(f"'{score_col}' not present; falling back to 'confidence'.")
            score_col = "confidence"
            if score_col not in sig_src.columns:
                sig_src[score_col] = np.nan

        # Pivot to wide: rows=date, cols=ticker, values=score
        sc = sig_src[["date", "ticker", score_col]].dropna(subset=["date", "ticker"])
        sc["ticker"] = sc["ticker"].astype(str)
        raw_wide = sc.pivot_table(index="date", columns="ticker", values=score_col, aggfunc="last").sort_index()

        # Smooth the raw daily scores before stabilizing -> calmer weights
        raw_wide = ema_smooth_wide(raw_wide, ema_span)

        if raw_wide.shape[1] == 0:
            st.info("No tickers available after processing.")
        else:
            # --- Returns source (per-ticker daily returns) ---
            mkt = load_csv("market_by_ticker.csv", RESULTS_DIR)

            if mkt.empty:
                # Fallback: compute returns from each ticker’s parquet (…/results/{T}.parquet)
                rets_list = []
                for t in raw_wide.columns:
                    pq = RESULTS_DIR / f"{t}.parquet"
                    px = load_parquet(pq)
                    if px.empty or "date" not in px.columns or "close" not in px.columns:
                        continue
                    parse_dates_inplace(px, ("date",))
                    px = px.dropna(subset=["date", "close"]).sort_values("date")
                    px["ret"] = pd.to_numeric(px["close"], errors="coerce").pct_change()
                    tmp = px[["date", "ret"]].copy()
                    tmp["ticker"] = t
                    rets_list.append(tmp)
                mkt = pd.concat(rets_list, ignore_index=True) if rets_list else pd.DataFrame()

            if mkt.empty or not {"date", "ticker"}.issubset(mkt.columns):
                st.info("Could not find per-ticker market returns to build portfolio.")
            else:
                if "ret" not in mkt.columns:
                    cand = next((c for c in ["return", "daily_return", "market_return"] if c in mkt.columns), None)
                    if cand:
                        mkt["ret"] = pd.to_numeric(mkt[cand], errors="coerce")
                mkt = mkt[["date", "ticker", "ret"]].copy()
                mkt["ticker"] = mkt["ticker"].astype(str)
                mkt["ret"] = pd.to_numeric(mkt["ret"], errors="coerce").fillna(0.0)
                mkt["date"] = pd.to_datetime(mkt["date"], errors="coerce", utc=True).dt.tz_localize(None)

                R = mkt.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last").sort_index()

                # Align universe
                common_cols = sorted(set(raw_wide.columns) & set(R.columns))
                if not common_cols:
                    st.info("No overlap between weights universe and returns universe.")
                else:
                    def build_portfolio(max_cap: float, a: float):
                        W_s = stabilize_weights(raw_wide[common_cols], max_cap)
                        W_e = pd.DataFrame(np.full((len(W_s.index), len(common_cols)), 1.0 / len(common_cols)),
                                           index=W_s.index, columns=common_cols)
                        Wf = a * W_s + (1.0 - a) * W_e
                        Wf = Wf.div(Wf.sum(axis=1), axis=0)

                        idx = Wf.index.intersection(R.index)
                        Wf = Wf.loc[idx, common_cols]
                        Rt = R.loc[idx, common_cols]

                        gross_ret = (Wf * Rt).sum(axis=1).fillna(0.0)
                        tvr = _daily_turnover(Wf).reindex(idx).fillna(0.0)
                        cost = cost_bps * tvr
                        net_ret = gross_ret - cost

                        bench_ret = Rt.mean(axis=1).fillna(0.0)

                        port_lvl_g = (1.0 + gross_ret).cumprod()
                        port_lvl_n = (1.0 + net_ret).cumprod()
                        bench_lvl  = (1.0 + bench_ret).cumprod()
                        if normalize_curves:
                            port_lvl_g = normalize_to_one(port_lvl_g)
                            port_lvl_n = normalize_to_one(port_lvl_n)
                            bench_lvl  = normalize_to_one(bench_lvl)
                        stats_g = perf_stats_from_levels(port_lvl_g)
                        stats_n = perf_stats_from_levels(port_lvl_n)
                        return Wf, port_lvl_g, port_lvl_n, bench_lvl, tvr, stats_g, stats_n

                    W, port_lvl_g, port_lvl_n, bench_lvl, tvr, ps_g, ps_n = build_portfolio(max_pct, alpha)
                    bs = perf_stats_from_levels(bench_lvl)

                    if show_kpis:
                        k1, k2, k3, k4, k5, k6 = st.columns(6)
                        with k1: st.metric("Net Total",   f"{ps_n['total_return']:.1%}" if np.isfinite(ps_n['total_return']) else "—")
                        with k2: st.metric("Net CAGR",    f"{ps_n['cagr']:.1%}"          if np.isfinite(ps_n['cagr'])          else "—")
                        with k3: st.metric("Net Sharpe",  f"{ps_n['sharpe']:.2f}"        if np.isfinite(ps_n['sharpe'])        else "—")
                        with k4: st.metric("Bench Total", f"{bs['total_return']:.1%}"    if np.isfinite(bs['total_return'])    else "—")
                        with k5: st.metric("Bench CAGR",  f"{bs['cagr']:.1%}"            if np.isfinite(bs['cagr'])            else "—")
                        with k6:
                            avg_mo_tvr = float(tvr.resample("M").mean().mean()) if not tvr.empty else np.nan
                            st.metric("Avg Monthly Turnover", f"{avg_mo_tvr:.1%}" if np.isfinite(avg_mo_tvr) else "—")

                    if not PLOTLY_OK:
                        st.warning("Plotly not installed — `pip install plotly` for charts.")
                    else:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=port_lvl_n.index,  y=port_lvl_n.values,  name="Portfolio (Net)",   mode="lines"))
                        fig.add_trace(go.Scatter(x=port_lvl_g.index,  y=port_lvl_g.values,  name="Portfolio (Gross)", mode="lines", opacity=0.55))
                        fig.add_trace(go.Scatter(x=bench_lvl.index,   y=bench_lvl.values,   name="Benchmark",         mode="lines"))
                        ttl = "Smart-Weight Portfolio vs Benchmark" + (" (normalized)" if normalize_curves else "")
                        fig.update_layout(title=ttl, xaxis_title="Date", yaxis_title="Cumulative Level")
                        st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Average weights by ticker (over period)"):
                        avg_w = W.mean(axis=0).sort_values(ascending=False)
                        st.dataframe(avg_w.rename("avg_weight").to_frame(), use_container_width=True)

                    with st.expander("🗺️ Weights heatmap (last ~120 days)"):
                        if PLOTLY_OK and not W.empty:
                            W_hm = W.tail(120)
                            hm = go.Figure(data=go.Heatmap(
                                z=W_hm.T.values,
                                x=W_hm.index.astype(str),
                                y=W_hm.columns.astype(str),
                                colorbar=dict(title="Weight")
                            ))
                            hm.update_layout(title="Daily Weights Heatmap", xaxis_title="Date", yaxis_title="Ticker")
                            st.plotly_chart(hm, use_container_width=True)
                        else:
                            st.caption("No weights to display.")

                    with st.expander("🔎 Quick grid search (max cap × α)"):
                        do_grid = st.checkbox("Run grid search near current params", value=False, key="t26_grid")
                        if do_grid:
                            caps = sorted(set([max(0.01, min(0.50, round(x, 2))) for x in [max_pct, max_pct-0.05, max_pct-0.02, max_pct+0.02, max_pct+0.05, 0.10, 0.15, 0.20]]))
                            alphas = sorted(set([max(0.0, min(1.0, round(x, 2))) for x in [alpha, 0.20, 0.30, 0.40, 0.50]]))
                            rows, best = [], None
                            for c in caps:
                                for a in alphas:
                                    try:
                                        _, _, p_lvl_n, _, _, _, stats_n = build_portfolio(c, a)
                                        sharpe_n = stats_n.get("sharpe", np.nan)
                                        rows.append({"max_cap": c, "alpha": a, "total_return_net": stats_n.get("total_return", np.nan), "cagr_net": stats_n.get("cagr", np.nan), "sharpe_net": sharpe_n})
                                        if best is None or (np.isfinite(sharpe_n) and sharpe_n > best["sharpe_net"]):
                                            best = {"max_cap": c, "alpha": a, "stats_n": stats_n, "curve_n": p_lvl_n}
                                    except Exception:
                                        pass
                            if rows:
                                grid_df = pd.DataFrame(rows).sort_values("sharpe_net", ascending=False)
                                st.dataframe(grid_df, use_container_width=True)
                                st.caption("Top by Net Sharpe (portfolio).")
                                if not PLOTLY_OK:
                                    st.info(f"Best combo: cap={best['max_cap']:.2f}, α={best['alpha']:.2f}, Net Sharpe={best['stats_n']['sharpe']:.2f}")
                                else:
                                    st.markdown(f"**Best combo (net):** max cap **{best['max_cap']:.2f}**, α **{best['alpha']:.2f}** — Sharpe **{best['stats_n']['sharpe']:.2f}**, CAGR **{best['stats_n']['cagr']:.1%}**")
                                    bf = go.Figure()
                                    bf.add_trace(go.Scatter(x=best["curve_n"].index, y=best["curve_n"].values, name="Best Smart-Weight (Net)", mode="lines"))
                                    bf.add_trace(go.Scatter(x=bench_lvl.index, y=bench_lvl.values, name="Benchmark", mode="lines"))
                                    bf.update_layout(title="Best grid result vs Benchmark" + (" (normalized)" if normalize_curves else ""), xaxis_title="Date", yaxis_title="Cumulative Level")
                                    st.plotly_chart(bf, use_container_width=True)

                    with st.expander("⬇️ Export data"):
                        curves = pd.DataFrame({
                            "date": port_lvl_n.index,
                            "portfolio_net": port_lvl_n.values,
                            "portfolio_gross": port_lvl_g.reindex(port_lvl_n.index).values,
                            "benchmark": bench_lvl.reindex(port_lvl_n.index).values,
                        }).set_index("date")
                        daily = pd.DataFrame({
                            "date": W.index,
                            "turnover": _daily_turnover(W).reindex(W.index).values
                        }).set_index("date")
                        st.download_button("Download equity curves (CSV)", curves.to_csv().encode("utf-8"), "tab26_curves.csv", "text/csv", key="t26_dl_curves")
                        st.download_button("Download turnover (CSV)", daily.to_csv().encode("utf-8"), "tab26_turnover.csv", "text/csv", key="t26_dl_tvr")

# ──────────────────────────────
# Tab 27 — Confidence Calibration (no 'fwd_ret' dependency)
# ──────────────────────────────
with tabs[27]:
    st.subheader("🧪 Confidence Calibration")

    # -- helpers
    def _safe_decile_map(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() == 0:
            return pd.Series(np.nan, index=s.index)
        qs = np.nanquantile(s.values, np.linspace(0, 1, 11))
        def _to_dec(x):
            if not np.isfinite(x):
                return np.nan
            d = int(np.searchsorted(qs, x, side="right"))
            return min(max(d, 1), 10)
        return s.map(_to_dec)

    def _summarize_calibration(df: pd.DataFrame, conf_col="confidence", ret_col="target_ret"):
        out = pd.DataFrame()
        if df.empty or conf_col not in df.columns or ret_col not in df.columns:
            return out
        d = df[[conf_col, ret_col]].copy()
        d["decile"] = _safe_decile_map(d[conf_col])
        d = d.dropna(subset=["decile"])
        grp = d.groupby("decile")
        out = pd.DataFrame({
            "count": grp.size(),
            "avg_return": grp[ret_col].mean(),
            "hitrate_%": grp.apply(lambda g: (g[ret_col] > 0).mean() * 100.0)
        }).reset_index()
        out["decile"] = out["decile"].astype(int)
        out = out.sort_values("decile")
        return out

    svs = load_csv("strategy_vs_market.csv", RESULTS_DIR)
    if svs.empty:
        st.info("No strategy_vs_market.csv yet.")
        st.stop()

    parse_dates_inplace(svs, ("date",))
    svs = svs.dropna(subset=["date", "ticker"]).copy().sort_values(["ticker", "date"])

    has_ret = {"strategy_return", "market_return"}.issubset(svs.columns)
    has_cum = {"cumulative_strategy", "cumulative_market"}.issubset(svs.columns)
    if has_ret and not has_cum:
        svs["strategy_return"] = pd.to_numeric(svs["strategy_return"], errors="coerce").fillna(0)
        svs["market_return"]   = pd.to_numeric(svs["market_return"], errors="coerce").fillna(0)
        svs["cumulative_strategy"] = svs.groupby("ticker")["strategy_return"].apply(lambda s: (1 + s).cumprod())
        svs["cumulative_market"]   = svs.groupby("ticker")["market_return"].apply(lambda s: (1 + s).cumprod())
    else:
        to_numeric(svs, ["cumulative_strategy", "cumulative_market"])

    panel = (
        svs[["date", "ticker", "cumulative_strategy", "cumulative_market"]]
        .set_index(["date", "ticker"])
        .sort_index()
    )
    panel["strat_ret"] = panel.groupby(level=1)["cumulative_strategy"].pct_change().fillna(0)
    panel["mkt_ret"]   = panel.groupby(level=1)["cumulative_market"].pct_change().fillna(0)

    panel["fwd_strat_ret"] = panel.groupby(level=1)["strat_ret"].shift(-1)
    panel["fwd_mkt_ret"]   = panel.groupby(level=1)["mkt_ret"].shift(-1)

    sig = load_csv("signals_with_rationale.csv", RESULTS_DIR)
    if sig.empty:
        sig = load_csv("signals.csv", RESULTS_DIR)

    if sig.empty:
        st.info("Need signals_with_rationale.csv (or signals.csv) with at least [date, ticker, confidence].")
        st.stop()

    sig = sig.copy()
    sig = ensure_date(sig, candidates=["date", "as_of", "timestamp", "time", "datetime", "Date"], normalize=False)
    sig = sig.dropna(subset=["date", "ticker"])
    to_numeric(sig, ["confidence"])
    sig_idxed = sig.set_index(["date", "ticker"]).sort_index()

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        target = st.selectbox("Calibration target (what confidence tries to predict)",
                              ["Next-day Market Return", "Next-day Strategy Return"], 0, key="t27_target")
    with c2:
        min_obs = st.slider("Min observations per decile", 10, 500, 30, 10, key="t27_minobs")
    with c3:
        show_scatter = st.checkbox("Show scatter (confidence vs next-day return)", True, key="t27_scatter")

    target_series = panel["fwd_mkt_ret"] if target.startswith("Next-day Market") else panel["fwd_strat_ret"]

    merged = sig_idxed.join(target_series.rename("target_ret"), how="left").dropna(subset=["confidence", "target_ret"])
    if merged.empty:
        st.info("After alignment, no rows found (check dates/tickers overlap).")
        st.stop()

    dec_tbl = _summarize_calibration(merged[["confidence", "target_ret"]])
    if dec_tbl.empty:
        st.info("No decile stats available."); st.stop()

    dec_tbl_f = dec_tbl[dec_tbl["count"] >= min_obs]
    if dec_tbl_f.empty:
        st.warning("All deciles are under the min observation threshold; showing raw deciles.")
        dec_tbl_f = dec_tbl

    st.markdown("**Confidence deciles vs next-day return**")
    st.dataframe(dec_tbl_f, use_container_width=True)

    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dec_tbl_f["decile"], y=dec_tbl_f["avg_return"], name="Avg next-day return"))
        fig.add_trace(go.Scatter(x=dec_tbl_f["decile"], y=dec_tbl_f["hitrate_%"], name="Hitrate (%)",
                                 mode="lines+markers", yaxis="y2"))
        fig.update_layout(title="Calibration: Confidence deciles vs next-day return / hitrate",
                          xaxis_title="Confidence decile (1=low … 10=high)",
                          yaxis=dict(title="Avg next-day return"),
                          yaxis2=dict(title="Hitrate (%)", overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)

    if show_scatter and PLOTLY_OK:
        samp = merged.sample(min(len(merged), 4000), random_state=42) if len(merged) > 4000 else merged
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=samp["confidence"], y=samp["target_ret"], mode="markers", name="obs", opacity=0.45))
        fig2.update_layout(title="Confidence vs next-day return (sampled)", xaxis_title="Confidence", yaxis_title="Next-day return")
        st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────
# Tab 28 — Confidence-Filtered Portfolio vs Benchmark
# ──────────────────────────────
with tabs[28]:
    st.subheader("🧪 Confidence-Filtered Portfolio vs Benchmark")

    # helpers (guarded)
    if "ema_smooth_wide" not in globals():
        def ema_smooth_wide(W: pd.DataFrame, span: int) -> pd.DataFrame:
            if W is None or W.empty or span <= 1:
                return W
            return W.ewm(span=span, adjust=False, min_periods=1).mean()
    if "enforce_min_hold" not in globals():
        def enforce_min_hold(mask_df: pd.DataFrame, min_days: int) -> pd.DataFrame:
            if mask_df is None or mask_df.empty or min_days <= 1:
                return mask_df
            out = mask_df.copy()
            for k in range(1, min_days):
                out |= mask_df.shift(k).fillna(False)
            return out

    def _daily_turnover(W: pd.DataFrame) -> pd.Series:
        if W is None or W.empty:
            return pd.Series(dtype=float)
        dW = W.diff().abs()
        return 0.5 * dW.sum(axis=1)

    svs = load_csv("strategy_vs_market.csv", RESULTS_DIR)
    if svs.empty:
        st.info("No strategy_vs_market.csv yet.")
    else:
        parse_dates_inplace(svs, ("date",))
        svs = svs.dropna(subset=["date", "ticker"]).copy().sort_values(["ticker", "date"])

        has_ret = {"strategy_return", "market_return"}.issubset(svs.columns)
        has_cum = {"cumulative_strategy", "cumulative_market"}.issubset(svs.columns)
        if has_ret and not has_cum:
            svs["strategy_return"] = pd.to_numeric(svs["strategy_return"], errors="coerce").fillna(0)
            svs["market_return"]   = pd.to_numeric(svs["market_return"], errors="coerce").fillna(0)
            svs["cumulative_strategy"] = svs.groupby("ticker")["strategy_return"].apply(lambda s: (1 + s).cumprod())
            svs["cumulative_market"]   = svs.groupby("ticker")["market_return"].apply(lambda s: (1 + s).cumprod())
        else:
            to_numeric(svs, ["cumulative_strategy", "cumulative_market"])

        panel = (
            svs[["date", "ticker", "cumulative_strategy", "cumulative_market"]]
            .set_index(["date", "ticker"])
            .sort_index()
        )
        panel["strat_ret"] = panel.groupby(level=1)["cumulative_strategy"].pct_change().fillna(0)
        panel["mkt_ret"]   = panel.groupby(level=1)["cumulative_market"].pct_change().fillna(0)
        panel["fwd_strat_ret"] = panel.groupby(level=1)["strat_ret"].shift(-1)
        panel["fwd_mkt_ret"]   = panel.groupby(level=1)["mkt_ret"].shift(-1)

        sig = load_csv("signals_with_rationale.csv", RESULTS_DIR)
        if sig.empty:
            sig = load_csv("signals.csv", RESULTS_DIR)

        if sig.empty:
            st.info("Need signals_with_rationale.csv (or signals.csv) with at least [date, ticker, confidence].")
        else:
            sig = ensure_date(sig, candidates=["date","as_of","timestamp","time","datetime","Date"], normalize=False)
            sig = sig.dropna(subset=["date","ticker"]).copy()
            to_numeric(sig, ["confidence"])
            sidx = sig.set_index(["date","ticker"]).sort_index()

            c1, c2, c3, c4 = st.columns([1.4, 1.1, 1.1, 1.2])
            with c1:
                bench_choice = st.selectbox("Benchmark", ["Avg Market (across tickers)", "SPY (ticker)"], 0, key="t28_bench")
            with c2:
                thr = st.slider("Confidence threshold (≥)", 0.0, 1.0, 0.70, 0.01, key="t28_thr")
            with c3:
                cost_bps = st.slider("Trading cost (bps per $ traded)", 0, 50, 5, 1, key="t28_cost") / 1e4
            with c4:
                normalize = st.checkbox("Normalize curves to 1.0 at start", True, key="t28_norm")
            ema_span = st.slider("Confidence smoothing (EMA days)", 1, 10, 1, 1, key="t28_ema")
            min_hold = st.slider("Min hold days", 1, 10, 1, 1, key="t28_hold")
            show_kpis = st.checkbox("Show KPIs", True, key="t28_kpi")

            c_mat = sidx["confidence"].unstack("ticker").sort_index().fillna(0.0)
            c_mat = ema_smooth_wide(c_mat, ema_span)
            mask_wide = enforce_min_hold((c_mat >= thr), min_hold)

            r = panel["fwd_strat_ret"]
            w_mat = mask_wide
            if w_mat is None or w_mat.empty:
                port_ret_gross = pd.Series(0.0, index=panel.index.get_level_values(0).unique())
                W = pd.DataFrame(index=panel.index.get_level_values(0).unique())
            else:
                n = w_mat.sum(axis=1)
                eq_w = w_mat.div(n.replace(0, np.nan), axis=0).fillna(0.0)
                W = eq_w.copy()
                w_long = eq_w.stack(dropna=False).fillna(0.0)
                w_long.index.set_names(panel.index.names, inplace=True)
                port_ret_gross = (w_long * r.reindex(w_long.index).fillna(0.0)).groupby(level=0).sum()

            tvr = _daily_turnover(W)
            cost = cost_bps * tvr.reindex(port_ret_gross.index).fillna(0.0)
            port_ret_net = port_ret_gross - cost

            avg_mkt_ret = panel["mkt_ret"].groupby(level=0).mean()
            dates_index = port_ret_net.index
            tickers_unique = svs.get("ticker", pd.Series(dtype=str)).astype(str).unique()
            if bench_choice.startswith("SPY") and ("SPY" in tickers_unique):
                spy = svs[svs["ticker"] == "SPY"].copy().sort_values("date")
                if has_ret and not has_cum:
                    spy["bench_ret"] = pd.to_numeric(spy["market_return"], errors="coerce").fillna(0)
                else:
                    to_numeric(spy, ["cumulative_market"])
                    spy["bench_ret"] = spy["cumulative_market"].pct_change().fillna(0)
                bench_ret = spy.set_index("date")["bench_ret"].reindex(dates_index).fillna(0)
            else:
                bench_ret = avg_mkt_ret.reindex(dates_index).fillna(0)

            port_curve_net  = (1 + port_ret_net.fillna(0)).cumprod()
            port_curve_gross= (1 + port_ret_gross.fillna(0)).cumprod()
            bench_curve     = (1 + bench_ret.fillna(0)).cumprod()
            if normalize:
                port_curve_net   = normalize_to_one(port_curve_net)
                port_curve_gross = normalize_to_one(port_curve_gross)
                bench_curve      = normalize_to_one(bench_curve)

            if show_kpis:
                def _ps(levels: pd.Series) -> Dict[str, float]:
                    if levels.empty:
                        return {"total_return":np.nan,"cagr":np.nan,"sharpe":np.nan,"max_dd":np.nan}
                    rets = levels.pct_change().dropna()
                    avg, vol = (float(rets.mean()), float(rets.std())) if not rets.empty else (np.nan, np.nan)
                    sharpe = (avg / vol) * np.sqrt(252) if (np.isfinite(avg) and vol and vol>0) else np.nan
                    start, end = float(levels.iloc[0]), float(levels.iloc[-1])
                    total = (end/start - 1.0) if (np.isfinite(start) and start>0) else np.nan
                    days = (levels.index[-1] - levels.index[0]).days if len(levels.index)>1 else 0
                    years = days/365.25 if days>0 else np.nan
                    cagr = (end/start)**(1/years)-1 if (np.isfinite(years) and years>0 and start>0) else np.nan
                    dd = (levels/levels.cummax()-1.0) if not levels.empty else pd.Series(dtype=float)
                    maxdd = float(dd.min()) if not dd.empty else np.nan
                    return {"total_return":total,"cagr":cagr,"sharpe":sharpe,"max_dd":maxdd}
                s_stats = _ps(port_curve_net.dropna())
                b_stats = _ps(bench_curve.dropna())
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                with k1: st.metric("Net Total",   f"{s_stats['total_return']:.1%}" if np.isfinite(s_stats['total_return']) else "—")
                with k2: st.metric("Net CAGR",    f"{s_stats['cagr']:.1%}"  if np.isfinite(s_stats['cagr'])  else "—")
                with k3: st.metric("Net Sharpe",  f"{s_stats['sharpe']:.2f}" if np.isfinite(s_stats['sharpe']) else "—")
                with k4: st.metric("Bench Total", f"{b_stats['total_return']:.1%}" if np.isfinite(b_stats['total_return']) else "—")
                with k5: st.metric("Bench CAGR",  f"{b_stats['cagr']:.1%}"  if np.isfinite(b_stats['cagr'])  else "—")
                with k6:
                    avg_mo_tvr = float(tvr.resample("M").mean().mean()) if not tvr.empty else np.nan
                    st.metric("Avg Monthly Turnover", f"{avg_mo_tvr:.1%}" if np.isfinite(avg_mo_tvr) else "—")

            if not PLOTLY_OK:
                st.warning("Plotly not installed — `pip install plotly` for charts.")
            else:
                plot_df = pd.DataFrame({
                    "Portfolio (Net)":   port_curve_net,
                    "Portfolio (Gross)": port_curve_gross.reindex(port_curve_net.index),
                    "Benchmark":         bench_curve.reindex(port_curve_net.index)
                }).dropna(how="all")
                fig = go.Figure()
                if plot_df["Portfolio (Net)"].notna().any():
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Portfolio (Net)"],   name="Portfolio (Net)",   mode="lines"))
                if plot_df["Portfolio (Gross)"].notna().any():
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Portfolio (Gross)"], name="Portfolio (Gross)", mode="lines", opacity=0.55))
                if plot_df["Benchmark"].notna().any():
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Benchmark"], name="Benchmark", mode="lines"))
                ttl = f"Confidence ≥ {thr:.2f} — Portfolio vs Benchmark" + (" (normalized)" if normalize else "")
                fig.update_layout(title=ttl, xaxis_title="Date", yaxis_title="Cumulative Level")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("🗺️ Weights heatmap (last ~120 days)"):
                if PLOTLY_OK and not W.empty:
                    W_hm = W.tail(120)
                    hm = go.Figure(data=go.Heatmap(
                        z=W_hm.T.values,
                        x=W_hm.index.astype(str),
                        y=W_hm.columns.astype(str),
                        colorbar=dict(title="Weight")
                    ))
                    hm.update_layout(title="Daily Weights Heatmap", xaxis_title="Date", yaxis_title="Ticker")
                    st.plotly_chart(hm, use_container_width=True)
                else:
                    st.caption("No weights to display.")

            with st.expander("⬇️ Export data"):
                curves = pd.DataFrame({
                    "date": port_curve_net.index,
                    "portfolio_net": port_curve_net.values,
                    "portfolio_gross": port_curve_gross.reindex(port_curve_net.index).values,
                    "benchmark": bench_curve.reindex(port_curve_net.index).values,
                }).set_index("date")
                daily = pd.DataFrame({
                    "date": port_ret_gross.index if 'port_ret_gross' in locals() else pd.Index([]),
                    "gross_ret": port_ret_gross.values if 'port_ret_gross' in locals() else [],
                    "net_ret": port_ret_net.reindex(port_ret_gross.index).values if 'port_ret_gross' in locals() else [],
                    "bench_ret": bench_ret.reindex(port_ret_gross.index).values if 'port_ret_gross' in locals() else [],
                    "turnover": tvr.reindex(port_ret_gross.index).values if 'port_ret_gross' in locals() else [],
                    "cost_applied": (port_ret_net - port_ret_gross).reindex(port_ret_gross.index).values if 'port_ret_gross' in locals() else [],
                }).set_index("date")
                st.download_button("Download equity curves (CSV)", curves.to_csv().encode("utf-8"), "tab28_curves.csv", "text/csv", key="t28_dl_curves")
                if not daily.empty:
                    st.download_button("Download daily returns (CSV)", daily.to_csv().encode("utf-8"), "tab28_daily.csv", "text/csv", key="t28_dl_daily")

# ──────────────────────────────
# Tab 29 — Confidence × Sharpe Portfolio vs Benchmark
# ──────────────────────────────
with tabs[29]:
    st.subheader("📊 Confidence × Sharpe Portfolio vs Benchmark")

    # helpers (guarded)
    if "ema_smooth_wide" not in globals():
        def ema_smooth_wide(W: pd.DataFrame, span: int) -> pd.DataFrame:
            if W is None or W.empty or span <= 1:
                return W
            return W.ewm(span=span, adjust=False, min_periods=1).mean()
    if "enforce_min_hold" not in globals():
        def enforce_min_hold(mask_df: pd.DataFrame, min_days: int) -> pd.DataFrame:
            if mask_df is None or mask_df.empty or min_days <= 1:
                return mask_df
            out = mask_df.copy()
            for k in range(1, min_days):
                out |= mask_df.shift(k).fillna(False)
            return out

    def _daily_turnover_from_series(weights_long: pd.Series) -> pd.Series:
        if weights_long is None or weights_long.empty:
            return pd.Series(dtype=float)
        W = weights_long.unstack("ticker").fillna(0.0).sort_index()
        dW = W.diff().abs()
        return 0.5 * dW.sum(axis=1)

    svs = load_csv("strategy_vs_market.csv", RESULTS_DIR)
    if svs.empty:
        st.info("No strategy_vs_market.csv yet.")
    else:
        parse_dates_inplace(svs, ("date",))
        svs = svs.dropna(subset=["date", "ticker"]).copy().sort_values(["ticker", "date"])

        has_ret = {"strategy_return", "market_return"}.issubset(svs.columns)
        has_cum = {"cumulative_strategy", "cumulative_market"}.issubset(svs.columns)
        if has_ret and not has_cum:
            svs["strategy_return"] = pd.to_numeric(svs["strategy_return"], errors="coerce").fillna(0)
            svs["market_return"]   = pd.to_numeric(svs["market_return"], errors="coerce").fillna(0)
            svs["cumulative_strategy"] = svs.groupby("ticker")["strategy_return"].apply(lambda s: (1 + s).cumprod())
            svs["cumulative_market"]   = svs.groupby("ticker")["market_return"].apply(lambda s: (1 + s).cumprod())
        else:
            to_numeric(svs, ["cumulative_strategy", "cumulative_market"])

        panel = (
            svs[["date", "ticker", "cumulative_strategy", "cumulative_market"]]
            .set_index(["date", "ticker"])
            .sort_index()
        )
        panel["strat_ret"] = panel.groupby(level=1)["cumulative_strategy"].pct_change().fillna(0)
        panel["mkt_ret"]   = panel.groupby(level=1)["cumulative_market"].pct_change().fillna(0)
        panel["fwd_strat_ret"] = panel.groupby(level=1)["strat_ret"].shift(-1)
        panel["fwd_mkt_ret"]   = panel.groupby(level=1)["mkt_ret"].shift(-1)

        sig = load_csv("signals_with_rationale.csv", RESULTS_DIR)
        if sig.empty:
            sig = load_csv("signals.csv", RESULTS_DIR)
        if not sig.empty:
            sig = ensure_date(sig, candidates=["date","as_of","timestamp","time","datetime","Date"], normalize=False)
            sig = sig.dropna(subset=["date","ticker"])
            to_numeric(sig, ["confidence"])
            conf = sig.set_index(["date","ticker"])["confidence"].sort_index()
        else:
            conf = pd.Series(np.nan, index=panel.index)

        c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
        with c1:
            lookback = st.slider("Sharpe lookback (days)", 10, 120, 30, 5, key="t29_lb")
        with c2:
            max_w_pct = st.slider("Max weight per ticker (%)", 1, 50, 15, 1, key="t29_cap")
        with c3:
            cost_bps = st.slider("Trading cost (bps per $ traded)", 0, 50, 5, 1, key="t29_cost") / 1e4

        c4, c5 = st.columns([1.2, 1.2])
        with c4:
            return_source = st.selectbox("Return family used for Sharpe & PnL",
                                         ["Market returns", "Strategy returns"], 0, key="t29_source")
        with c5:
            bench_choice = st.selectbox("Benchmark", ["Avg Market (across tickers)", "SPY (ticker)"], 0, key="t29_bench")
        ema_span = st.slider("Weight smoothing (EMA days)", 1, 10, 1, 1, key="t29_ema")
        min_hold = st.slider("Min hold days (soft)", 1, 10, 1, 1, key="t29_hold")
        normalize = st.checkbox("Normalize to 1.0 at start", True, key="t29_norm")
        show_kpis = st.checkbox("Show KPIs", True, key="t29_kpis")

        if return_source == "Market returns":
            base_ret      = panel["mkt_ret"]
            base_fwd_ret  = panel["fwd_mkt_ret"]
        else:
            base_ret      = panel["strat_ret"]
            base_fwd_ret  = panel["fwd_strat_ret"]

        r_mat = base_ret.unstack("ticker")
        mu = r_mat.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
        sd = r_mat.rolling(lookback, min_periods=max(5, lookback // 3)).std()
        sharpe_mat = (mu / sd.replace(0, np.nan)) * np.sqrt(252)
        sharpe_mat = sharpe_mat.clip(lower=0.0).shift(1).fillna(0.0)

        sharpe = sharpe_mat.stack(dropna=False).fillna(0.0)
        sharpe.index.set_names(panel.index.names, inplace=True)

        c_mat = conf.reindex(panel.index).fillna(0.0).unstack("ticker")
        c_min = c_mat.min(axis=1)
        c_span = (c_mat.max(axis=1) - c_min).replace(0, np.nan)
        conf01 = (c_mat.sub(c_min, axis=0)).div(c_span, axis=0).fillna(0.0).stack(dropna=False)
        conf01.index.set_names(panel.index.names, inplace=True)

        raw_w = (sharpe * conf01).clip(lower=0.0)
        cap = max_w_pct / 100.0
        raw_w = raw_w.clip(upper=cap)

        w_mat = raw_w.unstack("ticker").fillna(0.0)
        w_mat = ema_smooth_wide(w_mat, ema_span)
        if min_hold > 1 and not w_mat.empty:
            pos = w_mat.gt(0)
            pos = enforce_min_hold(pos, min_hold)
            w_mat = w_mat.where(pos, 0.0)

        w_mat = w_mat.clip(lower=0.0, upper=cap)
        row_sums = w_mat.sum(axis=1)
        w_norm = pd.DataFrame(0.0, index=w_mat.index, columns=w_mat.columns)
        pos_rows = row_sums > 0
        if pos_rows.any():
            w_norm.loc[pos_rows] = w_mat.loc[pos_rows].div(row_sums.loc[pos_rows], axis=0)
        if (~pos_rows).any():
            w0 = w_mat.loc[~pos_rows]
            def _eq_row(r):
                n = r.notna().sum()
                return pd.Series(np.where(r.notna(), 1.0 / n, 0.0), index=r.index) if n and n > 0 else pd.Series(0.0, index=r.index)
            w_norm.loc[~pos_rows] = w0.apply(_eq_row, axis=1).fillna(0.0)

        weights = w_norm.stack(dropna=False).fillna(0.0)
        weights.index.set_names(panel.index.names, inplace=True)

        port_ret_gross = (weights * base_fwd_ret.reindex(weights.index).fillna(0.0)).groupby(level=0).sum()

        tvr = _daily_turnover_from_series(weights)
        cost = cost_bps * tvr.reindex(port_ret_gross.index).fillna(0.0)
        port_ret_net = port_ret_gross - cost

        avg_mkt_ret = panel["mkt_ret"].groupby(level=0).mean()
        dates_index = port_ret_net.index
        tickers_unique = svs.get("ticker", pd.Series(dtype=str)).astype(str).unique()
        if bench_choice.startswith("SPY") and ("SPY" in tickers_unique):
            spy = svs[svs["ticker"] == "SPY"].copy().sort_values("date")
            if has_ret and not has_cum:
                spy["bench_ret"] = pd.to_numeric(spy["market_return"], errors="coerce").fillna(0)
            else:
                to_numeric(spy, ["cumulative_market"])
                spy["bench_ret"] = spy["cumulative_market"].pct_change().fillna(0)
            bench_ret = spy.set_index("date")["bench_ret"].reindex(dates_index).fillna(0)
        else:
            bench_ret = avg_mkt_ret.reindex(dates_index).fillna(0)

        port_curve_net  = (1 + port_ret_net.fillna(0)).cumprod()
        port_curve_gross= (1 + port_ret_gross.fillna(0)).cumprod()
        bench_curve     = (1 + bench_ret.fillna(0)).cumprod()
        if normalize:
            port_curve_net   = normalize_to_one(port_curve_net)
            port_curve_gross = normalize_to_one(port_curve_gross)
            bench_curve      = normalize_to_one(bench_curve)

        if show_kpis:
            def _ps(levels: pd.Series) -> Dict[str, float]:
                if levels.empty:
                    return {"total_return":np.nan,"cagr":np.nan,"sharpe":np.nan,"max_dd":np.nan}
                rets = levels.pct_change().dropna()
                avg, vol = (float(rets.mean()), float(rets.std())) if not rets.empty else (np.nan, np.nan)
                sharpe = (avg / vol) * np.sqrt(252) if (np.isfinite(avg) and vol and vol>0) else np.nan
                start, end = float(levels.iloc[0]), float(levels.iloc[-1])
                total = (end/start - 1.0) if (np.isfinite(start) and start>0) else np.nan
                days = (levels.index[-1] - levels.index[0]).days if len(levels.index)>1 else 0
                years = days/365.25 if days>0 else np.nan
                cagr = (end/start)**(1/years)-1 if (np.isfinite(years) and years>0 and start>0) else np.nan
                dd = (levels/levels.cummax()-1.0) if not levels.empty else pd.Series(dtype=float)
                maxdd = float(dd.min()) if not dd.empty else np.nan
                return {"total_return":total,"cagr":cagr,"sharpe":sharpe,"max_dd":maxdd}
            s_stats = _ps(port_curve_net.dropna())
            b_stats = _ps(bench_curve.dropna())

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            with k1: st.metric("Net Total", f"{s_stats['total_return']:.1%}" if np.isfinite(s_stats['total_return']) else "—")
            with k2: st.metric("Net CAGR",  f"{s_stats['cagr']:.1%}"  if np.isfinite(s_stats['cagr'])  else "—")
            with k3: st.metric("Net Sharpe",f"{s_stats['sharpe']:.2f}" if np.isfinite(s_stats['sharpe']) else "—")
            with k4: st.metric("Bench Total", f"{b_stats['total_return']:.1%}" if np.isfinite(b_stats['total_return']) else "—")
            with k5: st.metric("Bench CAGR",  f"{b_stats['cagr']:.1%}"  if np.isfinite(b_stats['cagr'])  else "—")
            with k6:
                avg_mo_tvr = float(tvr.resample("M").mean().mean()) if not tvr.empty else np.nan
                st.metric("Avg Monthly Turnover", f"{avg_mo_tvr:.1%}" if np.isfinite(avg_mo_tvr) else "—")

        if not PLOTLY_OK:
            st.warning("Plotly not installed — `pip install plotly` for charts.")
        else:
            plot_df = pd.DataFrame(
                {"Portfolio (Net)": port_curve_net,
                 "Portfolio (Gross)": port_curve_gross.reindex(port_curve_net.index),
                 "Benchmark": bench_curve.reindex(port_curve_net.index)}
            ).dropna(how="all")
            fig = go.Figure()
            if plot_df["Portfolio (Net)"].notna().any():
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Portfolio (Net)"], name="Portfolio (Net)", mode="lines"))
            if plot_df["Portfolio (Gross)"].notna().any():
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Portfolio (Gross)"], name="Portfolio (Gross)", mode="lines", opacity=0.55))
            if plot_df["Benchmark"].notna().any():
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Benchmark"], name="Benchmark", mode="lines"))
            ttl = "Confidence × Sharpe Portfolio vs Benchmark" + (" (normalized)" if normalize else "")
            fig.update_layout(title=ttl, xaxis_title="Date", yaxis_title="Cumulative Level")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📎 Average weights by ticker (over period)"):
            w_avg = weights.groupby(level=1).mean().sort_values(ascending=False)
            st.dataframe(w_avg.reset_index().rename(columns={0: "avg_weight"}), use_container_width=True)

        with st.expander("🗺️ Weights heatmap (last ~120 days)"):
            if PLOTLY_OK and not weights.empty:
                W_hm = weights.unstack("ticker").fillna(0.0).tail(120)
                hm = go.Figure(data=go.Heatmap(
                    z=W_hm.T.values,
                    x=W_hm.index.astype(str),
                    y=W_hm.columns.astype(str),
                    colorbar=dict(title="Weight")
                ))
                hm.update_layout(title="Daily Weights Heatmap", xaxis_title="Date", yaxis_title="Ticker")
                st.plotly_chart(hm, use_container_width=True)
            else:
                st.caption("No weights to display.")


# footer
st.caption("Tip: If new files aren’t appearing, use the Diagnostics expander or click ↻ in Raw CSV to clear cache and rescan.")
