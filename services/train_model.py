# services/train_model.py
# Multi-model training + Trade Rationale 2.0 (+ Confidence & Position Sizing)
# Built-in retry pass:
#   1) Rebuild from stock_data.parquet
#   2) Optional live fetch via yfinance + feature generation (merged from your retry script)

import os
import sys
import math
import argparse
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Optional XGBoost
HAS_XGB = False
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Optional yfinance (only used if --live-fetch)
try:
    import yfinance as yf  # type: ignore
    HAS_YF = True
except Exception:
    HAS_YF = False

# Allow importing from project root for feature generator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.feature_generator import add_technical_indicators  # your existing helper

# ---------- Paths ----------
RESULTS_DIR = "data/results"
PREDICTIONS_DIR = "data/predictions"
LOGS_DIR = "data/logs"
FAILED_LIST_DEFAULT = os.path.join(LOGS_DIR, "failed_tickers_unique.txt")
RETRY_FAILED_LOG = os.path.join(LOGS_DIR, "failed_tickers_retry.txt")
RETRIED_MERGED_OUT = "data/processed/retried_stock_data.parquet"

FUNDAMENTALS_PATH = os.path.join(RESULTS_DIR, "fundamentals.csv")
SCORES_PATH = os.path.join(RESULTS_DIR, "stock_scores.csv")
NEWS_SENTIMENT_PATH = os.path.join(RESULTS_DIR, "news_sentiment.csv")  # optional
STOCK_DATA_MERGED = os.path.join(RESULTS_DIR, "stock_data.parquet")    # optional backfill source

FEATURES_OUT_PATH = os.path.join(RESULTS_DIR, "feature_importance.csv")
SKIPPED_TICKERS_LOG = os.path.join(RESULTS_DIR, "skipped_tickers.csv")
MODEL_COMPARISON_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")

SIGNALS_WITH_RATIONALE_PATH = os.path.join(RESULTS_DIR, "signals_with_rationale.csv")
LEGACY_SIGNALS_PATH = os.path.join(RESULTS_DIR, "signals.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RETRIED_MERGED_OUT), exist_ok=True)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Train models + rationale with built-in retry and optional live fetch.")
    p.add_argument("--min-rows", type=int, default=30, help="Minimum rows on first pass")
    p.add_argument("--retry-min-rows", type=int, default=20, help="Minimum rows on retry")
    p.add_argument("--retry", action="store_true", help="Force a retry pass even if nothing skipped")
    p.add_argument("--live-fetch", action="store_true", help="Enable yfinance fallback to fetch missing tickers")
    p.add_argument("--failed-list", type=str, default=FAILED_LIST_DEFAULT, help="Path to failed tickers list")
    p.add_argument("--sleep-min", type=float, default=0.5, help="Min sleep between live fetches (anti-rate-limit)")
    p.add_argument("--sleep-max", type=float, default=2.5, help="Max sleep between live fetches")
    return p.parse_args()

args = parse_args()

print("🧠 Starting multi-model training with feature importance, model comparison, and Trade Rationale 2.0...")

# ---------- Helpers ----------
def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window // 2)).mean()

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def pct_diff(a, b):
    if b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (a - b) / b

def build_rationale_row(row, score_pct, sentiment_val):
    parts = []
    if pd.notna(row.get("sma20")) and pd.notna(row.get("sma50")):
        parts.append("Uptrend (SMA20 > SMA50)" if row["sma20"] > row["sma50"] else "Downtrend (SMA20 < SMA50)")
    rsi = row.get("rsi14")
    if pd.notna(rsi):
        if rsi <= 30: parts.append(f"RSI {rsi:.0f} (oversold)")
        elif rsi >= 70: parts.append(f"RSI {rsi:.0f} (overbought)")
        else: parts.append(f"RSI {rsi:.0f}")
    if pd.notna(row.get("atr14")) and pd.notna(row.get("close")) and row["close"] != 0:
        parts.append(f"Volatility {(row['atr14']/row['close']):.1%} (ATR/Price)")
    if pd.notna(row.get("pe_ratio")): parts.append(f"P/E {row['pe_ratio']:.1f}")
    if pd.notna(row.get("dividend_yield")) and row["dividend_yield"] > 0: parts.append(f"Dividend {row['dividend_yield']:.2%}")
    if pd.notna(score_pct): parts.append(f"Score pct {score_pct:.0%}")
    if pd.notna(sentiment_val): parts.append(f"Sentiment {sentiment_val:+.2f}")
    if pd.notna(row.get("predicted_close")) and pd.notna(row.get("close")) and row["close"] != 0:
        parts.append(f"Predicted edge {pct_diff(row['predicted_close'], row['close']):.2%}")
    sig = row.get("signal", "HOLD")
    return f"{sig}: " + ", ".join(parts)

def safe_percentile_rank(series: pd.Series, value):
    if len(series) < 2 or pd.isna(value): return np.nan
    return (series < value).mean()

# ---------- Confidence & Position Size ----------
def _nz(x, default=0.0):
    try:
        xv = float(x)
        return xv if xv == xv and np.isfinite(xv) else float(default)
    except Exception:
        return float(default)

def compute_confidence_row(row):
    close = _nz(row.get("close"))
    pred  = _nz(row.get("predicted_close"), close)
    edge  = (pred - close) / close if close > 0 else 0.0

    atrp  = _nz(row.get("atr14")) / close if close > 0 else 0.0
    rsi   = _nz(row.get("rsi14"))
    sma20 = _nz(row.get("sma20"))
    sma50 = _nz(row.get("sma50"))
    sent  = _nz(row.get("sentiment"))
    score = _nz(row.get("total_score"))

    edge_norm = np.tanh(abs(edge) / 0.02)
    edge_sign_ok = 1.0 if (edge >= 0 and row.get("signal") == "BUY") or (edge < 0 and row.get("signal") == "SELL") else 0.3
    w_edge = edge_norm * edge_sign_ok

    trend = 1.0 if (row.get("signal") == "BUY" and sma20 > sma50) or (row.get("signal") == "SELL" and sma20 < sma50) else 0.35
    rsi_conf = 0.7 if (row.get("signal") == "BUY" and rsi < 40) or (row.get("signal") == "SELL" and rsi > 60) else 0.4

    atrp_clamped = min(max(atrp, 0.0), 0.10)
    if atrp_clamped <= 0.01: vol_factor = 0.6
    elif atrp_clamped <= 0.03: vol_factor = 1.0
    elif atrp_clamped <= 0.06: vol_factor = 0.8
    else: vol_factor = 0.55

    sent_norm  = (sent + 1.0) / 2.0
    score_norm = score / 100.0 if score > 1.0 else score
    qual = (0.5 * sent_norm + 0.5 * score_norm) if row.get("signal") == "BUY" else (0.5 * (1 - sent_norm) + 0.5 * (1 - score_norm))
    conf_raw = (0.45 * w_edge + 0.30 * trend + 0.10 * rsi_conf + 0.10 * qual) * vol_factor
    confidence = float(np.clip(conf_raw, 0.0, 1.0))

    k_edge = np.clip(edge, -0.15, 0.15)
    risk = max(atrp_clamped, 0.005)
    kelly_like = np.clip((k_edge / (2 * risk)), -0.10, 0.10)
    pos_size = float(np.clip(confidence * max(kelly_like, 0.0) * 0.5, 0.0, 0.05))
    return confidence, pos_size, edge

# ---------- I/O helpers ----------
def must_load_csv(path, name):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"❌ {name} missing or empty at {path}")
        raise SystemExit(1)
    return pd.read_csv(path)

def load_ticker_parquet(ticker: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, f"{ticker}.parquet")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"⚠️ {ticker}: could not read parquet ({e})")
    return pd.DataFrame()

def rebuild_from_merged_stockdata(ticker: str) -> pd.DataFrame:
    if not os.path.exists(STOCK_DATA_MERGED) or os.path.getsize(STOCK_DATA_MERGED) == 0:
        return pd.DataFrame()
    try:
        merged = pd.read_parquet(STOCK_DATA_MERGED)
    except Exception as e:
        print(f"⚠️ Could not open stock_data.parquet: {e}")
        return pd.DataFrame()
    if "ticker" not in merged.columns:
        tick_col = [c for c in merged.columns if c.lower() == "ticker"]
        if tick_col: merged = merged.rename(columns={tick_col[0]: "ticker"})
        else: return pd.DataFrame()
    sub = merged[merged["ticker"].astype(str).str.upper() == ticker].copy()
    return sub

# ---------- Live fetch (from your retry script) ----------
def fetch_yf_raw(ticker: str, retries=3, wait=2):
    if not HAS_YF:
        return None, "yfinance not installed"
    for attempt in range(1, retries + 1):
        try:
            print(f"\n📥 Fetching {ticker} (Attempt {attempt})...")
            df = yf.Ticker(ticker).history(period="10y", interval="1d", auto_adjust=False)
            if df.empty or df.isna().all().all():
                raise ValueError("Empty or invalid DataFrame")
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if "close" not in df.columns:
                raise ValueError("Missing 'close' column")
            df["ticker"] = ticker
            return df, ""
        except Exception as e:
            print(f"⚠️ Error fetching {ticker}: {e}")
            if attempt < retries:
                print(f"⏳ Retrying in {wait} sec...")
                time.sleep(wait)
            else:
                return None, str(e)

def add_features_and_save(df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame | None) -> pd.DataFrame:
    try:
        out = add_technical_indicators(df.copy(), spy_df if spy_df is not None else pd.DataFrame())
        out_path = os.path.join(RESULTS_DIR, f"{ticker}.parquet")
        out.to_parquet(out_path, index=False)
        return out
    except Exception as e:
        print(f"⚠️ Skipping {ticker} (indicator error): {e}")
        with open(RETRY_FAILED_LOG, "a") as log:
            log.write(f"{ticker} (indicator error)\n")
        return pd.DataFrame()

def get_spy_ref_df(existing_results_universe):
    # Prefer local SPY parquet; else try from merged; else fetch if allowed
    spy = "SPY"
    df = load_ticker_parquet(spy)
    if not df.empty: return df
    df = rebuild_from_merged_stockdata(spy)
    if not df.empty: return df
    if args.live_fetch and spy not in existing_results_universe:
        raw, _ = fetch_yf_raw(spy)
        if raw is not None:
            return raw
    return pd.DataFrame()

# ---------- Load core inputs ----------
fundamentals_df = must_load_csv(FUNDAMENTALS_PATH, "fundamentals.csv")
fundamentals_df["ticker"] = fundamentals_df["ticker"].astype(str).str.upper()

scores_df = must_load_csv(SCORES_PATH, "stock_scores.csv")
scores_df["ticker"] = scores_df["ticker"].astype(str).str.upper()

sent_df = pd.DataFrame()
if os.path.exists(NEWS_SENTIMENT_PATH) and os.path.getsize(NEWS_SENTIMENT_PATH) > 0:
    try:
        raw_sent = pd.read_csv(NEWS_SENTIMENT_PATH)
        if "ticker" in raw_sent.columns:
            raw_sent["ticker"] = raw_sent["ticker"].astype(str).str.upper()
        if "date" in raw_sent.columns:
            raw_sent["date"] = pd.to_datetime(raw_sent["date"], errors="coerce").dt.date
        sent_df = raw_sent.groupby(["ticker", "date"], as_index=False).agg(sentiment=("sentiment", "mean"))
        print("📰 Loaded news_sentiment.csv")
    except Exception as e:
        print(f"⚠️ Could not parse news_sentiment.csv: {e} — continuing without sentiment")

universe = sorted(set(fundamentals_df["ticker"]).union(set(scores_df["ticker"])))
existing_results_universe = {fn.replace(".parquet", "").upper()
                             for fn in os.listdir(RESULTS_DIR) if fn.endswith(".parquet")}
spy_ref_df = get_spy_ref_df(existing_results_universe)

# ---------- Model zoo ----------
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "LinearRegression": LinearRegression()
}
if HAS_XGB:
    models["XGBoost"] = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, random_state=42, n_jobs=-1
    )

# ---------- Collectors ----------
skipped_tickers = []
all_feature_importance = []
all_model_comparison = []
signals_with_rationale_rows = []
legacy_signals_rows = []
retried_frames = []

score_values = scores_df[["ticker", "total_score"]].dropna()["total_score"]

# ---------- Core trainer ----------
def train_one_ticker(ticker: str, raw_df: pd.DataFrame, min_rows: int, reason_prefix="") -> tuple:
    try:
        df = raw_df.copy()
        if df.empty or len(df) < min_rows:
            return False, f"{reason_prefix}Empty dataset or < {min_rows} rows"
        if "date" not in df.columns:
            return False, f"{reason_prefix}Missing 'date' column"
        for required in ["open", "high", "low", "close", "volume"]:
            if required not in df.columns:
                return False, f"{reason_prefix}Missing OHLCV columns"

        df = df.sort_values("date").reset_index(drop=True)
        df["target"] = df["close"].shift(-1)
        df.dropna(inplace=True)

        fund_row = fundamentals_df[fundamentals_df["ticker"] == ticker]
        score_row = scores_df[scores_df["ticker"] == ticker]
        if fund_row.empty or score_row.empty:
            return False, f"{reason_prefix}Missing fundamentals or score"

        for col in ["pe_ratio", "eps", "market_cap", "pb_ratio", "dividend_yield"]:
            df[col] = fund_row.iloc[0][col]
        df["total_score"] = score_row.iloc[0]["total_score"]

        df["sma20"] = compute_sma(df["close"], 20)
        df["sma50"] = compute_sma(df["close"], 50)
        df["rsi14"] = compute_rsi(df["close"], 14)
        df["atr14"] = compute_atr(df["high"], df["low"], df["close"], 14)

        base_cols = ["open", "high", "low", "close", "volume"]
        feature_cols = base_cols + ["pe_ratio", "eps", "market_cap", "pb_ratio", "dividend_yield", "total_score"]
        X = df[feature_cols].fillna(0)
        y = df["target"]

        rf_model = models["RandomForest"]
        rf_model.fit(X, y)
        df["predicted_close"] = rf_model.predict(X)

        df["signal"] = "HOLD"
        df.loc[df["predicted_close"] > df["close"], "signal"] = "BUY"
        df.loc[df["predicted_close"] < df["close"], "signal"] = "SELL"

        conf_df = df.apply(lambda r: pd.Series(
            compute_confidence_row(r), index=["confidence", "position_size", "delta_pct"]), axis=1)
        df["confidence"] = conf_df["confidence"].fillna(0.0)
        df["position_size"] = conf_df["position_size"].fillna(0.0)
        df["delta_pct"] = conf_df["delta_pct"].fillna(0.0)

        output_df = df[["date", "close", "predicted_close", "signal"]].copy()
        output_df["ticker"] = ticker
        output_path = os.path.join(PREDICTIONS_DIR, f"{ticker}_predictions.parquet")
        output_df.to_parquet(output_path, index=False)

        fi = getattr(rf_model, "feature_importances_", None)
        if fi is not None:
            all_feature_importance.append(pd.DataFrame({
                "ticker": ticker, "model": "RandomForest",
                "feature": feature_cols, "importance": fi
            }))

        for name, model in models.items():
            try:
                model.fit(X, y)
                preds = model.predict(X)
                temp = pd.DataFrame({
                    "ticker": ticker,
                    "date": df["date"],
                    "model": name,
                    "close": df["close"].values,
                    "predicted_close": preds
                })
                all_model_comparison.append(temp)

                if hasattr(model, "feature_importances_"):
                    imp_vals = model.feature_importances_
                    all_feature_importance.append(pd.DataFrame({
                        "ticker": ticker, "model": name,
                        "feature": feature_cols, "importance": imp_vals
                    }))
                elif hasattr(model, "coef_"):
                    coef = model.coef_
                    coef = coef.ravel() if hasattr(coef, "ravel") else coef
                    denom = abs(coef).sum()
                    abs_coef = abs(coef) / (denom if denom != 0 else 1.0)
                    if len(abs_coef) == len(feature_cols):
                        all_feature_importance.append(pd.DataFrame({
                            "ticker": ticker, "model": name,
                            "feature": feature_cols, "importance": abs_coef
                        }))
            except Exception as me:
                print(f"⚠️ {ticker} / {name}: model failed — {me}")
                continue

        df_dates = pd.to_datetime(df["date"], errors="coerce").dt.date
        if not sent_df.empty:
            merged = pd.DataFrame({"date": df_dates, "idx": df.index})
            merged["ticker"] = ticker
            merged = merged.merge(sent_df, on=["ticker", "date"], how="left")
            df["sentiment"] = merged.set_index("idx")["sentiment"].reindex(df.index)
        else:
            df["sentiment"] = np.nan

        t_score = df["total_score"].iloc[0] if "total_score" in df.columns and not df.empty else np.nan
        score_pct = safe_percentile_rank(score_values, t_score)

        for _, row in df.iterrows():
            rationale = build_rationale_row(row, score_pct, row.get("sentiment", np.nan))
            signals_with_rationale_rows.append({
                "date": pd.to_datetime(row["date"]).date(),
                "ticker": ticker,
                "close": row.get("close", np.nan),
                "predicted_close": row.get("predicted_close", np.nan),
                "delta_pct": row.get("delta_pct", np.nan),
                "signal": row.get("signal", "HOLD"),
                "confidence": float(row.get("confidence", 0.0)),
                "position_size": float(row.get("position_size", 0.0)),
                "rationale": rationale,
                "rsi14": row.get("rsi14", np.nan),
                "sma20": row.get("sma20", np.nan),
                "sma50": row.get("sma50", np.nan),
                "atr14": row.get("atr14", np.nan),
                "sentiment": row.get("sentiment", np.nan),
                "total_score": row.get("total_score", np.nan),
                "pe_ratio": row.get("pe_ratio", np.nan),
                "eps": row.get("eps", np.nan),
                "market_cap": row.get("market_cap", np.nan),
                "pb_ratio": row.get("pb_ratio", np.nan),
                "dividend_yield": row.get("dividend_yield", np.nan),
            })

            legacy_signals_rows.append({
                "date": pd.to_datetime(row["date"]).date(),
                "ticker": ticker,
                "close": row.get("close", np.nan),
                "signal": row.get("signal", "HOLD"),
            })

        print(f"✅ Trained {ticker} ({', '.join(models.keys())})")
        return True, ""

    except Exception as e:
        return False, f"Error processing — {e}"

# ---------- First pass ----------
first_pass_skips = []
for ticker in sorted(universe):
    df_t = load_ticker_parquet(ticker)
    ok, reason = train_one_ticker(ticker, df_t, min_rows=args.min_rows)
    if not ok:
        first_pass_skips.append((ticker, reason))

# ---------- Optional failed list (to prioritize live fetch) ----------
failed_set = set()
if os.path.exists(args.failed_list):
    with open(args.failed_list, "r") as f:
        failed_set = {line.strip().upper() for line in f if line.strip()}

# ---------- Retry pass ----------
need_retry = args.retry or len(first_pass_skips) > 0
retry_results = []
if need_retry:
    print("\n🔁 Retry pass: rebuild from stock_data.parquet, then optional live fetch…")
    # Build SPY reference if we only had raw prices (for feature generation)
    full_retried = []

    for ticker, prev_reason in first_pass_skips:
        # 1) Try merged stock_data.parquet
        rebuilt = rebuild_from_merged_stockdata(ticker)
        tried_live = False
        if rebuilt.empty and args.live_fetch:
            # 2) Live fetch via yfinance if allowed (and either on failed_list or generally allowed)
            tried_live = True
            raw, err = fetch_yf_raw(ticker)
            if raw is None:
                with open(RETRY_FAILED_LOG, "a") as log:
                    log.write(f"{ticker} (fetch error: {err})\n")
            else:
                # add features + save per-ticker parquet
                raw_feat = add_features_and_save(raw, ticker, spy_ref_df)
                if not raw_feat.empty:
                    rebuilt = raw_feat
                time.sleep(random.uniform(args.sleep_min, args.sleep_max))

        if rebuilt.empty:
            skipped_tickers.append({"ticker": ticker, "reason": prev_reason})
            retry_results.append((ticker, False, f"{'fetch failed; ' if tried_live else ''}{prev_reason}"))
            continue

        ok, reason = train_one_ticker(ticker, rebuilt, min_rows=args.retry_min_rows, reason_prefix="(retry) ")
        if not ok:
            skipped_tickers.append({"ticker": ticker, "reason": reason})
            retry_results.append((ticker, False, reason))
        else:
            retry_results.append((ticker, True, ""))
            # keep for optional merged output
            full_retried.append(rebuilt)

    # Save merged retried parquet for audit
    if full_retried:
        try:
            merged_out = pd.concat(full_retried, ignore_index=True)
            merged_out.dropna(subset=["close"], inplace=True)
            merged_out.to_parquet(RETRIED_MERGED_OUT, index=False)
            print(f"✅ Retrained data saved to: {RETRIED_MERGED_OUT}")
        except Exception as e:
            print(f"⚠️ Could not save merged retried parquet: {e}")

# ---------- Save combined feature importances ----------
if all_feature_importance:
    combined_fi = pd.concat(all_feature_importance, ignore_index=True)
    combined_fi.to_csv(FEATURES_OUT_PATH, index=False)
    print(f"\n📊 Feature importance (all models) saved to: {FEATURES_OUT_PATH}")

# ---------- Save model comparison table ----------
if all_model_comparison:
    mc_df = pd.concat(all_model_comparison, ignore_index=True)
    if "date" in mc_df.columns:
        mc_df["date"] = pd.to_datetime(mc_df["date"], errors="coerce")
    mc_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"📈 Model comparison saved to: {MODEL_COMPARISON_PATH}")
else:
    print("⚠️ No model comparison rows were generated.")

# ---------- Save signals with rationale ----------
if signals_with_rationale_rows:
    swr = pd.DataFrame(signals_with_rationale_rows)
    swr["date"] = pd.to_datetime(swr["date"], errors="coerce").dt.date
    swr.sort_values(["ticker", "date"], inplace=True)
    swr.to_csv(SIGNALS_WITH_RATIONALE_PATH, index=False)
    print(f"🗒️  Signals + Rationale saved to: {SIGNALS_WITH_RATIONALE_PATH}")

# ---------- Save legacy signals.csv ----------
if legacy_signals_rows:
    leg = pd.DataFrame(legacy_signals_rows).drop_duplicates(subset=["date", "ticker"])
    leg["date"] = pd.to_datetime(leg["date"], errors="coerce").dt.date
    leg.sort_values(["ticker", "date"], inplace=True)
    leg.to_csv(LEGACY_SIGNALS_PATH, index=False)
    print(f"📄 Legacy signals saved to: {LEGACY_SIGNALS_PATH}")

# ---------- Save skipped tickers log ----------
if skipped_tickers:
    skipped_df = pd.DataFrame(skipped_tickers)
    skipped_df.to_csv(SKIPPED_TICKERS_LOG, index=False)
    print(f"📄 Skipped tickers log saved to: {SKIPPED_TICKERS_LOG}")

# ---------- Summary ----------
if first_pass_skips:
    n_retry_ok = sum(1 for _, ok, _ in retry_results if ok)
    n_retry_fail = sum(1 for _, ok, _ in retry_results if not ok)
    print(f"\n🔚 Retry summary: {n_retry_ok} fixed, {n_retry_fail} still skipped.")
print("\n🏁 All models trained. RF predictions written per ticker, model comparison & feature importances exported, rationales generated.")
