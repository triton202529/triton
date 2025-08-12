# services/train_model.py
# Multi-model training + Trade Rationale 2.0 (+ Confidence & Position Sizing)
# NOTE: This version intentionally has **NO retry** and **NO live fetch**.
# It only trains on tickers that already have valid per‑ticker parquet files
# in data/results/{TICKER}.parquet. Any failures are logged to skipped_tickers.csv.

import os
import sys
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Optional XGBoost
HAS_XGB = False
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Allow importing from project root for feature generator (not used here, features are already in parquet)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------- Paths ----------
RESULTS_DIR = "data/results"
PREDICTIONS_DIR = "data/predictions"
LOGS_DIR = "data/logs"

FUNDAMENTALS_PATH = os.path.join(RESULTS_DIR, "fundamentals.csv")
SCORES_PATH = os.path.join(RESULTS_DIR, "stock_scores.csv")
NEWS_SENTIMENT_PATH = os.path.join(RESULTS_DIR, "news_sentiment.csv")  # optional

FEATURES_OUT_PATH = os.path.join(RESULTS_DIR, "feature_importance.csv")
SKIPPED_TICKERS_LOG = os.path.join(RESULTS_DIR, "skipped_tickers.csv")
MODEL_COMPARISON_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")

SIGNALS_WITH_RATIONALE_PATH = os.path.join(RESULTS_DIR, "signals_with_rationale.csv")
LEGACY_SIGNALS_PATH = os.path.join(RESULTS_DIR, "signals.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Train models + rationale (no retry).")
    p.add_argument("--min-rows", type=int, default=30, help="Minimum rows required per ticker parquet.")
    return p.parse_args()

args = parse_args()

print("🧠 Starting multi-model training with feature importance, model comparison, and Trade Rationale 2.0 (no-retry)…")

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
    # Trend
    if pd.notna(row.get("sma20")) and pd.notna(row.get("sma50")):
        parts.append("Uptrend (SMA20 > SMA50)" if row["sma20"] > row["sma50"] else "Downtrend (SMA20 < SMA50)")
    # RSI
    rsi = row.get("rsi14")
    if pd.notna(rsi):
        if rsi <= 30: parts.append(f"RSI {rsi:.0f} (oversold)")
        elif rsi >= 70: parts.append(f"RSI {rsi:.0f} (overbought)")
        else: parts.append(f"RSI {rsi:.0f}")
    # Volatility
    if pd.notna(row.get("atr14")) and pd.notna(row.get("close")) and row["close"] != 0:
        parts.append(f"Volatility {(row['atr14']/row['close']):.1%} (ATR/Price)")
    # Fundamentals
    if pd.notna(row.get("pe_ratio")): parts.append(f"P/E {row['pe_ratio']:.1f}")
    if pd.notna(row.get("dividend_yield")) and row["dividend_yield"] > 0: parts.append(f"Dividend {row['dividend_yield']:.2%}")
    # Score percentile
    if pd.notna(score_pct): parts.append(f"Score pct {score_pct:.0%}")
    # Sentiment
    if pd.notna(sentiment_val): parts.append(f"Sentiment {sentiment_val:+.2f}")
    # Pred edge
    if pd.notna(row.get("predicted_close")) and pd.notna(row.get("close")) and row["close"] != 0:
        parts.append(f"Predicted edge {pct_diff(row['predicted_close'], row['close']):.2%}")
    sig = row.get("signal", "HOLD")
    return f"{sig}: " + ", ".join(parts)

def safe_percentile_rank(series: pd.Series, value):
    if len(series) < 2 or pd.isna(value):
        return np.nan
    return (series < value).mean()

# Confidence & position sizing (deterministic)
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

# ---------- Load core inputs ----------
fundamentals_df = must_load_csv(FUNDAMENTALS_PATH, "fundamentals.csv")
fundamentals_df["ticker"] = fundamentals_df["ticker"].astype(str).str.upper()

scores_df = must_load_csv(SCORES_PATH, "stock_scores.csv")
scores_df["ticker"] = scores_df["ticker"].astype(str).str.upper()

# Optional: daily news sentiment
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

# Universe = intersection of those that have a parquet and appear in inputs
available_parquets = {fn.replace(".parquet", "").upper()
                      for fn in os.listdir(RESULTS_DIR) if fn.endswith(".parquet")}
universe = sorted(set(fundamentals_df["ticker"]).union(set(scores_df["ticker"])))

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

score_values = scores_df[["ticker", "total_score"]].dropna()["total_score"]

# ---------- Core trainer ----------
def train_one_ticker(ticker: str, raw_df: pd.DataFrame, min_rows: int) -> bool:
    try:
        df = raw_df.copy()
        if df.empty or len(df) < min_rows:
            skipped_tickers.append({"ticker": ticker, "reason": f"Empty dataset or < {min_rows} rows"})
            return False
        if "date" not in df.columns:
            skipped_tickers.append({"ticker": ticker, "reason": "Missing 'date' column"})
            return False
        for required in ["open", "high", "low", "close", "volume"]:
            if required not in df.columns:
                skipped_tickers.append({"ticker": ticker, "reason": "Missing OHLCV columns"})
                return False

        df = df.sort_values("date").reset_index(drop=True)
        df["target"] = df["close"].shift(-1)
        df.dropna(inplace=True)

        fund_row = fundamentals_df[fundamentals_df["ticker"] == ticker]
        score_row = scores_df[scores_df["ticker"] == ticker]
        if fund_row.empty or score_row.empty:
            skipped_tickers.append({"ticker": ticker, "reason": "Missing fundamentals or score"})
            return False

        for col in ["pe_ratio", "eps", "market_cap", "pb_ratio", "dividend_yield"]:
            df[col] = fund_row.iloc[0][col]
        df["total_score"] = score_row.iloc[0]["total_score"]

        # Compute light tech (in case parquet didn't have them)
        if "sma20" not in df.columns:
            df["sma20"] = compute_sma(df["close"], 20)
        if "sma50" not in df.columns:
            df["sma50"] = compute_sma(df["close"], 50)
        if "rsi14" not in df.columns:
            df["rsi14"] = compute_rsi(df["close"], 14)
        if "atr14" not in df.columns:
            df["atr14"] = compute_atr(df["high"], df["low"], df["close"], 14)

        # Features
        base_cols = ["open", "high", "low", "close", "volume"]
        feature_cols = base_cols + ["pe_ratio", "eps", "market_cap", "pb_ratio", "dividend_yield", "total_score"]
        X = df[feature_cols].fillna(0)
        y = df["target"]

        # Baseline RF predictions
        rf_model = models["RandomForest"]
        rf_model.fit(X, y)
        df["predicted_close"] = rf_model.predict(X)

        # Signals
        df["signal"] = "HOLD"
        df.loc[df["predicted_close"] > df["close"], "signal"] = "BUY"
        df.loc[df["predicted_close"] < df["close"], "signal"] = "SELL"

        # Confidence / pos size / edge
        conf_df = df.apply(lambda r: pd.Series(
            compute_confidence_row(r), index=["confidence", "position_size", "delta_pct"]), axis=1)
        df["confidence"] = conf_df["confidence"].fillna(0.0)
        df["position_size"] = conf_df["position_size"].fillna(0.0)
        df["delta_pct"] = conf_df["delta_pct"].fillna(0.0)

        # Save per‑ticker RF predictions
        output_df = df[["date", "close", "predicted_close", "signal"]].copy()
        output_df["ticker"] = ticker
        output_path = os.path.join(PREDICTIONS_DIR, f"{ticker}_predictions.parquet")
        output_df.to_parquet(output_path, index=False)

        # RF feature importance
        fi = getattr(rf_model, "feature_importances_", None)
        if fi is not None:
            all_feature_importance.append(pd.DataFrame({
                "ticker": ticker, "model": "RandomForest",
                "feature": feature_cols, "importance": fi
            }))

        # Model comparison
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

        # Sentiment merge (optional)
        df_dates = pd.to_datetime(df["date"], errors="coerce").dt.date
        if not sent_df.empty:
            merged = pd.DataFrame({"date": df_dates, "idx": df.index})
            merged["ticker"] = ticker
            merged = merged.merge(sent_df, on=["ticker", "date"], how="left")
            df["sentiment"] = merged.set_index("idx")["sentiment"].reindex(df.index)
        else:
            df["sentiment"] = np.nan

        # Score percentile
        t_score = df["total_score"].iloc[0] if "total_score" in df.columns and not df.empty else np.nan
        score_pct = safe_percentile_rank(score_values, t_score)

        # Export rationale rows
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
        return True

    except Exception as e:
        skipped_tickers.append({"ticker": ticker, "reason": f"Error processing — {e}"})
        return False

# ---------- Train ONLY on tickers with an existing parquet ----------
trained = 0
for ticker in sorted(universe):
    if ticker not in available_parquets:
        skipped_tickers.append({"ticker": ticker, "reason": "No parquet data (data/results/{TICKER}.parquet)"})
        continue
    df_t = load_ticker_parquet(ticker)
    ok = train_one_ticker(ticker, df_t, min_rows=args.min_rows)
    trained += int(ok)

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

print(f"\n🏁 Done. Trained {trained} tickers. For the rest, run your retry script AFTER rebuilding/fetching parquets.")
