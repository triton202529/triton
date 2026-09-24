# services/train_model.py
# Multi-model training + Trade Rationale 2.0 (+ Confidence & Position Sizing)
# LEAKAGE-SAFE: Predict today's close (t) using only lagged features from (t-1).
#
# FIXES (Dec 2025):
#   ✅ Prefer per-ticker parquets from data/processed/{TICKER}.parquet
#   ✅ Fallback to legacy data/results/{TICKER}.parquet
#   ✅ Remove NaN/blank tickers so you never get "ticker = NAN" in skipped_tickers.csv
#
# Run via:
#   python -m services.train_model

from __future__ import annotations

import hashlib
import os
import sys
import time
import io
import argparse
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Ensure imports work when run as module (-m)
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.artifacts_writer import write_heartbeat

# --- Ensure stdout/stderr use UTF-8 (prevents Windows cp1252 UnicodeEncodeError when printing emojis) ---
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    try:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# ----- Optional XGBoost -----
HAS_XGB = False
try:
    from xgboost import XGBRegressor  # type: ignore

    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ---------- Paths ----------
RESULTS_DIR = "data/results"
PROCESSED_DIR = "data/processed"  # ✅ preferred parquet location
PREDICTIONS_DIR = "data/predictions"
LOGS_DIR = "data/logs"

FUNDAMENTALS_PATH = os.path.join(RESULTS_DIR, "fundamentals.csv")
SCORES_PATH = os.path.join(RESULTS_DIR, "stock_scores.csv")
NEWS_SENTIMENT_PATH = os.path.join(RESULTS_DIR, "news_sentiment.csv")  # optional

FEATURES_OUT_PATH = os.path.join(RESULTS_DIR, "feature_importance.csv")
SKIPPED_TICKERS_LOG = os.path.join(RESULTS_DIR, "skipped_tickers.csv")
MODEL_COMPARISON_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")


def _unique_sibling_tmp(target: str) -> str:
    abs_target = os.path.abspath(target)
    parent = os.path.dirname(abs_target) or "."
    name = os.path.basename(abs_target)
    return os.path.join(parent, f".{name}.{os.getpid()}.{time.time_ns()}.tmp")


def atomic_publish_csv(df: pd.DataFrame, target: str) -> None:
    """Write CSV to a unique sibling tmp, flush, then os.replace onto target."""
    abs_target = os.path.abspath(target)
    parent = os.path.dirname(abs_target) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = _unique_sibling_tmp(abs_target)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            df.to_csv(fh, index=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, abs_target)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def atomic_publish_parquet(df: pd.DataFrame, target: str) -> None:
    """Write parquet to a unique sibling tmp, fsync, then os.replace onto target."""
    abs_target = os.path.abspath(target)
    parent = os.path.dirname(abs_target) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = _unique_sibling_tmp(abs_target)
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        with open(tmp, "wb") as fh:
            fh.write(buf.getvalue())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, abs_target)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def _ensure_output_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train models + rationale (no retry, leakage-safe).")
    p.add_argument(
        "--min-rows",
        type=int,
        default=30,
        help="Minimum rows required per ticker parquet (pre-lag).",
    )
    p.add_argument("--run-id", default="train", help="Stable run identity bound into predictions.")
    return p.parse_args(argv)


def main(argv=None):
    _ensure_output_dirs()
    try:
        return _train(argv)
    except BaseException as e:
        try:
            write_heartbeat(
                status="fail",
                stage="train",
                last_success_stage="train",
                message="Training failed.",
                error=str(e),
            )
        except Exception:
            pass
        raise


def _train(argv=None):
    args = parse_args(argv)
    print(
        "🧠 Starting multi-model training (lagged features), feature importance, model comparison, Trade Rationale 2.0…"
    )
    print(
        f"📌 Parquet source priority: {PROCESSED_DIR}/{{TICKER}}.parquet → fallback {RESULTS_DIR}/{{TICKER}}.parquet"
    )

    # ---------- Helpers ----------
    def compute_sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=max(2, window // 2)).mean()

    def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    def pct_diff(a, b):
        if b == 0 or pd.isna(a) or pd.isna(b):
            return np.nan
        return (a - b) / b

    def safe_percentile_rank(series: pd.Series, value):
        if len(series) < 2 or pd.isna(value):
            return np.nan
        return (series < value).mean()

    def build_rationale_row(row, score_pct, sentiment_val):
        parts = []
        if pd.notna(row.get("sma20")) and pd.notna(row.get("sma50")):
            parts.append(
                "Uptrend (SMA20 > SMA50)"
                if row["sma20"] > row["sma50"]
                else "Downtrend (SMA20 < SMA50)"
            )

        rsi = row.get("rsi14")
        if pd.notna(rsi):
            if rsi <= 30:
                parts.append(f"RSI {rsi:.0f} (oversold)")
            elif rsi >= 70:
                parts.append(f"RSI {rsi:.0f} (overbought)")
            else:
                parts.append(f"RSI {rsi:.0f}")

        if pd.notna(row.get("atr14")) and pd.notna(row.get("close")) and row["close"] != 0:
            parts.append(f"Volatility {(row['atr14'] / row['close']):.1%} (ATR/Price)")

        if pd.notna(row.get("pe_ratio")):
            parts.append(f"P/E {row['pe_ratio']:.1f}")
        if pd.notna(row.get("dividend_yield")) and row["dividend_yield"] > 0:
            parts.append(f"Dividend {row['dividend_yield']:.2%}")

        if pd.notna(score_pct):
            parts.append(f"Score pct {score_pct:.0%}")
        if pd.notna(sentiment_val):
            parts.append(f"Sentiment {sentiment_val:+.2f}")

        if (
            pd.notna(row.get("predicted_close"))
            and pd.notna(row.get("close"))
            and row["close"] != 0
        ):
            parts.append(f"Predicted edge {pct_diff(row['predicted_close'], row['close']):.2%}")

        sig = row.get("signal", "HOLD")
        return f"{sig}: " + ", ".join(parts)

    def _nz(x, default=0.0):
        try:
            xv = float(x)
            return xv if xv == xv and np.isfinite(xv) else float(default)
        except Exception:
            return float(default)

    def compute_confidence_row(row):
        close = _nz(row.get("close"))
        pred = _nz(row.get("predicted_close"), close)
        edge = (pred - close) / close if close > 0 else 0.0

        atrp = _nz(row.get("atr14")) / close if close > 0 else 0.0
        rsi = _nz(row.get("rsi14"))
        sma20 = _nz(row.get("sma20"))
        sma50 = _nz(row.get("sma50"))
        sent = _nz(row.get("sentiment"))
        score = _nz(row.get("total_score"))

        edge_norm = np.tanh(abs(edge) / 0.02)
        edge_sign_ok = (
            1.0
            if (edge >= 0 and row.get("signal") == "BUY")
            or (edge < 0 and row.get("signal") == "SELL")
            else 0.3
        )
        w_edge = edge_norm * edge_sign_ok

        trend = (
            1.0
            if (
                (row.get("signal") == "BUY" and sma20 > sma50)
                or (row.get("signal") == "SELL" and sma20 < sma50)
            )
            else 0.35
        )
        rsi_conf = (
            0.7
            if (
                (row.get("signal") == "BUY" and rsi < 40)
                or (row.get("signal") == "SELL" and rsi > 60)
            )
            else 0.4
        )

        atrp_clamped = min(max(atrp, 0.0), 0.10)
        if atrp_clamped <= 0.01:
            vol_factor = 0.6
        elif atrp_clamped <= 0.03:
            vol_factor = 1.0
        elif atrp_clamped <= 0.06:
            vol_factor = 0.8
        else:
            vol_factor = 0.55

        sent_norm = (sent + 1.0) / 2.0
        score_norm = score / 100.0 if score > 1.0 else score
        qual = (
            (0.5 * sent_norm + 0.5 * score_norm)
            if row.get("signal") == "BUY"
            else (0.5 * (1 - sent_norm) + 0.5 * (1 - score_norm))
        )

        conf_raw = (0.45 * w_edge + 0.30 * trend + 0.10 * rsi_conf + 0.10 * qual) * vol_factor
        confidence = float(np.clip(conf_raw, 0.0, 1.0))

        k_edge = np.clip(edge, -0.15, 0.15)
        risk = max(atrp_clamped, 0.005)
        kelly_like = np.clip((k_edge / (2 * risk)), -0.10, 0.10)
        pos_size = float(np.clip(confidence * max(kelly_like, 0.0) * 0.5, 0.0, 0.05))
        return confidence, pos_size, edge

    # ---------- I/O ----------
    def must_load_csv(path, name):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"❌ {name} missing or empty at {path}")
            raise SystemExit(1)
        return pd.read_csv(path)

    def _available_parquets_in(dir_path: str) -> set[str]:
        if not os.path.exists(dir_path):
            return set()
        return {
            fn.replace(".parquet", "").upper()
            for fn in os.listdir(dir_path)
            if fn.endswith(".parquet")
        }

    def load_ticker_parquet(ticker: str) -> pd.DataFrame:
        path1 = os.path.join(PROCESSED_DIR, f"{ticker}.parquet")
        if os.path.exists(path1) and os.path.getsize(path1) > 0:
            try:
                return pd.read_parquet(path1)
            except Exception as e:
                print(f"⚠️ {ticker}: could not read processed parquet ({e})")

        path2 = os.path.join(RESULTS_DIR, f"{ticker}.parquet")
        if os.path.exists(path2) and os.path.getsize(path2) > 0:
            try:
                return pd.read_parquet(path2)
            except Exception as e:
                print(f"⚠️ {ticker}: could not read results parquet ({e})")

        return pd.DataFrame()

    # ---------- Load inputs ----------
    fundamentals_df = must_load_csv(FUNDAMENTALS_PATH, "fundamentals.csv")
    scores_df = must_load_csv(SCORES_PATH, "stock_scores.csv")

    # ✅ Clean tickers FIRST (this prevents NAN universe)
    fundamentals_df["ticker"] = fundamentals_df.get("ticker", pd.Series(dtype=str))
    scores_df["ticker"] = scores_df.get("ticker", pd.Series(dtype=str))

    fundamentals_df["ticker"] = fundamentals_df["ticker"].astype(str).str.strip().str.upper()
    scores_df["ticker"] = scores_df["ticker"].astype(str).str.strip().str.upper()

    fundamentals_df = fundamentals_df[fundamentals_df["ticker"].notna()]
    scores_df = scores_df[scores_df["ticker"].notna()]

    fundamentals_df = fundamentals_df[~fundamentals_df["ticker"].isin(["", "NAN", "NONE"])]
    scores_df = scores_df[~scores_df["ticker"].isin(["", "NAN", "NONE"])]

    # Optional sentiment
    sent_df = pd.DataFrame()
    if os.path.exists(NEWS_SENTIMENT_PATH) and os.path.getsize(NEWS_SENTIMENT_PATH) > 0:
        try:
            raw_sent = pd.read_csv(NEWS_SENTIMENT_PATH)
            if "ticker" in raw_sent.columns:
                raw_sent["ticker"] = raw_sent["ticker"].astype(str).str.strip().str.upper()
            if "date" in raw_sent.columns:
                raw_sent["date"] = pd.to_datetime(raw_sent["date"], errors="coerce").dt.date
            sent_df = raw_sent.groupby(["ticker", "date"], as_index=False).agg(
                sentiment=("sentiment", "mean")
            )
            print("📰 Loaded news_sentiment.csv")
        except Exception as e:
            print(f"⚠️ Could not parse news_sentiment.csv: {e} — continuing without sentiment")

    # ✅ Discover parquets from processed + results
    available_parquets = _available_parquets_in(PROCESSED_DIR).union(
        _available_parquets_in(RESULTS_DIR)
    )

    from pathlib import Path

    from services.daily_prediction_refresh_validate import required_symbol_sets
    from services.lifecycle_artifact_guard import parse_universe_ticker_identities
    from services.market_calendar import last_completed_nyse_session

    governed_path = Path(PROJECT_ROOT) / "data" / "config" / "tickers.txt"
    processed_under_project = (
        Path(PROCESSED_DIR).resolve() == (Path(PROJECT_ROOT) / "data" / "processed").resolve()
    )
    governed_batch = governed_path.is_file() and processed_under_project
    if governed_batch:
        governed = parse_universe_ticker_identities(governed_path)
        required_symbols = list(required_symbol_sets(governed)["prediction_required"])
        target_session = last_completed_nyse_session().isoformat()
    else:
        required_symbols = sorted(set(fundamentals_df["ticker"]).union(set(scores_df["ticker"])))
        target_session = ""

    # ---------- Models ----------
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )

    skipped_tickers: list[dict] = []
    staged_predictions: dict[str, pd.DataFrame] = {}
    all_feature_importance = []
    all_model_comparison = []

    score_values = (
        scores_df[["ticker", "total_score"]].dropna()["total_score"]
        if "total_score" in scores_df.columns
        else pd.Series(dtype=float)
    )

    model_identity = "+".join(models.keys())
    run_id = str(args.run_id)

    def train_one_ticker(
        ticker: str,
        raw_df: pd.DataFrame,
        min_rows: int,
        processed_sha256: str,
    ) -> bool:
        try:
            df = raw_df.copy()
            if df.empty or len(df) < min_rows:
                skipped_tickers.append(
                    {"ticker": ticker, "reason": f"Empty dataset or < {min_rows} rows"}
                )
                return False

            if "date" not in df.columns:
                skipped_tickers.append({"ticker": ticker, "reason": "Missing 'date' column"})
                return False

            for required in ["open", "high", "low", "close", "volume"]:
                if required not in df.columns:
                    skipped_tickers.append({"ticker": ticker, "reason": "Missing OHLCV columns"})
                    return False

            df = df.sort_values("date").reset_index(drop=True)

            fund_row = fundamentals_df[fundamentals_df["ticker"] == ticker]
            score_row = scores_df[scores_df["ticker"] == ticker]
            fund_cols = ["pe_ratio", "eps", "market_cap", "pb_ratio", "dividend_yield"]
            for col in fund_cols:
                df[col] = (
                    fund_row.iloc[0][col]
                    if not fund_row.empty and col in fund_row.columns
                    else np.nan
                )

            df[fund_cols] = df[fund_cols].fillna(0.0)
            df["total_score"] = (
                score_row.iloc[0]["total_score"]
                if not score_row.empty and "total_score" in score_row.columns
                else 0.0
            )

            if "sma20" not in df.columns:
                df["sma20"] = compute_sma(df["close"], 20)
            if "sma50" not in df.columns:
                df["sma50"] = compute_sma(df["close"], 50)
            if "rsi14" not in df.columns:
                df["rsi14"] = compute_rsi(df["close"], 14)
            if "atr14" not in df.columns:
                df["atr14"] = compute_atr(df["high"], df["low"], df["close"], 14)

            base_cols = ["open", "high", "low", "close", "volume"]
            tech_cols = ["sma20", "sma50", "rsi14", "atr14"]
            feature_cols = base_cols + tech_cols + fund_cols + ["total_score"]

            X_raw = df[feature_cols].copy()
            X = X_raw.shift(1)
            y = df["close"]

            required_for_mask = base_cols + tech_cols + ["total_score"]
            mask = X[required_for_mask].notna().all(axis=1) & y.notna()

            if mask.sum() < min_rows:
                skipped_tickers.append(
                    {
                        "ticker": ticker,
                        "reason": (
                            f"Too few rows after lagging (have {int(mask.sum())}) "
                            f"session={target_session}"
                        ),
                    }
                )
                return False

            X = X.loc[mask]
            y = y.loc[mask]
            df_model = df.loc[mask].copy()

            rf_model = models["RandomForest"]
            rf_model.fit(X, y)
            df_model["predicted_close"] = rf_model.predict(X)

            df_model["signal"] = "HOLD"
            df_model.loc[df_model["predicted_close"] > df_model["close"], "signal"] = "BUY"
            df_model.loc[df_model["predicted_close"] < df_model["close"], "signal"] = "SELL"

            df_model_dates = pd.to_datetime(df_model["date"], errors="coerce").dt.date
            if not sent_df.empty:
                merged = pd.DataFrame({"date": df_model_dates, "idx": df_model.index})
                merged["ticker"] = ticker
                merged = merged.merge(sent_df, on=["ticker", "date"], how="left")
                df_model["sentiment"] = merged.set_index("idx")["sentiment"].reindex(df_model.index)
            else:
                df_model["sentiment"] = np.nan

            conf_df = df_model.apply(
                lambda r: pd.Series(
                    compute_confidence_row(r), index=["confidence", "position_size", "delta_pct"]
                ),
                axis=1,
            )
            df_model["confidence"] = conf_df["confidence"].fillna(0.0)
            df_model["position_size"] = conf_df["position_size"].fillna(0.0)
            df_model["delta_pct"] = conf_df["delta_pct"].fillna(0.0)

            output_df = df_model[["date", "close", "predicted_close", "signal"]].copy()
            output_df["ticker"] = ticker
            output_df["completed_session"] = target_session
            output_df["processed_sha256"] = processed_sha256
            output_df["model_identity"] = model_identity
            output_df["run_id"] = run_id
            staged_predictions[ticker] = output_df

            fi = getattr(rf_model, "feature_importances_", None)
            if fi is not None and len(fi) == len(feature_cols):
                all_feature_importance.append(
                    pd.DataFrame(
                        {
                            "ticker": ticker,
                            "model": "RandomForest",
                            "feature": feature_cols,
                            "importance": fi,
                        }
                    )
                )

            for name, model in models.items():
                try:
                    model.fit(X, y)
                    preds = model.predict(X)
                    all_model_comparison.append(
                        pd.DataFrame(
                            {
                                "ticker": ticker,
                                "date": df_model["date"].values,
                                "model": name,
                                "close": df_model["close"].values,
                                "predicted_close": preds,
                            }
                        )
                    )

                    if hasattr(model, "feature_importances_"):
                        imp_vals = model.feature_importances_
                        if len(imp_vals) == len(feature_cols):
                            all_feature_importance.append(
                                pd.DataFrame(
                                    {
                                        "ticker": ticker,
                                        "model": name,
                                        "feature": feature_cols,
                                        "importance": imp_vals,
                                    }
                                )
                            )
                    elif hasattr(model, "coef_"):
                        coef = model.coef_
                        coef = coef.ravel() if hasattr(coef, "ravel") else coef
                        denom = abs(coef).sum()
                        abs_coef = abs(coef) / (denom if denom != 0 else 1.0)
                        if len(abs_coef) == len(feature_cols):
                            all_feature_importance.append(
                                pd.DataFrame(
                                    {
                                        "ticker": ticker,
                                        "model": name,
                                        "feature": feature_cols,
                                        "importance": abs_coef,
                                    }
                                )
                            )
                except Exception as me:
                    print(f"⚠️ {ticker} / {name}: model failed — {me}")

            print(f"✅ Trained {ticker} ({', '.join(models.keys())})")
            return True

        except Exception as e:
            skipped_tickers.append({"ticker": ticker, "reason": f"Error processing — {e}"})
            return False

    trained = 0
    for ticker in required_symbols:
        t = str(ticker).strip().upper()
        if t in ("", "NAN", "NONE"):
            continue

        if t not in available_parquets:
            skipped_tickers.append(
                {
                    "ticker": t,
                    "reason": f"No parquet data (preferred {PROCESSED_DIR}/{t}.parquet, fallback {RESULTS_DIR}/{t}.parquet)",
                }
            )
            continue

        df_t = load_ticker_parquet(t)
        if "ticker" in df_t.columns:
            found = {str(v).strip().upper() for v in df_t["ticker"].dropna().tolist()}
            found.discard("")
            if found and found != {t}:
                skipped_tickers.append({"ticker": t, "reason": "wrong_symbol"})
                continue
        dates = (
            pd.to_datetime(df_t["date"], errors="coerce")
            if "date" in df_t.columns
            else pd.Series(dtype="datetime64[ns]")
        )
        if not dates.empty and bool(dates.duplicated().any()):
            skipped_tickers.append({"ticker": t, "reason": "duplicate_session"})
            continue
        if governed_batch and not dates.empty and dates.max().date().isoformat() != target_session:
            skipped_tickers.append({"ticker": t, "reason": "stale_processed_maximum"})
            continue
        source = os.path.join(PROCESSED_DIR, f"{t}.parquet")
        if not os.path.exists(source):
            source = os.path.join(RESULTS_DIR, f"{t}.parquet")
        with open(source, "rb") as source_fh:
            processed_sha = hashlib.sha256(source_fh.read()).hexdigest()
        ok = train_one_ticker(t, df_t, args.min_rows, processed_sha)
        trained += int(ok)

    missing_required = [symbol for symbol in required_symbols if symbol not in staged_predictions]
    if governed_batch and missing_required:
        for symbol in missing_required:
            if not any(row.get("ticker") == symbol for row in skipped_tickers):
                skipped_tickers.append({"ticker": symbol, "reason": "missing_required_prediction"})
        print(f"Prediction batch incomplete: {missing_required}")
        if skipped_tickers:
            skipped_df = pd.DataFrame(skipped_tickers)
            skipped_df["ticker"] = skipped_df["ticker"].astype(str).str.strip().str.upper()
            atomic_publish_csv(skipped_df, SKIPPED_TICKERS_LOG)
        return 4

    staged_tmp: list[tuple[str, str]] = []
    try:
        for symbol, frame in staged_predictions.items():
            target = os.path.join(PREDICTIONS_DIR, f"{symbol}_predictions.parquet")
            abs_target = os.path.abspath(target)
            os.makedirs(os.path.dirname(abs_target) or ".", exist_ok=True)
            tmp = _unique_sibling_tmp(abs_target)
            buf = io.BytesIO()
            frame.to_parquet(buf, index=False)
            with open(tmp, "wb") as fh:
                fh.write(buf.getvalue())
                fh.flush()
                os.fsync(fh.fileno())
            staged_tmp.append((tmp, abs_target))
    except Exception as exc:
        for tmp, _abs_target in staged_tmp:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        print(f"Prediction staging failed; nothing published: {exc}")
        return 5

    replaced: list[str] = []
    try:
        for tmp, abs_target in staged_tmp:
            os.replace(tmp, abs_target)
            replaced.append(abs_target)
    except Exception as exc:
        print(f"Atomic prediction replacement failed once after {replaced}: {exc}")
        return 5

    if all_feature_importance:
        atomic_publish_csv(pd.concat(all_feature_importance, ignore_index=True), FEATURES_OUT_PATH)
        print(f"\n📊 Feature importance saved to: {FEATURES_OUT_PATH}")

    if all_model_comparison:
        mc_df = pd.concat(all_model_comparison, ignore_index=True)
        mc_df["date"] = pd.to_datetime(mc_df["date"], errors="coerce")
        atomic_publish_csv(mc_df, MODEL_COMPARISON_PATH)
        print(f"📈 Model comparison saved to: {MODEL_COMPARISON_PATH}")
    else:
        print("⚠️ No model comparison rows were generated.")

    if skipped_tickers:
        skipped_df = pd.DataFrame(skipped_tickers)
        skipped_df["ticker"] = skipped_df["ticker"].astype(str).str.strip().str.upper()
        skipped_df = skipped_df[~skipped_df["ticker"].isin(["", "NAN", "NONE"])]
        atomic_publish_csv(skipped_df, SKIPPED_TICKERS_LOG)
        print(f"📄 Skipped tickers log saved to: {SKIPPED_TICKERS_LOG}")

    print(f"\n🏁 Done. Trained {trained} tickers.")

    write_heartbeat(
        status="ok",
        stage="train",
        last_success_stage="train",
        message=f"Training complete. Trained {trained} tickers.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
