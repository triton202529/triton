import os
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

# Paths
RESULTS_DIR = "data/results"
PREDICTIONS_DIR = "data/predictions"
FUNDAMENTALS_PATH = os.path.join(RESULTS_DIR, "fundamentals.csv")
SCORES_PATH = os.path.join(RESULTS_DIR, "stock_scores.csv")
FEATURES_OUT_PATH = os.path.join(RESULTS_DIR, "feature_importance.csv")
SKIPPED_TICKERS_LOG = os.path.join(RESULTS_DIR, "skipped_tickers.csv")
MODEL_COMPARISON_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")

# Ensure output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

print("🧠 Starting multi-model training with feature importance & model comparison export...")

# Prepare logs/collectors
skipped_tickers = []
all_feature_importance = []
all_model_comparison = []

# --- Load fundamentals ---
if not os.path.exists(FUNDAMENTALS_PATH) or os.path.getsize(FUNDAMENTALS_PATH) == 0:
    print(f"❌ fundamentals.csv missing or empty at {FUNDAMENTALS_PATH}")
    raise SystemExit(1)
fundamentals_df = pd.read_csv(FUNDAMENTALS_PATH)
fundamentals_df["ticker"] = fundamentals_df["ticker"].str.upper()

# --- Load scores ---
if not os.path.exists(SCORES_PATH) or os.path.getsize(SCORES_PATH) == 0:
    print(f"❌ stock_scores.csv missing or empty at {SCORES_PATH} — run score_stocks.py first")
    raise SystemExit(1)
scores_df = pd.read_csv(SCORES_PATH)
scores_df["ticker"] = scores_df["ticker"].str.upper()

# --- Collect all ticker parquet files prepared previously ---
parquet_files = [
    f for f in os.listdir(RESULTS_DIR)
    if f.endswith(".parquet") and f != "stock_data.parquet"
]

if not parquet_files:
    print(f"❌ No .parquet files found in {RESULTS_DIR} — run fetch_and_prepare.py first")
    raise SystemExit(1)

# --- Define model zoo ---
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "LinearRegression": LinearRegression()
}
if HAS_XGB:
    # Reasonable defaults; tree_method left to auto
    models["XGBoost"] = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, random_state=42, n_jobs=-1
    )

# --- Train per ticker ---
for file in parquet_files:
    ticker = file.replace(".parquet", "").upper()
    file_path = os.path.join(RESULTS_DIR, file)

    try:
        df = pd.read_parquet(file_path)

        if df.empty or len(df) < 30:
            print(f"⚠️ {ticker}: Empty dataset or too few rows, skipping.")
            skipped_tickers.append({"ticker": ticker, "reason": "Empty or insufficient data"})
            continue

        df = df.sort_values("date").reset_index(drop=True)
        df["target"] = df["close"].shift(-1)
        df.dropna(inplace=True)

        # Merge fundamental + score data
        fund_row = fundamentals_df[fundamentals_df["ticker"] == ticker]
        score_row = scores_df[scores_df["ticker"] == ticker]

        if fund_row.empty or score_row.empty:
            print(f"⚠️ {ticker}: Missing fundamentals or score, skipping.")
            skipped_tickers.append({"ticker": ticker, "reason": "Missing fundamentals or score"})
            continue

        # Add fundamentals + score to dataset
        for col in ["pe_ratio", "eps", "market_cap", "pb_ratio", "dividend_yield"]:
            df[col] = fund_row.iloc[0][col]
        df["total_score"] = score_row.iloc[0]["total_score"]

        # Define features
        base_cols = ["open", "high", "low", "close", "volume"]
        feature_cols = base_cols + [
            "pe_ratio", "eps", "market_cap", "pb_ratio", "dividend_yield", "total_score"
        ]

        if not all(col in df.columns for col in base_cols):
            print(f"⚠️ {ticker}: Missing OHLCV columns, skipping.")
            skipped_tickers.append({"ticker": ticker, "reason": "Missing OHLCV columns"})
            continue

        X = df[feature_cols].fillna(0)
        y = df["target"]

        # Keep RandomForest as baseline output for backwards compatibility
        rf_model = models["RandomForest"]
        rf_model.fit(X, y)
        df["predicted_close"] = rf_model.predict(X)

        # Generate simple signals from RF baseline
        df["signal"] = "HOLD"
        df.loc[df["predicted_close"] > df["close"], "signal"] = "BUY"
        df.loc[df["predicted_close"] < df["close"], "signal"] = "SELL"

        # Save per‑ticker RF predictions (unchanged path/format)
        output_df = df[["date", "close", "predicted_close", "signal"]].copy()
        output_df["ticker"] = ticker
        output_path = os.path.join(PREDICTIONS_DIR, f"{ticker}_predictions.parquet")
        output_df.to_parquet(output_path, index=False)

        # Collect RF feature importance
        fi = getattr(rf_model, "feature_importances_", None)
        if fi is not None:
            all_feature_importance.append(pd.DataFrame({
                "ticker": ticker,
                "model": "RandomForest",
                "feature": feature_cols,
                "importance": fi
            }))

        # --- Train all comparison models and gather predictions ---
        for name, model in models.items():
            try:
                # For LinearRegression, GradientBoosting, XGB etc.
                model.fit(X, y)
                preds = model.predict(X)

                # Append to model comparison table
                temp = pd.DataFrame({
                    "ticker": ticker,
                    "date": df["date"],
                    "model": name,
                    "close": df["close"].values,          # actual close at t
                    "predicted_close": preds              # predicted close at t+1 target space, but chart will align on t
                })
                all_model_comparison.append(temp)

                # Feature importance / coefficients if available
                if hasattr(model, "feature_importances_"):
                    imp_vals = model.feature_importances_
                    all_feature_importance.append(pd.DataFrame({
                        "ticker": ticker,
                        "model": name,
                        "feature": feature_cols,
                        "importance": imp_vals
                    }))
                elif hasattr(model, "coef_"):
                    # LinearRegression: use absolute normalized coefficients as "importance"
                    coef = model.coef_
                    # Guard for shapes (coef may be 1-D)
                    coef = coef.ravel() if hasattr(coef, "ravel") else coef
                    abs_coef = (abs(coef) / (abs(coef).sum() if abs(coef).sum() != 0 else 1.0))
                    # Match feature length if needed
                    if len(abs_coef) == len(feature_cols):
                        all_feature_importance.append(pd.DataFrame({
                            "ticker": ticker,
                            "model": name,
                            "feature": feature_cols,
                            "importance": abs_coef
                        }))

            except Exception as me:
                print(f"⚠️ {ticker} / {name}: model failed — {me}")
                continue

        print(f"✅ Trained {ticker} ({', '.join(models.keys())})")

    except Exception as e:
        print(f"❌ {ticker}: Error processing — {e}")
        skipped_tickers.append({"ticker": ticker, "reason": f"Error: {e}"})

# --- Save combined feature importances ---
if all_feature_importance:
    combined_fi = pd.concat(all_feature_importance, ignore_index=True)
    combined_fi.to_csv(FEATURES_OUT_PATH, index=False)
    print(f"\n📊 Feature importance (all models) saved to: {FEATURES_OUT_PATH}")

# --- Save model comparison table ---
if all_model_comparison:
    mc_df = pd.concat(all_model_comparison, ignore_index=True)
    # Ensure consistent dtypes
    if "date" in mc_df.columns:
        mc_df["date"] = pd.to_datetime(mc_df["date"], errors="coerce")
    mc_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"📈 Model comparison saved to: {MODEL_COMPARISON_PATH}")
else:
    print("⚠️ No model comparison rows were generated.")

# --- Save skipped tickers log ---
if skipped_tickers:
    skipped_df = pd.DataFrame(skipped_tickers)
    skipped_df.to_csv(SKIPPED_TICKERS_LOG, index=False)
    print(f"📄 Skipped tickers log saved to: {SKIPPED_TICKERS_LOG}")

print("\n🏁 All models trained. RF predictions written per ticker, model comparison + feature importances exported.")
