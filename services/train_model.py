# services/train_model.py

import pandas as pd
import os
import subprocess
from sklearn.ensemble import RandomForestRegressor

# Paths
RESULTS_DIR = "data/results"
PROCESSED_DIR = "data/processed"
PREDICTIONS_DIR = "data/predictions"
FUNDAMENTALS_PATH = os.path.join(RESULTS_DIR, "fundamentals.csv")
SCORES_PATH = os.path.join(RESULTS_DIR, "stock_scores.csv")  # ✅ fixed filename
FEATURES_OUT_PATH = os.path.join(RESULTS_DIR, "feature_importance.csv")

# Ensure output directory exists
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

print("🧠 Starting model training with feature importance extraction...")

# 🔹 Step 1: Ensure price data exists
if not any(f.endswith(".parquet") and f != "stock_data.parquet" for f in os.listdir(RESULTS_DIR) if os.path.exists(RESULTS_DIR)):
    print(f"⚠️ No ticker .parquet files found in {RESULTS_DIR} — running fetch_and_prepare.py...")
    try:
        subprocess.run(["python", "scripts/fetch_and_prepare.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run fetch_and_prepare.py: {e}")
        exit(1)

# 🔹 Step 2: Ensure fundamentals.csv exists
if not os.path.exists(FUNDAMENTALS_PATH):
    print(f"⚠️ {FUNDAMENTALS_PATH} not found — running fetch_fundamentals.py...")
    try:
        subprocess.run(["python", "services/fetch_fundamentals.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run fetch_fundamentals.py: {e}")
        exit(1)

if not os.path.exists(FUNDAMENTALS_PATH):
    print(f"❌ fundamentals.csv still not found at {FUNDAMENTALS_PATH} after running fetch_fundamentals.py")
    exit(1)

fundamentals_df = pd.read_csv(FUNDAMENTALS_PATH)
fundamentals_df["ticker"] = fundamentals_df["ticker"].str.upper()

# 🔹 Step 3: Ensure stock_scores.csv exists
if not os.path.exists(SCORES_PATH):
    print(f"⚠️ {SCORES_PATH} not found — running score_stocks.py...")
    try:
        subprocess.run(["python", "services/score_stocks.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run score_stocks.py: {e}")
        exit(1)

if not os.path.exists(SCORES_PATH):
    print(f"❌ stock_scores.csv still not found at {SCORES_PATH} after running score_stocks.py")
    exit(1)

scores_df = pd.read_csv(SCORES_PATH)
scores_df["ticker"] = scores_df["ticker"].str.upper()

# 🔹 Step 4: Get all per-ticker .parquet files
parquet_files = [
    f for f in os.listdir(RESULTS_DIR)
    if f.endswith(".parquet") and f != "stock_data.parquet"
]

if not parquet_files:
    print(f"❌ No .parquet files found in {RESULTS_DIR} even after running fetch_and_prepare.py")
    exit(1)

# 🔹 Step 5: Train model for each ticker
all_feature_importance = []

for file in parquet_files:
    ticker = file.replace(".parquet", "").upper()
    file_path = os.path.join(RESULTS_DIR, file)

    try:
        df = pd.read_parquet(file_path)

        if df.empty or len(df) < 30:
            print(f"⚠️ Skipping {ticker} — not enough data")
            continue

        df = df.sort_values("date")
        df["target"] = df["close"].shift(-1)
        df.dropna(inplace=True)

        # Merge fundamental + score data
        fund_row = fundamentals_df[fundamentals_df["ticker"] == ticker]
        score_row = scores_df[scores_df["ticker"] == ticker]

        if fund_row.empty or score_row.empty:
            print(f"⚠️ Skipping {ticker} — missing fundamentals or score")
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
            print(f"⚠️ Skipping {ticker} — missing OHLCV columns")
            continue

        # Train model
        X = df[feature_cols].fillna(0)
        y = df["target"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        df["predicted_close"] = model.predict(X)

        # Generate signals
        df["signal"] = "HOLD"
        df.loc[df["predicted_close"] > df["close"], "signal"] = "BUY"
        df.loc[df["predicted_close"] < df["close"], "signal"] = "SELL"

        # Save predictions
        output_df = df[["date", "close", "predicted_close", "signal"]].copy()
        output_df["ticker"] = ticker
        output_path = os.path.join(PREDICTIONS_DIR, f"{ticker}_predictions.parquet")
        output_df.to_parquet(output_path, index=False)

        # Save feature importances
        importance_df = pd.DataFrame({
            "ticker": ticker,
            "feature": feature_cols,
            "importance": model.feature_importances_
        })
        all_feature_importance.append(importance_df)

        print(f"✅ Trained + feature importance extracted for {ticker}")

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

# 🔹 Step 6: Save all feature importances combined
if all_feature_importance:
    combined = pd.concat(all_feature_importance, ignore_index=True)
    combined.to_csv(FEATURES_OUT_PATH, index=False)
    print(f"\n📊 Feature importance saved to: {FEATURES_OUT_PATH}")

print("\n🏁 All models trained with feature importance extracted.")
