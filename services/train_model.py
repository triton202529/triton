import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor

# Paths
RESULTS_DIR = "data/results"
PREDICTIONS_DIR = "data/predictions"
FUNDAMENTALS_PATH = os.path.join(RESULTS_DIR, "fundamentals.csv")
SCORES_PATH = os.path.join(RESULTS_DIR, "stock_scores.csv")  # ✅ fixed filename
FEATURES_OUT_PATH = os.path.join(RESULTS_DIR, "feature_importance.csv")

# Ensure output directory exists
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

print("🧠 Starting model training with feature importance extraction...")

# Load fundamentals
if not os.path.exists(FUNDAMENTALS_PATH):
    print(f"❌ fundamentals.csv not found at {FUNDAMENTALS_PATH}")
    exit(1)
fundamentals_df = pd.read_csv(FUNDAMENTALS_PATH)
fundamentals_df["ticker"] = fundamentals_df["ticker"].str.upper()

# Load scores
if not os.path.exists(SCORES_PATH):
    print(f"❌ stock_scores.csv not found at {SCORES_PATH} — run score_stocks.py first")
    exit(1)
scores_df = pd.read_csv(SCORES_PATH)
scores_df["ticker"] = scores_df["ticker"].str.upper()

# Collect all ticker parquet files
parquet_files = [
    f for f in os.listdir(RESULTS_DIR)
    if f.endswith(".parquet") and f != "stock_data.parquet"
]

if not parquet_files:
    print(f"❌ No .parquet files found in {RESULTS_DIR} — run fetch_and_prepare.py first")
    exit(1)

all_feature_importance = []

for file in parquet_files:
    ticker = file.replace(".parquet", "").upper()
    file_path = os.path.join(RESULTS_DIR, file)

    try:
        df = pd.read_parquet(file_path)

        if df.empty or len(df) < 30:
            print(f"⚠️ {ticker}: not enough data, but proceeding with available records")

        df = df.sort_values("date")
        df["target"] = df["close"].shift(-1)
        df.dropna(inplace=True)

        # Merge fundamentals (fill defaults if missing)
        fund_row = fundamentals_df[fundamentals_df["ticker"] == ticker]
        if fund_row.empty:
            print(f"⚠️ Missing fundamentals for {ticker} — using defaults")
            fund_row = pd.DataFrame([{
                "ticker": ticker,
                "pe_ratio": 15,
                "eps": 5,
                "market_cap": 1e10,
                "pb_ratio": 1.5,
                "dividend_yield": 0
            }])

        # Merge scores (fill defaults if missing)
        score_row = scores_df[scores_df["ticker"] == ticker]
        if score_row.empty:
            print(f"⚠️ Missing score for {ticker} — using default score=50")
            score_row = pd.DataFrame([{"ticker": ticker, "total_score": 50}])

        # Add fundamentals + score to dataset
        for col in ["pe_ratio", "eps", "market_cap", "pb_ratio", "dividend_yield"]:
            df[col] = fund_row.iloc[0][col]
        df["total_score"] = score_row.iloc[0]["total_score"]

        # Define features
        base_cols = ["open", "high", "low", "close", "volume"]
        feature_cols = base_cols + [
            "pe_ratio", "eps", "market_cap", "pb_ratio", "dividend_yield", "total_score"
        ]

        # Fill missing OHLCV columns with zeros
        for col in base_cols:
            if col not in df.columns:
                print(f"⚠️ {ticker}: Missing {col}, filling with 0")
                df[col] = 0

        # Train model
        X = df[feature_cols].fillna(0)
        y = df["target"].fillna(df["target"].median())  # fill missing targets with median
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

        print(f"✅ Trained {ticker} + feature importance extracted")

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

# Save all feature importances combined
if all_feature_importance:
    combined = pd.concat(all_feature_importance, ignore_index=True)
    combined.to_csv(FEATURES_OUT_PATH, index=False)
    print(f"\n📊 Feature importance saved to: {FEATURES_OUT_PATH}")

print("\n🏁 All models trained with feature importance extracted.")
