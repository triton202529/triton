import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor

# Paths
RESULTS_DIR = "data/results"
PREDICTIONS_DIR = "data/predictions"
FUNDAMENTALS_PATH = os.path.join(RESULTS_DIR, "fundamentals.csv")
SCORES_PATH = os.path.join(RESULTS_DIR, "stock_scores.csv")
FEATURES_OUT_PATH = os.path.join(RESULTS_DIR, "feature_importance.csv")

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

print("🧠 Starting model training with feature importance extraction...")

# Load fundamentals
fundamentals_df = pd.read_csv(FUNDAMENTALS_PATH) if os.path.exists(FUNDAMENTALS_PATH) else pd.DataFrame()
fundamentals_df["ticker"] = fundamentals_df.get("ticker", pd.Series()).str.upper()

# Load scores
scores_df = pd.read_csv(SCORES_PATH) if os.path.exists(SCORES_PATH) else pd.DataFrame()
scores_df["ticker"] = scores_df.get("ticker", pd.Series()).str.upper()

# Ticker files
parquet_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".parquet") and f != "stock_data.parquet"]

all_feature_importance = []

for file in parquet_files:
    ticker = file.replace(".parquet", "").upper()
    file_path = os.path.join(RESULTS_DIR, file)
    try:
        df = pd.read_parquet(file_path).sort_values("date")
        if df.empty:
            print(f"⚠️ {ticker}: Empty dataset, skipping.")
            continue

        df["target"] = df["close"].shift(-1).fillna(method="ffill")

        # Fundamentals defaults
        fund_row = fundamentals_df[fundamentals_df["ticker"] == ticker]
        if fund_row.empty:
            print(f"⚠️ Missing fundamentals for {ticker} — using defaults")
            fund_row = pd.DataFrame([{
                "ticker": ticker, "pe_ratio": 15, "eps": 5,
                "market_cap": 1e10, "pb_ratio": 1.5, "dividend_yield": 0
            }])

        # Score defaults
        score_row = scores_df[scores_df["ticker"] == ticker]
        if score_row.empty:
            print(f"⚠️ Missing score for {ticker} — using default=50")
            score_row = pd.DataFrame([{"ticker": ticker, "total_score": 50}])

        # Add features
        for col in ["pe_ratio", "eps", "market_cap", "pb_ratio", "dividend_yield"]:
            df[col] = fund_row.iloc[0][col]
        df["total_score"] = score_row.iloc[0]["total_score"]

        base_cols = ["open", "high", "low", "close", "volume"]
        for col in base_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[base_cols + ["pe_ratio", "eps", "market_cap", "pb_ratio", "dividend_yield", "total_score"]].fillna(0)
        y = df["target"].fillna(df["target"].median())

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        df["predicted_close"] = model.predict(X)

        df["signal"] = "HOLD"
        df.loc[df["predicted_close"] > df["close"], "signal"] = "BUY"
        df.loc[df["predicted_close"] < df["close"], "signal"] = "SELL"

        # Save predictions
        df[["date", "close", "predicted_close", "signal"]].assign(ticker=ticker).to_parquet(
            os.path.join(PREDICTIONS_DIR, f"{ticker}_predictions.parquet"), index=False
        )

        all_feature_importance.append(pd.DataFrame({
            "ticker": ticker, "feature": X.columns, "importance": model.feature_importances_
        }))

        print(f"✅ Trained {ticker}")

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

# Save feature importances
if all_feature_importance:
    pd.concat(all_feature_importance, ignore_index=True).to_csv(FEATURES_OUT_PATH, index=False)
    print(f"📊 Feature importance saved to {FEATURES_OUT_PATH}")

print("\n🏁 All models trained with feature importance extracted.")
