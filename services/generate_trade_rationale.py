import pandas as pd

# Load inputs
signals = pd.read_csv("data/results/signals.csv")
fundamentals = pd.read_csv("data/results/fundamentals.csv")
scores = pd.read_csv("data/results/stock_scores.csv")
news = pd.read_csv("data/results/news_sentiment.csv", low_memory=False)

# Merge all data
merged = signals.merge(fundamentals, on="ticker", how="left")
merged = merged.merge(scores[["ticker", "total_score"]], on="ticker", how="left")
merged = merged.merge(news[["ticker", "sentiment"]], on="ticker", how="left")

# Define rationale logic
def get_rationale(row):
    reasons = []

    if row["signal"] == "BUY":
        if row["pe_ratio"] and row["pe_ratio"] < 25:
            reasons.append("undervalued (low PE)")
        if row["total_score"] and row["total_score"] > 50:
            reasons.append("strong fundamentals")
        if row.get("sentiment", 0) > 0.1:
            reasons.append("positive news sentiment")

    elif row["signal"] == "SELL":
        if row["pe_ratio"] and row["pe_ratio"] > 40:
            reasons.append("overvalued (high PE)")
        if row.get("sentiment", 0) < -0.1:
            reasons.append("negative news")
        if row["total_score"] and row["total_score"] < 40:
            reasons.append("weak fundamentals")

    return "; ".join(reasons) if reasons else "N/A"

# Apply rationale logic
merged["rationale"] = merged.apply(get_rationale, axis=1)

# Add 'confidence' column if missing
if "confidence" not in merged.columns:
    merged["confidence"] = (merged["close"] - merged["predicted_close"]).abs() / merged["close"]
    merged["confidence"] = merged["confidence"].round(4)

# Save output
output_path = "data/results/signals_with_rationale.csv"
merged.to_csv(output_path, index=False)
print(f"✅ Rationale file saved to: {output_path}")
