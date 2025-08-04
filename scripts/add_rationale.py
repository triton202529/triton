import pandas as pd

# Load inputs
signals = pd.read_csv("data/results/signals.csv", parse_dates=["date"])
fundamentals = pd.read_csv("data/results/fundamentals.csv")
scores = pd.read_csv("data/results/stock_scores.csv")
news = pd.read_csv("data/results/news_sentiment.csv", low_memory=False)

# Merge all data
merged = signals.merge(fundamentals, on="ticker", how="left")
merged = merged.merge(scores[["ticker", "total_score"]], on="ticker", how="left")
merged = merged.merge(news[["ticker", "sentiment"]], on="ticker", how="left")

# ✅ Generate 'confidence' column if missing
if "confidence" not in merged.columns:
    merged["confidence"] = abs((merged["predicted_close"] - merged["close"]) / merged["close"]).round(4)

# Define rationale logic
def get_rationale(row):
    reasons = []

    if row["signal"] == "BUY":
        if pd.notna(row["pe_ratio"]) and row["pe_ratio"] < 25:
            reasons.append("undervalued (low PE)")
        if pd.notna(row["total_score"]) and row["total_score"] > 50:
            reasons.append("strong fundamentals")
        if pd.notna(row.get("sentiment")) and row["sentiment"] > 0.1:
            reasons.append("positive news sentiment")

    elif row["signal"] == "SELL":
        if pd.notna(row["pe_ratio"]) and row["pe_ratio"] > 40:
            reasons.append("overvalued (high PE)")
        if pd.notna(row.get("sentiment")) and row["sentiment"] < -0.1:
            reasons.append("negative news")
        if pd.notna(row["total_score"]) and row["total_score"] < 40:
            reasons.append("weak fundamentals")

    return "; ".join(reasons) if reasons else "N/A"

# Apply rationale
merged["rationale"] = merged.apply(get_rationale, axis=1)

# Final column order
final_columns = [
    "date", "ticker", "close", "predicted_close", "signal",
    "pe_ratio", "eps", "revenue", "market_cap", "pb_ratio", "dividend_yield",
    "total_score", "sentiment", "rationale", "confidence"
]

# Save
output_path = "data/results/signals_with_rationale.csv"
merged[final_columns].to_csv(output_path, index=False)
print(f"✅ Rationale file saved to: {output_path}")
