import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from textblob import TextBlob

RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SIGNALS_FILE = os.path.join(RESULTS_DIR, "signals_with_rationale.csv")
NEWS_FILE = os.path.join(RESULTS_DIR, "news_sentiment.csv")
OUT_FILE = os.path.join(RESULTS_DIR, "alerts.csv")

def load_csv(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else:
        print(f"⚠ File not found: {path}")
        return pd.DataFrame()

def make_alerts(lookback_days=30):
    sig = load_csv(SIGNALS_FILE)
    news = load_csv(NEWS_FILE)

    if sig.empty:
        print("❌ No signals data found.")
        return pd.DataFrame()
    if news.empty:
        print("⚠ No news data found, proceeding without sentiment fusion.")

    # Parse dates
    sig["date"] = pd.to_datetime(sig["date"], errors="coerce", utc=True)
    if "date" in news.columns:
        news["date"] = pd.to_datetime(news["date"], errors="coerce", utc=True)

    # Filter signals by lookback
    cutoff = pd.Timestamp(datetime.now(timezone.utc)).normalize() - pd.Timedelta(days=lookback_days)
    sig = sig[sig["date"] >= cutoff]

    # Merge with news sentiment
    if not news.empty:
        merged = pd.merge(
            sig,
            news[["ticker", "date", "sentiment", "title", "url"]],
            on=["ticker", "date"],
            how="left"
        )
    else:
        merged = sig.copy()

    # Classify priority
    def classify_priority(row):
        conf = row.get("confidence", 0)
        sent = row.get("sentiment", 0)
        if pd.isna(conf):
            conf = 0
        if pd.isna(sent):
            sent = 0
        if conf > 0.5 and sent > 0.2:
            return "HIGH"
        elif conf > 0.2:
            return "MEDIUM"
        return "LOW"

    merged["priority"] = merged.apply(classify_priority, axis=1)

    # Clickable link formatting
    if "url" in merged.columns:
        merged["url"] = merged["url"].fillna("").apply(
            lambda x: f'<a href="{x}" target="_blank">Link</a>' if x else ""
        )

    # Message column
    merged["message"] = merged.apply(
        lambda r: f"{r['ticker']}: {r['type']} | Δ={r.get('price_change',''):.2%} | conf={r.get('confidence',0):.2f}",
        axis=1
    )

    # Timestamp
    merged["timestamp"] = datetime.now(timezone.utc).isoformat()

    cols = ["date", "ticker", "type", "priority", "confidence", "sentiment", "title", "url", "message", "timestamp"]
    final_df = merged[cols].sort_values(["priority", "confidence"], ascending=[False, False])

    final_df.to_csv(OUT_FILE, index=False)
    print(f"✅ Saved {len(final_df)} alerts → {OUT_FILE}")
    return final_df

if __name__ == "__main__":
    df = make_alerts(lookback_days=30)
    if not df.empty:
        print(df.head())
