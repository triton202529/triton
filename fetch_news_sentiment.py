import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from textblob import TextBlob
from newsapi import NewsApiClient

# === Configuration ===
API_KEY = "36536909146a411683bc7ccecab398b7"  # Replace with your own if needed
TICKERS = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "JPM", "XOM"]
OUTPUT_PATH = "data/results/news_sentiment.csv"

# === Setup ===
newsapi = NewsApiClient(api_key=API_KEY)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("📰 Fetching news and analyzing sentiment...")

all_data = []
from_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

for ticker in TICKERS:
    try:
        articles = newsapi.get_everything(
            q=ticker,
            language="en",
            sort_by="relevancy",
            from_param=from_date,
            to=to_date,
            page_size=20,
        )

        for article in articles.get("articles", []):
            title = article.get("title", "")
            description = article.get("description", "")
            content = f"{title} {description}"

            sentiment_score = TextBlob(content).sentiment.polarity

            all_data.append({
                "ticker": ticker,
                "title": title,
                "description": description,
                "publishedAt": article.get("publishedAt"),
                "sentiment": sentiment_score
            })

    except Exception as e:
        print(f"❌ Error fetching for {ticker}: {e}")

# === Save ===
if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ News sentiment saved to {OUTPUT_PATH}")
else:
    print("⚠️ No news data collected.")
