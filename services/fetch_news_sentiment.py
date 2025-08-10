# services/fetch_news_sentiment.py

import os
import time
import argparse
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# === Config ===
NEWSAPI_KEY = "36536909146a411683bc7ccecab398b7"
RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_LOOKBACK_DAYS = 30
DEFAULT_PAGE_SIZE = 20
DEFAULT_MAX_PAGES = 1
REQUEST_TIMEOUT = 20
RATE_LIMIT_SLEEP = 15
HTTP_RETRY = 2
MAX_429_RETRIES = 3
PER_BATCH_SLEEP = 2
PER_TICKER_SLEEP = 1

NAME_SYNONYMS = {
    "AAPL": ["AAPL", "Apple"],
    "MSFT": ["MSFT", "Microsoft"],
    "NVDA": ["NVDA", "Nvidia", "NVIDIA"],
    "AMZN": ["AMZN", "Amazon"],
    "META": ["META", "Meta", "Facebook"],
    "GOOGL": ["GOOGL", "Google", "Alphabet"],
    "TSLA": ["TSLA", "Tesla"],
}

analyzer = SentimentIntensityAnalyzer()

def clamp_window(days_requested: int) -> int:
    if days_requested > MAX_LOOKBACK_DAYS:
        print(f"ℹ️ Window {days_requested}d > max {MAX_LOOKBACK_DAYS}d. Clamping to {MAX_LOOKBACK_DAYS} days.")
    return min(days_requested, MAX_LOOKBACK_DAYS)

def analyze_sentiment(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    return analyzer.polarity_scores(text)["compound"]

def _request(url: str, params: dict, label: str):
    tries = 0
    while True:
        tries += 1
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            if tries <= HTTP_RETRY:
                time.sleep(2); continue
            print(f"⚠️ {label}: request error: {e}")
            return None
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            if tries >= MAX_429_RETRIES:
                print(f"🛑 Too many 429s for {label}; skipping.")
                return None
            print(f"⏳ Rate limited on {label}. Sleeping {RATE_LIMIT_SLEEP}s… ({tries}/{MAX_429_RETRIES})")
            time.sleep(RATE_LIMIT_SLEEP); continue
        print(f"⚠️ {label}: {r.status_code} {r.text[:200]}...")
        return None

def fetch_everything(q: str, from_str: str, to_str: str, page_size: int, page: int):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q, "from": from_str, "to": to_str,
        "sortBy": "publishedAt", "language": "en",
        "pageSize": page_size, "page": page, "apiKey": NEWSAPI_KEY,
    }
    return _request(url, params, f"everything p{page}")

def fetch_top_headlines(page_size: int = 100, page: int = 1, country: str = "us", category: str = "business"):
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": country, "category": category,
        "pageSize": page_size, "page": page, "apiKey": NEWSAPI_KEY,
    }
    return _request(url, params, f"top-headlines p{page}")

def tag_tickers(article: dict, tickers: list[str]) -> list[str]:
    text = f"{article.get('title','')} {article.get('description','')} {article.get('content','')}"
    text_lower = text.lower()
    matched = set()
    for t in tickers:
        for s in NAME_SYNONYMS.get(t, [t]):
            if s and s.lower() in text_lower:
                matched.add(t); break
    return sorted(matched)

def aggregate_mode(tickers: list[str], window: int, output_path: str,
                   page_size: int, max_pages: int, batch_size: int):
    """Batch query /everything; if rate-limited or empty, fallback to /top-headlines once."""
    end_dt = datetime.now(timezone.utc)
    window = clamp_window(window)
    start_dt = end_dt - timedelta(days=window)
    from_str, to_str = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    all_rows = []
    any_success = False

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        q_parts = []
        for t in batch:
            for n in NAME_SYNONYMS.get(t, [t]):
                q_parts.append(f"\"{n}\"")
        q = f"({' OR '.join(q_parts)})"

        print(f"📰 Batch {i // batch_size + 1}: fetching for {', '.join(batch)}")
        batch_articles = []
        for page in range(1, max_pages + 1):
            r = fetch_everything(q, from_str, to_str, page_size, page)
            if r is None:
                break
            items = r.json().get("articles", []) or []
            batch_articles.extend(items)
            if len(items) < page_size:
                break

        if batch_articles:
            any_success = True

        if not batch_articles:
            time.sleep(PER_BATCH_SLEEP)
            continue

        for a in batch_articles:
            hits = tag_tickers(a, batch)
            if not hits:
                continue
            title = a.get("title", "") or ""
            desc = a.get("description", "") or ""
            pub_at = a.get("publishedAt", "") or ""
            sent = analyze_sentiment(f"{title} {desc}")
            pub_dt = pd.to_datetime(pub_at, utc=True, errors="coerce")
            row_date = (pub_dt.date() if pd.notna(pub_dt) else start_dt.date())
            for t in hits:
                all_rows.append({
                    "ticker": t,
                    "title": title,
                    "description": desc,
                    "publishedAt": pub_at,
                    "sentiment": sent,
                    "date": row_date
                })

        if all_rows:
            pd.DataFrame(all_rows).drop_duplicates(subset=["ticker", "title", "publishedAt"]).to_csv(output_path, index=False)
            print(f"💾 Progress saved → {output_path} (rows: {len(all_rows)})")

        time.sleep(PER_BATCH_SLEEP)

    # If /everything produced nothing, fallback to /top-headlines once
    if not any_success and not all_rows:
        print("↩️ Falling back to /top-headlines (single request)…")
        r = fetch_top_headlines(page_size=100, page=1, country="us", category="business")
        if r is None:
            print("🚫 No news articles found (fallback failed)."); return
        items = r.json().get("articles", []) or []
        if not items:
            print("🚫 No news articles found in top-headlines."); return
        for a in items:
            hits = tag_tickers(a, tickers)
            if not hits:
                continue
            title = a.get("title", "") or ""
            desc = a.get("description", "") or ""
            pub_at = a.get("publishedAt", "") or ""
            sent = analyze_sentiment(f"{title} {desc}")
            pub_dt = pd.to_datetime(pub_at, utc=True, errors="coerce")
            row_date = (pub_dt.date() if pd.notna(pub_dt) else end_dt.date())
            for t in hits:
                all_rows.append({
                    "ticker": t,
                    "title": title,
                    "description": desc,
                    "publishedAt": pub_at,
                    "sentiment": sent,
                    "date": row_date
                })

    if not all_rows:
        print("🚫 No news articles found."); return

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["ticker", "title", "publishedAt"])
    df.to_csv(output_path, index=False)
    print(f"✅ Saved news sentiment to: {output_path}")
    print(df.head(8))

def by_ticker_mode(tickers: list[str], window: int, output_path: str,
                   page_size: int, max_pages: int):
    end_dt = datetime.now(timezone.utc)
    window = clamp_window(window)
    start_dt = end_dt - timedelta(days=window)
    from_str, to_str = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    all_rows = []

    for t in tickers:
        print(f"📰 Fetching news for {t} ({from_str} → {to_str})...")
        for page in range(1, max_pages + 1):
            r = fetch_everything(f"\"{t}\"", from_str, to_str, page_size, page)
            if r is None:
                break
            items = r.json().get("articles", []) or []
            if not items:
                break
            for a in items:
                title = a.get("title", "") or ""
                desc = a.get("description", "") or ""
                pub_at = a.get("publishedAt", "") or ""
                sent = analyze_sentiment(f"{title} {desc}")
                pub_dt = pd.to_datetime(pub_at, utc=True, errors="coerce")
                row_date = (pub_dt.date() if pd.notna(pub_dt) else start_dt.date())
                all_rows.append({
                    "ticker": t,
                    "title": title,
                    "description": desc,
                    "publishedAt": pub_at,
                    "sentiment": sent,
                    "date": row_date
                })
            if len(items) < page_size:
                break

        if all_rows:
            pd.DataFrame(all_rows).drop_duplicates(subset=["ticker", "title", "publishedAt"]).to_csv(output_path, index=False)
            print(f"💾 Progress saved → {output_path} (rows: {len(all_rows)})")
        time.sleep(PER_TICKER_SLEEP)

    if not all_rows:
        print("🚫 No news articles found."); return

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["ticker", "title", "publishedAt"])
    df.to_csv(output_path, index=False)
    print(f"✅ Saved news sentiment to: {output_path}")
    print(df.head(8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default="all", help="Comma-separated tickers or 'all'")
    parser.add_argument("--window", type=int, default=7, help="Days back (auto-clamped to 30)")
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "news_sentiment.csv"))
    parser.add_argument("--page_size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max_pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--strategy", choices=["aggregate", "by_ticker"], default="aggregate")
    parser.add_argument("--batch", type=int, default=10, help="Tickers per batch in aggregate mode")
    args = parser.parse_args()

    if args.tickers == "all":
        tickers_file = os.path.join(RESULTS_DIR, "stock_scores.csv")
        if os.path.exists(tickers_file):
            df_tickers = pd.read_csv(tickers_file)
            tickers = sorted(df_tickers["ticker"].dropna().unique().tolist())
        else:
            tickers = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.strategy == "aggregate":
        aggregate_mode(tickers, args.window, args.out, args.page_size, args.max_pages, args.batch)
    else:
        by_ticker_mode(tickers, args.window, args.out, args.page_size, args.max_pages)
