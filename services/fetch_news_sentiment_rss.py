# services/fetch_news_sentiment_rss.py
import os, time, argparse, html
import pandas as pd
import feedparser
from urllib.parse import quote_plus
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()

# Optional synonyms used for tagging/queries
NAME_SYNONYMS = {
    "AAPL": ["AAPL", "Apple"],
    "MSFT": ["MSFT", "Microsoft"],
    "NVDA": ["NVDA", "Nvidia", "NVIDIA"],
    "AMZN": ["AMZN", "Amazon"],
    "META": ["META", "Meta", "Facebook"],
    "GOOGL": ["GOOGL", "Google", "Alphabet"],
    "TSLA": ["TSLA", "Tesla"],
}

def analyze(text: str) -> float:
    if not isinstance(text, str): return 0.0
    return analyzer.polarity_scores(text)["compound"]

def google_news_rss_url(query: str, window_days: int | None):
    # Google News RSS search. If window_days is provided, add when:dN
    q = query if window_days is None else f"{query} when:d{window_days}"
    qp = quote_plus(q)
    return f"https://news.google.com/rss/search?q={qp}&hl=en-US&gl=US&ceid=US:en"

def yahoo_finance_rss_url(ticker: str):
    # Yahoo Finance RSS (not rate-limited like NewsAPI; works for many tickers)
    # If this ever fails, try feeds.finance.yahoo.com variant.
    return f"https://finance.yahoo.com/rss/headline?s={quote_plus(ticker)}"

def parse_entries(feed):
    rows = []
    for e in feed.entries:
        title = html.unescape(getattr(e, "title", "") or "")
        desc = html.unescape(getattr(e, "summary", "") or "")
        pub = getattr(e, "published", "") or getattr(e, "updated", "") or ""
        link = getattr(e, "link", "") or ""
        pub_dt = pd.to_datetime(pub, utc=True, errors="coerce")
        rows.append({
            "title": title,
            "description": desc,
            "publishedAt": pub,
            "published_dt": pub_dt,
            "link": link,
            "text": f"{title} {desc}",
        })
    return rows

def tag_mentions(text: str, ticker: str) -> bool:
    lower = (text or "").lower()
    for s in NAME_SYNONYMS.get(ticker, [ticker]):
        if s.lower() in lower:
            return True
    return False

def fetch_for_ticker(ticker: str, window: int, sleep_s: float = 0.8):
    """Try Google (with window) → Google (no window) → Yahoo Finance; return rows tagged with ticker."""
    rows = []

    # 1) Google News RSS — with date window
    synonyms = NAME_SYNONYMS.get(ticker, [ticker])
    # Build a query like: "AAPL" OR Apple (quotes help exact tickers/names)
    quoted = [f"\"{s}\"" for s in synonyms]
    query = " OR ".join(quoted)

    url_g_with = google_news_rss_url(query, window_days=window)
    feed = feedparser.parse(url_g_with)
    items = parse_entries(feed)
    if not items:
        # 2) Google News RSS — without date window (broader results)
        url_g_no = google_news_rss_url(query, window_days=None)
        feed = feedparser.parse(url_g_no)
        items = parse_entries(feed)

    # 3) Yahoo Finance RSS fallback
    if not items:
        url_y = yahoo_finance_rss_url(ticker)
        feed = feedparser.parse(url_y)
        items = parse_entries(feed)

    # Tag and keep only items that mention the ticker/synonyms
    for it in items:
        if tag_mentions(it["text"], ticker):
            sent = analyze(it["text"])
            # Default date if missing
            row_date = (it["published_dt"].date()
                        if pd.notna(it["published_dt"])
                        else datetime.now(timezone.utc).date())
            rows.append({
                "ticker": ticker,
                "title": it["title"],
                "description": it["description"],
                "publishedAt": it["publishedAt"],
                "sentiment": sent,
                "date": row_date
            })

    time.sleep(sleep_s)
    return rows

def main(tickers: list[str], window: int, out_path: str, per_ticker_sleep: float):
    all_rows = []
    for i, t in enumerate(tickers, start=1):
        print(f"📰 RSS: {t} ({i}/{len(tickers)})")
        try:
            rows = fetch_for_ticker(t, window, sleep_s=per_ticker_sleep)
        except Exception as e:
            print(f"⚠️ {t}: fetch error: {e}")
            rows = []

        if rows:
            all_rows.extend(rows)
            # incremental save
            dfp = pd.DataFrame(all_rows).drop_duplicates(subset=["ticker","title","publishedAt"])
            dfp.to_csv(out_path, index=False)
            print(f"💾 Progress → {out_path} (rows: {len(dfp)})")

    if not all_rows:
        print("🚫 No news from RSS. Try a smaller window (e.g., 7) or different tickers.")
        return

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["ticker","title","publishedAt"])
    df.to_csv(out_path, index=False)
    print(f"✅ Saved news sentiment to: {out_path}")
    print(df.head(8))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="AAPL,MSFT,NVDA,AMZN,META")
    p.add_argument("--window", type=int, default=14, help="Days back for Google News filter (best-effort)")
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "news_sentiment.csv"))
    p.add_argument("--sleep", type=float, default=0.8, help="Seconds to sleep per ticker")
    args = p.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    main(tickers, args.window, args.out, args.sleep)
