# services/fetch_econ_calendar_rss.py
import os, time, html, feedparser
import pandas as pd
from urllib.parse import quote_plus
from datetime import datetime, timezone

RESULTS_DIR = "data/results"
OUT_PATH = os.path.join(RESULTS_DIR, "economic_calendar.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

LOOKBACK_DAYS = 21  # show only the last N days
MIN_IMPORTANCE = {"High", "Medium"}  # filter
PER_QUERY_SLEEP = 0.4  # seconds

KEYWORDS = [
    "CPI", "inflation", "PPI", "FOMC", "Federal Reserve",
    "Nonfarm Payrolls", "jobs report", "unemployment rate",
    "GDP", "Retail Sales", "PMI", "ISM", "PCE"
]

SOURCES = [
    ("GoogleNews", lambda q: f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"),
    ("YahooFinance", lambda q: f"https://news.search.yahoo.com/rss?p={quote_plus(q)}"),
]

def parse_entries(feed):
    rows = []
    for e in getattr(feed, "entries", []):
        title = html.unescape(getattr(e, "title", "") or "")
        desc  = html.unescape(getattr(e, "summary", "") or "")
        link  = getattr(e, "link", "") or ""
        published = getattr(e, "published", "") or getattr(e, "updated", "") or ""
        pub_dt = pd.to_datetime(published, utc=True, errors="coerce")
        rows.append((title, desc, link, pub_dt))
    return rows

def classify_importance(title):
    t = title.lower()
    if any(k in t for k in ["cpi", "fomc", "nonfarm", "payroll", "pce"]):
        return "High"
    if any(k in t for k in ["gdp", "ppi", "unemployment"]):
        return "Medium"
    return "Low"

def main():
    items = []
    for kw in KEYWORDS:
        query = f"{kw} United States calendar"
        for src, url_fn in SOURCES:
            url = url_fn(query)
            try:
                feed = feedparser.parse(url)
                for title, desc, link, dt in parse_entries(feed):
                    if not title:
                        continue
                    date_val = (dt.date() if pd.notna(dt)
                                else datetime.now(timezone.utc).date())
                    items.append({
                        "date": date_val,
                        "event": title,
                        "source": src,
                        "country": "US",
                        "importance": classify_importance(title),
                        "link": link,
                        "notes": (desc or "")[:240]
                    })
            except Exception as e:
                print(f"⚠️ {src} parse error for '{kw}': {e}")
            time.sleep(PER_QUERY_SLEEP)

    if not items:
        print("🚫 No economic RSS items found.")
        return

    df = pd.DataFrame(items)

    # filter importance + lookback
    cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=LOOKBACK_DAYS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[df["date"] >= cutoff]
    df = df[df["importance"].isin(MIN_IMPORTANCE)]

    # dedup & sort
    df = (df.drop_duplicates(subset=["event", "date", "link"])
            .sort_values(["date", "importance"], ascending=[False, False]))

    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved {len(df)} events → {OUT_PATH}")
    print(df.head(8))

if __name__ == "__main__":
    main()
