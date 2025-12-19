#!/usr/bin/env python3
# services/fetch_news_sentiment.py

import os
import re
import time
import argparse
import urllib.parse as up
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import sys  # ← added

# ---- UTF-8 / Windows console safety (avoid UnicodeEncodeError) ----
for _s in (sys.stdout, sys.stderr):
    try:
        # Keep current encoding, but never crash on characters console can't encode
        _s.reconfigure(errors="replace")
    except Exception:
        pass
# -------------------------------------------------------------------

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ──────────────────────────────────────────────────────────────────────────────
# Config / constants
# ──────────────────────────────────────────────────────────────────────────────
# Prefer env var; fall back to literal if present
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "").strip() or "36536909146a411683bc7ccecab398b7"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_CSV = os.path.join(RESULTS_DIR, "news_sentiment.csv")

MAX_LOOKBACK_DAYS = 30
DEFAULT_PAGE_SIZE = 20
DEFAULT_MAX_PAGES = 1
REQUEST_TIMEOUT = 20
RATE_LIMIT_SLEEP = 15
HTTP_RETRY = 2
MAX_429_RETRIES = 3
PER_BATCH_SLEEP = 2
PER_TICKER_SLEEP = 1

# Canonical “trusted” outlets mapping (domain → pretty label)
TRUSTED_CANON = {
    "bloomberg.com": "Bloomberg",
    "reuters.com": "Reuters",
    "wsj.com": "WSJ",
    "cnbc.com": "CNBC",
    "ft.com": "Financial Times",
    "marketwatch.com": "MarketWatch",
    "barrons.com": "Barron's",
    "forbes.com": "Forbes",
    "businessinsider.com": "Business Insider",
    "finance.yahoo.com": "Yahoo Finance",
    "yahoo.com": "Yahoo Finance",
    "thestreet.com": "TheStreet",
    "seekingalpha.com": "Seeking Alpha",
    "investing.com": "Investing.com",
}
DEFAULT_PREFERRED_DOMAINS = list(TRUSTED_CANON.keys())

# Ticker name synonyms for matching
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def now_naive_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def clamp_window(days_requested: int) -> int:
    if days_requested > MAX_LOOKBACK_DAYS:
        print(f"Info: window {days_requested}d > max {MAX_LOOKBACK_DAYS}d. Clamping.")
    return min(days_requested, MAX_LOOKBACK_DAYS)


def analyze_sentiment(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return float(analyzer.polarity_scores(text)["compound"])


def _request(url: str, params: dict, label: str) -> Optional[requests.Response]:
    tries = 0
    while True:
        tries += 1
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            if tries <= HTTP_RETRY:
                time.sleep(2)
                continue
            print(f"Warn {label}: request error: {e}")
            return None

        if r.status_code == 200:
            return r

        if r.status_code == 429:
            if tries >= MAX_429_RETRIES:
                print(f"Stop: too many 429s for {label}; skipping.")
                return None
            print(
                f"Rate limited on {label}. Sleeping {RATE_LIMIT_SLEEP}s… ({tries}/{MAX_429_RETRIES})"
            )
            time.sleep(RATE_LIMIT_SLEEP)
            continue

        print(f"Warn {label}: {r.status_code} {r.text[:200]}...")
        return None


def fetch_everything(
    q: str,
    from_str: str,
    to_str: str,
    page_size: int,
    page: int,
    domains: Optional[List[str]] = None,
):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "from": from_str,
        "to": to_str,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "page": page,
        "apiKey": NEWSAPI_KEY,
    }
    if domains:
        params["domains"] = ",".join(sorted(set(d.lower() for d in domains)))
    return _request(url, params, f"everything p{page}")


def fetch_top_headlines(
    page_size: int = 100,
    page: int = 1,
    country: str = "us",
    category: str = "business",
    domains: Optional[List[str]] = None,
):
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": country,
        "category": category,
        "pageSize": page_size,
        "page": page,
        "apiKey": NEWSAPI_KEY,
    }
    if domains:
        params["domains"] = ",".join(sorted(set(d.lower() for d in domains)))
    return _request(url, params, f"top-headlines p{page}")


def tag_tickers(article: dict, tickers: List[str]) -> List[str]:
    text = " ".join(
        str(article.get(k, "") or "") for k in ("title", "description", "content")
    ).lower()
    matched = set()
    for t in tickers:
        for s in NAME_SYNONYMS.get(t, [t]):
            if s and s.lower() in text:
                matched.add(t)
                break
    return sorted(matched)


def get_domain(u: str) -> str:
    try:
        n = up.urlparse(str(u)).netloc.lower()
        parts = [q for q in n.split(".") if q not in ("", "www", "m")]
        return ".".join(parts[-2:]) if len(parts) >= 2 else n
    except Exception:
        return ""


def canonical_source(url: str, fallback: str = "") -> str:
    d = get_domain(url)
    if d in TRUSTED_CANON:
        return TRUSTED_CANON[d]
    return fallback or d or "Unknown"


def _normalize_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return DataFrame with dashboard-friendly columns."""
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "sentiment",
                "source_display",
                "news",
                "description",
                "url",
                "source",
                "author",
                "domain",
            ]
        )

    df = pd.DataFrame(rows)

    # date → naive UTC midnight
    df["date"] = (
        pd.to_datetime(df.get("date"), errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # Ensure columns exist & have type
    for c in ["ticker", "news", "description", "url", "source", "author"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str)

    # Domain & pretty source name
    if "domain" not in df.columns:
        df["domain"] = df["url"].map(get_domain)

    if "source_display" not in df.columns:
        df["source_display"] = df.apply(
            lambda r: TRUSTED_CANON.get(r.get("domain", ""), r.get("source", "")), axis=1
        )

    # Dedupe by URL (keep latest date)
    df = df.sort_values(["url", "date"]).drop_duplicates(subset=["url"], keep="last")

    # Keep only relevant columns, rename title→news if present
    if "title" in df.columns and "news" not in df.columns:
        df = df.rename(columns={"title": "news"})

    keep = [
        "date",
        "ticker",
        "sentiment",
        "source_display",
        "news",
        "description",
        "url",
        "source",
        "author",
        "domain",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = "" if c not in ("sentiment",) else 0.0

    return df[keep].reset_index(drop=True)


def _merge_into_csv(new_df: pd.DataFrame, out_csv: str):
    if os.path.exists(out_csv) and os.stat(out_csv).st_size > 0:
        old = pd.read_csv(out_csv)
        if "date" in old.columns:
            old["date"] = (
                pd.to_datetime(old["date"], errors="coerce", utc=True)
                .dt.tz_localize(None)
                .dt.normalize()
            )
        merged = pd.concat([old, new_df], ignore_index=True)
        merged = merged.sort_values(["url", "date"]).drop_duplicates(subset=["url"], keep="last")
    else:
        merged = new_df.copy()

    merged = merged.sort_values("date", ascending=True)
    merged.to_csv(out_csv, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Fetch modes
# ──────────────────────────────────────────────────────────────────────────────
def aggregate_mode(
    tickers: List[str],
    window: int,
    output_path: str,
    page_size: int,
    max_pages: int,
    batch_size: int,
    domains: Optional[List[str]],
    trusted_only: bool,
):
    """Batch query /everything; fallback to /top-headlines if nothing fetched."""
    end_dt = datetime.now(timezone.utc)
    window = clamp_window(window)
    start_dt = end_dt - timedelta(days=window)
    from_str, to_str = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    all_rows: List[Dict[str, Any]] = []
    any_success = False

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        q_parts = []
        for t in batch:
            for n in NAME_SYNONYMS.get(t, [t]):
                q_parts.append(f'"{n}"')
        q = f"({' OR '.join(q_parts)})"

        # ↓ replaced emoji with ASCII
        print(f"[News] Batch {i // batch_size + 1}: fetching for {', '.join(batch)}")
        batch_articles: List[Dict[str, Any]] = []

        for page in range(1, max_pages + 1):
            r = fetch_everything(q, from_str, to_str, page_size, page, domains)
            if r is None:
                break
            js = r.json()
            items = js.get("articles", []) or []
            if not items:
                break
            for a in items:
                # Filter by domain if trusted_only (or domains passed)
                url = a.get("url") or ""
                dom = get_domain(url)
                if trusted_only and dom not in TRUSTED_CANON:
                    continue

                title = a.get("title", "") or ""
                desc = a.get("description", "") or ""
                author = a.get("author", "") or ""
                source_name = (a.get("source", {}) or {}).get("name", "")
                pub_at = a.get("publishedAt", "") or ""
                pub_dt = pd.to_datetime(pub_at, utc=True, errors="coerce")
                row_date = pub_dt.tz_convert(None) if hasattr(pub_dt, "tz_convert") else pub_dt
                if pd.isna(row_date):
                    row_date = start_dt
                row_date = row_date.normalize()

                hits = tag_tickers(a, batch)
                if not hits:
                    continue

                sent = analyze_sentiment(f"{title} {desc}")

                for t in hits:
                    batch_articles.append(
                        {
                            "date": row_date,
                            "ticker": t,
                            "title": title,
                            "description": desc,
                            "url": url,
                            "source": source_name,
                            "author": author,
                            "sentiment": sent,
                            "domain": dom,
                            "source_display": canonical_source(url, source_name),
                        }
                    )

            if len(items) < page_size:
                break

        if batch_articles:
            any_success = True
            all_rows.extend(batch_articles)

        if all_rows:
            df = _normalize_rows(all_rows)
            _merge_into_csv(df, output_path)
            print(f"Saved → {output_path} (rows: {len(df)})")

        time.sleep(PER_BATCH_SLEEP)

    # Fallback to top-headlines if absolutely nothing
    if not any_success and not all_rows:
        print("Fallback to /top-headlines…")
        r = fetch_top_headlines(
            page_size=100, page=1, country="us", category="business", domains=domains
        )
        if r is not None:
            items = r.json().get("articles", []) or []
            for a in items:
                url = a.get("url") or ""
                dom = get_domain(url)
                if trusted_only and dom not in TRUSTED_CANON:
                    continue
                title = a.get("title", "") or ""
                desc = a.get("description", "") or ""
                author = a.get("author", "") or ""
                source_name = (a.get("source", {}) or {}).get("name", "")
                pub_at = a.get("publishedAt", "") or ""
                pub_dt = pd.to_datetime(pub_at, utc=True, errors="coerce")
                row_date = pub_dt.tz_convert(None) if hasattr(pub_dt, "tz_convert") else pub_dt
                if pd.isna(row_date):
                    row_date = end_dt
                row_date = row_date.normalize()

                hits = tag_tickers(a, tickers)
                if not hits:
                    continue

                sent = analyze_sentiment(f"{title} {desc}")
                for t in hits:
                    all_rows.append(
                        {
                            "date": row_date,
                            "ticker": t,
                            "title": title,
                            "description": desc,
                            "url": url,
                            "source": source_name,
                            "author": author,
                            "sentiment": sent,
                            "domain": dom,
                            "source_display": canonical_source(url, source_name),
                        }
                    )

    if not all_rows:
        print("No news articles found.")
        return

    final_df = _normalize_rows(all_rows)
    _merge_into_csv(final_df, output_path)
    print(f"Saved news sentiment to: {output_path} (rows now: {len(final_df)})")
    print(final_df.head(8))


def by_ticker_mode(
    tickers: List[str],
    window: int,
    output_path: str,
    page_size: int,
    max_pages: int,
    domains: Optional[List[str]],
    trusted_only: bool,
):
    end_dt = datetime.now(timezone.utc)
    window = clamp_window(window)
    start_dt = end_dt - timedelta(days=window)
    from_str, to_str = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    all_rows: List[Dict[str, Any]] = []

    for t in tickers:
        # ↓ replaced emoji with ASCII
        print(f"[News] Fetching news for {t} ({from_str} → {to_str})...")
        for page in range(1, max_pages + 1):
            r = fetch_everything(f'"{t}"', from_str, to_str, page_size, page, domains)
            if r is None:
                break
            items = r.json().get("articles", []) or []
            if not items:
                break
            for a in items:
                url = a.get("url") or ""
                dom = get_domain(url)
                if trusted_only and dom not in TRUSTED_CANON:
                    continue

                title = a.get("title", "") or ""
                desc = a.get("description", "") or ""
                author = a.get("author", "") or ""
                source_name = (a.get("source", {}) or {}).get("name", "")
                pub_at = a.get("publishedAt", "") or ""
                sent = analyze_sentiment(f"{title} {desc}")
                pub_dt = pd.to_datetime(pub_at, utc=True, errors="coerce")
                row_date = pub_dt.tz_convert(None) if hasattr(pub_dt, "tz_convert") else pub_dt
                if pd.isna(row_date):
                    row_date = start_dt
                row_date = row_date.normalize()

                all_rows.append(
                    {
                        "date": row_date,
                        "ticker": t,
                        "title": title,
                        "description": desc,
                        "url": url,
                        "source": source_name,
                        "author": author,
                        "sentiment": sent,
                        "domain": dom,
                        "source_display": canonical_source(url, source_name),
                    }
                )

            if len(items) < page_size:
                break

        if all_rows:
            df = _normalize_rows(all_rows)
            _merge_into_csv(df, output_path)
            print(f"Saved → {output_path} (rows: {len(df)})")

        time.sleep(PER_TICKER_SLEEP)

    if not all_rows:
        print("No news articles found.")
        return

    final_df = _normalize_rows(all_rows)
    _merge_into_csv(final_df, output_path)
    print(f"Saved news sentiment to: {output_path} (rows now: {len(final_df)})")
    print(final_df.head(8))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch market news → data/results/news_sentiment.csv"
    )
    parser.add_argument("--tickers", default="all", help="Comma-separated tickers or 'all'")
    parser.add_argument("--window", type=int, default=7, help="Days back (auto-clamped to 30)")
    parser.add_argument("--out", default=OUT_CSV)
    parser.add_argument("--page_size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max_pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--strategy", choices=["aggregate", "by_ticker"], default="aggregate")
    parser.add_argument("--batch", type=int, default=10, help="Tickers per batch in aggregate mode")
    parser.add_argument(
        "--domains", default="", help="Comma-separated preferred domains (defaults to majors)"
    )
    parser.add_argument(
        "--trusted-only", action="store_true", help="Keep only canonical trusted outlets"
    )
    args = parser.parse_args()

    # Build ticker list
    if args.tickers == "all":
        # Prefer scores file, else a small default basket
        scores = os.path.join(RESULTS_DIR, "stock_scores.csv")
        if os.path.exists(scores) and os.stat(scores).st_size > 0:
            df_tickers = pd.read_csv(scores)
            tickers = sorted(
                df_tickers.get("ticker", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        else:
            tickers = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    # Domains preference
    domains = [
        d.strip().lower() for d in args.domains.split(",") if d.strip()
    ] or DEFAULT_PREFERRED_DOMAINS
    trusted_only = bool(args.trusted_only)

    if not NEWSAPI_KEY:
        print("Warn: NEWSAPI_KEY not set; NewsAPI requests may fail.")

    if args.strategy == "aggregate":
        aggregate_mode(
            tickers,
            args.window,
            args.out,
            args.page_size,
            args.max_pages,
            args.batch,
            domains,
            trusted_only,
        )
    else:
        by_ticker_mode(
            tickers, args.window, args.out, args.page_size, args.max_pages, domains, trusted_only
        )
