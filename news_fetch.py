# news_fetch.py
import os, time, argparse, re
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR  = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Skip pure indices
STOP_TICKERS = {"^DJI", "^GSPC", "^IXIC", "^VIX"}

# Helpful aliases (companies & ETFs)
ALIASES = {
    # Tricky symbols / hyphenated
    "BRK-B": ["Berkshire Hathaway", "Berkshire Hathaway Class B", "BRK.B"],
    "GOOGL": ["Alphabet Inc", "Google"],
    "META":  ["Meta Platforms", "Facebook"],
    "V":     ["Visa Inc"],
    "MA":    ["Mastercard"],
    "T":     ["AT&T"],
    "GE":    ["General Electric", "GE Vernova"],

    # ETFs you track
    "SPY": ["SPDR S&P 500 ETF Trust"],
    "QQQ": ["Invesco QQQ Trust"],
    "ARKK":["ARK Innovation ETF"],
    "BITO":["ProShares Bitcoin Strategy ETF"],
    "GBTC":["Grayscale Bitcoin Trust"],
    "DIA": ["SPDR Dow Jones Industrial Average ETF"],
    "SLV": ["iShares Silver Trust"],
    "GLD": ["SPDR Gold Shares"],
    "DBA": ["Invesco DB Agriculture Fund"],
    "UNG": ["United States Natural Gas Fund"],
    "USO": ["United States Oil Fund"],

    # Sector SPDRs
    "XLB": ["Materials Select Sector SPDR Fund"],
    "XLE": ["Energy Select Sector SPDR Fund"],
    "XLF": ["Financial Select Sector SPDR Fund"],
    "XLI": ["Industrial Select Sector SPDR Fund"],
    "XLK": ["Technology Select Sector SPDR Fund"],
    "XLP": ["Consumer Staples Select Sector SPDR Fund"],
    "XLRE":["Real Estate Select Sector SPDR Fund"],
    "XLU": ["Utilities Select Sector SPDR Fund"],
    "XLV": ["Health Care Select Sector SPDR Fund"],
    "XLY": ["Consumer Discretionary Select Sector SPDR Fund"],
}

POS = set("beat beats surges record upgrade bullish strong growth jumps rally profit profits upside gain gains".split())
NEG = set("miss misses plunges downgrade bearish weak slump falls lawsuit probe cut cuts downside loss losses".split())

def naive_sentiment(text: str) -> float:
    if not text:
        return 0.0
    words = re.sub(r"[^a-zA-Z ]", " ", text).lower().split()
    if not words:
        return 0.0
    pos = sum(w in POS for w in words)
    neg = sum(w in NEG for w in words)
    total = pos + neg
    if total == 0:
        return 0.0
    return float((pos - neg) / total)  # ~[-1,1]

def best_query_for(ticker: str) -> str:
    """Build a broad but precise NewsAPI query for this ticker."""
    terms = {ticker}
    # Hyphen variants (e.g., BRK-B)
    if "-" in ticker:
        terms.add(ticker.replace("-", "."))
        terms.add(ticker.replace("-", " "))
    # Known aliases
    for alt in ALIASES.get(ticker, []):
        terms.add(alt)
    # Try yfinance longName/shortName for better recall
    if YF_OK and ticker not in STOP_TICKERS:
        try:
            tk = yf.Ticker(ticker)
            info = None
            if hasattr(tk, "get_info"):
                info = tk.get_info()
            else:
                info = getattr(tk, "info", None)
            if isinstance(info, dict):
                for k in ("longName", "shortName"):
                    v = (info.get(k) or "").strip()
                    if v:
                        terms.add(v)
        except Exception:
            pass
    # Quote multi-word terms
    quoted = []
    for t in sorted(terms):
        tt = t.strip()
        if not tt:
            continue
        if " " in tt or "." in tt:
            quoted.append(f"\"{tt}\"")
        else:
            quoted.append(tt)
    # Favor name OR symbol; avoid being too broad with common words
    return " OR ".join(quoted)

def fetch_for_ticker(t: str, key: str, since_days: int, per: int, sleep_s: float):
    if t in STOP_TICKERS or t.startswith("^"):
        return []
    q = best_query_for(t)
    since = (datetime.now(timezone.utc) - timedelta(days=since_days)).date().isoformat()
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "language": "en",
        "searchIn": "title,description,content",
        "sortBy": "publishedAt",
        "pageSize": per,
        "from": since,
    }
    try:
        resp = requests.get(url, headers={"X-Api-Key": key}, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    rows = []
    for a in (data or {}).get("articles", []):
        title = a.get("title") or ""
        desc  = a.get("description") or ""
        s     = naive_sentiment(f"{title}. {desc}")
        rows.append({
            "date": a.get("publishedAt"),
            "ticker": t,
            "title": title,
            "url": a.get("url"),
            "sentiment": s,
            "source": (a.get("source") or {}).get("name"),
            "author": a.get("author"),
            "description": desc,
            "query": q,
        })
    time.sleep(sleep_s)
    return rows

def main(tickers, days, per, fill_missing, sleep_s):
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        raise SystemExit("NEWSAPI_KEY not set")

    all_rows = []
    for i, t in enumerate(tickers, 1):
        all_rows.extend(fetch_for_ticker(t, key, days, per, sleep_s))

    df = pd.DataFrame(all_rows)

    # Optional: ensure at least one placeholder row per non-index ticker
    base_set = {t for t in tickers if not (t in STOP_TICKERS or t.startswith("^"))}
    have_set = set(df["ticker"].unique()) if not df.empty and "ticker" in df.columns else set()

    if fill_missing:
        missing = sorted(base_set - have_set)
        if missing:
            today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            filler = pd.DataFrame({
                "date": [today.isoformat()] * len(missing),
                "ticker": missing,
                "title": ["" for _ in missing],
                "url": ["" for _ in missing],
                "sentiment": [0.0 for _ in missing],
                "source": [""] * len(missing),
                "author": [""] * len(missing),
                "description": ["No recent articles found"] * len(missing),
                "query": [best_query_for(t) for t in missing],
            })
            df = pd.concat([df, filler], ignore_index=True)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["ticker"]).sort_values(["ticker", "date"], ascending=[True, False])
        df = df.drop_duplicates(subset=["ticker", "url"], keep="first")

    out = RESULTS_DIR / "news_sentiment.csv"
    df.to_csv(out, index=False)
    uniq = df["ticker"].nunique() if "ticker" in df.columns else 0
    print(f"Wrote {len(df):,} rows across {uniq} tickers -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tickers", nargs="+", required=True)
    ap.add_argument("--days", type=int, default=7, help="Look back N days for news (default 7)")
    ap.add_argument("--per",  type=int, default=8, help="Max articles per ticker (default 8)")
    ap.add_argument("--fill-missing", action="store_true", help="Add neutral placeholder rows for tickers with no hits")
    ap.add_argument("--sleep", type=float, default=0.35, help="Delay between requests (seconds)")
    args = ap.parse_args()
    main(args.tickers, args.days, args.per, args.fill_missing, args.sleep)
