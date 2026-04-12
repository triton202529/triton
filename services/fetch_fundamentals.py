# services/fetch_fundamentals.py
"""
Fetch fundamentals from yfinance and save to data/results/fundamentals.csv

Clean upgrades:
- Discovers tickers from per-ticker parquet files (data/processed/*.parquet preferred)
  with fallback to data/results/*.parquet
- Uses robust info fetching (get_info -> info fallback)
- Never fails the run because of one ticker
- Always writes fundamentals.csv (even if empty/partial)
- Safe numeric coercion + defaults
"""

import os
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yfinance as yf
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_PATH = RESULTS_DIR / "fundamentals.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def safe_int(x: Any, default: int) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, bool):
            return int(default)
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return int(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return int(default)
        return int(float(s))
    except Exception:
        return int(default)


def safe_get_info(ticker: str) -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    try:
        info = t.get_info()
        if isinstance(info, dict) and info:
            return info
    except Exception:
        pass
    try:
        info = t.info
        if isinstance(info, dict) and info:
            return info
    except Exception:
        pass
    return {}


def discover_tickers() -> List[str]:
    """
    Preferred: data/processed/{TICKER}.parquet
    Fallback:  data/results/{TICKER}.parquet
    """
    tickers: List[str] = []

    if PROCESSED_DIR.exists():
        for p in PROCESSED_DIR.glob("*.parquet"):
            t = p.stem.upper().strip()
            if t not in ("STOCK_DATA", "STOCK_DATA_MERGED"):
                tickers.append(t)

    if not tickers and RESULTS_DIR.exists():
        for p in RESULTS_DIR.glob("*.parquet"):
            t = p.stem.upper().strip()
            tickers.append(t)

    # dedupe + sort
    tickers = sorted(list(dict.fromkeys([t for t in tickers if t])))
    return tickers


# -----------------------------
# Main
# -----------------------------
def main():
    tickers = discover_tickers()
    print(f"📡 Fetching fundamentals for {len(tickers)} tickers...")

    fundamentals: List[Dict[str, Any]] = []
    ok_count = 0

    for ticker in tickers:
        print(f"🔍 {ticker}")
        try:
            info = safe_get_info(ticker)

            # Defaults (kept intentionally conservative)
            pe_ratio = safe_float(info.get("trailingPE"), 15.0)
            eps = safe_float(info.get("trailingEps"), 5.0)
            revenue = safe_float(info.get("totalRevenue"), 1e9)
            market_cap = safe_float(info.get("marketCap"), 1e10)
            pb_ratio = safe_float(info.get("priceToBook"), 1.5)
            dividend_yield = safe_float(info.get("dividendYield"), 0.0)

            fundamentals.append(
                {
                    "ticker": ticker.upper(),
                    "pe_ratio": pe_ratio,
                    "eps": eps,
                    "revenue": revenue,
                    "market_cap": market_cap,
                    "pb_ratio": pb_ratio,
                    "dividend_yield": dividend_yield,
                    "ok": bool(info),
                    "fetched_at_utc": utc_now_iso(),
                    "source": "yfinance",
                }
            )

            if info:
                ok_count += 1

        except Exception as e:
            print(f"⚠️ Error fetching {ticker}, using defaults: {e}")
            fundamentals.append(
                {
                    "ticker": ticker.upper(),
                    "pe_ratio": 15.0,
                    "eps": 5.0,
                    "revenue": 1e9,
                    "market_cap": 1e10,
                    "pb_ratio": 1.5,
                    "dividend_yield": 0.0,
                    "ok": False,
                    "fetched_at_utc": utc_now_iso(),
                    "source": "yfinance",
                    "error": str(e),
                }
            )

        # tiny delay helps avoid rate limiting
        time.sleep(0.15)

    df = pd.DataFrame(fundamentals)

    # Always write output (even if empty)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"✅ Fundamentals saved to {OUTPUT_PATH} (ok={ok_count}/{len(tickers)})")


if __name__ == "__main__":
    main()
