# services/market_sentinels.py
# Robust market sentinel fetch + quick health metrics

import os
import sys
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Optional provider (yfinance)
try:
    import yfinance as yf

    YF_OK = True
except Exception:
    yf = None
    YF_OK = False

RESULTS_DIR = Path("data") / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default universe of quick "market health" sentinels
# (You can tweak freely; non-existent symbols will be skipped gracefully)
DEFAULT_SENTINELS = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "^VIX",  # VIX (CBOE)
    "DIA",   # Dow Jones
]


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _extract_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a 1-D close price Series from various possible DataFrame shapes:
    - Simple column 'Close' / 'close'
    - MultiIndex columns (e.g., ('Close','AAPL') or ('AAPL','Close'))
    - Wide DataFrame with several 'Close' columns (pick first non-all-NaN)
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # MultiIndex columns: try common layouts, else flatten
    if isinstance(df.columns, pd.MultiIndex):
        for level in (0, 1):
            try:
                if "Close" in df.columns.get_level_values(level):
                    sub = df.xs("Close", axis=1, level=level, drop_level=False)
                    if isinstance(sub, pd.DataFrame) and not sub.empty:
                        # If one column left, squeeze
                        if sub.shape[1] == 1:
                            return pd.to_numeric(sub.iloc[:, 0], errors="coerce")
                        # Else pick first non-empty column
                        non_empty = [c for c in sub.columns if sub[c].notna().any()]
                        col = non_empty[0] if non_empty else sub.columns[0]
                        return pd.to_numeric(sub[col], errors="coerce")
            except Exception:
                pass

        # Fallback: flatten the MultiIndex to strings like "AAPL_Close"
        df = df.copy()
        df.columns = [
            "_".join([str(x) for x in tup if str(x) != ""]) for tup in df.columns.to_list()
        ]

    # Case-insensitive exact "close"
    lower_map = {str(c).lower(): c for c in df.columns}
    if "close" in lower_map:
        c = df[lower_map["close"]]
        if isinstance(c, pd.DataFrame):
            if c.shape[1] == 1:
                return pd.to_numeric(c.iloc[:, 0], errors="coerce")
            non_empty = [name for name in c.columns if c[name].notna().any()]
            sel = non_empty[0] if non_empty else c.columns[0]
            return pd.to_numeric(c[sel], errors="coerce")
        return pd.to_numeric(c, errors="coerce")

    # Last resort: any column whose name endswith 'close' (e.g., 'Adj Close', 'something_close')
    for col in df.columns:
        if str(col).lower().endswith("close"):
            c = df[col]
            if isinstance(c, pd.DataFrame):
                if c.shape[1] == 1:
                    return pd.to_numeric(c.iloc[:, 0], errors="coerce")
                non_empty = [name for name in c.columns if c[name].notna().any()]
                sel = non_empty[0] if non_empty else c.columns[0]
                return pd.to_numeric(c[sel], errors="coerce")
            return pd.to_numeric(c, errors="coerce")

    # Nothing found
    return pd.Series(dtype=float)


def _normalize_date_index_to_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index()
    if "date" not in out.columns:
        # yfinance uses 'Date' index name sometimes; normalize to 'date'
        if "Date" in out.columns:
            out = out.rename(columns={"Date": "date"})
        elif "index" in out.columns:
            out = out.rename(columns={"index": "date"})
        else:
            # create a date anyway to keep shape consistent
            out["date"] = pd.NaT
    # Make the date tz-naive UTC (midnight)
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return out


def fetch_hist(symbol: str, days: int = 60) -> pd.DataFrame:
    """
    Fetch daily history and return DataFrame with columns ['date', 'close'].
    Tries to be robust to different shapes/column names.
    """
    if not YF_OK:
        raise RuntimeError(
            "yfinance is not installed. Install with: pip install yfinance"
        )

    # yfinance period accepts strings like "60d"
    period = f"{max(days + 5, days)}d"
    try:
        raw = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    except Exception as e:
        print(f"⚠️ yfinance download failed for {symbol}: {e}")
        return pd.DataFrame(columns=["date", "close"])

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "close"])

    close = _extract_close_series(raw)
    if close.empty:
        return pd.DataFrame(columns=["date", "close"])

    out = pd.DataFrame({"close": close})
    out = _normalize_date_index_to_column(out)
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    # Keep most recent N valid rows
    if days and days > 0:
        out = out.tail(days)
    return out[["date", "close"]]


def realized_vol(returns: pd.Series, freq: int = 252) -> float:
    r = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return np.nan
    return float(r.std() * math.sqrt(freq))


def rsi(series: pd.Series, period: int = 14) -> float:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.size < period + 1:
        return np.nan
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(period, min_periods=period).mean()
    roll_down = down.rolling(period, min_periods=period).mean()
    rs = roll_up / (roll_down + 1e-12)
    last = rs.iloc[-1] if not rs.dropna().empty else np.nan
    if not np.isfinite(last):
        return np.nan
    return float(100.0 - (100.0 / (1.0 + last)))


def summarize_symbol(symbol: str, days: int = 60) -> Dict[str, Optional[float]]:
    df = fetch_hist(symbol, days=days)
    if df.empty:
        return {
            "symbol": symbol,
            "last_date": None,
            "last_close": None,
            "ret_5d": np.nan,
            "ret_20d": np.nan,
            "vol_20d": np.nan,
            "rsi_14": np.nan,
            "ma_20_above_ma_50": None,
        }

    df = df.sort_values("date").dropna(subset=["close"])
    closes = pd.to_numeric(df["close"], errors="coerce")
    rets = closes.pct_change()

    last_close = float(closes.iloc[-1])
    last_date = str(df["date"].iloc[-1].date())

    # Rolling stats
    ma20 = closes.rolling(20, min_periods=1).mean()
    ma50 = closes.rolling(50, min_periods=1).mean()

    ret_5d = float((closes.iloc[-1] / closes.iloc[-6]) - 1.0) if len(closes) > 6 else np.nan
    ret_20d = float((closes.iloc[-1] / closes.iloc[-21]) - 1.0) if len(closes) > 21 else np.nan
    vol_20d = realized_vol(rets.tail(21))
    rsi_14 = rsi(closes, period=14)

    cross = None
    if len(ma20) and len(ma50):
        cross = bool(ma20.iloc[-1] > ma50.iloc[-1])

    return {
        "symbol": symbol,
        "last_date": last_date,
        "last_close": last_close,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "vol_20d": vol_20d,
        "rsi_14": rsi_14,
        "ma_20_above_ma_50": cross,
    }


def main(symbols: Optional[List[str]] = None, lookback_days: int = 60):
    if symbols is None:
        symbols = DEFAULT_SENTINELS

    if not YF_OK:
        print("❌ yfinance not installed. Install it with: pip install yfinance")
        sys.exit(2)

    rows = []
    for s in symbols:
        try:
            print(f"⏳ Fetching {s}…")
            rows.append(summarize_symbol(s, days=lookback_days))
            time.sleep(0.2)  # be gentle
        except Exception as e:
            print(f"⚠️ Failed {s}: {e}")

    if not rows:
        print("🚫 No sentinel data produced.")
        return

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "market_sentinels.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Wrote {csv_path} ({len(df)} rows)")

    # Also produce a tiny JSON with a few global flags you might consume elsewhere
    # e.g., risk dashboard can read these and show a banner
    global_flags = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signals": rows,
        "notes": "Basic market health sentinels. Non-blocking best-effort output.",
    }
    json_path = RESULTS_DIR / "market_sentinels.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(global_flags, f, indent=2)
    print(f"💾 Wrote {json_path}")


if __name__ == "__main__":
    # Minimal CLI: allow symbols via env var SENTINELS, comma-separated
    # or just edit DEFAULT_SENTINELS above.
    env_syms = os.environ.get("SENTINELS", "").strip()
    symbols = (
        [s.strip() for s in env_syms.split(",") if s.strip()] if env_syms else DEFAULT_SENTINELS
    )
    # Allow LOOKBACK_DAYS override
    try:
        lookback = int(os.environ.get("SENTINEL_LOOKBACK_DAYS", "60"))
    except Exception:
        lookback = 60

    main(symbols=symbols, lookback_days=lookback)
