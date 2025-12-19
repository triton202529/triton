"""
record_equity.py
Capture the live account equity from Alpaca and append it to
data/results/portfolio_history.csv so Triton can measure drawdowns.

We keep it dead simple and fail-soft:
- If we can't talk to Alpaca, we just return None.
- If the CSV doesn't exist yet, we create it with a header.
- Each row is: timestamp,equity
"""

import os
import csv
import datetime as dt
from pathlib import Path
import requests

PORTFOLIO_HISTORY_PATH = Path("data/results/portfolio_history.csv")


def _alpaca_headers():
    key    = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        return None
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def _alpaca_base():
    base = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets").rstrip("/")
    return base


def fetch_live_equity():
    """
    Ask Alpaca /v2/account for current equity.

    We try fields in this order:
    - 'equity'
    - 'portfolio_value'
    - 'cash' (fallback, worst case)
    Returns float or None.
    """
    H = _alpaca_headers()
    if H is None:
        return None

    B = _alpaca_base()
    try:
        resp = requests.get(f"{B}/v2/account", headers=H, timeout=10)
        data = resp.json()
    except Exception:
        return None

    # Try common fields
    for key in ("equity", "portfolio_value", "cash"):
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except:
                continue

    return None


def append_equity_snapshot():
    """
    Get current equity and append a row to portfolio_history.csv.

    Returns the equity value written (float) or None if we couldn't record.
    """
    eq = fetch_live_equity()
    if eq is None:
        return None

    ts = dt.datetime.now(dt.timezone.utc).isoformat()

    # Ensure parent folder exists
    PORTFOLIO_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = PORTFOLIO_HISTORY_PATH.exists()

    with PORTFOLIO_HISTORY_PATH.open("a", newline="") as f:
        fieldnames = ["timestamp", "equity"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({
            "timestamp": ts,
            "equity": f"{eq:.4f}",
        })

    return eq
