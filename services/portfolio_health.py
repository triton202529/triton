"""
portfolio_health.py
Compute live portfolio health stats (drawdown, etc.) for Triton.

This feeds risk governance (pretrade_guard).

We support TWO schemas in data/results/portfolio_history.csv:

1. Legacy / backtest / paper-tracking format:
   date,cash,market_value
   2025-10-20 16:00:00,10000.0,90500.0
   -> equity = cash + market_value

2. Live runtime snapshots from record_equity.append_equity_snapshot():
   timestamp,equity
   2025-10-31T00:18:25.834915+00:00,97221.0300

We:
- read both styles
- normalize them into (timestamp, equity_float)
- sort chronologically
- compute drawdown from peak to latest

We deliberately fail soft. If anything goes wrong, we return safe defaults.
"""

import csv
from pathlib import Path
from dataclasses import dataclass
import datetime as dt
from typing import List, Tuple, Optional

PORTFOLIO_HISTORY_PATH = Path("data/results/portfolio_history.csv")

# Only need recent history to judge pain/regime.
MAX_ROWS_FOR_ANALYSIS = 5000


@dataclass
class PortfolioHealth:
    drawdown_pct: float    # e.g. 0.12 means "down 12% from peak"
    latest_equity: float   # most recent equity value we saw
    peak_equity: float     # max equity in window
    data_points: int       # how many usable rows we had


def _safe_float(val, default=None):
    try:
        return float(val)
    except:
        return default


def _parse_timestamp(raw: str) -> Optional[dt.datetime]:
    """
    Try to interpret timestamps from both formats and normalize to UTC-aware.
    Supported examples:
    - "2025-10-20 16:00:00"
    - "2025-10-31T00:18:25.834915+00:00"
    """
    if not raw:
        return None

    # Try Python's ISO parser first (handles T + offset)
    try:
        ts = dt.datetime.fromisoformat(raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        return ts.astimezone(dt.timezone.utc)
    except Exception:
        pass

    # Try "YYYY-MM-DD HH:MM:SS"
    try:
        ts = dt.datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        ts = ts.replace(tzinfo=dt.timezone.utc)
        return ts
    except Exception:
        pass

    return None


def _load_equity_series() -> List[Tuple[dt.datetime, float]]:
    """
    Load portfolio_history.csv and return a list of (timestamp, equity_float),
    oldest -> newest.

    We support mixed formats in ONE file:
    - legacy rows with (date, cash, market_value)
    - new rows with (timestamp, equity)

    We gather everything we can parse.
    """
    p = PORTFOLIO_HISTORY_PATH
    if not p.exists() or not p.is_file():
        return []

    rows: List[Tuple[dt.datetime, float]] = []

    try:
        with p.open("r", newline="") as f:
            r = csv.DictReader(f)
            # normalize headers (DictReader preserves header row)
            headers = [h.strip().lower() for h in (r.fieldnames or [])]

            legacy_schema = (
                "date" in headers and
                "cash" in headers and
                "market_value" in headers
            )
            live_schema = (
                "timestamp" in headers and
                "equity" in headers
            )

            # NOTE:
            # Even if DictReader saw only one header style,
            # the file might still contain appended rows from the other style.
            # So for each row we will TRY BOTH parsers.

            for row in r:
                # Try legacy parser
                # equity = cash + market_value
                ts_legacy = _parse_timestamp(row.get("date", "")) if "date" in row else None
                cash_val = _safe_float(row.get("cash")) if "cash" in row else None
                mv_val   = _safe_float(row.get("market_value")) if "market_value" in row else None

                if ts_legacy is not None and (cash_val is not None or mv_val is not None):
                    eq_val = (cash_val or 0.0) + (mv_val or 0.0)
                    rows.append((ts_legacy, eq_val))

                # Try live parser
                ts_live = _parse_timestamp(row.get("timestamp", "")) if "timestamp" in row else None
                eq_live = _safe_float(row.get("equity")) if "equity" in row else None

                if ts_live is not None and eq_live is not None:
                    rows.append((ts_live, eq_live))

    except Exception:
        # On any read/parsing failure, just return empty so caller fails soft
        return []

    # sort chronologically
    rows.sort(key=lambda x: x[0])

    # optional tail cutoff so file can grow forever
    if len(rows) > MAX_ROWS_FOR_ANALYSIS:
        rows = rows[-MAX_ROWS_FOR_ANALYSIS:]

    return rows


def _compute_drawdown(series: List[Tuple[dt.datetime, float]]):
    """
    Given [(ts, equity), ...] oldest->newest,
    compute:
        latest_equity
        peak_equity
        drawdown_pct  (positive number, e.g. 0.12 means -12%)

    Return tuple (drawdown_pct, latest_equity, peak_equity).
    If not enough data, return (0.0, last_equity_or_0, peak_or_0).
    """
    if not series:
        return 0.0, 0.0, 0.0

    # pull just the equity floats
    vals = [eq for (_, eq) in series]

    latest_equity = vals[-1]
    peak_equity   = max(vals) if vals else 0.0

    if peak_equity <= 0.0:
        return 0.0, latest_equity, peak_equity

    dd_raw = (peak_equity - latest_equity) / peak_equity
    if dd_raw < 0:
        dd_raw = 0.0  # if we're above peak, drawdown is 0

    return dd_raw, latest_equity, peak_equity


def get_portfolio_health() -> PortfolioHealth:
    """
    Public entry point Triton calls.

    - Reads/merges equity history (legacy + live).
    - Computes drawdown from high-water mark.
    - Returns PortfolioHealth.
    - Never raises.
    """
    series = _load_equity_series()

    if not series:
        return PortfolioHealth(
            drawdown_pct=0.0,
            latest_equity=0.0,
            peak_equity=0.0,
            data_points=0,
        )

    dd_pct, latest_eq, peak_eq = _compute_drawdown(series)

    return PortfolioHealth(
        drawdown_pct=dd_pct,
        latest_equity=latest_eq,
        peak_equity=peak_eq,
        data_points=len(series),
    )
