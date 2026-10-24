"""Temporary processed-bar fixtures for the prior-session start gate.

Writes only under the caller-supplied root. Does not touch production data.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from services.generate_signals import MIN_FEATURE_BARS, _is_index_ticker, _load_optional_tickers
from services.nyse_signal_session import required_signal_session

REPO = Path(__file__).resolve().parents[1]
TICKERS = REPO / "data" / "config" / "tickers.txt"


def governed_symbols() -> list[str]:
    return _load_optional_tickers(str(TICKERS))


def tradable_symbols() -> list[str]:
    return [symbol for symbol in governed_symbols() if not _is_index_ticker(symbol)]


def index_symbols() -> list[str]:
    return [symbol for symbol in governed_symbols() if _is_index_ticker(symbol)]


def install_ticker_universe(root: Path) -> Path:
    dest = root / "data" / "config" / "tickers.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(TICKERS.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def write_processed_bars(
    root: Path,
    symbol: str,
    session: date,
    *,
    rows: int = MIN_FEATURE_BARS,
    open_px: float = 10.0,
    high_px: float = 12.0,
    low_px: float = 9.0,
    close_px: object = 11.0,
    volume: object = 1000,
) -> Path:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    start = session - timedelta(days=max(rows, 1) - 1)
    lines = ["date,open,high,low,close,volume"]
    for offset in range(max(rows, 1)):
        day = start + timedelta(days=offset)
        lines.append(f"{day.isoformat()},{open_px},{high_px},{low_px},{close_px},{volume}")
    path = processed / f"{symbol}.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_unreadable_processed_file(root: Path, symbol: str) -> Path:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    path = processed / f"{symbol}.parquet"
    csv_path = processed / f"{symbol}.csv"
    if csv_path.is_file():
        csv_path.unlink()
    path.write_bytes(b"not-a-parquet-file")
    return path


def write_non_coverage_artifacts(root: Path, session: date) -> None:
    results = root / "data" / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "signals.csv").write_text("ticker,signal\nSPY,BUY\n", encoding="utf-8")
    (results / "pipeline_heartbeat.json").write_text(
        json.dumps({"status": "ok", "generated_at": "2026-10-23T13:00:00Z"}),
        encoding="utf-8",
    )
    (results / "daily_refresh_fetch_receipt.json").write_text(
        json.dumps(
            {
                "passed": True,
                "target_session": session.isoformat(),
                "failure": None,
                "schema": "fetch-only",
            }
        ),
        encoding="utf-8",
    )
    (results / "data_freshness_refresh_audit.json").write_text(
        json.dumps({"errors": [], "light_only": False, "timestamp": "2026-10-23T13:00:00Z"}),
        encoding="utf-8",
    )


def seed_prior_session_coverage(
    root: Path,
    *,
    covered: Optional[int] = None,
    now: Optional[datetime] = None,
    session: Optional[date] = None,
    include_indexes: bool = False,
) -> date:
    """Install the production universe and exact session bars under ``root``."""
    install_ticker_universe(root)
    session_date = session or required_signal_session(now)
    names = tradable_symbols()
    limit = len(names) if covered is None else covered
    for symbol in names[:limit]:
        write_processed_bars(root, symbol, session_date)
    if include_indexes:
        for symbol in index_symbols():
            write_processed_bars(root, symbol, session_date)
    return session_date
