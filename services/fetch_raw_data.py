# services/fetch_raw_data.py
# --------------------------
# Fetch daily OHLCV from yfinance into data/raw/{TICKER}.csv
# Guarantees columns: date, open, high, low, close, adj_close, volume
#
# Deterministic AS_OF hardening (Jan 2026):
# - Default fetch window is driven by "last completed NYSE session" (ET),
#   same AS_OF used by preprocessing/training/signals/contracts.
# - yfinance 'end' is treated as EXCLUSIVE → fetch end = (AS_OF_DATE + 1 day)
# - AS_OF freshness allows a small calendar lag (ALLOW_DATA_LAG_DAYS): latest bar may
#   trail AS_OF by up to that many days and still count toward asof_ok / min-asof-ok.
#
# Hardening patch:
# - Retries with exponential backoff for flaky yfinance timeouts
# - Supports a longer timeout (default 30s)
# - If download fails, reads existing data/raw/{TICKER}.csv and reports last_date
#   (so we can see whether we're truly behind AS_OF or it was already OK)
# - Writes data/results/raw_fetch_diagnostics.csv with per-ticker status + last_date

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
from typing import List, Optional, Tuple

import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise RuntimeError("yfinance is required. pip install yfinance") from e


RAW_DIR = Path("data") / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("data") / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DIAG_PATH = RESULTS_DIR / "raw_fetch_diagnostics.csv"
SUMMARY_JSON = RESULTS_DIR / "raw_fetch_summary.json"

DEFAULT_TICKERS_FILE = Path("data") / "config" / "tickers.txt"

# Latest bar may trail calendar AS_OF (e.g. provider delay); still treat as OK within this window.
ALLOW_DATA_LAG_DAYS = 2

# Accept common index tickers and BRK-B style
TICKER_RE = re.compile(r"^[A-Z0-9.\-\^]{1,15}$")

CANON_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "adjclose": "adj_close",
    "volume": "volume",
    "date": "date",
    "datetime": "date",
}

NEEDED = ["date", "open", "high", "low", "close", "adj_close", "volume"]


# ─────────────────────────────────────────────────────────────
# AS_OF helpers
# ─────────────────────────────────────────────────────────────


def _compute_asof_date(now_utc: Optional[datetime] = None) -> date:
    try:
        from services.market_calendar import last_completed_nyse_session  # type: ignore

        d = last_completed_nyse_session()
        if isinstance(d, date):
            return d
        return pd.to_datetime(str(d), errors="coerce").date()
    except Exception:
        n = now_utc or datetime.now(timezone.utc)
        d = n.date()
        if d.weekday() == 5:
            return d - timedelta(days=1)
        if d.weekday() == 6:
            return d - timedelta(days=2)
        return d - timedelta(days=1)


def _asof_to_end_exclusive(asof: date) -> str:
    return (asof + timedelta(days=1)).isoformat()


def _parse_last_date(last: Optional[str]) -> Optional[date]:
    if not last:
        return None
    try:
        d = pd.to_datetime(str(last).strip(), errors="coerce")
        if pd.isna(d):
            return None
        return d.date()
    except Exception:
        return None


def _asof_lag_days(expected: date, latest: Optional[date]) -> Optional[int]:
    """Calendar days expected is after latest (positive = data behind). None if unknown latest."""
    if latest is None:
        return None
    return (expected - latest).days


def _asof_within_tolerance(expected: date, latest: Optional[date], allow_days: int) -> bool:
    if latest is None:
        return False
    lag = _asof_lag_days(expected, latest)
    if lag is None:
        return False
    return lag <= allow_days


# ─────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    lvl0 = {str(x).strip().lower() for x in df.columns.get_level_values(0)}
    lvl1 = {str(x).strip().lower() for x in df.columns.get_level_values(1)}
    ohlcv = {"open", "high", "low", "close", "adj close", "adj_close", "volume"}

    if len(lvl0 & ohlcv) >= 3:
        df.columns = [c[0] for c in df.columns]
    elif len(lvl1 & ohlcv) >= 3:
        df.columns = [c[1] for c in df.columns]
    else:
        df.columns = ["_".join([str(a), str(b)]) for a, b in df.columns]

    return df


def _normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _flatten_cols(df)

    if "date" not in {str(c).strip().lower() for c in df.columns}:
        df = df.reset_index()

    df.columns = [str(c).strip().lower() for c in df.columns]

    rename = {c: CANON_MAP[c] for c in df.columns if c in CANON_MAP}
    df = df.rename(columns=rename)

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    for c in NEEDED:
        if c not in df.columns:
            df[c] = pd.NA

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df[NEEDED]


def _is_valid_token(tok: str) -> bool:
    tok = tok.strip().upper()
    if not tok:
        return False
    if tok.startswith("#"):
        return False
    if not TICKER_RE.match(tok):
        return False
    return True


def _split_tickers(items: List[str]) -> List[str]:
    out: List[str] = []
    for x in items:
        if not x:
            continue
        parts = [p.strip() for p in str(x).split(",") if p.strip()]
        out.extend(parts)

    out = [t.strip().upper() for t in out]
    out = [t for t in out if _is_valid_token(t)]

    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _load_tickers_from_file(path: Path) -> List[str]:
    if not (path.exists() and path.stat().st_size > 0):
        return []

    raw = path.read_text(encoding="utf-8-sig").splitlines()

    cleaned: List[str] = []
    for ln in raw:
        if not ln:
            continue
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        cleaned.append(ln)

    return _split_tickers(cleaned)


def _existing_last_date(ticker: str) -> Optional[str]:
    p = RAW_DIR / f"{ticker}.csv"
    if not (p.exists() and p.stat().st_size > 0):
        return None
    try:
        df = pd.read_csv(p)
        if "date" not in df.columns or df.empty:
            return None
        d = pd.to_datetime(df["date"], errors="coerce")
        if d.isna().all():
            return None
        return str(d.max().date())
    except Exception:
        return None


def _download_one(
    ticker: str, start: str, end: str, timeout_s: int, verbose: bool
) -> Tuple[bool, str, Optional[str], int]:
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            timeout=timeout_s,
        )

        if df is None or df.empty:
            return False, "EMPTY", None, 0

        norm = _normalize_yf_df(df)
        if norm is None or norm.empty:
            cols = list(df.columns)[:12]
            msg = (
                f"EMPTY_AFTER_NORMALIZE cols_sample={cols}" if verbose else "EMPTY_AFTER_NORMALIZE"
            )
            return False, msg, None, 0

        out = RAW_DIR / f"{ticker}.csv"
        norm.to_csv(out, index=False)

        last = str(norm["date"].max()) if "date" in norm.columns else None
        msg = f"rows={len(norm)} max={last}" if verbose else "OK"
        return True, msg, last, int(len(norm))

    except Exception as e:
        return False, f"ERROR {e}", None, 0


def fetch_one_with_retries(
    ticker: str,
    start: str,
    end: str,
    retries: int,
    timeout_s: int,
    verbose: bool,
) -> Tuple[bool, str, Optional[str], int, str]:
    """
    Returns: (ok, msg, last_date, rows, source)
    source: 'download' or 'existing'
    """
    last_err = ""
    for i in range(max(1, retries)):
        ok, msg, last_date, rows = _download_one(ticker, start, end, timeout_s, verbose)
        if ok:
            return True, msg, last_date, rows, "download"

        last_err = msg
        # backoff before retry
        if i < retries - 1:
            time.sleep(min(8.0, 1.5**i))

    # download failed; fall back to existing raw file info (do NOT overwrite)
    existing = _existing_last_date(ticker)
    if existing:
        return False, f"{last_err} | using existing last_date={existing}", existing, 0, "existing"

    return False, last_err, None, 0, "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tickers", nargs="*", default=None, help="Tickers (space or comma separated)."
    )
    ap.add_argument("--tickers-file", default=None, help="Path to a tickers file (one per line).")
    ap.add_argument("--start", default="2020-01-01")

    ap.add_argument("--asof", default=None, help="Override AS_OF_DATE (YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (exclusive). Default: AS_OF + 1 day.")

    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--min-ok", type=int, default=1)
    ap.add_argument("--min-asof-ok", type=int, default=0)
    ap.add_argument(
        "--allow-data-lag-days",
        type=int,
        default=ALLOW_DATA_LAG_DAYS,
        help=f"Max calendar days latest bar may trail AS_OF and still count as ok (default: {ALLOW_DATA_LAG_DAYS}).",
    )

    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=30, help="yfinance timeout seconds")

    args = ap.parse_args()

    # AS_OF + end
    if args.asof:
        asof_dt = pd.to_datetime(args.asof, errors="coerce")
        if pd.isna(asof_dt):
            raise SystemExit(f"Invalid --asof: {args.asof!r}")
        asof = asof_dt.date()
    else:
        asof = _compute_asof_date()

    end = args.end or _asof_to_end_exclusive(asof)

    # tickers
    tickers: List[str] = []
    if args.tickers:
        tickers = _split_tickers(args.tickers)
    if not tickers and args.tickers_file:
        tickers = _load_tickers_from_file(Path(args.tickers_file))
    if not tickers and DEFAULT_TICKERS_FILE.exists():
        tickers = _load_tickers_from_file(DEFAULT_TICKERS_FILE)
    if not tickers:
        raise SystemExit("No tickers provided and no default tickers file found.")

    allow_lag = max(0, int(args.allow_data_lag_days))
    print(
        f"[fetch_raw] asof={asof.isoformat()} end(exclusive)={end} start={args.start} "
        f"tickers={len(tickers)} retries={args.retries} timeout={args.timeout}s "
        f"allow_data_lag_days={allow_lag}"
    )

    ok_count = 0
    asof_ok = 0
    lagging_ok_count = 0
    lagging_exceeded_count = 0
    diag_rows: List[dict] = []

    for t in tickers:
        ok, msg, last_date, rows, source = fetch_one_with_retries(
            t,
            start=args.start,
            end=end,
            retries=args.retries,
            timeout_s=args.timeout,
            verbose=args.verbose,
        )

        # 'ok' means a fresh download occurred; but ASOF coverage is based on last_date.
        if ok:
            ok_count += 1

        latest_dt = _parse_last_date(last_date)
        lag_days = _asof_lag_days(asof, latest_dt)
        reached_strict = bool(latest_dt and latest_dt >= asof)
        reached = _asof_within_tolerance(asof, latest_dt, allow_lag)
        if reached:
            asof_ok += 1
            if lag_days is not None and lag_days > 0:
                lagging_ok_count += 1
        else:
            if lag_days is not None and lag_days > allow_lag:
                lagging_exceeded_count += 1
            elif latest_dt is None:
                lagging_exceeded_count += 1

        print(f"[fetch_raw] {t}: {msg}")
        if reached and lag_days is not None and lag_days > 0:
            print(f"[fetch_raw] {t}: lag={lag_days} days (within tolerance)")

        diag_rows.append(
            {
                "ticker": t,
                "download_ok": bool(ok),
                "rows": int(rows),
                "last_date": last_date or "",
                "asof": asof.isoformat(),
                "reached_asof": bool(reached),
                "reached_asof_strict": bool(reached_strict),
                "lag_days": "" if lag_days is None else int(lag_days),
                "allow_data_lag_days": allow_lag,
                "start": args.start,
                "end_exclusive": end,
                "source": source,
                "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "message": msg,
            }
        )

    try:
        pd.DataFrame(diag_rows).to_csv(DIAG_PATH, index=False)
        print(f"[fetch_raw] diagnostics: {DIAG_PATH}")
    except Exception:
        pass

    summary_payload = {
        "asof": asof.isoformat(),
        "allow_data_lag_days": allow_lag,
        "asof_ok": asof_ok,
        "tickers": len(tickers),
        "lagging_ok_count": lagging_ok_count,
        "lagging_exceeded_count": lagging_exceeded_count,
        "download_ok": ok_count,
    }
    try:
        SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"[fetch_raw] summary: {SUMMARY_JSON}")
    except Exception:
        pass

    print(
        f"[fetch_raw] asof_ok={asof_ok}/{len(tickers)} lagging_ok_count={lagging_ok_count} "
        f"lagging_exceeded_count={lagging_exceeded_count} download_ok={ok_count}/{len(tickers)}"
    )

    if ok_count < args.min_ok:
        return 2
    if args.min_asof_ok and asof_ok < args.min_asof_ok:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
