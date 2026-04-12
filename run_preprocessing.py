# run_preprocessing.py
# --------------------
# Preprocess raw OHLCV CSVs into per-ticker parquet files in data/processed.
#
# Key behavior:
# 1) Prefer "live" raw files:     data/raw/{TICKER}.csv
# 2) Fall back to "legacy" files: data/raw/{TICKER}_YYYY-MM-DD_to_YYYY-MM-DD.csv
# 3) If multiple legacy files exist for a ticker, use the most recent by end-date (or mtime fallback)
#
# PERMANENT AS_OF FIX (Jan 2026):
# - Compute AS_OF_DATE as the *last completed NYSE session* (ET) (authoritative)
# - After preprocessing, verify df.max(date) >= AS_OF_DATE (date-grain)
# - If behind: SKIP writing parquet (always). If strict flag is on, exit non-zero at end.
#
# NEW (Step B hardening):
# - If a LIVE file exists but is stale, auto-fallback to freshest legacy file (if newer).
#   This prevents Triton from repeatedly reprocessing stale "live" CSVs.

from __future__ import annotations

import os
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, List

import pandas as pd

from services.preprocess_data import preprocess_stock_csv

RAW_DIR = Path("data") / "raw"
PROCESSED_DIR = Path("data") / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Strictness
# -----------------------------
STRICT_ASOF = os.getenv("TRITON_STRICT_ASOF", "0").strip().lower() in ("1", "true", "yes", "y")

# If true (default), allow stale live files to fall back to newer legacy files
ALLOW_LIVE_STALE_FALLBACK = os.getenv("TRITON_ALLOW_LIVE_STALE_FALLBACK", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
)

# -----------------------------
# Patterns
# -----------------------------
LIVE_PAT = re.compile(r"^([A-Z0-9\-\^]+)\.csv$", re.IGNORECASE)
LEGACY_PAT = re.compile(
    r"^([A-Z0-9\-\^]+)_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE
)


# -----------------------------
# AS_OF helpers (session-authoritative)
# -----------------------------
def _et_now() -> datetime:
    """Best-effort current time in America/New_York; fallback approximates ET from UTC."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now(timezone.utc) - timedelta(hours=4)


def _fallback_last_completed_session_date() -> str:
    """
    Fallback (NOT holiday-aware; avoids intraday drift):
    - If Mon–Fri and time >= 16:00 ET -> today
    - Otherwise -> previous weekday
    """
    now = _et_now()
    d = now.date()

    def prev_weekday(x):
        while x.weekday() >= 5:
            x = x - timedelta(days=1)
        return x

    # after close -> today counts as completed
    if d.weekday() < 5 and (now.hour > 16 or (now.hour == 16 and now.minute >= 0)):
        return d.isoformat()

    # before close/weekend -> previous weekday
    d2 = prev_weekday(d - timedelta(days=1))
    return d2.isoformat()


def get_as_of_date() -> str:
    """
    Canonical AS_OF_DATE (date string YYYY-MM-DD):
    Prefer services.market_calendar.last_completed_nyse_session() if available.
    Otherwise fallback (weekday + 16:00 ET rule).
    """
    try:
        # ✅ Authoritative (holiday-aware when exchange_calendars is installed)
        from services.market_calendar import last_completed_nyse_session  # type: ignore

        v = str(last_completed_nyse_session()).strip()
        if v:
            return v
    except Exception:
        pass

    return _fallback_last_completed_session_date()


def normalize_date_series(s: pd.Series) -> pd.Series:
    """
    Parse dates robustly:
    - Accept date strings, timestamps, tz-aware or naive
    - Normalize to date-grain (midnight, tz-naive)
    """
    dt = pd.to_datetime(s, errors="coerce", utc=False)

    # If tz-aware, convert to UTC then drop tz
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        # If dt isn't a datetime series for any reason, fall through
        pass

    # Normalize to midnight
    try:
        dt = dt.dt.normalize()
    except Exception:
        pass

    return dt


def _legacy_end_date(fn: str) -> Optional[pd.Timestamp]:
    m = LEGACY_PAT.match(fn)
    if not m:
        return None
    end = pd.to_datetime(m.group(3), errors="coerce")
    if pd.isna(end):
        return None
    return end.normalize()


def pick_best_legacy(raw_dir: Path, files_for_ticker: List[str]) -> str:
    """
    Pick the best legacy file:
    - Prefer the one with the latest end-date in the filename
    - Fall back to most recent file mtime if parsing fails
    """
    parsed: List[Tuple[str, str]] = []
    for fn in files_for_ticker:
        m = LEGACY_PAT.match(fn)
        if not m:
            continue
        end_date = m.group(3)
        parsed.append((end_date, fn))

    if parsed:
        parsed.sort(key=lambda x: x[0])
        return parsed[-1][1]

    files_for_ticker.sort(key=lambda fn: (raw_dir / fn).stat().st_mtime)
    return files_for_ticker[-1]


def _quick_last_date_from_preprocessed(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if not isinstance(df, pd.DataFrame) or df.empty or "date" not in df.columns:
        return None
    dts = normalize_date_series(df["date"])
    mx = pd.to_datetime(dts.max(), errors="coerce")
    if pd.isna(mx):
        return None
    return mx.normalize()


def _load_and_preprocess(path: Path) -> pd.DataFrame:
    df = preprocess_stock_csv(str(path))
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df


def main() -> int:
    print("🔄 Starting preprocessing...")

    as_of_str = get_as_of_date()
    as_of_dt = pd.to_datetime(as_of_str, errors="coerce")
    if pd.isna(as_of_dt):
        raise RuntimeError(f"Invalid AS_OF_DATE computed: {as_of_str!r}")
    as_of_dt = as_of_dt.normalize()

    print(
        f"📌 AS_OF_DATE (last completed NYSE session): {as_of_str}  | strict={STRICT_ASOF} | live_fallback={ALLOW_LIVE_STALE_FALLBACK}"
    )

    if not RAW_DIR.exists():
        print(f"❌ Raw data folder not found: {RAW_DIR}")
        return 1

    raw_files = [f.name for f in RAW_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".csv"]
    if not raw_files:
        print(f"⚠️ No CSV files found in {RAW_DIR}")
        return 0

    # Build ticker -> chosen raw file map
    chosen: Dict[str, str] = {}

    # 1) live files always win initially
    for fn in raw_files:
        m = LIVE_PAT.match(fn)
        if m:
            ticker = m.group(1).upper()
            chosen[ticker] = fn

    # 2) collect legacy files
    legacy_by_ticker: Dict[str, List[str]] = {}
    for fn in raw_files:
        m = LEGACY_PAT.match(fn)
        if not m:
            continue
        ticker = m.group(1).upper()
        legacy_by_ticker.setdefault(ticker, []).append(fn)

    # 3) for tickers without live, choose best legacy
    for ticker, files_for_ticker in legacy_by_ticker.items():
        if ticker in chosen:
            continue
        chosen[ticker] = pick_best_legacy(RAW_DIR, files_for_ticker)

    processed_count = 0
    skipped_count = 0
    error_count = 0
    behind_count = 0
    behind_details: List[dict] = []

    # Optional: reuse DF if we already preloaded live for stale-check
    preloaded_df: Dict[str, pd.DataFrame] = {}

    for ticker in sorted(chosen.keys()):
        fn = chosen[ticker]
        file_path = RAW_DIR / fn

        # Step B: If LIVE is stale, optionally fall back to freshest legacy (if newer).
        if (
            ALLOW_LIVE_STALE_FALLBACK
            and LIVE_PAT.match(fn)
            and ticker in legacy_by_ticker
            and legacy_by_ticker[ticker]
        ):
            try:
                df_live = _load_and_preprocess(file_path)
                preloaded_df[ticker] = df_live
                last_live = _quick_last_date_from_preprocessed(df_live)
            except Exception:
                last_live = None

            if last_live is not None and last_live < as_of_dt:
                best_legacy = pick_best_legacy(RAW_DIR, legacy_by_ticker[ticker])
                legacy_end = _legacy_end_date(best_legacy)

                # If legacy end date is newer than live last date, use legacy
                if legacy_end is not None and legacy_end > last_live:
                    print(
                        f"↩️ {ticker}: LIVE stale ({last_live.date()}) → using legacy {best_legacy} (end={legacy_end.date()})"
                    )
                    fn = best_legacy
                    file_path = RAW_DIR / fn
                    # If we switch away from live, drop preload
                    preloaded_df.pop(ticker, None)
                else:
                    print(
                        f"⚠️ {ticker}: LIVE stale ({last_live.date()}) and no newer legacy found → keeping LIVE"
                    )

        try:
            # Load/preprocess (reuse preload when possible)
            if ticker in preloaded_df and file_path.name == chosen[ticker]:
                df = preloaded_df[ticker]
            else:
                df = _load_and_preprocess(file_path)

            if not isinstance(df, pd.DataFrame) or df.empty:
                print(f"⚠️ Skipped {fn} (empty after preprocessing)")
                skipped_count += 1
                continue

            if "date" not in df.columns:
                print(f"❌ {ticker}: preprocessed df missing 'date' column. Skipping.")
                error_count += 1
                continue

            df = df.copy()
            df["date"] = normalize_date_series(df["date"])
            df = df.dropna(subset=["date"])

            if df.empty:
                print(f"⚠️ {ticker}: no valid dates after parsing. Skipping.")
                skipped_count += 1
                continue

            last_dt = pd.to_datetime(df["date"].max(), errors="coerce")
            if pd.isna(last_dt):
                print(f"⚠️ {ticker}: could not parse max(date). Skipping.")
                skipped_count += 1
                continue
            last_dt = last_dt.normalize()

            # ✅ AS_OF completeness check (ALWAYS skip write if behind)
            if last_dt < as_of_dt:
                behind_count += 1
                behind_details.append(
                    {
                        "ticker": ticker,
                        "raw_file": fn,
                        "last_date": str(last_dt.date()),
                        "as_of_date": as_of_str,
                    }
                )
                msg = f"⛔ {ticker}: STALE RAW/PROCESSED DATA (last_date={last_dt.date()} < AS_OF_DATE={as_of_str})"
                print(msg + " — skipping parquet write")
                skipped_count += 1
                continue

            output_path = PROCESSED_DIR / f"{ticker}.parquet"
            df.to_parquet(output_path, index=False)
            print(f"📦 Saved processed {ticker} to {output_path}")
            processed_count += 1

        except Exception as e:
            print(f"❌ Error processing {fn}: {e}")
            error_count += 1

    # Optional: write a behind_asof report to logs
    if behind_details:
        try:
            report_path = Path("data") / "logs" / "preprocess_behind_asof.csv"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(behind_details).to_csv(report_path, index=False)
            print(f"🧾 Behind-ASOF report: {report_path}")
        except Exception:
            pass

    print(
        f"✅ All preprocessing done. "
        f"processed={processed_count} skipped={skipped_count} behind_asof={behind_count} errors={error_count}"
    )

    # If strict mode and anything was behind AS_OF, fail hard so upstream fetch gets fixed.
    if STRICT_ASOF and behind_details:
        raise SystemExit(
            "Preprocessing strict AS_OF failed: some tickers are missing the last completed session bar.\n"
            "Fix upstream fetch: download daily bars so the dataset includes AS_OF_DATE, "
            "then rerun: fetch_raw_data -> run_preprocessing -> train -> signals."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
