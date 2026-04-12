from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_NAME_RE = re.compile(
    r"^([A-Za-z0-9.\-]+)_\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}\.csv$",
    re.IGNORECASE,
)


def infer_ticker(name: str) -> str | None:
    m = RAW_NAME_RE.match(name)
    if not m:
        return None
    return m.group(1).upper()


def get_date_col(df: pd.DataFrame) -> str | None:
    if "date" in df.columns:
        return "date"
    if "Date" in df.columns:
        return "Date"
    return None


def read_last_date(path: Path) -> pd.Timestamp | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    c = get_date_col(df)
    if not c:
        return None
    mx = pd.to_datetime(df[c], errors="coerce").max()
    if pd.isna(mx):
        return None
    return pd.Timestamp(mx).normalize()


def _flatten_cols(cols) -> list[str]:
    # Handles MultiIndex/tuples from yfinance
    if isinstance(cols, pd.MultiIndex):
        out = []
        for col in cols:
            parts = [str(x) for x in col if x not in (None, "", " ")]
            out.append("_".join(parts).strip())
        return out
    return [str(c).strip() for c in cols]


def yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Sometimes columns are MultiIndex like ('Open','AAPL')
    df.columns = _flatten_cols(df.columns)

    df = df.reset_index()

    # Normalize date column name
    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Date": "date"})

    # Lowercase + snake_case
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    return df


def normalize_existing(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Date": "date"})
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df


def main():
    print(f"📦 RAW_DIR: {RAW_DIR}")

    files = sorted(RAW_DIR.glob("*.csv"))
    file_map: dict[str, Path] = {}
    for p in files:
        t = infer_ticker(p.name)
        if t:
            file_map[t] = p

    tickers = sorted(file_map.keys())
    print(f"🎯 Tickers: {len(tickers)}")

    today = pd.Timestamp.today().normalize()
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # exclusive
    print(f"📅 Target end (exclusive): {end}")

    updated = 0
    for t in tickers:
        p = file_map[t]
        last = read_last_date(p)
        if last is None:
            print(f"⚠️  {t}: cannot read last date from {p.name}, skipping")
            continue

        if last >= today:
            print(f"✅ {t}: already up-to-date (last={last.date()})")
            continue

        start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"⬇️  {t}: downloading {start} → {end} (last={last.date()})")

        new = yf_download(t, start=start, end=end)
        if new.empty:
            print(f"⚠️  {t}: no new rows returned")
            continue

        old = normalize_existing(pd.read_csv(p))
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.dropna(subset=["date"]).sort_values("date")
        merged = merged.drop_duplicates(subset=["date"], keep="last")

        merged.to_csv(p, index=False)
        updated += 1
        print(f"✔ {t}: wrote {len(merged)} rows; latest={merged['date'].max().date()}")

    print(f"\n🏁 Done. Updated files: {updated}/{len(tickers)}")


if __name__ == "__main__":
    main()
