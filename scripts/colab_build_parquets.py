import os, time, random, math, concurrent.futures as fut
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LOGS = ROOT / "data" / "logs"
RESULTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# Map Yahoo indices to ETF proxies (more reliable)
INDEX_PROXIES = {"^GSPC":"SPY", "^DJI":"DIA", "^IXIC":"QQQ", "^VIX":"VIXY"}
ALIASES = {"BRK-B":["BRK-B","BRK.B"], "BF-B":["BF-B","BF.B"]}

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume","Date":"date"})
    return df

def fetch_one(ticker: str, attempts=3, wait=2.0):
    fetch_symbol = INDEX_PROXIES.get(ticker, ticker)
    candidates = ALIASES.get(fetch_symbol, [fetch_symbol])
    last_err = "unknown"
    for a in range(1, attempts+1):
        for sym in candidates:
            try:
                df = yf.download(sym, period="10y", interval="1d", auto_adjust=False, progress=False, threads=False, repair=True, timeout=30)
                if df is None or df.empty: raise ValueError("empty df")
                df = normalize_cols(df).reset_index()
                if "close" not in df.columns: raise ValueError("no close col")
                df["ticker"] = ticker
                return df
            except Exception as e:
                last_err = str(e)
        if a < attempts:
            time.sleep(wait)
    raise RuntimeError(last_err)

def write_parquet(df: pd.DataFrame, out_path: Path):
    df.to_parquet(out_path, index=False)

def main():
    # Ticker universe from fundamentals + scores (assumes both exist)
    fin = RESULTS / "fundamentals.csv"
    sco = RESULTS / "stock_scores.csv"
    if not fin.exists() or not sco.exists():
        raise SystemExit("Place fundamentals.csv and stock_scores.csv in data/results/ first.")

    fundamentals = pd.read_csv(fin)
    scores = pd.read_csv(sco)
    fundamentals["ticker"] = fundamentals["ticker"].astype(str).str.upper()
    scores["ticker"] = scores["ticker"].astype(str).str.upper()
    universe = sorted(set(fundamentals["ticker"]).union(set(scores["ticker"])))

    # Skip raw index tickers (we’ll fetch proxies instead)
    universe = [t for t in universe if not t.startswith("^")]

    existing = {p.stem.upper() for p in RESULTS.glob("*.parquet")}
    todo = [t for t in universe if t not in existing]
    print(f"Need to fetch {len(todo)} tickers (skipping {len(existing)} already present).")

    errors = []
    saved = 0

    def worker(t):
        try:
            df = fetch_one(t)
            write_parquet(df, RESULTS / f"{t}.parquet")
            return (t, True, "")
        except Exception as e:
            return (t, False, str(e))

    # Parallel with small pool (avoid hammering Yahoo)
    max_workers = 6
    with fut.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for t, ok, err in ex.map(worker, todo):
            if ok:
                saved += 1
                if saved % 10 == 0:
                    print(f"Saved {saved}/{len(todo)}...")
            else:
                errors.append((t, err))

    # Log errors
    if errors:
        log = LOGS / "colab_fetch_errors.txt"
        with open(log, "w") as f:
            for t, e in errors:
                f.write(f"{t}: {e}\n")
        print(f"⚠️ Errors for {len(errors)} tickers. See {log}")

    print(f"✅ Done. Parquets written: {saved}")

if __name__ == "__main__":
    main()
