import os
import glob
from pathlib import Path

import pandas as pd


def exists(p: str) -> bool:
    q = Path(p)
    return q.exists() and q.stat().st_size > 0


def head(df: pd.DataFrame, n: int = 3) -> None:
    try:
        print(df.head(n).to_string(index=False))
    except Exception:
        print(df.head(n))


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Removes BOM + trims whitespace; keeps original case
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def read_csv_norm(path: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    return _normalize_cols(df)


print("=== Triton data audit ===")

# A) Per-ticker parquets
paths = sorted(glob.glob(r"data/results/*.parquet"))
print(f"per-ticker parquet files: {len(paths)}")
if paths:
    sample = paths[0]
    try:
        df0 = pd.read_parquet(sample)
        print("sample parquet:", os.path.basename(sample))
        print("columns:", list(df0.columns))
        head(df0, 3)
    except Exception as e:
        print("?? could not read sample parquet:", e)

# B) Merged parquet
p_merged = Path(r"data/processed/stock_data.parquet")
print(
    "\nmerged exists:",
    p_merged.exists(),
    "size:",
    p_merged.stat().st_size if p_merged.exists() else 0,
)
if p_merged.exists():
    try:
        dfm = pd.read_parquet(p_merged)
        print("merged cols (first 15):", list(dfm.columns)[:15], "...")
        print("rows:", len(dfm))
        if "ticker" in dfm.columns:
            print("tickers:", dfm["ticker"].nunique())
            print(dfm.groupby("ticker").size().sort_values(ascending=False).head(10))
    except Exception as e:
        print("?? could not read merged parquet:", e)

# C) fundamentals & stock_scores presence/shape
for name in [r"data/results/fundamentals.csv", r"data/results/stock_scores.csv"]:
    ok = exists(name)
    print(f"\n{name} exists:", ok)
    if ok:
        try:
            d = read_csv_norm(name)
            print("rows:", len(d), "cols:", list(d.columns))
            if "ticker" in d.columns:
                print("tickers sample:", d["ticker"].dropna().unique()[:8])
        except Exception as e:
            print(f"?? could not read {name}:", e)

# D) Signals files (for simulate_portfolio date/timestamp issue)
for name in [r"data/results/signals_with_rationale.csv", r"data/results/signals.csv"]:
    ok = exists(name)
    print(f"\n{name} exists:", ok)
    if ok:
        try:
            d = read_csv_norm(name, nrows=5)
            print("cols:", list(d.columns))
            print(
                "has 'date'?", "date" in d.columns, "| has 'timestamp'?", "timestamp" in d.columns
            )
            cols = [c for c in ["ticker", "signal"] if c in d.columns]
            if cols:
                print(d[cols].head(3).to_string(index=False))
        except Exception as e:
            print(f"?? could not read {name}:", e)

# E) Portfolio history (for drawdown tab)
ph_path = r"data/results/portfolio_history.csv"
ok = exists(ph_path)
print(f"\n{ph_path} exists:", ok)
if ok:
    try:
        ph = read_csv_norm(ph_path, nrows=5)
        print("portfolio_history cols:", list(ph.columns))
        head(ph, 3)
    except Exception as e:
        print("?? could not read portfolio_history:", e)

# F) Positions snapshot (validator-critical)
pos_path = r"data/results/positions_snapshot.csv"
ok = exists(pos_path)
print(f"\n{pos_path} exists:", ok)
if ok:
    try:
        ps = read_csv_norm(pos_path, nrows=5)
        print("positions_snapshot cols:", [repr(c) for c in ps.columns])
        print("has 'symbol'?", "symbol" in ps.columns, "| has 'ticker'?", "ticker" in ps.columns)

        # Auditing safety (does NOT rewrite file):
        if "symbol" not in ps.columns and "ticker" in ps.columns:
            print("[audit] NOTE: symbol missing; could be derived from ticker.")

        cols = [
            c
            for c in ["snapshot_ts", "date", "symbol", "ticker", "qty", "market_value", "value"]
            if c in ps.columns
        ]
        if cols:
            print(ps[cols].head(3).to_string(index=False))
        else:
            head(ps, 3)
    except Exception as e:
        print("?? could not read positions_snapshot:", e)

print("\n✓ Audit complete.")
