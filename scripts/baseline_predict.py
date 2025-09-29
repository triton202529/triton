from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("data/results")
PRED_DIR    = Path("data/predictions")
PRED_DIR.mkdir(parents=True, exist_ok=True)

rows = []
files = sorted(RESULTS_DIR.glob("*.parquet"))
if not files:
    raise SystemExit("No per-ticker parquet files found in data/results")

for fp in files:
    ticker = fp.stem  # filename -> ticker
    try:
        df = pd.read_parquet(fp)
        # normalize column names
        lower = {c: c.lower() for c in df.columns}
        df = df.rename(columns=lower)
        # if the datetime is the index, expose it
        if not any(c in df.columns for c in ("date","datetime","time")) and df.index.inferred_type in ("datetime64","datetime","date"):
            df = df.reset_index().rename(columns={df.columns[0]: "date"})
        # find close column
        close_col = next((c for c in ("close","adjclose","adj_close") if c in df.columns), None)
        if close_col is None:
            print(f"⚠️  {ticker}: no Close column; skipping.")
            continue
        df["sma20"] = df[close_col].rolling(20).mean()
        df["sma50"] = df[close_col].rolling(50).mean()
        if len(df) < 50 or pd.isna(df["sma50"].iloc[-1]):
            print(f"⚠️  {ticker}: not enough data for SMA50; skipping.")
            continue
        last = df.iloc[-1]
        score = float((last.sma20 - last.sma50) / last.sma50)
        signal = "BUY" if score > 0 else "SELL"
        rationale = f"SMA20 {'>' if score>0 else '<='} SMA50; spread={score:.3%}; close≈{float(last[close_col]):.2f}"
        rows.append({
            "ticker": ticker,
            "date":  pd.to_datetime(last.get("date", pd.Timestamp.utcnow())).normalize(),
            "score": score,
            "signal": signal,
            "rationale": rationale,
        })
    except Exception as e:
        print(f"❌ {ticker}: {e}")

preds = pd.DataFrame(rows)
if preds.empty:
    raise SystemExit("No predictions generated.")

preds = preds.sort_values(["score","ticker"], ascending=[False, True])
preds.to_parquet(PRED_DIR / "baseline_preds.parquet", index=False)
preds[["ticker","signal"]].to_csv(PRED_DIR / "signals.csv", index=False)

print(f"✅ Wrote {len(preds)} predictions -> {PRED_DIR/'baseline_preds.parquet'}")
print(f"✅ Wrote CSV signals         -> {PRED_DIR/'signals.csv'}")
