import os
import glob
import pandas as pd

PRED_DIR = "data/predictions"
RESULTS_DIR = "data/results"
OUT_CSV = os.path.join(RESULTS_DIR, "signals_with_rationale.csv")
SCORES_CSV = os.path.join(RESULTS_DIR, "stock_scores.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

def normalize_date(series):
    # Make everything tz-aware UTC, then drop tz -> tz-naive
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_localize(None)

def load_scores():
    if os.path.exists(SCORES_CSV):
        try:
            s = pd.read_csv(SCORES_CSV)
            s.columns = [c.strip().lower() for c in s.columns]
            if "ticker" in s.columns:
                s["ticker"] = s["ticker"].astype(str).str.upper()
            return s[["ticker", "total_score"]].drop_duplicates()
        except Exception as e:
            print(f"⚠️ Could not read {SCORES_CSV}: {e}")
    return pd.DataFrame(columns=["ticker", "total_score"])

def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "*_predictions.parquet")))
    if not files:
        print(f"❌ No prediction parquet files found in {PRED_DIR}")
        return

    frames = []
    for fp in files:
        try:
            df = pd.read_parquet(fp)
            # expected columns: date, close, predicted_close, signal, ticker
            df.columns = [c.strip().lower() for c in df.columns]

            # Some writers might not include ticker; infer from filename if missing
            if "ticker" not in df.columns or df["ticker"].isna().all():
                base = os.path.basename(fp).replace("_predictions.parquet", "")
                df["ticker"] = base

            # Normalize
            if "date" not in df.columns:
                # sometimes index is date
                df = df.reset_index().rename(columns={"index": "date"})
            df["date"] = normalize_date(df["date"])

            for col in ["close", "predicted_close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["ticker"] = df["ticker"].astype(str).str.upper()
            if "signal" in df.columns:
                df["signal"] = df["signal"].astype(str).str.upper()
            else:
                df["signal"] = "HOLD"

            # keep only what we need
            keep = ["date", "ticker", "close", "predicted_close", "signal"]
            df = df[[c for c in keep if c in df.columns]]
            frames.append(df.dropna(subset=["date"]))
        except Exception as e:
            print(f"⚠️ Skipping {fp}: {e}")

    if not frames:
        print("❌ No valid prediction frames to combine.")
        return

    signals = pd.concat(frames, ignore_index=True)
    # Merge scores if available
    scores = load_scores()
    if not scores.empty:
        signals = signals.merge(scores, on="ticker", how="left")

    # Build a lightweight rationale string
    def mk_rat(row):
        bits = []
        if pd.notna(row.get("predicted_close", None)) and pd.notna(row.get("close", None)):
            bits.append(f"Pred {row['predicted_close']:.2f} vs {row['close']:.2f}")
        bits.append(f"Signal {row.get('signal','HOLD')}")
        if pd.notna(row.get("total_score", None)):
            bits.append(f"Score {row['total_score']:.1f}")
        return " | ".join(bits)

    signals["rationale"] = signals.apply(mk_rat, axis=1)

    # Sort consistently
    signals = signals.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Final column order
    cols = ["date", "ticker", "close", "predicted_close", "signal", "total_score", "rationale"]
    cols = [c for c in cols if c in signals.columns]
    signals = signals[cols]

    signals.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {OUT_CSV} with {len(signals):,} rows")

if __name__ == "__main__":
    main()
