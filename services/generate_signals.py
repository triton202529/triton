# services/generate_signals.py

import os
import glob
import pandas as pd

print("⚙️ Generating signals with rationale...")

PREDICTIONS_DIR = "data/predictions"
RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_WITH_RATIONALE = os.path.join(RESULTS_DIR, "signals_with_rationale.csv")
OUT_SIGNALS = os.path.join(RESULTS_DIR, "signals.csv")  # backward compatibility

# Signal thresholds (as pct moves vs close)
BUY_DELTA = 0.002   # +0.20%
SELL_DELTA = -0.002 # -0.20%

def load_pred_file(path: str) -> pd.DataFrame:
    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as e:
        print(f"🔥 Error reading {path}: {e}")
        return pd.DataFrame()

def normalize_date(s: pd.Series) -> pd.Series:
    # Convert any mix of tz-aware/naive to UTC then drop tz (naive)
    s = pd.to_datetime(s, errors="coerce", utc=True)
    # s is datetime64[ns, UTC]; drop tz so everything is naive & sortable
    return s.dt.tz_convert("UTC").dt.tz_localize(None)

def build_rationale(delta_pct: float, buy_thr: float, sell_thr: float) -> str:
    pct = f"{delta_pct*100:.2f}%"
    if delta_pct >= buy_thr:
        return f"Predicted ↑ {pct} vs close (>{buy_thr*100:.2f}%). Upside expected; BUY bias."
    if delta_pct <= sell_thr:
        return f"Predicted ↓ {pct} vs close (<{sell_thr*100:.2f}%). Downside risk; SELL bias."
    return f"Predicted {pct} vs close within band; momentum unclear; HOLD."

all_rows = []

pred_files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*_predictions.parquet"))) \
           + sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*_predictions.csv")))

if not pred_files:
    print(f"🚫 No predictions found in {PREDICTIONS_DIR}. Run train_model.py first.")
    raise SystemExit(0)

for path in pred_files:
    ticker = os.path.basename(path).split("_")[0].upper()
    df = load_pred_file(path)
    if df.empty:
        print(f"⚠️ {ticker}: empty predictions, skipping.")
        continue

    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})

    required = {"date", "close", "predicted_close"}
    if not required.issubset(df.columns):
        print(f"⚠️ {ticker}: missing columns {sorted(required - set(df.columns))}, skipping.")
        continue

    # Normalize dates (fix tz-aware/naive mix) and basic types
    df["date"] = normalize_date(df["date"])
    df = df.dropna(subset=["date", "close", "predicted_close"])
    if df.empty:
        print(f"⚠️ {ticker}: predictions have no valid dates after parsing, skipping.")
        continue

    df = df.sort_values("date").copy()
    df["delta_pct"] = (pd.to_numeric(df["predicted_close"], errors="coerce") /
                       pd.to_numeric(df["close"], errors="coerce")) - 1.0

    def decide(delta):
        if delta >= BUY_DELTA:
            return "BUY"
        if delta <= SELL_DELTA:
            return "SELL"
        return "HOLD"

    df["signal"] = df["delta_pct"].apply(decide)
    df["confidence"] = df["delta_pct"].abs().round(4)
    df["rationale"] = df["delta_pct"].apply(lambda d: build_rationale(d, BUY_DELTA, SELL_DELTA))
    df["ticker"] = str(ticker)

    all_rows.append(df[["date", "ticker", "close", "predicted_close", "delta_pct",
                        "signal", "confidence", "rationale"]])

if not all_rows:
    print("🚫 No signals generated (no valid prediction files).")
    raise SystemExit(0)

signals = pd.concat(all_rows, ignore_index=True)

# Ensure sortable dtypes
signals["ticker"] = signals["ticker"].astype(str)
signals["date"] = pd.to_datetime(signals["date"], errors="coerce")  # naive
signals = signals.sort_values(["ticker", "date"], kind="mergesort")

signals.to_csv(OUT_WITH_RATIONALE, index=False)

# Back-compat: drop rationale, rename confidence
signals_no_rat = signals.drop(columns=["rationale"]).rename(columns={"confidence": "confidence_score"})
signals_no_rat.to_csv(OUT_SIGNALS, index=False)

print(f"✅ signals_with_rationale.csv → {OUT_WITH_RATIONALE}")
print(f"✅ signals.csv               → {OUT_SIGNALS}")

print("\n📊 Signal counts (overall):")
print(signals_no_rat["signal"].value_counts())
print("\n📈 Signal counts by ticker:")
print(signals_no_rat.groupby("ticker")["signal"].value_counts().unstack(fill_value=0))
