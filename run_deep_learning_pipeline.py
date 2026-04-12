#!/usr/bin/env python3
"""
Deep Learning Signal Fusion Pipeline for Triton (robust schema + safe fallbacks)
"""

import sys, glob, json, warnings, re, argparse, datetime as dt
from pathlib import Path
from tools.io_utils import smart_read_price_csv
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# ----------------------------- Helpers ---------------------------------

DATE_CANDIDATES = ["date", "Date", "DATE", "datetime", "Datetime", "timestamp", "Timestamp"]
CLOSE_CANDIDATES = [
    "close",
    "Close",
    "CLOSE",
    "adj_close",
    "Adj Close",
    "Adj_Close",
    "AdjClose",
    "Adj. Close",
    "Adj. close",
    "adjusted_close",
    "Adjusted Close",
]
TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}$")  # basic US-style ticker heuristic


def _normalize_ohlc_df(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    # date
    dcol = next((c for c in DATE_CANDIDATES if c in df.columns), None)
    if not dcol:
        return None
    out = (
        pd.DataFrame({"date": pd.to_datetime(df[dcol], errors="coerce")})
        .dropna()
        .reset_index(drop=True)
    )

    # close (prefer adjusted if present)
    ccol = None
    for c in CLOSE_CANDIDATES:
        if c in df.columns:
            ccol = c
            if any(k in c for k in ("Adj", "adj", "Adjusted")):
                break
    if not ccol:
        # sometimes multiindex -> ('Close','AAPL') style
        for c in df.columns:
            if isinstance(c, tuple) and str(c[0]).lower() in ("close", "adj close"):
                df["__tmp_close__"] = df[c]
                ccol = "__tmp_close__"
                break
    if not ccol:
        return None

    out["close"] = pd.to_numeric(df[ccol], errors="coerce")
    # optional volume
    vcol = None
    for cand in ("volume", "Volume", "VOLUME"):
        if cand in df.columns:
            vcol = cand
            break
    out["volume"] = pd.to_numeric(df[vcol], errors="coerce") if vcol else np.nan
    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return out if not out.empty else None


def load_universe_data(
    limit: int | None = None, as_of_today: bool = False
) -> dict[str, pd.DataFrame]:
    files = glob.glob("data/*.csv")
    if not files:
        print("⚠️ No CSV files found in data/")
        return {}

    universe, ok = {}, 0
    cutoff = pd.Timestamp(dt.date.today()) if as_of_today else None

    for fp in (files if limit is None else files[:limit]):
        # validate ticker from filename first (skip non-price files early)
        ticker = Path(fp).stem.split("_")[0]
        if not TICKER_RE.match(ticker):
            continue

        # use robust reader
        df = smart_read_price_csv(fp)
        if df is None:
            print(f"⚠️  Skipping {fp}: no usable date/close")
            continue

        norm = _normalize_ohlc_df(df)
        if norm is None:
            print(f"⚠️  Skipping {fp}: could not normalize to [date, close]")
            continue

        if cutoff is not None:
            norm = norm[norm["date"] <= cutoff].copy()

        if len(norm) < 2:
            continue

        universe[ticker] = norm
        ok += 1

    print(f"✅ Loaded {ok} tickers")
    return universe


def load_model_predictions() -> dict[str, float]:
    pred_file = Path("predictions/latest_predictions.csv")
    if not pred_file.exists():
        print("⚠️ No predictions file found")
        return {}
    try:
        df = pd.read_csv(pred_file)
        preds: dict[str, float] = {}
        if {"ticker", "predicted_close", "close"}.issubset(df.columns):
            for _, r in df.iterrows():
                pc, c = r.get("predicted_close"), r.get("close")
                if pd.notna(pc) and pd.notna(c) and float(c) != 0:
                    preds[str(r["ticker"])] = (float(pc) - float(c)) / float(c)
        print(f"✅ Loaded {len(preds)} predictions")
        return preds
    except Exception as e:
        print(f"⚠️ Failed to load predictions: {e}")
        return {}


def load_sentiment_data() -> dict[str, float]:
    sentiment_file = Path("predictions/signals.csv")
    if not sentiment_file.exists():
        return {}
    try:
        df = pd.read_csv(sentiment_file)
        sent: dict[str, float] = {}
        if "ticker" in df.columns:
            for _, row in df.iterrows():
                sig = (row.get("signal") or "").upper()
                sent[row["ticker"]] = 0.7 if sig == "BUY" else (-0.3 if sig == "SELL" else 0.0)
        print(f"✅ Loaded sentiment for {len(sent)} tickers")
        return sent
    except Exception as e:
        print(f"⚠️ Failed to load sentiment: {e}")
        return {}


# ----------------------------- Main ------------------------------------


def _generate_baseline_signals(universe_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tkr, df in universe_data.items():
        if len(df) < 21:
            continue
        ret20 = (df["close"].iloc[-1] / df["close"].iloc[-21]) - 1.0
        sig = "BUY" if ret20 > 0 else "HOLD"
        rows.append({"ticker": tkr, "signal": sig, "confidence": float(abs(ret20))})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> bool:
    parser = argparse.ArgumentParser(description="Deep Learning Signal Fusion Pipeline")
    parser.add_argument("--as-of-today", action="store_true", help="Trim rows after today's date")
    parser.add_argument("--limit", type=int, default=None, help="Max number of CSVs to read")
    args = parser.parse_args(argv)

    print("🚀 Deep Learning Signal Fusion Pipeline")
    print("=" * 70)

    # Step 1: Data
    print("\n📥 Step 1: Loading data...")
    universe_data = load_universe_data(limit=args.limit, as_of_today=args.as_of_today)
    if not universe_data:
        print("❌ No data available - cannot proceed")
        return False

    # Step 2: Predictions + Sentiment
    print("\n🔮 Step 2: Loading model predictions and sentiment...")
    model_predictions = load_model_predictions()
    sentiment_data = load_sentiment_data()

    # Step 3: Enhanced Signal Generator (safe import)
    print("\n🧠 Step 3: Initializing Enhanced Signal Generator...")
    generator = None
    fusion_available = False
    try:
        from services.enhanced_signal_generator import EnhancedSignalGenerator

        generator = EnhancedSignalGenerator(use_fusion=True, use_adaptive_risk=True, verbose=True)
        fusion_available = True
    except Exception as e:
        print(f"⚠️ Fusion path unavailable ({e}). Falling back to baseline momentum signals.")

    # Step 4: Generate signals
    print("\n🎯 Step 4: Generating enhanced signals...")
    try:
        if fusion_available and generator is not None:
            signals_df = generator.generate_signals(
                universe_data=universe_data,
                model_predictions=model_predictions,
                sentiment_data=sentiment_data,
            )
        else:
            signals_df = _generate_baseline_signals(universe_data)

        if signals_df.empty:
            print("⚠️ No signals generated")
            return False

        print(f"✅ Generated {len(signals_df)} signals")
    except Exception as e:
        print(f"❌ Failed to generate signals: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 5: Adaptive risk (best-effort)
    print("\n⚡ Step 5: Applying adaptive risk adjustments...")
    try:
        if fusion_available and generator is not None:
            risk_adjusted_df = generator.apply_adaptive_risk(signals_df, universe_data)
        else:
            risk_adjusted_df = signals_df
    except Exception as e:
        print(f"⚠️ Risk adjustment failed: {e}")
        risk_adjusted_df = signals_df

    # Step 6: Save
    print("\n💾 Step 6: Saving results...")
    try:
        outdir = Path("predictions/enhanced")
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / "fused_signals.csv"
        risk_adjusted_df.to_csv(outfile, index=False)
        print(f"✅ Saved to {outfile}")
        if fusion_available and generator is not None:
            generator.save_config(outdir / "config.json")
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")

    # Step 7: Sample
    print("\n📋 Sample signals (first 10):")
    if not risk_adjusted_df.empty:
        print(risk_adjusted_df.head(10).to_string())

    print("\n" + "=" * 70)
    print("✅ Deep Learning Signal Fusion Pipeline Completed!")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
