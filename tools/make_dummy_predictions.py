#!/usr/bin/env python3
"""
Builds predictions/latest_predictions.csv from today's fused_signals
- close is taken from your latest price file per ticker (prefers *.fixed.csv)
- predicted_close is close * (1 + view), where view is mapped from the signal
"""

from pathlib import Path
import argparse, glob, sys
import pandas as pd
import numpy as np

# Optional helper if present in your repo
try:
    from tools.io_utils import smart_read_price_csv
except Exception:
    smart_read_price_csv = None

DATE_CANDS = ["date", "Date", "DATE", "datetime", "Datetime", "timestamp", "Timestamp"]
CLOSE_CANDS = [
    "adj_close",
    "Adj Close",
    "Adj_Close",
    "AdjClose",
    "Adj. Close",
    "Adj. close",
    "adjusted_close",
    "Adjusted Close",
    "close",
    "Close",
    "CLOSE",
]


def norm_df(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    dcol = next((c for c in DATE_CANDS if c in df.columns), None)
    if not dcol:
        if df.index.name and str(df.index.name).lower() in ("date", "datetime", "timestamp"):
            df = df.reset_index().rename(columns={df.index.name: "date"})
            dcol = "date"
        else:
            return None
    out = pd.DataFrame({"date": pd.to_datetime(df[dcol], errors="coerce")})
    ccol = next((c for c in CLOSE_CANDS if c in df.columns), None)
    if not ccol:
        # try multiindex columns ('Adj Close','AAPL') or ('Close','AAPL')
        for c in df.columns:
            if isinstance(c, tuple) and str(c[0]).lower() in ("adj close", "close"):
                df["__tmp_close__"] = df[c]
                ccol = "__tmp_close__"
                break
        if not ccol:
            return None
    out["close"] = pd.to_numeric(df[ccol], errors="coerce")
    out = out.dropna().sort_values("date")
    return out if not out.empty else None


def last_close_for(ticker: str) -> float | None:
    data_dir = Path("data")
    # prefer fixed/normalized.fixed variants, then fall back to any CSV
    patterns = [f"{ticker}_*normalized.fixed.csv", f"{ticker}_*.fixed.csv", f"{ticker}_*.csv"]
    for pat in patterns:
        files = sorted(data_dir.glob(pat))
        for f in reversed(files):  # prefer most recent filename
            try:
                if smart_read_price_csv:
                    df = smart_read_price_csv(f)
                else:
                    df = pd.read_csv(f)
                df = norm_df(df)
                if df is not None and not df.empty:
                    return float(df["close"].iloc[-1])
            except Exception:
                continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--signals",
        default="predictions/enhanced/fused_signals.csv",
        help="Input signals file (ticker,signal,confidence)",
    )
    ap.add_argument(
        "--outfile",
        default="predictions/latest_predictions.csv",
        help="Where to write the predictions CSV",
    )
    ap.add_argument(
        "--buy-bps",
        type=float,
        default=150.0,
        help="View for BUY in basis points (default 150 = +1.5%)",
    )
    ap.add_argument(
        "--sell-bps",
        type=float,
        default=-150.0,
        help="View for SELL in basis points (default -150 = -1.5%)",
    )
    ap.add_argument(
        "--hold-bps",
        type=float,
        default=0.0,
        help="View for HOLD in basis points (default 0 = flat)",
    )
    args = ap.parse_args()

    sig_path = Path(args.signals)
    if not sig_path.exists():
        print(f"❌ Signals file not found: {sig_path}")
        return 2

    df = pd.read_csv(sig_path)
    if "ticker" not in df.columns or "signal" not in df.columns:
        print("❌ Signals file must have 'ticker' and 'signal' columns")
        return 2

    view_map = {"BUY": args.buy_bps / 1e4, "SELL": args.sell_bps / 1e4, "HOLD": args.hold_bps / 1e4}

    rows = []
    missing = []
    for _, r in df.iterrows():
        tkr = str(r["ticker"]).upper().strip()
        sig = str(r["signal"]).upper().strip()
        view = view_map.get(sig, 0.0)
        px = last_close_for(tkr)
        if px is None:
            missing.append(tkr)
            continue
        pred = float(px) * (1.0 + float(view))
        rows.append({"ticker": tkr, "close": float(px), "predicted_close": float(pred)})

    if not rows:
        print("❌ No predictions generated (no closes found)")
        if missing:
            print("   Missing price files for:", ", ".join(sorted(set(missing))))
        return 2

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print(f"✅ Wrote {out}  (tickers={len(rows)})")
    if missing:
        print(f"⚠️ Skipped (no close found): {', '.join(sorted(set(missing)))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
