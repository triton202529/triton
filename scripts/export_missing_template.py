# scripts/export_missing_template.py
from pathlib import Path
import datetime as dt
import pandas as pd

FUND_HINT = {"BITO", "GBTC", "GLD", "ARKK", "DIA", "DBA"}  # tweak if needed


def main():
    weights = Path("data/results/baseline/weights.csv")
    if not weights.exists():
        raise SystemExit(f"[err] Not found: {weights}")
    df = pd.read_csv(weights)
    notes = df["notes"].astype(str)
    missing = df[notes.str.contains(r"\bgroup=missing\b", regex=True, na=False)].copy()

    tickers = sorted(set(missing["ticker"].astype(str).str.upper()))
    if not tickers:
        print("[ok] No missing tickers in current baseline.")
        return

    out = pd.DataFrame(
        {
            "ticker": tickers,
            # Fill ONE of these for companies:
            "market_cap": ["" for _ in tickers],  # e.g. 350B, 1.2T, or raw number
            "shares_outstanding": ["" for _ in tickers],  # optional if market_cap provided
            # Optionally give a price to compute cap if shares provided:
            "price": ["" for _ in tickers],  # optional; builder also uses universe price
            # If it’s a fund/ETF/trust, fill AUM instead of market_cap:
            "totalAssets": ["" for _ in tickers],  # a.k.a. AUM / netAssets / totalNetAssets
            "is_fund_hint": [t in FUND_HINT for t in tickers],
            "notes": ["" for _ in tickers],
        }
    )

    outdir = Path("data/results/baseline")
    outdir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = outdir / f"fundamentals_fill.{ts}.csv"
    out.to_csv(path, index=False)
    print(f"[ok] Fill template written -> {path}")


if __name__ == "__main__":
    main()
