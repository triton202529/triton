# make_backtest_summary.py
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR  = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def max_drawdown(cum):
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan

def main():
    sig_path = RESULTS_DIR / "signals_with_rationale.csv"
    if not sig_path.exists() or sig_path.stat().st_size == 0:
        raise SystemExit("signals_with_rationale.csv not found or empty")

    df = pd.read_csv(sig_path)
    if "ticker" not in df.columns or "close" not in df.columns:
        raise SystemExit("signals_with_rationale.csv needs 'ticker' and 'close'")

    # Dates
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["ticker"]).sort_values(["ticker", "date"])
    df["signal"] = df.get("signal", "").astype(str).str.upper()

    rows = []
    svm_frames = []
    for t, g in df.groupby("ticker", sort=True):
        g = g.dropna(subset=["date"]).sort_values("date").copy()
        if g.empty:
            continue
        price = pd.to_numeric(g["close"], errors="coerce")
        ret   = price.pct_change().fillna(0.0)

        # Position follows yesterday's signal: BUY=+1, SELL=-1, HOLD=0
        pos = g["signal"].map({"BUY": 1.0, "SELL": -1.0}).fillna(0.0).shift(1).fillna(0.0)
        strat_ret = pos * ret

        cum_market   = (1.0 + ret).cumprod()
        cum_strategy = (1.0 + strat_ret).cumprod()

        svm_frames.append(pd.DataFrame({
            "ticker": t,
            "date": g["date"],
            "cumulative_strategy": cum_strategy.values,
            "cumulative_market":   cum_market.values,
        }))

        n = len(g)
        if n == 0:
            continue
        total = float(cum_strategy.iloc[-1] - 1.0)
        # Annualization on trading days
        ann_factor = np.sqrt(252.0)
        vol = float(np.std(strat_ret, ddof=1)) * ann_factor if n > 1 else np.nan
        # Simple CAGR approximation: final_cum^(252/n) - 1
        cagr = float(cum_strategy.iloc[-1] ** (252.0 / max(1.0, n)) - 1.0) if n > 1 else np.nan
        sharpe = float(cagr / vol) if vol and np.isfinite(vol) and vol != 0 else np.nan
        mdd = max_drawdown(cum_strategy)

        rows.append({
            "ticker": t,
            "start_date": g["date"].iloc[0].date(),
            "end_date":   g["date"].iloc[-1].date(),
            "n_days": n,
            "total_return": total,
            "cagr": cagr,
            "volatility": vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "trades": int((g["signal"].isin(["BUY","SELL"])).sum()),
        })

    # Write strategy_vs_market
    if svm_frames:
        svm = pd.concat(svm_frames, ignore_index=True)
        svm.to_csv(RESULTS_DIR / "strategy_vs_market.csv", index=False)

    # Write backtest_summary
    out = pd.DataFrame(rows).sort_values("total_return", ascending=False)
    out.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False)
    print(f"Wrote {len(out):,} rows -> {RESULTS_DIR / 'backtest_summary.csv'}")
    if svm_frames:
        print(f"Wrote {len(svm):,} rows -> {RESULTS_DIR / 'strategy_vs_market.csv'}")

if __name__ == "__main__":
    main()
