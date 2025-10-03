# scripts/diff_weights.py
import sys, re, glob, os, datetime as dt
from pathlib import Path
import pandas as pd

BASE = Path("data/results/baseline")


def latest_two_snapshots():
    snaps = sorted(BASE.glob("weights.*.csv"))
    return (snaps[-2], snaps[-1]) if len(snaps) >= 2 else (None, snaps[-1] if snaps else None)


def load_csv(p):
    return pd.read_csv(p) if p and p.exists() else None


def main():
    curr = BASE / "weights.csv"
    prev, last = latest_two_snapshots()
    # prefer comparing two snapshots; fall back to current vs last snapshot
    left = load_csv(prev) if prev else None
    right = load_csv(last) if last else None
    if left is None or right is None:
        left = load_csv(last)
        right = load_csv(curr)
    if left is None or right is None:
        print("[err] Not enough files to diff.")
        sys.exit(1)

    a = left[["ticker", "target_weight"]].rename(columns={"target_weight": "w_old"})
    b = right[["ticker", "target_weight"]].rename(columns={"target_weight": "w_new"})
    df = a.merge(b, on="ticker", how="outer").fillna(0.0)
    df["delta"] = df["w_new"] - df["w_old"]
    df["abs_delta"] = df["delta"].abs()

    total_l1 = df["abs_delta"].sum()
    turnover = total_l1 / 2.0
    top = df.sort_values("abs_delta", ascending=False).head(15)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = BASE / "diffs"
    outdir.mkdir(parents=True, exist_ok=True)
    outcsv = outdir / f"weights_diff.{ts}.csv"
    df.sort_values("abs_delta", ascending=False).to_csv(outcsv, index=False, float_format="%.6f")

    print("[diff]")
    print(f"  compared: {prev.name if prev else last.name}  ->  {last.name if last else curr.name}")
    print(f"  turnover (sum|Δ|/2): {turnover:.6f}")
    print("\n[top moves]")
    for _, r in top.iterrows():
        print(f"  {r['ticker']:>6}  {r['w_old']:.4f} -> {r['w_new']:.4f}  Δ={r['delta']:.4f}")

    print(f"\n[ok] Diff saved -> {outcsv}")
    sys.exit(0)


if __name__ == "__main__":
    main()
