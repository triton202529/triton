# scripts/audit_baseline.py
import argparse, math, sys
from pathlib import Path
import pandas as pd


def hhi(weights):
    return float((weights**2).sum())


def entropy(weights):
    w = weights[weights > 0]
    return float(-(w * (w.apply(math.log))).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("data/results/baseline/weights.csv"))
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.path)
    if df.empty:
        print("[err] Empty weights file.")
        sys.exit(2)

    w = df["target_weight"].astype(float)
    h = hhi(w)  # Herfindahl-Hirschman Index
    n_eff = 1.0 / h if h > 0 else float("nan")
    ent = entropy(w)  # Shannon entropy
    n_exp = math.exp(ent) if ent == ent else float("nan")  # effective breadth

    # Groups if present
    notes = df["notes"].astype(str)
    missing_sum = w[notes.str.contains(r"\bgroup=missing\b", regex=True, na=False)].sum()
    known_sum = w[notes.str.contains(r"\bgroup=known\b", regex=True, na=False)].sum()

    topdf = df.sort_values("target_weight", ascending=False).head(args.top)[
        ["ticker", "target_weight"]
    ]
    print("[audit]")
    print(f"  sum= {w.sum():.4f} | max= {w.max():.4f} | min= {w.min():.4f}")
    print(f"  HHI= {h:.6f} | 1/HHI (eff N)= {n_eff:.2f}")
    print(f"  Entropy= {ent:.6f} | exp(H)= {n_exp:.2f}")
    if missing_sum > 0:
        print(f"  pool: known={known_sum:.4f}, missing={missing_sum:.4f}")
    print("\n[top weights]")
    for _, r in topdf.iterrows():
        print(f"  {r['ticker']:>6}  {r['target_weight']:.4f}")

    sys.exit(0)


if __name__ == "__main__":
    main()
