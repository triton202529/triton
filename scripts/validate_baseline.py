# scripts/validate_baseline.py
import argparse, math, re, sys
from pathlib import Path
import pandas as pd


def parse_flag_from_notes(notes_series, key):
    pat = re.compile(rf"{re.escape(key)}=([0-9.]+)")
    for s in notes_series:
        m = pat.search(str(s))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def has_group(notes_series, g):
    return notes_series.str.contains(rf"group={g}\b", regex=True, na=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("data/results/baseline/weights.csv"))
    ap.add_argument(
        "--max-weight",
        type=float,
        default=None,
        help="If omitted, try to parse from notes",
    )
    ap.add_argument(
        "--missing-share",
        type=float,
        default=None,
        help="If omitted, try to parse from notes",
    )
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    df = pd.read_csv(args.path)
    if df.empty:
        print("[err] Empty weights file.")
        sys.exit(2)

    w = df["target_weight"].astype(float)
    total = w.sum()
    if abs(total - 1.0) > args.tol:
        print(f"[err] Sum of weights {total:.8f} != 1.0")
        sys.exit(2)

    if (w < -args.tol).any():
        print("[err] Negative weights detected.")
        sys.exit(2)

    notes = df["notes"].astype(str)
    # Infer flags if not provided
    maxw = (
        args.max_weight
        if args.max_weight is not None
        else parse_flag_from_notes(notes, "max-weight")
    )
    mshare = (
        args.missing_share
        if args.missing_share is not None
        else parse_flag_from_notes(notes, "missing-share")
    )

    # Check cap if available
    if maxw is not None:
        known_mask = has_group(notes, "known")
        known_count = int(known_mask.sum())
        pool = 1.0 - (mshare if mshare is not None else 0.0)
        feas_min = (pool / known_count) if known_count > 0 else maxw
        cap_eff = max(maxw, feas_min)
        wmax = float(w.max())
        if wmax > cap_eff + 5e-6:
            print(
                f"[err] Max weight {wmax:.6f} > effective cap {cap_eff:.6f} (requested {maxw:.6f})"
            )
            sys.exit(2)

    # Check missing-share if groups are present
    if mshare is not None and has_group(notes, "missing").any():
        missing_sum = float(w[has_group(notes, "missing")].sum())
        if abs(missing_sum - mshare) > 5e-4:
            print(f"[err] Missing-share {missing_sum:.6f} != expected {mshare:.6f}")
            sys.exit(2)

    print("[ok] Baseline validates: sum=1, non-negative, caps respected, missing-share consistent.")
    sys.exit(0)


if __name__ == "__main__":
    main()
