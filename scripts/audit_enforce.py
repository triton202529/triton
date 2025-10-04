#!/usr/bin/env python3
"""
Audit + Enforce baseline constraints.

Reads a weights CSV (expects a 'target_weight' column), prints metrics,
and exits non-zero if any thresholds are violated.

Example:
  python scripts/audit_enforce.py \
    --path data/results/baseline/weights.csv \
    --max-weight 0.1251 --min-weight 0.0005 \
    --max-hhi 0.1005 --min-entropy 2.45 --min-effn 10.0
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(x: float, n: int = 6) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{n}f}"


def compute_metrics(weights: pd.Series) -> dict:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    total = float(w.sum())
    # Avoid division by zero in info metrics; metrics are computed on the actual vector
    hhi = float((w**2).sum()) if len(w) else float("nan")
    eff_n = (1.0 / hhi) if (hhi > 0) else float("nan")
    # Shannon entropy (natural log); define 0 * log(0) := 0 via mask
    mask = w > 0
    ent = float(-(w[mask] * np.log(w[mask])).sum()) if mask.any() else float("nan")
    exp_h = float(np.exp(ent)) if not math.isnan(ent) else float("nan")
    mx = float(w.max()) if len(w) else float("nan")
    mn = float(w.min()) if len(w) else float("nan")
    return {
        "sum": total,
        "hhi": hhi,
        "eff_n": eff_n,
        "entropy": ent,
        "exp_h": exp_h,
        "max": mx,
        "min": mn,
        "count": int(len(w)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit + enforce baseline constraints.")
    ap.add_argument("--path", required=True, type=Path, help="Path to weights CSV")
    ap.add_argument("--sum-tol", type=float, default=5e-5, help="Allowed deviation from 1.0")
    ap.add_argument("--max-weight", type=float, default=None, help="Upper bound for any name")
    ap.add_argument("--min-weight", type=float, default=None, help="Lower bound for any name")
    ap.add_argument("--max-hhi", type=float, default=None, help="Upper bound on HHI")
    ap.add_argument(
        "--min-entropy", type=float, default=None, help="Lower bound on Shannon entropy (nats)"
    )
    ap.add_argument(
        "--min-effn", type=float, default=None, help="Lower bound on effective N (=1/HHI)"
    )
    args = ap.parse_args()

    if not args.path.exists():
        print(f"[error] Weights file not found: {args.path}", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(args.path)
    except Exception as e:
        print(f"[error] Failed to read CSV: {e}", file=sys.stderr)
        return 2

    # Find the weight column
    candidates = ["target_weight", "weight", "Weight", "targetWeight"]
    c_weight = next((c for c in candidates if c in df.columns), None)
    if not c_weight:
        print(f"[error] No weight column found (tried: {', '.join(candidates)}).", file=sys.stderr)
        return 2

    metrics = compute_metrics(df[c_weight])

    print("[audit]")
    print(
        f"  sum= {_fmt(metrics['sum'], 4)} | max= {_fmt(metrics['max'], 4)} | min= {_fmt(metrics['min'], 4)}"
    )
    print(f"  HHI= {_fmt(metrics['hhi'], 6)} | 1/HHI (eff N)= {_fmt(metrics['eff_n'], 2)}")
    print(f"  Entropy= {_fmt(metrics['entropy'], 6)} | exp(H)= {_fmt(metrics['exp_h'], 2)}")

    violations = []

    if abs(metrics["sum"] - 1.0) > args.sum_tol:
        violations.append(f"sum != 1 (sum={metrics['sum']:.6f}, tol={args.sum_tol:g})")

    if args.max_weight is not None and metrics["max"] - args.max_weight > 1e-12:
        violations.append(f"max_weight {metrics['max']:.6f} > {args.max_weight:g}")

    if args.min_weight is not None and (args.min_weight - metrics["min"]) > 1e-12:
        violations.append(f"min_weight {metrics['min']:.6f} < {args.min_weight:g}")

    if args.max_hhi is not None and metrics["hhi"] - args.max_hhi > 1e-12:
        violations.append(f"HHI {metrics['hhi']:.6f} > {args.max_hhi:g}")

    if args.min_entropy is not None and (args.min_entropy - metrics["entropy"]) > 1e-12:
        violations.append(f"entropy {metrics['entropy']:.6f} < {args.min_entropy:g}")

    if args.min_effn is not None and (args.min_effn - metrics["eff_n"]) > 1e-12:
        violations.append(f"effN {metrics['eff_n']:.4f} < {args.min_effn:g}")

    if violations:
        print("\n[violations]")
        for v in violations:
            print(f"  - {v}")
        return 2

    print("\n[ok] All audit checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
