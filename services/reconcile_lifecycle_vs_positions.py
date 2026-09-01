"""
reconcile_lifecycle_vs_positions.py
-------------------------------------
Read-only compare: signal lifecycle STATE vs broker positions snapshot.

Inputs:
  - data/results/signal_lifecycle.csv  (RAW_SIGNAL_LIFECYCLE — advisory only)
  - data/results/positions_snapshot.csv

Output:
  - data/results/lifecycle_reconciliation.csv

Authority note:
  This module reads raw lifecycle for diagnostic reconciliation only.
  Operational lifecycle authority is signal_lifecycle_effective.csv
  (EFFECTIVE_LIFECYCLE_AUTHORITY).

No broker API calls; does not modify lifecycle CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from services.schema_guard import safe_merge
from services.authority_labels import lifecycle_raw_metadata
from services.symbol_identity import (
    COLLISION_REASON,
    SymbolIdentityCollision,
    collect_identity_collisions,
    identity_key,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_LIFECYCLE = RESULTS_DIR / "signal_lifecycle.csv"
DEFAULT_POSITIONS = RESULTS_DIR / "positions_snapshot.csv"
DEFAULT_OUT = RESULTS_DIR / "lifecycle_reconciliation.csv"


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _ticker_series(df: pd.DataFrame) -> pd.Series:
    if "ticker" in df.columns:
        s = df["ticker"]
    elif "symbol" in df.columns:
        s = df["symbol"]
    else:
        raise ValueError("positions snapshot needs ticker or symbol column")
    return s.astype(str).str.strip().str.upper()


def load_lifecycle(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing lifecycle file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return _norm_cols(df)
    df = _norm_cols(df)
    if "ticker" not in df.columns:
        raise ValueError("signal_lifecycle.csv must have ticker")
    df = df.copy()
    raw_tickers = df["ticker"].astype(str).tolist()
    collisions = collect_identity_collisions(raw_tickers)
    if collisions:
        raise SymbolIdentityCollision(collisions, source="lifecycle")
    df["ticker"] = df["ticker"].map(identity_key)
    df = df[df["ticker"].astype(str).str.len() > 0]
    # latest row per identity if duplicates of the same raw form
    df = df.drop_duplicates(subset=["ticker"], keep="last")
    return df


def load_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(
            f"[WARN] Positions file missing; treating as no broker positions: {path}",
            file=sys.stderr,
        )
        return pd.DataFrame(columns=["ticker", "broker_qty", "broker_market_value"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "broker_qty", "broker_market_value"])
    df = _norm_cols(df)
    df = df.copy()
    raw_tickers = _ticker_series(df)
    collisions = collect_identity_collisions(raw_tickers.tolist())
    if collisions:
        raise SymbolIdentityCollision(collisions, source="positions")
    df["ticker"] = raw_tickers.map(lambda x: identity_key(x))
    q = pd.to_numeric(df.get("qty", np.nan), errors="coerce").fillna(0.0)
    df["qty"] = q
    mv_col = None
    for c in ("market_value", "value", "marketvalue"):
        if c in df.columns:
            mv_col = c
            break
    if mv_col:
        mv = pd.to_numeric(df[mv_col], errors="coerce").fillna(0.0)
    else:
        mv = pd.Series(0.0, index=df.index)
    df["market_value"] = mv
    agg = df.groupby("ticker", as_index=False).agg(
        broker_qty=("qty", "sum"), broker_market_value=("market_value", "sum")
    )
    return agg


def reconcile_simple(lc: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """Outer merge on ticker; broker truth for reconciled_state."""
    if lc.empty:
        lc = pd.DataFrame(
            columns=["ticker", "lifecycle_stance", "lifecycle_action", "lifecycle_position_state"]
        )
    else:
        base = lc[["ticker"]].copy()
        for src, dst in (
            ("stance", "lifecycle_stance"),
            ("lifecycle_action", "lifecycle_action"),
            ("position_state", "lifecycle_position_state"),
        ):
            if src in lc.columns:
                base[dst] = lc[src].fillna("").astype(str).str.strip()
            else:
                base[dst] = ""
        lc = base

    if pos.empty:
        pos = pd.DataFrame(columns=["ticker", "broker_qty", "broker_market_value"])

    m = safe_merge(lc, pos, on="ticker", how="outer", label="lifecycle_vs_positions")
    m = m.dropna(subset=["ticker"])
    m["ticker"] = m["ticker"].map(identity_key)
    m["lifecycle_stance"] = m["lifecycle_stance"].fillna("").map(lambda x: str(x).strip())
    m["lifecycle_action"] = m["lifecycle_action"].fillna("").map(lambda x: str(x).strip())
    m["lifecycle_position_state"] = (
        m["lifecycle_position_state"].fillna("").map(lambda x: str(x).strip().upper())
    )
    m["broker_qty"] = pd.to_numeric(m["broker_qty"], errors="coerce").fillna(0.0)
    m["broker_market_value"] = pd.to_numeric(m["broker_market_value"], errors="coerce").fillna(0.0)

    m["broker_has_position"] = m["broker_qty"] > 0
    m["reconciled_state"] = np.where(m["broker_has_position"], "LONG", "FLAT")

    def _reason(row: pd.Series) -> tuple[bool, str]:
        lp = row["lifecycle_position_state"]
        bh = bool(row["broker_has_position"])
        if lp == "LONG" and not bh:
            return True, "lifecycle_long_broker_flat"
        if lp == "FLAT" and bh:
            return True, "lifecycle_flat_broker_long"
        return False, ""

    reasons = m.apply(_reason, axis=1, result_type="expand")
    m["mismatch"] = reasons[0].astype(bool)
    m["mismatch_reason"] = reasons[1].astype(str)

    out = m[
        [
            "ticker",
            "lifecycle_stance",
            "lifecycle_action",
            "lifecycle_position_state",
            "broker_qty",
            "broker_market_value",
            "broker_has_position",
            "reconciled_state",
            "mismatch",
            "mismatch_reason",
        ]
    ]
    return out.sort_values("ticker").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Reconcile signal_lifecycle.csv (RAW_SIGNAL_LIFECYCLE — advisory only) "
            "vs positions_snapshot.csv"
        )
    )
    ap.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    ap.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    try:
        lc = load_lifecycle(args.lifecycle)
        pos = load_positions(args.positions)
    except SymbolIdentityCollision as e:
        print(f"[ERROR] {COLLISION_REASON} fail_closed=True {e}", file=sys.stderr)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        ev_path = args.out.with_name("lifecycle_reconciliation_identity_collision.json")
        ev_path.write_text(json.dumps(e.to_dict(), indent=2), encoding="utf-8")
        return 3
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    out_df = reconcile_simple(lc, pos)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    meta_path = args.out.with_name("lifecycle_reconciliation_authority.json")
    meta_path.write_text(
        json.dumps(
            {
                "lifecycle_source": lifecycle_raw_metadata(),
                "note": (
                    "Reconciliation uses RAW_SIGNAL_LIFECYCLE for diagnostics. "
                    "EFFECTIVE_LIFECYCLE_AUTHORITY is signal_lifecycle_effective.csv."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    n = len(out_df)
    n_mis = int(out_df["mismatch"].sum()) if n else 0
    print(f"[reconcile_lifecycle_vs_positions] wrote {args.out}")
    print(f"[reconcile_lifecycle_vs_positions] total_rows={n} mismatch_count={n_mis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
