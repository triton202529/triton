"""
build_effective_lifecycle.py
----------------------------
Overlay broker reconciliation truth onto signal_lifecycle rows.

Reads:
  - data/results/signal_lifecycle.csv  (unchanged on disk)
  - data/results/lifecycle_reconciliation.csv
  - data/results/open_orders_snapshot.csv (optional; in-flight BUY preserves effective BUY when broker flat)

Writes:
  - data/results/signal_lifecycle_effective.csv

Read-only with respect to signal_lifecycle.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional, Set

import pandas as pd

from services.signal_lifecycle import (
    LifecycleLogicConfig,
    lifecycle_logic_from_dict,
    long_buy_qualifies_for_add,
    long_qualifies_for_exit_delta,
    long_qualifies_for_trim,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_LIFECYCLE = RESULTS_DIR / "signal_lifecycle.csv"
DEFAULT_RECON = RESULTS_DIR / "lifecycle_reconciliation.csv"
DEFAULT_OUT = RESULTS_DIR / "signal_lifecycle_effective.csv"
DEFAULT_OPEN_ORDERS = RESULTS_DIR / "open_orders_snapshot.csv"

# Status values meaning the order is still working (not fully filled / terminal).
_TERMINAL_ORDER_STATUSES = frozenset(
    {
        "filled",
        "done",
        "canceled",
        "cancelled",
        "expired",
        "failed",
        "replaced",
        "rejected",
        "closed",
    }
)


def load_lifecycle_config() -> dict:
    """Load config/lifecycle_logic.json merged onto defaults (project-root path)."""
    defaults = {
        "enabled": True,
        "add_confidence_min": 0.60,
        "add_delta_pct_min": 0.008,
        "hold_delta_floor": -0.002,
        "hold_delta_ceiling": 0.006,
        "trim_delta_pct_threshold": -0.001,
        "exit_delta_pct_threshold": -0.004,
        "exit_confidence_min": 0.55,
    }
    path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "..", "config", "lifecycle_logic.json"
    )
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        if isinstance(user_cfg, dict):
            defaults.update(user_cfg)
    except Exception as e:
        print(f"[CONFIG] Failed to load lifecycle config: {e}")
    return defaults


def _norm_symbol(s: str) -> str:
    """Align tickers across lifecycle vs broker snapshot (e.g. BRK.B vs BRK-B)."""
    return str(s or "").strip().upper().replace(".", "-")


def _delta_pct_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (float, int)) and pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def load_reconciliation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing reconciliation file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "mismatch_reason"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("lifecycle_reconciliation.csv must have ticker")
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    if "mismatch_reason" not in df.columns:
        df["mismatch_reason"] = ""
    df = df[["ticker", "mismatch_reason"]].drop_duplicates(subset=["ticker"], keep="last")
    return df


def load_open_buy_tickers(path: Path) -> Set[str]:
    """
    Symbols with an in-flight BUY (open/accepted/pending/etc., not fully terminal).
    Uses same snapshot as snapshot_live_orders / manage_positions.
    """
    out: Set[str] = set()
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return out
        df = pd.read_csv(path)
    except Exception:
        return out
    if df is None or df.empty:
        return out
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    sym_col = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
    if not sym_col:
        return out
    side_col = "side" if "side" in df.columns else None
    st_col = "status" if "status" in df.columns else None
    fq_col = "filled_qty" if "filled_qty" in df.columns else None
    qcol = "qty" if "qty" in df.columns else None

    for _, row in df.iterrows():
        side = str(row.get(side_col) or "").strip().lower() if side_col else ""
        if "buy" not in side:
            continue
        st = str(row.get(st_col) or "").strip().lower() if st_col else ""
        if st in _TERMINAL_ORDER_STATUSES:
            continue
        if fq_col and qcol:
            try:
                fq = float(row.get(fq_col) or 0)
                qq = float(row.get(qcol) or 0)
                if qq > 0 and fq >= qq:
                    continue
            except Exception:
                pass
        sym = _norm_symbol(str(row.get(sym_col) or ""))
        if sym:
            out.add(sym)
    return out


def load_lifecycle(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing lifecycle file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "ticker" not in df.columns:
        raise ValueError("signal_lifecycle.csv must have ticker")
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["ticker"], keep="last")
    return df


def apply_heal(
    stance: str,
    position_state: str,
    mismatch_reason: str,
    *,
    ticker: str = "",
    open_buy_tickers: Optional[Set[str]] = None,
) -> tuple[str, str, bool, str, str, bool]:
    """
    Broker vs persisted lifecycle reconciliation only.

    Returns:
      effective_position, effective_stance, healed, heal_reason,
      lifecycle_authoritative_source, reconciled_with_broker

    `healed` is True only for rare broker/persisted mismatch resolution (not stance normalization).
    Open-BUY in-flight keeps BUY without counting as a broker "heal" for metrics.
    """
    stance_u = str(stance or "").strip().upper()
    reason = str(mismatch_reason or "").strip()
    sym_u = _norm_symbol(str(ticker or ""))
    ob = open_buy_tickers or set()

    if reason == "lifecycle_long_broker_flat":
        ep = "FLAT"
        if stance_u == "BUY" and sym_u and sym_u in ob:
            return ep, "BUY", False, "", "open_buy_in_flight", False
        if stance_u in ("HOLD", "ADD", "BUY", "EXIT", "TRIM"):
            es = "WAIT"
        else:
            es = stance_u
        return ep, es, True, "broker_flat_overrides_lifecycle_long", "broker_reconciled", True

    if reason == "lifecycle_flat_broker_long":
        ep = "LONG"
        if stance_u == "WAIT":
            es = "HOLD"
        else:
            es = stance_u
        return ep, es, True, "broker_long_overrides_lifecycle_flat", "broker_reconciled", True

    ps = str(position_state or "").strip()
    st = str(stance or "").strip()
    return ps, st, False, "", "persisted_lifecycle", False


def _deterministic_effective_stance(
    effective_position_state: str,
    effective_stance: str,
    *,
    delta_pct: Any = None,
    confidence: Any = None,
    logic: Optional[LifecycleLogicConfig] = None,
) -> tuple[str, str, str]:
    """
    Deterministic stance from persisted signal + thresholds (not broker healing).

    Returns (effective_position, effective_stance, stance_adjustment_tag_or_empty).

    - FLAT cannot EXIT/TRIM → WAIT.
    - LONG+BUY should not appear from apply_lifecycle; if present, map to EXIT/TRIM/ADD/HOLD.
    """
    _logic = logic or lifecycle_logic_from_dict(load_lifecycle_config())
    ep = str(effective_position_state or "").strip()
    ep_u = ep.upper()
    es_u = str(effective_stance or "").strip().upper()
    st_out = str(effective_stance or "").strip()

    if ep_u == "FLAT" and es_u in ("EXIT", "TRIM"):
        return ep, "WAIT", "effective_flat_suppresses_exit_trim"

    if ep_u == "LONG" and es_u == "ADD":
        return ep, st_out, ""

    if ep_u == "LONG" and es_u == "BUY":
        if long_qualifies_for_exit_delta(delta_pct, logic=_logic):
            return ep, "EXIT", "effective_long_buy_resolved_to_exit_delta"
        if long_qualifies_for_trim(delta_pct, logic=_logic):
            return ep, "TRIM", "effective_long_buy_resolved_to_trim_delta"
        if long_buy_qualifies_for_add(delta_pct, confidence, logic=_logic):
            return ep, "ADD", "effective_long_buy_resolved_to_add_thresholds"
        return ep, "HOLD", "effective_long_buy_resolved_to_hold_weak"

    return ep, st_out, ""


def build_effective(
    lc: pd.DataFrame,
    recon: pd.DataFrame,
    *,
    open_orders_path: Path = DEFAULT_OPEN_ORDERS,
    lifecycle_logic: Any = None,
) -> pd.DataFrame:
    merged = lc.merge(recon, on="ticker", how="left")
    merged["mismatch_reason"] = merged["mismatch_reason"].fillna("").astype(str)

    open_buy_tickers = load_open_buy_tickers(open_orders_path)
    logic = lifecycle_logic or lifecycle_logic_from_dict(load_lifecycle_config())

    eps: list[str] = []
    es: list[str] = []
    healed: list[bool] = []
    reasons: list[str] = []
    inflight: list[bool] = []
    auth_src: list[str] = []
    recon_broker: list[bool] = []
    stance_adj: list[str] = []
    lc_consistency: list[str] = []

    for _, row in merged.iterrows():
        ep, est, h, hr, asrc, rbrok = apply_heal(
            str(row.get("stance", "")),
            str(row.get("position_state", "")),
            str(row.get("mismatch_reason", "")),
            ticker=str(row.get("ticker", "")),
            open_buy_tickers=open_buy_tickers,
        )
        eps.append(ep)
        es.append(est)
        healed.append(h)
        reasons.append(hr)
        inflight.append(asrc == "open_buy_in_flight")
        auth_src.append(asrc)
        recon_broker.append(rbrok)

    for i in range(len(eps)):
        row = merged.iloc[i]
        ep0, es0, adj_tag = _deterministic_effective_stance(
            eps[i],
            es[i],
            delta_pct=row.get("delta_pct"),
            confidence=row.get("confidence"),
            logic=logic,
        )
        eps[i] = ep0
        es[i] = es0
        stance_adj.append(adj_tag)
        if adj_tag:
            lc_consistency.append("ADJUSTED_STANCE")
        elif recon_broker[i]:
            lc_consistency.append("RECONCILED_BROKER")
        else:
            lc_consistency.append("OK")

    out = merged.copy()
    out["effective_position_state"] = eps
    out["effective_stance"] = es
    out["healed"] = healed
    out["heal_reason"] = reasons
    out["effective_in_flight"] = inflight
    out["lifecycle_authoritative_source"] = auth_src
    out["reconciled_with_broker"] = recon_broker
    out["reconciled_reason"] = [
        (reasons[i] if recon_broker[i] else "") for i in range(len(reasons))
    ]
    out["stance_adjustment"] = stance_adj
    out["lifecycle_consistency"] = lc_consistency

    # Reconciliation field was join-only; output = lifecycle columns + effective fields
    if "mismatch_reason" in out.columns:
        out = out.drop(columns=["mismatch_reason"])

    n_recon = int(sum(recon_broker)) if recon_broker else 0
    print(f"[RECONCILIATION] broker_reconciled_rows={n_recon} (snapshot+reconciliation.csv)")

    try:
        from services.lifecycle_truth import (
            enrich_execution_flags,
            print_effective_summary_stats,
            print_lifecycle_summary_from_effective,
        )

        out = enrich_execution_flags(out)
        print_lifecycle_summary_from_effective(out)
        print_effective_summary_stats(out)
    except Exception:
        pass

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build signal_lifecycle_effective.csv from lifecycle + reconciliation"
    )
    ap.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    ap.add_argument("--recon", type=Path, default=DEFAULT_RECON)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--open-orders",
        type=Path,
        default=DEFAULT_OPEN_ORDERS,
        help="Snapshot CSV for open orders (default data/results/open_orders_snapshot.csv)",
    )
    args = ap.parse_args()

    try:
        lc = load_lifecycle(args.lifecycle)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    try:
        recon = load_reconciliation(args.recon)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    lc_cfg = load_lifecycle_config()
    print("[LIFECYCLE CONFIG]", lc_cfg)
    lifecycle_logic = lifecycle_logic_from_dict(lc_cfg)

    if lc.empty:
        print("[WARN] signal_lifecycle.csv is empty; writing empty effective file.")
        out = lc.copy()
        out["effective_position_state"] = pd.Series(dtype=object)
        out["effective_stance"] = pd.Series(dtype=object)
        out["healed"] = pd.Series(dtype=bool)
        out["heal_reason"] = pd.Series(dtype=object)
        out["effective_in_flight"] = pd.Series(dtype=bool)
        out["lifecycle_authoritative_source"] = pd.Series(dtype=object)
        out["reconciled_with_broker"] = pd.Series(dtype=bool)
        out["reconciled_reason"] = pd.Series(dtype=object)
        out["stance_adjustment"] = pd.Series(dtype=object)
        out["lifecycle_consistency"] = pd.Series(dtype=object)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"[build_effective_lifecycle] wrote {args.out} total_rows=0 healed_count=0")
        return 0

    out = build_effective(
        lc,
        recon,
        open_orders_path=args.open_orders,
        lifecycle_logic=lifecycle_logic,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    n = len(out)
    n_healed = int(out["healed"].sum()) if "healed" in out.columns else 0
    n_inflight = (
        int(out["effective_in_flight"].sum()) if "effective_in_flight" in out.columns else 0
    )
    print(f"[build_effective_lifecycle] wrote {args.out}")
    print(
        f"[build_effective_lifecycle] total_rows={n} healed_count={n_healed} open_buy_in_flight_preserved={n_inflight}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
