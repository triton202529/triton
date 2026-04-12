"""
build_trade_opportunities.py
----------------------------
Build execution-ready trade opportunities from signal_lifecycle_effective.csv.

Input:  data/results/signal_lifecycle_effective.csv (read-only)
Output: data/results/trade_opportunities.csv

Classification uses (effective_position_state, lifecycle_action) so broker-truth position
pairs with lifecycle intent — not effective_stance, which may be WAIT/HOLD after reconciliation.

Post-lifecycle drops (only): invalid price, qty=0, hard risk blocks (read-only from state files).
Diagnostics: data/results/trade_opportunity_build_diagnostics.json, trade_opportunity_build_drops.csv

If strict opportunity_count is 0 after the above, optionally inject up to N exploratory FLAT→ENTRY rows
(exploration_flag=True). Exploration pool may apply confidence/delta filters (diagnostic only for that path).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_IN = RESULTS_DIR / "signal_lifecycle_effective.csv"
DEFAULT_OUT = RESULTS_DIR / "trade_opportunities.csv"
DIAG_JSON = RESULTS_DIR / "trade_opportunity_build_diagnostics.json"
DROPS_CSV = RESULTS_DIR / "trade_opportunity_build_drops.csv"
EXPLORATION_TOP_N = 3

# Symbols not executable via typical broker APIs (indices, etc.)
INVALID_SYMBOLS = ["^VIX"]

CONTEXT_COLS = [
    "ticker",
    "effective_stance",
    "effective_position_state",
    "lifecycle_decision_reason",
    "confidence",
    "delta_pct",
    "rationale",
    "healed",
    "heal_reason",
    "reason_code",
]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_effective(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing effective lifecycle file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("signal_lifecycle_effective.csv must include ticker")
    if "effective_position_state" not in df.columns:
        raise ValueError("signal_lifecycle_effective.csv must include effective_position_state")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df


def _series_lifecycle_action(df: pd.DataFrame) -> pd.Series:
    """Prefer lifecycle_action; else stance (raw lifecycle output before effective_stance overlay)."""
    if "lifecycle_action" in df.columns:
        s = df["lifecycle_action"]
    elif "stance" in df.columns:
        s = df["stance"]
    else:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    return s.fillna("").astype(str).str.strip().str.upper()


def classify_opportunity_from_lifecycle(pos_u: str, lifecycle_action_u: str) -> Optional[str]:
    """
    Map broker effective position + lifecycle intent → opportunity_type.
    LONG+BUY → ADD (add to existing); FLAT+BUY → ENTRY.
    """
    p = str(pos_u or "").strip().upper()
    a = str(lifecycle_action_u or "").strip().upper()
    if p == "FLAT" and a == "BUY":
        return "ENTRY"
    if p == "LONG" and a == "BUY":
        return "ADD"
    if p == "LONG" and a == "ADD":
        return "ADD"
    if p == "LONG" and a == "TRIM":
        return "TRIM"
    if p == "LONG" and a == "EXIT":
        return "EXIT"
    return None


def _intent_stance_for_opportunity(opportunity_type: str) -> str:
    return {"ENTRY": "BUY", "ADD": "ADD", "TRIM": "TRIM", "EXIT": "EXIT"}.get(
        str(opportunity_type or "").strip().upper(), ""
    )


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_hard_risk_flags() -> Tuple[bool, bool]:
    """
    allow_new_orders, allow_new_trades — False means hard block (mirror execute_trades / risk snapshot).
    """
    allow_new_orders = True
    allow_new_trades = True

    rj = _load_json(RESULTS_DIR / "adaptive_risk_state.json")
    if isinstance(rj, dict):
        ctrl = rj.get("controls") if isinstance(rj.get("controls"), dict) else {}
        if "risk_on" in ctrl:
            allow_new_orders = allow_new_orders and bool(ctrl.get("risk_on", True))
        if "allow_new_orders" in ctrl:
            allow_new_orders = allow_new_orders and bool(ctrl.get("allow_new_orders", True))

    for cpm_path in (
        RESULTS_DIR / "capital_preservation_mode.json",
        RESULTS_DIR / "capital_preservation_state.json",
    ):
        cj = _load_json(cpm_path)
        if isinstance(cj, dict) and "allow_new_trades" in cj:
            allow_new_trades = allow_new_trades and bool(cj.get("allow_new_trades", True))
            break

    return allow_new_orders, allow_new_trades


def _row_invalid_price(row: pd.Series) -> bool:
    """True if close is unusable for sizing reference."""
    if "close" not in row.index:
        return False
    c = pd.to_numeric(row.get("close"), errors="coerce")
    if pd.isna(c) or float(c) <= 0:
        return True
    return False


def _row_invalid_qty_zero(row: pd.Series) -> bool:
    """True only when an explicit qty column exists and is numerically zero."""
    for q in ("qty", "target_qty", "planned_qty", "order_qty"):
        if q not in row.index:
            continue
        v = row.get(q)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        try:
            if float(v) == 0.0:
                return True
        except Exception:
            continue
    return False


def _risk_guard_drops_opportunity(
    opportunity_type: str,
    allow_new_orders: bool,
    allow_new_trades: bool,
) -> bool:
    ot = str(opportunity_type or "").strip().upper()
    if not allow_new_trades:
        return True
    if ot in ("ENTRY", "ADD") and not allow_new_orders:
        return True
    return False


def _output_columns(sub: pd.DataFrame) -> list[str]:
    head = ["ticker", "effective_stance", "effective_position_state"]
    optional = [c for c in CONTEXT_COLS if c not in head and c in sub.columns]
    tail = ["opportunity_type", "exploration_flag"]
    return [c for c in head if c in sub.columns] + optional + tail


def _exploration_entry_fallback(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """When strict opportunity_count == 0: filtered FLAT rows → top-N exploratory ENTRY (BUY)."""
    base_cols = [c for c in CONTEXT_COLS if c in df.columns] + [
        "opportunity_type",
        "exploration_flag",
    ]
    pos = df["effective_position_state"].fillna("").astype(str).str.strip().str.upper()
    flat = df.loc[pos == "FLAT"].copy()
    if flat.empty:
        print("[EXPLORATION] filtered_candidates=", 0)
        print("[EXPLORATION] selected=", 0)
        return pd.DataFrame(columns=base_cols)

    mask = pd.Series(True, index=flat.index)
    if "confidence" in flat.columns:
        cnum = pd.to_numeric(flat["confidence"], errors="coerce")
        mask = mask & (cnum >= 0.55)
    if "delta_pct" in flat.columns:
        dnum = pd.to_numeric(flat["delta_pct"], errors="coerce")
        mask = mask & (dnum.abs() >= 0.01)

    filtered = flat.loc[mask].copy()
    print("[EXPLORATION] filtered_candidates=", len(filtered))

    if filtered.empty:
        print("[EXPLORATION] selected=", 0)
        return pd.DataFrame(columns=base_cols)

    conf = (
        pd.to_numeric(filtered["confidence"], errors="coerce")
        if "confidence" in filtered.columns
        else None
    )
    if conf is not None and conf.notna().any():
        filtered["_sort"] = conf
    elif "delta_pct" in filtered.columns:
        filtered["_sort"] = pd.to_numeric(filtered["delta_pct"], errors="coerce")
    else:
        filtered["_sort"] = 0.0
    filtered["_sort"] = filtered["_sort"].fillna(float("-inf"))
    filtered = filtered.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    selected = filtered.head(n).copy()
    print("[EXPLORATION] selected=", len(selected))

    selected["effective_stance"] = "BUY"
    selected["opportunity_type"] = "ENTRY"
    selected["exploration_flag"] = True
    if "reason_code" not in selected.columns:
        selected["reason_code"] = "OK"
    else:
        selected["reason_code"] = "OK"

    cols = _output_columns(selected)
    return selected[cols].reset_index(drop=True)


def build_opportunities_clean(
    df: pd.DataFrame,
    *,
    allow_new_orders: bool,
    allow_new_trades: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Returns (opportunities_df, diagnostics_dict, drops_df).
    """
    out_cols = CONTEXT_COLS + ["opportunity_type", "exploration_flag"]
    empty_diag: Dict[str, Any] = {
        "timestamp": _utc_iso(),
        "input_rows": 0,
        "lifecycle_actionable_rows": 0,
        "opportunities_emitted": 0,
        "dropped_after_lifecycle": 0,
        "not_actionable_lifecycle": 0,
        "drop_reason_counts": {},
    }
    if df.empty:
        return pd.DataFrame(columns=out_cols), empty_diag, pd.DataFrame()

    pos = df["effective_position_state"].fillna("").astype(str).str.strip().str.upper()
    lc = _series_lifecycle_action(df)

    kept_idx: List[int] = []
    otypes: List[str] = []
    reason_codes: List[str] = []
    drops: List[Dict[str, Any]] = []

    actionable_mask = []
    for i in range(len(df)):
        row = df.iloc[i]
        ot = classify_opportunity_from_lifecycle(str(pos.iloc[i]), str(lc.iloc[i]))
        actionable_mask.append(ot is not None)
        if ot is None:
            continue

        reason = ""
        if _row_invalid_price(row):
            reason = "INVALID_PRICE"
        elif _row_invalid_qty_zero(row):
            reason = "INVALID_QTY"
        elif _risk_guard_drops_opportunity(ot, allow_new_orders, allow_new_trades):
            reason = "RISK_GUARD"
        else:
            kept_idx.append(i)
            otypes.append(ot)
            reason_codes.append("OK")
            continue

        drops.append(
            {
                "ticker": str(row.get("ticker", "")).strip().upper(),
                "effective_position_state": str(pos.iloc[i]),
                "lifecycle_action": str(lc.iloc[i]),
                "opportunity_type_would_be": ot,
                "reason_code": reason,
            }
        )

    lifecycle_actionable = int(sum(actionable_mask))
    dropped_after = len(drops)

    drop_counts: Dict[str, int] = {}
    for d in drops:
        rc = str(d.get("reason_code") or "UNKNOWN_FILTER")
        drop_counts[rc] = drop_counts.get(rc, 0) + 1

    not_actionable = len(df) - lifecycle_actionable

    if not kept_idx:
        empty_df = pd.DataFrame(columns=out_cols)
        diag = {
            "timestamp": _utc_iso(),
            "input_rows": len(df),
            "lifecycle_actionable_rows": lifecycle_actionable,
            "opportunities_emitted": 0,
            "dropped_after_lifecycle": dropped_after,
            "not_actionable_lifecycle": not_actionable,
            "drop_reason_counts": drop_counts,
            "hard_risk_flags": {
                "allow_new_orders": allow_new_orders,
                "allow_new_trades": allow_new_trades,
            },
        }
        return empty_df, diag, pd.DataFrame(drops)

    sub = df.iloc[kept_idx].copy()
    sub["opportunity_type"] = otypes
    sub["exploration_flag"] = False
    # Execution handoff uses effective_stance when non-empty; set intent so WAIT/HOLD does not override mapped stance.
    sub["effective_stance"] = [_intent_stance_for_opportunity(o) for o in otypes]
    sub["reason_code"] = reason_codes

    cols = _output_columns(sub)
    diag = {
        "timestamp": _utc_iso(),
        "input_rows": len(df),
        "lifecycle_actionable_rows": lifecycle_actionable,
        "opportunities_emitted": len(sub),
        "dropped_after_lifecycle": dropped_after,
        "not_actionable_lifecycle": not_actionable,
        "drop_reason_counts": drop_counts,
        "hard_risk_flags": {
            "allow_new_orders": allow_new_orders,
            "allow_new_trades": allow_new_trades,
        },
    }
    return sub[cols].reset_index(drop=True), diag, pd.DataFrame(drops)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build trade_opportunities.csv from signal_lifecycle_effective.csv"
    )
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    try:
        df = load_effective(args.in_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    allow_new_orders, allow_new_trades = _parse_hard_risk_flags()
    out_df, diag, drops_df = build_opportunities_clean(
        df, allow_new_orders=allow_new_orders, allow_new_trades=allow_new_trades
    )

    if out_df.empty:
        out_df = _exploration_entry_fallback(df, EXPLORATION_TOP_N)
        diag["exploration_fallback_used"] = True
        diag["opportunities_emitted_after_exploration"] = len(out_df)
    else:
        diag["exploration_fallback_used"] = False

    if not out_df.empty and "ticker" in out_df.columns:
        out_df = out_df[~out_df["ticker"].isin(INVALID_SYMBOLS)].copy()
        diag["opportunities_emitted"] = len(out_df)
        if diag.get("exploration_fallback_used"):
            diag["opportunities_emitted_after_exploration"] = len(out_df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    try:
        diag["reason_code_semantics"] = {
            "OK": "Included in trade_opportunities.csv (strict path).",
            "INVALID_PRICE": "close missing, NaN, or <= 0 (hard drop).",
            "INVALID_QTY": "explicit qty/target_qty/planned_qty/order_qty column equals 0.",
            "RISK_GUARD": "Hard block from adaptive_risk_state / capital_preservation (no sizing changes).",
            "LOW_CONFIDENCE": "Not used to drop strict lifecycle rows (reserved / exploration diagnostics only).",
            "LOW_DELTA": "Not used to drop strict lifecycle rows (reserved / exploration diagnostics only).",
            "UNKNOWN_FILTER": "Unexpected drop reason (should not occur in strict path).",
        }
        DIAG_JSON.parent.mkdir(parents=True, exist_ok=True)
        DIAG_JSON.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    except Exception:
        pass

    try:
        if not drops_df.empty:
            drops_df.to_csv(DROPS_CSV, index=False)
        else:
            pd.DataFrame(
                columns=[
                    "ticker",
                    "effective_position_state",
                    "lifecycle_action",
                    "opportunity_type_would_be",
                    "reason_code",
                ]
            ).to_csv(DROPS_CSV, index=False)
    except Exception:
        pass

    try:
        from services.signal_pressure_diagnostics import refresh_signal_pressure_diagnostics

        refresh_signal_pressure_diagnostics()
    except Exception:
        pass

    n = len(out_df)
    print(f"[build_trade_opportunities] wrote {args.out}")
    print(f"[build_trade_opportunities] total_rows={n} opportunity_count={n}")
    print(
        f"[build_trade_opportunities] dropped_after_lifecycle={diag.get('dropped_after_lifecycle', 0)} "
        f"lifecycle_actionable={diag.get('lifecycle_actionable_rows', 0)} input_rows={diag.get('input_rows', 0)}"
    )
    if n == 0:
        print(
            "[build_trade_opportunities] idle: no strict opportunities; exploration pool empty or failed filters"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
