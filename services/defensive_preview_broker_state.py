"""
defensive_preview_broker_state.py
Phase 198N — Invalidate defensive previews that no longer match broker position.

Broker position is the authority after dispatch. Completed/stale defensive
batches must not keep advertising sell quantities the account can no longer
support (or that have already been completed).

Read-only classification helper. Does not create approvals, broker auth, or orders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
POSITIONS_CSV = RESULTS / "positions_snapshot.csv"

STALE_DEFENSIVE_PREVIEW_INVALIDATED = "STALE_DEFENSIVE_PREVIEW_INVALIDATED"
TARGET_ALREADY_COMPLETED = "TARGET_ALREADY_COMPLETED"
DEFENSIVE_PREVIEW_FRESH = "DEFENSIVE_PREVIEW_FRESH"
DEFENSIVE_PREVIEW_ABSENT = "DEFENSIVE_PREVIEW_ABSENT"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def load_broker_symbol_qty(symbol: str = "GLD") -> Dict[str, Any]:
    """Load current broker/paper position qty for a symbol from positions snapshot."""
    symbol_u = str(symbol or "").strip().upper()
    if not POSITIONS_CSV.is_file():
        return {
            "ok": False,
            "symbol": symbol_u,
            "qty": 0,
            "reason": "positions_snapshot_missing",
        }
    try:
        import pandas as pd

        df = pd.read_csv(POSITIONS_CSV, low_memory=False)
    except Exception as exc:
        return {
            "ok": False,
            "symbol": symbol_u,
            "qty": 0,
            "reason": f"positions_snapshot_read_failed:{exc}",
        }
    if df.empty:
        return {"ok": False, "symbol": symbol_u, "qty": 0, "reason": "positions_snapshot_empty"}
    sym_col = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
    if not sym_col:
        return {
            "ok": False,
            "symbol": symbol_u,
            "qty": 0,
            "reason": "positions_snapshot_missing_symbol_column",
        }
    mask = df[sym_col].astype(str).str.strip().str.upper() == symbol_u
    if not mask.any():
        return {
            "ok": True,
            "symbol": symbol_u,
            "qty": 0,
            "market_value": 0.0,
            "reason": "symbol_not_held",
        }
    row = df.loc[mask].iloc[0]
    qty = int(_safe_float(row.get("qty") or row.get("qty_available")))
    return {
        "ok": True,
        "symbol": symbol_u,
        "qty": max(0, qty),
        "market_value": round(abs(_safe_float(row.get("market_value") or row.get("value"))), 2),
        "current_price": round(_safe_float(row.get("current_price")), 6),
        "snapshot_ts": str(row.get("snapshot_ts") or ""),
        "source": str(POSITIONS_CSV.relative_to(ROOT)).replace("\\", "/"),
    }


def evaluate_defensive_preview_broker_freshness(
    *,
    symbol: str = "GLD",
    expected_pre_dispatch_qty: int = 0,
    preview_sell_qty: int = 0,
    preview_remaining_qty: int = 0,
    broker_qty: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compare a defensive preview's expected pre-dispatch position to broker qty.

    Invalidates when broker qty is not the preview's expected pre-dispatch qty.
    Marks TARGET_ALREADY_COMPLETED when broker qty is at/below the preview's
    remaining qty (batch already finished).
    """
    symbol_u = str(symbol or "GLD").strip().upper()
    if broker_qty is None:
        pos = load_broker_symbol_qty(symbol_u)
        broker_qty_i = int(pos.get("qty") or 0)
        broker_ok = bool(pos.get("ok"))
        broker_meta = pos
    else:
        broker_qty_i = int(broker_qty)
        broker_ok = True
        broker_meta = {"ok": True, "symbol": symbol_u, "qty": broker_qty_i}

    expected = int(expected_pre_dispatch_qty or 0)
    sell_qty = int(preview_sell_qty or 0)
    remaining = int(preview_remaining_qty or 0)

    base: Dict[str, Any] = {
        "symbol": symbol_u,
        "broker_qty": broker_qty_i,
        "expected_pre_dispatch_qty": expected,
        "preview_sell_qty": sell_qty,
        "preview_remaining_qty": remaining,
        "broker_position": broker_meta,
        "fresh": False,
        "invalidated": False,
        "eligible_qty": 0,
        "bridge_qty": 0,
        "final_state": DEFENSIVE_PREVIEW_ABSENT,
        "reason": "",
        "approval_cannot_revive": True,
    }

    if expected <= 0 and sell_qty <= 0:
        base["reason"] = "no_defensive_preview_quantities"
        base["final_state"] = DEFENSIVE_PREVIEW_ABSENT
        return base

    if not broker_ok:
        base["invalidated"] = True
        base["final_state"] = STALE_DEFENSIVE_PREVIEW_INVALIDATED
        base["reason"] = f"broker_position_unavailable:{broker_meta.get('reason')}"
        return base

    # Completed target: broker already at/below the preview's remaining quantity.
    if remaining > 0 and broker_qty_i <= remaining and expected > 0 and broker_qty_i != expected:
        base["invalidated"] = True
        base["final_state"] = TARGET_ALREADY_COMPLETED
        base["reason"] = (
            f"broker_qty={broker_qty_i}<=preview_remaining={remaining}; "
            f"expected_pre_dispatch_qty={expected}"
        )
        return base

    # Strict rule: broker qty must match the preview's expected pre-dispatch qty.
    if expected > 0 and broker_qty_i != expected:
        base["invalidated"] = True
        base["final_state"] = STALE_DEFENSIVE_PREVIEW_INVALIDATED
        base["reason"] = (
            f"broker_qty={broker_qty_i}!=expected_pre_dispatch_qty={expected}; "
            f"preview_sell_qty={sell_qty}"
        )
        return base

    # Cannot advertise a sell larger than the held position.
    if sell_qty > 0 and broker_qty_i < sell_qty:
        base["invalidated"] = True
        base["final_state"] = STALE_DEFENSIVE_PREVIEW_INVALIDATED
        base["reason"] = f"broker_qty={broker_qty_i}<preview_sell_qty={sell_qty}"
        return base

    base["fresh"] = True
    base["invalidated"] = False
    base["eligible_qty"] = sell_qty
    base["bridge_qty"] = sell_qty
    base["final_state"] = DEFENSIVE_PREVIEW_FRESH
    base["reason"] = "broker_qty_matches_expected_pre_dispatch_qty"
    return base


def apply_broker_invalidation_to_partial_fields(
    partial: Dict[str, Any],
    *,
    projected_qty: int = 0,
) -> Dict[str, Any]:
    """
    Return a copy of partial contract fields with sell eligibility cleared when
    broker state shows the defensive preview is stale/completed.
    """
    out = dict(partial or {})
    symbol = str(out.get("symbol") or "GLD").upper()
    sell_qty = int(out.get("partial_qty") or projected_qty or 0)
    if projected_qty > sell_qty:
        sell_qty = int(projected_qty)
    full_qty = int(out.get("full_qty") or 0)
    remaining = int(out.get("remaining_qty_after_preview") or max(0, full_qty - sell_qty))
    expected = full_qty if full_qty > 0 else (sell_qty + remaining if sell_qty > 0 else 0)

    freshness = evaluate_defensive_preview_broker_freshness(
        symbol=symbol,
        expected_pre_dispatch_qty=expected,
        preview_sell_qty=sell_qty,
        preview_remaining_qty=remaining,
    )
    out["defensive_preview_broker_state"] = freshness
    out["defensive_preview_state"] = freshness.get("final_state")
    out["defensive_preview_invalidated"] = bool(freshness.get("invalidated"))
    out["broker_qty"] = freshness.get("broker_qty")
    out["expected_pre_dispatch_qty"] = freshness.get("expected_pre_dispatch_qty")

    if freshness.get("invalidated"):
        out["partial_preview_available"] = False
        out["partial_qty"] = 0
        out["approved_quantities"] = {}
        out["stale_preview_original_qty"] = sell_qty
        out["stale_preview_original_full_qty"] = full_qty
        out["eligible_qty"] = 0
        out["invalidation_reason"] = freshness.get("reason")
    return out
