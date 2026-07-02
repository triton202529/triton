"""
paper_execution_handoff_scope.py
Phase 197Z-HS — Enforce approved 197X scope at paper execution handoff boundary.
Read-only helpers; does not submit broker orders unless caller writes filtered CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAP = 50000.0

FORBIDDEN_OUTSIDE_SCOPE_SYMBOLS = frozenset(
    {"GLD", "GE", "TSLA", "ARKK", "MA", "PG", "ABBV", "UNG"}
)


def should_enforce_approved_197x_handoff_scope(*, mode: str = "paper") -> bool:
    """True when valid 197W/197X paper execution scope should constrain handoff rows."""
    if str(mode or "").lower() != "paper":
        return False
    from services.defensive_exit_dispatch_interlock import load_paper_approved_execution_scope

    _, _, scope_valid = load_paper_approved_execution_scope()
    return bool(scope_valid)


def _norm_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _symbols_from_planned(planned: Sequence[Any]) -> List[str]:
    return sorted(
        {
            _norm_symbol(getattr(p, "symbol", ""))
            for p in planned
            if _norm_symbol(getattr(p, "symbol", ""))
        }
    )


def estimate_handoff_notional_from_dataframe(df: Any) -> float:
    if df is None or getattr(df, "empty", True):
        return 0.0
    total = 0.0
    for _, row in df.iterrows():
        qty = row.get("qty")
        try:
            q = float(qty)
        except (TypeError, ValueError):
            continue
        if q <= 0:
            continue
        price = None
        for col in ("limit_price", "close", "last", "ref_price"):
            if col in row.index:
                try:
                    val = float(row.get(col))
                    if val > 0:
                        price = val
                        break
                except (TypeError, ValueError):
                    continue
        if price is not None:
            total += q * price
    return round(total, 2)


def filter_handoff_lifecycle_dataframe(
    df: Any,
    *,
    mode: str = "paper",
    batch_notional_cap: float = DEFAULT_CAP,
) -> Tuple[Any, Dict[str, Any]]:
    """Filter manage_positions handoff dataframe to approved 197X symbols only."""
    import pandas as pd

    from services.defensive_exit_dispatch_interlock import load_paper_approved_execution_scope

    if df is None or df.empty:
        return df, {
            "enforced": False,
            "pre_handoff_symbols": [],
            "approved_scope_symbols": [],
            "excluded_from_handoff_symbols": [],
            "final_handoff_symbols": [],
            "final_handoff_notional": 0.0,
            "batch_notional_cap": batch_notional_cap,
            "cap_passed": True,
            "outside_scope_rows_reached_placement": False,
        }

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    if "ticker" not in work.columns:
        return work, {"enforced": False, "reason": "missing_ticker_column"}

    work["ticker"] = work["ticker"].astype(str).str.strip().str.upper()
    pre_symbols = sorted(work["ticker"].unique())
    pre_notional = estimate_handoff_notional_from_dataframe(work)

    if not should_enforce_approved_197x_handoff_scope(mode=mode):
        final_symbols = pre_symbols
        return work, {
            "enforced": False,
            "pre_handoff_symbols": pre_symbols,
            "approved_scope_symbols": [],
            "excluded_from_handoff_symbols": [],
            "final_handoff_symbols": final_symbols,
            "pre_handoff_notional": pre_notional,
            "final_handoff_notional": pre_notional,
            "batch_notional_cap": batch_notional_cap,
            "cap_passed": pre_notional <= batch_notional_cap,
            "outside_scope_rows_reached_placement": False,
        }

    approved_symbols, scope_source, scope_valid = load_paper_approved_execution_scope()
    if not scope_valid or not approved_symbols:
        return work, {
            "enforced": False,
            "reason": "approved_scope_not_valid",
            "pre_handoff_symbols": pre_symbols,
            "pre_handoff_notional": pre_notional,
        }

    approved_set = set(approved_symbols)
    filtered = work[work["ticker"].isin(approved_set)].copy()
    final_symbols = sorted(filtered["ticker"].unique())
    excluded = sorted(set(pre_symbols) - set(final_symbols))
    final_notional = estimate_handoff_notional_from_dataframe(filtered)
    forbidden_present = sorted(set(final_symbols) & FORBIDDEN_OUTSIDE_SCOPE_SYMBOLS)
    excluded_forbidden = sorted(set(excluded) & FORBIDDEN_OUTSIDE_SCOPE_SYMBOLS)

    return filtered, {
        "enforced": True,
        "scope_source": scope_source,
        "pre_handoff_symbols": pre_symbols,
        "approved_scope_symbols": approved_symbols,
        "excluded_from_handoff_symbols": excluded,
        "final_handoff_symbols": final_symbols,
        "pre_handoff_notional": pre_notional,
        "final_handoff_notional": final_notional,
        "batch_notional_cap": batch_notional_cap,
        "cap_passed": final_notional <= batch_notional_cap,
        "forbidden_symbols_in_final_handoff": forbidden_present,
        "forbidden_symbols_excluded": excluded_forbidden,
        "outside_scope_rows_reached_placement": bool(forbidden_present),
    }


def build_handoff_scope_diagnostics(
    *,
    pre_handoff_symbols: List[str],
    final_planned: Sequence[Any],
    scope_meta: Dict[str, Any],
    pre_planned: Optional[Sequence[Any]] = None,
    batch_notional_cap: float = DEFAULT_CAP,
) -> Dict[str, Any]:
    """Diagnostics for manage_positions planned-order handoff scope filtering."""

    def _planned_notional(planned: Sequence[Any]) -> float:
        total = 0.0
        for p in planned:
            try:
                q = float(getattr(p, "qty", 0) or 0)
                lp = getattr(p, "limit_price", None)
                if lp is not None and q > 0:
                    total += q * float(lp)
            except (TypeError, ValueError):
                pass
        return round(total, 2)

    final_symbols = _symbols_from_planned(final_planned)
    excluded = sorted(set(pre_handoff_symbols) - set(final_symbols))
    pre_notional = _planned_notional(pre_planned) if pre_planned is not None else 0.0
    final_notional = _planned_notional(final_planned)
    approved_symbols = list(scope_meta.get("paper_scope_approved_symbols") or final_symbols)
    return {
        "pre_handoff_symbols": sorted(pre_handoff_symbols),
        "approved_scope_symbols": approved_symbols,
        "excluded_from_handoff_symbols": excluded,
        "final_handoff_symbols": final_symbols,
        "pre_handoff_notional_estimate": round(pre_notional, 2),
        "final_handoff_notional_estimate": round(final_notional, 2),
        "batch_notional_cap": batch_notional_cap,
        "cap_passed": final_notional <= batch_notional_cap,
        "outside_scope_rows_reached_placement": False,
        "paper_scope_constrained": bool(scope_meta.get("paper_scope_constrained")),
        "paper_scope_source": scope_meta.get("paper_scope_source"),
    }


def filter_order_rows_to_approved_197x_scope(
    rows: Sequence[Any],
    *,
    mode: str = "paper",
) -> Tuple[List[Any], Dict[str, Any]]:
    """Filter place_live_orders OrderRow list to approved 197X scope when active."""
    if not should_enforce_approved_197x_handoff_scope(mode=mode):
        symbols = sorted(
            {
                _norm_symbol(getattr(r, "symbol", ""))
                for r in rows
                if _norm_symbol(getattr(r, "symbol", ""))
            }
        )
        return list(rows), {
            "enforced": False,
            "pre_handoff_symbols": symbols,
            "final_handoff_symbols": symbols,
            "outside_scope_rows_reached_placement": False,
        }

    from services.defensive_exit_dispatch_interlock import load_paper_approved_execution_scope

    approved_symbols, scope_source, scope_valid = load_paper_approved_execution_scope()
    if not scope_valid:
        return list(rows), {"enforced": False, "reason": "approved_scope_not_valid"}

    approved_set = set(approved_symbols)
    pre_symbols = sorted(
        {
            _norm_symbol(getattr(r, "symbol", ""))
            for r in rows
            if _norm_symbol(getattr(r, "symbol", ""))
        }
    )
    kept = [r for r in rows if _norm_symbol(getattr(r, "symbol", "")) in approved_set]
    final_symbols = sorted(
        {
            _norm_symbol(getattr(r, "symbol", ""))
            for r in kept
            if _norm_symbol(getattr(r, "symbol", ""))
        }
    )
    excluded = sorted(set(pre_symbols) - set(final_symbols))
    forbidden_present = sorted(set(final_symbols) & FORBIDDEN_OUTSIDE_SCOPE_SYMBOLS)
    return kept, {
        "enforced": True,
        "scope_source": scope_source,
        "pre_handoff_symbols": pre_symbols,
        "approved_scope_symbols": approved_symbols,
        "excluded_from_handoff_symbols": excluded,
        "final_handoff_symbols": final_symbols,
        "forbidden_symbols_in_final_handoff": forbidden_present,
        "outside_scope_rows_reached_placement": bool(forbidden_present),
    }
