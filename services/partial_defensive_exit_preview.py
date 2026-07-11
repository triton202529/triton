"""
partial_defensive_exit_preview.py
Phase 198D — Cap-aware partial defensive exit preview from 198C sizing analysis.

Preview/governance integration only. Does not submit orders or create approvals.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
SIZING_JSON = RESULTS / "gld_partial_exit_sizing_analysis.json"
POSITIONS_CSV = RESULTS / "positions_snapshot.csv"
READINESS_JSON = RESULTS / "execution_readiness.json"
OPEN_ORDERS_CSV = RESULTS / "open_orders_snapshot.csv"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

PREVIEW_TYPE = "PARTIAL_DEFENSIVE_EXIT_PREVIEW"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _count_open_orders() -> int:
    if not OPEN_ORDERS_CSV.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(OPEN_ORDERS_CSV, low_memory=False))
    except Exception:
        return 0


def _held_symbols() -> Set[str]:
    if not POSITIONS_CSV.is_file():
        return set()
    try:
        import pandas as pd

        df = pd.read_csv(POSITIONS_CSV, low_memory=False)
        sym_col = "symbol" if "symbol" in df.columns else "ticker"
        if sym_col not in df.columns:
            return set()
        out: Set[str] = set()
        for _, row in df.iterrows():
            sym = str(row.get(sym_col) or "").strip().upper()
            if not sym:
                continue
            qty = _safe_float(row.get("qty") or row.get("quantity"))
            if qty != 0:
                out.add(sym)
        return out
    except Exception:
        return set()


def _live_blocked() -> bool:
    readiness = _read_json(READINESS_JSON)
    if bool(readiness.get("live_execution_permitted", False)):
        return False
    try:
        from services.portfolio_history_sync import load_live_execution_flag

        if load_live_execution_flag():
            return False
    except Exception:
        pass
    return True


def ensure_gld_sizing_analysis(*, refresh: bool = True) -> Dict[str, Any]:
    """Load 198C analysis; optionally refresh from current positions."""
    if refresh:
        try:
            from services.gld_partial_exit_sizing_analysis import (
                run_gld_partial_exit_sizing_analysis,
            )

            return run_gld_partial_exit_sizing_analysis(verbose=False)
        except Exception:
            pass
    return _read_json(SIZING_JSON)


def build_partial_defensive_exit_preview(
    *,
    configured_cap: float,
    deferred_symbols: Optional[List[str]] = None,
    handoff_symbols: Optional[List[str]] = None,
    refresh_sizing: bool = True,
) -> Dict[str, Any]:
    """
    Build a governed partial-exit preview candidate when full exits exceed cap.

    Returns empty/unavailable payload when preconditions fail.
    Never creates approval or broker-auth artifacts.
    """
    deferred = {str(s).upper() for s in (deferred_symbols or []) if s}
    handoff = {str(s).upper() for s in (handoff_symbols or []) if s}
    held = _held_symbols()
    open_orders = _count_open_orders()
    live_ok = _live_blocked()

    base: Dict[str, Any] = {
        "partial_preview_available": False,
        "partial_preview_source": "gld_partial_exit_sizing_analysis.json",
        "partial_preview_type": PREVIEW_TYPE,
        "partial_preview_symbol": None,
        "partial_preview_qty": 0,
        "partial_preview_notional": 0.0,
        "partial_preview_classification": None,
        "partial_preview_scenario": None,
        "full_exit_notional": None,
        "full_exit_deferred_reason": None,
        "partial_preview_requires_operator_approval": True,
        "partial_preview_analysis_only": True,
        "paper_only": True,
        "broker_orders_submitted": False,
        "eligible_for_paper_execution": False,
        "open_orders_count": open_orders,
        "live_execution_blocked": live_ok,
        "approval_artifact_created": False,
        "broker_authorization_artifact_created": False,
        "generated_at": _utc_now_iso(),
    }

    if open_orders != 0:
        base["block_reason"] = f"open_orders_count={open_orders}"
        return base
    if not live_ok:
        base["block_reason"] = "live_execution_not_blocked"
        return base

    sizing = ensure_gld_sizing_analysis(refresh=refresh_sizing)
    if not sizing or sizing.get("status") != "PASS":
        base["block_reason"] = "gld_sizing_analysis_unavailable"
        return base

    symbol = str(sizing.get("symbol") or "GLD").upper()
    if symbol not in held:
        base["block_reason"] = f"{symbol}_not_held"
        return base
    if deferred and symbol not in deferred and symbol not in handoff:
        # Only apply when GLD is in the defensive deferred/handoff set.
        base["block_reason"] = f"{symbol}_not_in_deferred_or_handoff"
        return base

    full = sizing.get("full_exit") or {}
    full_notional = _safe_float(full.get("estimated_notional"))
    full_qty = _safe_float((sizing.get("current_position") or {}).get("quantity"))
    fits_full = bool(full.get("fits_configured_cap"))
    if fits_full or full_notional <= configured_cap + 1e-6:
        base["block_reason"] = "full_exit_already_fits_cap"
        base["full_exit_notional"] = full_notional
        return base

    rec = sizing.get("recommendation") or {}
    scenario_name = str(rec.get("recommended_scenario") or "")
    scenarios = list(sizing.get("scenarios") or [])
    chosen = next((s for s in scenarios if s.get("scenario_name") == scenario_name), None)
    if chosen is None:
        # Fallback: first REASONABLE_PARTIAL that passes configured cap.
        for s in scenarios:
            if (
                s.get("classification") == "REASONABLE_PARTIAL"
                and s.get("cap_pass")
                and _safe_float(s.get("cap")) == float(configured_cap)
            ):
                chosen = s
                scenario_name = str(s.get("scenario_name") or "")
                break
    if not chosen:
        base["block_reason"] = "no_feasible_partial_scenario"
        base["full_exit_notional"] = full_notional
        base["full_exit_deferred_reason"] = "FULL_EXIT_EXCEEDS_CAP"
        return base

    sell_qty = _safe_float(chosen.get("sell_qty"))
    notional = _safe_float(chosen.get("estimated_notional"))
    classification = str(chosen.get("classification") or "")
    if sell_qty <= 0 or sell_qty >= full_qty:
        base["block_reason"] = "invalid_partial_qty"
        return base
    if notional > configured_cap + 1e-6 or not chosen.get("cap_pass"):
        base["block_reason"] = "partial_notional_exceeds_cap"
        return base

    limit_price = _safe_float(full.get("estimated_limit_price"))
    if limit_price <= 0 and sell_qty > 0:
        limit_price = round(notional / sell_qty, 4)

    order = {
        "symbol": symbol,
        "qty": int(sell_qty),
        "limit_price": limit_price,
        "notional": round(notional, 2),
        "opportunity_type": "TRIM",
        "effective_stance": "TRIM",
        "preview_type": PREVIEW_TYPE,
        "partial_defensive_exit": True,
        "scenario_name": scenario_name,
        "classification": classification,
        "full_position_qty": full_qty,
        "remaining_qty": _safe_float(chosen.get("remaining_qty")),
        "remaining_notional": _safe_float(chosen.get("remaining_notional")),
        "paper_only": True,
        "broker_orders_submitted": False,
        "discipline_allowed": True,
        "rank": 1,
        "reason_code": "PARTIAL_EXIT_CAP_AWARE",
        "priority_bucket": "EXIT",
    }

    base.update(
        {
            "partial_preview_available": True,
            "partial_preview_symbol": symbol,
            "partial_preview_qty": int(sell_qty),
            "partial_preview_notional": round(notional, 2),
            "partial_preview_classification": classification,
            "partial_preview_scenario": scenario_name,
            "full_exit_notional": full_notional,
            "full_exit_deferred_reason": (
                f"FULL_EXIT_EXCEEDS_CAP notional={full_notional} cap={configured_cap}"
            ),
            "partial_preview_order": order,
            "remaining_qty": order["remaining_qty"],
            "remaining_notional": order["remaining_notional"],
            "cap_used_pct": _safe_float(chosen.get("cap_used_pct")),
            "configured_cap": configured_cap,
            "block_reason": "",
            "approval_artifact_present": APPROVAL_PATH.is_file(),
            "broker_auth_artifact_present": BROKER_AUTH_PATH.is_file(),
        }
    )
    return base


def apply_partial_preview_to_capped_block(
    capped_block: Dict[str, Any],
    partial: Dict[str, Any],
    *,
    batch_cap: float,
) -> Dict[str, Any]:
    """Inject partial order into an empty capped preview block (copy)."""
    out = dict(capped_block or {})
    if not partial.get("partial_preview_available"):
        return out
    order = dict(partial.get("partial_preview_order") or {})
    if not order:
        return out
    if out.get("capped_preview_symbols"):
        # Full-symbol capped preview already exists; do not override.
        return out

    symbol = str(order.get("symbol") or "").upper()
    deferred = [str(s).upper() for s in (out.get("deferred_symbols") or [])]
    if symbol and symbol not in deferred:
        deferred.append(symbol)

    out["batch_cap"] = batch_cap
    out["batch_notional_capped_preview"] = round(_safe_float(order.get("notional")), 2)
    out["capped_preview_symbols"] = [symbol]
    out["deferred_symbols"] = deferred
    out["capped_preview_orders"] = [order]
    out["next_eligible_batch_symbols"] = [symbol]
    out["preview_type"] = PREVIEW_TYPE
    out["partial_defensive_exit_preview"] = True
    out["concentration_risk_note"] = (
        f"Partial {symbol} defensive exit preview ({partial.get('partial_preview_scenario')}) "
        f"fits cap={batch_cap}; full exit deferred because notional exceeds cap."
    )
    # Keep deferred_orders for full-exit visibility if present.
    deferred_orders = list(out.get("deferred_orders") or [])
    if not any(str(o.get("symbol") or "").upper() == symbol for o in deferred_orders):
        deferred_orders.append(
            {
                "symbol": symbol,
                "qty": order.get("full_position_qty"),
                "notional": partial.get("full_exit_notional"),
                "defer_reason": "full_exit_exceeds_batch_notional_cap",
                "partial_preview_available": True,
            }
        )
    out["deferred_orders"] = deferred_orders
    return out


def enrich_sizing_analysis_with_partial_preview(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach GLD partial preview when full GLD exit exceeds the manage_positions cap.
    If no full-symbol orders fit the runtime cap, inject the partial into capped symbols.
    """
    out = dict(analysis)
    runtime_cap = _safe_float(
        out.get("configured_batch_cap_manage_positions")
        or out.get("configured_batch_cap_effective")
        or 50000.0
    )
    runtime_preview = dict(out.get("capped_preview_runtime_cap") or {})
    deferred = list(runtime_preview.get("deferred_symbols") or out.get("blocked_symbols") or [])
    handoff = list(out.get("handoff_symbols") or [])
    existing_symbols = [
        str(s).upper() for s in (runtime_preview.get("capped_preview_symbols") or [])
    ]

    # Prefer GLD as deferred/handoff candidate even when other symbols already fit.
    candidate_syms = sorted(set(deferred) | set(handoff) | {"GLD"})
    partial = build_partial_defensive_exit_preview(
        configured_cap=runtime_cap,
        deferred_symbols=candidate_syms,
        handoff_symbols=candidate_syms,
        refresh_sizing=True,
    )
    out["partial_defensive_exit_preview"] = partial
    out["partial_preview_available"] = bool(partial.get("partial_preview_available"))
    out["partial_preview_source"] = partial.get("partial_preview_source")
    out["partial_preview_symbol"] = partial.get("partial_preview_symbol")
    out["partial_preview_qty"] = partial.get("partial_preview_qty")
    out["partial_preview_notional"] = partial.get("partial_preview_notional")
    out["partial_preview_classification"] = partial.get("partial_preview_classification")
    out["full_exit_notional"] = partial.get("full_exit_notional")
    out["full_exit_deferred_reason"] = partial.get("full_exit_deferred_reason")
    out["partial_preview_requires_operator_approval"] = True
    out["partial_preview_analysis_only"] = True
    out["broker_orders_submitted"] = False

    if not partial.get("partial_preview_available"):
        return out

    # Inject into empty runtime capped preview (classic deadlock case).
    if not existing_symbols:
        enriched_runtime = apply_partial_preview_to_capped_block(
            runtime_preview, partial, batch_cap=runtime_cap
        )
        out["capped_preview_runtime_cap"] = enriched_runtime
        out["capped_preview_available"] = True
        if out.get("handoff_status") == "HANDOFF_READY_BUT_BATCH_NOTIONAL_BLOCKED":
            out["handoff_status"] = "HANDOFF_PARTIAL_PREVIEW_AVAILABLE"
    else:
        # Keep existing capped symbols; annotate deferred GLD full exit + partial availability.
        runtime_preview = dict(runtime_preview)
        runtime_preview["partial_defensive_exit_preview"] = True
        runtime_preview["preview_type"] = PREVIEW_TYPE
        runtime_preview["partial_preview_symbol"] = partial.get("partial_preview_symbol")
        runtime_preview["partial_preview_qty"] = partial.get("partial_preview_qty")
        runtime_preview["partial_preview_notional"] = partial.get("partial_preview_notional")
        deferred_syms = [str(s).upper() for s in (runtime_preview.get("deferred_symbols") or [])]
        sym = str(partial.get("partial_preview_symbol") or "GLD").upper()
        if sym not in deferred_syms:
            deferred_syms.append(sym)
        runtime_preview["deferred_symbols"] = deferred_syms
        runtime_preview["full_exit_deferred_reason"] = partial.get("full_exit_deferred_reason")
        runtime_preview["concentration_risk_note"] = (
            f"Capped batch includes {existing_symbols}; full {sym} exit deferred "
            f"({partial.get('full_exit_deferred_reason')}). Partial preview "
            f"{partial.get('partial_preview_scenario')} available for operator review."
        )
        out["capped_preview_runtime_cap"] = runtime_preview
        out["capped_preview_available"] = True

    primary = dict(out.get("capped_preview") or {})
    primary_cap = _safe_float(primary.get("batch_cap") or out.get("configured_batch_cap_effective"))
    partial_notional = _safe_float(partial.get("partial_preview_notional"))
    if not primary.get("capped_preview_symbols") and partial_notional <= primary_cap + 1e-6:
        out["capped_preview"] = apply_partial_preview_to_capped_block(
            primary, partial, batch_cap=primary_cap
        )
        out["capped_preview_available"] = True
    return out


def extract_partial_approval_contract_fields(
    preview_doc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalize partial defensive-exit fields for operator packet / approval / dry-run.
    """
    preview_doc = (
        preview_doc
        if preview_doc is not None
        else _read_json(RESULTS / "defensive_exit_capped_batch_preview.json")
    )
    nested = preview_doc.get("partial_defensive_exit_preview") or {}
    runtime = preview_doc.get("capped_preview_runtime_cap") or {}
    order = nested.get("partial_preview_order") or {}

    available = bool(
        preview_doc.get("partial_preview_available")
        or nested.get("partial_preview_available")
        or runtime.get("partial_defensive_exit_preview")
    )
    symbol = str(
        preview_doc.get("partial_preview_symbol")
        or nested.get("partial_preview_symbol")
        or runtime.get("partial_preview_symbol")
        or order.get("symbol")
        or ""
    ).upper()
    partial_qty = int(
        _safe_float(
            preview_doc.get("partial_preview_qty")
            or nested.get("partial_preview_qty")
            or runtime.get("partial_preview_qty")
            or order.get("qty")
        )
    )
    full_qty = int(
        _safe_float(
            order.get("full_position_qty")
            or (nested.get("partial_preview_order") or {}).get("full_position_qty")
        )
    )
    if full_qty <= 0 and symbol:
        held = _held_symbols()
        # Fall back to positions snapshot qty.
        try:
            import pandas as pd

            if POSITIONS_CSV.is_file():
                df = pd.read_csv(POSITIONS_CSV, low_memory=False)
                sym_col = "symbol" if "symbol" in df.columns else "ticker"
                row = df[df[sym_col].astype(str).str.upper() == symbol]
                if not row.empty:
                    full_qty = int(_safe_float(row.iloc[0].get("qty")))
        except Exception:
            pass
    remaining_qty = int(
        _safe_float(
            nested.get("remaining_qty")
            or order.get("remaining_qty")
            or max(0, full_qty - partial_qty)
        )
    )
    notional = round(
        _safe_float(
            preview_doc.get("partial_preview_notional")
            or nested.get("partial_preview_notional")
            or runtime.get("partial_preview_notional")
            or order.get("notional")
        ),
        2,
    )
    classification = str(
        preview_doc.get("partial_preview_classification")
        or nested.get("partial_preview_classification")
        or ""
    )
    scenario = str(nested.get("partial_preview_scenario") or order.get("scenario_name") or "")
    full_exit_notional = _safe_float(
        preview_doc.get("full_exit_notional") or nested.get("full_exit_notional")
    )
    full_exit_deferred_reason = str(
        preview_doc.get("full_exit_deferred_reason")
        or nested.get("full_exit_deferred_reason")
        or runtime.get("full_exit_deferred_reason")
        or ""
    )
    limit_price = _safe_float(order.get("limit_price"))

    if not available or not symbol or partial_qty <= 0:
        return {
            "partial_preview_available": False,
            "preview_type": None,
            "approved_action": None,
            "symbol": None,
            "partial_qty": 0,
            "full_qty": 0,
            "remaining_qty_after_preview": 0,
            "estimated_notional": 0.0,
            "full_exit_deferred": False,
            "full_exit_deferred_reason": None,
            "approved_symbols": [],
            "approved_quantities": {},
            "broker_orders_submitted": False,
        }

    return {
        "partial_preview_available": True,
        "preview_type": PREVIEW_TYPE,
        "approved_action": "PARTIAL_EXIT",
        "symbol": symbol,
        "partial_qty": partial_qty,
        "full_qty": full_qty,
        "remaining_qty_after_preview": remaining_qty,
        "estimated_notional": notional,
        "classification": classification,
        "scenario": scenario,
        "limit_price": limit_price,
        "full_exit_notional": full_exit_notional,
        "full_exit_deferred": True,
        "full_exit_deferred_reason": full_exit_deferred_reason
        or f"FULL_EXIT_EXCEEDS_CAP notional={full_exit_notional}",
        "approved_symbols": [symbol],
        "approved_quantities": {symbol: partial_qty},
        "partial_preview_requires_operator_approval": True,
        "partial_preview_analysis_only": True,
        "acknowledgement_partial_exit": True,
        "acknowledgement_full_exit_deferred": True,
        "acknowledgement_live_blocked": True,
        "acknowledgement_no_broker_authorization": True,
        "paper_only": True,
        "broker_orders_submitted": False,
        "eligible_for_paper_execution": False,
    }
