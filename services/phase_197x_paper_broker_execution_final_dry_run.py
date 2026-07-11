"""
phase_197x_paper_broker_execution_final_dry_run.py
Phase 197X — Final paper broker execution dry-run (projected orders only, zero submissions).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json"
PROJECTED_ORDERS_CSV = RESULTS / "phase_197x_projected_paper_orders.csv"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
V197_JSON = RESULTS / "phase_197v_operator_approval_packet_review.json"
W197_JSON = RESULTS / "phase_197w_create_paper_operator_approval_artifact.json"
POSITIONS_CSV = RESULTS / "positions_snapshot.csv"
OPEN_ORDERS_CSV = RESULTS / "open_orders_snapshot.csv"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
PREVIOUS_COMMIT = "bdd044c6"
PREVIOUS_PHASE = "197W-FIX2"

READY = "PAPER_BROKER_EXECUTION_DRY_RUN_READY_OPERATOR_EXECUTION_REQUIRED"
BLOCKED_ABSENT = "PAPER_BROKER_EXECUTION_DRY_RUN_BLOCKED_APPROVAL_ABSENT"
BLOCKED_EXPIRED = "PAPER_BROKER_EXECUTION_DRY_RUN_BLOCKED_APPROVAL_EXPIRED"
BLOCKED_INVALID = "PAPER_BROKER_EXECUTION_DRY_RUN_BLOCKED_APPROVAL_INVALID"
BLOCKED_MARKET_CLOSED = "PAPER_BROKER_EXECUTION_DRY_RUN_BLOCKED_MARKET_CLOSED"
BLOCKED_OPEN_ORDERS = "PAPER_BROKER_EXECUTION_DRY_RUN_BLOCKED_OPEN_ORDERS"
BLOCKED_LIVE_NOT_BLOCKED = "PAPER_BROKER_EXECUTION_DRY_RUN_BLOCKED_LIVE_NOT_BLOCKED"
BLOCKED_BATCH_MISMATCH = "PAPER_BROKER_EXECUTION_DRY_RUN_BLOCKED_BATCH_MISMATCH"
BLOCKED_NO_ORDERS = "PAPER_BROKER_EXECUTION_DRY_RUN_BLOCKED_NO_PROJECTED_ORDERS"

SAFE_BLOCKED_STATES = frozenset(
    {
        BLOCKED_EXPIRED,
        BLOCKED_ABSENT,
        BLOCKED_INVALID,
        BLOCKED_MARKET_CLOSED,
        BLOCKED_OPEN_ORDERS,
        BLOCKED_LIVE_NOT_BLOCKED,
        BLOCKED_BATCH_MISMATCH,
        BLOCKED_NO_ORDERS,
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _norm_symbols(symbols: Any) -> List[str]:
    return sorted({str(s).strip().upper() for s in (symbols or []) if str(s).strip()})


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _runtime_capped_preview(doc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        doc.get("capped_preview_runtime_cap")
        or doc.get("capped_preview")
        or doc.get("capped_preview_cli_cap")
        or {}
    )


def _refresh_broker_snapshot() -> Tuple[bool, Dict[str, Any]]:
    proc = subprocess.run(
        [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    snapshot: Dict[str, Any] = {"exit_code": proc.returncode}
    jstart = (proc.stdout or "").find("{")
    if jstart >= 0:
        try:
            snapshot = json.loads(proc.stdout[jstart:])
        except Exception:
            pass
    return bool(snapshot.get("ok")), snapshot


def _load_positions() -> Dict[str, Dict[str, Any]]:
    positions: Dict[str, Dict[str, Any]] = {}
    if not POSITIONS_CSV.is_file():
        return positions
    try:
        import pandas as pd

        df = pd.read_csv(POSITIONS_CSV)
        sym_col = (
            "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
        )
        if not sym_col:
            return positions
        for _, row in df.iterrows():
            sym = str(row.get(sym_col) or "").strip().upper()
            if not sym:
                continue
            qty = float(row.get("qty") or row.get("qty_available") or 0)
            if qty <= 0:
                continue
            positions[sym] = {
                "symbol": sym,
                "qty": qty,
                "current_price": float(row.get("current_price") or 0),
                "market_value": float(row.get("market_value") or row.get("value") or 0),
            }
    except Exception:
        pass
    return positions


def _count_open_orders() -> int:
    if not OPEN_ORDERS_CSV.is_file():
        return 0
    try:
        import pandas as pd

        df = pd.read_csv(OPEN_ORDERS_CSV)
        return int(len(df))
    except Exception:
        return 0


def _build_projected_orders(
    *,
    approved_symbols: List[str],
    preview_block: Dict[str, Any],
    contract: Dict[str, Any],
    positions: Dict[str, Dict[str, Any]],
    preview_doc: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    notes: List[str] = []
    preview_orders = list(preview_block.get("capped_preview_orders") or [])
    preview_symbols = _norm_symbols(preview_block.get("capped_preview_symbols"))
    approved_quantities_raw = contract.get("approved_quantities") or {}
    approved_quantities: Dict[str, int] = {}
    for sym, qty in (approved_quantities_raw or {}).items():
        try:
            approved_quantities[str(sym).strip().upper()] = int(float(qty))
        except (TypeError, ValueError):
            notes.append(f"invalid_approved_quantity={sym}")

    is_partial = str(
        contract.get("preview_type") or ""
    ).upper() == "PARTIAL_DEFENSIVE_EXIT_PREVIEW" or bool(approved_quantities)

    if not is_partial and preview_symbols != approved_symbols:
        notes.append(
            f"preview_symbols_mismatch preview={preview_symbols} approved={approved_symbols}"
        )

    by_symbol = {str(o.get("symbol") or "").upper(): o for o in preview_orders}

    # Partial GLD may be deferred from capped orders; pull from nested partial preview order.
    if is_partial and preview_doc is not None:
        nested = preview_doc.get("partial_defensive_exit_preview") or {}
        partial_order = nested.get("partial_preview_order") or {}
        psym = str(partial_order.get("symbol") or "").upper()
        if psym and psym not in by_symbol:
            by_symbol[psym] = partial_order

    projected: List[Dict[str, Any]] = []
    packet_id = str(contract.get("packet_id") or contract.get("source_packet_id") or "")
    expires_at = str(contract.get("expires_at") or "")

    for sym in approved_symbols:
        pos = positions.get(sym)
        if not pos:
            notes.append(f"position_missing={sym}")
            continue
        order = by_symbol.get(sym)
        if not order and not (is_partial and sym in approved_quantities):
            notes.append(f"preview_order_missing={sym}")
            continue

        if is_partial and sym in approved_quantities:
            qty = int(approved_quantities[sym])
        else:
            qty = int((order or {}).get("qty") or pos.get("qty") or 0)

        # Never project full position when partial quantities are approved.
        if is_partial and sym in approved_quantities and qty != int(approved_quantities[sym]):
            qty = int(approved_quantities[sym])
        if qty <= 0:
            notes.append(f"non_positive_qty={sym}")
            continue
        held_qty = int(float(pos.get("qty") or 0))
        if is_partial and held_qty > 0 and qty >= held_qty:
            notes.append(f"partial_qty_not_less_than_full symbol={sym} qty={qty} held={held_qty}")
            continue

        limit_price = float((order or {}).get("limit_price") or pos.get("current_price") or 0)
        notional = float((order or {}).get("notional") or (qty * limit_price))
        if is_partial and sym in approved_quantities:
            notional = round(qty * limit_price, 2)
        reason = str(
            (order or {}).get("reason_code")
            or (order or {}).get("reason")
            or ("PARTIAL_EXIT_CAP_AWARE" if is_partial else "DEFENSIVE_EXIT")
        )
        projected.append(
            {
                "symbol": sym,
                "side": "sell",
                "quantity": qty,
                "notional_estimate": round(notional, 2),
                "reason": reason,
                "source": (
                    "partial_defensive_exit_preview"
                    if is_partial
                    else "defensive_exit_capped_batch_preview"
                ),
                "approval_packet_id": packet_id,
                "approval_expires_at": expires_at,
                "eligible_for_execution": False,
                "broker_order_submitted": False,
            }
        )
    return projected, notes


def _write_projected_csv(orders: List[Dict[str, Any]]) -> None:
    PROJECTED_ORDERS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "side",
        "quantity",
        "notional_estimate",
        "reason",
        "source",
        "approval_packet_id",
        "approval_expires_at",
        "eligible_for_execution",
        "broker_order_submitted",
    ]
    with PROJECTED_ORDERS_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in orders:
            writer.writerow(row)


def run_phase_197x_dry_run(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    projected_orders: List[Dict[str, Any]] = []

    snapshot_ok, snapshot = _refresh_broker_snapshot()
    open_orders = _count_open_orders()
    positions = _load_positions()
    positions_count = len(positions)

    from services.defensive_exit_governance_review_criteria import (
        evaluate_governance_review_criteria,
    )
    from services.guard_auto import is_market_open_et
    from services.paper_operator_approval_artifact_state import (
        PAPER_APPROVAL_ABSENT,
        PAPER_APPROVAL_EXPIRED,
        PAPER_APPROVAL_VALID_CURRENT,
        INVALID_STATES,
        classify_paper_operator_approval_artifact,
    )
    from services.partial_defensive_exit_preview import extract_partial_approval_contract_fields

    criteria = evaluate_governance_review_criteria(broker_snapshot_ok=snapshot_ok)
    live_blocked = bool(criteria.get("live_blocked"))
    market_open = bool(is_market_open_et(datetime.now(timezone.utc)))

    ap_state, ap_reason, ap_details = classify_paper_operator_approval_artifact(
        broker_snapshot_ok=snapshot_ok
    )
    contract = ap_details.get("contract") or _read_json(APPROVAL_PATH)
    approval_valid = ap_state == PAPER_APPROVAL_VALID_CURRENT
    w197 = _read_json(W197_JSON)
    preview_doc = _read_json(PREVIEW_JSON)
    partial_fields = extract_partial_approval_contract_fields(preview_doc)

    # Compatibility path: project from 197W dry-run validated partial contract when
    # no governance approval artifact is present (no broker auth / no execution).
    using_dry_run_contract = False
    if (
        ap_state == PAPER_APPROVAL_ABSENT
        and bool(w197.get("dry_run_validated"))
        and bool(w197.get("contract"))
    ):
        contract = dict(w197.get("contract") or {})
        using_dry_run_contract = True
        notes.append("projecting_from_197w_dry_run_contract_no_governance_artifact")

    approved_symbols = _norm_symbols(contract.get("approved_symbols"))
    deferred_symbols = _norm_symbols(contract.get("deferred_symbols"))
    approved_quantities_raw = contract.get("approved_quantities") or {}
    if not approved_quantities_raw and partial_fields.get("partial_preview_available"):
        approved_quantities_raw = partial_fields.get("approved_quantities") or {}
    projected_quantities: Dict[str, int] = {}
    for sym, qty in (approved_quantities_raw or {}).items():
        try:
            projected_quantities[str(sym).strip().upper()] = int(float(qty))
        except (TypeError, ValueError):
            pass

    final_state = READY
    block_reason = ""

    if ap_state == PAPER_APPROVAL_ABSENT and not using_dry_run_contract:
        final_state, block_reason = BLOCKED_ABSENT, ap_reason
    elif ap_state == PAPER_APPROVAL_EXPIRED:
        final_state, block_reason = BLOCKED_EXPIRED, ap_reason
    elif ap_state in INVALID_STATES:
        final_state, block_reason = BLOCKED_INVALID, ap_reason
    elif not live_blocked:
        final_state, block_reason = BLOCKED_LIVE_NOT_BLOCKED, "live_blocked=false"
    elif not market_open and not using_dry_run_contract:
        final_state, block_reason = BLOCKED_MARKET_CLOSED, "market_closed"
    elif open_orders != 0:
        final_state, block_reason = BLOCKED_OPEN_ORDERS, f"open_orders={open_orders}"
    elif positions_count == 0:
        final_state, block_reason = BLOCKED_BATCH_MISMATCH, "no_positions_in_snapshot"
    elif approval_valid or using_dry_run_contract:
        preview_block = _runtime_capped_preview(preview_doc)
        if using_dry_run_contract and not approved_symbols and partial_fields.get("symbol"):
            approved_symbols = _norm_symbols([partial_fields.get("symbol")])
            projected_quantities = dict(partial_fields.get("approved_quantities") or {})
            contract = dict(contract)
            contract["approved_symbols"] = approved_symbols
            contract["approved_quantities"] = projected_quantities
            contract["preview_type"] = partial_fields.get("preview_type")
        projected_orders, build_notes = _build_projected_orders(
            approved_symbols=approved_symbols,
            preview_block=preview_block,
            contract=contract,
            positions=positions,
            preview_doc=preview_doc,
        )
        notes.extend(build_notes)
        missing_positions = [s for s in approved_symbols if s not in positions]
        if missing_positions:
            final_state = BLOCKED_BATCH_MISMATCH
            block_reason = f"missing_positions={missing_positions}"
        elif len(projected_orders) != len(approved_symbols):
            final_state = BLOCKED_BATCH_MISMATCH
            block_reason = (
                f"projected_count={len(projected_orders)} approved_count={len(approved_symbols)}"
            )
        elif not projected_orders:
            final_state = BLOCKED_NO_ORDERS
            block_reason = "no_projected_orders"
        else:
            projected_quantities = {
                str(o.get("symbol")).upper(): int(o.get("quantity") or 0) for o in projected_orders
            }

    if final_state == READY and projected_orders:
        _write_projected_csv(projected_orders)
    else:
        if PROJECTED_ORDERS_CSV.is_file():
            try:
                PROJECTED_ORDERS_CSV.unlink()
            except Exception:
                pass

    status = "PASS"
    if final_state == BLOCKED_EXPIRED:
        status = "PASS_SAFE_BLOCKED"
    elif final_state != READY:
        status = "FAIL" if final_state == BLOCKED_LIVE_NOT_BLOCKED else "PASS_SAFE_BLOCKED"

    if not live_blocked:
        status = "FAIL"
        notes.append("live_block_regression")
    if open_orders != 0 and final_state != BLOCKED_OPEN_ORDERS:
        status = "FAIL"
        notes.append("open_orders_must_be_zero")
    if BROKER_AUTH_PATH.is_file():
        status = "FAIL"
        notes.append("broker_authorization_artifact_must_not_exist")

    batch_notional = round(sum(float(o.get("notional_estimate") or 0) for o in projected_orders), 2)
    batch_cap = float(
        contract.get("batch_notional_cap")
        or preview_doc.get("configured_batch_cap_manage_positions")
        or 50000.0
    )
    v197 = _read_json(V197_JSON)
    full_exit_deferred = bool(
        contract.get("full_exit_deferred")
        or partial_fields.get("full_exit_deferred")
        or str(contract.get("preview_type") or "").upper() == "PARTIAL_DEFENSIVE_EXIT_PREVIEW"
    )

    payload: Dict[str, Any] = {
        "phase": "197X",
        "generated_at": _utc_now_iso(),
        "status": status,
        "final_state": final_state,
        "block_reason": block_reason or None,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "approval_artifact_state": ap_state,
        "approval_artifact_reason": ap_reason,
        "approval_artifact_valid": approval_valid,
        "using_dry_run_contract": using_dry_run_contract,
        "approval_artifact_path": (
            str(APPROVAL_PATH.relative_to(ROOT)).replace("\\", "/")
            if APPROVAL_PATH.is_file()
            else None
        ),
        "approval_scope": contract.get("approval_scope"),
        "approval_packet_id": contract.get("packet_id") or contract.get("source_packet_id"),
        "approval_expires_at": contract.get("expires_at"),
        "preview_type": contract.get("preview_type") or partial_fields.get("preview_type"),
        "approved_symbols": approved_symbols,
        "deferred_symbols": deferred_symbols,
        "projected_symbols": [str(o.get("symbol")).upper() for o in projected_orders],
        "projected_quantities": projected_quantities,
        "batch_notional_estimate": batch_notional,
        "projected_notional": batch_notional,
        "batch_notional_cap": batch_cap,
        "projected_notional_under_cap": (
            bool(batch_notional <= batch_cap) if projected_orders else None
        ),
        "full_exit_deferred": full_exit_deferred,
        "full_exit_deferred_reason": contract.get("full_exit_deferred_reason")
        or partial_fields.get("full_exit_deferred_reason"),
        "projected_order_count": len(projected_orders),
        "projected_orders_path": (
            str(PROJECTED_ORDERS_CSV.relative_to(ROOT)).replace("\\", "/")
            if PROJECTED_ORDERS_CSV.is_file()
            else None
        ),
        "projected_orders": projected_orders,
        "positions_count": positions_count,
        "approved_symbols_with_positions": sorted([s for s in approved_symbols if s in positions]),
        "market_open": market_open,
        "live_blocked": live_blocked,
        "live_block_reason": criteria.get("live_block_reason"),
        "open_orders": open_orders,
        "broker_snapshot_ok": snapshot_ok,
        "broker_snapshot": {
            "open_orders_written": snapshot.get("open_orders_written"),
            "positions_written": snapshot.get("paths", {}),
        },
        "broker_orders_submitted": False,
        "broker_authorization_artifact_created": False,
        "requires_explicit_broker_execution_authorization": True,
        "exits_executed": 0,
        "trims_executed": 0,
        "operator_execution_required": True,
        "eligible_for_execution": False,
        "governance_review_state": criteria.get("governance_review_state"),
        "dispatch_interlock_note": (
            "197X projects orders only. Broker submission requires explicit operator execution phase."
        ),
        "operator_review_symbols": _norm_symbols(v197.get("symbols_to_exit")),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197X-DRY-RUN] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197X-DRY-RUN] status={status} final_state={final_state} "
            f"projected={len(projected_orders)} open_orders={open_orders} "
            f"broker_orders_submitted=False "
            f"projected_quantities={projected_quantities} "
            f"full_exit_deferred={full_exit_deferred}"
        )
    return payload


def get_approved_execution_scope_symbols() -> Tuple[List[str], str, bool]:
    """Expose 197W/197X approved execution scope for loop dispatch authority."""
    from services.defensive_exit_dispatch_interlock import load_paper_approved_execution_scope

    return load_paper_approved_execution_scope()


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197X paper broker execution final dry-run")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197x_dry_run(verbose=not args.quiet)
    return 0 if result.get("status") in ("PASS", "PASS_SAFE_BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
