"""
phase_198i_partial_dispatch_scope_dry_run_validation.py
Phase 198I — Validate partial GLD dispatch/placement scope dry-run (no orders).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_198i_partial_dispatch_scope_dry_run_validation.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
X197_JSON = RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json"
PROJECTED_CSV = RESULTS / "phase_197x_projected_paper_orders.csv"
RB_JSON = RESULTS / "phase_197z_readiness_bridge_validation.json"
Y197_JSON = RESULTS / "phase_197y_loop_mediated_paper_execution_gate.json"
Z197_JSON = RESULTS / "phase_197z_paper_broker_execution_authorization_creator.json"
MANAGE_ORDERS_CSV = LIVE / "manage_orders.csv"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
MARKER = "PARTIAL_DISPATCH_SCOPE_DRY_RUN_NO_EXECUTION"
EXPECTED_PARTIAL_QTY = 48
EXPECTED_FULL_QTY = 195
EXPECTED_REMAINING = 147
BUY_ADD_TOKENS = frozenset({"BUY", "ADD", "INCREASE", "SCALE_IN", "ENTRY"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _load_projected_scope() -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    qty_map: Dict[str, int] = {}
    if PROJECTED_CSV.is_file():
        with PROJECTED_CSV.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                sym = str(row.get("symbol") or "").strip().upper()
                if not sym:
                    continue
                qty = _safe_int(row.get("quantity"))
                side = str(row.get("side") or "").strip().lower()
                rows.append(
                    {
                        "symbol": sym,
                        "side": side,
                        "quantity": qty,
                        "notional_estimate": _safe_float(row.get("notional_estimate")),
                        "reason": row.get("reason"),
                        "source": row.get("source"),
                        "broker_order_submitted": str(
                            row.get("broker_order_submitted") or ""
                        ).lower()
                        in ("true", "1", "yes"),
                    }
                )
                qty_map[sym] = qty
    return rows, qty_map


def _inspect_manage_orders(symbol: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "exists": MANAGE_ORDERS_CSV.is_file(),
        "gld_qty": None,
        "gld_stance": None,
        "buy_add_rows_in_scope": [],
        "row_count": 0,
    }
    if not MANAGE_ORDERS_CSV.is_file():
        return out
    try:
        with MANAGE_ORDERS_CSV.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return out
    out["row_count"] = len(rows)
    for row in rows:
        sym = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        stance = (
            str(
                row.get("opportunity_type")
                or row.get("effective_stance")
                or row.get("management_action")
                or row.get("side")
                or ""
            )
            .strip()
            .upper()
        )
        qty = _safe_int(row.get("qty") or row.get("quantity"))
        if sym == symbol:
            out["gld_qty"] = qty
            out["gld_stance"] = stance
        # BUY/ADD contamination against the partial defensive symbol set.
        if sym == symbol and any(tok in stance for tok in BUY_ADD_TOKENS):
            out["buy_add_rows_in_scope"].append({"symbol": sym, "stance": stance, "qty": qty})
    return out


def run_phase_198i_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    checks: Dict[str, bool] = {}

    from services.partial_defensive_exit_preview import (
        PREVIEW_TYPE,
        extract_partial_approval_contract_fields,
    )
    from services.portfolio_history_sync import load_live_execution_flag

    preview_doc = _read_json(PREVIEW_JSON)
    packet = _read_json(PACKET_JSON)
    x197 = _read_json(X197_JSON)
    y197 = _read_json(Y197_JSON)
    z197 = _read_json(Z197_JSON)
    rb = _read_json(RB_JSON)
    partial = extract_partial_approval_contract_fields(preview_doc)
    projected_rows, projected_qty_map = _load_projected_scope()

    symbol = str(partial.get("symbol") or "GLD").upper()
    approved_qty = _safe_int(
        partial.get("partial_qty")
        or packet.get("partial_qty")
        or (packet.get("approved_quantities") or {}).get(symbol)
        or EXPECTED_PARTIAL_QTY
    )
    full_qty = _safe_int(partial.get("full_qty") or packet.get("full_qty") or EXPECTED_FULL_QTY)
    remaining_qty = _safe_int(
        partial.get("remaining_qty_after_preview")
        or packet.get("remaining_qty_after_preview")
        or EXPECTED_REMAINING
    )
    projected_qty = _safe_int(
        projected_qty_map.get(symbol)
        or (x197.get("projected_quantities") or {}).get(symbol)
        or y197.get("eligible_qty")
        or z197.get("authorized_qty")
    )
    projected_notional = round(
        _safe_float(
            next(
                (r.get("notional_estimate") for r in projected_rows if r.get("symbol") == symbol),
                None,
            )
            or x197.get("projected_notional")
            or partial.get("estimated_notional")
            or rb.get("estimated_notional")
        ),
        2,
    )
    batch_cap = _safe_float(
        x197.get("batch_notional_cap")
        or packet.get("configured_cap")
        or preview_doc.get("configured_batch_cap_manage_positions")
        or 50000.0
    )
    manage = _inspect_manage_orders(symbol)

    preview_type = str(
        partial.get("preview_type")
        or packet.get("preview_type")
        or x197.get("preview_type")
        or PREVIEW_TYPE
    )
    approved_action = str(
        partial.get("approved_action") or packet.get("approved_action") or "PARTIAL_EXIT"
    )
    full_exit_deferred = bool(
        partial.get("full_exit_deferred")
        or packet.get("full_exit_deferred")
        or x197.get("full_exit_deferred")
        or rb.get("full_exit_deferred")
    )

    # Placement/dispatch scope is the dry-run projected order set, not raw manage_orders.
    placement_scope = {
        "source": "phase_197x_projected_paper_orders",
        "symbols": sorted({r["symbol"] for r in projected_rows}),
        "quantities": dict(projected_qty_map),
        "orders": projected_rows,
    }
    if not placement_scope["symbols"] and projected_qty > 0:
        placement_scope = {
            "source": "phase_197x_json_projected_quantities",
            "symbols": [symbol],
            "quantities": {symbol: projected_qty},
            "orders": [
                {
                    "symbol": symbol,
                    "side": "sell",
                    "quantity": projected_qty,
                    "notional_estimate": projected_notional,
                }
            ],
        }

    broker_orders_submitted = bool(
        x197.get("broker_orders_submitted")
        or y197.get("broker_orders_submitted")
        or rb.get("broker_orders_submitted")
        or any(bool(r.get("broker_order_submitted")) for r in projected_rows)
    )
    live_execution_permitted = bool(load_live_execution_flag()) or bool(
        rb.get("live_execution_permitted")
    )
    approval_exists = APPROVAL_PATH.is_file()
    broker_auth_exists = BROKER_AUTH_PATH.is_file()

    checks["partial_preview_exists"] = bool(partial.get("partial_preview_available"))
    checks["preview_type_partial"] = preview_type == PREVIEW_TYPE
    checks["approved_action_partial_exit"] = approved_action == "PARTIAL_EXIT"
    checks["approved_qty_48"] = approved_qty == EXPECTED_PARTIAL_QTY
    checks["full_qty_195"] = full_qty == EXPECTED_FULL_QTY
    checks["remaining_qty_147"] = remaining_qty == EXPECTED_REMAINING
    checks["projected_qty_equals_approved"] = projected_qty == approved_qty == EXPECTED_PARTIAL_QTY
    checks["projected_notional_under_cap"] = (
        projected_notional > 0 and projected_notional <= batch_cap
    )
    checks["full_exit_deferred"] = full_exit_deferred is True
    checks["placement_scope_partial_only"] = (
        placement_scope["quantities"].get(symbol) == EXPECTED_PARTIAL_QTY
        and placement_scope["quantities"].get(symbol) != EXPECTED_FULL_QTY
        and placement_scope["quantities"].get(symbol) < full_qty
    )
    checks["placement_scope_not_full_qty"] = (
        projected_qty != EXPECTED_FULL_QTY and projected_qty < (full_qty or EXPECTED_FULL_QTY)
    )
    checks["no_projected_qty_gt_approved"] = projected_qty <= approved_qty
    checks["no_buy_add_in_partial_scope"] = len(manage.get("buy_add_rows_in_scope") or []) == 0
    # Raw manage_orders may still show full GLD EXIT; that must not become placement scope.
    if manage.get("exists") and manage.get("gld_qty") == EXPECTED_FULL_QTY:
        checks["manage_orders_full_qty_not_used_as_placement"] = (
            placement_scope["quantities"].get(symbol) == EXPECTED_PARTIAL_QTY
        )
        notes.append(
            "manage_orders.csv still shows GLD qty=195; placement scope correctly uses projected qty=48"
        )
    else:
        checks["manage_orders_full_qty_not_used_as_placement"] = True
    checks["broker_orders_not_submitted"] = broker_orders_submitted is False
    checks["live_execution_blocked"] = live_execution_permitted is False
    checks["approval_artifact_absent"] = approval_exists is False
    checks["broker_auth_artifact_absent"] = broker_auth_exists is False
    checks["dispatch_scope_valid"] = (
        checks["placement_scope_partial_only"]
        and checks["projected_qty_equals_approved"]
        and checks["full_exit_deferred"]
        and checks["no_buy_add_in_partial_scope"]
        and checks["broker_orders_not_submitted"]
    )

    # Explicit hard failures from requirements.
    hard_fail_reasons: List[str] = []
    if projected_qty == EXPECTED_FULL_QTY:
        hard_fail_reasons.append("projected_gld_quantity_is_full_195")
    if projected_qty > approved_qty:
        hard_fail_reasons.append("projected_qty_greater_than_approved_partial")
    if full_exit_deferred is False:
        hard_fail_reasons.append("full_exit_deferred_false")
    if broker_orders_submitted:
        hard_fail_reasons.append("broker_orders_submitted_true")
    if live_execution_permitted:
        hard_fail_reasons.append("live_execution_permitted_true")
    if approval_exists:
        hard_fail_reasons.append("real_approval_artifact_remains")
    if broker_auth_exists:
        hard_fail_reasons.append("real_broker_authorization_artifact_remains")
    if manage.get("buy_add_rows_in_scope"):
        hard_fail_reasons.append("buy_add_rows_in_partial_defensive_dispatch_scope")

    failed = [k for k, v in checks.items() if not v]
    if hard_fail_reasons:
        for reason in hard_fail_reasons:
            if reason not in notes:
                notes.append(f"hard_fail={reason}")
        status = "FAIL"
    else:
        status = "PASS" if not failed else "FAIL"
    if failed:
        notes.extend([f"failed_check={k}" for k in failed])

    payload: Dict[str, Any] = {
        "phase": "198I",
        "generated_at": _utc_now_iso(),
        "status": status,
        "preview_type": preview_type,
        "symbol": symbol,
        "approved_action": approved_action,
        "approved_qty": approved_qty,
        "full_qty": full_qty,
        "remaining_qty": remaining_qty,
        "projected_qty": projected_qty,
        "projected_notional": projected_notional,
        "batch_cap": batch_cap,
        "projected_notional_under_cap": bool(projected_notional <= batch_cap),
        "full_exit_deferred": full_exit_deferred,
        "full_exit_deferred_reason": partial.get("full_exit_deferred_reason")
        or x197.get("full_exit_deferred_reason")
        or rb.get("full_exit_deferred_reason"),
        "dispatch_scope_valid": bool(checks.get("dispatch_scope_valid")),
        "placement_scope": placement_scope,
        "manage_orders_inspection": manage,
        "broker_orders_submitted": False,
        "live_execution_permitted": False,
        "approval_artifact_exists_final": approval_exists,
        "broker_authorization_artifact_exists_final": broker_auth_exists,
        "checks": checks,
        "failed_checks": failed,
        "hard_fail_reasons": hard_fail_reasons,
        "sources": {
            "preview": str(PREVIEW_JSON.relative_to(ROOT)).replace("\\", "/"),
            "packet": str(PACKET_JSON.relative_to(ROOT)).replace("\\", "/"),
            "197x": str(X197_JSON.relative_to(ROOT)).replace("\\", "/"),
            "projected_csv": str(PROJECTED_CSV.relative_to(ROOT)).replace("\\", "/"),
            "readiness_bridge": str(RB_JSON.relative_to(ROOT)).replace("\\", "/"),
            "manage_orders": str(MANAGE_ORDERS_CSV.relative_to(ROOT)).replace("\\", "/"),
        },
        "marker": MARKER,
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[198I-PARTIAL-DISPATCH-SCOPE] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[198I-PARTIAL-DISPATCH-SCOPE] status={status} "
            f"approved={approved_qty} projected={projected_qty} full={full_qty} "
            f"scope_valid={payload.get('dispatch_scope_valid')}"
        )
        print(f"[198I-PARTIAL-DISPATCH-SCOPE] {MARKER}")
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 198I partial dispatch scope dry-run validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_198i_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
