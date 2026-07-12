"""
phase_198j_partial_dispatch_evidence_reconciliation.py
Phase 198J — Reconcile raw handoff vs approved/projected partial GLD dispatch quantities.
Evidence/operator clarity only. Does not submit orders or create approvals.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_198j_partial_dispatch_evidence_reconciliation.json"
OUTPUT_CSV = RESULTS / "partial_dispatch_quantity_reconciliation.csv"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
X197_JSON = RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json"
I198_JSON = RESULTS / "phase_198i_partial_dispatch_scope_dry_run_validation.json"
RB_JSON = RESULTS / "phase_197z_readiness_bridge_validation.json"
MANAGE_ORDERS_CSV = LIVE / "manage_orders.csv"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

MARKER = "PARTIAL_DISPATCH_EVIDENCE_RECONCILED_NO_EXECUTION"
OPERATOR_WARNING = (
    "Raw handoff quantity may represent the full defensive candidate. "
    "Broker dispatch must use the governed projected dispatch quantity only."
)
EXPECTED_PARTIAL_QTY = 48
EXPECTED_FULL_QTY = 195

RECONCILED_PARTIAL_SCOPE = "RECONCILED_PARTIAL_SCOPE"
RAW_HANDOFF_FULL_BUT_DISPATCH_PARTIAL = "RAW_HANDOFF_FULL_BUT_DISPATCH_PARTIAL"
BLOCKED_QUANTITY_MISMATCH = "BLOCKED_QUANTITY_MISMATCH"
BLOCKED_FULL_EXIT_EXPANSION = "BLOCKED_FULL_EXIT_EXPANSION"
BLOCKED_MISSING_PARTIAL_EVIDENCE = "BLOCKED_MISSING_PARTIAL_EVIDENCE"


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


def _raw_handoff_qty(symbol: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "exists": MANAGE_ORDERS_CSV.is_file(),
        "qty": None,
        "stance": None,
        "source": str(MANAGE_ORDERS_CSV.relative_to(ROOT)).replace("\\", "/"),
    }
    if not MANAGE_ORDERS_CSV.is_file():
        return out
    try:
        with MANAGE_ORDERS_CSV.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                sym = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
                if sym != symbol:
                    continue
                out["qty"] = _safe_int(row.get("qty") or row.get("quantity"))
                out["stance"] = str(
                    row.get("opportunity_type")
                    or row.get("effective_stance")
                    or row.get("management_action")
                    or ""
                ).upper()
                break
    except Exception:
        pass
    return out


def _classify(
    *,
    approved_partial_qty: int,
    projected_dispatch_qty: int,
    full_position_qty: int,
    raw_handoff_qty: Optional[int],
    full_exit_deferred: bool,
    partial_evidence_present: bool,
) -> str:
    if not partial_evidence_present:
        return BLOCKED_MISSING_PARTIAL_EVIDENCE
    if projected_dispatch_qty >= full_position_qty > 0:
        return BLOCKED_FULL_EXIT_EXPANSION
    if projected_dispatch_qty != approved_partial_qty or approved_partial_qty <= 0:
        return BLOCKED_QUANTITY_MISMATCH
    if not full_exit_deferred:
        return BLOCKED_QUANTITY_MISMATCH
    if (
        raw_handoff_qty is not None
        and raw_handoff_qty == full_position_qty
        and projected_dispatch_qty == approved_partial_qty
        and projected_dispatch_qty < full_position_qty
    ):
        return RAW_HANDOFF_FULL_BUT_DISPATCH_PARTIAL
    if (
        projected_dispatch_qty == approved_partial_qty
        and projected_dispatch_qty < full_position_qty
        and full_exit_deferred
    ):
        return RECONCILED_PARTIAL_SCOPE
    return BLOCKED_QUANTITY_MISMATCH


def run_phase_198j_reconciliation(*, verbose: bool = True) -> Dict[str, Any]:
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
    i198 = _read_json(I198_JSON)
    rb = _read_json(RB_JSON)
    partial = extract_partial_approval_contract_fields(preview_doc)

    symbol = str(
        partial.get("symbol")
        or i198.get("symbol")
        or x197.get("projected_symbols", [None])[0]
        or "GLD"
    ).upper()
    preview_type = str(
        partial.get("preview_type")
        or i198.get("preview_type")
        or packet.get("preview_type")
        or PREVIEW_TYPE
    )
    approved_partial_qty = _safe_int(
        i198.get("approved_qty")
        or partial.get("partial_qty")
        or packet.get("partial_qty")
        or (packet.get("approved_quantities") or {}).get(symbol)
        or EXPECTED_PARTIAL_QTY
    )
    projected_dispatch_qty = _safe_int(
        i198.get("projected_qty")
        or (x197.get("projected_quantities") or {}).get(symbol)
        or approved_partial_qty
    )
    full_position_qty = _safe_int(
        i198.get("full_qty")
        or partial.get("full_qty")
        or packet.get("full_qty")
        or EXPECTED_FULL_QTY
    )
    remaining_qty_after_partial = _safe_int(
        i198.get("remaining_qty")
        or partial.get("remaining_qty_after_preview")
        or packet.get("remaining_qty_after_preview")
        or max(0, full_position_qty - approved_partial_qty)
    )
    deferred_qty = remaining_qty_after_partial
    raw = _raw_handoff_qty(symbol)
    raw_handoff_qty = raw.get("qty")
    full_exit_deferred = bool(
        i198.get("full_exit_deferred")
        if "full_exit_deferred" in i198
        else (
            partial.get("full_exit_deferred")
            or packet.get("full_exit_deferred")
            or x197.get("full_exit_deferred")
            or rb.get("full_exit_deferred")
        )
    )
    full_exit_deferred_reason = (
        partial.get("full_exit_deferred_reason")
        or i198.get("full_exit_deferred_reason")
        or x197.get("full_exit_deferred_reason")
        or rb.get("full_exit_deferred_reason")
        or ""
    )
    partial_evidence_present = bool(
        partial.get("partial_preview_available")
        or i198.get("dispatch_scope_valid")
        or preview_type == PREVIEW_TYPE
    )

    authoritative_dispatch_qty = projected_dispatch_qty
    authoritative_dispatch_source = (
        "phase_197x_projected_paper_orders / phase_198i_partial_dispatch_scope"
    )
    raw_handoff_is_full_exit_candidate = bool(
        raw_handoff_qty is not None
        and raw_handoff_qty == full_position_qty
        and full_position_qty > 0
    )
    dispatch_scope_is_partial = bool(
        authoritative_dispatch_qty == approved_partial_qty
        and authoritative_dispatch_qty < full_position_qty
        and full_exit_deferred
    )

    classification = _classify(
        approved_partial_qty=approved_partial_qty,
        projected_dispatch_qty=projected_dispatch_qty,
        full_position_qty=full_position_qty,
        raw_handoff_qty=raw_handoff_qty if isinstance(raw_handoff_qty, int) else None,
        full_exit_deferred=full_exit_deferred,
        partial_evidence_present=partial_evidence_present,
    )

    broker_orders_submitted = bool(
        i198.get("broker_orders_submitted")
        or x197.get("broker_orders_submitted")
        or rb.get("broker_orders_submitted")
    )
    live_execution_permitted = bool(load_live_execution_flag()) or bool(
        rb.get("live_execution_permitted")
    )
    approval_exists = APPROVAL_PATH.is_file()
    broker_auth_exists = BROKER_AUTH_PATH.is_file()

    checks["projected_equals_approved"] = projected_dispatch_qty == approved_partial_qty
    checks["projected_lt_full"] = projected_dispatch_qty < full_position_qty
    checks["authoritative_dispatch_qty_48"] = authoritative_dispatch_qty == EXPECTED_PARTIAL_QTY
    checks["full_exit_deferred"] = full_exit_deferred is True
    checks["broker_orders_not_submitted"] = broker_orders_submitted is False
    checks["approval_absent"] = approval_exists is False
    checks["broker_auth_absent"] = broker_auth_exists is False
    checks["live_execution_blocked"] = live_execution_permitted is False
    checks["classification_passable"] = classification in (
        RECONCILED_PARTIAL_SCOPE,
        RAW_HANDOFF_FULL_BUT_DISPATCH_PARTIAL,
    )
    checks["dispatch_scope_is_partial"] = dispatch_scope_is_partial

    failed = [k for k, v in checks.items() if not v]
    status = (
        "PASS"
        if not failed
        and classification
        in (
            RECONCILED_PARTIAL_SCOPE,
            RAW_HANDOFF_FULL_BUT_DISPATCH_PARTIAL,
        )
        else "FAIL"
    )
    if failed:
        notes.extend([f"failed_check={k}" for k in failed])
    if classification.startswith("BLOCKED_"):
        notes.append(f"classification={classification}")
    if raw_handoff_is_full_exit_candidate:
        notes.append("raw_handoff_shows_full_gld_candidate_qty")

    row = {
        "symbol": symbol,
        "preview_type": preview_type,
        "raw_handoff_qty": raw_handoff_qty,
        "approved_partial_qty": approved_partial_qty,
        "projected_dispatch_qty": projected_dispatch_qty,
        "full_position_qty": full_position_qty,
        "remaining_qty_after_partial": remaining_qty_after_partial,
        "deferred_qty": deferred_qty,
        "authoritative_dispatch_qty": authoritative_dispatch_qty,
        "authoritative_dispatch_source": authoritative_dispatch_source,
        "full_exit_deferred": full_exit_deferred,
        "full_exit_deferred_reason": full_exit_deferred_reason,
        "raw_handoff_is_full_exit_candidate": raw_handoff_is_full_exit_candidate,
        "dispatch_scope_is_partial": dispatch_scope_is_partial,
        "reconciliation_classification": classification,
        "operator_warning": OPERATOR_WARNING,
    }

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    payload: Dict[str, Any] = {
        "phase": "198J",
        "generated_at": _utc_now_iso(),
        "status": status,
        "reconciliation_classification": classification,
        "symbol": symbol,
        "preview_type": preview_type,
        "raw_handoff_qty": raw_handoff_qty,
        "raw_handoff_source": raw.get("source"),
        "raw_handoff_stance": raw.get("stance"),
        "approved_partial_qty": approved_partial_qty,
        "projected_dispatch_qty": projected_dispatch_qty,
        "full_position_qty": full_position_qty,
        "remaining_qty_after_partial": remaining_qty_after_partial,
        "deferred_qty": deferred_qty,
        "authoritative_dispatch_qty": authoritative_dispatch_qty,
        "authoritative_dispatch_source": authoritative_dispatch_source,
        "full_exit_deferred": full_exit_deferred,
        "full_exit_deferred_reason": full_exit_deferred_reason,
        "raw_handoff_is_full_exit_candidate": raw_handoff_is_full_exit_candidate,
        "dispatch_scope_is_partial": dispatch_scope_is_partial,
        "operator_warning": OPERATOR_WARNING,
        "broker_orders_submitted": False,
        "live_execution_permitted": False,
        "approval_artifact_exists_final": approval_exists,
        "broker_authorization_artifact_exists_final": broker_auth_exists,
        "checks": checks,
        "failed_checks": failed,
        "csv_path": str(OUTPUT_CSV.relative_to(ROOT)).replace("\\", "/"),
        "sources": {
            "manage_orders": str(MANAGE_ORDERS_CSV.relative_to(ROOT)).replace("\\", "/"),
            "preview": str(PREVIEW_JSON.relative_to(ROOT)).replace("\\", "/"),
            "packet": str(PACKET_JSON.relative_to(ROOT)).replace("\\", "/"),
            "197x": str(X197_JSON.relative_to(ROOT)).replace("\\", "/"),
            "198i": str(I198_JSON.relative_to(ROOT)).replace("\\", "/"),
            "readiness_bridge": str(RB_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
        "marker": MARKER,
        "notes": notes,
        "row": row,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[198J-PARTIAL-RECON] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198J-PARTIAL-RECON] wrote {OUTPUT_CSV.relative_to(ROOT)}")
        print(
            f"[198J-PARTIAL-RECON] status={status} class={classification} "
            f"raw={raw_handoff_qty} approved={approved_partial_qty} "
            f"projected={projected_dispatch_qty} auth={authoritative_dispatch_qty}"
        )
        print(f"[198J-PARTIAL-RECON] {MARKER}")
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 198J partial dispatch evidence reconciliation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_198j_reconciliation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
