"""
phase_198e_partial_approval_packet_compatibility_validation.py
Phase 198E — Validate partial defensive exit approval packet compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
OUTPUT_JSON = RESULTS / "phase_198e_partial_approval_packet_compatibility_validation.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
V197_JSON = RESULTS / "phase_197v_operator_approval_packet_review.json"
W197_JSON = RESULTS / "phase_197w_create_paper_operator_approval_artifact.json"
X197_JSON = RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
MARKER = "PARTIAL_APPROVAL_COMPATIBILITY_NO_EXECUTION"


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


def run_phase_198e_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    checks: Dict[str, bool] = {}

    approval_before = APPROVAL_PATH.is_file()
    auth_before = BROKER_AUTH_PATH.is_file()

    from services.partial_defensive_exit_preview import (
        PREVIEW_TYPE,
        extract_partial_approval_contract_fields,
    )
    from services.portfolio_history_sync import load_live_execution_flag

    preview_doc = _read_json(PREVIEW_JSON)
    partial = extract_partial_approval_contract_fields(preview_doc)
    v197 = _read_json(V197_JSON)
    w197 = _read_json(W197_JSON)
    x197 = _read_json(X197_JSON)

    symbol = str(partial.get("symbol") or "").upper()
    partial_qty = int(partial.get("partial_qty") or 0)
    full_qty = int(partial.get("full_qty") or 0)
    remaining = int(partial.get("remaining_qty_after_preview") or 0)

    checks["partial_preview_exists"] = bool(partial.get("partial_preview_available"))
    checks["preview_type_partial"] = str(partial.get("preview_type") or "") == PREVIEW_TYPE
    checks["review_represents_partial_gld"] = (
        str(v197.get("preview_type") or "") == PREVIEW_TYPE
        and str(v197.get("partial_preview_symbol") or "").upper() == "GLD"
        and str(v197.get("approved_action") or "") == "PARTIAL_EXIT"
        and int(v197.get("partial_qty") or 0) == partial_qty
        and int(v197.get("full_qty") or 0) == full_qty
        and int(v197.get("remaining_qty_after_preview") or 0) == remaining
    )
    checks["full_gld_exit_deferred"] = bool(
        v197.get("full_exit_deferred") and partial.get("full_exit_deferred")
    )
    checks["197w_dry_run_validated_partial_qty"] = bool(w197.get("dry_run_validated")) and (
        int((w197.get("approved_quantities") or {}).get("GLD") or 0) == partial_qty
        or int((w197.get("contract") or {}).get("approved_quantities", {}).get("GLD") or 0)
        == partial_qty
    )
    checks["197w_not_full_quantity"] = (
        int(
            (
                w197.get("approved_quantities")
                or (w197.get("contract") or {}).get("approved_quantities")
                or {}
            ).get("GLD")
            or 0
        )
        == partial_qty
        and partial_qty < full_qty
    )
    checks["197w_no_broker_authorization"] = not bool(
        w197.get("broker_authorization_artifact_created")
    ) and not bool((w197.get("contract") or {}).get("broker_execution_authorized"))
    checks["197x_projects_partial_only"] = (
        int(x197.get("projected_order_count") or 0) == 1
        and list(x197.get("projected_symbols") or []) == ["GLD"]
        and int((x197.get("projected_quantities") or {}).get("GLD") or 0) == partial_qty
        and partial_qty < full_qty
    )
    checks["197x_projected_notional_under_cap"] = bool(x197.get("projected_notional_under_cap"))
    checks["197x_full_exit_deferred"] = bool(x197.get("full_exit_deferred"))
    checks["197x_requires_broker_auth"] = bool(
        x197.get("requires_explicit_broker_execution_authorization")
    )
    checks["no_broker_orders_submitted"] = (
        v197.get("broker_orders_submitted") is False
        and w197.get("broker_orders_submitted") is False
        and x197.get("broker_orders_submitted") is False
    )

    approval_after = APPROVAL_PATH.is_file()
    auth_after = BROKER_AUTH_PATH.is_file()
    checks["no_approval_artifact_left"] = not approval_after
    checks["no_broker_auth_artifact"] = not auth_after
    checks["live_execution_blocked"] = not bool(load_live_execution_flag())
    checks["approval_absent_or_removed"] = (not approval_before and not approval_after) or (
        not approval_after
    )
    if approval_before and approval_after:
        checks["approval_absent_or_removed"] = False
        notes.append("approval_artifact_still_present")
    if auth_before or auth_after:
        notes.append("broker_auth_artifact_present")

    # Symbol/qty sanity for GLD partial scenario.
    checks["gld_partial_qty_48"] = symbol == "GLD" and partial_qty == 48
    checks["gld_full_qty_195"] = full_qty == 195
    checks["gld_remaining_147"] = remaining == 147

    failed = [k for k, v in checks.items() if not v]
    status = "PASS" if not failed else "FAIL"
    if failed:
        notes.extend([f"failed_check={k}" for k in failed])

    payload: Dict[str, Any] = {
        "phase": "198E",
        "generated_at": _utc_now_iso(),
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "partial_preview": {
            "available": partial.get("partial_preview_available"),
            "preview_type": partial.get("preview_type"),
            "symbol": symbol,
            "approved_action": partial.get("approved_action"),
            "partial_qty": partial_qty,
            "full_qty": full_qty,
            "remaining_qty_after_preview": remaining,
            "estimated_notional": partial.get("estimated_notional"),
            "full_exit_deferred": partial.get("full_exit_deferred"),
            "full_exit_deferred_reason": partial.get("full_exit_deferred_reason"),
        },
        "review_197v": {
            "status": v197.get("status"),
            "preview_type": v197.get("preview_type"),
            "symbol": v197.get("partial_preview_symbol"),
            "approved_action": v197.get("approved_action"),
            "partial_qty": v197.get("partial_qty"),
            "full_qty": v197.get("full_qty"),
            "remaining_qty_after_preview": v197.get("remaining_qty_after_preview"),
            "full_exit_deferred": v197.get("full_exit_deferred"),
            "operator_approval_required": v197.get("operator_approval_required"),
            "broker_orders_submitted": v197.get("broker_orders_submitted"),
        },
        "approval_197w": {
            "status": w197.get("status"),
            "dry_run": w197.get("dry_run"),
            "dry_run_validated": w197.get("dry_run_validated"),
            "partial_mode": w197.get("partial_mode"),
            "approved_symbols": w197.get("approved_symbols"),
            "approved_quantities": w197.get("approved_quantities")
            or (w197.get("contract") or {}).get("approved_quantities"),
            "approval_artifact_created": w197.get("approval_artifact_created"),
            "broker_authorization_artifact_created": w197.get(
                "broker_authorization_artifact_created"
            ),
            "broker_orders_submitted": w197.get("broker_orders_submitted"),
        },
        "dry_run_197x": {
            "status": x197.get("status"),
            "final_state": x197.get("final_state"),
            "projected_order_count": x197.get("projected_order_count"),
            "projected_symbols": x197.get("projected_symbols"),
            "projected_quantities": x197.get("projected_quantities"),
            "projected_notional": x197.get("projected_notional"),
            "projected_notional_under_cap": x197.get("projected_notional_under_cap"),
            "full_exit_deferred": x197.get("full_exit_deferred"),
            "broker_orders_submitted": x197.get("broker_orders_submitted"),
            "requires_explicit_broker_execution_authorization": x197.get(
                "requires_explicit_broker_execution_authorization"
            ),
            "using_dry_run_contract": x197.get("using_dry_run_contract"),
        },
        "approval_artifact_present": approval_after,
        "broker_authorization_artifact_present": auth_after,
        "live_execution_blocked": not bool(load_live_execution_flag()),
        "broker_orders_submitted": False,
        "marker": MARKER,
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[198E-PARTIAL-APPROVAL] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198E-PARTIAL-APPROVAL] status={status} failed={failed}")
        print(f"[198E-PARTIAL-APPROVAL] {MARKER}")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 198E partial approval packet compatibility validation"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_198e_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
