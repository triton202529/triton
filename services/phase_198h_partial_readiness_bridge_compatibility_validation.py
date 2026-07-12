"""
phase_198h_partial_readiness_bridge_compatibility_validation.py
Phase 198H — Validate partial readiness bridge compatibility.
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
OUTPUT_JSON = RESULTS / "phase_198h_partial_readiness_bridge_compatibility_validation.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
X197_JSON = RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json"
Y197_JSON = RESULTS / "phase_197y_loop_mediated_paper_execution_gate.json"
Z197_JSON = RESULTS / "phase_197z_paper_broker_execution_authorization_creator.json"
RB_JSON = RESULTS / "phase_197z_readiness_bridge_validation.json"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
MARKER = "PARTIAL_READINESS_BRIDGE_NO_EXECUTION"


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


def run_phase_198h_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    checks: Dict[str, bool] = {}

    from services.partial_defensive_exit_preview import (
        PREVIEW_TYPE,
        extract_partial_approval_contract_fields,
    )
    from services.portfolio_history_sync import load_live_execution_flag

    preview_doc = _read_json(PREVIEW_JSON)
    partial = extract_partial_approval_contract_fields(preview_doc)
    x197 = _read_json(X197_JSON)
    y197 = _read_json(Y197_JSON)
    z197 = _read_json(Z197_JSON)
    rb = _read_json(RB_JSON)

    symbol = str(partial.get("symbol") or "").upper()
    partial_qty = int(partial.get("partial_qty") or 0)
    full_qty = int(partial.get("full_qty") or 0)

    checks["partial_gld_preview_exists"] = bool(partial.get("partial_preview_available")) and (
        symbol == "GLD"
    )
    checks["197x_projects_qty_48_only"] = (
        int((x197.get("projected_quantities") or {}).get("GLD") or 0) == 48
        and int(x197.get("projected_order_count") or 0) == 1
        and partial_qty == 48
    )
    checks["197z_dry_run_recognizes_qty_48"] = (
        int(z197.get("authorized_qty") or 0) == 48
        and z197.get("broker_authorization_created") is False
    )
    checks["197y_recognizes_partial_gate"] = (
        str(y197.get("preview_type") or "") == PREVIEW_TYPE
        and int(y197.get("eligible_qty") or 0) == 48
        and y197.get("eligible_for_execution") is False
    )
    checks["readiness_bridge_detects_partial_path"] = (
        bool(rb.get("partial_path_detected"))
        and str(rb.get("preview_type") or "") == PREVIEW_TYPE
        and str(rb.get("symbol") or "").upper() == "GLD"
        and str(rb.get("bridge_action") or "") == "PARTIAL_EXIT"
        and int(rb.get("bridge_qty") or 0) == 48
        and int(rb.get("full_qty") or 0) == full_qty
        and int(rb.get("remaining_qty_after_preview") or 0) == 147
    )
    checks["bridge_inactive_without_artifacts"] = (
        rb.get("bridge_active") is False
        and bool(rb.get("requires_operator_approval"))
        and bool(rb.get("requires_broker_authorization"))
    )
    checks["dispatch_blocked_approval_required"] = str(rb.get("dispatch") or "") in (
        "DISPATCH_BLOCKED_APPROVAL_REQUIRED",
        "DISPATCH_BLOCKED_BROKER_AUTH_REQUIRED",
    ) and (
        "approval" in str(rb.get("missing_requirement") or "").lower()
        or "operator" in str(rb.get("missing_requirement") or "").lower()
        or str(rb.get("dispatch") or "") == "DISPATCH_BLOCKED_APPROVAL_REQUIRED"
    )
    checks["full_gld_exit_deferred"] = bool(rb.get("full_exit_deferred")) and (
        int(rb.get("bridge_qty") or 0) < full_qty
    )
    checks["no_broker_orders_submitted"] = rb.get("broker_orders_submitted") is False
    checks["no_real_approval_artifact"] = not APPROVAL_PATH.is_file()
    checks["no_real_broker_auth_artifact"] = not BROKER_AUTH_PATH.is_file()
    checks["live_execution_blocked"] = (
        not bool(load_live_execution_flag())
        and bool(rb.get("live_blocked"))
        and rb.get("live_execution_permitted") is False
    )
    checks["rb_status_safe"] = rb.get("status") in ("PASS", "PASS_SAFE_BLOCKED")

    failed = [k for k, v in checks.items() if not v]
    status = "PASS" if not failed else "FAIL"
    if failed:
        notes.extend([f"failed_check={k}" for k in failed])

    payload: Dict[str, Any] = {
        "phase": "198H",
        "generated_at": _utc_now_iso(),
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "partial_preview": {
            "available": partial.get("partial_preview_available"),
            "preview_type": partial.get("preview_type"),
            "symbol": symbol,
            "partial_qty": partial_qty,
            "full_qty": full_qty,
            "remaining_qty_after_preview": partial.get("remaining_qty_after_preview"),
            "full_exit_deferred": partial.get("full_exit_deferred"),
        },
        "197x": {
            "projected_quantities": x197.get("projected_quantities"),
            "projected_order_count": x197.get("projected_order_count"),
        },
        "197z": {
            "authorized_qty": z197.get("authorized_qty"),
            "broker_authorization_created": z197.get("broker_authorization_created"),
        },
        "197y": {
            "preview_type": y197.get("preview_type"),
            "eligible_qty": y197.get("eligible_qty"),
            "eligible_for_execution": y197.get("eligible_for_execution"),
        },
        "readiness_bridge": {
            "status": rb.get("status"),
            "bridge_active": rb.get("bridge_active"),
            "partial_path_detected": rb.get("partial_path_detected"),
            "preview_type": rb.get("preview_type"),
            "symbol": rb.get("symbol"),
            "bridge_action": rb.get("bridge_action"),
            "bridge_qty": rb.get("bridge_qty"),
            "full_qty": rb.get("full_qty"),
            "remaining_qty_after_preview": rb.get("remaining_qty_after_preview"),
            "full_exit_deferred": rb.get("full_exit_deferred"),
            "requires_operator_approval": rb.get("requires_operator_approval"),
            "requires_broker_authorization": rb.get("requires_broker_authorization"),
            "missing_requirement": rb.get("missing_requirement"),
            "dispatch": rb.get("dispatch"),
            "broker_orders_submitted": rb.get("broker_orders_submitted"),
            "live_execution_permitted": rb.get("live_execution_permitted"),
        },
        "approval_artifact_present": APPROVAL_PATH.is_file(),
        "broker_authorization_artifact_present": BROKER_AUTH_PATH.is_file(),
        "live_execution_blocked": not bool(load_live_execution_flag()),
        "broker_orders_submitted": False,
        "marker": MARKER,
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[198H-PARTIAL-READINESS-BRIDGE] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198H-PARTIAL-READINESS-BRIDGE] status={status} failed={failed}")
        print(f"[198H-PARTIAL-READINESS-BRIDGE] {MARKER}")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 198H partial readiness bridge compatibility validation"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_198h_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
