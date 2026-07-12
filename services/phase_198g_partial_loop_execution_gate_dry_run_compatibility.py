"""
phase_198g_partial_loop_execution_gate_dry_run_compatibility.py
Phase 198G — Validate partial loop-mediated execution gate dry-run compatibility.
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
OUTPUT_JSON = RESULTS / "phase_198g_partial_loop_execution_gate_dry_run_compatibility.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
X197_JSON = RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json"
Y197_JSON = RESULTS / "phase_197y_loop_mediated_paper_execution_gate.json"
Z197_JSON = RESULTS / "phase_197z_paper_broker_execution_authorization_creator.json"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
MARKER = "PARTIAL_LOOP_GATE_DRY_RUN_NO_EXECUTION"


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


def run_phase_198g_validation(*, verbose: bool = True) -> Dict[str, Any]:
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

    symbol = str(partial.get("symbol") or "").upper()
    partial_qty = int(partial.get("partial_qty") or 0)
    full_qty = int(partial.get("full_qty") or 0)

    checks["partial_gld_preview_exists"] = bool(partial.get("partial_preview_available")) and (
        symbol == "GLD"
    )
    checks["197x_projects_qty_48"] = (
        int((x197.get("projected_quantities") or {}).get("GLD") or 0) == 48 and partial_qty == 48
    )
    checks["197z_dry_run_recognizes_qty_48"] = (
        int(z197.get("authorized_qty") or 0) == 48
        and z197.get("broker_authorization_created") is False
    )
    checks["197y_recognizes_partial_eligibility"] = (
        str(y197.get("preview_type") or "") == PREVIEW_TYPE
        and str(y197.get("symbol") or "").upper() == "GLD"
        and str(y197.get("eligible_action") or "") == "PARTIAL_EXIT"
        and int(y197.get("eligible_qty") or 0) == 48
        and int(y197.get("full_qty") or 0) == full_qty
        and int(y197.get("remaining_qty_after_preview") or 0) == 147
        and bool(y197.get("partial_loop_gate_supported"))
    )
    checks["full_gld_exit_deferred"] = bool(y197.get("full_exit_deferred")) and (
        int(y197.get("eligible_qty") or 0) < full_qty
    )
    checks["execution_blocked_without_artifacts"] = (
        y197.get("eligible_for_execution") is False
        and bool(y197.get("requires_operator_approval"))
        and bool(y197.get("requires_broker_authorization"))
        and y197.get("operator_approval_present") is False
        and y197.get("broker_authorization_present") is False
    )
    checks["no_broker_orders_submitted"] = y197.get("broker_orders_submitted") is False
    checks["no_approval_artifact"] = not APPROVAL_PATH.is_file() and (
        y197.get("approval_artifact_created") is False
    )
    checks["no_broker_auth_artifact"] = not BROKER_AUTH_PATH.is_file() and (
        y197.get("broker_authorization_artifact_created") is False
    )
    checks["live_execution_blocked"] = (
        not bool(load_live_execution_flag())
        and bool(y197.get("live_blocked"))
        and y197.get("live_execution_permitted") is False
    )
    checks["197y_safe_blocked"] = bool(y197.get("safe_blocked")) or str(
        y197.get("final_state") or ""
    ).startswith("PAPER_EXECUTION_GATE_BLOCKED")

    failed = [k for k, v in checks.items() if not v]
    status = "PASS" if not failed else "FAIL"
    if failed:
        notes.extend([f"failed_check={k}" for k in failed])

    payload: Dict[str, Any] = {
        "phase": "198G",
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
            "status": z197.get("status"),
            "authorized_qty": z197.get("authorized_qty"),
            "broker_authorization_created": z197.get("broker_authorization_created"),
        },
        "197y": {
            "status": y197.get("status"),
            "final_state": y197.get("final_state"),
            "preview_type": y197.get("preview_type"),
            "symbol": y197.get("symbol"),
            "eligible_action": y197.get("eligible_action"),
            "eligible_qty": y197.get("eligible_qty"),
            "full_qty": y197.get("full_qty"),
            "remaining_qty_after_preview": y197.get("remaining_qty_after_preview"),
            "full_exit_deferred": y197.get("full_exit_deferred"),
            "requires_operator_approval": y197.get("requires_operator_approval"),
            "requires_broker_authorization": y197.get("requires_broker_authorization"),
            "eligible_for_execution": y197.get("eligible_for_execution"),
            "broker_orders_submitted": y197.get("broker_orders_submitted"),
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
        print(f"[198G-PARTIAL-LOOP-GATE] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198G-PARTIAL-LOOP-GATE] status={status} failed={failed}")
        print(f"[198G-PARTIAL-LOOP-GATE] {MARKER}")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 198G partial loop execution gate dry-run compatibility"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_198g_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
