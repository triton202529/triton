"""
phase_198f_partial_broker_authorization_dry_run_compatibility.py
Phase 198F — Validate partial broker authorization dry-run compatibility.
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
OUTPUT_JSON = RESULTS / "phase_198f_partial_broker_authorization_dry_run_compatibility.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
X197_JSON = RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json"
Z197_JSON = RESULTS / "phase_197z_paper_broker_execution_authorization_creator.json"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"
MARKER = "PARTIAL_BROKER_AUTH_DRY_RUN_NO_EXECUTION"


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


def run_phase_198f_validation(*, verbose: bool = True) -> Dict[str, Any]:
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
    z197 = _read_json(Z197_JSON)

    symbol = str(partial.get("symbol") or "").upper()
    partial_qty = int(partial.get("partial_qty") or 0)
    full_qty = int(partial.get("full_qty") or 0)

    checks["partial_gld_preview_exists"] = bool(partial.get("partial_preview_available")) and (
        symbol == "GLD"
    )
    checks["197x_projects_gld_qty_48"] = (
        int(x197.get("projected_order_count") or 0) == 1
        and int((x197.get("projected_quantities") or {}).get("GLD") or 0) == 48
        and partial_qty == 48
    )
    checks["197z_dry_run_recognizes_qty_48"] = (
        z197.get("mode") == "dry_run"
        and int(z197.get("authorized_qty") or 0) == 48
        and int(z197.get("partial_authorization_qty") or 0) == 48
        and str(z197.get("symbol") or "").upper() == "GLD"
        and str(z197.get("preview_type") or "") == PREVIEW_TYPE
        and str(z197.get("authorized_action") or "") == "PARTIAL_EXIT"
    )
    checks["full_gld_exit_not_authorized"] = (
        z197.get("full_exit_authorized") is False
        and bool(z197.get("full_exit_deferred"))
        and int(z197.get("authorized_qty") or 0) < full_qty
        and int(z197.get("remaining_qty_after_authorized_preview") or 0) == 147
    )
    checks["broker_auth_would_be_created"] = bool(
        z197.get("broker_authorization_would_be_created")
    ) or bool(z197.get("would_create_authorization"))
    checks["no_real_broker_authorization_artifact"] = (
        not BROKER_AUTH_PATH.is_file()
        and z197.get("broker_authorization_created") is False
        and z197.get("real_authorization_created") is False
        and z197.get("authorization_artifact_exists") is False
    )
    checks["no_broker_orders_submitted"] = z197.get("broker_orders_submitted") is False
    checks["live_execution_blocked"] = (
        not bool(load_live_execution_flag())
        and z197.get("live_execution_permitted") is False
        and bool(z197.get("live_blocked"))
    )
    checks["approval_artifact_absent"] = not APPROVAL_PATH.is_file()
    checks["197z_status_pass_dry_run"] = z197.get("status") in ("PASS_DRY_RUN", "PASS")
    checks["estimated_notional_under_cap"] = bool(z197.get("estimated_notional_under_cap"))
    checks["partial_broker_authorization_supported"] = bool(
        z197.get("partial_broker_authorization_supported")
    )

    failed = [k for k, v in checks.items() if not v]
    status = "PASS" if not failed else "FAIL"
    if failed:
        notes.extend([f"failed_check={k}" for k in failed])

    payload: Dict[str, Any] = {
        "phase": "198F",
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
            "estimated_notional": partial.get("estimated_notional"),
            "full_exit_deferred": partial.get("full_exit_deferred"),
        },
        "dry_run_197x": {
            "status": x197.get("status"),
            "projected_order_count": x197.get("projected_order_count"),
            "projected_quantities": x197.get("projected_quantities"),
            "projected_notional": x197.get("projected_notional"),
            "full_exit_deferred": x197.get("full_exit_deferred"),
        },
        "creator_197z": {
            "status": z197.get("status"),
            "mode": z197.get("mode"),
            "final_state": z197.get("final_state"),
            "preview_type": z197.get("preview_type"),
            "symbol": z197.get("symbol"),
            "authorized_action": z197.get("authorized_action"),
            "authorized_qty": z197.get("authorized_qty"),
            "full_qty": z197.get("full_qty"),
            "remaining_qty_after_authorized_preview": z197.get(
                "remaining_qty_after_authorized_preview"
            ),
            "estimated_notional": z197.get("estimated_notional"),
            "full_exit_deferred": z197.get("full_exit_deferred"),
            "full_exit_authorized": z197.get("full_exit_authorized"),
            "broker_authorization_would_be_created": z197.get(
                "broker_authorization_would_be_created"
            ),
            "broker_authorization_created": z197.get("broker_authorization_created"),
            "broker_orders_submitted": z197.get("broker_orders_submitted"),
            "live_execution_permitted": z197.get("live_execution_permitted"),
            "partial_broker_authorization_supported": z197.get(
                "partial_broker_authorization_supported"
            ),
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
        print(f"[198F-PARTIAL-BROKER-AUTH] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198F-PARTIAL-BROKER-AUTH] status={status} failed={failed}")
        print(f"[198F-PARTIAL-BROKER-AUTH] {MARKER}")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 198F partial broker authorization dry-run compatibility"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_198f_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
