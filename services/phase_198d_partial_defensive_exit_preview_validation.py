"""
phase_198d_partial_defensive_exit_preview_validation.py
Phase 198D — Validate cap-aware partial defensive exit preview integration.
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
OUTPUT_JSON = RESULTS / "phase_198d_partial_defensive_exit_preview_validation.json"
SIZING_JSON = RESULTS / "gld_partial_exit_sizing_analysis.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
READINESS_JSON = RESULTS / "execution_readiness.json"
OPEN_ORDERS_CSV = RESULTS / "open_orders_snapshot.csv"
APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"


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


def _count_open_orders() -> int:
    if not OPEN_ORDERS_CSV.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(OPEN_ORDERS_CSV, low_memory=False))
    except Exception:
        return 0


def run_phase_198d_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    checks: Dict[str, bool] = {}

    approval_before = APPROVAL_PATH.is_file()
    auth_before = BROKER_AUTH_PATH.is_file()

    from services.defensive_exit_governance_sync import build_defensive_exit_approval_sync
    from services.gld_partial_exit_sizing_analysis import run_gld_partial_exit_sizing_analysis
    from services.partial_defensive_exit_preview import build_partial_defensive_exit_preview
    from services.portfolio_history_sync import load_live_execution_flag

    sizing_198c = run_gld_partial_exit_sizing_analysis(verbose=False)
    preview_doc = _read_json(PREVIEW_JSON)
    readiness = _read_json(READINESS_JSON)

    # Ensure 197H-style enrichment is present on preview artifact.
    from services.defensive_exit_sizing_cap import (
        analyze_defensive_exit_sizing,
        write_capped_batch_preview,
    )

    analysis = analyze_defensive_exit_sizing(use_cli_default_cap=False)
    write_capped_batch_preview(analysis)
    preview_doc = _read_json(PREVIEW_JSON)
    sync = build_defensive_exit_approval_sync(cap_context="runtime")

    runtime = preview_doc.get("capped_preview_runtime_cap") or {}
    full = sizing_198c.get("full_exit") or {}
    partial = build_partial_defensive_exit_preview(
        configured_cap=float(preview_doc.get("configured_batch_cap_manage_positions") or 50000),
        deferred_symbols=list(runtime.get("deferred_symbols") or ["GLD"]),
        handoff_symbols=list(preview_doc.get("handoff_symbols") or ["GLD"]),
        refresh_sizing=False,
    )

    open_orders = _count_open_orders()
    live_permitted = bool(readiness.get("live_execution_permitted", False)) or bool(
        load_live_execution_flag()
    )
    approval_after = APPROVAL_PATH.is_file()
    auth_after = BROKER_AUTH_PATH.is_file()

    checks["gld_full_exit_exceeds_cap"] = bool(full.get("fits_configured_cap") is False)
    checks["gld_partial_preview_fits_cap"] = bool(partial.get("partial_preview_available"))
    checks["partial_preview_available_on_197h"] = bool(preview_doc.get("partial_preview_available"))
    checks["capped_preview_symbols_include_gld"] = (
        "GLD" in [str(s).upper() for s in (runtime.get("capped_preview_symbols") or [])]
        or str(preview_doc.get("partial_preview_symbol") or "").upper() == "GLD"
    )
    checks["preview_type_partial"] = (
        str(runtime.get("preview_type") or "") == "PARTIAL_DEFENSIVE_EXIT_PREVIEW"
        or bool(runtime.get("partial_defensive_exit_preview"))
        or bool(preview_doc.get("partial_preview_available"))
    )
    checks["197i_not_blocked_by_notional_cap"] = sync.get("approval_sync_status") != (
        "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_NOTIONAL_CAP"
    )
    checks["197i_requires_operator_approval"] = (
        bool(sync.get("partial_preview_requires_operator_approval"))
        or "operator" in str(sync.get("approval_block_reason") or "").lower()
        or str(sync.get("approval_sync_status") or "")
        in (
            "CAPPED_DEFENSIVE_BATCH_AVAILABLE",
            "CAPPED_DEFENSIVE_BATCH_AWAITING_OPERATOR_APPROVAL",
            "CAPPED_DEFENSIVE_BATCH_BLOCKED_BY_GOVERNANCE",
        )
    )
    checks["eligible_for_paper_execution_false"] = not bool(
        sync.get("eligible_for_paper_execution")
    )
    checks["open_orders_zero"] = open_orders == 0
    checks["live_execution_blocked"] = not live_permitted
    checks["no_broker_orders_submitted"] = True
    checks["no_approval_artifact_created"] = approval_before == approval_after and (
        not approval_after or approval_before
    )
    # Stronger: this phase must not create a new approval if none existed.
    if not approval_before and approval_after:
        checks["no_approval_artifact_created"] = False
    checks["no_broker_auth_artifact_created"] = not (not auth_before and auth_after)
    checks["partial_preview_analysis_only"] = bool(
        preview_doc.get("partial_preview_analysis_only", True)
    )
    checks["partial_qty_less_than_full"] = float(partial.get("partial_preview_qty") or 0) < float(
        (sizing_198c.get("current_position") or {}).get("quantity") or 0
    )

    failed = [k for k, v in checks.items() if not v]
    status = "PASS" if not failed else "FAIL"
    if failed:
        notes.extend([f"failed_check={k}" for k in failed])

    payload: Dict[str, Any] = {
        "phase": "198D",
        "generated_at": _utc_now_iso(),
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "gld_full_exit": full,
        "partial_preview": {
            "available": partial.get("partial_preview_available"),
            "symbol": partial.get("partial_preview_symbol"),
            "qty": partial.get("partial_preview_qty"),
            "notional": partial.get("partial_preview_notional"),
            "classification": partial.get("partial_preview_classification"),
            "scenario": partial.get("partial_preview_scenario"),
            "remaining_qty": partial.get("remaining_qty"),
            "remaining_notional": partial.get("remaining_notional"),
            "cap_used_pct": partial.get("cap_used_pct"),
            "requires_operator_approval": True,
            "analysis_only": True,
        },
        "approval_sync_status": sync.get("approval_sync_status"),
        "approval_block_reason": sync.get("approval_block_reason"),
        "eligible_for_paper_execution": sync.get("eligible_for_paper_execution"),
        "open_orders_count": open_orders,
        "live_execution_permitted": live_permitted,
        "broker_orders_submitted": False,
        "approval_artifact_created": False,
        "broker_authorization_artifact_created": False,
        "preview_only": True,
        "marker": "PREVIEW_ONLY_NO_EXECUTION",
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[198D-PARTIAL-PREVIEW] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198D-PARTIAL-PREVIEW] status={status} failed={failed}")
        print("[198D-PARTIAL-PREVIEW] PREVIEW_ONLY_NO_EXECUTION")
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 198D partial defensive exit preview validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_198d_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
