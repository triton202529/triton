#!/usr/bin/env python3
"""
show_partial_approval_packet_compatibility.py
Display Phase 198E partial defensive approval packet compatibility (read-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
VALIDATION_JSON = RESULTS / "phase_198e_partial_approval_packet_compatibility_validation.json"
V197_JSON = RESULTS / "phase_197v_operator_approval_packet_review.json"
W197_JSON = RESULTS / "phase_197w_create_paper_operator_approval_artifact.json"
X197_JSON = RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
MARKER = "PARTIAL_APPROVAL_COMPATIBILITY_NO_EXECUTION"


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def main() -> int:
    validation = _load(VALIDATION_JSON)
    v197 = _load(V197_JSON)
    w197 = _load(W197_JSON)
    x197 = _load(X197_JSON)
    preview = _load(PREVIEW_JSON)

    if not validation and not v197:
        print("Report missing. Run:")
        print("  python -m services.phase_198e_partial_approval_packet_compatibility_validation")
        return 1

    partial = validation.get("partial_preview") or {}
    if not partial.get("available"):
        nested = preview.get("partial_defensive_exit_preview") or {}
        partial = {
            "preview_type": v197.get("preview_type")
            or nested.get("partial_preview_type")
            or "PARTIAL_DEFENSIVE_EXIT_PREVIEW",
            "symbol": v197.get("partial_preview_symbol") or preview.get("partial_preview_symbol"),
            "partial_qty": v197.get("partial_qty") or preview.get("partial_preview_qty"),
            "full_qty": v197.get("full_qty"),
            "remaining_qty_after_preview": v197.get("remaining_qty_after_preview")
            or nested.get("remaining_qty"),
            "estimated_notional": v197.get("estimated_notional")
            or preview.get("partial_preview_notional"),
        }

    print("=== TRITON Phase 198E — Partial Approval Packet Compatibility ===")
    print("")
    print(f"  preview_type:          {partial.get('preview_type') or v197.get('preview_type')}")
    print(f"  symbol:                {partial.get('symbol') or v197.get('partial_preview_symbol')}")
    print(f"  partial_quantity:      {partial.get('partial_qty') or v197.get('partial_qty')}")
    print(f"  full_quantity:         {partial.get('full_qty') or v197.get('full_qty')}")
    print(
        f"  remaining_quantity:    {partial.get('remaining_qty_after_preview') or v197.get('remaining_qty_after_preview')}"
    )
    print(
        f"  projected_notional:    {x197.get('projected_notional') or partial.get('estimated_notional')}"
    )
    print("")
    print("Approval status (197W)")
    print(f"  status:                {w197.get('status')}")
    print(f"  dry_run:               {w197.get('dry_run')}")
    print(f"  dry_run_validated:     {w197.get('dry_run_validated')}")
    print(
        f"  approved_quantities:   {w197.get('approved_quantities') or (w197.get('contract') or {}).get('approved_quantities')}"
    )
    print(f"  artifact_created:      {w197.get('approval_artifact_created')}")
    print("")
    print("Dry-run status (197X)")
    print(f"  status:                {x197.get('status')}")
    print(f"  final_state:           {x197.get('final_state')}")
    print(f"  projected_order_count: {x197.get('projected_order_count')}")
    print(f"  projected_quantities:  {x197.get('projected_quantities')}")
    print(f"  full_exit_deferred:    {x197.get('full_exit_deferred')}")
    print(f"  broker_orders:         {x197.get('broker_orders_submitted')}")
    print("")
    print(f"Validation status:       {validation.get('status')}")
    print("")
    print(MARKER)
    print("=== End report ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
