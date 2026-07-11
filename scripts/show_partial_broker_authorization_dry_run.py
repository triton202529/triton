#!/usr/bin/env python3
"""
show_partial_broker_authorization_dry_run.py
Display Phase 198F partial broker authorization dry-run compatibility (read-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
VALIDATION_JSON = RESULTS / "phase_198f_partial_broker_authorization_dry_run_compatibility.json"
Z197_JSON = RESULTS / "phase_197z_paper_broker_execution_authorization_creator.json"
X197_JSON = RESULTS / "phase_197x_paper_broker_execution_final_dry_run.json"
MARKER = "PARTIAL_BROKER_AUTH_DRY_RUN_NO_EXECUTION"


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
    z197 = _load(Z197_JSON)
    x197 = _load(X197_JSON)

    if not validation and not z197:
        print("Report missing. Run:")
        print("  python -m services.phase_198f_partial_broker_authorization_dry_run_compatibility")
        return 1

    partial = validation.get("partial_preview") or {}
    creator = validation.get("creator_197z") or z197

    print("=== TRITON Phase 198F — Partial Broker Authorization Dry-Run ===")
    print("")
    print(
        f"  preview_type:              {creator.get('preview_type') or partial.get('preview_type')}"
    )
    print(f"  symbol:                    {creator.get('symbol') or partial.get('symbol')}")
    print(
        f"  authorized_partial_qty:    {creator.get('authorized_qty') or partial.get('partial_qty')}"
    )
    print(f"  full_quantity:             {creator.get('full_qty') or partial.get('full_qty')}")
    print(
        "  remaining_quantity:        "
        f"{creator.get('remaining_qty_after_authorized_preview') or partial.get('remaining_qty_after_preview')}"
    )
    print("")
    print("Authorization dry-run status (197Z)")
    print(f"  status:                    {creator.get('status') or z197.get('status')}")
    print(f"  mode:                      {creator.get('mode') or z197.get('mode')}")
    print(f"  final_state:               {creator.get('final_state') or z197.get('final_state')}")
    print(f"  would_create:              {z197.get('broker_authorization_would_be_created')}")
    print(
        f"  estimated_notional:        {creator.get('estimated_notional') or z197.get('estimated_notional')}"
    )
    print(f"  full_exit_deferred:        {creator.get('full_exit_deferred')}")
    print("")
    print("197X projection")
    print(f"  projected_quantities:      {x197.get('projected_quantities')}")
    print("")
    print("  real broker authorization created: false")
    print(f"  broker_authorization_created: {z197.get('broker_authorization_created', False)}")
    print(f"  broker_orders_submitted:      {z197.get('broker_orders_submitted', False)}")
    print("")
    print(f"Validation status:           {validation.get('status')}")
    print("")
    print(MARKER)
    print("=== End report ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
