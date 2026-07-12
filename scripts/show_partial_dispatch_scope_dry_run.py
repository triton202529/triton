#!/usr/bin/env python3
"""
show_partial_dispatch_scope_dry_run.py
Display Phase 198I partial dispatch scope dry-run validation (read-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
VALIDATION_JSON = RESULTS / "phase_198i_partial_dispatch_scope_dry_run_validation.json"
MARKER = "PARTIAL_DISPATCH_SCOPE_DRY_RUN_NO_EXECUTION"


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def main() -> int:
    doc = _load(VALIDATION_JSON)
    if not doc:
        print("Report missing. Run:")
        print("  python -m services.phase_198i_partial_dispatch_scope_dry_run_validation")
        return 1

    print("=== TRITON Phase 198I — Partial Dispatch Scope Dry-Run ===")
    print("")
    print(f"  preview_type:                 {doc.get('preview_type')}")
    print(f"  symbol:                       {doc.get('symbol')}")
    print(f"  approved_quantity:            {doc.get('approved_qty')}")
    print(f"  projected_dispatch_quantity:  {doc.get('projected_qty')}")
    print(f"  full_quantity:                {doc.get('full_qty')}")
    print(f"  remaining_quantity:           {doc.get('remaining_qty')}")
    print(f"  full_exit_deferred:           {doc.get('full_exit_deferred')}")
    print(f"  dispatch_scope_valid:         {doc.get('dispatch_scope_valid')}")
    print(f"  projected_notional:           {doc.get('projected_notional')}")
    print(f"  broker_orders_submitted:      {doc.get('broker_orders_submitted')}")
    print("")
    print(f"Validation status:              {doc.get('status')}")
    print("")
    print(MARKER)
    print("=== End report ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
