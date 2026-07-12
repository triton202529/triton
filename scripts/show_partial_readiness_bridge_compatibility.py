#!/usr/bin/env python3
"""
show_partial_readiness_bridge_compatibility.py
Display Phase 198H partial readiness bridge compatibility (read-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
VALIDATION_JSON = RESULTS / "phase_198h_partial_readiness_bridge_compatibility_validation.json"
RB_JSON = RESULTS / "phase_197z_readiness_bridge_validation.json"
MARKER = "PARTIAL_READINESS_BRIDGE_NO_EXECUTION"


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
    rb = _load(RB_JSON)
    bridge = validation.get("readiness_bridge") or rb

    if not validation and not rb:
        print("Report missing. Run:")
        print("  python -m services.phase_198h_partial_readiness_bridge_compatibility_validation")
        return 1

    print("=== TRITON Phase 198H — Partial Readiness Bridge Compatibility ===")
    print("")
    print(f"  preview_type:           {bridge.get('preview_type') or rb.get('preview_type')}")
    print(f"  symbol:                 {bridge.get('symbol') or rb.get('symbol')}")
    print(f"  bridge_quantity:        {bridge.get('bridge_qty') or rb.get('bridge_qty')}")
    print(f"  full_quantity:          {bridge.get('full_qty') or rb.get('full_qty')}")
    print(
        "  remaining_quantity:     "
        f"{bridge.get('remaining_qty_after_preview') or rb.get('remaining_qty_after_preview')}"
    )
    print("")
    print(f"  bridge_active:          {bridge.get('bridge_active', rb.get('bridge_active'))}")
    print(f"  dispatch:               {bridge.get('dispatch') or rb.get('dispatch')}")
    print(
        f"  missing_requirement:    {bridge.get('missing_requirement') or rb.get('missing_requirement')}"
    )
    print(
        f"  full_exit_deferred:     {bridge.get('full_exit_deferred', rb.get('full_exit_deferred'))}"
    )
    print(
        f"  broker_orders_submitted:{bridge.get('broker_orders_submitted', rb.get('broker_orders_submitted'))}"
    )
    print("")
    print(f"Validation status:        {validation.get('status')}")
    print("")
    print(MARKER)
    print("=== End report ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
