#!/usr/bin/env python3
"""
show_partial_loop_execution_gate_dry_run.py
Display Phase 198G partial loop execution gate dry-run compatibility (read-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
VALIDATION_JSON = RESULTS / "phase_198g_partial_loop_execution_gate_dry_run_compatibility.json"
Y197_JSON = RESULTS / "phase_197y_loop_mediated_paper_execution_gate.json"
MARKER = "PARTIAL_LOOP_GATE_DRY_RUN_NO_EXECUTION"


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
    y197 = _load(Y197_JSON)
    gate = validation.get("197y") or y197

    if not validation and not y197:
        print("Report missing. Run:")
        print("  python -m services.phase_198g_partial_loop_execution_gate_dry_run_compatibility")
        return 1

    print("=== TRITON Phase 198G — Partial Loop Execution Gate Dry-Run ===")
    print("")
    print(f"  preview_type:                 {gate.get('preview_type') or y197.get('preview_type')}")
    print(f"  symbol:                       {gate.get('symbol') or y197.get('symbol')}")
    print(f"  eligible_partial_quantity:    {gate.get('eligible_qty') or y197.get('eligible_qty')}")
    print(f"  full_quantity:                {gate.get('full_qty') or y197.get('full_qty')}")
    print(
        "  remaining_quantity:           "
        f"{gate.get('remaining_qty_after_preview') or y197.get('remaining_qty_after_preview')}"
    )
    print("")
    print("Gate requirements")
    print(
        f"  operator_approval_required:   {gate.get('requires_operator_approval', y197.get('requires_operator_approval'))}"
    )
    print(
        f"  broker_authorization_required:{gate.get('requires_broker_authorization', y197.get('requires_broker_authorization'))}"
    )
    print(
        f"  eligible_for_execution:       {gate.get('eligible_for_execution', y197.get('eligible_for_execution'))}"
    )
    print(f"  final_state:                  {gate.get('final_state') or y197.get('final_state')}")
    print(
        f"  broker_orders_submitted:      {gate.get('broker_orders_submitted', y197.get('broker_orders_submitted'))}"
    )
    print("")
    print(f"Validation status:              {validation.get('status')}")
    print("")
    print(MARKER)
    print("=== End report ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
