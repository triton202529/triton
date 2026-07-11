#!/usr/bin/env python3
"""
show_partial_defensive_exit_preview.py
Display Phase 198D cap-aware partial defensive exit preview (read-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
PREVIEW_JSON = RESULTS / "defensive_exit_capped_batch_preview.json"
VALIDATION_JSON = RESULTS / "phase_198d_partial_defensive_exit_preview_validation.json"
SIZING_JSON = RESULTS / "gld_partial_exit_sizing_analysis.json"


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def main() -> int:
    preview = _load(PREVIEW_JSON)
    validation = _load(VALIDATION_JSON)
    sizing = _load(SIZING_JSON)

    if not preview and not validation:
        print("Report missing. Run:")
        print("  python -m services.phase_198d_partial_defensive_exit_preview_validation")
        return 1

    full = sizing.get("full_exit") or validation.get("gld_full_exit") or {}
    partial = validation.get("partial_preview") or {}
    if not partial.get("available"):
        partial = {
            "available": preview.get("partial_preview_available"),
            "symbol": preview.get("partial_preview_symbol"),
            "qty": preview.get("partial_preview_qty"),
            "notional": preview.get("partial_preview_notional"),
            "classification": preview.get("partial_preview_classification"),
            "scenario": (preview.get("partial_defensive_exit_preview") or {}).get(
                "partial_preview_scenario"
            ),
            "remaining_qty": (preview.get("partial_defensive_exit_preview") or {}).get(
                "remaining_qty"
            ),
            "remaining_notional": (preview.get("partial_defensive_exit_preview") or {}).get(
                "remaining_notional"
            ),
            "cap_used_pct": (preview.get("partial_defensive_exit_preview") or {}).get(
                "cap_used_pct"
            ),
            "requires_operator_approval": preview.get("partial_preview_requires_operator_approval"),
        }

    print("=== TRITON Phase 198D — Partial Defensive Exit Preview ===")
    print("")
    print("Full GLD exit status")
    print(f"  estimated_notional:    {full.get('estimated_notional')}")
    print(f"  configured_cap:        {full.get('configured_cap')}")
    print(f"  fits_configured_cap:   {full.get('fits_configured_cap')}")
    print(f"  cap_excess_amount:     {full.get('cap_excess_amount')}")
    print(f"  classification:        {full.get('classification')}")
    print("")
    print("Selected partial scenario")
    print(f"  available:             {partial.get('available')}")
    print(f"  scenario:              {partial.get('scenario')}")
    print(f"  symbol:                {partial.get('symbol')}")
    print(f"  sell_qty:              {partial.get('qty')}")
    print(f"  estimated_notional:    {partial.get('notional')}")
    print(f"  classification:        {partial.get('classification')}")
    print(f"  cap_used_pct:          {partial.get('cap_used_pct')}")
    print("")
    print("Remaining GLD exposure after partial")
    print(f"  remaining_qty:         {partial.get('remaining_qty')}")
    print(f"  remaining_notional:    {partial.get('remaining_notional')}")
    print("")
    print("Operator approval")
    print(f"  required_later:        {bool(partial.get('requires_operator_approval', True))}")
    print(f"  eligible_for_execution:{validation.get('eligible_for_paper_execution', False)}")
    print("")
    print("PREVIEW_ONLY_NO_EXECUTION")
    print("=== End report ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
