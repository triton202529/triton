#!/usr/bin/env python3
"""
show_gld_partial_exit_sizing_analysis.py
Display Phase 198C GLD partial exit sizing analysis (read-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT_JSON = RESULTS / "gld_partial_exit_sizing_analysis.json"


def main() -> int:
    if not OUTPUT_JSON.is_file():
        print("Report missing. Run:")
        print("  python -m services.gld_partial_exit_sizing_analysis")
        return 1

    doc = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    pos = doc.get("current_position") or {}
    full = doc.get("full_exit") or {}
    rec = doc.get("recommendation") or {}
    top = list(doc.get("top_feasible_partial_scenarios") or [])[:3]

    print("=== TRITON Phase 198C — GLD Partial Exit Sizing Analysis ===")
    print("")
    print("Current GLD exposure")
    print(f"  quantity:              {pos.get('quantity')}")
    print(f"  market_value:          {pos.get('market_value')}")
    print(f"  unrealized_pl:         {pos.get('unrealized_pl')}")
    print(f"  unrealized_pl_pct:     {pos.get('unrealized_pl_pct')}")
    print(f"  concentration_share:   {pos.get('concentration_share_pct')}%")
    print("")
    print("Full exit cap status")
    print(f"  estimated_limit_price: {full.get('estimated_limit_price')}")
    print(f"  estimated_notional:    {full.get('estimated_notional')}")
    print(f"  configured_cap:        {full.get('configured_cap')}")
    print(f"  fits_configured_cap:   {full.get('fits_configured_cap')}")
    print(f"  cap_excess_amount:     {full.get('cap_excess_amount')}")
    print(f"  classification:        {full.get('classification')}")
    print("")
    print("Top 3 feasible partial scenarios")
    if not top:
        print("  (none)")
    for i, s in enumerate(top, 1):
        print(
            f"  {i}. {s.get('scenario_name')}: sell_qty={s.get('sell_qty')} "
            f"notional={s.get('estimated_notional')} "
            f"cap_used={s.get('cap_used_pct')}% "
            f"class={s.get('classification')}"
        )
    print("")
    print("Recommended next review option")
    print(f"  scenario: {rec.get('recommended_scenario')}")
    print(f"  classification: {rec.get('classification')}")
    print(f"  sell_qty: {rec.get('sell_qty')}")
    print(f"  estimated_notional: {rec.get('estimated_notional')}")
    print(f"  next_review_option: {rec.get('next_review_option')}")
    print("")
    print("ANALYSIS_ONLY_NO_EXECUTION")
    print("=== End report ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
