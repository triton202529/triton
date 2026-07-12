#!/usr/bin/env python3
"""
show_partial_gld_no_execution_certification.py
Display Phase 198K partial GLD no-execution certification (read-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
CERT_JSON = RESULTS / "phase_198k_partial_gld_no_execution_certification.json"
MARKER = "PARTIAL_GLD_NO_EXECUTION_CERTIFIED"


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def main() -> int:
    doc = _load(CERT_JSON)
    if not doc:
        print("Report missing. Run:")
        print("  python -m services.phase_198k_partial_gld_no_execution_certification")
        return 1

    print("=== TRITON Phase 198K — Partial GLD No-Execution Certification ===")
    print("")
    print(f"  certification_status:         {doc.get('certification_status')}")
    print(f"  certification_score:          {doc.get('certification_score')}")
    print(f"  certified_symbol:             {doc.get('certified_symbol')}")
    print(f"  full_qty:                     {doc.get('full_qty')}")
    print(f"  partial_qty:                  {doc.get('approved_partial_qty')}")
    print(f"  remaining_qty:                {doc.get('remaining_qty_after_partial')}")
    print(f"  authoritative_dispatch_qty:   {doc.get('authoritative_dispatch_qty')}")
    print(f"  raw_handoff_qty:              {doc.get('raw_handoff_qty')}")
    print(f"  execution_allowed_now:        {doc.get('execution_allowed_now')}")
    print("")
    print("Missing requirements for real execution")
    for item in doc.get("missing_requirements_for_real_execution") or []:
        print(f"  - {item}")
    print("")
    print(MARKER)
    print("=== End report ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
