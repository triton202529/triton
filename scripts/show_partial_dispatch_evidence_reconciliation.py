#!/usr/bin/env python3
"""
show_partial_dispatch_evidence_reconciliation.py
Display Phase 198J partial dispatch quantity evidence reconciliation (read-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
VALIDATION_JSON = RESULTS / "phase_198j_partial_dispatch_evidence_reconciliation.json"
MARKER = "PARTIAL_DISPATCH_EVIDENCE_RECONCILED_NO_EXECUTION"


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
        print("  python -m services.phase_198j_partial_dispatch_evidence_reconciliation")
        return 1

    print("=== TRITON Phase 198J — Partial Dispatch Evidence Reconciliation ===")
    print("")
    print(f"  raw_handoff_qty:              {doc.get('raw_handoff_qty')}")
    print(f"  approved_partial_qty:         {doc.get('approved_partial_qty')}")
    print(f"  projected_dispatch_qty:       {doc.get('projected_dispatch_qty')}")
    print(f"  authoritative_dispatch_qty:   {doc.get('authoritative_dispatch_qty')}")
    print(f"  deferred_qty:                 {doc.get('deferred_qty')}")
    print(f"  full_position_qty:            {doc.get('full_position_qty')}")
    print(f"  reconciliation_classification:{doc.get('reconciliation_classification')}")
    print("")
    print("Operator warning")
    print(f"  {doc.get('operator_warning')}")
    print("")
    print(f"Validation status:              {doc.get('status')}")
    print("")
    print(MARKER)
    print("=== End report ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
