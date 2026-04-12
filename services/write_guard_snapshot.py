# services/write_guard_snapshot.py
"""
Write Guard Snapshot artifact for TRITON Phase 1.5

Produces:
  data/results/guard_snapshot.json

This is intentionally lightweight.
Real guard logic (risk modes, capital locks, broker sync)
will be layered in later phases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone


# ──────────────────────────────
# Paths
# ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = RESULTS_DIR / "guard_snapshot.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    snapshot = {
        "timestamp": utc_now_iso(),
        "mode": "NORMAL",  # NORMAL | DEFENSIVE | LOCKDOWN (future)
        "reason": "All systems nominal",
        "buying_power": None,  # None until broker is wired here
        "reserve_pct": 0.20,  # Capital Preservation Doctrine default
    }

    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)

    print(f"🛡️  Guard snapshot written → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
