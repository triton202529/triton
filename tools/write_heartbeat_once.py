# tools/write_heartbeat_once.py
"""
One-time heartbeat writer.

Run from repo root:
  python tools/write_heartbeat_once.py

Fixes Windows/CLI import issues by bootstrapping project root onto sys.path.
"""

from __future__ import annotations

import os
import sys

# ─────────────────────────────────────────────────────────────
# Path bootstrap (REQUIRED when running as a script)
# ─────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.artifacts_writer import write_heartbeat, write_pipeline_status_fallback  # noqa: E402


def main() -> int:
    write_heartbeat(
        status="ok",
        stage="snapshot",
        last_success_stage="snapshot",
        message="Manual heartbeat (one-time). Pipeline health online.",
        error="",
    )

    # Optional fallback (harmless; helps older UI)
    write_pipeline_status_fallback(
        status="ok",
        stage="snapshot",
        message="Manual pipeline status (fallback).",
        error="",
    )

    print("✅ Wrote data/results/heartbeat.json (+ pipeline_status.json fallback)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
