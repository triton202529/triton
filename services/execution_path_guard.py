"""
execution_path_guard.py
THP-H003 — Safe exit for deprecated / archived execution paths.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

DEPRECATION_BANNER = """
WARNING:
This execution path is deprecated and not authorized for capital deployment.

TRITON institutional execution chain:
  Continuous Loop → Signal Pipeline → Lifecycle → Allocation
  → Risk Protection → Execution Gate → Governance Council
  → Broker Authorization → Alpaca

Use the approved workflow instead:
  python -m services.run_scheduled_paper_cycle
  python -m services.run_execution_loop
  python -m services.execute_trades --execute
"""


def normalize_execution_path(path: str | Path) -> str:
    """Return repo-relative path with forward slashes."""
    p = Path(path)
    if not p.is_absolute():
        return str(p).replace("\\", "/")
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _load_catalog() -> dict:
    from services.execution_path_registry import EXECUTION_PATHS

    return EXECUTION_PATHS


def get_path_entry(path: str | Path) -> Optional[dict]:
    rel = normalize_execution_path(path)
    return _load_catalog().get(rel)


def exit_if_deprecated(path: str | Path) -> None:
    """Exit safely when path is DEPRECATED or ARCHIVED."""
    rel = normalize_execution_path(path)
    entry = get_path_entry(rel)
    if not entry:
        return
    status = str(entry.get("status", "")).upper()
    if status not in ("DEPRECATED", "ARCHIVED"):
        return

    alt = entry.get("approved_alternative") or ("python -m services.run_scheduled_paper_cycle")
    print(DEPRECATION_BANNER.strip(), file=sys.stderr)
    print(f"\nBlocked path: {rel}", file=sys.stderr)
    print(f"Status: {status}", file=sys.stderr)
    print(f"Approved alternative: {alt}", file=sys.stderr)
    sys.exit(1)
