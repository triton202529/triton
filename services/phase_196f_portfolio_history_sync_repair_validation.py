"""
phase_196f_portfolio_history_sync_repair_validation.py
Phase 196F — Portfolio History dashboard sync repair validation.
"""

from __future__ import annotations

import sys

from services.portfolio_history_sync import run_phase_196f_report


def main() -> int:
    report = run_phase_196f_report(append_if_stale=True, verbose=True)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
