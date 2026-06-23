"""
phase_196g_heartbeat_status_display_hardening_validation.py
Phase 196G — Dashboard heartbeat and execution status display hardening.
"""

from __future__ import annotations

import sys

from services.dashboard_status_sync import run_phase_196g_report


def main() -> int:
    report = run_phase_196g_report(verbose=True)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
