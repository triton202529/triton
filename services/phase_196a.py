"""
phase_196a.py
Compatibility wrapper for operator-facing `python -m services.phase_196a` hints.
Delegates to phase_196a_monday_paper_readiness_hardening without changing readiness logic.
"""

from __future__ import annotations

import sys

from services.phase_196a_monday_paper_readiness_hardening import main

if __name__ == "__main__":
    sys.exit(main())
