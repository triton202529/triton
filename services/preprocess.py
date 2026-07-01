"""
services/preprocess.py
Official entry point for raw OHLCV → data/processed/{TICKER}.parquet.

Delegates to run_preprocessing.main() so phase repair chains and operators can run:
  python -m services.preprocess
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from run_preprocessing import main


def run_preprocess() -> int:
    return int(main())


if __name__ == "__main__":
    raise SystemExit(run_preprocess())
