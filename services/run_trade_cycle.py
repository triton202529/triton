# services/run_trade_cycle.py
"""Optional orchestrator: execute_trades (entries) then manage_positions (exits/trims). Does not modify schedulers."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run execute_trades then manage_positions (same mode/execute flags)."
    )
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Pass --execute to both stages (otherwise plan-only).",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ignore-market-closed", action="store_true")
    args = ap.parse_args(argv)

    exe = sys.executable
    base = [exe, "-m"]
    extra: list[str] = []
    if args.verbose:
        extra.append("--verbose")
    if args.ignore_market_closed:
        extra.append("--ignore-market-closed")

    cmd1 = (
        base
        + ["services.execute_trades", "--mode", args.mode]
        + (["--execute"] if args.execute else [])
        + extra
    )
    r1 = subprocess.call(cmd1, cwd=str(ROOT))
    cmd2 = (
        base
        + ["services.manage_positions", "--mode", args.mode]
        + (["--execute"] if args.execute else [])
        + extra
    )
    r2 = subprocess.call(cmd2, cwd=str(ROOT))

    if r1 != 0 or r2 != 0:
        return max(r1, r2) if max(r1, r2) != 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
