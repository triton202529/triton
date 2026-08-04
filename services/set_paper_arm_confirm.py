"""
Explicit operator PAPER arm create / renew / disarm CLI.

This is the sole supported writer of data/live/paper_arm_confirm.json for
normal operations. Runtime paths (market-open, freshness refresh, watchdog,
continuous loop) must validate the arm read-only and must not call write_confirm.

Arm:
  python -m services.set_paper_arm_confirm --ttl <minutes>

Disarm (allow_execute=false; does not change standing paper authorization):
  python -m services.set_paper_arm_confirm --ttl <minutes> --no-execute
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
CONFIRM_PATH = LIVE / "paper_arm_confirm.json"
MAX_SUPPORTED_TTL_MINUTES = 600


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_confirm(ttl_minutes: int, allow_execute: bool = True) -> Path:
    """Write a fixed-expiry paper arm latch (explicit operator use only)."""
    LIVE.mkdir(parents=True, exist_ok=True)
    ttl = max(1, int(ttl_minutes))
    if ttl > MAX_SUPPORTED_TTL_MINUTES:
        ttl = MAX_SUPPORTED_TTL_MINUTES
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=ttl)
    doc = {
        "timestamp": _utc_iso(),
        "mode": "ASSISTED",
        "expires_at": exp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "allow_execute": bool(allow_execute),
    }
    CONFIRM_PATH.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return CONFIRM_PATH


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Explicit operator create/renew/disarm for ASSISTED paper ARM. "
            "Runtime paths must not invoke this for auto-renewal."
        )
    )
    ap.add_argument(
        "--ttl",
        type=int,
        default=30,
        help=f"Minutes until fixed expiry (max {MAX_SUPPORTED_TTL_MINUTES})",
    )
    ap.add_argument(
        "--no-execute",
        action="store_true",
        help="Write allow_execute=false (disarm); does not alter standing paper authorization",
    )
    args = ap.parse_args(argv)
    p = write_confirm(args.ttl, allow_execute=not args.no_execute)
    print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
