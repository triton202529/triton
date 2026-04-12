"""Write ASSISTED-mode paper execution confirmation (TTL-based)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
CONFIRM_PATH = LIVE / "paper_arm_confirm.json"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_confirm(ttl_minutes: int, allow_execute: bool = True) -> Path:
    LIVE.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=max(1, int(ttl_minutes)))
    doc = {
        "timestamp": _utc_iso(),
        "mode": "ASSISTED",
        "expires_at": exp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "allow_execute": bool(allow_execute),
    }
    CONFIRM_PATH.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return CONFIRM_PATH


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create paper ARM confirmation for ASSISTED mode")
    ap.add_argument("--ttl", type=int, default=30, help="Minutes until expiry")
    ap.add_argument("--no-execute", action="store_true", help="Write allow_execute=false")
    args = ap.parse_args(argv)
    p = write_confirm(args.ttl, allow_execute=not args.no_execute)
    print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
