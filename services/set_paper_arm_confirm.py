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
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
CONFIRM_PATH = LIVE / "paper_arm_confirm.json"
MAX_SUPPORTED_TTL_MINUTES = 600


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_confirm(path: Path, doc: dict) -> None:
    """Same schema bytes as the historical writer, with replace+flush+fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(doc, indent=2)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def write_confirm(
    ttl_minutes: int,
    allow_execute: bool = True,
    *,
    confirm_path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Path:
    """Write a fixed-expiry paper arm latch (explicit operator use only)."""
    dest = Path(confirm_path) if confirm_path is not None else CONFIRM_PATH
    ttl = max(1, int(ttl_minutes))
    if ttl > MAX_SUPPORTED_TTL_MINUTES:
        ttl = MAX_SUPPORTED_TTL_MINUTES
    clock = now if now is not None else datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    else:
        clock = clock.astimezone(timezone.utc)
    exp = clock + timedelta(minutes=ttl)
    ts = clock.strftime("%Y-%m-%dT%H:%M:%SZ") if now is not None else _utc_iso()
    doc = {
        "timestamp": ts,
        "mode": "ASSISTED",
        "expires_at": exp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "allow_execute": bool(allow_execute),
    }
    _atomic_write_confirm(dest, doc)
    return dest


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
