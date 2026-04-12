from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Write data/live/live_confirm.json for execution gating."
    )
    ap.add_argument("--mode", default="paper", help="paper|live (informational tag).")
    ap.add_argument("--path", default="data/live/live_confirm.json", help="Confirm file path.")
    ap.add_argument("--note", default="manual confirm", help="Optional note.")
    args = ap.parse_args(argv)

    p = Path(args.path)
    p.parent.mkdir(parents=True, exist_ok=True)

    now = utc_iso()
    payload = {
        "confirmed": True,  # ✅ REQUIRED by live_gate.py
        "confirmed_at": now,  # ✅ REQUIRED by live_gate.py (or ts/time)
        "mode": str(args.mode),
        "note": str(args.note),
        "schema": "triton.live_confirm.v1",
    }

    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[write_live_confirm] OK -> {p} confirmed_at={now}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
