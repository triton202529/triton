# services/guard_set.py
"""
Guard Snapshot helper.

Writes data/results/guard_snapshot.json in a consistent way (UTF-8 no BOM),
so the "Guard Snapshot Gate - SUPREME" in place_live_orders.py behaves
predictably.

Examples:
  python services/guard_set.py --on --code RECONCILE_FREEZE --message "test"
  python services/guard_set.py --off
  python services/guard_set.py --on --kill --message "HARD STOP"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "data", "results")
GUARD_PATH = os.path.join(RESULTS_DIR, "guard_snapshot.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_read(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8-sig") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}
    return {}


def _write(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # UTF-8 *without* BOM
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Set/clear Triton guard_snapshot.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--on", action="store_true", help="Enable guard (block live).")
    g.add_argument("--off", action="store_true", help="Disable guard (allow live).")

    ap.add_argument("--blocked", action="store_true", help="Set blocked=true (default for --on).")
    ap.add_argument("--kill", action="store_true", help="Set kill_switch=true (default for --on).")
    ap.add_argument("--code", default=None, help="Code string (e.g., RECONCILE_FREEZE).")
    ap.add_argument("--message", default=None, help="Message/reason.")
    ap.add_argument("--reason", default=None, help="Alias for --message.")
    ap.add_argument(
        "--preserve-extra", action="store_true", help="Preserve existing 'extra' object if present."
    )
    ap.add_argument("--show", action="store_true", help="Print the resulting JSON after writing.")

    args = ap.parse_args()

    existing = _safe_read(GUARD_PATH)
    extra = {}
    if args.preserve_extra and isinstance(existing.get("extra"), dict):
        extra = existing.get("extra", {}) or {}

    if args.off:
        payload = {
            "updated_at": _now_iso(),
            "blocked": False,
            "kill_switch": False,
            "code": "CLEAR",
            "message": "cleared",
            "reason": "cleared",
            "extra": extra,
        }
        _write(GUARD_PATH, payload)
        if args.show:
            print(json.dumps(payload, indent=2))
        print(f"[guard_set] CLEARED -> {GUARD_PATH}")
        return

    # --on path
    msg = args.message or args.reason or "manual guard block"
    code = args.code or "MANUAL_GUARD_BLOCK"

    blocked = True if (args.blocked or (not args.blocked and not args.kill)) else True
    kill = True if (args.kill or (not args.blocked and not args.kill)) else False

    payload = {
        "updated_at": _now_iso(),
        "blocked": bool(blocked),
        "kill_switch": bool(kill),
        "code": str(code),
        "message": str(msg),
        "reason": str(msg),
        "extra": extra,
    }
    _write(GUARD_PATH, payload)
    if args.show:
        print(json.dumps(payload, indent=2))
    print(
        f"[guard_set] SET blocked={payload['blocked']} kill_switch={payload['kill_switch']} code={payload['code']} -> {GUARD_PATH}"
    )


if __name__ == "__main__":
    main()
