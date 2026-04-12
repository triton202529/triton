# services/guard_status.py
"""
Print guard_snapshot.json state in a friendly way.

Example:
  python services/guard_status.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GUARD_PATH = os.path.join(ROOT, "data", "results", "guard_snapshot.json")


def _safe_read(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8-sig") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}
    return {}


def _guard_is_active(snapshot: Dict[str, Any]) -> Tuple[bool, str]:
    blocked = bool(snapshot.get("blocked", False))
    kill = bool(snapshot.get("kill_switch", False))
    if blocked or kill:
        code = str(snapshot.get("code", "GUARD_BLOCKED"))
        msg = snapshot.get("message") or snapshot.get("reason") or "Guard snapshot active"
        return True, f"{code}: {msg}"
    return False, ""


def main() -> None:
    exists = os.path.exists(GUARD_PATH)
    size = os.path.getsize(GUARD_PATH) if exists else None
    snap = _safe_read(GUARD_PATH)

    print(f"[guard_status] path={GUARD_PATH}")
    print(f"[guard_status] exists={exists} size={size}")
    if not snap:
        print("[guard_status] snapshot is empty/invalid -> treated as NOT ACTIVE")
        return

    active, reason = _guard_is_active(snap)
    print(f"[guard_status] keys={sorted(list(snap.keys()))}")
    print(
        f"[guard_status] blocked={bool(snap.get('blocked', False))} kill_switch={bool(snap.get('kill_switch', False))}"
    )
    if snap.get("updated_at"):
        print(f"[guard_status] updated_at={snap.get('updated_at')}")
    if snap.get("code"):
        print(f"[guard_status] code={snap.get('code')}")
    if snap.get("message") or snap.get("reason"):
        print(f"[guard_status] message={snap.get('message') or snap.get('reason')}")

    if active:
        print(f"[guard_status] ACTIVE -> {reason}")
    else:
        print("[guard_status] NOT ACTIVE")


if __name__ == "__main__":
    main()
