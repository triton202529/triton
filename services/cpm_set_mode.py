# services/cpm_set_mode.py
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OVERRIDE_PATH = Path("data/results/cpm_override.json")

VALID = {"NORMAL", "DEFENSIVE", "CPM", "LOCKDOWN", "CLEAR"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python services/cpm_set_mode.py LOCKDOWN "reason"')
        print('  python services/cpm_set_mode.py DEFENSIVE "reason"')
        print("  python services/cpm_set_mode.py CLEAR")
        sys.exit(2)

    mode = sys.argv[1].upper().strip()
    if mode not in VALID:
        print(f"Invalid mode: {mode}. Valid: {sorted(VALID)}")
        sys.exit(2)

    OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if mode == "CLEAR":
        if OVERRIDE_PATH.exists():
            OVERRIDE_PATH.unlink()
        print(f"[{_utc_now_iso()}] CPM override CLEARED")
        return

    reason = " ".join(sys.argv[2:]).strip() if len(sys.argv) > 2 else ""
    payload = {
        "force_mode": mode,
        "reason": reason or "manual",
        "set_at": _utc_now_iso(),
    }

    OVERRIDE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[{_utc_now_iso()}] CPM override SET -> {mode}  reason={payload['reason']}")


if __name__ == "__main__":
    main()
