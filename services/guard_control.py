# services/guard_control.py
# ------------------------------------------------------------
# TRITON — Global Guard / Kill-Switch Controller
#
# Controls execution safety state via data/results/guard_snapshot.json
#
# Commands:
#   --status           Show current guard state
#   --kill --reason X  Activate kill-switch with reason
#   --block --reason X Block execution without full kill
#   --unblock          Clear block / kill-switch (NORMAL mode)
#
# Capital Preservation Doctrine:
#   This file is authoritative. If guard is blocked, nothing trades.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


RESULTS_DIR = Path("data") / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GUARD_PATH = RESULTS_DIR / "guard_snapshot.json"


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────
@dataclass
class GuardState:
    blocked: bool
    kill_switch: bool
    mode: str
    code: str
    message: str
    updated_at: str


def utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ─────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────
def load_guard() -> GuardState:
    if not GUARD_PATH.exists():
        return GuardState(
            blocked=False,
            kill_switch=False,
            mode="NORMAL",
            code="",
            message="All systems nominal",
            updated_at=utc_now_z(),
        )

    try:
        data = json.loads(GUARD_PATH.read_text(encoding="utf-8"))
        return GuardState(
            blocked=bool(data.get("blocked", False)),
            kill_switch=bool(data.get("kill_switch", False)),
            mode=str(data.get("mode", "NORMAL")),
            code=str(data.get("code", "")),
            message=str(data.get("message", "")),
            updated_at=str(data.get("updated_at", utc_now_z())),
        )
    except Exception:
        return GuardState(
            blocked=True,
            kill_switch=True,
            mode="HALT",
            code="CORRUPT_STATE",
            message="Guard snapshot unreadable",
            updated_at=utc_now_z(),
        )


def save_guard(state: GuardState) -> None:
    GUARD_PATH.write_text(
        json.dumps(asdict(state), indent=2),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────
def cmd_status() -> int:
    s = load_guard()
    print(f"[guard] path={GUARD_PATH}")
    print(
        f"[guard] blocked={s.blocked} kill_switch={s.kill_switch} " f"mode={s.mode} code={s.code}"
    )
    print(f"[guard] message={s.message}")
    print(f"[guard] updated_at={s.updated_at}")
    return 0


def cmd_kill(reason: Optional[str]) -> int:
    s = load_guard()

    s.blocked = True
    s.kill_switch = True
    s.mode = "HALT"
    s.code = "MANUAL_KILL"
    s.message = reason or "Manual kill-switch activated"
    s.updated_at = utc_now_z()

    save_guard(s)
    print("[guard] KILL-SWITCH ACTIVATED")
    return 0


def cmd_block(reason: Optional[str]) -> int:
    s = load_guard()

    s.blocked = True
    s.kill_switch = False
    s.mode = "BLOCKED"
    s.code = "MANUAL_BLOCK"
    s.message = reason or "Manual execution block"
    s.updated_at = utc_now_z()

    save_guard(s)
    print("[guard] BLOCKED")
    return 0


def cmd_unblock() -> int:
    s = load_guard()

    s.blocked = False
    s.kill_switch = False

    # Clean, auditable reset (do NOT leave fields blank)
    s.mode = "NORMAL"
    s.code = "CLEAR"
    s.message = "Cleared from dashboard"
    s.updated_at = utc_now_z()

    save_guard(s)
    print("[guard] UNBLOCKED")
    return 0


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--kill", action="store_true")
    ap.add_argument("--block", action="store_true")
    ap.add_argument("--unblock", action="store_true")
    ap.add_argument("--reason", type=str, default=None)

    args = ap.parse_args()

    if args.status:
        return cmd_status()
    if args.kill:
        return cmd_kill(args.reason)
    if args.block:
        return cmd_block(args.reason)
    if args.unblock:
        return cmd_unblock()

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
