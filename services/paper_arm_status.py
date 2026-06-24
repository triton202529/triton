"""
paper_arm_status.py
Read-only paper ARM / operator latch status (Phase 196Q).

Does not write paper_arm_confirm.json or alter execution/governance logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
RESULTS = ROOT / "data" / "results"
ARM_CONFIG_PATH = ROOT / "config" / "arm_mode.json"
PAPER_ARM_CONFIRM_PATH = LIVE / "paper_arm_confirm.json"
LIVE_CONFIRM_PATH = LIVE / "live_confirm.json"


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        from services.arm_mode import _parse_dt as arm_parse

        return arm_parse(value)
    except Exception:
        return None


def classify_paper_arm_confirm_status() -> str:
    """Return VALID | EXPIRED | MISSING | INVALID."""
    try:
        from services.arm_mode import (
            is_paper_arm_confirm_valid,
            load_arm_config,
            read_paper_arm_confirm,
        )

        doc = read_paper_arm_confirm()
        if doc is None:
            return "MISSING"
        if not bool(doc.get("allow_execute", True)):
            return "INVALID"
        if is_paper_arm_confirm_valid(load_arm_config()):
            return "VALID"
        return "EXPIRED"
    except Exception:
        return "MISSING"


def classify_live_confirm_status(*, ttl_minutes: int = 10) -> str:
    """Return VALID | EXPIRED | MISSING | INVALID for live_confirm.json."""
    if not LIVE_CONFIRM_PATH.is_file():
        return "MISSING"
    try:
        doc = json.loads(LIVE_CONFIRM_PATH.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            return "INVALID"
        confirmed = bool(doc.get("confirmed")) or bool(doc.get("ok"))
        cat = _parse_dt(doc.get("confirmed_at") or doc.get("ts") or doc.get("time"))
        if not confirmed or cat is None:
            return "INVALID"
        age_min = (datetime.now(timezone.utc) - cat).total_seconds() / 60.0
        return "VALID" if age_min <= ttl_minutes else "EXPIRED"
    except Exception:
        return "INVALID"


def build_paper_arm_status(*, verbose_notes: bool = True) -> Dict[str, Any]:
    """Read-only paper arm operator status payload."""
    notes: List[str] = []
    try:
        from services.arm_mode import (
            get_arm_mode,
            is_paper_arm_confirm_valid,
            load_arm_config,
            read_paper_arm_confirm,
            resolve_arm_mutation_permissions,
            validate_mode,
        )
        from services.portfolio_history_sync import load_live_execution_flag

        arm_cfg = load_arm_config()
        mode = validate_mode(arm_cfg.get("mode"))
        doc = read_paper_arm_confirm()
        status = classify_paper_arm_confirm_status()
        ttl = int(arm_cfg.get("confirmation_ttl_minutes", 30) or 30)
        operator_required = mode == "ASSISTED"

        minutes_remaining: Optional[float] = None
        if doc and status == "VALID":
            exp = _parse_dt(doc.get("expires_at"))
            if exp:
                minutes_remaining = max(
                    0.0, (exp - datetime.now(timezone.utc)).total_seconds() / 60.0
                )

        perms, blocks = resolve_arm_mutation_permissions(arm_cfg)
        paper_entry_armed = bool(
            mode != "ASSISTED" or (operator_required and is_paper_arm_confirm_valid(arm_cfg))
        ) and bool(perms.get("execute_trades"))

        live_execution = load_live_execution_flag()
        live_confirm_status = classify_live_confirm_status()

        if verbose_notes:
            notes.append("Read-only status; does not submit orders or write confirmation files.")
            if operator_required and status != "VALID":
                notes.append(
                    "ASSISTED mode: paper entry requires fresh paper_arm_confirm "
                    "(operator latch, distinct from live_confirm)."
                )
            if live_confirm_status != "VALID":
                notes.append(
                    "live_confirm.json is separate; expired/missing live confirm does not arm paper."
                )
            notes.append("live_execution remains blocked by policy.")

        return {
            "ok": True,
            "mode": mode,
            "arm_config_path": str(ARM_CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
            "paper_arm_confirm_path": str(PAPER_ARM_CONFIRM_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "exists": doc is not None,
            "status": status,
            "timestamp": str(doc.get("timestamp") or "") if doc else "",
            "expires_at": str(doc.get("expires_at") or "") if doc else "",
            "allow_execute": doc.get("allow_execute") if doc else None,
            "confirmation_ttl_minutes": ttl,
            "minutes_remaining": (
                round(minutes_remaining, 1) if minutes_remaining is not None else None
            ),
            "operator_required": operator_required,
            "paper_entry_armed": paper_entry_armed,
            "arm_permissions": perms,
            "arm_block_reasons": blocks,
            "live_execution": live_execution,
            "live_execution_blocked": not live_execution,
            "live_confirm_path": str(LIVE_CONFIRM_PATH.relative_to(ROOT)).replace("\\", "/"),
            "live_confirm_status": live_confirm_status,
            "live_confirm_distinct_from_paper_arm": True,
            "set_paper_arm_confirm_command": "python -m services.set_paper_arm_confirm --ttl <minutes>",
            "status_check_command": "python -m services.paper_arm_status",
            "notes": notes,
        }
    except Exception as exc:
        return {
            "ok": False,
            "mode": "UNKNOWN",
            "paper_arm_confirm_path": str(PAPER_ARM_CONFIRM_PATH),
            "exists": PAPER_ARM_CONFIRM_PATH.is_file(),
            "status": classify_paper_arm_confirm_status(),
            "operator_required": True,
            "paper_entry_armed": False,
            "live_execution": False,
            "notes": [f"status error: {exc}"],
        }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only paper ARM operator status")
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args(argv)
    payload = build_paper_arm_status()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"mode={payload.get('mode')} status={payload.get('status')}")
        print(f"paper_entry_armed={payload.get('paper_entry_armed')}")
        print(f"live_execution_blocked={payload.get('live_execution_blocked')}")
        print(f"path={payload.get('paper_arm_confirm_path')}")
        if payload.get("expires_at"):
            print(f"expires_at={payload.get('expires_at')}")
        if payload.get("minutes_remaining") is not None:
            print(f"minutes_remaining={payload.get('minutes_remaining')}")
        for n in payload.get("notes") or []:
            print(f"note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
