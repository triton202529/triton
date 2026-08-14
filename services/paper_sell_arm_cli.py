"""
paper_sell_arm_cli.py
Phase 198H — Non-executing inspection / challenge / arm interface.

Defaults fail closed. Never writes live governance unless both:
  --store-root points at governance AND --explicit-runtime-write is set.
Phase 198H validation must not use --explicit-runtime-write.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper sell-arm authority (198H)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_insp = sub.add_parser("inspect", help="Read-only authority/challenge inspection")
    p_insp.add_argument("--store-root", type=Path, required=True)

    p_chal = sub.add_parser(
        "prepare-challenge",
        help="Arm Reviewed Sells — create confirmation challenge only (no authority)",
    )
    p_chal.add_argument("--store-root", type=Path, required=True)
    p_chal.add_argument("--readiness-json", type=Path, required=True)
    p_chal.add_argument("--account-raw", default="paper-test-account")
    p_chal.add_argument("--explicit-runtime-write", action="store_true")

    p_arm = sub.add_parser(
        "confirm-and-arm",
        help="Confirm & Arm Paper Sells — create authority only (no dispatch)",
    )
    p_arm.add_argument("--store-root", type=Path, required=True)
    p_arm.add_argument("--readiness-json", type=Path, required=True)
    p_arm.add_argument("--challenge-json", type=Path, required=True)
    p_arm.add_argument("--acknowledgement-file", type=Path, required=True)
    p_arm.add_argument("--current-plan-hash", default="")
    p_arm.add_argument("--newer-plan-exists", action="store_true")
    p_arm.add_argument("--explicit-runtime-write", action="store_true")

    p_val = sub.add_parser("validate-authority", help="Validate authority JSON (no writes)")
    p_val.add_argument("--authority-json", type=Path, required=True)

    args = parser.parse_args(argv)

    from services.paper_sell_arm_authority import (
        OPERATOR_ACKNOWLEDGEMENT_TEXT,
        confirm_and_create_paper_sell_arm,
        inspect_paper_sell_arm,
        prepare_confirmation_challenge,
        validate_paper_sell_arm_authority,
    )

    if args.cmd == "inspect":
        print(json.dumps(inspect_paper_sell_arm(store_root=args.store_root), indent=2))
        return 0

    if args.cmd == "prepare-challenge":
        if args.explicit_runtime_write:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "explicit_runtime_write_refused_in_default_cli_policy",
                        "hint": "Use programmatic API only after a dedicated operator phase.",
                    }
                )
            )
            return 2
        readiness = _load_json(args.readiness_json)
        try:
            chal = prepare_confirmation_challenge(
                store_root=args.store_root,
                readiness_packet=readiness,
                account_raw=args.account_raw,
                explicit_runtime_write=False,
            )
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}))
            return 2
        print(json.dumps({"ok": True, "challenge": chal, "authority_created": False}, indent=2))
        return 0

    if args.cmd == "confirm-and-arm":
        if args.explicit_runtime_write:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "explicit_runtime_write_refused_in_default_cli_policy",
                    }
                )
            )
            return 2
        readiness = _load_json(args.readiness_json)
        challenge = _load_json(args.challenge_json)
        ack = args.acknowledgement_file.read_text(encoding="utf-8").strip()
        if ack != OPERATOR_ACKNOWLEDGEMENT_TEXT:
            print(json.dumps({"ok": False, "error": "acknowledgement_mismatch"}))
            return 2
        try:
            authority = confirm_and_create_paper_sell_arm(
                store_root=args.store_root,
                readiness_packet=readiness,
                challenge=challenge,
                operator_acknowledgement_text=ack,
                current_plan_hash=args.current_plan_hash or None,
                newer_plan_exists=bool(args.newer_plan_exists),
                explicit_runtime_write=False,
            )
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}))
            return 2
        print(
            json.dumps(
                {
                    "ok": True,
                    "authority_created": True,
                    "orders_submitted": False,
                    "dispatch_performed": False,
                    "authority_status": authority.get("status"),
                    "audit_id": authority.get("audit_id"),
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "validate-authority":
        doc = _load_json(args.authority_json)
        ok, reason, details = validate_paper_sell_arm_authority(doc)
        print(json.dumps({"ok": ok, "reason": reason, "details": details}, indent=2))
        return 0 if ok else 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
