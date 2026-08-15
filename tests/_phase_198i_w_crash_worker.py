"""Subprocess crash-injection worker for Phase 198I-W SQLite recovery tests.

PAPER ONLY. No network. No real broker. Isolated db_path argument.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.paper_sell_arm_store import (  # noqa: E402
    consume_authority_insert_dispatch,
    consume_challenge_and_create_authority,
    persist_open_challenge,
    record_dispatch_outcome,
)


def _install_commit_crash() -> None:
    real = sqlite3.Connection.execute

    def wrapped(self, sql, *args, **kwargs):
        if str(sql).strip().upper() == "COMMIT":
            os._exit(17)
        return real(self, sql, *args, **kwargs)

    sqlite3.Connection.execute = wrapped  # type: ignore[method-assign]


def main() -> int:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    stage = payload["stage"]
    db = Path(payload["db_path"])
    now = datetime.fromisoformat(payload["now"])
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    challenge = payload.get("challenge") or {}
    authority = payload.get("authority") or {}
    requested = payload.get("requested") or []
    plan_hash = payload.get("plan_hash") or ""
    batch_hash = payload.get("batch_hash") or ""
    authority_id = payload.get("authority_id") or ""
    dispatch_id = payload.get("dispatch_id") or ""
    live_blocked = bool(payload.get("live_blocked", True))

    if stage == "crash_before_challenge_commit":
        _install_commit_crash()
        persist_open_challenge(db, challenge=challenge, now=now)
        return 0

    if stage == "crash_after_challenge_commit":
        persist_open_challenge(db, challenge=challenge, now=now)
        os._exit(18)

    if stage == "crash_before_authority_commit":
        persist_open_challenge(db, challenge=challenge, now=now)
        _install_commit_crash()
        consume_challenge_and_create_authority(
            db,
            challenge_id=str(challenge.get("challenge_id") or ""),
            readiness_plan_hash=str(challenge.get("plan_hash") or ""),
            readiness_batch_id=str(challenge.get("batch_id") or ""),
            readiness_batch_hash=batch_hash,
            authority=authority,
            now=now,
        )
        return 0

    if stage == "crash_after_authority_commit":
        persist_open_challenge(db, challenge=challenge, now=now)
        consume_challenge_and_create_authority(
            db,
            challenge_id=str(challenge.get("challenge_id") or ""),
            readiness_plan_hash=str(challenge.get("plan_hash") or ""),
            readiness_batch_id=str(challenge.get("batch_id") or ""),
            readiness_batch_hash=batch_hash,
            authority=authority,
            now=now,
        )
        os._exit(19)

    if stage == "crash_after_dispatch_before_broker":
        consume_authority_insert_dispatch(
            db,
            authority_id=authority_id,
            plan_hash=plan_hash,
            batch_hash=batch_hash,
            expected_scope=payload.get("expected_scope") or [],
            mode="paper",
            live_blocked=live_blocked,
            requested=requested,
            now=now,
        )
        os._exit(20)

    if stage == "crash_after_broker_before_outcome":
        os._exit(21)

    if stage == "outcome_unknown_after_broker":
        record_dispatch_outcome(
            db,
            dispatch_id=dispatch_id,
            outcome_state="OUTCOME_UNKNOWN",
            result_class="CRASH_AFTER_FAKE_BROKER",
            reconciliation_required=True,
            now=now,
        )
        return 0

    raise SystemExit(f"unknown_stage:{stage}")


if __name__ == "__main__":
    raise SystemExit(main())
