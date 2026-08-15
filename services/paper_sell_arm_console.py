"""Operator Console orchestration for paper SELL arm buttons.

Thin controller: public authority APIs for writes, store snapshot for reads.
Creates no dispatch, permit, or broker call. SQLite remains the authority source.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.live_block_classifier import evaluate_live_block_status
from services.paper_sell_arm_authority import (
    BUY_ACTIONS,
    LIVE_GOVERNANCE,
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    READINESS_REFERENCE_TTL_MINUTES,
    SELL_ACTIONS,
    _parse_iso,
    _scope_from_readiness,
    append_audit,
    classify_readiness_packet,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
    PaperSellArmPaths,
)
from services.paper_sell_arm_store import (
    PaperSellArmStoreError,
    batch_hash_from_scope,
    inspect_sqlite_arm_snapshot,
    production_path_overrides_present,
    resolve_runtime_sqlite_path,
)

CONFIRM_PAPER_SELLS_PHRASE = "CONFIRM PAPER SELLS"
READINESS_PACKET_NAME = "defensive_exit_approval_readiness.json"
ARM_SUCCESS = "Reviewed SELL challenge created. No orders were submitted."
CONFIRM_SUCCESS = "Paper SELL authority armed. No orders were submitted."
NO_SUBMIT_CAPTION = "Button clicks do not submit orders."

FAIL_CLOSED_MESSAGES = {
    "no_reviewed_sell_actions": "No reviewed SELL actions are available.",
    "stale_readiness_packet": "Reviewed SELL readiness packet is stale.",
    "malformed_packet": "Reviewed SELL readiness packet is malformed or unavailable.",
    "live_execution_not_blocked": "Live execution is not blocked. Paper SELL arming is refused.",
    "non_paper_mode": "Paper mode is required. Paper SELL arming is refused.",
    "sqlite_unavailable": "SQLite SELL store is unavailable.",
    "sqlite_pragma_or_schema_failure": "SQLite SELL store failed a PRAGMA, schema, or integrity check.",
    "sqlite_corrupt": "SQLite SELL store is corrupt or incompatible.",
    "challenge_missing": "No OPEN paper SELL confirmation challenge is available.",
    "challenge_expired": "The paper SELL confirmation challenge has expired.",
    "challenge_already_consumed": "The paper SELL confirmation challenge was already consumed.",
    "plan_mismatch": "Current reviewed plan does not match the confirmation challenge.",
    "batch_mismatch": "Current reviewed batch does not match the confirmation challenge.",
    "confirmation_phrase_mismatch": "Confirmation phrase does not match CONFIRM PAPER SELLS.",
    "duplicate_click": "An OPEN challenge already exists for this reviewed batch.",
    "authority_already_created": "Paper SELL authority already exists for this challenge.",
    "configuration_blocked": "Paper SELL arming is blocked by a non-production configuration setting.",
    "unknown_exception": "Paper SELL arming failed closed. No orders were submitted.",
}


def default_results_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "results"


def default_audit_root() -> Path:
    return LIVE_GOVERNANCE


def default_store_root() -> Path:
    """JSON audit export root. SQLite authority lives at canonical live path."""
    return default_audit_root()


def default_sqlite_db_path() -> Path:
    return resolve_runtime_sqlite_path()


def _production_override_block(
    *, db_path: Optional[Path], results_root: Optional[Path]
) -> Optional[str]:
    if db_path is not None:
        return None
    blocked, reason = production_path_overrides_present()
    if blocked:
        return reason
    return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def operator_fail_message(reason: str) -> str:
    return FAIL_CLOSED_MESSAGES.get(reason, FAIL_CLOSED_MESSAGES["unknown_exception"])


def _operator_message(reason: str) -> str:
    return operator_fail_message(reason)


def _sanitize_detail(text: str) -> str:
    raw = str(text or "")
    lowered = raw.replace("\\", "/").lower()
    if "c:/users/" in lowered or "/users/" in lowered or "traceback" in lowered:
        return FAIL_CLOSED_MESSAGES["unknown_exception"]
    return raw


def _map_store_reason(reason: str) -> str:
    r = str(reason or "")
    if r in (
        "sqlite_challenge_insert_conflict",
        "sqlite_duplicate_open_challenge",
    ):
        return "duplicate_click"
    if "pragma" in r or r in (
        "sqlite_integrity_check_failed",
        "sqlite_wal_required",
        "sqlite_synchronous_full_required",
    ):
        return "sqlite_pragma_or_schema_failure"
    if "corrupt" in r:
        return "sqlite_corrupt"
    if r.startswith("sqlite_"):
        return "sqlite_unavailable"
    return "unknown_exception"


def _map_arm_exception(exc: BaseException) -> str:
    msg = str(exc)
    if isinstance(exc, PermissionError):
        return "sqlite_unavailable"
    if isinstance(exc, PaperSellArmStoreError):
        return _map_store_reason(exc.reason)
    if isinstance(exc, ValueError):
        if "challenge_blocked:" in msg:
            tail = msg.split("challenge_blocked:", 1)[-1]
            return _map_store_reason(tail) if tail.startswith("sqlite_") else "malformed_packet"
        if "arm_rejected:" in msg:
            tail = msg.split("arm_rejected:", 1)[-1]
            mapping = {
                "acknowledgement_mismatch": "confirmation_phrase_mismatch",
                "challenge_expired": "challenge_expired",
                "challenge_plan_hash_mismatch": "plan_mismatch",
                "challenge_batch_id_mismatch": "batch_mismatch",
                "challenge_replay": "challenge_already_consumed",
                "existing_armed_authority_present": "authority_already_created",
                "sqlite_challenge_not_open": "challenge_already_consumed",
                "sqlite_challenge_not_found": "challenge_missing",
                "sqlite_challenge_expired": "challenge_expired",
                "sqlite_challenge_plan_hash_mismatch": "plan_mismatch",
                "sqlite_challenge_batch_hash_mismatch": "batch_mismatch",
                "sqlite_challenge_insert_conflict": "duplicate_click",
                "sqlite_duplicate_open_challenge": "duplicate_click",
                "sqlite_authority_insert_conflict": "authority_already_created",
            }
            for key, mapped in mapping.items():
                if key in tail:
                    return mapped
            if tail.startswith("sqlite_"):
                return _map_store_reason(tail)
            if "readiness" in tail:
                return "malformed_packet"
            return "unknown_exception"
    return "unknown_exception"


def _record_failure(store_root: Path, event: str, reason: str) -> None:
    try:
        paths = PaperSellArmPaths(root=store_root)
        append_audit(
            paths,
            {
                "event": event,
                "reason": reason,
                "source": "operator_console_sell_arm",
                "orders_submitted": False,
            },
        )
    except Exception:
        return


def load_reviewed_sell_packet(results_root: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    path = Path(results_root) / READINESS_PACKET_NAME
    if not path.is_file():
        return None, "malformed_packet"
    try:
        raw = path.read_text(encoding="utf-8")
        doc = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None, "malformed_packet"
    if not isinstance(doc, dict):
        return None, "malformed_packet"
    return doc, "ok"


def packet_freshness(
    packet: Dict[str, Any], *, now: datetime
) -> Tuple[bool, str, Optional[datetime]]:
    generated = _parse_iso(str(packet.get("generated_at") or packet.get("timestamp") or ""))
    if generated is None:
        return False, "malformed_packet", None
    age = now - generated
    if age > timedelta(minutes=READINESS_REFERENCE_TTL_MINUTES):
        return False, "stale_readiness_packet", generated
    return True, "ok", generated


def action_summaries(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in packet.get("approval_ready_actions") or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "symbol": str(row.get("symbol") or "").upper(),
                "action": str(row.get("action") or "").upper(),
                "quantity": row.get("qty"),
            }
        )
    return rows


def _sell_only_nonempty(packet: Dict[str, Any]) -> Tuple[bool, str]:
    actions = list(packet.get("approval_ready_actions") or [])
    if not actions:
        return False, "no_reviewed_sell_actions"
    for row in actions:
        action = str((row or {}).get("action") or "").upper()
        if action in BUY_ACTIONS:
            return False, "malformed_packet"
        if action not in SELL_ACTIONS:
            return False, "malformed_packet"
    scope = _scope_from_readiness(packet)
    if not scope:
        return False, "no_reviewed_sell_actions"
    return True, "ok"


def _paper_and_live_ok(
    packet: Dict[str, Any],
    *,
    paper_mode: Optional[bool],
    live_blocked: Optional[bool],
) -> Tuple[bool, str, bool, bool]:
    packet_paper = str(packet.get("mode") or "").upper() == "PAPER"
    if paper_mode is None:
        paper_mode = packet_paper
    else:
        paper_mode = bool(paper_mode) and packet_paper
    if live_blocked is None:
        live_blocked = bool(evaluate_live_block_status(paper_mode=True).get("live_blocked"))
    if not paper_mode:
        return False, "non_paper_mode", False, bool(live_blocked)
    if not live_blocked:
        return False, "live_execution_not_blocked", True, False
    return True, "ok", True, True


def sqlite_status(
    store_root: Path, *, now: datetime, db_path: Optional[Path] = None
) -> Dict[str, Any]:
    path = resolve_runtime_sqlite_path(db_path=db_path)
    snap = inspect_sqlite_arm_snapshot(path, now=now)
    if snap.get("db_present") and not snap.get("ok"):
        mapped = _map_store_reason(str(snap.get("reason") or ""))
        snap["operator_reason"] = mapped
        return snap
    return snap


def matching_open_challenge(
    snap: Dict[str, Any],
    *,
    plan_hash: str,
    batch_hash: str,
    now: datetime,
) -> Optional[Dict[str, Any]]:
    for rec in snap.get("open_challenges") or []:
        if str(rec.get("plan_hash")) != str(plan_hash):
            continue
        if str(rec.get("batch_hash")) != str(batch_hash):
            continue
        exp = _parse_iso(str(rec.get("expires_at") or ""))
        if exp is None or exp <= now:
            continue
        return rec
    return None


def build_presentation(
    *,
    store_root: Optional[Path] = None,
    results_root: Optional[Path] = None,
    now: Optional[datetime] = None,
    paper_mode: Optional[bool] = None,
    live_blocked: Optional[bool] = None,
    confirmation_phrase: str = "",
    session_challenge_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    now = now or _utc_now()
    blocked = _production_override_block(db_path=db_path, results_root=results_root)
    store_root = Path(store_root or default_store_root())
    results_root = Path(results_root or default_results_root())
    if blocked:
        empty_snap = inspect_sqlite_arm_snapshot(
            resolve_runtime_sqlite_path(db_path=db_path), now=now
        )
        return {
            "packet": None,
            "generated_at": None,
            "freshness": "unavailable",
            "action_count": 0,
            "actions": [],
            "challenge_id": None,
            "challenge_state": "ABSENT",
            "challenge_expires_at": None,
            "authority_state": "ABSENT",
            "authority_expires_at": None,
            "authority_single_use": None,
            "plan_hash": "",
            "batch_hash": "",
            "arm_enabled": False,
            "confirm_enabled": False,
            "arm_deny_reason": blocked,
            "confirm_deny_reason": blocked,
            "arm_duplicate": False,
            "snapshot": empty_snap,
            "paper_mode": paper_mode,
            "live_blocked": live_blocked,
            "no_submit_caption": NO_SUBMIT_CAPTION,
            "store_root": store_root,
            "results_root": results_root,
            "now": now,
            "resolved_db_path": str(resolve_runtime_sqlite_path(db_path=db_path)),
        }
    packet, load_reason = load_reviewed_sell_packet(results_root)
    snap = sqlite_status(store_root, now=now, db_path=db_path)
    deny_arm = None
    deny_confirm = None
    generated_at = None
    summaries: List[Dict[str, Any]] = []
    plan_hash = ""
    batch_hash = ""
    action_count = 0
    if packet is None:
        deny_arm = load_reason
        deny_confirm = load_reason
    else:
        summaries = action_summaries(packet)
        action_count = len(summaries)
        fresh, fresh_reason, generated_at = packet_freshness(packet, now=now)
        ok_cls, cls_reason = classify_readiness_packet(packet)
        sell_ok, sell_reason = _sell_only_nonempty(packet)
        mode_ok, mode_reason, paper_mode, live_blocked = _paper_and_live_ok(
            packet, paper_mode=paper_mode, live_blocked=live_blocked
        )
        if not fresh:
            deny_arm = fresh_reason
        elif not ok_cls:
            if cls_reason == "readiness_mode_not_paper":
                deny_arm = "non_paper_mode"
            elif cls_reason == "readiness_live_not_blocked":
                deny_arm = "live_execution_not_blocked"
            elif cls_reason == "readiness_has_no_approval_ready_actions":
                deny_arm = "no_reviewed_sell_actions"
            else:
                deny_arm = "malformed_packet"
        elif not sell_ok:
            deny_arm = sell_reason
        elif not mode_ok:
            deny_arm = mode_reason
        if deny_arm is None:
            plan_hash = str(packet.get("plan_hash") or "")
            batch_hash = batch_hash_from_scope(_scope_from_readiness(packet))
        deny_confirm = deny_arm
        if snap.get("db_present") and not snap.get("ok"):
            deny_arm = snap.get("operator_reason") or "sqlite_unavailable"
            deny_confirm = deny_arm

    challenge = None
    if packet is not None and plan_hash and batch_hash:
        challenge = matching_open_challenge(
            snap, plan_hash=plan_hash, batch_hash=batch_hash, now=now
        )
    if challenge is None and session_challenge_id:
        for rec in snap.get("open_challenges") or []:
            if rec.get("challenge_id") == session_challenge_id:
                challenge = rec
                break
        if challenge is None and snap.get("latest_challenge"):
            latest = snap["latest_challenge"]
            if latest.get("challenge_id") == session_challenge_id:
                challenge = latest

    challenge_state = None
    challenge_id = None
    challenge_expires = None
    if challenge:
        challenge_id = challenge.get("challenge_id")
        challenge_state = challenge.get("state")
        challenge_expires = challenge.get("expires_at")
        exp = _parse_iso(str(challenge_expires or ""))
        if challenge_state == "OPEN" and exp and exp <= now:
            deny_confirm = "challenge_expired"
            challenge_state = "EXPIRED"
        elif challenge_state == "CONSUMED":
            deny_confirm = "challenge_already_consumed"

    auth = snap.get("latest_authority") or {"state": "ABSENT"}
    confirm_enabled = False
    chal_plan = ""
    if challenge is not None:
        chal_plan = str(
            (challenge.get("payload") or {}).get("plan_hash") or challenge.get("plan_hash") or ""
        )
    if deny_confirm is None and not challenge_id:
        deny_confirm = "challenge_missing"
    elif deny_confirm is None and challenge is not None:
        if challenge_state == "CONSUMED":
            deny_confirm = "challenge_already_consumed"
        elif challenge_state != "OPEN":
            deny_confirm = "challenge_expired"
        elif chal_plan != plan_hash:
            deny_confirm = "plan_mismatch"
        elif str(challenge.get("batch_hash") or "") != batch_hash:
            deny_confirm = "batch_mismatch"
        elif confirmation_phrase and confirmation_phrase != CONFIRM_PAPER_SELLS_PHRASE:
            deny_confirm = "confirmation_phrase_mismatch"
    if (
        deny_confirm is None
        and challenge_id
        and challenge_state == "OPEN"
        and packet is not None
        and chal_plan == plan_hash
        and str(challenge.get("batch_hash") or "") == batch_hash
        and confirmation_phrase == CONFIRM_PAPER_SELLS_PHRASE
    ):
        confirm_enabled = True

    if deny_arm is None and matching_open_challenge(
        snap, plan_hash=plan_hash, batch_hash=batch_hash, now=now
    ):
        arm_enabled = False
        arm_duplicate = True
    else:
        arm_enabled = deny_arm is None
        arm_duplicate = False

    freshness_label = "unavailable"
    if generated_at is not None:
        freshness_label = (
            "fresh"
            if (now - generated_at) <= timedelta(minutes=READINESS_REFERENCE_TTL_MINUTES)
            else "stale"
        )

    return {
        "packet": packet,
        "generated_at": generated_at.isoformat() if generated_at else None,
        "freshness": freshness_label,
        "action_count": action_count,
        "actions": summaries,
        "challenge_id": challenge_id,
        "challenge_state": challenge_state
        or (
            (snap.get("latest_challenge") or {}).get("state")
            if snap.get("latest_challenge")
            else "ABSENT"
        ),
        "challenge_expires_at": challenge_expires
        or ((snap.get("latest_challenge") or {}).get("expires_at")),
        "authority_state": auth.get("state") or "ABSENT",
        "authority_expires_at": auth.get("expires_at"),
        "authority_single_use": auth.get("single_use"),
        "plan_hash": plan_hash,
        "batch_hash": batch_hash,
        "arm_enabled": arm_enabled,
        "confirm_enabled": confirm_enabled,
        "arm_deny_reason": deny_arm,
        "confirm_deny_reason": deny_confirm,
        "arm_duplicate": arm_duplicate,
        "snapshot": snap,
        "paper_mode": paper_mode,
        "live_blocked": live_blocked,
        "no_submit_caption": NO_SUBMIT_CAPTION,
        "store_root": store_root,
        "results_root": results_root,
        "now": now,
        "resolved_db_path": str(resolve_runtime_sqlite_path(db_path=db_path)),
    }


def arm_reviewed_sells(
    *,
    store_root: Optional[Path] = None,
    results_root: Optional[Path] = None,
    now: Optional[datetime] = None,
    paper_mode: Optional[bool] = None,
    live_blocked: Optional[bool] = None,
    account_raw: str = "paper-operator-console",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    now = now or _utc_now()
    counts = {"broker": 0, "network": 0, "permit": 0, "buy_add": 0}
    blocked = _production_override_block(db_path=db_path, results_root=results_root)
    if blocked:
        return _denied("arm", blocked, counts)
    store_root = Path(store_root or default_store_root())
    results_root = Path(results_root or default_results_root())
    packet, load_reason = load_reviewed_sell_packet(results_root)
    if packet is None:
        _record_failure(store_root, "sell_arm_button1_denied", load_reason)
        return _denied("arm", load_reason, counts)
    pres = build_presentation(
        store_root=store_root,
        results_root=results_root,
        now=now,
        paper_mode=paper_mode,
        live_blocked=live_blocked,
        db_path=db_path,
    )
    if pres.get("arm_duplicate"):
        _record_failure(store_root, "sell_arm_button1_denied", "duplicate_click")
        return _denied("arm", "duplicate_click", counts, snapshot=pres["snapshot"])
    if not pres.get("arm_enabled"):
        reason = str(pres.get("arm_deny_reason") or "unknown_exception")
        _record_failure(store_root, "sell_arm_button1_denied", reason)
        return _denied("arm", reason, counts, snapshot=pres["snapshot"])

    sqlite_path = resolve_runtime_sqlite_path(db_path=db_path)
    parent = sqlite_path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            _record_failure(store_root, "sell_arm_button1_denied", "sqlite_unavailable")
            return _denied("arm", "sqlite_unavailable", counts)

    try:
        challenge = prepare_confirmation_challenge(
            store_root=store_root,
            readiness_packet=packet,
            account_raw=account_raw,
            trading_date=now.date().isoformat(),
            now=now,
            explicit_runtime_write=True,
            db_path=sqlite_path,
        )
    except Exception as exc:
        reason = _map_arm_exception(exc)
        snap = sqlite_status(store_root, now=now, db_path=sqlite_path)
        open_match = matching_open_challenge(
            snap,
            plan_hash=str(packet.get("plan_hash") or ""),
            batch_hash=batch_hash_from_scope(_scope_from_readiness(packet)),
            now=now,
        )
        if not open_match:
            try:
                from services.paper_sell_arm_store import write_connection

                bhash = batch_hash_from_scope(_scope_from_readiness(packet))
                with write_connection(sqlite_path) as conn:
                    row = conn.execute(
                        """
                        SELECT challenge_id FROM sell_challenges
                        WHERE state='OPEN' AND plan_hash=? AND batch_hash=?
                        """,
                        (str(packet.get("plan_hash") or ""), bhash),
                    ).fetchone()
                if row:
                    open_match = {"challenge_id": row["challenge_id"], "state": "OPEN"}
            except Exception:
                open_match = None
        if open_match:
            reason = "duplicate_click"
        _record_failure(store_root, "sell_arm_button1_denied", reason)
        return _denied("arm", reason, counts, snapshot=snap)

    snap = sqlite_status(store_root, now=now, db_path=sqlite_path)
    open_match = matching_open_challenge(
        snap,
        plan_hash=str(packet.get("plan_hash") or ""),
        batch_hash=batch_hash_from_scope(_scope_from_readiness(packet)),
        now=now,
    )
    stored_open = bool(open_match) and str(open_match.get("state")) == "OPEN"
    if not stored_open:
        _record_failure(store_root, "sell_arm_button1_denied", "unknown_exception")
        return _denied("arm", "unknown_exception", counts, snapshot=snap)

    return {
        "ok": True,
        "button": "arm",
        "message": ARM_SUCCESS,
        "reason": None,
        "challenge_id": challenge.get("challenge_id"),
        "challenge_state": "OPEN",
        "challenge_expires_at": challenge.get("expires_at"),
        "action_count": len(action_summaries(packet)),
        "authority_count": snap.get("authority_count") or 0,
        "dispatch_count": snap.get("dispatch_count") or 0,
        "snapshot": snap,
        "counts": counts,
        "instruction": "Review the exact batch, then complete the second confirmation.",
        "resolved_db_path": str(sqlite_path),
    }


def confirm_and_arm_paper_sells(
    *,
    store_root: Optional[Path] = None,
    results_root: Optional[Path] = None,
    confirmation_phrase: str,
    challenge_id: Optional[str] = None,
    now: Optional[datetime] = None,
    paper_mode: Optional[bool] = None,
    live_blocked: Optional[bool] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    now = now or _utc_now()
    counts = {"broker": 0, "network": 0, "permit": 0, "buy_add": 0}
    blocked = _production_override_block(db_path=db_path, results_root=results_root)
    if blocked:
        return _denied("confirm", blocked, counts)
    store_root = Path(store_root or default_store_root())
    results_root = Path(results_root or default_results_root())
    sqlite_path = resolve_runtime_sqlite_path(db_path=db_path)
    packet, load_reason = load_reviewed_sell_packet(results_root)
    if packet is None:
        _record_failure(store_root, "sell_arm_button2_denied", load_reason)
        return _denied("confirm", load_reason, counts)

    if str(confirmation_phrase or "") != CONFIRM_PAPER_SELLS_PHRASE:
        _record_failure(store_root, "sell_arm_button2_denied", "confirmation_phrase_mismatch")
        return _denied("confirm", "confirmation_phrase_mismatch", counts)

    pres = build_presentation(
        store_root=store_root,
        results_root=results_root,
        now=now,
        paper_mode=paper_mode,
        live_blocked=live_blocked,
        confirmation_phrase=confirmation_phrase,
        session_challenge_id=challenge_id,
        db_path=sqlite_path,
    )
    snap = pres["snapshot"]
    if not pres.get("confirm_enabled"):
        reason = str(pres.get("confirm_deny_reason") or "unknown_exception")
        _record_failure(store_root, "sell_arm_button2_denied", reason)
        return _denied("confirm", reason, counts, snapshot=snap)

    cid = pres.get("challenge_id") or challenge_id
    challenge_rec = None
    for rec in snap.get("open_challenges") or []:
        if rec.get("challenge_id") == cid:
            challenge_rec = rec
            break
    if challenge_rec is None:
        _record_failure(store_root, "sell_arm_button2_denied", "challenge_missing")
        return _denied("confirm", "challenge_missing", counts, snapshot=snap)

    payload = dict(challenge_rec.get("payload") or {})
    current_plan = str(packet.get("plan_hash") or "")
    current_batch = batch_hash_from_scope(_scope_from_readiness(packet))
    if current_plan != str(challenge_rec.get("plan_hash") or payload.get("plan_hash") or ""):
        _record_failure(store_root, "sell_arm_button2_denied", "plan_mismatch")
        return _denied("confirm", "plan_mismatch", counts, snapshot=snap)
    if current_batch != str(challenge_rec.get("batch_hash") or ""):
        _record_failure(store_root, "sell_arm_button2_denied", "batch_mismatch")
        return _denied("confirm", "batch_mismatch", counts, snapshot=snap)

    try:
        authority = confirm_and_create_paper_sell_arm(
            store_root=store_root,
            readiness_packet=packet,
            challenge=payload,
            operator_acknowledgement_text=OPERATOR_ACKNOWLEDGEMENT_TEXT,
            issued_by="operator_console",
            current_plan_hash=current_plan,
            now=now,
            explicit_runtime_write=True,
            db_path=sqlite_path,
        )
    except Exception as exc:
        reason = _map_arm_exception(exc)
        _record_failure(store_root, "sell_arm_button2_denied", reason)
        return _denied("confirm", reason, counts, snapshot=snap)

    snap2 = sqlite_status(store_root, now=now, db_path=sqlite_path)
    latest_ch = snap2.get("latest_challenge") or {}
    auth = snap2.get("latest_authority") or {}
    unused = str(auth.get("state")) == "UNUSED"
    exp = _parse_iso(str(auth.get("expires_at") or ""))
    unexpired = exp is None or exp > now
    consumed = str(latest_ch.get("state")) == "CONSUMED"
    if not (
        consumed
        and unused
        and unexpired
        and int(snap2.get("authority_count") or 0) == 1
        and int(snap2.get("dispatch_count") or 0) == 0
    ):
        _record_failure(store_root, "sell_arm_button2_denied", "unknown_exception")
        return _denied("confirm", "unknown_exception", counts, snapshot=snap2)

    return {
        "ok": True,
        "button": "confirm",
        "message": CONFIRM_SUCCESS,
        "reason": None,
        "challenge_id": cid,
        "challenge_state": "CONSUMED",
        "authority_state": "UNUSED",
        "authority_expires_at": auth.get("expires_at"),
        "single_use": True,
        "action_count": len(action_summaries(packet)),
        "dispatch_count": 0,
        "snapshot": snap2,
        "counts": counts,
        "warning": (
            "A later eligible paper runtime cycle may consume this unused authority. "
            "No orders were submitted."
        ),
        "authority_id": auth.get("authority_id") or authority.get("authority_id"),
        "resolved_db_path": str(sqlite_path),
    }


def _denied(
    button: str, reason: str, counts: Dict[str, int], snapshot: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "ok": False,
        "button": button,
        "message": _operator_message(reason),
        "reason": reason,
        "snapshot": snapshot or {},
        "counts": counts,
        "authority_count": (snapshot or {}).get("authority_count") or 0,
        "dispatch_count": (snapshot or {}).get("dispatch_count") or 0,
    }
