"""Operator Console orchestration for paper SELL arm buttons.

Thin controller: public authority APIs for writes, store snapshot for reads.
Creates no dispatch, permit, or broker call. SQLite remains the authority source.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from services.live_block_classifier import evaluate_live_block_status
from services.paper_sell_arm_authority import (
    BUY_ACTIONS,
    LIVE_GOVERNANCE,
    OPERATOR_ACKNOWLEDGEMENT_TEXT,
    READINESS_REFERENCE_TTL_MINUTES,
    SELL_ACTIONS,
    evaluate_readiness_packet_freshness,
    _parse_iso,
    _scope_from_readiness,
    append_audit,
    applied_authority_cap,
    classify_readiness_packet,
    confirm_and_create_paper_sell_arm,
    prepare_confirmation_challenge,
    PaperSellArmPaths,
)
from services.paper_sell_arm_cap_slice import (
    STATUS_AUTHORITY_CAP_EXCEEDED,
    STATUS_SLICE_READY,
    authorized_scope_from_slice,
    money_round,
    presentation_token,
    slice_reviewed_sells,
    tokens_match,
)
from services.paper_sell_arm_store import (
    PaperSellArmStoreError,
    batch_hash_from_scope,
    count_current_unused_exact_batch_authorities,
    count_dispatches_for_authority,
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
    "no_reviewed_sell_actions": "No reviewed SELL actions.",
    "stale_readiness_packet": "Reviewed SELL readiness packet is stale.",
    "session_boundary_closed": "Reviewed SELL readiness is past the authorized session boundary.",
    "prior_session_packet": "Reviewed SELL readiness belongs to a prior session.",
    "session_derivation_unknown": "Market session could not be derived. Paper SELL arming is refused.",
    "malformed_packet": "Reviewed SELL readiness packet is malformed or unavailable.",
    "missing_generated_at": "Missing generated_at",
    "naive_generated_at": "Reviewed SELL readiness generated_at is not timezone-aware.",
    "future_generated_at": "Reviewed SELL readiness generated_at is in the future.",
    "unparseable_generated_at": "Reviewed SELL readiness generated_at is unparseable.",
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
    "basis_unproven_hold_policy_unavailable": (
        "Basis-unproven hold policy could not be validated. No sell packet is offered."
    ),
    "basis_unproven_hold_symbol_in_packet": (
        "Reviewed packet includes a basis-unproven hold symbol. "
        "It is not offered for arming. A replacement must be newly generated."
    ),
    "authority_already_created": "Paper SELL authority already exists for this challenge.",
    "configuration_blocked": "Paper SELL arming is blocked by a non-production configuration setting.",
    "unknown_exception": "Paper SELL arming failed closed. No orders were submitted.",
    "post_create_authority_missing": (
        "Paper SELL confirmation did not persist an authority id. No orders were submitted."
    ),
    "post_create_authority_mismatch": (
        "Paper SELL confirmation did not match the newly created authority. "
        "No orders were submitted."
    ),
    "challenge_not_consumed_after_create": (
        "Paper SELL confirmation challenge was not consumed. No orders were submitted."
    ),
    "authority_not_unused_after_create": (
        "Paper SELL authority was not unused after create. No orders were submitted."
    ),
    "authority_expired_after_create": (
        "Paper SELL authority expired during confirmation. No orders were submitted."
    ),
    "unused_authority_count_mismatch": (
        "Paper SELL confirmation did not leave exactly one unused authority for this batch. "
        "No orders were submitted."
    ),
    "authority_dispatch_already_present": (
        "Paper SELL authority already has a dispatch and is not newly unused. "
        "No orders were submitted."
    ),
    "post_create_dispatch_count_unreadable": (
        "Paper SELL dispatch history for the new authority could not be read. "
        "No orders were submitted."
    ),
    "source_unbound": "Reviewed SELL readiness is not bound to the current cycle.",
    "blocked_other_gate": "Reviewed SELL actions are blocked by a non-authority gate.",
    "sell_outcome_unknown": "Reviewed SELL outcome is unknown and requires reconciliation.",
    "authority_cap_exceeded": "No reviewed SELL quantity fits the $5,000 authority cap.",
    "packet_changed_since_render": "Reviewed SELL packet changed. Refresh and review the new slice.",
    "notional_expansion_vs_authorized_slice": (
        "Planned SELL notional exceeds the authorized slice maximum."
    ),
    "notional_expansion_vs_authorized_slice_total": (
        "Planned SELL total exceeds the authorized slice maximum."
    ),
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
    key = str(reason or "")
    if key in FAIL_CLOSED_MESSAGES:
        return FAIL_CLOSED_MESSAGES[key]
    if key.startswith("notional_expansion_vs_authorized_slice:"):
        return FAIL_CLOSED_MESSAGES["notional_expansion_vs_authorized_slice"]
    base = key.split(":", 1)[0]
    if base in FAIL_CLOSED_MESSAGES:
        return FAIL_CLOSED_MESSAGES[base]
    return FAIL_CLOSED_MESSAGES["unknown_exception"]


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
            if tail.startswith("sqlite_"):
                return _map_store_reason(tail)
            if "scope_exceeds_approval_cli_cap" in tail or "symbol_exceeds_cap" in tail:
                return "authority_cap_exceeded"
            return "malformed_packet"
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
                "total_notional_exceeds_approval_cli_cap": "authority_cap_exceeded",
            }
            for key, mapped in mapping.items():
                if key in tail:
                    return mapped
            if tail.startswith("sqlite_"):
                return _map_store_reason(tail)
            if "readiness" in tail:
                return "malformed_packet"
            if "exceeds_approval_cli_cap" in tail or "symbol_exceeds_cap" in tail:
                return "authority_cap_exceeded"
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


def load_reviewed_sell_packet(
    results_root: Path,
) -> Tuple[Optional[Dict[str, Any]], str]:
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
    return evaluate_readiness_packet_freshness(packet, now=now)


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


def hold_arm_denial(packet: Dict[str, Any]) -> Optional[str]:
    """Refuse Ready-to-Arm when the hold policy fails or a blocked symbol is included.

    Does not rewrite the packet. A hold symbol requires a newly generated packet.
    """
    from services.basis_unproven_hold import evaluate_basis_unproven_hold_for_packet

    hold = evaluate_basis_unproven_hold_for_packet()
    if not hold.get("ok"):
        return "basis_unproven_hold_policy_unavailable"
    blocked = {str(symbol).upper() for symbol in (hold.get("blocked_symbols") or [])}
    symbols = []
    for row in packet.get("approval_ready_actions") or []:
        if isinstance(row, dict):
            symbols.append(str(row.get("symbol") or "").upper())
    for symbol in packet.get("approval_ready_symbols") or []:
        symbols.append(str(symbol or "").upper())
    if blocked.intersection(symbol for symbol in symbols if symbol):
        return "basis_unproven_hold_symbol_in_packet"
    return None


def _sell_only_nonempty(packet: Dict[str, Any]) -> Tuple[bool, str]:
    actions = list(packet.get("approval_ready_actions") or [])
    if not actions:
        return False, "no_reviewed_sell_actions"
    for row in actions:
        if not isinstance(row, dict):
            return False, "malformed_packet"
        action = str(row.get("action") or "").upper()
        if action in BUY_ACTIONS:
            return False, "malformed_packet"
        if action not in SELL_ACTIONS:
            return False, "malformed_packet"
        try:
            qty = float(row.get("qty") or 0)
        except (TypeError, ValueError):
            return False, "malformed_packet"
        if not str(row.get("symbol") or "").strip() or qty <= 0:
            return False, "malformed_packet"
    scope = _scope_from_readiness(packet)
    if not scope or len(scope) != len(actions):
        return False, "malformed_packet"
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


def _packet_file_sha256(results_root: Path) -> Optional[str]:
    path = Path(results_root) / READINESS_PACKET_NAME
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_slice_fields() -> Dict[str, Any]:
    return {
        "applied_cap": None,
        "cap_source": None,
        "slice": None,
        "slice_actions": [],
        "slice_action_count": 0,
        "slice_total_notional": 0.0,
        "deferred_actions": [],
        "deferred_action_count": 0,
        "deferred_total_notional": 0.0,
        "unsliceable_actions": [],
        "slice_hash": "",
        "estimated_remaining_slice_count": 0,
        "estimated_remaining_slice_count_is_estimate": True,
        "full_total_notional": 0.0,
        "presentation_token": None,
        "packet_sha256": None,
        "source_packet_fingerprint": None,
    }


def _slice_presentation_fields(
    packet: Dict[str, Any], *, results_root: Path
) -> Tuple[Dict[str, Any], Optional[str]]:
    cap_value, cap_source = applied_authority_cap()
    sliced = slice_reviewed_sells(packet, cap=cap_value, cap_source=cap_source)
    file_sha = _packet_file_sha256(results_root)
    token = presentation_token(packet=packet, slice_result=sliced, packet_sha256=file_sha)
    fields = {
        "applied_cap": sliced.get("applied_cap"),
        "cap_source": sliced.get("cap_source"),
        "slice": sliced,
        "slice_actions": sliced.get("slice_actions") or [],
        "slice_action_count": int(sliced.get("slice_action_count") or 0),
        "slice_total_notional": sliced.get("slice_total_notional") or 0.0,
        "deferred_actions": sliced.get("deferred_actions") or [],
        "deferred_action_count": int(sliced.get("deferred_action_count") or 0),
        "deferred_total_notional": sliced.get("deferred_total_notional") or 0.0,
        "unsliceable_actions": sliced.get("unsliceable_actions") or [],
        "slice_hash": sliced.get("slice_hash") or "",
        "estimated_remaining_slice_count": int(sliced.get("estimated_remaining_slice_count") or 0),
        "estimated_remaining_slice_count_is_estimate": True,
        "full_total_notional": sliced.get("full_total_notional")
        or packet.get("total_proposed_notional")
        or 0.0,
        "presentation_token": token,
        "packet_sha256": file_sha,
        "source_packet_fingerprint": sliced.get("source_packet_fingerprint"),
    }
    status = sliced.get("status")
    if status == STATUS_SLICE_READY:
        if int(fields["slice_action_count"] or 0) < 1:
            return fields, "authority_cap_exceeded"
        if money_round(fields["slice_total_notional"]) > money_round(
            sliced.get("applied_cap") or 5000.0
        ):
            return fields, "authority_cap_exceeded"
        return fields, None
    if status == STATUS_AUTHORITY_CAP_EXCEEDED:
        return fields, "authority_cap_exceeded"
    return fields, "malformed_packet"


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


def _dispatch_visibility_fields(
    *,
    authority_state: str,
    authority_id: str,
    results_root: Path,
) -> Dict[str, Any]:
    empty = {
        "last_dispatch_visibility": None,
        "last_dispatch_message": None,
        "suppress_no_orders_submitted": False,
    }
    try:
        from services.paper_order_ledger import (
            consumed_authority_message,
            dispatch_visibility,
            reduce_ledger,
        )

        root = Path(results_root)
        if root.name == "results":
            root = root.parent.parent
        rows = reduce_ledger(root=root)
        vis = dispatch_visibility(rows, authority_id=authority_id or None)
        if authority_state == "CONSUMED" and vis.get("has_dispatch"):
            return {
                "last_dispatch_visibility": vis,
                "last_dispatch_message": consumed_authority_message(vis),
                "suppress_no_orders_submitted": True,
            }
        if authority_state == "UNUSED":
            return {
                "last_dispatch_visibility": vis,
                "last_dispatch_message": (
                    "Paper SELL authority is unused and available for a later eligible cycle."
                ),
                "suppress_no_orders_submitted": False,
            }
        if authority_state == "EXPIRED" and not vis.get("has_dispatch"):
            return {
                "last_dispatch_visibility": vis,
                "last_dispatch_message": "Paper SELL authority expired and was never used.",
                "suppress_no_orders_submitted": False,
            }
    except Exception:
        return empty
    return empty


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
            **_empty_slice_fields(),
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
    slice_fields = _empty_slice_fields()
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
        else:
            deny_arm = hold_arm_denial(packet)
        slice_fields = _empty_slice_fields()
        if deny_arm is None and packet is not None:
            slice_fields, slice_deny = _slice_presentation_fields(packet, results_root=results_root)
            if slice_deny:
                deny_arm = slice_deny
            else:
                plan_hash = str(packet.get("plan_hash") or "")
                batch_hash = batch_hash_from_scope(
                    authorized_scope_from_slice(slice_fields["slice"])
                )
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
    matching_unused = (
        str(auth.get("state") or "") == "UNUSED"
        and bool(plan_hash)
        and bool(batch_hash)
        and str(auth.get("plan_hash") or "") == plan_hash
        and str(auth.get("batch_hash") or "") == batch_hash
    )
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
    elif deny_arm is None and matching_unused:
        arm_enabled = False
        arm_duplicate = False
    else:
        arm_enabled = deny_arm is None
        arm_duplicate = False

    pkt_status = str((packet or {}).get("packet_status") or "")
    if pkt_status == "BLOCKED_OTHER_GATE" and deny_arm in (
        None,
        "no_reviewed_sell_actions",
    ):
        deny_arm = "blocked_other_gate"
        deny_confirm = deny_arm
        arm_enabled = False
        confirm_enabled = False
    elif pkt_status == "OUTCOME_UNKNOWN":
        deny_arm = "sell_outcome_unknown"
        deny_confirm = deny_arm
        arm_enabled = False
        confirm_enabled = False
    elif pkt_status == "SOURCE_UNBOUND":
        deny_arm = "source_unbound"
        deny_confirm = deny_arm
        arm_enabled = False
        confirm_enabled = False

    freshness_label = "UNAVAILABLE"
    if packet is None:
        freshness_label = "MALFORMED"
    elif deny_arm == "missing_generated_at":
        freshness_label = "MALFORMED"
    elif deny_arm in (
        "unparseable_generated_at",
        "naive_generated_at",
        "future_generated_at",
        "malformed_packet",
    ):
        freshness_label = "MALFORMED"
    elif deny_arm == "source_unbound":
        freshness_label = "SOURCE_UNBOUND"
    elif deny_arm == "blocked_other_gate":
        freshness_label = "BLOCKED_OTHER_GATE"
    elif deny_arm == "sell_outcome_unknown":
        freshness_label = "OUTCOME_UNKNOWN"
    elif deny_arm in (
        "stale_readiness_packet",
        "session_boundary_closed",
        "prior_session_packet",
        "session_derivation_unknown",
    ):
        freshness_label = "STALE"
    elif deny_arm == "no_reviewed_sell_actions":
        freshness_label = "EMPTY"
    elif deny_arm == "authority_cap_exceeded":
        freshness_label = "UNAVAILABLE"
    elif generated_at is not None:
        freshness_label = (
            "FRESH"
            if (now - generated_at) <= timedelta(minutes=READINESS_REFERENCE_TTL_MINUTES)
            else "STALE"
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
        **slice_fields,
        **_dispatch_visibility_fields(
            authority_state=str(auth.get("state") or "ABSENT"),
            authority_id=str(auth.get("authority_id") or ""),
            results_root=results_root,
        ),
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
    expected_presentation_token: Optional[Dict[str, Any]] = None,
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
    if expected_presentation_token is not None and not tokens_match(
        expected_presentation_token, pres.get("presentation_token")
    ):
        _record_failure(store_root, "sell_arm_button1_denied", "packet_changed_since_render")
        return _denied("arm", "packet_changed_since_render", counts, snapshot=pres["snapshot"])
    if pres.get("arm_duplicate"):
        _record_failure(store_root, "sell_arm_button1_denied", "duplicate_click")
        return _denied("arm", "duplicate_click", counts, snapshot=pres["snapshot"])
    if not pres.get("arm_enabled"):
        reason = str(pres.get("arm_deny_reason") or "unknown_exception")
        _record_failure(store_root, "sell_arm_button1_denied", reason)
        return _denied("arm", reason, counts, snapshot=pres["snapshot"])

    authorized_slice = pres.get("slice") or {}
    sliced_batch = batch_hash_from_scope(authorized_scope_from_slice(authorized_slice))
    sliced_plan = str(packet.get("plan_hash") or "")

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
            authorized_slice=authorized_slice,
        )
    except Exception as exc:
        reason = _map_arm_exception(exc)
        snap = sqlite_status(store_root, now=now, db_path=sqlite_path)
        open_match = matching_open_challenge(
            snap,
            plan_hash=sliced_plan,
            batch_hash=sliced_batch,
            now=now,
        )
        if not open_match:
            try:
                from services.paper_sell_arm_store import write_connection

                with write_connection(sqlite_path) as conn:
                    row = conn.execute(
                        """
                        SELECT challenge_id FROM sell_challenges
                        WHERE state='OPEN' AND plan_hash=? AND batch_hash=?
                        """,
                        (sliced_plan, sliced_batch),
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
        plan_hash=sliced_plan,
        batch_hash=sliced_batch,
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
        "action_count": int(pres.get("slice_action_count") or 0),
        "slice_hash": pres.get("slice_hash"),
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
    authorized_slice = pres.get("slice") or {}
    current_plan = str(packet.get("plan_hash") or "")
    current_batch = batch_hash_from_scope(authorized_scope_from_slice(authorized_slice))
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
            authorized_slice=authorized_slice,
        )
    except Exception as exc:
        reason = _map_arm_exception(exc)
        _record_failure(store_root, "sell_arm_button2_denied", reason)
        return _denied("confirm", reason, counts, snapshot=snap)

    snap2 = sqlite_status(store_root, now=now, db_path=sqlite_path)
    latest_ch = snap2.get("latest_challenge") or {}
    auth = snap2.get("latest_authority") or {}
    created_id = str(authority.get("authority_id") or "")
    unused = str(auth.get("state")) == "UNUSED"
    exp = _parse_iso(str(auth.get("expires_at") or ""))
    unexpired = exp is None or exp > now
    consumed = str(latest_ch.get("state")) == "CONSUMED"
    current_unused = 0
    try:
        current_unused = count_current_unused_exact_batch_authorities(
            sqlite_path,
            now=now,
            plan_hash=current_plan,
            batch_hash=current_batch,
        )
    except Exception:
        current_unused = 0
    created_dispatch_count = -1
    try:
        created_dispatch_count = count_dispatches_for_authority(sqlite_path, created_id)
    except Exception:
        created_dispatch_count = -1
    post_reason = classify_post_create_confirm(
        created_id=created_id,
        latest_authority=auth,
        challenge_consumed=consumed,
        unused=unused,
        unexpired=unexpired,
        current_unused=current_unused,
        created_dispatch_count=created_dispatch_count,
    )
    if post_reason:
        _record_failure(store_root, "sell_arm_button2_denied", post_reason)
        return _denied("confirm", post_reason, counts, snapshot=snap2)

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
        "action_count": int(pres.get("slice_action_count") or 0),
        "slice_hash": pres.get("slice_hash"),
        "dispatch_count": created_dispatch_count,
        "snapshot": snap2,
        "counts": counts,
        "warning": (
            "A later eligible paper runtime cycle may consume this unused authority. "
            "No orders were submitted."
        ),
        "authority_id": auth.get("authority_id") or authority.get("authority_id"),
        "resolved_db_path": str(sqlite_path),
    }


def classify_post_create_confirm(
    *,
    created_id: str,
    latest_authority: Mapping[str, Any],
    challenge_consumed: bool,
    unused: bool,
    unexpired: bool,
    current_unused: int,
    created_dispatch_count: int,
) -> Optional[str]:
    """Return an explicit post-create denial reason, or None when newly unused."""
    if not created_id:
        return "post_create_authority_missing"
    if str(latest_authority.get("authority_id") or "") != created_id:
        return "post_create_authority_mismatch"
    if not challenge_consumed:
        return "challenge_not_consumed_after_create"
    if not unused:
        return "authority_not_unused_after_create"
    if not unexpired:
        return "authority_expired_after_create"
    if int(current_unused) != 1:
        return "unused_authority_count_mismatch"
    if int(created_dispatch_count) < 0:
        return "post_create_dispatch_count_unreadable"
    if int(created_dispatch_count) != 0:
        return "authority_dispatch_already_present"
    return None


def _denied(
    button: str,
    reason: str,
    counts: Dict[str, int],
    snapshot: Optional[Dict[str, Any]] = None,
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
