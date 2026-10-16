"""One-time alignment for the October 15 and 16 manual-exit store.

Disabled unless both module flags are true. Dry-run reads a root and writes
nothing. A future execute call also needs an explicit root, a one-time
manifest, and the audited pre-repair bytes. This module does not call the
broker, append ledgers, or publish a close.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sqlite3
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from services.manual_portfolio_exit import (
    DISPATCH_TERMINAL_DIR,
    DISPATCH_TERMINAL_SCHEMA,
    STORE_DB_NAME,
    TERMINAL_RECONCILIATION,
    evaluate_manual_exit_close_gate,
)
from services.paper_sell_arm_runtime_enforcement import DURABLE_PERMIT_NAME
from services.paper_sell_arm_store import canonical_json
from services.performance_publication_authority import inspect_runtime_state

WRITE_MODE_ENABLED = False
EXECUTOR_ENABLED = False

STORE_RELATIVE = Path("data/governance/manual_portfolio_exit")
PAPER_LEDGER_RELATIVE = Path("data/results/paper_order_events.jsonl")
ACCOUNTING_LEDGER_RELATIVE = Path("data/results/performance_accounting_events.jsonl")
MANIFEST_RELATIVE = Path("data/governance/october15_16_manual_exit_store.authorization")
GOOGL_BROKER_ORDER_ID = "924938e3-813d-4007-9384-b36fb639cd66"
TERMINAL_STATUS = "filled"
_RUNTIME_LOCKS = (
    Path("data/live/continuous_loop.lock"),
    Path("data/live/watchdog_loop.lock"),
    Path("data/live/daily_refresh.lock"),
    Path("data/daily_refresh.lock"),
)

_OCT15 = {
    "broker_order_id": "507112c1-8984-4450-8c67-f9d1bca5a8ab",
    "client_order_id": "manual-BITO-0f628c831b6e31d43583ceb6",
    "request_id": "0f628c831b6e31d43583ceb610e13db1ed32c1bd727ad32e5856d5e6e0a2215f",
    "authority_id": "manual-auth-0f628c831b6e31d43583ceb6",
    "challenge_id": "manual-chal-0f628c831b6e31d43583ceb6",
    "dispatch_id": "sell-dispatch-f40ccbd63cf54e52",
    "qty": Decimal("450"),
    "final_result": "DISPATCH_INTENT",
    "result_class": None,
    "receipt_linked": False,
    "identity_present": False,
    "action_submitted_event_id": (
        "3a979aa39b42013e02bb3c431c10d788c57277daa9dcd18ac220cc3b94a489d4"
    ),
    "terminal_event_id": "fe597bdac633f99aff0142b2862d33ebf8b419f5b20c9293d571bab023a83528",
    "fill_sell_event_id": "074134fa1bdbfc863192763a87803f2c2f7b3d204428906214f3504088a1d6ab",
}
_OCT16_400 = {
    "broker_order_id": "7443d112-0191-4eb5-89dd-69b59b950577",
    "client_order_id": "manual-BITO-191fcd3e5a3bac36cc0e5200",
    "request_id": "191fcd3e5a3bac36cc0e5200c9ab4c91bc00cbb51a9f3398ad2e20fbaf1afd16",
    "authority_id": "manual-auth-191fcd3e5a3bac36cc0e5200",
    "challenge_id": "manual-chal-191fcd3e5a3bac36cc0e5200",
    "dispatch_id": "sell-dispatch-ed4565837b154c44",
    "qty": Decimal("400"),
    "final_result": "SUBMITTED",
    "result_class": "SUBMITTED",
    "receipt_linked": True,
    "identity_present": True,
    "action_submitted_event_id": (
        "7c1a1c8e03416373d032617e558ddad0be7ac08771bb0e1c5ec4fd37c520c3f2"
    ),
    "terminal_event_id": "bb987376d9bc57665965d94a5f856b38158ba20b1a17e95f05648cc38cf02896",
    "fill_sell_event_id": "bc0f958b4df0c01a1fd4655d8f4e17c36f13134e26673bdcf19f1bb60e266478",
}
_OCT16_19 = {
    "broker_order_id": "99bd5524-40c6-439b-8600-533d937140dc",
    "client_order_id": "manual-BITO-eedd181cd423bc6382ee97ed",
    "request_id": "eedd181cd423bc6382ee97ed9f01778cbd03ba6d95172bd1e06a1221bc9e670a",
    "authority_id": "manual-auth-eedd181cd423bc6382ee97ed",
    "challenge_id": "manual-chal-eedd181cd423bc6382ee97ed",
    "dispatch_id": "sell-dispatch-d17896a20e9846af",
    "qty": Decimal("19"),
    "final_result": "SUBMITTED",
    "result_class": "SUBMITTED",
    "receipt_linked": True,
    "identity_present": True,
    "action_submitted_event_id": (
        "c38e0cce149f0c06d80c18f7c30934626187a00e737c326ef046fd81d2b91a09"
    ),
    "terminal_event_id": "218135558759444cfa834963642e71cd5ac93d43cfe3ef7ddd3bc6641680da59",
    "fill_sell_event_id": "bcf0279d5272a237a47511848db04a8c7551b270de8cd045f42c7f25c2732b4a",
}
TARGETS: tuple[Dict[str, Any], ...] = (_OCT15, _OCT16_400, _OCT16_19)

_PRE_REPAIR_FILES = frozenset(
    {
        STORE_DB_NAME,
        STORE_DB_NAME + "-wal",
        STORE_DB_NAME + "-shm",
        DURABLE_PERMIT_NAME,
        f"order_identities/{_OCT16_400['broker_order_id']}.json",
        f"order_identities/{_OCT16_19['broker_order_id']}.json",
        f"requests/{_OCT15['request_id']}.json",
        f"requests/{_OCT16_400['request_id']}.json",
        f"requests/{_OCT16_19['request_id']}.json",
    }
)
_POST_REPAIR_FILES = frozenset(
    (_PRE_REPAIR_FILES - {STORE_DB_NAME + "-wal", STORE_DB_NAME + "-shm"})
    | {
        f"order_identities/{_OCT15['broker_order_id']}.json",
        *(f"{DISPATCH_TERMINAL_DIR}/{target['dispatch_id']}.json" for target in TARGETS),
    }
)
_HASH_FIELDS = (
    "paper_order_events_sha256",
    "performance_accounting_events_sha256",
    "manual_portfolio_exit_sqlite3_sha256",
    "manual_portfolio_exit_sqlite3_wal_sha256",
    "manual_portfolio_exit_sqlite3_shm_sha256",
    "permit_sha256",
    "request_oct15_sha256",
    "request_oct16_400_sha256",
    "request_oct16_19_sha256",
    "identity_oct16_400_sha256",
    "identity_oct16_19_sha256",
)
_MANIFEST_FIELDS = _HASH_FIELDS + ("nonce",)
_HASH_PATHS = {
    "paper_order_events_sha256": PAPER_LEDGER_RELATIVE,
    "performance_accounting_events_sha256": ACCOUNTING_LEDGER_RELATIVE,
    "manual_portfolio_exit_sqlite3_sha256": STORE_RELATIVE / STORE_DB_NAME,
    "manual_portfolio_exit_sqlite3_wal_sha256": STORE_RELATIVE / (STORE_DB_NAME + "-wal"),
    "manual_portfolio_exit_sqlite3_shm_sha256": STORE_RELATIVE / (STORE_DB_NAME + "-shm"),
    "permit_sha256": STORE_RELATIVE / DURABLE_PERMIT_NAME,
    "request_oct15_sha256": STORE_RELATIVE / "requests" / f"{_OCT15['request_id']}.json",
    "request_oct16_400_sha256": STORE_RELATIVE / "requests" / f"{_OCT16_400['request_id']}.json",
    "request_oct16_19_sha256": STORE_RELATIVE / "requests" / f"{_OCT16_19['request_id']}.json",
    "identity_oct16_400_sha256": (
        STORE_RELATIVE / "order_identities" / f"{_OCT16_400['broker_order_id']}.json"
    ),
    "identity_oct16_19_sha256": (
        STORE_RELATIVE / "order_identities" / f"{_OCT16_19['broker_order_id']}.json"
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fail(reason: str, *, dry_run: bool, **details: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "wrote": False,
        "dry_run": dry_run,
        "reason": reason,
        "broker_called": False,
        "executor_enabled": EXECUTOR_ENABLED,
        "write_mode_enabled": WRITE_MODE_ENABLED,
    }
    if details:
        payload["details"] = details
    return payload


def _decimal(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError):
        return None
    if not number.is_finite():
        return None
    return number


def _store(root: Path) -> Path:
    return Path(root) / STORE_RELATIVE


def _sha256_file(path: Path) -> Optional[str]:
    try:
        data = path.read_bytes()
    except PermissionError as exc:
        if path.name == STORE_DB_NAME or path.name.startswith(STORE_DB_NAME + "-"):
            raise PermissionError("sqlite_handle_active") from exc
        return None
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def baseline_hashes(root: Path) -> Dict[str, Any]:
    """Hash the audited files. Missing files are reported and nothing is written."""
    found: Dict[str, str] = {}
    missing = []
    for field, relative in _HASH_PATHS.items():
        try:
            digest = _sha256_file(Path(root) / relative)
        except PermissionError:
            return _fail("sqlite_handle_active", dry_run=True)
        if digest is None:
            missing.append(field)
        else:
            found[field] = digest
    if missing:
        return _fail("baseline_hash_unavailable", dry_run=True, fields=sorted(missing))
    return {"ok": True, "wrote": False, "hashes": found}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _dump(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def _child_set(store: Path) -> Optional[set[str]]:
    if not store.is_dir():
        return None
    found = set()
    for path in store.rglob("*"):
        if path.is_file():
            found.add(path.relative_to(store).as_posix())
    return found


def _immutable(db_path: Path) -> sqlite3.Connection:
    uri = db_path.resolve().as_posix()
    connection = sqlite3.connect(f"file:{uri}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _ledger_events(path: Path) -> list[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("ledger row")
        events.append(payload)
    return events


def _planned_changes() -> list[Dict[str, Any]]:
    changes: list[Dict[str, Any]] = []
    for target in TARGETS:
        request_path = f"requests/{target['request_id']}.json"
        if target["receipt_linked"]:
            changes.append({"path": request_path, "action": "confirm_terminal"})
        else:
            changes.append(
                {
                    "path": request_path,
                    "action": "add_linkage",
                    "fields": ["authority_id", "client_order_id", "dispatch_id"],
                }
            )
        identity_path = f"order_identities/{target['broker_order_id']}.json"
        changes.append(
            {
                "path": identity_path,
                "action": "set_terminal_status" if target["identity_present"] else "create",
                "broker_status": TERMINAL_STATUS,
                "reconciliation_state": TERMINAL_RECONCILIATION,
            }
        )
        changes.append(
            {
                "path": f"{DISPATCH_TERMINAL_DIR}/{target['dispatch_id']}.json",
                "action": "create",
                "broker_status": TERMINAL_STATUS,
                "reconciliation_state": TERMINAL_RECONCILIATION,
            }
        )
        changes.append(
            {
                "path": STORE_DB_NAME,
                "action": "consume_challenge",
                "challenge_id": target["challenge_id"],
                "authority_id": target["authority_id"],
            }
        )
        changes.append(
            {
                "path": STORE_DB_NAME,
                "action": "record_terminal_state",
                "dispatch_id": target["dispatch_id"],
                "broker_status": TERMINAL_STATUS,
                "reconciliation_state": TERMINAL_RECONCILIATION,
                "preserve_final_result": target["final_result"],
            }
        )
    changes.append(
        {
            "path": DURABLE_PERMIT_NAME,
            "action": "replace_non_authoritative",
            "dispatch_id": _OCT16_19["dispatch_id"],
            "permit_token_removed": True,
        }
    )
    changes.append({"path": STORE_DB_NAME + "-wal", "action": "remove"})
    changes.append({"path": STORE_DB_NAME + "-shm", "action": "remove"})
    return changes


def _receipt_problems(target: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    problems = []
    pairs = {
        "request_id": target["request_id"],
        "broker_order_id": target["broker_order_id"],
        "symbol": "BITO",
    }
    for field, wanted in pairs.items():
        found = str(payload.get(field) or "")
        if field == "symbol":
            found = found.upper()
        if found != wanted:
            problems.append(field)
    if _decimal(payload.get("selected_qty")) != target["qty"]:
        problems.append("selected_qty")
    if str(payload.get("broker_status") or "") != TERMINAL_STATUS:
        problems.append("broker_status")
    if str(payload.get("reconciliation_state") or "") != TERMINAL_RECONCILIATION:
        problems.append("reconciliation_state")
    if str(payload.get("status") or "") != "submitted":
        problems.append("status")
    linked = all(
        str(payload.get(field) or "") == target[field]
        for field in ("authority_id", "dispatch_id", "client_order_id")
    )
    if target["receipt_linked"] and not linked:
        problems.append("linkage")
    if not target["receipt_linked"] and any(
        payload.get(field)
        for field in (
            "authority_id",
            "dispatch_id",
            "client_order_id",
        )
    ):
        problems.append("linkage_already_present")
    return problems


def _identity_problems(target: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    problems = []
    for field in (
        "broker_order_id",
        "request_id",
        "authority_id",
        "dispatch_id",
        "client_order_id",
    ):
        if str(payload.get(field) or "") != target[field]:
            problems.append(field)
    if str(payload.get("symbol") or "").upper() != "BITO":
        problems.append("symbol")
    if _decimal(payload.get("selected_qty")) != target["qty"]:
        problems.append("selected_qty")
    if str(payload.get("broker_status") or "") != "pending_new":
        problems.append("broker_status")
    if str(payload.get("reconciliation_state") or "") != "submission_recorded":
        problems.append("reconciliation_state")
    return problems


def _event_matches(event: Mapping[str, Any], target: Mapping[str, Any], kind: str) -> bool:
    if str(event.get("broker_order_id") or "") != target["broker_order_id"]:
        body = event.get("body") if isinstance(event.get("body"), Mapping) else {}
        if str(body.get("broker_order_id") or "") != target["broker_order_id"]:
            return False
    if kind == "FILL_SELL":
        body = event.get("body") if isinstance(event.get("body"), Mapping) else {}
        return (
            str(event.get("event_type") or "") == "FILL_SELL"
            and str(event.get("event_id") or "") == target["fill_sell_event_id"]
            and str(body.get("symbol") or "").upper() == "BITO"
            and str(body.get("client_order_id") or "") == target["client_order_id"]
            and _decimal(body.get("qty")) == target["qty"]
        )
    wanted_id = (
        target["action_submitted_event_id"]
        if kind == "ACTION_SUBMITTED"
        else target["terminal_event_id"]
    )
    qty_field = "selected_qty" if kind == "ACTION_SUBMITTED" else "filled_qty"
    matched = (
        str(event.get("event_type") or "") == kind
        and str(event.get("event_id") or "") == wanted_id
        and str(event.get("symbol") or "").upper() == "BITO"
        and str(event.get("authority_id") or "") == target["authority_id"]
        and str(event.get("dispatch_id") or "") == target["dispatch_id"]
        and str(event.get("client_order_id") or "") == target["client_order_id"]
        and _decimal(event.get(qty_field)) == target["qty"]
    )
    if kind == "ORDER_TERMINAL_RECONCILED":
        matched = matched and event.get("terminal_status_proven") is True
        matched = matched and str(event.get("broker_terminal_status") or "") == "FILLED"
    return matched


def _ledger_problems(root: Path) -> Optional[Dict[str, Any]]:
    try:
        paper = _ledger_events(Path(root) / PAPER_LEDGER_RELATIVE)
        accounting = _ledger_events(Path(root) / ACCOUNTING_LEDGER_RELATIVE)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _fail("ledger_unreadable", dry_run=True, error=type(exc).__name__)
    for target in TARGETS:
        if target["broker_order_id"] == GOOGL_BROKER_ORDER_ID:
            return _fail("not_recovery_target", dry_run=True)
        counts = {
            "ACTION_SUBMITTED": 0,
            "ORDER_TERMINAL_RECONCILED": 0,
            "FILL_SELL": 0,
        }
        for event in paper:
            broker = str(event.get("broker_order_id") or "")
            if broker != target["broker_order_id"]:
                continue
            kind = str(event.get("event_type") or "")
            if kind not in {"ACTION_SUBMITTED", "ORDER_TERMINAL_RECONCILED"}:
                return _fail(
                    "duplicated_evidence",
                    dry_run=True,
                    broker_order_id=target["broker_order_id"],
                    event_type=kind,
                )
            counts[kind] += 1
            if counts[kind] > 1:
                return _fail(
                    "duplicated_evidence",
                    dry_run=True,
                    broker_order_id=target["broker_order_id"],
                    event_type=kind,
                )
            if not _event_matches(event, target, kind):
                return _fail(
                    "target_mismatch",
                    dry_run=True,
                    fields=[f"ledger_{kind.lower()}"],
                    broker_order_id=target["broker_order_id"],
                )
        for event in accounting:
            body = event.get("body") if isinstance(event.get("body"), Mapping) else {}
            broker = str(body.get("broker_order_id") or event.get("broker_order_id") or "")
            if broker != target["broker_order_id"]:
                continue
            if str(event.get("event_type") or "") != "FILL_SELL":
                return _fail(
                    "duplicated_evidence",
                    dry_run=True,
                    broker_order_id=target["broker_order_id"],
                    event_type=str(event.get("event_type") or ""),
                )
            counts["FILL_SELL"] += 1
            if counts["FILL_SELL"] > 1:
                return _fail(
                    "duplicated_evidence",
                    dry_run=True,
                    broker_order_id=target["broker_order_id"],
                    event_type="FILL_SELL",
                )
            if not _event_matches(event, target, "FILL_SELL"):
                return _fail(
                    "target_mismatch",
                    dry_run=True,
                    fields=["ledger_fill_sell"],
                    broker_order_id=target["broker_order_id"],
                )
        missing = [kind for kind, count in counts.items() if count == 0]
        if missing:
            return _fail(
                "missing_evidence",
                dry_run=True,
                broker_order_id=target["broker_order_id"],
                fields=missing,
            )
    return None


def _sqlite_problems(store: Path) -> Optional[Dict[str, Any]]:
    db_path = store / STORE_DB_NAME
    try:
        connection = _immutable(db_path)
    except sqlite3.OperationalError as exc:
        text = str(exc).lower()
        if "locked" in text or "busy" in text:
            return _fail("sqlite_handle_active", dry_run=True)
        return _fail("sqlite_unreadable", dry_run=True, error=type(exc).__name__)
    except sqlite3.Error as exc:
        return _fail("sqlite_unreadable", dry_run=True, error=type(exc).__name__)
    try:
        challenges = list(connection.execute("SELECT * FROM sell_challenges"))
        authorities = list(connection.execute("SELECT * FROM sell_authorities"))
        dispatches = list(connection.execute("SELECT * FROM sell_dispatches"))
    except sqlite3.OperationalError as exc:
        text = str(exc).lower()
        if "locked" in text or "busy" in text:
            return _fail("sqlite_handle_active", dry_run=True)
        return _fail("sqlite_unreadable", dry_run=True, error=type(exc).__name__)
    except sqlite3.Error as exc:
        return _fail("sqlite_unreadable", dry_run=True, error=type(exc).__name__)
    finally:
        connection.close()
    if len(challenges) != 3 or len(authorities) != 3 or len(dispatches) != 3:
        return _fail("unexpected_store_row", dry_run=True)
    by_challenge = {str(row["challenge_id"]): row for row in challenges}
    by_authority = {str(row["authority_id"]): row for row in authorities}
    by_dispatch = {str(row["dispatch_id"]): row for row in dispatches}
    aligned = False
    problems: list[str] = []
    for target in TARGETS:
        challenge = by_challenge.get(target["challenge_id"])
        authority = by_authority.get(target["authority_id"])
        dispatch = by_dispatch.get(target["dispatch_id"])
        if challenge is None or authority is None or dispatch is None:
            return _fail("target_mismatch", dry_run=True, fields=["sqlite_identity"])
        if str(challenge["state"] or "") != "OPEN" or challenge["consuming_authority_id"]:
            aligned = True
        if str(authority["state"] or "") != "CONSUMED":
            problems.append("authority_state")
        if str(authority["final_result"] or "") != target["final_result"]:
            problems.append("final_result")
        if str(authority["challenge_id"] or "") != target["challenge_id"]:
            problems.append("challenge_id")
        if str(authority["consuming_dispatch_id"] or "") != target["dispatch_id"]:
            problems.append("consuming_dispatch_id")
        if not str(authority["consumption_time"] or ""):
            problems.append("consumption_time")
        if str(dispatch["authority_id"] or "") != target["authority_id"]:
            problems.append("dispatch_authority")
        if str(dispatch["state"] or "") != "SUBMITTED":
            problems.append("dispatch_state")
        if str(dispatch["broker_ref"] or "") != target["broker_order_id"]:
            problems.append("broker_ref")
        result_class = dispatch["result_class"]
        if result_class != target["result_class"]:
            problems.append("result_class")
        try:
            intent = json.loads(str(dispatch["intent_json"] or "{}"))
            payload = json.loads(str(authority["payload_json"] or "{}"))
        except json.JSONDecodeError:
            return _fail("sqlite_unreadable", dry_run=True)
        requested = intent.get("requested") if isinstance(intent, dict) else None
        if (
            not isinstance(requested, list)
            or len(requested) != 1
            or str(requested[0].get("symbol") or "").upper() != "BITO"
            or str(requested[0].get("action") or "").upper() != "EXIT"
            or str(requested[0].get("side") or "").lower() != "sell"
            or _decimal(requested[0].get("qty")) != target["qty"]
        ):
            problems.append("requested_qty")
        if isinstance(intent, dict) and (
            intent.get("broker_status") or intent.get("reconciliation_state")
        ):
            aligned = True
        if (
            isinstance(payload, dict)
            and str(payload.get("request_id") or "") != target["request_id"]
        ):
            problems.append("payload_request_id")
        if str(challenge["operator_action_label"] or "") != "MANUAL_PORTFOLIO_EXIT":
            problems.append("operator_action_label")
    if aligned and not problems:
        return _fail("already_aligned", dry_run=True)
    if problems:
        return _fail("target_mismatch", dry_run=True, fields=sorted(set(problems)))
    return None


def _permit_problems(payload: Mapping[str, Any]) -> list[str]:
    problems = []
    if str(payload.get("status") or "") != "PERMIT_ISSUED_PRE_SUBMIT":
        problems.append("status")
    if str(payload.get("json_role") or "") != "AUDIT_EXPORT_ONLY":
        problems.append("json_role")
    if str(payload.get("dispatch_intent_id") or "") != _OCT16_19["dispatch_id"]:
        problems.append("dispatch_intent_id")
    if str(payload.get("idempotency_key") or "") != _OCT16_19["request_id"]:
        problems.append("idempotency_key")
    if payload.get("symbols") != ["BITO"]:
        problems.append("symbols")
    actions = payload.get("actions")
    if (
        not isinstance(actions, list)
        or len(actions) != 1
        or str(actions[0].get("symbol") or "").upper() != "BITO"
        or str(actions[0].get("action") or "").upper() != "EXIT"
        or _decimal(actions[0].get("qty")) != _OCT16_19["qty"]
    ):
        problems.append("actions")
    token = payload.get("permit_token")
    if not isinstance(token, str) or not token:
        problems.append("permit_token")
    if payload.get("status") == "NON_AUTHORITATIVE" and payload.get("close_gate_evidence") is False:
        return ["already_non_authoritative"]
    return problems


def _assess(root: Path) -> Dict[str, Any]:
    store = _store(root)
    children = _child_set(store)
    if children is None:
        return _fail("store_missing", dry_run=True)
    bare = children
    if bare == _POST_REPAIR_FILES:
        gate = evaluate_manual_exit_close_gate(store)
        if gate.get("ok"):
            return _fail("already_aligned", dry_run=True)
        return _fail("stale_store_state", dry_run=True)
    if bare != _PRE_REPAIR_FILES:
        extra = sorted(bare - _PRE_REPAIR_FILES)
        missing = sorted(_PRE_REPAIR_FILES - bare)
        if extra:
            return _fail("unexpected_child", dry_run=True, paths=extra)
        return _fail("missing_store_child", dry_run=True, paths=missing)
    file_problems: list[str] = []
    aligned = False
    for target in TARGETS:
        receipt = _load_json(store / "requests" / f"{target['request_id']}.json")
        if receipt is None:
            return _fail("stale_store_state", dry_run=True, fields=["receipt"])
        problems = _receipt_problems(target, receipt)
        if "linkage_already_present" in problems:
            aligned = True
        file_problems.extend(
            f"receipt_{item}" for item in problems if item != "linkage_already_present"
        )
        identity_path = store / "order_identities" / f"{target['broker_order_id']}.json"
        if target["identity_present"]:
            identity = _load_json(identity_path)
            if identity is None:
                return _fail("stale_store_state", dry_run=True, fields=["identity"])
            identity_problems = _identity_problems(target, identity)
            if (
                str(identity.get("broker_status") or "") == TERMINAL_STATUS
                and str(identity.get("reconciliation_state") or "") == TERMINAL_RECONCILIATION
            ):
                aligned = True
                identity_problems = [
                    item
                    for item in identity_problems
                    if item not in {"broker_status", "reconciliation_state"}
                ]
            file_problems.extend(f"identity_{item}" for item in identity_problems)
        elif identity_path.exists():
            aligned = True
    if aligned and not file_problems:
        return _fail("already_aligned", dry_run=True)
    permit = _load_json(store / DURABLE_PERMIT_NAME)
    if permit is None:
        return _fail("stale_store_state", dry_run=True, fields=["permit"])
    permit_problems = _permit_problems(permit)
    if "already_non_authoritative" in permit_problems and len(permit_problems) == 1:
        return _fail("already_aligned", dry_run=True)
    file_problems.extend(f"permit_{item}" for item in permit_problems)
    if any(item.startswith("receipt_broker_status") for item in file_problems):
        return _fail("stale_store_state", dry_run=True, fields=sorted(set(file_problems)))
    if file_problems:
        return _fail("target_mismatch", dry_run=True, fields=sorted(set(file_problems)))
    sqlite_problem = _sqlite_problems(store)
    if sqlite_problem is not None:
        return sqlite_problem
    ledger_problem = _ledger_problems(root)
    if ledger_problem is not None:
        return ledger_problem
    hashes = baseline_hashes(root)
    if not hashes.get("ok"):
        return hashes
    return {
        "ok": True,
        "wrote": False,
        "dry_run": True,
        "reason": "dry_run",
        "broker_called": False,
        "executor_enabled": EXECUTOR_ENABLED,
        "write_mode_enabled": WRITE_MODE_ENABLED,
        "changes": _planned_changes(),
        "baseline_hashes": hashes["hashes"],
        "ledger_appended": False,
    }


def dry_run_october15_16_manual_exit_store_recovery(
    *, root: Optional[Path] = None
) -> Dict[str, Any]:
    """Report the alignment plan. Does not create a manifest or write a file."""
    if root is None:
        return _fail("root_required", dry_run=True)
    if not Path(root).is_dir():
        return _fail("root_missing", dry_run=True)
    return _assess(Path(root))


def _lock_held(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text or set(text) <= {"0"}:
        return False
    return True


def _runtime_obstacles(root: Path) -> Dict[str, Any]:
    held = [path.as_posix() for path in _RUNTIME_LOCKS if _lock_held(Path(root) / path)]
    if held:
        return _fail("runtime_lock", dry_run=False, locks=held)
    if Path(root).resolve() != _repo_root().resolve():
        return {"ok": True}
    try:
        state = inspect_runtime_state(Path(root))
    except (OSError, RuntimeError) as exc:
        return _fail("runtime_inspect_failed", dry_run=False, error=type(exc).__name__)
    if state.get("inspect_failed"):
        return _fail("runtime_inspect_failed", dry_run=False)
    if not state.get("streamlit_stopped"):
        return _fail("streamlit_active", dry_run=False)
    if not state.get("runtime_stopped"):
        return _fail("triton_runtime_active", dry_run=False)
    return {"ok": True}


def _authorize(root: Path, token: str, hashes: Mapping[str, str]) -> Dict[str, Any]:
    path = Path(root) / MANIFEST_RELATIVE
    if not path.is_file():
        return _fail("authorization_missing", dry_run=False)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail("authorization_unreadable", dry_run=False, error=type(exc).__name__)
    if not isinstance(manifest, dict) or set(manifest) != set(_MANIFEST_FIELDS):
        return _fail("manifest_rejected", dry_run=False, fields=["shape"])
    if any(
        not isinstance(manifest.get(field), str) or manifest.get(field) == ""
        for field in _MANIFEST_FIELDS
    ):
        return _fail("manifest_rejected", dry_run=False, fields=["shape"])
    nonce = str(manifest.get("nonce") or "")
    if not hmac.compare_digest(str(token or ""), nonce):
        return _fail("authorization_mismatch", dry_run=False)
    mismatched = [field for field in _HASH_FIELDS if manifest.get(field) != hashes.get(field)]
    if mismatched:
        return _fail("baseline_hash_mismatch", dry_run=False, fields=sorted(mismatched))
    return {"ok": True}


def _snapshot(store: Path) -> Dict[str, Optional[bytes]]:
    saved: Dict[str, Optional[bytes]] = {}
    for relative in _PRE_REPAIR_FILES:
        path = store / relative
        saved[relative] = path.read_bytes() if path.is_file() else None
    return saved


def _restore(store: Path, saved: Mapping[str, Optional[bytes]]) -> None:
    current = _child_set(store) or set()
    for relative in current:
        if relative in saved:
            continue
        (store / relative).unlink()
    for relative, data in saved.items():
        path = store / relative
        if data is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _terminal_document(target: Mapping[str, Any], receipt: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": DISPATCH_TERMINAL_SCHEMA,
        "dispatch_id": target["dispatch_id"],
        "authority_id": target["authority_id"],
        "request_id": target["request_id"],
        "broker_order_id": target["broker_order_id"],
        "client_order_id": target["client_order_id"],
        "symbol": "BITO",
        "selected_qty": receipt.get("selected_qty"),
        "broker_status": TERMINAL_STATUS,
        "reconciliation_state": TERMINAL_RECONCILIATION,
        "close_gate_evidence": True,
    }


def _retired_permit(current: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": current.get("schema_version"),
        "status": "NON_AUTHORITATIVE",
        "json_role": "NON_AUTHORITATIVE",
        "close_gate_evidence": False,
        "authorizes_submission": False,
        "replay_forbidden": True,
        "dispatch_intent_id": _OCT16_19["dispatch_id"],
        "idempotency_key": _OCT16_19["request_id"],
        "broker_order_id": _OCT16_19["broker_order_id"],
        "broker_status": TERMINAL_STATUS,
        "reconciliation_state": TERMINAL_RECONCILIATION,
    }


def _apply(store: Path) -> None:
    db_path = store / STORE_DB_NAME
    connection = sqlite3.connect(str(db_path), timeout=0.2, isolation_level=None)
    connection.row_factory = sqlite3.Row
    busy = 1
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for target in TARGETS:
            authority = connection.execute(
                "SELECT consumption_time, final_result FROM sell_authorities WHERE authority_id=?",
                (target["authority_id"],),
            ).fetchone()
            if authority is None or str(authority["final_result"] or "") != target["final_result"]:
                raise RuntimeError("authority_changed")
            updated = connection.execute(
                """
                UPDATE sell_challenges
                SET state='CONSUMED', consumed_at=?, consuming_authority_id=?
                WHERE challenge_id=? AND state='OPEN' AND consuming_authority_id IS NULL
                """,
                (authority["consumption_time"], target["authority_id"], target["challenge_id"]),
            )
            if updated.rowcount != 1:
                raise RuntimeError("challenge_not_open")
            dispatch = connection.execute(
                "SELECT intent_json, broker_ref, result_class FROM sell_dispatches WHERE dispatch_id=?",
                (target["dispatch_id"],),
            ).fetchone()
            payload_row = connection.execute(
                "SELECT payload_json FROM sell_authorities WHERE authority_id=?",
                (target["authority_id"],),
            ).fetchone()
            if dispatch is None or payload_row is None:
                raise RuntimeError("row_missing")
            if str(dispatch["broker_ref"] or "") != target["broker_order_id"]:
                raise RuntimeError("broker_ref_changed")
            if dispatch["result_class"] != target["result_class"]:
                raise RuntimeError("result_class_changed")
            intent = json.loads(str(dispatch["intent_json"]))
            payload = json.loads(str(payload_row["payload_json"]))
            for document in (intent, payload):
                document["broker_status"] = TERMINAL_STATUS
                document["reconciliation_state"] = TERMINAL_RECONCILIATION
                document["broker_order_id"] = target["broker_order_id"]
            connection.execute(
                "UPDATE sell_dispatches SET intent_json=? WHERE dispatch_id=?",
                (canonical_json(intent), target["dispatch_id"]),
            )
            connection.execute(
                "UPDATE sell_authorities SET payload_json=? WHERE authority_id=?",
                (canonical_json(payload), target["authority_id"]),
            )
        connection.execute("COMMIT")
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        busy = int(checkpoint[0]) if checkpoint else 1
    except sqlite3.OperationalError as exc:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        message = str(exc).lower()
        if "locked" in message or "busy" in message:
            raise PermissionError("sqlite_handle_active") from exc
        raise
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        connection.close()
    if busy != 0:
        raise RuntimeError("sqlite_checkpoint_busy")
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    for target in TARGETS:
        receipt_path = store / "requests" / f"{target['request_id']}.json"
        receipt = _load_json(receipt_path)
        if receipt is None:
            raise RuntimeError("receipt_missing")
        if not target["receipt_linked"]:
            receipt["authority_id"] = target["authority_id"]
            receipt["dispatch_id"] = target["dispatch_id"]
            receipt["client_order_id"] = target["client_order_id"]
            receipt_path.write_text(_dump(receipt), encoding="utf-8")
        identity_path = store / "order_identities" / f"{target['broker_order_id']}.json"
        if target["identity_present"]:
            identity = _load_json(identity_path) or {}
        else:
            identity = {
                "authority_id": target["authority_id"],
                "broker_order_id": target["broker_order_id"],
                "client_order_id": target["client_order_id"],
                "dispatch_id": target["dispatch_id"],
                "orders_submitted": True,
                "reason": receipt.get("reason"),
                "request_id": target["request_id"],
                "selected_qty": receipt.get("selected_qty"),
                "status": "submitted",
                "submitted": True,
                "symbol": "BITO",
            }
        identity["broker_status"] = TERMINAL_STATUS
        identity["reconciliation_state"] = TERMINAL_RECONCILIATION
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(_dump(identity), encoding="utf-8")
        evidence_path = store / DISPATCH_TERMINAL_DIR / f"{target['dispatch_id']}.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            _dump(_terminal_document(target, receipt)),
            encoding="utf-8",
        )
    permit_path = store / DURABLE_PERMIT_NAME
    current_permit = _load_json(permit_path)
    if current_permit is None:
        raise RuntimeError("permit_missing")
    permit_path.write_text(_dump(_retired_permit(current_permit)), encoding="utf-8")


def _consume_manifest(root: Path) -> Dict[str, Any]:
    path = Path(root) / MANIFEST_RELATIVE
    try:
        path.unlink()
    except OSError as exc:
        return _fail("manifest_not_consumed", dry_run=False, error=type(exc).__name__)
    if path.exists():
        return _fail("manifest_not_consumed", dry_run=False)
    return {"ok": True}


def execute_october15_16_manual_exit_store_recovery(
    *,
    authorization_token: str,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Align the three audited BITO rows when flags, manifest, and bytes match.

    The flags stay false in source. This does not default to the production
    root, create a manifest, or consume a manifest before the post-check passes.
    """
    if EXECUTOR_ENABLED is not True or WRITE_MODE_ENABLED is not True:
        return _fail("executor_not_enabled", dry_run=False)
    if root is None:
        return _fail("root_required", dry_run=False)
    root_p = Path(root)
    if not root_p.is_dir():
        return _fail("root_missing", dry_run=False)
    assessed = _assess(root_p)
    if not assessed.get("ok"):
        assessed["dry_run"] = False
        return assessed
    authorized = _authorize(root_p, authorization_token, assessed["baseline_hashes"])
    if not authorized.get("ok"):
        return authorized
    runtime = _runtime_obstacles(root_p)
    if not runtime.get("ok"):
        return runtime
    paper_before = (root_p / PAPER_LEDGER_RELATIVE).read_bytes()
    accounting_before = (root_p / ACCOUNTING_LEDGER_RELATIVE).read_bytes()
    store = _store(root_p)
    saved = _snapshot(store)
    try:
        _apply(store)
    except PermissionError:
        _restore(store, saved)
        return _fail("sqlite_handle_active", dry_run=False)
    except (OSError, sqlite3.Error, RuntimeError, json.JSONDecodeError) as exc:
        _restore(store, saved)
        return _fail("recovery_not_applied", dry_run=False, error=type(exc).__name__)
    paper_after = (root_p / PAPER_LEDGER_RELATIVE).read_bytes()
    accounting_after = (root_p / ACCOUNTING_LEDGER_RELATIVE).read_bytes()
    gate = evaluate_manual_exit_close_gate(store)
    if paper_after != paper_before or accounting_after != accounting_before or not gate.get("ok"):
        _restore(store, saved)
        return _fail("recovery_not_verified", dry_run=False, gate=gate.get("reason"))
    consumed = _consume_manifest(root_p)
    if not consumed.get("ok"):
        return consumed
    return {
        "ok": True,
        "wrote": True,
        "dry_run": False,
        "reason": "recovered",
        "broker_called": False,
        "executor_enabled": True,
        "write_mode_enabled": True,
        "manifest_consumed": True,
        "exclusive_access": True,
        "ledger_appended": False,
        "changes": assessed["changes"],
        "verified_broker_order_ids": [
            _OCT15["broker_order_id"],
            _OCT16_400["broker_order_id"],
            _OCT16_19["broker_order_id"],
        ],
    }


def execute_legacy_store_recovery(**_ignored: Any) -> Dict[str, Any]:
    """Refuse every write. The module flags stay false."""
    del _ignored
    return _fail("executor_not_enabled", dry_run=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"--write", "--execute", "--apply"} for arg in args):
        print(json.dumps(execute_legacy_store_recovery(), sort_keys=True))
        return 2
    parser = argparse.ArgumentParser(description="Dry-run the manual-exit store alignment.")
    parser.add_argument("--root", type=Path, required=True)
    parsed = parser.parse_args(args)
    report = dry_run_october15_16_manual_exit_store_recovery(root=parsed.root)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
