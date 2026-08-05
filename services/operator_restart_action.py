"""
operator_restart_action.py
Phase DO-006 — Protected RESTART_TRITON orchestration.

Orchestrates two separately verified transactions:
  1) DO-005 protected STOP_TRITON (cooperative shutdown) when Stop is required
  2) Fresh DO-003 START_TRITON only after independently verified full Stop completion

Never force-terminates. Never deletes/repairs locks. Never reuses Restart token as Start authority.
Never accepts frontend PIDs, paths, commands, targets, env, or timeouts.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from services.operator_cooperative_shutdown import (
    CONTROLLER_VERSION,
    SCHEMA_VERSION,
    _iso,
    _sanitize,
    _utc_now,
    atomic_write_json,
    is_guard_active,
    read_json,
    recommended_ack_timeout_sec,
)
from services.operator_runtime_controller import (
    ACTION_START,
    execute_start_triton,
    live_execution_appears_enabled,
    start_eligibility,
)
from services.operator_shutdown_controller import shutdown_active
from services.operator_stop_action import (
    ACTION_STOP,
    CONFIRMATION_PHRASE as STOP_PHRASE,
    ELIGIBLE_STOP_DECISIONS,
    execute_stop_triton,
    issue_stop_preflight,
    stop_eligibility,
    _identity_fingerprint as _stop_identity_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
RESULTS = ROOT / "data" / "results"

ACTION_RESTART = "RESTART_TRITON"
ALLOWED_ACTIONS = frozenset({ACTION_RESTART})
CONFIRMATION_PHRASE = "RESTART TRITON"
PREFLIGHT_TTL_SEC = 120

TOKEN_STORE = LIVE / "operator_restart_preflight_tokens.json"
IDEMPOTENCY_STORE = LIVE / "operator_restart_idempotency.json"
GUARD_PATH = LIVE / "operator_restart_action.guard.json"
TX_PATH = LIVE / "operator_restart_transaction.json"
AUDIT_PATH = RESULTS / "operator_restart_actions.jsonl"
GUARD_TTL_SEC = 600

DENIED_FRONTEND_FIELDS = frozenset(
    {
        "target",
        "targets",
        "target_components",
        "pid",
        "pids",
        "path",
        "command",
        "script",
        "arguments",
        "args",
        "environment",
        "env",
        "timeout",
        "timeout_sec",
        "shutdown_mode",
        "start_mode",
        "controller_operation",
        "broker",
        "credentials",
        "force",
        "override",
        "scheduled_task",
        "task_name",
    }
)

ELIGIBLE_RESTART_DECISIONS = frozenset(
    {
        "RESTART_ELIGIBLE_RUNNING",
        "RESTART_ELIGIBLE_CONTINUOUS_ONLY",
        "RESTART_ELIGIBLE_WATCHDOG_ONLY",
        "RESTART_ALREADY_STOPPED",
    }
)

STOP_COMPLETED_RESULTS = frozenset(
    {
        "STOP_COMPLETED",
        "STOP_ALREADY_STOPPED",
    }
)

START_SUCCESS_RESULTS = frozenset({"STARTED", "ALREADY_RUNNING"})
START_PARTIAL_RESULTS = frozenset({"PARTIAL_START", "POSTCONDITION_FAILED"})


def _hash_token(token: str) -> str:
    return hashlib.sha256(f"restart-token:{token}".encode("utf-8")).hexdigest()


def _hash_nonce(nonce: str) -> str:
    return hashlib.sha256(f"restart-nonce:{nonce}".encode("utf-8")).hexdigest()


def append_restart_audit(record: Dict[str, Any], *, path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or AUDIT_PATH
    safe = _sanitize(record)
    for k in ("preflight_token", "confirmation_token", "token", "raw_token", "confirmation_phrase"):
        if k in safe:
            safe[k] = "<redacted>"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
        return {"ok": True, "path": str(path)}
    except OSError as exc:
        return {"ok": False, "error": f"audit_write_failed:{exc}", "path": str(path)}


def read_recent_restart_actions(
    *, path: Optional[Path] = None, limit: int = 20
) -> List[Dict[str, Any]]:
    path = path or AUDIT_PATH
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: List[Dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
            if isinstance(doc, dict):
                out.append(_sanitize(doc))
        except Exception:
            continue
    out.reverse()
    return out


def _load_token_store(path: Path) -> Dict[str, Any]:
    doc = read_json(path)
    if not doc or doc.get("__malformed__"):
        return {"tokens": {}}
    if "tokens" not in doc or not isinstance(doc["tokens"], dict):
        return {"tokens": {}}
    return doc


def _save_token_store(path: Path, doc: Dict[str, Any]) -> Dict[str, Any]:
    return atomic_write_json(path, doc)


def _idempotency_lookup(nonce: str, *, path: Path) -> Optional[Dict[str, Any]]:
    doc = read_json(path)
    if not doc or doc.get("__malformed__"):
        return None
    entries = doc.get("entries") if isinstance(doc.get("entries"), dict) else {}
    rec = entries.get(_hash_nonce(nonce))
    return rec if isinstance(rec, dict) else None


def _idempotency_store(nonce: str, result: Dict[str, Any], *, path: Path) -> None:
    doc = read_json(path)
    if not doc or doc.get("__malformed__") or not isinstance(doc, dict):
        doc = {"entries": {}}
    entries = doc.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        doc["entries"] = entries
    entries[_hash_nonce(nonce)] = {
        "stored_at": _iso(_utc_now()),
        "result": _sanitize(result),
    }
    if len(entries) > 50:
        for k in list(entries.keys())[:-40]:
            entries.pop(k, None)
    atomic_write_json(path, doc)


def acquire_restart_guard(
    *,
    action_id: str,
    path: Path,
    ttl_sec: int = GUARD_TTL_SEC,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now = now or _utc_now()
    if is_guard_active(path, now=now):
        return {"acquired": False, "reason": "restart_guard_active"}
    doc = {
        "schema_version": SCHEMA_VERSION,
        "action_id": action_id,
        "action": ACTION_RESTART,
        "acquired_at": _iso(now),
        "expires_at": _iso(now + timedelta(seconds=ttl_sec)),
    }
    wr = atomic_write_json(path, doc)
    if not wr.get("ok"):
        return {"acquired": False, "reason": "guard_write_failed"}
    return {"acquired": True, "guard": doc}


def release_restart_guard(*, action_id: str, path: Path) -> None:
    doc = read_json(path)
    if not doc or doc.get("__malformed__"):
        return
    if str(doc.get("action_id") or "") != action_id:
        return
    # Expire rather than delete — preserve evidence
    doc["expires_at"] = _iso(_utc_now() - timedelta(seconds=1))
    doc["released_at"] = _iso(_utc_now())
    atomic_write_json(path, doc)


def write_restart_transaction(doc: Dict[str, Any], *, path: Path) -> Dict[str, Any]:
    return atomic_write_json(path, _sanitize(doc))


def read_restart_transaction(*, path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or TX_PATH
    doc = read_json(path)
    if not doc or doc.get("__malformed__"):
        return {}
    return doc if isinstance(doc, dict) else {}


def evaluate_restart_decision(
    stop_decision: Dict[str, Any],
    *,
    live_execution_blocked: bool = True,
    start_guard_active: bool = False,
    shutdown_guard_active: bool = False,
    restart_guard_active: bool = False,
    unresolved_shutdown: bool = False,
    unresolved_restart: bool = False,
) -> Dict[str, Any]:
    """Explicit Restart decision matrix (dashboard-facing)."""
    if not live_execution_blocked:
        return {
            "decision": "RESTART_REFUSED_LIVE_SAFETY",
            "restart_button_enabled": False,
            "stop_required": False,
            "target_components": [],
            "explanation": "Restart refused: live execution appears enabled.",
            "ui_state": "RESTART_UNAVAILABLE",
            "safe_to_retry": False,
        }
    if restart_guard_active or unresolved_restart:
        return {
            "decision": (
                "RESTART_IN_PROGRESS"
                if restart_guard_active
                else "RESTART_REFUSED_UNRESOLVED_TRANSACTION"
            ),
            "restart_button_enabled": False,
            "stop_required": False,
            "target_components": [],
            "explanation": "A Restart transaction is already active or unresolved.",
            "ui_state": "RESTART_IN_PROGRESS",
            "safe_to_retry": False,
        }
    if start_guard_active:
        return {
            "decision": "RESTART_REFUSED_START_CONFLICT",
            "restart_button_enabled": False,
            "stop_required": False,
            "target_components": [],
            "explanation": "Restart unavailable while a Start transaction is active.",
            "ui_state": "RESTART_UNAVAILABLE",
            "safe_to_retry": True,
        }
    if shutdown_guard_active or unresolved_shutdown:
        return {
            "decision": (
                "RESTART_REFUSED_STOP_CONFLICT"
                if shutdown_guard_active
                else "RESTART_REFUSED_UNRESOLVED_TRANSACTION"
            ),
            "restart_button_enabled": False,
            "stop_required": False,
            "target_components": [],
            "explanation": "Restart unavailable while Stop/shutdown is active or unresolved.",
            "ui_state": (
                "RESTART_STOP_IN_PROGRESS" if shutdown_guard_active else "RESTART_UNAVAILABLE"
            ),
            "safe_to_retry": True,
        }

    stop_d = str(stop_decision.get("decision") or "UNKNOWN")
    targets = list(stop_decision.get("target_components") or [])

    if stop_d == "STOP_ALREADY_STOPPED":
        return {
            "decision": "RESTART_ALREADY_STOPPED",
            "restart_button_enabled": True,
            "stop_required": False,
            "target_components": [],
            "explanation": (
                "Runtime components appear stopped. Restart may proceed to a fresh "
                "DO-003 Start eligibility evaluation after confirmation."
            ),
            "ui_state": "RESTART_CONFIRMATION_REQUIRED",
            "safe_to_retry": True,
        }

    mapping = {
        "STOP_ELIGIBLE_BOTH": "RESTART_ELIGIBLE_RUNNING",
        "STOP_ELIGIBLE_CONTINUOUS_ONLY": "RESTART_ELIGIBLE_CONTINUOUS_ONLY",
        "STOP_ELIGIBLE_WATCHDOG_ONLY": "RESTART_ELIGIBLE_WATCHDOG_ONLY",
        "STOP_ELIGIBLE_UNHEALTHY_VERIFIED": "RESTART_ELIGIBLE_RUNNING",
        "STOP_IN_PROGRESS": "RESTART_STOP_IN_PROGRESS",
        "STOP_REFUSED_START_IN_PROGRESS": "RESTART_REFUSED_START_CONFLICT",
        "STOP_REFUSED_LIVE_SAFETY": "RESTART_REFUSED_LIVE_SAFETY",
        "STOP_REFUSED_UNSAFE_STATE": "RESTART_REFUSED_UNSAFE_STATE",
        "STOP_REFUSED_SHUTDOWN_UNRESOLVED": "RESTART_REFUSED_UNRESOLVED_TRANSACTION",
    }
    if stop_d in mapping:
        decision = mapping[stop_d]
        eligible = decision in ELIGIBLE_RESTART_DECISIONS
        return {
            "decision": decision,
            "restart_button_enabled": eligible,
            "stop_required": eligible and stop_d != "STOP_ALREADY_STOPPED",
            "target_components": targets if eligible else [],
            "explanation": stop_decision.get("explanation")
            or ("Restart eligible." if eligible else "Restart unavailable."),
            "ui_state": "RESTART_CONFIRMATION_REQUIRED" if eligible else "RESTART_UNAVAILABLE",
            "safe_to_retry": True,
            "unhealthy_verified": stop_d == "STOP_ELIGIBLE_UNHEALTHY_VERIFIED",
        }

    return {
        "decision": "RESTART_REFUSED_UNSAFE_STATE",
        "restart_button_enabled": False,
        "stop_required": False,
        "target_components": [],
        "explanation": stop_decision.get("explanation")
        or "Restart refused: unsafe or unknown runtime classification.",
        "ui_state": "RESTART_UNAVAILABLE",
        "safe_to_retry": False,
    }


def restart_eligibility(
    *,
    root: Optional[Path] = None,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    root = Path(root) if root is not None else ROOT
    live = root / "data" / "live"
    now = _utc_now()
    stop_el = stop_eligibility(root=root, process_snapshot=process_snapshot)
    live_blocked = bool(stop_el.get("live_execution_blocked", True))
    start_g = is_guard_active(live / "operator_start_action.guard.json", now=now)
    shut_g = is_guard_active(live / "operator_shutdown_action.guard.json", now=now)
    restart_g = is_guard_active(live / "operator_restart_action.guard.json", now=now)
    active = shutdown_active(root=root, now=now)
    tx = read_restart_transaction(path=live / "operator_restart_transaction.json")
    unresolved_restart = bool(tx.get("unresolved")) and not restart_g

    decision = evaluate_restart_decision(
        stop_el,
        live_execution_blocked=live_blocked,
        start_guard_active=start_g,
        shutdown_guard_active=shut_g or bool(active.get("active")),
        restart_guard_active=restart_g,
        unresolved_shutdown=bool(active.get("active")),
        unresolved_restart=unresolved_restart,
    )
    return {
        **decision,
        "stop_decision": stop_el.get("decision"),
        "stop_eligibility": {
            "decision": stop_el.get("decision"),
            "stop_button_enabled": stop_el.get("stop_button_enabled"),
            "target_components": stop_el.get("target_components"),
        },
        "live_execution_blocked": live_blocked,
        "start_guard_active": start_g,
        "shutdown_guard_active": shut_g,
        "restart_guard_active": restart_g,
        "shutdown_active": active,
        "protected_restart_exposed": True,
        "confirmation_phrase_required": CONFIRMATION_PHRASE,
        "preflight_token_required": True,
        "allowed_dashboard_action": ACTION_RESTART,
        "stop_backend_operation": "PREPARE_COOPERATIVE_SHUTDOWN",
        "start_backend_operation": ACTION_START,
        "classifications": stop_el.get("classifications"),
        "recommended_action": (
            "Confirm Restart with exact phrase RESTART TRITON when eligible."
            if decision.get("restart_button_enabled")
            else "Investigate Diagnostics before Restart."
        ),
    }


def issue_restart_preflight(
    *,
    root: Optional[Path] = None,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
    ttl_sec: int = PREFLIGHT_TTL_SEC,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    from services.operator_runtime_classifier import classify_runtime_loops

    root = Path(root) if root is not None else ROOT
    live = root / "data" / "live"
    now = now or _utc_now()
    classifications = classify_runtime_loops(root=root, process_snapshot=process_snapshot)
    el = restart_eligibility(root=root, process_snapshot=process_snapshot)
    targets = list(el.get("target_components") or [])
    identity = _identity_fingerprint(classifications, targets or ["continuous", "watchdog"])
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash_token(raw_token)
    record = {
        "schema_version": SCHEMA_VERSION,
        "token_hash": token_hash,
        "action": ACTION_RESTART,
        "decision": el.get("decision"),
        "stop_required": bool(el.get("stop_required")),
        "target_components": targets,
        "identity": identity,
        "live_execution_blocked": el.get("live_execution_blocked"),
        "start_guard_active": el.get("start_guard_active"),
        "shutdown_guard_active": el.get("shutdown_guard_active"),
        "restart_guard_active": el.get("restart_guard_active"),
        "issued_at": _iso(now),
        "expires_at": _iso(now + timedelta(seconds=ttl_sec)),
        "used": False,
        "nonce": str(uuid.uuid4()),
        "controller_version": CONTROLLER_VERSION,
    }
    store_path = live / "operator_restart_preflight_tokens.json"
    store = _load_token_store(store_path)
    pruned: Dict[str, Any] = {}
    for h, rec in (store.get("tokens") or {}).items():
        try:
            exp = datetime.fromisoformat(str(rec.get("expires_at")).replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if exp > now and not rec.get("used"):
                pruned[h] = rec
        except Exception:
            continue
    pruned[token_hash] = record
    store["tokens"] = pruned
    wr = _save_token_store(store_path, store)
    if not wr.get("ok"):
        return {
            "ok": False,
            "error": "token_store_failed",
            "decision": el.get("decision"),
            "restart_button_enabled": False,
        }
    return {
        "ok": True,
        "preflight_token": raw_token,
        "preflight_nonce": record["nonce"],
        "expires_at": record["expires_at"],
        "issued_at": record["issued_at"],
        "decision": el.get("decision"),
        "explanation": el.get("explanation"),
        "stop_required": record["stop_required"],
        "target_components": targets,
        "restart_button_enabled": bool(el.get("restart_button_enabled")),
        "ui_state": el.get("ui_state"),
        "live_execution_blocked": el.get("live_execution_blocked"),
        "identity_bound": True,
        "token_presented": True,
        "allowed_dashboard_action": ACTION_RESTART,
        "safe_message": (
            "Restart TRITON uses two separately verified transactions: a cooperative Stop "
            "followed by a fresh protected Start. Restart does not force-terminate processes, "
            "delete locks, bypass Start eligibility, arm paper execution, or enable live execution."
        ),
    }


def _consume_restart_token(
    raw_token: str,
    *,
    store_path: Path,
    now: datetime,
) -> Dict[str, Any]:
    store = _load_token_store(store_path)
    th = _hash_token(raw_token)
    rec = (store.get("tokens") or {}).get(th)
    if not isinstance(rec, dict):
        return {"ok": False, "reason": "STOP_REFUSED_CONFIRMATION", "detail": "token_unknown"}
    if rec.get("action") != ACTION_RESTART:
        return {"ok": False, "reason": "RESTART_REFUSED_CONFIRMATION", "detail": "action_mismatch"}
    if rec.get("used"):
        return {"ok": False, "reason": "RESTART_REFUSED_REPLAY", "detail": "token_already_used"}
    try:
        exp = datetime.fromisoformat(str(rec.get("expires_at")).replace("Z", "+00:00"))
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp <= now:
            return {
                "ok": False,
                "reason": "RESTART_REFUSED_CONFIRMATION_EXPIRED",
                "detail": "token_expired",
            }
    except Exception:
        return {
            "ok": False,
            "reason": "RESTART_REFUSED_CONFIRMATION",
            "detail": "token_expiry_invalid",
        }
    rec["used"] = True
    rec["used_at"] = _iso(now)
    store["tokens"][th] = rec
    wr = _save_token_store(store_path, store)
    if not wr.get("ok"):
        return {"ok": False, "reason": "RESTART_AUDIT_FAILED", "detail": "token_mark_failed"}
    return {"ok": True, "record": rec}


def _identity_fingerprint(classifications: Dict[str, Any], targets: List[str]) -> Dict[str, Any]:
    """Restart identity uses the same stable fingerprint as protected Stop."""
    return _stop_identity_fingerprint(classifications, targets)


def _identities_match(
    expected: Dict[str, Any], current: Dict[str, Any], targets: List[str]
) -> bool:
    """Stable identity only — heartbeat/lock_refreshed_at must not refuse Restart."""
    from services.operator_stop_action import _identities_match as _stop_match

    return _stop_match(expected, current, list(targets or expected.keys()))


def _verify_stop_postconditions(
    stop_result: Dict[str, Any],
    *,
    stop_required: bool,
    root: Path,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Independently verify Stop completed before Start evaluation."""
    result = str(stop_result.get("result") or "")
    if not stop_required:
        if result in STOP_COMPLETED_RESULTS or result == "":
            return {"ok": True, "reason": "stop_not_required"}
    if result == "STOP_TIMED_OUT":
        return {"ok": False, "advance": False, "mapped": "RESTART_TIMED_OUT"}
    if result in ("STOP_PARTIAL_COMPLETION",):
        return {"ok": False, "advance": False, "mapped": "RESTART_PARTIAL_COMPLETION"}
    if result == "STOP_AUDIT_FAILED":
        return {"ok": False, "advance": False, "mapped": "RESTART_AUDIT_FAILED"}
    if result not in STOP_COMPLETED_RESULTS:
        return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}

    if stop_result.get("forced_termination_used") is True:
        return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}
    if stop_result.get("runtime_locks_deleted_by_controller") is True:
        return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}
    if stop_result.get("runtime_locks_repaired_by_controller") is True:
        return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}

    ack = stop_result.get("acknowledgement_results") or {}
    exits = stop_result.get("process_exit_results") or {}
    locks = stop_result.get("lock_release_results") or {}

    # Acknowledgement alone never authorizes Start.
    if isinstance(ack, dict) and ack and result == "STOP_COMPLETED":
        if not isinstance(exits, dict) or not exits:
            return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}
        if not isinstance(locks, dict) or not locks:
            return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}

    for _comp, er in exits.items() if isinstance(exits, dict) else []:
        if isinstance(er, dict) and er.get("exited") is False:
            return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}
    for _comp, lr in locks.items() if isinstance(locks, dict) else []:
        if isinstance(lr, dict) and lr.get("released") is False:
            return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}

    # When exit/lock evidence is supplied, both postconditions must be present and true.
    if stop_required and (exits or locks):
        if not exits or not locks:
            return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}
        for _comp, er in exits.items():
            if not (isinstance(er, dict) and er.get("exited") is True):
                return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}
        for _comp, lr in locks.items():
            if not (isinstance(lr, dict) and lr.get("released") is True):
                return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}

    active = shutdown_active(root=root)
    if active.get("active"):
        return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}

    # Fresh classification must show no conflicting remnants for targeted components
    stop_el = stop_eligibility(root=root, process_snapshot=process_snapshot)
    if stop_el.get("decision") not in ("STOP_ALREADY_STOPPED",) and stop_required:
        # After successful stop, expect already stopped
        if stop_el.get("decision") in ELIGIBLE_STOP_DECISIONS or stop_el.get("stop_button_enabled"):
            return {"ok": False, "advance": False, "mapped": "RESTART_REFUSED_STOP_INCOMPLETE"}

    return {"ok": True, "mapped": "STOP_COMPLETED"}


def _start_has_explicit_timeout_evidence(start_result: Dict[str, Any]) -> bool:
    """True only when DO-003 Start (or an injected clock) reports verified timeout evidence.

    Ordinary START_IN_PROGRESS (e.g. concurrent Start guard) is not a timeout.
    """
    if start_result.get("timed_out") is True or start_result.get("start_timed_out") is True:
        return True
    result = str(start_result.get("result") or "")
    if result in ("START_TIMED_OUT",):
        return True
    error_code = str(start_result.get("error_code") or "")
    if error_code in (
        "START_TIMED_OUT",
        "POSTCONDITION_TIMEOUT",
        "START_POSTCONDITION_TIMEOUT",
    ):
        return True
    for key in ("reason", "postcondition_reason", "safe_message"):
        text = str(start_result.get(key) or "").lower()
        if "postcondition_timeout" in text or "start_timed_out" in text:
            return True
    # DO-003 embeds postcondition wait timeout under component_results.*.postcondition.reason
    for cres in (start_result.get("component_results") or {}).values():
        if not isinstance(cres, dict):
            continue
        pc = cres.get("postcondition") or {}
        if str(pc.get("reason") or "") == "postcondition_timeout":
            return True
        if pc.get("timed_out") is True:
            return True
    return False


def _map_start_to_restart(start_result: Dict[str, Any]) -> str:
    result = str(start_result.get("result") or "")
    if result in START_SUCCESS_RESULTS:
        return "RESTART_COMPLETED"
    # Explicit timeout evidence wins over generic START_IN_PROGRESS / POSTCONDITION_FAILED labels.
    if _start_has_explicit_timeout_evidence(start_result):
        return "RESTART_TIMED_OUT"
    if result in START_PARTIAL_RESULTS:
        return "RESTART_PARTIAL_COMPLETION"
    if result.startswith("START_REFUSED"):
        return "RESTART_REFUSED_START_INELIGIBLE"
    if result == "START_IN_PROGRESS":
        # Ordinary non-expired in-progress Start (e.g. concurrent guard) — not a timeout.
        return "RESTART_PARTIAL_COMPLETION"
    if result == "START_FAILED":
        return "RESTART_REFUSED"
    return "RESTART_REFUSED"


def _verify_start_postconditions(
    start_result: Dict[str, Any],
    classifications: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify replacement process/lock ownership before RESTART_COMPLETED."""
    result = str(start_result.get("result") or "")
    if result not in START_SUCCESS_RESULTS:
        return {"ok": False, "mapped": _map_start_to_restart(start_result)}

    launched = list(start_result.get("launched_components") or [])
    check = launched or ["continuous", "watchdog"]
    for comp in check:
        loop = classifications.get(comp) or {}
        pids = list(loop.get("process_pids") or [])
        lock_exists = bool(loop.get("lock_exists"))
        lock_pid = loop.get("lock_pid")
        if not pids:
            if result == "ALREADY_RUNNING" and not launched:
                continue
            if launched and comp in launched:
                return {
                    "ok": False,
                    "mapped": "RESTART_PARTIAL_COMPLETION",
                    "reason": f"{comp}_process_missing_after_start",
                }
            continue
        if len(pids) > 1:
            return {
                "ok": False,
                "mapped": "RESTART_PARTIAL_COMPLETION",
                "reason": f"{comp}_duplicate_replacement_process",
            }
        if not lock_exists:
            return {
                "ok": False,
                "mapped": "RESTART_PARTIAL_COMPLETION",
                "reason": f"{comp}_lock_missing_after_start",
            }
        if lock_pid not in pids:
            return {
                "ok": False,
                "mapped": "RESTART_PARTIAL_COMPLETION",
                "reason": f"{comp}_lock_pid_mismatch_after_start",
            }
    return {"ok": True, "mapped": "RESTART_COMPLETED"}


def controller_has_termination_paths() -> bool:
    src = Path(__file__).read_text(encoding="utf-8")
    kill_call = "os" + "." + "kill" + "("
    proc_kill = "process" + "." + "kill" + "("
    taskkill = "task" + "kill" + ".exe"
    stop_proc = "Stop" + "-" + "Process" + " -"
    popen = "P" + "open("
    subproc_import = "import " + "subprocess"
    return any(d in src for d in (kill_call, proc_kill, taskkill, stop_proc, popen, subproc_import))


def execute_restart_triton(
    *,
    action: str = ACTION_RESTART,
    confirmation_phrase: str = "",
    confirmation_checked: bool = False,
    preflight_token: Optional[str] = None,
    request_nonce: Optional[str] = None,
    source: str = "operator_console",
    root: Optional[Path] = None,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
    stop_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    start_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    issue_stop_preflight_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ack_timeout_sec: float = float(recommended_ack_timeout_sec()),
    exit_timeout_sec: float = 60.0,
    lock_timeout_sec: float = 30.0,
    poll_sec: float = 0.5,
    sleep_fn: Optional[Callable[[float], None]] = None,
    # Explicitly rejected
    target: Any = None,
    targets: Any = None,
    target_components: Any = None,
    pid: Any = None,
    path: Any = None,
    command: Any = None,
    script: Any = None,
    arguments: Any = None,
    env: Any = None,
    environment: Any = None,
    timeout: Any = None,
    shutdown_mode: Any = None,
    start_mode: Any = None,
    controller_operation: Any = None,
    force: Any = None,
    override: Any = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Typed RESTART_TRITON — Stop (if needed) then fresh Start as separate transactions."""
    from services.operator_runtime_classifier import classify_runtime_loops

    root = Path(root) if root is not None else ROOT
    live = root / "data" / "live"
    results = root / "data" / "results"
    audit_path = results / "operator_restart_actions.jsonl"
    token_store = live / "operator_restart_preflight_tokens.json"
    idem_path = live / "operator_restart_idempotency.json"
    guard_path = live / "operator_restart_action.guard.json"
    tx_path = live / "operator_restart_transaction.json"
    now = _utc_now()
    action_id = str(uuid.uuid4())
    stages: List[Dict[str, Any]] = []

    def _stage(name: str, **kwargs: Any) -> None:
        stages.append({"stage": name, "at": _iso(_utc_now()), **kwargs})

    def _base(**kwargs: Any) -> Dict[str, Any]:
        out = {
            "schema_version": SCHEMA_VERSION,
            "action_id": action_id,
            "restart_request_id": action_id,
            "action": ACTION_RESTART,
            "source": source,
            "controller_version": CONTROLLER_VERSION,
            "allowed_orchestration": [
                "PREPARE_COOPERATIVE_SHUTDOWN",
                ACTION_START,
            ],
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "runtime_locks_repaired_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "orders_cancelled": False,
            "approvals_created": False,
            "authorizations_created": False,
            "scheduled_task_modified": False,
            "governance_modified": False,
            "lifecycle_modified": False,
            "allocation_modified": False,
            "risk_modified": False,
            "automatic_start_before_stop_completion": False,
            "restart_token_reused_for_start": False,
            "stages": list(stages),
        }
        out.update(kwargs)
        return out

    injected = any(
        v is not None
        for v in (
            target,
            targets,
            target_components,
            pid,
            path,
            command,
            script,
            arguments,
            env,
            environment,
            timeout,
            shutdown_mode,
            start_mode,
            controller_operation,
            force,
            override,
        )
    ) or bool(extra)
    if injected:
        rec = _base(
            result="RESTART_REFUSED_CONFIRMATION",
            error_code="FRONTEND_INJECTION_REFUSED",
            safe_message=(
                "Restart refused: frontend must not supply targets, PID, path, command, "
                "environment, timeout, modes, force, or override values."
            ),
            safe_to_retry=False,
        )
        append_restart_audit(rec, path=audit_path)
        return rec

    if action != ACTION_RESTART:
        return _base(
            result="RESTART_REFUSED_CONFIRMATION",
            error_code="ACTION_NOT_ALLOWED",
            safe_message="Restart refused: only RESTART_TRITON is permitted.",
            safe_to_retry=False,
        )

    if not confirmation_checked:
        return _base(
            result="RESTART_REFUSED_CONFIRMATION",
            error_code="CHECKBOX_REQUIRED",
            safe_message="Restart refused: confirmation checkbox is required.",
            safe_to_retry=True,
        )

    if confirmation_phrase != CONFIRMATION_PHRASE:
        return _base(
            result="RESTART_REFUSED_CONFIRMATION",
            error_code="PHRASE_MISMATCH",
            safe_message="Restart refused: confirmation phrase must match exactly: RESTART TRITON",
            safe_to_retry=True,
        )

    if not preflight_token:
        return _base(
            result="RESTART_REFUSED_CONFIRMATION",
            error_code="TOKEN_REQUIRED",
            safe_message="Restart refused: missing preflight token.",
            safe_to_retry=True,
        )

    nonce = request_nonce or str(uuid.uuid4())
    prior = _idempotency_lookup(nonce, path=idem_path)
    if prior and isinstance(prior.get("result"), dict):
        replayed = dict(prior["result"])
        replayed["replayed"] = True
        replayed["safe_message"] = (
            replayed.get("safe_message")
            or "Duplicate Restart submission with same nonce — prior result returned."
        )
        return replayed

    consumed = _consume_restart_token(preflight_token, store_path=token_store, now=now)
    if not consumed.get("ok"):
        reason = str(consumed.get("reason") or "RESTART_REFUSED_CONFIRMATION")
        rec = _base(
            result=reason,
            error_code=str(consumed.get("detail") or "token_invalid"),
            safe_message=f"Restart refused: {reason}",
            safe_to_retry=reason.endswith("EXPIRED"),
            confirmation_valid=True,
            token_valid=False,
        )
        append_restart_audit(rec, path=audit_path)
        _idempotency_store(nonce, rec, path=idem_path)
        return rec

    token_rec = consumed["record"]
    _stage("RESTART_PREFLIGHT_CREATED")
    _stage("RESTART_CONFIRMED")

    live_blocked_before = not live_execution_appears_enabled(root=root)
    classifications = classify_runtime_loops(root=root, process_snapshot=process_snapshot)
    el = restart_eligibility(root=root, process_snapshot=process_snapshot)
    expected_targets = list(token_rec.get("target_components") or [])
    current_identity = _identity_fingerprint(
        classifications, expected_targets or ["continuous", "watchdog"]
    )
    expected_identity = token_rec.get("identity") or {}

    if not live_blocked_before:
        rec = _base(
            result="RESTART_REFUSED_LIVE_SAFETY",
            safe_message="Restart refused: live execution appears enabled.",
            live_execution_blocked_before=False,
            live_execution_blocked_after=False,
        )
        append_restart_audit(rec, path=audit_path)
        _idempotency_store(nonce, rec, path=idem_path)
        return rec

    if el.get("decision") in (
        "RESTART_REFUSED_START_CONFLICT",
        "RESTART_REFUSED_STOP_CONFLICT",
        "RESTART_REFUSED_UNRESOLVED_TRANSACTION",
        "RESTART_IN_PROGRESS",
    ):
        rec = _base(
            result=str(el.get("decision")),
            safe_message=el.get("explanation") or "Restart refused due to conflicting transaction.",
            live_execution_blocked_before=True,
            live_execution_blocked_after=True,
        )
        append_restart_audit(rec, path=audit_path)
        _idempotency_store(nonce, rec, path=idem_path)
        return rec

    stop_required = bool(token_rec.get("stop_required"))
    if stop_required and not _identities_match(
        expected_identity, current_identity, expected_targets
    ):
        rec = _base(
            result="RESTART_REFUSED_STATE_CHANGED",
            state_changed=True,
            safe_message=(
                "Restart refused: verified process/lock identity changed since confirmation."
            ),
            live_execution_blocked_before=True,
            live_execution_blocked_after=True,
            preflight_classification=token_rec.get("decision"),
            submission_classification=el.get("decision"),
        )
        append_restart_audit(rec, path=audit_path)
        _idempotency_store(nonce, rec, path=idem_path)
        return rec

    if str(el.get("decision") or "") not in ELIGIBLE_RESTART_DECISIONS and not (
        str(token_rec.get("decision") or "") in ELIGIBLE_RESTART_DECISIONS
        and str(el.get("decision") or "") == "RESTART_ALREADY_STOPPED"
    ):
        # Allow already-stopped path; otherwise refuse state change / unsafe
        if str(el.get("decision") or "") != str(token_rec.get("decision") or ""):
            rec = _base(
                result="RESTART_REFUSED_STATE_CHANGED",
                state_changed=True,
                safe_message="Restart refused: eligibility changed since confirmation.",
                live_execution_blocked_before=True,
                live_execution_blocked_after=True,
            )
            append_restart_audit(rec, path=audit_path)
            _idempotency_store(nonce, rec, path=idem_path)
            return rec
        if str(el.get("decision") or "") not in ELIGIBLE_RESTART_DECISIONS:
            rec = _base(
                result="RESTART_REFUSED_UNSAFE_STATE",
                safe_message=el.get("explanation") or "Restart refused: not eligible.",
                live_execution_blocked_before=True,
                live_execution_blocked_after=True,
            )
            append_restart_audit(rec, path=audit_path)
            _idempotency_store(nonce, rec, path=idem_path)
            return rec

    # Initial audit must succeed before Stop
    initial = _base(
        result="RESTART_SUBMITTED",
        stage="RESTART_SUBMITTED",
        confirmation_valid=True,
        token_valid=True,
        token_expired=False,
        replay_detected=False,
        stop_required=stop_required,
        stop_target_components=expected_targets,
        preflight_classification=token_rec.get("decision"),
        submission_classification=el.get("decision"),
        expected_instances=_sanitize(expected_identity),
        live_execution_blocked_before=True,
        request_nonce_hash=_hash_nonce(nonce),
    )
    _stage("RESTART_SUBMITTED")
    audit_wr = append_restart_audit(initial, path=audit_path)
    if not audit_wr.get("ok"):
        rec = _base(
            result="RESTART_AUDIT_FAILED",
            error_code="INITIAL_AUDIT_FAILED",
            safe_message="Restart refused: initial audit persistence failed before Stop delivery.",
            live_execution_blocked_before=True,
            live_execution_blocked_after=True,
        )
        _idempotency_store(nonce, rec, path=idem_path)
        return rec

    guard = acquire_restart_guard(action_id=action_id, path=guard_path)
    if not guard.get("acquired"):
        rec = _base(
            result="RESTART_IN_PROGRESS",
            safe_message="Restart already in progress (guard active).",
            live_execution_blocked_before=True,
            live_execution_blocked_after=True,
        )
        append_restart_audit(rec, path=audit_path)
        _idempotency_store(nonce, rec, path=idem_path)
        return rec

    tx = {
        "schema_version": SCHEMA_VERSION,
        "action_id": action_id,
        "action": ACTION_RESTART,
        "unresolved": True,
        "stage": "RESTART_SUBMITTED",
        "stages": stages,
        "stop_required": stop_required,
        "stop_target_components": expected_targets,
        "created_at": _iso(now),
    }
    write_restart_transaction(tx, path=tx_path)

    stop_result: Dict[str, Any] = {
        "result": "STOP_ALREADY_STOPPED",
        "forced_termination_used": False,
        "runtime_locks_deleted_by_controller": False,
        "runtime_locks_repaired_by_controller": False,
    }
    stop_action_id = None
    stop_request_id = None

    try:
        if stop_required:
            _stage("STOP_PREFLIGHT_REVALIDATED")
            issue_fn = issue_stop_preflight_fn or issue_stop_preflight
            stop_pre = issue_fn(root=root, process_snapshot=process_snapshot)
            if not stop_pre.get("ok"):
                rec = _base(
                    result="RESTART_REFUSED_STOP_INCOMPLETE",
                    safe_message="Restart refused: could not issue DO-005 Stop preflight.",
                    stop_required=True,
                    live_execution_blocked_before=True,
                    live_execution_blocked_after=True,
                    unresolved=True,
                )
                tx.update({"stage": "RESTART_REFUSED", "unresolved": True, "final": rec})
                write_restart_transaction(tx, path=tx_path)
                append_restart_audit(rec, path=audit_path)
                _idempotency_store(nonce, rec, path=idem_path)
                return rec

            _stage("STOP_REQUEST_DELIVERED")
            do_stop = stop_fn or execute_stop_triton
            stop_result = do_stop(
                action=ACTION_STOP,
                confirmation_checked=True,
                confirmation_phrase=STOP_PHRASE,
                preflight_token=stop_pre.get("preflight_token"),
                request_nonce=stop_pre.get("preflight_nonce"),
                source="operator_restart_action",
                root=root,
                process_snapshot=process_snapshot,
                ack_timeout_sec=ack_timeout_sec,
                exit_timeout_sec=exit_timeout_sec,
                lock_timeout_sec=lock_timeout_sec,
                poll_sec=poll_sec,
                sleep_fn=sleep_fn,
            )
            stop_action_id = stop_result.get("action_id")
            stop_request_id = stop_result.get("request_id")
            _stage(
                (
                    "STOP_COMPLETED"
                    if stop_result.get("result") in STOP_COMPLETED_RESULTS
                    else "STOP_WAITING_FOR_PROCESS_EXIT"
                ),
                stop_result=stop_result.get("result"),
            )

            mid_audit = append_restart_audit(
                _base(
                    result="RESTART_STOP_IN_PROGRESS",
                    stop_result=stop_result.get("result"),
                    stop_action_id=stop_action_id,
                    stop_request_id=stop_request_id,
                    process_exit_results=stop_result.get("process_exit_results"),
                    lock_release_results=stop_result.get("lock_release_results"),
                ),
                path=audit_path,
            )
            if not mid_audit.get("ok"):
                # Preserve Stop evidence; do not Start
                rec = _base(
                    result="RESTART_AUDIT_FAILED",
                    error_code="MID_AUDIT_FAILED",
                    safe_message=(
                        "Restart audit failed after Stop delivery. Stop evidence preserved. "
                        "Start was not attempted."
                    ),
                    stop_required=True,
                    stop_result=stop_result.get("result"),
                    stop_action_id=stop_action_id,
                    stop_request_id=stop_request_id,
                    process_exit_results=stop_result.get("process_exit_results"),
                    lock_release_results=stop_result.get("lock_release_results"),
                    forced_termination_used=bool(stop_result.get("forced_termination_used")),
                    live_execution_blocked_before=True,
                    live_execution_blocked_after=not live_execution_appears_enabled(root=root),
                    unresolved=True,
                )
                tx.update({"stage": "RESTART_AUDIT_FAILED", "unresolved": True, "final": rec})
                write_restart_transaction(tx, path=tx_path)
                _idempotency_store(nonce, rec, path=idem_path)
                return rec

        post = _verify_stop_postconditions(
            stop_result,
            stop_required=stop_required,
            root=root,
            process_snapshot=process_snapshot,
        )
        if not post.get("ok"):
            mapped = str(post.get("mapped") or "RESTART_REFUSED_STOP_INCOMPLETE")
            rec = _base(
                result=mapped,
                safe_message=(
                    "Restart did not advance to Start: Stop postconditions incomplete "
                    f"({mapped}). Forced termination was not used."
                ),
                stop_required=stop_required,
                stop_result=stop_result.get("result"),
                stop_action_id=stop_action_id,
                stop_request_id=stop_request_id,
                process_exit_results=stop_result.get("process_exit_results"),
                lock_release_results=stop_result.get("lock_release_results"),
                acknowledgement_results=stop_result.get("acknowledgement_results"),
                forced_termination_used=bool(stop_result.get("forced_termination_used")),
                live_execution_blocked_before=True,
                live_execution_blocked_after=not live_execution_appears_enabled(root=root),
                unresolved=True,
                safe_to_retry=mapped in ("RESTART_TIMED_OUT", "RESTART_PARTIAL_COMPLETION"),
            )
            tx.update({"stage": mapped, "unresolved": True, "final": rec})
            write_restart_transaction(tx, path=tx_path)
            append_restart_audit(rec, path=audit_path)
            _idempotency_store(nonce, rec, path=idem_path)
            return rec

        _stage("STOP_COMPLETED")
        _stage("START_FRESH_CLASSIFICATION")
        start_el = start_eligibility(root=root, process_snapshot=process_snapshot)
        if not start_el.get("start_button_enabled"):
            rec = _base(
                result="RESTART_REFUSED_START_INELIGIBLE",
                safe_message=(
                    "Stop completed (or was not required), but fresh DO-003 Start eligibility "
                    f"failed: {start_el.get('decision')}. TRITON remains stopped."
                ),
                stop_required=stop_required,
                stop_result=stop_result.get("result"),
                stop_action_id=stop_action_id,
                fresh_start_classification=start_el.get("decision"),
                start_eligible=False,
                live_execution_blocked_before=True,
                live_execution_blocked_after=True,
                unresolved=False,
                safe_to_retry=True,
            )
            tx.update(
                {"stage": "RESTART_REFUSED_START_INELIGIBLE", "unresolved": False, "final": rec}
            )
            write_restart_transaction(tx, path=tx_path)
            append_restart_audit(rec, path=audit_path)
            _idempotency_store(nonce, rec, path=idem_path)
            release_restart_guard(action_id=action_id, path=guard_path)
            return rec

        _stage("START_ELIGIBILITY_CONFIRMED")
        start_idem = str(uuid.uuid4())  # fresh Start authority — never reuse Restart token
        _stage("START_TRANSACTION_CREATED", start_idempotency_key_hash=_hash_nonce(start_idem))
        do_start = start_fn or execute_start_triton
        start_result = do_start(
            action=ACTION_START,
            idempotency_key=start_idem,
            source="operator_restart_action",
            operator_context={"ui": "restart_orchestration", "restart_action_id": action_id},
            root=root,
            process_snapshot=process_snapshot,
        )
        live_after = not live_execution_appears_enabled(root=root)
        final_cls = classify_runtime_loops(root=root, process_snapshot=process_snapshot)
        start_post = _verify_start_postconditions(start_result, final_cls)
        mapped = str(start_post.get("mapped") or _map_start_to_restart(start_result))
        _stage("RESTART_COMPLETED" if mapped == "RESTART_COMPLETED" else mapped)

        rec = _base(
            result=mapped,
            start_postcondition_ok=bool(start_post.get("ok")),
            start_postcondition_reason=start_post.get("reason"),
            stop_required=stop_required,
            stop_result=stop_result.get("result"),
            stop_action_id=stop_action_id,
            stop_request_id=stop_request_id,
            process_exit_results=stop_result.get("process_exit_results"),
            lock_release_results=stop_result.get("lock_release_results"),
            fresh_start_classification=start_el.get("decision"),
            start_eligible=True,
            start_transaction_id=start_result.get("action_id"),
            start_result=start_result.get("result"),
            launched_components=start_result.get("launched_components"),
            final_runtime_classification=_sanitize(final_cls),
            live_execution_blocked_before=True,
            live_execution_blocked_after=live_after,
            paper_arm_mutated=bool(start_result.get("paper_arm_mutated")),
            forced_termination_used=bool(stop_result.get("forced_termination_used")),
            runtime_locks_deleted_by_controller=bool(
                stop_result.get("runtime_locks_deleted_by_controller")
            ),
            safe_message=(
                start_result.get("safe_message")
                or (
                    "Restart completed: cooperative Stop verified and fresh protected Start succeeded."
                    if mapped == "RESTART_COMPLETED"
                    else f"Restart finished with result {mapped}."
                )
            ),
            safe_to_retry=mapped != "RESTART_COMPLETED",
            unresolved=mapped
            in ("RESTART_PARTIAL_COMPLETION", "RESTART_TIMED_OUT", "RESTART_AUDIT_FAILED"),
            restart_token_reused_for_start=False,
        )

        final_audit = append_restart_audit(rec, path=audit_path)
        if not final_audit.get("ok"):
            rec = {
                **rec,
                "result": "RESTART_AUDIT_FAILED",
                "error_code": "FINAL_AUDIT_FAILED",
                "safe_message": (
                    "Start may have completed, but final Restart audit failed. "
                    "Coordination evidence preserved."
                ),
                "unresolved": True,
            }
        tx.update(
            {
                "stage": rec["result"],
                "unresolved": bool(rec.get("unresolved")),
                "final": _sanitize(rec),
                "completed_at": _iso(_utc_now()),
            }
        )
        write_restart_transaction(tx, path=tx_path)
        _idempotency_store(nonce, rec, path=idem_path)
        if not rec.get("unresolved"):
            release_restart_guard(action_id=action_id, path=guard_path)
        return rec
    finally:
        # Leave unresolved guards for partial/timeout; release only handled above
        pass
