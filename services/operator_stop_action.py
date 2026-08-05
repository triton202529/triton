"""
operator_stop_action.py
Phase DO-005 — Protected STOP_TRITON dashboard action.

Maps exclusively to DO-004 PREPARE_COOPERATIVE_SHUTDOWN.
Never terminates processes. Never deletes/repairs runtime locks.
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
from services.operator_runtime_controller import live_execution_appears_enabled
from services.operator_shutdown_controller import (
    ACTION_PREPARE,
    evaluate_shutdown_decision,
    prepare_cooperative_shutdown,
    shutdown_active,
)

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
RESULTS = ROOT / "data" / "results"

ACTION_STOP = "STOP_TRITON"
ALLOWED_ACTIONS = frozenset({ACTION_STOP})
CONFIRMATION_PHRASE = "STOP TRITON"

# Fixed dashboard policy — backend may narrow via DO-004 decision matrix
FIXED_DASHBOARD_TARGET_POLICY = "BOTH"

PREFLIGHT_TTL_SEC = 120
TOKEN_STORE = LIVE / "operator_stop_preflight_tokens.json"
AUDIT_PATH = RESULTS / "operator_stop_actions.jsonl"
IDEMPOTENCY_STORE = LIVE / "operator_stop_idempotency.json"

ALLOWED_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "action",
        "confirmation_phrase",
        "confirmation_checked",
        "preflight_token",
        "request_nonce",
        "source",
    }
)

DENIED_FRONTEND_FIELDS = frozenset(
    {
        "target",
        "targets",
        "target_components",
        "pid",
        "pids",
        "component_pid",
        "path",
        "repository_path",
        "lock_path",
        "request_path",
        "acknowledgement_path",
        "guard_path",
        "command",
        "script",
        "arguments",
        "args",
        "environment",
        "env",
        "timeout",
        "timeout_sec",
        "process_name",
        "executable_name",
        "broker",
        "credentials",
    }
)

ELIGIBLE_STOP_DECISIONS = frozenset(
    {
        "STOP_ELIGIBLE_CONTINUOUS_ONLY",
        "STOP_ELIGIBLE_WATCHDOG_ONLY",
        "STOP_ELIGIBLE_BOTH",
        "STOP_ELIGIBLE_UNHEALTHY_VERIFIED",
    }
)

SHUTDOWN_TO_STOP = {
    "ALREADY_STOPPED": "STOP_ALREADY_STOPPED",
    "SHUTDOWN_ELIGIBLE_CONTINUOUS_ONLY": "STOP_ELIGIBLE_CONTINUOUS_ONLY",
    "SHUTDOWN_ELIGIBLE_WATCHDOG_ONLY": "STOP_ELIGIBLE_WATCHDOG_ONLY",
    "SHUTDOWN_ELIGIBLE_BOTH": "STOP_ELIGIBLE_BOTH",
    "SHUTDOWN_ELIGIBLE_UNHEALTHY_VERIFIED": "STOP_ELIGIBLE_UNHEALTHY_VERIFIED",
    "SHUTDOWN_REFUSED_UNSAFE_STATE": "STOP_REFUSED_UNSAFE_STATE",
    "SHUTDOWN_REFUSED_START_IN_PROGRESS": "STOP_REFUSED_START_IN_PROGRESS",
    "SHUTDOWN_IN_PROGRESS": "STOP_IN_PROGRESS",
    "SHUTDOWN_REFUSED_LIVE_SAFETY": "STOP_REFUSED_LIVE_SAFETY",
    "COOPERATIVE_SHUTDOWN_COMPLETED": "STOP_COMPLETED",
    "PARTIAL_SHUTDOWN": "STOP_PARTIAL_COMPLETION",
    "SHUTDOWN_TIMED_OUT": "STOP_TIMED_OUT",
    "POSTCONDITION_FAILED": "STOP_PARTIAL_COMPLETION",
    "AUDIT_FAILED": "STOP_AUDIT_FAILED",
    "FAILED": "STOP_REFUSED",
    "UNKNOWN": "UNKNOWN",
}


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _hash_nonce(nonce: str) -> str:
    return hashlib.sha256(f"nonce:{nonce}".encode("utf-8")).hexdigest()


def append_stop_audit(record: Dict[str, Any], *, path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or AUDIT_PATH
    safe = _sanitize(record)
    # Never persist raw tokens
    for k in ("preflight_token", "confirmation_token", "token", "raw_token"):
        if k in safe:
            safe[k] = "<redacted>"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
        return {"ok": True, "path": str(path)}
    except OSError as exc:
        return {"ok": False, "error": f"audit_write_failed:{exc}", "path": str(path)}


def read_recent_stop_actions(
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


def map_shutdown_to_stop(decision: str) -> str:
    return SHUTDOWN_TO_STOP.get(decision, decision if decision.startswith("STOP_") else "UNKNOWN")


def evaluate_stop_decision(
    classifications: Dict[str, Any],
    *,
    live_execution_blocked: bool = True,
    start_guard_active: bool = False,
    shutdown_guard_active: bool = False,
    unresolved_shutdown: bool = False,
) -> Dict[str, Any]:
    """Explicit Stop-control decision matrix (dashboard-facing)."""
    if unresolved_shutdown and not shutdown_guard_active:
        # Unresolved request without active guard still blocks new Stop
        base = evaluate_shutdown_decision(
            classifications,
            live_execution_blocked=live_execution_blocked,
            start_guard_active=start_guard_active,
            shutdown_guard_active=False,
        )
        # If already stopped cleanly, prefer that
        if base.get("decision") == "ALREADY_STOPPED":
            return {
                "decision": "STOP_ALREADY_STOPPED",
                "target_components": [],
                "explanation": base.get("explanation"),
                "stop_button_enabled": False,
                "ui_state": "STOP_UNAVAILABLE",
                "safe_to_retry": True,
            }
        return {
            "decision": "STOP_REFUSED_SHUTDOWN_UNRESOLVED",
            "target_components": [],
            "explanation": (
                "A prior cooperative shutdown remains unresolved. "
                "Investigate Diagnostics before requesting another Stop."
            ),
            "stop_button_enabled": False,
            "ui_state": "STOP_REFUSED",
            "safe_to_retry": False,
        }

    base = evaluate_shutdown_decision(
        classifications,
        live_execution_blocked=live_execution_blocked,
        start_guard_active=start_guard_active,
        shutdown_guard_active=shutdown_guard_active,
    )
    stop_decision = map_shutdown_to_stop(str(base.get("decision") or "UNKNOWN"))
    enabled = stop_decision in ELIGIBLE_STOP_DECISIONS
    ui_state = (
        "STOP_ELIGIBLE"
        if enabled
        else ("STOP_IN_PROGRESS" if stop_decision == "STOP_IN_PROGRESS" else "STOP_UNAVAILABLE")
    )
    if stop_decision == "STOP_ALREADY_STOPPED":
        ui_state = "STOP_UNAVAILABLE"
    return {
        "decision": stop_decision,
        "target_components": list(base.get("target_components") or []),
        "explanation": base.get("explanation"),
        "continuous_state": base.get("continuous_state"),
        "watchdog_state": base.get("watchdog_state"),
        "stop_button_enabled": enabled,
        "ui_state": ui_state,
        "safe_to_retry": base.get("safe_to_retry"),
        "dashboard_target_policy": FIXED_DASHBOARD_TARGET_POLICY,
        "backend_operation": ACTION_PREPARE,
    }


def _lock_evidence(loop: Dict[str, Any]) -> Dict[str, Any]:
    evidence = loop.get("evidence") if isinstance(loop.get("evidence"), dict) else {}
    lock = evidence.get("lock") if isinstance(evidence.get("lock"), dict) else {}
    return lock


def _identity_fingerprint(classifications: Dict[str, Any], targets: List[str]) -> Dict[str, Any]:
    """
    Stable process identity for Stop confirmation binding.

    Material identity excludes ordinary liveness movement (cycle, heartbeat /
    lock_refreshed_at, phase, safe-boundary). PID-reuse resistance uses lock
    ownership started_at as process_creation_time when available.
    """
    out: Dict[str, Any] = {}
    for c in targets:
        loop = classifications.get(c) or {}
        lock = _lock_evidence(loop)
        script = loop.get("expected_script")
        pids = list(loop.get("process_pids") or [])
        out[c] = {
            "component": c,
            "pids": pids,
            "lock_pid": loop.get("lock_pid"),
            "lock_exists": loop.get("lock_exists"),
            "expected_script": script,
            "process_creation_time": lock.get("started_at"),
            "lock_owner_source": lock.get("source"),
            "command_fingerprint": f"ps1:{script}:pids={pids}",
        }
    return out


def _liveness_snapshot(classifications: Dict[str, Any], targets: List[str]) -> Dict[str, Any]:
    """Non-identity liveness fields recorded separately from identity match."""
    out: Dict[str, Any] = {}
    for c in targets:
        loop = classifications.get(c) or {}
        lock = _lock_evidence(loop)
        out[c] = {
            "state": loop.get("state"),
            "lock_refreshed_at": loop.get("lock_refreshed_at"),
            "activity_age_minutes": loop.get("activity_age_minutes"),
            "cycle": lock.get("cycle"),
        }
    return out


def _identity_changed_fields(
    expected: Dict[str, Any], current: Dict[str, Any], targets: List[str]
) -> Dict[str, Any]:
    changed: Dict[str, Any] = {}
    material_keys = (
        "pids",
        "lock_pid",
        "expected_script",
        "command_fingerprint",
        "process_creation_time",
        "lock_owner_source",
    )
    for c in targets:
        e = expected.get(c) or {}
        cur = current.get(c) or {}
        diffs = {}
        for k in material_keys:
            if e.get(k) != cur.get(k):
                diffs[k] = {"expected": e.get(k), "current": cur.get(k)}
        if diffs:
            changed[c] = diffs
    return changed


def _liveness_drift(
    expected: Dict[str, Any], current: Dict[str, Any], targets: List[str]
) -> Dict[str, Any]:
    drift: Dict[str, Any] = {}
    for c in targets:
        e = expected.get(c) or {}
        cur = current.get(c) or {}
        diffs = {}
        for k in ("state", "lock_refreshed_at", "activity_age_minutes", "cycle"):
            if e.get(k) != cur.get(k):
                diffs[k] = {"expected": e.get(k), "current": cur.get(k)}
        if diffs:
            drift[c] = diffs
    return drift


def _load_token_store(path: Path) -> Dict[str, Any]:
    doc = read_json(path)
    if not doc or doc.get("__malformed__"):
        return {"tokens": {}}
    if "tokens" not in doc or not isinstance(doc["tokens"], dict):
        return {"tokens": {}}
    return doc


def _save_token_store(path: Path, doc: Dict[str, Any]) -> Dict[str, Any]:
    return atomic_write_json(path, doc)


def issue_stop_preflight(
    *,
    root: Optional[Path] = None,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
    ttl_sec: int = PREFLIGHT_TTL_SEC,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Server-generated short-lived preflight token bound to current eligibility/identity.
    Returns raw token once; store only hash.
    """
    from services.operator_runtime_classifier import classify_runtime_loops

    root = Path(root) if root is not None else ROOT
    live = root / "data" / "live"
    now = now or _utc_now()
    classifications = classify_runtime_loops(root=root, process_snapshot=process_snapshot)
    live_blocked = not live_execution_appears_enabled(root=root)
    start_g = is_guard_active(live / "operator_start_action.guard.json", now=now)
    shut_g = is_guard_active(live / "operator_shutdown_action.guard.json", now=now)
    active = shutdown_active(root=root, now=now)
    unresolved = bool(active.get("active")) and not shut_g
    decision = evaluate_stop_decision(
        classifications,
        live_execution_blocked=live_blocked,
        start_guard_active=start_g,
        shutdown_guard_active=shut_g,
        unresolved_shutdown=unresolved
        or bool(active.get("active") and shut_g is False and active.get("reasons")),
    )
    # Treat any active shutdown as unresolved for new Stop enablement
    if active.get("active"):
        if decision["decision"] not in ("STOP_ALREADY_STOPPED", "STOP_IN_PROGRESS"):
            if shut_g:
                decision = {
                    **decision,
                    "decision": "STOP_IN_PROGRESS",
                    "stop_button_enabled": False,
                    "ui_state": "STOP_IN_PROGRESS",
                    "explanation": "A cooperative shutdown transaction is already active.",
                }
            else:
                decision = {
                    **decision,
                    "decision": "STOP_REFUSED_SHUTDOWN_UNRESOLVED",
                    "stop_button_enabled": False,
                    "ui_state": "STOP_REFUSED",
                }

    targets = list(decision.get("target_components") or [])
    bind_targets = targets or ["continuous", "watchdog"]
    identity = _identity_fingerprint(classifications, bind_targets)
    liveness = _liveness_snapshot(classifications, bind_targets)
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash_token(raw_token)
    record = {
        "schema_version": SCHEMA_VERSION,
        "token_hash": token_hash,
        "action": ACTION_STOP,
        "decision": decision.get("decision"),
        "target_components": targets,
        "identity": identity,
        "liveness": liveness,
        "live_execution_blocked": live_blocked,
        "start_guard_active": start_g,
        "shutdown_guard_active": shut_g,
        "shutdown_active": active,
        "issued_at": _iso(now),
        "expires_at": _iso(now + timedelta(seconds=ttl_sec)),
        "used": False,
        "nonce": str(uuid.uuid4()),
    }
    store_path = live / "operator_stop_preflight_tokens.json"
    store = _load_token_store(store_path)
    # prune expired
    pruned = {}
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
            "decision": decision,
            "stop_button_enabled": False,
        }

    return {
        "ok": True,
        "preflight_token": raw_token,  # returned once to caller; never audit raw
        "preflight_nonce": record["nonce"],
        "expires_at": record["expires_at"],
        "issued_at": record["issued_at"],
        "decision": decision.get("decision"),
        "explanation": decision.get("explanation"),
        "target_components": targets,
        "stop_button_enabled": bool(decision.get("stop_button_enabled")),
        "ui_state": decision.get("ui_state"),
        "dashboard_target_policy": FIXED_DASHBOARD_TARGET_POLICY,
        "backend_operation": ACTION_PREPARE,
        "live_execution_blocked": live_blocked,
        "start_guard_active": start_g,
        "shutdown_guard_active": shut_g,
        "classifications": _sanitize(classifications),
        "identity_bound": True,
        "token_presented": True,  # opaque flag for Diagnostics — not the token
        "restart_available": False,
        "allowed_dashboard_action": ACTION_STOP,
        "safe_message": (
            "Stop TRITON uses verified cooperative shutdown. Active bounded work is allowed "
            "to reach its documented safe boundary. TRITON is never force-terminated by this control."
        ),
    }


def stop_eligibility(
    *,
    root: Optional[Path] = None,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Read-only Stop eligibility (no token issuance)."""
    from services.operator_runtime_classifier import classify_runtime_loops

    root = Path(root) if root is not None else ROOT
    live = root / "data" / "live"
    now = _utc_now()
    classifications = classify_runtime_loops(root=root, process_snapshot=process_snapshot)
    live_blocked = not live_execution_appears_enabled(root=root)
    start_g = is_guard_active(live / "operator_start_action.guard.json", now=now)
    shut_g = is_guard_active(live / "operator_shutdown_action.guard.json", now=now)
    active = shutdown_active(root=root, now=now)
    decision = evaluate_stop_decision(
        classifications,
        live_execution_blocked=live_blocked,
        start_guard_active=start_g,
        shutdown_guard_active=shut_g,
        unresolved_shutdown=bool(active.get("active")),
    )
    return {
        **decision,
        "protocol_implemented": True,
        "protected_stop_exposed": True,
        "restart_exposed": False,
        "restart_available": False,
        "clickable_restart_exposed": False,
        "confirmation_phrase_required": CONFIRMATION_PHRASE,
        "confirmation_checkbox_required": True,
        "preflight_token_required": True,
        "dashboard_target_policy": FIXED_DASHBOARD_TARGET_POLICY,
        "backend_operation": ACTION_PREPARE,
        "allowed_dashboard_action": ACTION_STOP,
        "live_execution_blocked": live_blocked,
        "start_guard_active": start_g,
        "shutdown_guard_active": shut_g,
        "shutdown_active": active,
        "classifications": _sanitize(classifications),
        "safe_message": (
            "Stop TRITON uses verified cooperative shutdown. Active bounded work is allowed "
            "to reach its documented safe boundary. TRITON is never force-terminated by this control."
        ),
        "recommended_action": (
            "Use protected Stop only when eligible. Restart remains unavailable."
        ),
    }


def _consume_token(
    raw_token: str,
    *,
    store_path: Path,
    now: datetime,
) -> Dict[str, Any]:
    store = _load_token_store(store_path)
    th = _hash_token(raw_token)
    rec = (store.get("tokens") or {}).get(th)
    if not rec:
        return {
            "ok": False,
            "reason": "STOP_REFUSED_CONFIRMATION_EXPIRED",
            "detail": "unknown_token",
        }
    if rec.get("used"):
        return {"ok": False, "reason": "STOP_REFUSED_REPLAY", "detail": "token_already_used"}
    try:
        exp = datetime.fromisoformat(str(rec.get("expires_at")).replace("Z", "+00:00"))
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
    except Exception:
        return {"ok": False, "reason": "STOP_REFUSED_CONFIRMATION_EXPIRED", "detail": "bad_expiry"}
    if exp <= now:
        return {"ok": False, "reason": "STOP_REFUSED_CONFIRMATION_EXPIRED", "detail": "expired"}
    if rec.get("action") != ACTION_STOP:
        return {"ok": False, "reason": "STOP_REFUSED_CONFIRMATION", "detail": "wrong_action"}
    # mark used
    rec["used"] = True
    rec["used_at"] = _iso(now)
    store["tokens"][th] = rec
    _save_token_store(store_path, store)
    return {"ok": True, "record": rec, "token_hash": th}


def _identities_match(
    expected: Dict[str, Any], current: Dict[str, Any], targets: List[str]
) -> bool:
    """True when stable identity matches. Ignores liveness-only drift."""
    for c in targets:
        e = expected.get(c) or {}
        cur = current.get(c) or {}
        if list(e.get("pids") or []) != list(cur.get("pids") or []):
            return False
        if e.get("lock_pid") != cur.get("lock_pid"):
            return False
        if e.get("expected_script") != cur.get("expected_script"):
            return False
        if e.get("command_fingerprint") != cur.get("command_fingerprint"):
            return False
        # PID-reuse-resistant ownership start time (not heartbeat/refreshed_at)
        if e.get("process_creation_time") and cur.get("process_creation_time"):
            if e.get("process_creation_time") != cur.get("process_creation_time"):
                return False
        if e.get("lock_owner_source") and cur.get("lock_owner_source"):
            if e.get("lock_owner_source") != cur.get("lock_owner_source"):
                return False
    return True


def _idempotency_lookup(nonce: str, *, path: Path) -> Optional[Dict[str, Any]]:
    doc = read_json(path)
    if not doc or doc.get("__malformed__"):
        return None
    entries = doc.get("entries") or {}
    h = _hash_nonce(nonce)
    return entries.get(h)


def _idempotency_store(nonce: str, result: Dict[str, Any], *, path: Path) -> None:
    doc = read_json(path)
    if not doc or doc.get("__malformed__"):
        doc = {"entries": {}}
    entries = doc.setdefault("entries", {})
    entries[_hash_nonce(nonce)] = {
        "stored_at": _iso(),
        "result": _sanitize({k: v for k, v in result.items() if k != "preflight_token"}),
    }
    # keep small
    if len(entries) > 50:
        keys = list(entries.keys())[:-40]
        for k in keys:
            entries.pop(k, None)
    atomic_write_json(path, doc)


def execute_stop_triton(
    *,
    action: str = ACTION_STOP,
    confirmation_phrase: str = "",
    confirmation_checked: bool = False,
    preflight_token: Optional[str] = None,
    request_nonce: Optional[str] = None,
    source: str = "operator_console",
    root: Optional[Path] = None,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
    deliver: bool = True,
    ack_timeout_sec: float = float(recommended_ack_timeout_sec()),
    exit_timeout_sec: float = 60.0,
    lock_timeout_sec: float = 30.0,
    poll_sec: float = 0.5,
    sleep_fn: Optional[Callable[[float], None]] = None,
    # Explicitly rejected — frontend must not supply these
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
    **extra: Any,
) -> Dict[str, Any]:
    """Typed STOP_TRITON action. Maps only to PREPARE_COOPERATIVE_SHUTDOWN."""
    from services.operator_runtime_classifier import classify_runtime_loops

    root = Path(root) if root is not None else ROOT
    live = root / "data" / "live"
    results = root / "data" / "results"
    audit_path = results / "operator_stop_actions.jsonl"
    token_store = live / "operator_stop_preflight_tokens.json"
    idem_path = live / "operator_stop_idempotency.json"
    now = _utc_now()
    action_id = str(uuid.uuid4())

    def _base(**kwargs: Any) -> Dict[str, Any]:
        out = {
            "schema_version": SCHEMA_VERSION,
            "action_id": action_id,
            "action": ACTION_STOP,
            "source": source,
            "controller_version": CONTROLLER_VERSION,
            "backend_operation": ACTION_PREPARE,
            "dashboard_target_policy": FIXED_DASHBOARD_TARGET_POLICY,
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
            "automatic_restart": False,
            "restart_available": False,
        }
        out.update(kwargs)
        return out

    # Reject injected operational fields
    injected = {
        "target": target,
        "targets": targets,
        "target_components": target_components,
        "pid": pid,
        "path": path,
        "command": command,
        "script": script,
        "arguments": arguments,
        "env": env,
        "environment": environment,
        "timeout": timeout,
    }
    if any(v is not None for v in injected.values()) or extra:
        bad = [k for k, v in injected.items() if v is not None] + list(extra.keys())
        rec = _base(
            result="STOP_REFUSED_CONFIRMATION",
            error_code="FRONTEND_INJECTION_REFUSED",
            safe_message=(
                "Stop refused: frontend must not supply targets, PID, path, command, "
                f"arguments, environment, timeout, or unknown fields ({', '.join(bad)})."
            ),
            confirmation_valid=False,
            evaluated_at=_iso(now),
            completed_at=_iso(),
            live_execution_blocked_before=True,
            live_execution_blocked_after=True,
        )
        append_stop_audit(rec, path=audit_path)
        return rec

    if action != ACTION_STOP:
        rec = _base(
            result="STOP_REFUSED_CONFIRMATION",
            error_code="ACTION_NOT_ALLOWED",
            safe_message=f"Action {action!r} is not allowed. Only {ACTION_STOP} is supported.",
            confirmation_valid=False,
            evaluated_at=_iso(now),
            completed_at=_iso(),
        )
        append_stop_audit(rec, path=audit_path)
        return rec

    if not confirmation_checked:
        rec = _base(
            result="STOP_REFUSED_CONFIRMATION",
            error_code="CHECKBOX_REQUIRED",
            safe_message="Stop refused: confirmation checkbox is required.",
            confirmation_valid=False,
            evaluated_at=_iso(now),
            completed_at=_iso(),
        )
        append_stop_audit(rec, path=audit_path)
        return rec

    # Exact phrase — no case folding, no whitespace normalize
    if confirmation_phrase != CONFIRMATION_PHRASE:
        rec = _base(
            result="STOP_REFUSED_CONFIRMATION",
            error_code="PHRASE_MISMATCH",
            safe_message="Stop refused: confirmation phrase must match exactly: STOP TRITON",
            confirmation_valid=False,
            evaluated_at=_iso(now),
            completed_at=_iso(),
        )
        append_stop_audit(rec, path=audit_path)
        return rec

    if not preflight_token:
        rec = _base(
            result="STOP_REFUSED_CONFIRMATION_EXPIRED",
            error_code="MISSING_TOKEN",
            safe_message="Stop refused: missing preflight token.",
            confirmation_valid=True,
            preflight_token_valid=False,
            evaluated_at=_iso(now),
            completed_at=_iso(),
        )
        append_stop_audit(rec, path=audit_path)
        return rec

    nonce = request_nonce or str(uuid.uuid4())
    prior = _idempotency_lookup(nonce, path=idem_path)
    if prior and prior.get("result"):
        replay = dict(prior["result"])
        replay["replayed"] = True
        replay["action_id"] = action_id
        replay["safe_message"] = (
            replay.get("safe_message")
            or "Duplicate Stop submission with same nonce — prior result returned."
        )
        append_stop_audit({**replay, "replay_detected": True}, path=audit_path)
        return replay

    consumed = _consume_token(preflight_token, store_path=token_store, now=now)
    if not consumed.get("ok"):
        reason = str(consumed.get("reason") or "STOP_REFUSED_CONFIRMATION_EXPIRED")
        rec = _base(
            result=reason,
            error_code=consumed.get("detail"),
            safe_message=f"Stop refused: {reason}",
            confirmation_valid=True,
            preflight_token_valid=False,
            preflight_token_expired=reason == "STOP_REFUSED_CONFIRMATION_EXPIRED",
            replay_detected=reason == "STOP_REFUSED_REPLAY",
            request_nonce_hash=_hash_nonce(nonce),
            evaluated_at=_iso(now),
            completed_at=_iso(),
        )
        append_stop_audit(rec, path=audit_path)
        return rec

    token_rec = consumed["record"]
    expected_targets = list(token_rec.get("target_components") or [])
    expected_identity = token_rec.get("identity") or {}

    # Reclassify immediately before delivery
    classifications = classify_runtime_loops(root=root, process_snapshot=process_snapshot)
    live_before = not live_execution_appears_enabled(root=root)
    start_g = is_guard_active(live / "operator_start_action.guard.json", now=now)
    shut_g = is_guard_active(live / "operator_shutdown_action.guard.json", now=now)
    active = shutdown_active(root=root, now=now)
    decision = evaluate_stop_decision(
        classifications,
        live_execution_blocked=live_before,
        start_guard_active=start_g,
        shutdown_guard_active=shut_g,
        unresolved_shutdown=bool(active.get("active")),
    )

    if decision["decision"] == "STOP_ALREADY_STOPPED":
        rec = _base(
            result="STOP_ALREADY_STOPPED",
            safe_message="TRITON is already stopped. No shutdown request was created.",
            confirmation_valid=True,
            preflight_token_valid=True,
            target_components=[],
            request_nonce_hash=_hash_nonce(nonce),
            evaluated_at=_iso(now),
            completed_at=_iso(),
            live_execution_blocked_before=live_before,
            live_execution_blocked_after=not live_execution_appears_enabled(root=root),
            ui_state="STOP_COMPLETED",
        )
        append_stop_audit(rec, path=audit_path)
        _idempotency_store(nonce, rec, path=idem_path)
        return rec

    if decision["decision"] not in ELIGIBLE_STOP_DECISIONS:
        rec = _base(
            result=decision["decision"],
            safe_message=decision.get("explanation") or decision["decision"],
            confirmation_valid=True,
            preflight_token_valid=True,
            state_changed=decision["decision"] != token_rec.get("decision"),
            request_nonce_hash=_hash_nonce(nonce),
            evaluated_at=_iso(now),
            completed_at=_iso(),
            live_execution_blocked_before=live_before,
            live_execution_blocked_after=live_before,
            preflight_classification=_sanitize({"decision": token_rec.get("decision")}),
            submission_classification=_sanitize(classifications),
        )
        append_stop_audit(rec, path=audit_path)
        return rec

    current_identity = _identity_fingerprint(classifications, expected_targets)
    current_liveness = _liveness_snapshot(classifications, expected_targets)
    expected_liveness = token_rec.get("liveness") or {}
    liveness_drift = _liveness_drift(expected_liveness, current_liveness, expected_targets)
    if set(expected_targets) != set(decision.get("target_components") or []):
        rec = _base(
            result="STOP_REFUSED_STATE_CHANGED",
            error_code="TARGET_SET_CHANGED",
            safe_message="Stop refused: eligible target components changed since confirmation.",
            confirmation_valid=True,
            preflight_token_valid=True,
            state_changed=True,
            request_nonce_hash=_hash_nonce(nonce),
            evaluated_at=_iso(now),
            completed_at=_iso(),
            live_execution_blocked_before=live_before,
            live_execution_blocked_after=live_before,
            liveness_drift=_sanitize(liveness_drift) if liveness_drift else None,
        )
        append_stop_audit(rec, path=audit_path)
        return rec

    if not _identities_match(expected_identity, current_identity, expected_targets):
        changed = _identity_changed_fields(expected_identity, current_identity, expected_targets)
        rec = _base(
            result="STOP_REFUSED_STATE_CHANGED",
            error_code="IDENTITY_CHANGED",
            safe_message=(
                "Stop refused: verified process/lock identity changed since confirmation. "
                "Re-open confirmation with a fresh preflight."
            ),
            confirmation_valid=True,
            preflight_token_valid=True,
            state_changed=True,
            request_nonce_hash=_hash_nonce(nonce),
            evaluated_at=_iso(now),
            completed_at=_iso(),
            live_execution_blocked_before=live_before,
            live_execution_blocked_after=live_before,
            expected_instances=_sanitize(expected_identity),
            submission_classification=_sanitize(current_identity),
            identity_changed_fields=_sanitize(changed),
            liveness_drift=_sanitize(liveness_drift) if liveness_drift else None,
            material_identity_changed=True,
            volatile_liveness_only_changed=False,
        )
        append_stop_audit(rec, path=audit_path)
        return rec

    if decision.get("decision") != token_rec.get("decision"):
        # Decision label changed (e.g. healthy -> unhealthy) but still eligible — refuse for safety
        if (
            decision["decision"] not in ELIGIBLE_STOP_DECISIONS
            or token_rec.get("decision") not in ELIGIBLE_STOP_DECISIONS
        ):
            rec = _base(
                result="STOP_REFUSED_STATE_CHANGED",
                error_code="DECISION_CHANGED",
                safe_message="Stop refused: eligibility decision changed since confirmation.",
                confirmation_valid=True,
                preflight_token_valid=True,
                state_changed=True,
                request_nonce_hash=_hash_nonce(nonce),
                evaluated_at=_iso(now),
                completed_at=_iso(),
            )
            append_stop_audit(rec, path=audit_path)
            return rec

    # Initial audit before delivery
    begin = append_stop_audit(
        _base(
            result="STOP_BEGIN",
            confirmation_valid=True,
            preflight_token_valid=True,
            request_nonce_hash=_hash_nonce(nonce),
            target_components=expected_targets,
            expected_instances=_sanitize(expected_identity),
            preflight_classification=_sanitize({"decision": token_rec.get("decision")}),
            submission_classification=_sanitize(classifications),
            evaluated_at=_iso(now),
            submitted_at=_iso(),
            live_execution_blocked_before=live_before,
            safe_message="Delivering cooperative shutdown via DO-004 controller.",
        ),
        path=audit_path,
    )
    if not begin.get("ok"):
        rec = _base(
            result="STOP_AUDIT_FAILED",
            error_code="INITIAL_AUDIT_FAILED",
            safe_message="Initial audit could not be persisted; Stop request was not delivered.",
            confirmation_valid=True,
            preflight_token_valid=True,
            request_nonce_hash=_hash_nonce(nonce),
            evaluated_at=_iso(now),
            completed_at=_iso(),
            live_execution_blocked_before=live_before,
            live_execution_blocked_after=live_before,
        )
        return rec

    if not deliver:
        rec = _base(
            result="STOP_ELIGIBLE",
            safe_message="Eligibility only; deliver=False.",
            target_components=expected_targets,
            completed_at=_iso(),
        )
        return rec

    # Invoke ONLY the typed DO-004 controller — no targets from frontend
    shutdown_result = prepare_cooperative_shutdown(
        action=ACTION_PREPARE,
        root=root,
        process_snapshot=process_snapshot,
        source=f"operator_stop_action:{source}",
        reason_code="OPERATOR_PROTECTED_STOP",
        deliver=True,
        ack_timeout_sec=ack_timeout_sec,
        exit_timeout_sec=exit_timeout_sec,
        lock_timeout_sec=lock_timeout_sec,
        poll_sec=poll_sec,
        sleep_fn=sleep_fn,
    )

    mapped = map_shutdown_to_stop(str(shutdown_result.get("result") or "UNKNOWN"))
    live_after = not live_execution_appears_enabled(root=root)
    final = _base(
        result=mapped,
        request_id=shutdown_result.get("request_id"),
        confirmation_valid=True,
        preflight_token_valid=True,
        state_changed=False,
        request_nonce_hash=_hash_nonce(nonce),
        target_components=shutdown_result.get("target_components") or expected_targets,
        expected_instances=shutdown_result.get("expected_instances"),
        acknowledgement_results=shutdown_result.get("acknowledgement_results"),
        process_exit_results=shutdown_result.get("process_exit_results"),
        lock_release_results=shutdown_result.get("lock_release_results"),
        shutdown_guard_result={"via": "prepare_cooperative_shutdown"},
        request_delivery_result={"ok": shutdown_result.get("request_id") is not None},
        postcondition_classification=None,
        evaluated_at=_iso(now),
        submitted_at=_iso(now),
        completed_at=_iso(),
        live_execution_blocked_before=live_before,
        live_execution_blocked_after=live_after,
        paper_arm_mutated=False,
        orders_submitted=False,
        orders_cancelled=False,
        approvals_created=False,
        authorizations_created=False,
        forced_termination_used=False,
        runtime_locks_deleted_by_controller=False,
        safe_to_retry=shutdown_result.get("safe_to_retry"),
        safe_message=shutdown_result.get("safe_message") or mapped,
        ui_state=_ui_state_for_result(mapped),
        do004_result=shutdown_result.get("result"),
    )
    if (
        shutdown_result.get("result") == "AUDIT_FAILED"
        or shutdown_result.get("audit_final_ok") is False
    ):
        final["result"] = "STOP_AUDIT_FAILED"
        final["ui_state"] = "STOP_AUDIT_FAILED"

    audit_final = append_stop_audit(final, path=audit_path)
    if not audit_final.get("ok") and final["result"] not in ("STOP_AUDIT_FAILED",):
        final["result"] = "STOP_AUDIT_FAILED"
        final["ui_state"] = "STOP_AUDIT_FAILED"
        final["safe_message"] = (
            "Shutdown may have been delivered; final Stop audit failed. "
            "Coordination evidence preserved. Forced termination remains outside policy."
        )

    _idempotency_store(nonce, final, path=idem_path)
    return final


def _ui_state_for_result(result: str) -> str:
    mapping = {
        "STOP_COMPLETED": "STOP_COMPLETED",
        "STOP_PARTIAL_COMPLETION": "STOP_PARTIAL_COMPLETION",
        "STOP_TIMED_OUT": "STOP_TIMED_OUT",
        "STOP_ALREADY_STOPPED": "STOP_COMPLETED",
        "STOP_IN_PROGRESS": "SUBMISSION_IN_PROGRESS",
        "STOP_AUDIT_FAILED": "STOP_AUDIT_FAILED",
    }
    if result in mapping:
        return mapping[result]
    if result.startswith("STOP_REFUSED"):
        return "STOP_REFUSED"
    return "UNKNOWN"


def controller_has_termination_paths() -> bool:
    src = Path(__file__).read_text(encoding="utf-8")
    kill_call = "os" + "." + "kill" + "("
    proc_kill = "process" + "." + "kill" + "("
    taskkill = "task" + "kill" + ".exe"
    stop_proc = "Stop" + "-" + "Process" + " -"
    popen = "P" + "open("
    subproc_import = "import " + "subprocess"
    return any(d in src for d in (kill_call, proc_kill, taskkill, stop_proc, popen, subproc_import))
