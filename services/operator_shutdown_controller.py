"""
operator_shutdown_controller.py
Phase DO-004 — Cooperative shutdown prepare/orchestrate controller.

Allowed action: PREPARE_COOPERATIVE_SHUTDOWN
Never terminates processes. Never deletes runtime loop locks.
Never creates/extends paper arm. Never enables live. No OS kill paths.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from services.operator_cooperative_shutdown import (
    ACTION,
    CONTROLLER_VERSION,
    REQUEST_PATH,
    SCHEMA_VERSION,
    SHUTDOWN_GUARD,
    acquire_shutdown_guard,
    append_shutdown_audit,
    atomic_write_json,
    build_expected_instance,
    create_shutdown_request,
    is_guard_active,
    read_json,
    read_recent_shutdown_actions,
    recommended_ack_timeout_sec,
    release_shutdown_guard,
    _iso,
    _sanitize,
    _utc_now,
)

# Re-export for Diagnostics / Operations
__all__ = [
    "ACTION_PREPARE",
    "ALLOWED_ACTIONS",
    "DENIED_ACTIONS",
    "evaluate_shutdown_decision",
    "prepare_cooperative_shutdown",
    "read_recent_shutdown_actions",
    "shutdown_active",
    "shutdown_eligibility",
]
from services.operator_runtime_controller import (
    live_execution_appears_enabled,
)

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"

ACTION_PREPARE = "PREPARE_COOPERATIVE_SHUTDOWN"
ALLOWED_ACTIONS = frozenset({ACTION_PREPARE})

DENIED_ACTIONS = frozenset(
    {
        "STOP_TRITON",
        "RESTART_TRITON",
        "FORCE_STOP",
        "KILL_PROCESS",
        "STOP_PROCESS",
        "TASKKILL",
        "SIGKILL",
        "SIGTERM",
        "REPAIR_LOCK",
        "REMOVE_LOCK",
        "DELETE_LOCK",
        "PAPER_ARM",
        "DISARM_PAPER",
        "APPROVE",
        "AUTHORIZE",
        "SUBMIT_ORDER",
        "CANCEL_ORDER",
        "ENABLE_LIVE",
        "LIVE_ENABLE",
    }
)

ELIGIBLE_STATES = frozenset({"HEALTHY_RUNNING", "STALE_OR_OVERDUE"})
UNSAFE_STATES = frozenset(
    {
        "STALE_LOCK_NO_PROCESS",
        "PROCESS_WITHOUT_VALID_LOCK",
        "PID_MISMATCH",
        "COMMAND_MISMATCH",
        "DUPLICATE_PROCESSES",
        "AMBIGUOUS",
        "UNKNOWN",
    }
)

# Aligned to interruptible cooperative observation interval (not full 45-minute sleep).
ACK_TIMEOUT_SEC = recommended_ack_timeout_sec()
EXIT_TIMEOUT_SEC = 60
LOCK_RELEASE_TIMEOUT_SEC = 30
POLL_SEC = 0.5

# Forbidden invocation patterns (tests assert absence of call forms)
_FORBIDDEN_TERMINATION = (
    "taskkill",
    "Stop-Process",
    "TerminateProcess",
    "SIGKILL",
)


def evaluate_shutdown_decision(
    classifications: Dict[str, Any],
    *,
    live_execution_blocked: bool = True,
    start_guard_active: bool = False,
    shutdown_guard_active: bool = False,
) -> Dict[str, Any]:
    """Explicit Stop/shutdown eligibility decision matrix (backend only)."""
    if not live_execution_blocked:
        return {
            "decision": "SHUTDOWN_REFUSED_LIVE_SAFETY",
            "target_components": [],
            "explanation": (
                "Live execution does not appear blocked. Shutdown refused. No override."
            ),
            "safe_to_retry": False,
        }
    if start_guard_active:
        return {
            "decision": "SHUTDOWN_REFUSED_START_IN_PROGRESS",
            "target_components": [],
            "explanation": "A Start transaction is active. Shutdown preparation refused.",
            "safe_to_retry": True,
        }
    if shutdown_guard_active:
        return {
            "decision": "SHUTDOWN_IN_PROGRESS",
            "target_components": [],
            "explanation": "A cooperative shutdown transaction is already active.",
            "safe_to_retry": True,
        }

    cont = (classifications.get("continuous") or {}).get("state") or "UNKNOWN"
    wd = (classifications.get("watchdog") or {}).get("state") or "UNKNOWN"

    if cont in UNSAFE_STATES or wd in UNSAFE_STATES:
        return {
            "decision": "SHUTDOWN_REFUSED_UNSAFE_STATE",
            "target_components": [],
            "explanation": (
                f"Unsafe or ambiguous runtime state "
                f"(continuous={cont}, watchdog={wd}). Write no request."
            ),
            "safe_to_retry": False,
            "continuous_state": cont,
            "watchdog_state": wd,
        }

    if cont == "NOT_RUNNING" and wd == "NOT_RUNNING":
        return {
            "decision": "ALREADY_STOPPED",
            "target_components": [],
            "explanation": "Both loops are absent with no conflicting locks. Already stopped.",
            "safe_to_retry": True,
            "continuous_state": cont,
            "watchdog_state": wd,
        }

    targets: List[str] = []
    if cont in ELIGIBLE_STATES:
        targets.append("continuous")
    if wd in ELIGIBLE_STATES:
        targets.append("watchdog")

    # Eligible only when the other side is NOT_RUNNING or also eligible
    if cont not in ELIGIBLE_STATES | {"NOT_RUNNING"} or wd not in ELIGIBLE_STATES | {"NOT_RUNNING"}:
        return {
            "decision": "SHUTDOWN_REFUSED_UNSAFE_STATE",
            "target_components": [],
            "explanation": f"No cooperatively eligible component (continuous={cont}, watchdog={wd}).",
            "safe_to_retry": False,
            "continuous_state": cont,
            "watchdog_state": wd,
        }

    if not targets:
        return {
            "decision": "SHUTDOWN_REFUSED_UNSAFE_STATE",
            "target_components": [],
            "explanation": f"No cooperatively eligible component (continuous={cont}, watchdog={wd}).",
            "safe_to_retry": False,
            "continuous_state": cont,
            "watchdog_state": wd,
        }

    unhealthy = any(
        (classifications.get(c) or {}).get("state") == "STALE_OR_OVERDUE" for c in targets
    )
    if unhealthy:
        decision = "SHUTDOWN_ELIGIBLE_UNHEALTHY_VERIFIED"
    elif targets == ["continuous"]:
        decision = "SHUTDOWN_ELIGIBLE_CONTINUOUS_ONLY"
    elif targets == ["watchdog"]:
        decision = "SHUTDOWN_ELIGIBLE_WATCHDOG_ONLY"
    else:
        decision = "SHUTDOWN_ELIGIBLE_BOTH"

    return {
        "decision": decision,
        "target_components": targets,
        "explanation": (
            f"Cooperative shutdown eligible for: {', '.join(targets) or 'none'} "
            f"(continuous={cont}, watchdog={wd})."
        ),
        "safe_to_retry": True,
        "continuous_state": cont,
        "watchdog_state": wd,
    }


def shutdown_active(
    *, root: Optional[Path] = None, now: Optional[datetime] = None
) -> Dict[str, Any]:
    """Whether Start must refuse due to unresolved cooperative shutdown."""
    root = Path(root) if root is not None else ROOT
    live = root / "data" / "live"
    now = now or _utc_now()
    guard_path = live / "operator_shutdown_action.guard.json"
    request_path = live / "operator_shutdown_request.json"
    reasons: List[str] = []

    if is_guard_active(guard_path, now=now):
        reasons.append("shutdown_guard_active")

    req = read_json(request_path)
    if req and not req.get("__malformed__"):
        from services.operator_cooperative_shutdown import _parse_dt

        exp = _parse_dt(req.get("expires_at"))
        state = str(req.get("state") or "CREATED")
        terminal = {
            "COMPLETED",
            "PARTIAL_COMPLETION",
            "EXPIRED",
            "SUPERSEDED",
            "TIMED_OUT",
            "FAILED",
            "REFUSED_IDENTITY_MISMATCH",
            "REFUSED_UNSAFE_STATE",
        }
        if exp and exp > now and state not in terminal:
            reasons.append("active_shutdown_request")
            if state in ("ACKNOWLEDGED", "EXIT_PENDING", "DELIVERED"):
                reasons.append("exit_pending_or_delivered")

    # Unresolved: process gone but lock remains after request targeting it
    if req and not req.get("__malformed__"):
        # Soft signal only — Start still refuses if request non-terminal
        pass

    return {
        "active": bool(reasons),
        "reasons": reasons,
        "request_id": req.get("request_id") if req else None,
        "request_state": req.get("state") if req else None,
    }


def shutdown_eligibility(
    *,
    root: Optional[Path] = None,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Read-only cooperative shutdown eligibility for Operations/Diagnostics."""
    from services.operator_runtime_classifier import classify_runtime_loops

    root = Path(root) if root is not None else ROOT
    live = root / "data" / "live"
    live_enabled = live_execution_appears_enabled(root=root)
    classifications = classify_runtime_loops(root=root, process_snapshot=process_snapshot)
    start_g = is_guard_active(live / "operator_start_action.guard.json")
    shut_g = is_guard_active(live / "operator_shutdown_action.guard.json")
    decision = evaluate_shutdown_decision(
        classifications,
        live_execution_blocked=not live_enabled,
        start_guard_active=start_g,
        shutdown_guard_active=shut_g,
    )
    req = read_json(live / "operator_shutdown_request.json")
    ack_c = read_json(live / "operator_shutdown_ack_continuous.json")
    ack_w = read_json(live / "operator_shutdown_ack_watchdog.json")
    active = shutdown_active(root=root)
    return {
        "protocol_implemented": True,
        "clickable_stop_exposed": False,
        "clickable_restart_exposed": False,
        "allowed_backend_action": ACTION_PREPARE,
        "denied_actions": sorted(DENIED_ACTIONS),
        "decision": decision.get("decision"),
        "explanation": decision.get("explanation"),
        "target_components": decision.get("target_components") or [],
        "classifications": classifications,
        "live_execution_blocked": not live_enabled,
        "start_guard_active": start_g,
        "shutdown_guard_active": shut_g,
        "shutdown_active": active,
        "active_request": _sanitize(req) if req and not req.get("__malformed__") else {},
        "ack_continuous": _sanitize(ack_c) if ack_c else {},
        "ack_watchdog": _sanitize(ack_w) if ack_w else {},
        "recommended_action": (
            "Do not use Stop yet — cooperative protocol is installed for backend validation. "
            "Operator Stop remains unavailable until protected Stop control validation is complete."
        ),
        "safe_message": (
            "Cooperative shutdown protocol installed. Operator Stop remains unavailable "
            "until protected Stop control validation is complete."
        ),
    }


def _component_exited(
    component: str,
    expected_pid: Optional[int],
    classifications: Dict[str, Any],
) -> bool:
    loop = classifications.get(component) or {}
    state = loop.get("state")
    pids = list(loop.get("process_pids") or [])
    if state == "NOT_RUNNING" and not pids:
        return True
    if expected_pid is not None and expected_pid not in pids and not pids:
        return True
    if (
        expected_pid is not None
        and expected_pid not in pids
        and state
        in (
            "NOT_RUNNING",
            "STALE_LOCK_NO_PROCESS",
        )
    ):
        return True
    return False


def _lock_released(component: str, classifications: Dict[str, Any]) -> bool:
    loop = classifications.get(component) or {}
    return not bool(loop.get("lock_exists"))


def _poll_completion(
    *,
    targets: Sequence[str],
    expected_instances: Dict[str, Any],
    classify_fn: Callable[[], Dict[str, Any]],
    root: Path,
    ack_timeout: float,
    exit_timeout: float,
    lock_timeout: float,
    poll_sec: float,
    sleep_fn: Callable[[float], None],
) -> Dict[str, Any]:
    live = root / "data" / "live"
    ack_paths = {
        "continuous": live / "operator_shutdown_ack_continuous.json",
        "watchdog": live / "operator_shutdown_ack_watchdog.json",
    }
    deadline_ack = time.time() + ack_timeout
    deadline_exit = time.time() + exit_timeout
    deadline_lock = time.time() + lock_timeout

    ack_results: Dict[str, Any] = {}
    exit_results: Dict[str, Any] = {}
    lock_results: Dict[str, Any] = {}
    forced_termination_used = False

    # Phase 1: acknowledgements
    while time.time() < deadline_ack:
        for c in targets:
            if c in ack_results:
                continue
            ack = read_json(ack_paths[c])
            if not ack or ack.get("__malformed__"):
                continue
            rid_ok = True  # request_id checked by caller via expected
            state = str(ack.get("state") or "")
            if state in (
                "ACCEPTED",
                "EXITING",
                "COMPLETING_CURRENT_SAFE_UNIT",
                "REFUSED_WRONG_INSTANCE",
                "REFUSED_EXPIRED",
                "REFUSED_MALFORMED",
                "REFUSED_UNSUPPORTED_ACTION",
                "REFUSED_UNSAFE_BOUNDARY",
                "FAILED",
            ):
                ack_results[c] = {
                    "state": state,
                    "actual_pid": ack.get("actual_pid"),
                    "request_id": ack.get("request_id"),
                    "safe_boundary": ack.get("safe_boundary"),
                    "accepted": state in ("ACCEPTED", "EXITING", "COMPLETING_CURRENT_SAFE_UNIT"),
                }
        if all(c in ack_results for c in targets):
            break
        sleep_fn(poll_sec)

    for c in targets:
        if c not in ack_results:
            ack_results[c] = {"state": "TIMEOUT", "accepted": False, "timed_out": True}

    # If any refused identity, stop early
    if any(
        str((ack_results.get(c) or {}).get("state") or "").startswith("REFUSED") for c in targets
    ):
        return {
            "result": "POSTCONDITION_FAILED",
            "ack_results": ack_results,
            "exit_results": exit_results,
            "lock_results": lock_results,
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
        }

    if any((ack_results.get(c) or {}).get("timed_out") for c in targets):
        return {
            "result": "SHUTDOWN_TIMED_OUT",
            "ack_results": ack_results,
            "exit_results": exit_results,
            "lock_results": lock_results,
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "safe_message": ("Acknowledgement timed out. Forced termination is outside policy."),
        }

    # Phase 2: process exit
    while time.time() < deadline_exit:
        snap = classify_fn()
        for c in targets:
            exp_pid = (expected_instances.get(c) or {}).get("expected_pid")
            exited = _component_exited(c, exp_pid, snap)
            exit_results[c] = {
                "exited": exited,
                "state": (snap.get(c) or {}).get("state"),
                "pids": list((snap.get(c) or {}).get("process_pids") or []),
            }
        if all((exit_results.get(c) or {}).get("exited") for c in targets):
            break
        sleep_fn(poll_sec)

    for c in targets:
        if c not in exit_results:
            snap = classify_fn()
            exp_pid = (expected_instances.get(c) or {}).get("expected_pid")
            exit_results[c] = {
                "exited": _component_exited(c, exp_pid, snap),
                "state": (snap.get(c) or {}).get("state"),
                "pids": list((snap.get(c) or {}).get("process_pids") or []),
                "timed_out": True,
            }
        elif not exit_results[c].get("exited"):
            exit_results[c]["timed_out"] = True

    if any(not (exit_results.get(c) or {}).get("exited") for c in targets):
        # Partial if at least one exited
        if any((exit_results.get(c) or {}).get("exited") for c in targets):
            result = "PARTIAL_SHUTDOWN"
        else:
            result = "SHUTDOWN_TIMED_OUT"
        return {
            "result": result,
            "ack_results": ack_results,
            "exit_results": exit_results,
            "lock_results": lock_results,
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "safe_message": (
                "Process exit incomplete. Forced termination is outside policy. "
                "Investigate Diagnostics; do not delete locks from the controller."
            ),
        }

    # Phase 3: lock release (normal ownership path — controller never deletes)
    while time.time() < deadline_lock:
        snap = classify_fn()
        for c in targets:
            lock_results[c] = {
                "released": _lock_released(c, snap),
                "lock_exists": bool((snap.get(c) or {}).get("lock_exists")),
                "state": (snap.get(c) or {}).get("state"),
            }
        if all((lock_results.get(c) or {}).get("released") for c in targets):
            break
        sleep_fn(poll_sec)

    for c in targets:
        if c not in lock_results:
            snap = classify_fn()
            lock_results[c] = {
                "released": _lock_released(c, snap),
                "lock_exists": bool((snap.get(c) or {}).get("lock_exists")),
                "state": (snap.get(c) or {}).get("state"),
                "timed_out": True,
            }
        elif not lock_results[c].get("released"):
            lock_results[c]["timed_out"] = True

    # Exit with stale lock
    if any(not (lock_results.get(c) or {}).get("released") for c in targets):
        if all((exit_results.get(c) or {}).get("exited") for c in targets):
            return {
                "result": "POSTCONDITION_FAILED",
                "ack_results": ack_results,
                "exit_results": exit_results,
                "lock_results": lock_results,
                "forced_termination_used": False,
                "runtime_locks_deleted_by_controller": False,
                "safe_message": (
                    "Process exited but runtime lock remains. Controller will not delete locks."
                ),
            }
        return {
            "result": "PARTIAL_SHUTDOWN",
            "ack_results": ack_results,
            "exit_results": exit_results,
            "lock_results": lock_results,
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
        }

    return {
        "result": "COOPERATIVE_SHUTDOWN_COMPLETED",
        "ack_results": ack_results,
        "exit_results": exit_results,
        "lock_results": lock_results,
        "forced_termination_used": False,
        "runtime_locks_deleted_by_controller": False,
        "safe_message": "All targeted components exited and released their own locks.",
    }


def prepare_cooperative_shutdown(
    *,
    action: str = ACTION_PREPARE,
    root: Optional[Path] = None,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
    source: str = "operator_shutdown_controller",
    reason_code: str = "OPERATOR_PREPARE",
    deliver: bool = True,
    ack_timeout_sec: float = ACK_TIMEOUT_SEC,
    exit_timeout_sec: float = EXIT_TIMEOUT_SEC,
    lock_timeout_sec: float = LOCK_RELEASE_TIMEOUT_SEC,
    poll_sec: float = POLL_SEC,
    sleep_fn: Optional[Callable[[float], None]] = None,
    clock: Optional[Callable[[], datetime]] = None,
) -> Dict[str, Any]:
    """
    Typed backend operation PREPARE_COOPERATIVE_SHUTDOWN.

    When deliver=False, evaluates eligibility only (no request write) — used by UI status.
    Tests may set deliver=True with fixtures and short timeouts.
    """
    from services.operator_runtime_classifier import classify_runtime_loops

    if action != ACTION_PREPARE:
        return {
            "result": "FAILED",
            "error_code": "DENIED_ACTION",
            "safe_message": f"Action {action} is not allowed. Only {ACTION_PREPARE} is permitted.",
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
        }

    root = Path(root) if root is not None else ROOT
    live = root / "data" / "live"
    results = root / "data" / "results"
    sleep_fn = sleep_fn or time.sleep
    now_fn = clock or _utc_now
    now = now_fn()
    action_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    audit_path = results / "operator_shutdown_actions.jsonl"

    live_before = not live_execution_appears_enabled(root=root)
    classifications = classify_runtime_loops(root=root, process_snapshot=process_snapshot)
    start_g = is_guard_active(live / "operator_start_action.guard.json", now=now)
    shut_g = is_guard_active(live / "operator_shutdown_action.guard.json", now=now)
    decision = evaluate_shutdown_decision(
        classifications,
        live_execution_blocked=live_before,
        start_guard_active=start_g,
        shutdown_guard_active=shut_g,
    )
    decision_code = decision.get("decision")

    base_audit = {
        "schema_version": SCHEMA_VERSION,
        "action_id": action_id,
        "request_id": request_id,
        "requested_at": _iso(now),
        "action": ACTION_PREPARE,
        "source": source,
        "controller_version": CONTROLLER_VERSION,
        "preflight_classification": _sanitize(classifications),
        "live_execution_blocked_before": live_before,
        "paper_arm_mutated": False,
        "orders_submitted": False,
        "approvals_created": False,
        "forced_termination_used": False,
        "runtime_locks_deleted_by_controller": False,
    }

    if decision_code == "ALREADY_STOPPED":
        rec = {
            **base_audit,
            "completed_at": _iso(now_fn()),
            "target_components": [],
            "result": "ALREADY_STOPPED",
            "safe_to_retry": True,
            "live_execution_blocked_after": not live_execution_appears_enabled(root=root),
            "safe_message": decision.get("explanation"),
            "guard_result": {"acquired": False, "reason": "not_needed"},
            "request_write_result": {"ok": False, "reason": "not_needed"},
        }
        append_shutdown_audit(rec, path=audit_path)
        return {
            "result": "ALREADY_STOPPED",
            "action_id": action_id,
            "request_id": None,
            "target_components": [],
            "safe_message": decision.get("explanation"),
            "safe_to_retry": True,
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": live_before,
            "live_execution_blocked_after": rec["live_execution_blocked_after"],
            "decision": decision_code,
        }

    refused = {
        "SHUTDOWN_REFUSED_UNSAFE_STATE",
        "SHUTDOWN_REFUSED_START_IN_PROGRESS",
        "SHUTDOWN_IN_PROGRESS",
        "SHUTDOWN_REFUSED_LIVE_SAFETY",
    }
    if decision_code in refused or not deliver:
        if decision_code in refused or not (str(decision_code).startswith("SHUTDOWN_ELIGIBLE")):
            rec = {
                **base_audit,
                "completed_at": _iso(now_fn()),
                "target_components": decision.get("target_components") or [],
                "result": decision_code,
                "safe_to_retry": decision.get("safe_to_retry"),
                "live_execution_blocked_after": not live_execution_appears_enabled(root=root),
                "safe_message": decision.get("explanation"),
                "guard_result": {"acquired": False},
                "request_write_result": {"ok": False, "reason": "refused_or_status_only"},
            }
            append_shutdown_audit(rec, path=audit_path)
            return {
                "result": decision_code,
                "action_id": action_id,
                "request_id": None,
                "target_components": decision.get("target_components") or [],
                "safe_message": decision.get("explanation"),
                "safe_to_retry": decision.get("safe_to_retry"),
                "forced_termination_used": False,
                "runtime_locks_deleted_by_controller": False,
                "paper_arm_mutated": False,
                "orders_submitted": False,
                "approvals_created": False,
                "live_execution_blocked_before": live_before,
                "live_execution_blocked_after": rec["live_execution_blocked_after"],
                "decision": decision_code,
                "classifications": classifications,
            }

    targets = list(decision.get("target_components") or [])
    expected_instances = {
        c: build_expected_instance(component=c, classification=classifications.get(c) or {})
        for c in targets
    }

    # Fail-safe: initial audit must succeed before request delivery
    initial = append_shutdown_audit(
        {
            **base_audit,
            "target_components": targets,
            "expected_instances": expected_instances,
            "result": "SHUTDOWN_BEGIN",
            "safe_message": "Preparing cooperative shutdown request.",
        },
        path=audit_path,
    )
    if not initial.get("ok"):
        return {
            "result": "AUDIT_FAILED",
            "action_id": action_id,
            "request_id": None,
            "error_code": "INITIAL_AUDIT_FAILED",
            "safe_message": "Initial audit could not be persisted; shutdown request was not delivered.",
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
            "live_execution_blocked_before": live_before,
            "live_execution_blocked_after": not live_execution_appears_enabled(root=root),
        }

    guard = acquire_shutdown_guard(
        action_id=action_id,
        request_id=request_id,
        target_components=targets,
        path=live / "operator_shutdown_action.guard.json",
        now=now,
    )
    # Also refuse if start guard became active (acquire checks sibling)
    if not guard.get("acquired"):
        reason = guard.get("reason") or "SHUTDOWN_IN_PROGRESS"
        result_code = (
            reason
            if reason in ("SHUTDOWN_IN_PROGRESS", "SHUTDOWN_REFUSED_START_IN_PROGRESS")
            else "FAILED"
        )
        append_shutdown_audit(
            {
                **base_audit,
                "completed_at": _iso(now_fn()),
                "target_components": targets,
                "result": result_code,
                "guard_result": guard,
                "request_write_result": {"ok": False, "reason": "guard_not_acquired"},
                "safe_message": f"Could not acquire shutdown guard: {reason}",
                "live_execution_blocked_after": not live_execution_appears_enabled(root=root),
            },
            path=audit_path,
        )
        return {
            "result": result_code,
            "action_id": action_id,
            "guard_result": guard,
            "safe_message": f"Shutdown guard not acquired: {reason}",
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
        }

    created = create_shutdown_request(
        target_components=targets,
        expected_instances=expected_instances,
        reason_code=reason_code,
        source=source,
        request_path=live / "operator_shutdown_request.json",
        now=now,
        request_id=request_id,
    )
    if not created.get("ok"):
        release_shutdown_guard(
            action_id=action_id, path=live / "operator_shutdown_action.guard.json"
        )
        append_shutdown_audit(
            {
                **base_audit,
                "completed_at": _iso(now_fn()),
                "target_components": targets,
                "result": "FAILED",
                "guard_result": guard,
                "request_write_result": created,
                "safe_message": "Atomic request write failed.",
                "live_execution_blocked_after": not live_execution_appears_enabled(root=root),
            },
            path=audit_path,
        )
        return {
            "result": "FAILED",
            "error_code": "REQUEST_WRITE_FAILED",
            "safe_message": "Failed to write shutdown request atomically.",
            "forced_termination_used": False,
            "runtime_locks_deleted_by_controller": False,
            "paper_arm_mutated": False,
            "orders_submitted": False,
            "approvals_created": False,
        }

    # Mark delivered
    req_doc = created.get("request") or {}
    req_doc["state"] = "DELIVERED"
    atomic_write_json(live / "operator_shutdown_request.json", req_doc)

    def classify_fn() -> Dict[str, Any]:
        return classify_runtime_loops(root=root, process_snapshot=process_snapshot)

    completion = _poll_completion(
        targets=targets,
        expected_instances=expected_instances,
        classify_fn=classify_fn,
        root=root,
        ack_timeout=ack_timeout_sec,
        exit_timeout=exit_timeout_sec,
        lock_timeout=lock_timeout_sec,
        poll_sec=poll_sec,
        sleep_fn=sleep_fn,
    )

    result = completion.get("result") or "UNKNOWN"
    # Update request state
    final_req = read_json(live / "operator_shutdown_request.json")
    if final_req and not final_req.get("__malformed__"):
        if result == "COOPERATIVE_SHUTDOWN_COMPLETED":
            final_req["state"] = "COMPLETED"
        elif result == "PARTIAL_SHUTDOWN":
            # Keep non-terminal so Start/Restart remain blocked until resolved or TTL.
            final_req["state"] = "EXIT_PENDING"
            final_req["partial_completion"] = True
            final_req["controller_timed_out"] = True
        elif result == "SHUTDOWN_TIMED_OUT":
            # Explicit controller timeout for the transaction, but keep the typed
            # request observable until expires_at so supported loops can still ack.
            ack_results = completion.get("ack_results") or {}
            any_accepted = any(bool((ack_results.get(c) or {}).get("accepted")) for c in targets)
            if any_accepted:
                final_req["state"] = "EXIT_PENDING"
                final_req["controller_timed_out"] = True
            else:
                final_req["state"] = "DELIVERED"
                final_req["controller_ack_timed_out"] = True
        elif result == "POSTCONDITION_FAILED":
            final_req["state"] = "EXIT_PENDING"
            final_req["postcondition_failed"] = True
        else:
            final_req["state"] = result
        atomic_write_json(live / "operator_shutdown_request.json", final_req)

    live_after = not live_execution_appears_enabled(root=root)
    final_audit = {
        **base_audit,
        "completed_at": _iso(now_fn()),
        "target_components": targets,
        "expected_instances": expected_instances,
        "guard_result": {"acquired": True, "action_id": action_id},
        "request_write_result": {"ok": True, "request_id": request_id},
        "acknowledgement_results": completion.get("ack_results"),
        "process_exit_results": completion.get("exit_results"),
        "lock_release_results": completion.get("lock_results"),
        "postcondition_classification": classify_fn(),
        "result": result,
        "safe_to_retry": result
        in ("PARTIAL_SHUTDOWN", "SHUTDOWN_TIMED_OUT", "POSTCONDITION_FAILED"),
        "live_execution_blocked_after": live_after,
        "forced_termination_used": False,
        "runtime_locks_deleted_by_controller": False,
        "safe_message": completion.get("safe_message") or result,
        "evidence": {
            "decision": decision_code,
            "protocol": ACTION,
        },
    }
    audit_final = append_shutdown_audit(final_audit, path=audit_path)
    if not audit_final.get("ok"):
        result = "AUDIT_FAILED"
        # Preserve request/ack evidence; no destructive rollback

    release_shutdown_guard(action_id=action_id, path=live / "operator_shutdown_action.guard.json")

    return {
        "result": result,
        "action_id": action_id,
        "request_id": request_id,
        "decision": decision_code,
        "target_components": targets,
        "expected_instances": expected_instances,
        "acknowledgement_results": completion.get("ack_results"),
        "process_exit_results": completion.get("exit_results"),
        "lock_release_results": completion.get("lock_results"),
        "safe_message": final_audit.get("safe_message"),
        "safe_to_retry": final_audit.get("safe_to_retry"),
        "forced_termination_used": False,
        "runtime_locks_deleted_by_controller": False,
        "paper_arm_mutated": False,
        "orders_submitted": False,
        "approvals_created": False,
        "live_execution_blocked_before": live_before,
        "live_execution_blocked_after": live_after,
        "audit_final_ok": audit_final.get("ok"),
    }


def controller_has_termination_paths() -> bool:
    """Static self-check — detect dangerous call forms without embedding them literally."""
    src = Path(__file__).read_text(encoding="utf-8")
    coop = Path(__file__).parent / "operator_cooperative_shutdown.py"
    coop_src = coop.read_text(encoding="utf-8") if coop.is_file() else ""
    combined = src + "\n" + coop_src
    # Build tokens so this function body does not contain the forbidden call forms.
    kill_call = "os" + "." + "kill" + "("
    proc_kill = "process" + "." + "kill" + "("
    taskkill = "task" + "kill" + ".exe"
    stop_proc = "Stop" + "-" + "Process" + " -"
    term = "Terminate" + "Process" + "("
    return any(d in combined for d in (kill_call, proc_kill, taskkill, stop_proc, term))
