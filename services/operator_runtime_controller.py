"""
operator_runtime_controller.py
Phase DO-003 — Protected local START_TRITON controller.

Accepts only action START_TRITON.
Fixed paths/scripts/arguments only. No frontend commands/PIDs/paths/env dicts.
Never terminates processes. Never deletes runtime loop locks.
Never creates/extends paper arm. Never enables live execution.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LIVE = ROOT / "data" / "live"
RESULTS = ROOT / "data" / "results"

WATCHDOG_SCRIPT = SCRIPTS / "run_watchdog_loop.ps1"
CONTINUOUS_SCRIPT = SCRIPTS / "run_continuous_loop.ps1"
POWERSHELL = (
    Path(os.environ.get("SystemRoot", r"C:\Windows"))
    / "System32"
    / "WindowsPowerShell"
    / "v1.0"
    / "powershell.exe"
)

ACTION_START = "START_TRITON"
DENIED_ACTIONS = frozenset(
    {
        "STOP_TRITON",
        "RESTART_TRITON",
        "FORCE_STOP",
        "KILL_PROCESS",
        "REPAIR_LOCK",
        "REMOVE_LOCK",
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

ALLOWED_ACTIONS = frozenset({ACTION_START})

AUDIT_PATH = RESULTS / "operator_runtime_actions.jsonl"
GUARD_PATH = LIVE / "operator_start_action.guard.json"
SCHEMA_VERSION = 1

WATCHDOG_INTERVAL = 1
CONTINUOUS_INTERVAL = 45
POSTCONDITION_TIMEOUT_SEC = 25
POSTCONDITION_POLL_SEC = 1.0
GUARD_TTL_SEC = 120
COOLDOWN_SEC = 15
IDEMPOTENCY_WINDOW_SEC = 300

# Fail-safe: if audit cannot be written before launch, refuse the Start.
AUDIT_FAIL_REFUSES_LAUNCH = True


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime] = None) -> str:
    return (dt or _utc_now()).isoformat()


def _paper_live_env() -> Dict[str, str]:
    """Canonical paper-only environment overlay (matches 931/supervisor)."""
    return {
        "TRITON_ENABLE_PAPER_EXECUTION": "1",
        "TRITON_ENABLE_AUTO_PAPER_ENTRIES": "1",
        "TRITON_ENABLE_AUTO_PAPER_APPROVAL": "1",
        "TRITON_ENABLE_LIVE_EXECUTION": "0",
        "TRITON_LIVE_EXECUTION": "0",
    }


def live_execution_appears_enabled(
    *,
    root: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> bool:
    root = Path(root) if root is not None else ROOT
    env = env if env is not None else dict(os.environ)
    if env.get("TRITON_ENABLE_LIVE_EXECUTION") == "1" or env.get("TRITON_LIVE_EXECUTION") == "1":
        return True
    readiness = root / "data" / "results" / "execution_readiness.json"
    if readiness.is_file():
        try:
            doc = json.loads(readiness.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and doc.get("live_execution_permitted") is True:
                return True
        except Exception:
            pass
    if root == ROOT:
        try:
            from services.portfolio_history_sync import load_live_execution_flag

            if load_live_execution_flag():
                return True
        except Exception:
            pass
    return False


def _sanitize_for_audit(obj: Any) -> Any:
    banned_exact = frozenset(
        {
            "api_key",
            "apikey",
            "token",
            "access_token",
            "refresh_token",
            "secret",
            "password",
            "credential",
            "authorization",
            "env",
            "environ",
            "environment",
        }
    )
    banned_substr = ("secret", "password", "credential", "api_key", "access_token")
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in banned_exact or any(b in lk for b in banned_substr):
                out[k] = "<redacted>"
            else:
                out[k] = _sanitize_for_audit(v)
        return out
    if isinstance(obj, list):
        return [_sanitize_for_audit(x) for x in obj]
    return obj


def append_runtime_action_audit(
    record: Dict[str, Any], *, path: Optional[Path] = None
) -> Dict[str, Any]:
    """Append-only JSONL audit. Returns {ok, path, error}."""
    path = path or AUDIT_PATH
    safe = _sanitize_for_audit(record)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
        return {"ok": True, "path": str(path), "error": None}
    except OSError as exc:
        return {"ok": False, "path": str(path), "error": f"audit_write_failed:{exc}"}


def read_recent_runtime_actions(
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
                out.append(_sanitize_for_audit(doc))
        except Exception:
            continue
    out.reverse()
    return out


def _read_guard(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {"corrupt": True}


def acquire_start_guard(
    *,
    action_id: str,
    idempotency_key: str,
    path: Optional[Path] = None,
    ttl_sec: int = GUARD_TTL_SEC,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Exclusive Start-action coordination guard (not proof TRITON is running)."""
    path = path or GUARD_PATH
    now = now or _utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_guard(path)
    if existing and not existing.get("corrupt"):
        exp = existing.get("expires_at")
        exp_dt = None
        try:
            exp_dt = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
        except Exception:
            exp_dt = None
        if exp_dt and exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        if exp_dt and exp_dt > now:
            # Same idempotency may observe in-progress
            if existing.get("idempotency_key") == idempotency_key:
                return {
                    "acquired": False,
                    "reason": "START_IN_PROGRESS",
                    "same_idempotency": True,
                    "guard": existing,
                }
            return {
                "acquired": False,
                "reason": "START_IN_PROGRESS",
                "same_idempotency": False,
                "guard": existing,
            }
        # expired — may overwrite (does not touch loop locks)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "action_id": action_id,
        "idempotency_key": idempotency_key,
        "action": ACTION_START,
        "created_at": _iso(now),
        "expires_at": _iso(now + timedelta(seconds=ttl_sec)),
        "owner": "operator_runtime_controller",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"acquired": True, "reason": "ACQUIRED", "guard": payload}


def release_start_guard(
    *,
    action_id: str,
    path: Optional[Path] = None,
) -> None:
    path = path or GUARD_PATH
    if not path.is_file():
        return
    existing = _read_guard(path)
    if existing.get("action_id") == action_id or existing.get("corrupt"):
        try:
            path.unlink()
        except OSError:
            pass


def _find_recent_idempotent(
    idempotency_key: str,
    *,
    audit_path: Path,
    window_sec: int = IDEMPOTENCY_WINDOW_SEC,
) -> Optional[Dict[str, Any]]:
    now = _utc_now()
    for rec in read_recent_runtime_actions(path=audit_path, limit=50):
        if rec.get("idempotency_key") != idempotency_key:
            continue
        if rec.get("action") != ACTION_START:
            continue
        if rec.get("result") in (None, "START_IN_PROGRESS", "START_BEGIN"):
            continue
        # Prefer terminal outcomes only
        if rec.get("result") not in (
            "STARTED",
            "PARTIAL_START",
            "ALREADY_RUNNING",
            "START_FAILED",
            "POSTCONDITION_FAILED",
            "START_REFUSED_UNSAFE_STATE",
            "START_REFUSED_LIVE_SAFETY",
            "START_REFUSED_EXISTING_UNHEALTHY_PROCESS",
        ):
            continue
        completed = rec.get("completed_at") or rec.get("requested_at")
        try:
            dt = datetime.fromisoformat(str(completed).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if (now - dt).total_seconds() <= window_sec:
            return rec
    return None


def verify_fixed_paths(*, root: Path = ROOT) -> Dict[str, Any]:
    scripts = root / "scripts"
    wd = scripts / "run_watchdog_loop.ps1"
    cont = scripts / "run_continuous_loop.ps1"
    ps = POWERSHELL
    missing = []
    if not wd.is_file():
        missing.append(str(wd))
    if not cont.is_file():
        missing.append(str(cont))
    if sys.platform == "win32" and not ps.is_file():
        missing.append(str(ps))
    return {
        "ok": len(missing) == 0,
        "watchdog_script": str(wd),
        "continuous_script": str(cont),
        "powershell": str(ps),
        "repo_root": str(root),
        "missing": missing,
    }


def default_launch_loop(component: str, *, root: Path = ROOT) -> Dict[str, Any]:
    """Launch one allowlisted loop script with fixed arguments. No shell=True."""
    if component == "watchdog":
        script = root / "scripts" / "run_watchdog_loop.ps1"
        interval = WATCHDOG_INTERVAL
    elif component == "continuous":
        script = root / "scripts" / "run_continuous_loop.ps1"
        interval = CONTINUOUS_INTERVAL
    else:
        return {"ok": False, "error": "unknown_component", "component": component}

    if not script.is_file():
        return {"ok": False, "error": "missing_script", "component": component, "path": str(script)}

    env = dict(os.environ)
    env.update(_paper_live_env())
    # Never pass through a live-enable if somehow present
    env["TRITON_ENABLE_LIVE_EXECUTION"] = "0"
    env["TRITON_LIVE_EXECUTION"] = "0"

    args = [
        str(POWERSHELL if POWERSHELL.is_file() else "powershell.exe"),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-IntervalMinutes",
        str(interval),
    ]
    try:
        creationflags = 0
        if sys.platform == "win32":
            # Streamlit/agent Python often runs inside a Windows Job Object.
            # CREATE_BREAKAWAY_FROM_JOB is required so child runtimes survive
            # st.rerun() / handler completion. DETACHED_PROCESS alone does not
            # break away from the parent job and children are killed on rerun.
            creationflags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
            )
        proc = subprocess.Popen(
            args,
            cwd=str(root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            shell=False,
            creationflags=creationflags,
        )
        return {
            "ok": True,
            "component": component,
            "child_pid": proc.pid,
            "script": str(script),
            "interval_minutes": interval,
            "argv_fixed": True,
            "creationflags": int(creationflags),
            "breakaway_from_job": bool(
                creationflags & getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
            ),
        }
    except OSError as exc:
        return {"ok": False, "error": f"launch_failed:{exc}", "component": component}


def _wait_postconditions(
    components: Sequence[str],
    *,
    classify_fn: Callable[[], Dict[str, Any]],
    timeout_sec: float = POSTCONDITION_TIMEOUT_SEC,
    poll_sec: float = POSTCONDITION_POLL_SEC,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    last: Dict[str, Any] = {}
    while time.time() < deadline:
        last = classify_fn()
        cont = (last.get("continuous") or {}).get("state")
        wd = (last.get("watchdog") or {}).get("state")
        ok = True
        for c in components:
            state = cont if c == "continuous" else wd
            if state != "HEALTHY_RUNNING":
                ok = False
                break
        # Also ensure no duplicates appeared on either loop
        if cont == "DUPLICATE_PROCESSES" or wd == "DUPLICATE_PROCESSES":
            return {"ok": False, "reason": "duplicates_appeared", "snapshot": last}
        if ok:
            return {"ok": True, "reason": "postconditions_met", "snapshot": last}
        time.sleep(poll_sec)
    return {"ok": False, "reason": "postcondition_timeout", "snapshot": last}


def start_eligibility(
    *,
    root: Optional[Path] = None,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Read-only Start eligibility for Operations/Diagnostics."""
    from services.operator_runtime_classifier import (
        classify_runtime_loops,
        evaluate_start_decision,
    )
    from services.operator_console_status import map_paper_arm_display

    root = Path(root) if root is not None else ROOT
    live_enabled = live_execution_appears_enabled(root=root)
    classifications = classify_runtime_loops(root=root, process_snapshot=process_snapshot)
    decision = evaluate_start_decision(classifications, live_execution_blocked=not live_enabled)

    # DO-004: refuse Start while cooperative shutdown is unresolved
    try:
        from services.operator_shutdown_controller import shutdown_active

        shut = shutdown_active(root=root)
        if shut.get("active"):
            decision = {
                "decision": "START_REFUSED_SHUTDOWN_IN_PROGRESS",
                "components_to_start": [],
                "preserved_components": [],
                "explanation": (
                    "Cooperative shutdown is active or unresolved. Start refused until "
                    "shutdown completes cleanly and runtime classification is clear."
                ),
                "safe_to_retry": True,
                "shutdown_active": shut,
            }
    except Exception:
        pass

    paths = verify_fixed_paths(root=root)
    if not paths["ok"] and decision["decision"] in (
        "START_ALLOWED",
        "PARTIAL_START_ALLOWED",
    ):
        decision = {
            **decision,
            "decision": "START_REFUSED_UNSAFE_STATE",
            "components_to_start": [],
            "explanation": f"Required fixed scripts missing: {paths['missing']}",
            "safe_to_retry": False,
        }

    paper_status = "UNKNOWN"
    try:
        if root == ROOT:
            from services.paper_arm_status import classify_paper_arm_confirm_status

            paper_status = map_paper_arm_display(classify_paper_arm_confirm_status())
        else:
            arm = root / "data" / "live" / "paper_arm_confirm.json"
            paper_status = "MISSING" if not arm.is_file() else "UNKNOWN"
    except Exception:
        paper_status = "UNKNOWN"

    guard = _read_guard(root / "data" / "live" / "operator_start_action.guard.json")
    guard_active = False
    if guard and not guard.get("corrupt"):
        try:
            exp = datetime.fromisoformat(str(guard.get("expires_at")).replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            guard_active = exp > _utc_now()
        except Exception:
            guard_active = False

    start_meaningful = decision["decision"] in (
        "START_ALLOWED",
        "PARTIAL_START_ALLOWED",
        "ALREADY_RUNNING",
    )
    button_enabled = (
        decision["decision"] in ("START_ALLOWED", "PARTIAL_START_ALLOWED") and not guard_active
    )

    return {
        "decision": decision["decision"],
        "explanation": decision.get("explanation"),
        "components_to_start": decision.get("components_to_start") or [],
        "preserved_components": decision.get("preserved_components") or [],
        "classifications": classifications,
        "live_execution_blocked": not live_enabled,
        "paper_arm_status": paper_status,
        "fixed_paths": paths,
        "guard_active": guard_active,
        "guard": guard if guard_active else {},
        "start_button_visible": start_meaningful or True,
        "start_button_enabled": button_enabled,
        "already_running": decision["decision"] == "ALREADY_RUNNING",
        "stop_available": False,
        "restart_available": False,
        "denied_actions": sorted(DENIED_ACTIONS),
        "allowed_action": ACTION_START,
    }


def execute_start_triton(
    *,
    action: str = ACTION_START,
    idempotency_key: Optional[str] = None,
    source: str = "operator_console",
    operator_context: Optional[Dict[str, Any]] = None,
    root: Optional[Path] = None,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
    launch_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    classify_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    audit_path: Optional[Path] = None,
    guard_path: Optional[Path] = None,
    skip_postcondition_wait: bool = False,
    # Explicitly rejected kwargs — frontend must not supply these
    command: Any = None,
    pid: Any = None,
    path: Any = None,
    env: Any = None,
    shell_argument: Any = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Execute START_TRITON with full safety policy.

    Rejects any frontend-supplied command/pid/path/env/shell_argument.
    """
    from services.operator_runtime_classifier import (
        classify_runtime_loops,
        evaluate_start_decision,
    )

    root = Path(root) if root is not None else ROOT
    audit_path = audit_path or (root / "data" / "results" / "operator_runtime_actions.jsonl")
    guard_path = guard_path or (root / "data" / "live" / "operator_start_action.guard.json")
    action_id = str(uuid.uuid4())
    idem = idempotency_key or str(uuid.uuid4())
    requested_at = _iso()
    operator_context = operator_context or {"source": source}

    def _result(**kwargs: Any) -> Dict[str, Any]:
        base = {
            "schema_version": SCHEMA_VERSION,
            "action_id": action_id,
            "idempotency_key": idem,
            "requested_at": requested_at,
            "completed_at": _iso(),
            "action": ACTION_START,
            "source": source,
            "operator_context": _sanitize_for_audit(operator_context),
            "paper_arm_mutated": False,
            "runtime_locks_deleted": False,
            "processes_terminated": False,
        }
        base.update(kwargs)
        return base

    # Reject unsupported / injected inputs
    if (
        command is not None
        or pid is not None
        or path is not None
        or env is not None
        or shell_argument is not None
        or extra
    ):
        rec = _result(
            result="START_REFUSED_UNSAFE_STATE",
            error_code="FRONTEND_INJECTION_REFUSED",
            safe_message=(
                "Start refused: frontend must not supply command, PID, path, environment, "
                "or shell arguments."
            ),
            safe_to_retry=False,
            live_execution_blocked_before=True,
            live_execution_blocked_after=True,
            launched_components=[],
            preserved_components=[],
            component_results={},
        )
        append_runtime_action_audit(rec, path=audit_path)
        return rec

    if action in DENIED_ACTIONS or action != ACTION_START:
        rec = _result(
            result="START_REFUSED_UNSAFE_STATE",
            error_code="ACTION_NOT_ALLOWED",
            safe_message=f"Action {action!r} is not allowed. Only START_TRITON is supported in DO-003.",
            safe_to_retry=False,
            live_execution_blocked_before=not live_execution_appears_enabled(root=root),
            live_execution_blocked_after=not live_execution_appears_enabled(root=root),
            launched_components=[],
            preserved_components=[],
            component_results={},
        )
        append_runtime_action_audit(rec, path=audit_path)
        return rec

    # Idempotency: return prior completed result
    prior = _find_recent_idempotent(idem, audit_path=audit_path)
    if prior is not None:
        replay = dict(prior)
        replay["replayed"] = True
        replay["action_id"] = action_id
        replay["requested_at"] = requested_at
        replay["completed_at"] = _iso()
        replay["safe_message"] = (
            prior.get("safe_message")
            or "Repeated Start request with the same idempotency key — prior result returned; no relaunch."
        )
        append_runtime_action_audit(replay, path=audit_path)
        return replay

    # Acquire guard
    guard = acquire_start_guard(action_id=action_id, idempotency_key=idem, path=guard_path)
    if not guard.get("acquired"):
        rec = _result(
            result="START_IN_PROGRESS",
            error_code="START_IN_PROGRESS",
            safe_message="A Start action is already in progress. Wait for it to finish.",
            safe_to_retry=True,
            live_execution_blocked_before=not live_execution_appears_enabled(root=root),
            live_execution_blocked_after=not live_execution_appears_enabled(root=root),
            launched_components=[],
            preserved_components=[],
            component_results={},
            evidence={"guard": guard.get("guard")},
        )
        append_runtime_action_audit(rec, path=audit_path)
        return rec

    try:
        live_before_blocked = not live_execution_appears_enabled(root=root)
        if not live_before_blocked:
            rec = _result(
                result="START_REFUSED_LIVE_SAFETY",
                error_code="LIVE_SAFETY",
                safe_message=(
                    "TRITON was not started because live execution may be enabled. "
                    "Critical safety refusal — no override."
                ),
                safe_to_retry=False,
                live_execution_blocked_before=False,
                live_execution_blocked_after=False,
                preflight_classification={},
                launched_components=[],
                preserved_components=[],
                component_results={},
            )
            append_runtime_action_audit(rec, path=audit_path)
            return rec

        paths = verify_fixed_paths(root=root)
        if not paths["ok"]:
            rec = _result(
                result="START_REFUSED_UNSAFE_STATE",
                error_code="MISSING_FIXED_SCRIPT",
                safe_message=f"Required fixed scripts missing: {paths['missing']}",
                safe_to_retry=False,
                live_execution_blocked_before=True,
                live_execution_blocked_after=True,
                evidence={"fixed_paths": paths},
                launched_components=[],
                preserved_components=[],
                component_results={},
            )
            append_runtime_action_audit(rec, path=audit_path)
            return rec

        def _classify() -> Dict[str, Any]:
            if classify_fn is not None:
                return classify_fn()
            return classify_runtime_loops(root=root, process_snapshot=process_snapshot)

        pre = _classify()
        decision = evaluate_start_decision(pre, live_execution_blocked=True)

        # DO-004: refuse Start while cooperative shutdown is unresolved
        from services.operator_shutdown_controller import shutdown_active

        shut = shutdown_active(root=root)
        if shut.get("active"):
            rec = _result(
                result="START_REFUSED_SHUTDOWN_IN_PROGRESS",
                error_code="SHUTDOWN_IN_PROGRESS",
                safe_message=(
                    "Start refused: cooperative shutdown is active or unresolved. "
                    "Wait for clean completion before starting."
                ),
                safe_to_retry=True,
                live_execution_blocked_before=True,
                live_execution_blocked_after=not live_execution_appears_enabled(root=root),
                preflight_classification=pre,
                launched_components=[],
                preserved_components=[],
                component_results={},
                evidence={"shutdown_active": shut},
            )
            append_runtime_action_audit(rec, path=audit_path)
            return rec

        if decision["decision"] == "ALREADY_RUNNING":
            rec = _result(
                result="ALREADY_RUNNING",
                error_code=None,
                safe_message="TRITON was already running safely. No additional process was started.",
                safe_to_retry=True,
                live_execution_blocked_before=True,
                live_execution_blocked_after=not live_execution_appears_enabled(root=root),
                preflight_classification=pre,
                postcondition_classification=pre,
                requested_components=[],
                preserved_components=decision.get("preserved_components") or [],
                launched_components=[],
                component_results={},
                evidence={"decision": decision},
            )
            append_runtime_action_audit(rec, path=audit_path)
            return rec

        if decision["decision"] not in ("START_ALLOWED", "PARTIAL_START_ALLOWED"):
            rec = _result(
                result=decision["decision"],
                error_code=decision["decision"],
                safe_message=decision.get("explanation")
                or "TRITON was not started because the current runtime state is unsafe or ambiguous.",
                safe_to_retry=bool(decision.get("safe_to_retry")),
                live_execution_blocked_before=True,
                live_execution_blocked_after=True,
                preflight_classification=pre,
                requested_components=[],
                preserved_components=[],
                launched_components=[],
                component_results={},
                evidence={"decision": decision},
            )
            append_runtime_action_audit(rec, path=audit_path)
            return rec

        # Audit must be writable before launch (fail-safe)
        probe = {
            "schema_version": SCHEMA_VERSION,
            "action_id": action_id,
            "idempotency_key": idem,
            "requested_at": requested_at,
            "action": ACTION_START,
            "result": "START_BEGIN",
            "safe_message": "Start preflight passed; launching allowlisted components.",
            "requested_components": decision.get("components_to_start") or [],
            "preserved_components": decision.get("preserved_components") or [],
            "live_execution_blocked_before": True,
            "paper_arm_mutated": False,
        }
        audit_ok = append_runtime_action_audit(probe, path=audit_path)
        if not audit_ok.get("ok") and AUDIT_FAIL_REFUSES_LAUNCH:
            rec = _result(
                result="START_REFUSED_UNSAFE_STATE",
                error_code="AUDIT_UNAVAILABLE",
                safe_message=(
                    "Start refused because the operator audit destination is unavailable. "
                    "Fail-safe policy: do not launch without auditability."
                ),
                safe_to_retry=True,
                live_execution_blocked_before=True,
                live_execution_blocked_after=True,
                evidence={"audit": audit_ok},
                launched_components=[],
                preserved_components=decision.get("preserved_components") or [],
                component_results={},
            )
            return rec

        to_start = list(decision.get("components_to_start") or [])
        preserved = list(decision.get("preserved_components") or [])
        # Fixed order: watchdog then continuous
        ordered = [c for c in ("watchdog", "continuous") if c in to_start]
        launch = launch_fn or (lambda component: default_launch_loop(component, root=root))
        component_results: Dict[str, Any] = {}
        launched: List[str] = []
        failed: List[str] = []

        for component in ordered:
            # Re-check race with scheduled task / concurrent start before each launch
            mid = _classify()
            mid_state = (mid.get(component) or {}).get("state")
            if mid_state == "HEALTHY_RUNNING":
                component_results[component] = {
                    "status": "PRESERVED_RACE",
                    "detail": "Became healthy before launch (possible Scheduled Task race).",
                }
                if component not in preserved:
                    preserved.append(component)
                continue
            if mid_state in (
                "DUPLICATE_PROCESSES",
                "STALE_LOCK_NO_PROCESS",
                "PROCESS_WITHOUT_VALID_LOCK",
                "PID_MISMATCH",
                "COMMAND_MISMATCH",
                "AMBIGUOUS",
                "UNKNOWN",
            ):
                component_results[component] = {
                    "status": "REFUSED_RACE",
                    "state": mid_state,
                }
                failed.append(component)
                break

            lr = launch(component)
            component_results[component] = {"launch": lr}
            if not lr.get("ok"):
                failed.append(component)
                break
            launched.append(component)

            if not skip_postcondition_wait:
                pc = _wait_postconditions(
                    [component],
                    classify_fn=_classify,
                    timeout_sec=POSTCONDITION_TIMEOUT_SEC,
                    poll_sec=POSTCONDITION_POLL_SEC,
                )
                component_results[component]["postcondition"] = {
                    "ok": pc.get("ok"),
                    "reason": pc.get("reason"),
                    "state": ((pc.get("snapshot") or {}).get(component) or {}).get("state"),
                }
                if not pc.get("ok"):
                    failed.append(component)
                    # Do not kill the launched component (DO-003)
                    break

        post = _classify()
        live_after_blocked = not live_execution_appears_enabled(root=root)

        # Final duplicate check
        cont_state = (post.get("continuous") or {}).get("state")
        wd_state = (post.get("watchdog") or {}).get("state")
        if cont_state == "DUPLICATE_PROCESSES" or wd_state == "DUPLICATE_PROCESSES":
            rec = _result(
                result="POSTCONDITION_FAILED",
                error_code="DUPLICATES_AFTER_START",
                safe_message=(
                    "Postcondition failed: duplicate loop processes detected after Start. "
                    "No automatic termination was performed."
                ),
                safe_to_retry=False,
                live_execution_blocked_before=True,
                live_execution_blocked_after=live_after_blocked,
                preflight_classification=pre,
                postcondition_classification=post,
                requested_components=ordered,
                preserved_components=preserved,
                launched_components=launched,
                component_results=component_results,
            )
            append_runtime_action_audit(rec, path=audit_path)
            return rec

        if not live_after_blocked:
            rec = _result(
                result="POSTCONDITION_FAILED",
                error_code="LIVE_ENABLED_AFTER_START",
                safe_message=(
                    "Critical: live execution appears enabled after Start. "
                    "Investigate immediately. No live-enable control is provided."
                ),
                safe_to_retry=False,
                live_execution_blocked_before=True,
                live_execution_blocked_after=False,
                preflight_classification=pre,
                postcondition_classification=post,
                requested_components=ordered,
                preserved_components=preserved,
                launched_components=launched,
                component_results=component_results,
            )
            append_runtime_action_audit(rec, path=audit_path)
            return rec

        # Determine overall result
        need = set(ordered)
        healthy_needed = set()
        for c in ordered:
            st = cont_state if c == "continuous" else wd_state
            if st == "HEALTHY_RUNNING":
                healthy_needed.add(c)
        # Preserved components must remain healthy
        for c in preserved:
            st = cont_state if c == "continuous" else wd_state
            if st != "HEALTHY_RUNNING":
                failed.append(c)

        if failed and healthy_needed:
            result = "PARTIAL_START"
            msg = (
                "Partial Start: at least one required loop is healthy, but another failed. "
                "The healthy loop was not terminated (destructive rollback is outside DO-003). "
                "Review Diagnostics."
            )
        elif (
            failed
            and not healthy_needed
            and not (cont_state == "HEALTHY_RUNNING" and wd_state == "HEALTHY_RUNNING")
        ):
            result = "START_FAILED" if launched else "POSTCONDITION_FAILED"
            msg = (
                "TRITON Start failed postcondition or launch validation. "
                "No claim of success is made from process creation alone."
            )
        elif cont_state == "HEALTHY_RUNNING" and wd_state == "HEALTHY_RUNNING":
            result = "STARTED" if launched else "ALREADY_RUNNING"
            msg = (
                "TRITON started successfully. One continuous loop and one watchdog are running. "
                "Live execution remains blocked."
                if launched
                else "TRITON was already running safely. No additional process was started."
            )
        elif need and healthy_needed == need:
            result = "STARTED"
            msg = (
                "TRITON started successfully for the requested components. "
                "Live execution remains blocked."
            )
        else:
            result = "POSTCONDITION_FAILED"
            msg = "Start did not meet verified process and lock postconditions."

        # Paper arm status (read-only)
        paper_arm_status = "UNKNOWN"
        try:
            if root == ROOT:
                from services.paper_arm_status import classify_paper_arm_confirm_status
                from services.operator_console_status import map_paper_arm_display

                paper_arm_status = map_paper_arm_display(classify_paper_arm_confirm_status())
        except Exception:
            pass

        rec = _result(
            result=result,
            error_code=(
                None if result in ("STARTED", "ALREADY_RUNNING", "PARTIAL_START") else result
            ),
            safe_message=msg,
            safe_to_retry=result in ("START_FAILED", "POSTCONDITION_FAILED", "PARTIAL_START"),
            live_execution_blocked_before=True,
            live_execution_blocked_after=True,
            paper_arm_status=paper_arm_status,
            preflight_classification=pre,
            postcondition_classification=post,
            requested_components=ordered,
            preserved_components=preserved,
            launched_components=launched,
            component_results=component_results,
            evidence={"decision": decision, "fixed_paths": paths},
        )
        append_runtime_action_audit(rec, path=audit_path)
        return rec
    finally:
        release_start_guard(action_id=action_id, path=guard_path)
