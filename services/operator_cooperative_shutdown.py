"""
operator_cooperative_shutdown.py
Phase DO-004 — Cooperative shutdown request/ack contract helpers.

No process termination. No runtime lock deletion by this module's controller APIs.
Loops may release only their own lock via normal finally after accepting shutdown.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
RESULTS = ROOT / "data" / "results"

REQUEST_PATH = LIVE / "operator_shutdown_request.json"
ACK_CONTINUOUS = LIVE / "operator_shutdown_ack_continuous.json"
ACK_WATCHDOG = LIVE / "operator_shutdown_ack_watchdog.json"
SHUTDOWN_GUARD = LIVE / "operator_shutdown_action.guard.json"
START_GUARD = LIVE / "operator_start_action.guard.json"
AUDIT_PATH = RESULTS / "operator_shutdown_actions.jsonl"

SCHEMA_VERSION = 1
CONTROLLER_VERSION = "DO-004"
ACTION = "REQUEST_COOPERATIVE_SHUTDOWN"

TARGETS = ("CONTINUOUS", "WATCHDOG", "BOTH")
COMPONENTS = ("continuous", "watchdog")

REQUEST_TTL_SEC = 900
GUARD_TTL_SEC = 900

# Interruptible wait check intervals (recognition targets while sleeping)
CONTINUOUS_WAIT_CHECK_SEC = 10
WATCHDOG_WAIT_CHECK_SEC = 5

# Exit codes for CLI poll used by loop scripts
EXIT_NO_REQUEST = 0
EXIT_ACCEPTED_EXIT_NOW = 10
EXIT_REFUSED = 11
EXIT_COMPLETING = 12
EXIT_ERROR = 2


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime] = None) -> str:
    return (dt or _utc_now()).isoformat()


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        s = str(value).strip().replace("Z", "+00:00")
        if " " in s and "T" not in s:
            s = s.replace(" ", "T")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _sanitize(obj: Any) -> Any:
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
            "command",
            "shell_argument",
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
                out[k] = _sanitize(v)
        return out
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    return obj


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Atomic replace of JSON file in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(_sanitize(payload), indent=2, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(prefix=".shutdown_", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))
        return {"ok": True, "path": str(path)}
    except OSError as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return {"ok": False, "error": str(exc), "path": str(path)}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {"__malformed__": True}
    except Exception:
        return {"__malformed__": True}


def append_shutdown_audit(record: Dict[str, Any], *, path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or AUDIT_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_sanitize(record), ensure_ascii=False) + "\n")
        return {"ok": True, "path": str(path)}
    except OSError as exc:
        return {"ok": False, "error": f"audit_write_failed:{exc}", "path": str(path)}


def read_recent_shutdown_actions(
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


def command_fingerprint(script_name: str, pid: Optional[int] = None) -> str:
    return f"ps1:{script_name}:pid={pid if pid is not None else '?'}"


def build_expected_instance(
    *,
    component: str,
    classification: Dict[str, Any],
) -> Dict[str, Any]:
    pids = list(classification.get("process_pids") or [])
    pid = int(pids[0]) if len(pids) == 1 else None
    script = str(classification.get("expected_script") or "")
    return {
        "component": component,
        "expected_pid": pid,
        "expected_lock_path": classification.get("expected_lock_path"),
        "expected_lock_pid": classification.get("lock_pid"),
        "expected_command_fingerprint": command_fingerprint(script, pid),
        "expected_script": script,
        "state": classification.get("state"),
    }


def is_guard_active(path: Path, *, now: Optional[datetime] = None) -> bool:
    now = now or _utc_now()
    doc = read_json(path)
    if not doc or doc.get("__malformed__") or doc.get("corrupt"):
        return False
    exp = _parse_dt(doc.get("expires_at"))
    return bool(exp and exp > now)


def acquire_shutdown_guard(
    *,
    action_id: str,
    request_id: str,
    target_components: Sequence[str],
    path: Optional[Path] = None,
    start_guard_path: Optional[Path] = None,
    ttl_sec: int = GUARD_TTL_SEC,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    path = path or SHUTDOWN_GUARD
    start_path = start_guard_path or (path.parent / "operator_start_action.guard.json")
    now = now or _utc_now()
    if is_guard_active(path, now=now):
        return {"acquired": False, "reason": "SHUTDOWN_IN_PROGRESS", "guard": read_json(path)}
    if is_guard_active(start_path, now=now):
        return {
            "acquired": False,
            "reason": "SHUTDOWN_REFUSED_START_IN_PROGRESS",
            "guard": read_json(start_path),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "action_id": action_id,
        "request_id": request_id,
        "owner": "operator_shutdown_controller",
        "created_at": _iso(now),
        "expires_at": _iso(now + timedelta(seconds=ttl_sec)),
        "target_components": list(target_components),
        "action": ACTION,
    }
    wr = atomic_write_json(path, payload)
    if not wr.get("ok"):
        return {"acquired": False, "reason": "GUARD_WRITE_FAILED", "error": wr.get("error")}
    return {"acquired": True, "reason": "ACQUIRED", "guard": payload}


def release_shutdown_guard(*, action_id: str, path: Optional[Path] = None) -> None:
    path = path or SHUTDOWN_GUARD
    doc = read_json(path)
    if not doc or doc.get("action_id") == action_id or doc.get("__malformed__"):
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def create_shutdown_request(
    *,
    target_components: Sequence[str],
    expected_instances: Dict[str, Any],
    reason_code: str = "OPERATOR_PREPARE",
    source: str = "operator_shutdown_controller",
    request_path: Optional[Path] = None,
    ttl_sec: int = REQUEST_TTL_SEC,
    now: Optional[datetime] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    now = now or _utc_now()
    rid = request_id or str(uuid.uuid4())
    comps = [c.lower() for c in target_components]
    if set(comps) == {"continuous", "watchdog"}:
        target_label = "BOTH"
    elif comps == ["continuous"]:
        target_label = "CONTINUOUS"
    elif comps == ["watchdog"]:
        target_label = "WATCHDOG"
    else:
        return {"ok": False, "error": "invalid_targets"}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "request_id": rid,
        "action": ACTION,
        "requested_at": _iso(now),
        "expires_at": _iso(now + timedelta(seconds=ttl_sec)),
        "target_components": comps,
        "target": target_label,
        "expected_instances": expected_instances,
        "reason_code": reason_code,
        "source": source,
        "controller_version": CONTROLLER_VERSION,
        "state": "CREATED",
    }
    # Strip any unsafe injected fields if present in expected_instances copies
    path = request_path or REQUEST_PATH
    wr = atomic_write_json(path, payload)
    if not wr.get("ok"):
        return {"ok": False, "error": wr.get("error"), "request_id": rid}
    return {"ok": True, "request": payload, "path": str(path)}


def validate_request_for_component(
    request: Dict[str, Any],
    *,
    component: str,
    actual_pid: int,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Loop-side validation: matching identity, not expired, correct action."""
    now = now or _utc_now()
    if not request or request.get("__malformed__"):
        return {
            "ok": False,
            "state": "REFUSED_MALFORMED",
            "safe_message": "Malformed shutdown request.",
        }
    if request.get("action") != ACTION:
        return {
            "ok": False,
            "state": "REFUSED_UNSUPPORTED_ACTION",
            "safe_message": "Unsupported shutdown action.",
        }
    exp = _parse_dt(request.get("expires_at"))
    if not exp or exp <= now:
        return {
            "ok": False,
            "state": "REFUSED_EXPIRED",
            "safe_message": "Shutdown request expired.",
        }
    targets = [str(t).lower() for t in (request.get("target_components") or [])]
    if component not in targets and request.get("target") not in (
        "BOTH",
        component.upper(),
        "CONTINUOUS" if component == "continuous" else "WATCHDOG",
    ):
        if component not in targets:
            return {
                "ok": False,
                "state": "REFUSED_WRONG_INSTANCE",
                "safe_message": f"Request does not target {component}.",
            }
    expected = (request.get("expected_instances") or {}).get(component) or {}
    expected_pid = expected.get("expected_pid")
    if expected_pid is None or int(expected_pid) != int(actual_pid):
        return {
            "ok": False,
            "state": "REFUSED_WRONG_INSTANCE",
            "safe_message": (
                f"Request expected PID {expected_pid} does not match this instance PID {actual_pid}."
            ),
        }
    return {
        "ok": True,
        "state": "ACCEPTED",
        "request_id": request.get("request_id"),
        "expected": expected,
        "safe_message": "Shutdown request accepted for this verified instance.",
    }


def write_ack(
    *,
    component: str,
    request_id: str,
    state: str,
    actual_pid: int,
    safe_boundary: str,
    current_cycle_state: str,
    next_action: str,
    safe_message: str,
    ack_path: Optional[Path] = None,
    script_name: str = "",
    lock_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = ack_path
    if path is None:
        path = ACK_CONTINUOUS if component == "continuous" else ACK_WATCHDOG
    payload = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "component": component,
        "acknowledged_at": _iso(),
        "state": state,
        "actual_pid": actual_pid,
        "command_fingerprint": command_fingerprint(script_name, actual_pid),
        "lock_identity": lock_identity or {},
        "safe_boundary": safe_boundary,
        "current_cycle_state": current_cycle_state,
        "next_action": next_action,
        "safe_message": safe_message,
    }
    return atomic_write_json(path, payload)


def poll_shutdown_for_loop(
    *,
    component: str,
    actual_pid: int,
    script_name: str,
    root: Optional[Path] = None,
    cycle_state: str = "BETWEEN_CYCLES",
    safe_boundary: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Called by continuous/watchdog loops at safe checkpoints.

    Returns action: NONE | EXIT_NOW | CONTINUE_REFUSED
    """
    root = Path(root) if root is not None else ROOT
    live = root / "data" / "live"
    request_path = live / "operator_shutdown_request.json"
    ack_path = (
        live / "operator_shutdown_ack_continuous.json"
        if component == "continuous"
        else live / "operator_shutdown_ack_watchdog.json"
    )
    if safe_boundary is None:
        if cycle_state in ("BETWEEN_CYCLES", "PRE_CYCLE"):
            safe_boundary = "PRE_CYCLE"
        elif cycle_state == "WAITING":
            safe_boundary = "INTERRUPTIBLE_WAIT"
        else:
            safe_boundary = "POST_CYCLE_PRE_SLEEP"
    req = read_json(request_path)
    if not req:
        return {"action": "NONE", "detail": "no_request"}
    result = validate_request_for_component(req, component=component, actual_pid=actual_pid)
    if not result.get("ok"):
        write_ack(
            component=component,
            request_id=str(req.get("request_id") or ""),
            state=str(result.get("state") or "REFUSED_MALFORMED"),
            actual_pid=actual_pid,
            safe_boundary=safe_boundary,
            current_cycle_state=cycle_state,
            next_action="CONTINUE",
            safe_message=str(result.get("safe_message") or "refused"),
            ack_path=ack_path,
            script_name=script_name,
        )
        return {"action": "CONTINUE_REFUSED", "detail": result}
    # Accepted — acknowledge and instruct exit at this safe boundary
    write_ack(
        component=component,
        request_id=str(req.get("request_id")),
        state="ACCEPTED",
        actual_pid=actual_pid,
        safe_boundary=safe_boundary,
        current_cycle_state=cycle_state,
        next_action="EXIT",
        safe_message="Accepted at safe boundary; exiting through normal cleanup.",
        ack_path=ack_path,
        script_name=script_name,
    )
    # Mark completing
    write_ack(
        component=component,
        request_id=str(req.get("request_id")),
        state="EXITING",
        actual_pid=actual_pid,
        safe_boundary=safe_boundary,
        current_cycle_state="EXIT_PENDING",
        next_action="EXIT",
        safe_message="Exiting after accepted cooperative shutdown.",
        ack_path=ack_path,
        script_name=script_name,
    )
    return {
        "action": "EXIT_NOW",
        "request_id": req.get("request_id"),
        "detail": result,
    }


def interruptible_sleep_seconds(
    total_seconds: int,
    *,
    component: str,
    actual_pid: int,
    script_name: str,
    check_interval_sec: int,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Sleep in chunks; return early if matching shutdown accepted."""
    remaining = max(0, int(total_seconds))
    interval = max(1, int(check_interval_sec))
    while remaining > 0:
        poll = poll_shutdown_for_loop(
            component=component,
            actual_pid=actual_pid,
            script_name=script_name,
            root=root,
            cycle_state="WAITING",
            safe_boundary="INTERRUPTIBLE_WAIT",
        )
        if poll.get("action") == "EXIT_NOW":
            return {"slept": total_seconds - remaining, "shutdown": True, "poll": poll}
        chunk = min(interval, remaining)
        time.sleep(chunk)
        remaining -= chunk
    # Final check
    poll = poll_shutdown_for_loop(
        component=component,
        actual_pid=actual_pid,
        script_name=script_name,
        root=root,
        cycle_state="WAITING",
        safe_boundary="INTERRUPTIBLE_WAIT",
    )
    return {
        "slept": total_seconds,
        "shutdown": poll.get("action") == "EXIT_NOW",
        "poll": poll,
    }


def _cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TRITON cooperative shutdown loop helper")
    parser.add_argument("--component", choices=["continuous", "watchdog"], required=True)
    parser.add_argument("--pid", type=int, default=os.getpid())
    parser.add_argument("--root", type=str, default=str(ROOT))
    parser.add_argument(
        "--mode",
        choices=["check", "sleep"],
        default="check",
        help="check=poll once; sleep=interruptible wait",
    )
    parser.add_argument("--seconds", type=int, default=0)
    parser.add_argument("--cycle-state", type=str, default="BETWEEN_CYCLES")
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(args.root)
    script = (
        "run_continuous_loop.ps1" if args.component == "continuous" else "run_watchdog_loop.ps1"
    )
    if args.mode == "sleep":
        check = (
            CONTINUOUS_WAIT_CHECK_SEC if args.component == "continuous" else WATCHDOG_WAIT_CHECK_SEC
        )
        result = interruptible_sleep_seconds(
            args.seconds,
            component=args.component,
            actual_pid=args.pid,
            script_name=script,
            check_interval_sec=check,
            root=root,
        )
        print(json.dumps({"mode": "sleep", **{k: v for k, v in result.items() if k != "poll"}}))
        return EXIT_ACCEPTED_EXIT_NOW if result.get("shutdown") else EXIT_NO_REQUEST

    poll = poll_shutdown_for_loop(
        component=args.component,
        actual_pid=args.pid,
        script_name=script,
        root=root,
        cycle_state=args.cycle_state,
        safe_boundary=(
            "PRE_CYCLE" if args.cycle_state == "BETWEEN_CYCLES" else "POST_CYCLE_PRE_SLEEP"
        ),
    )
    print(
        json.dumps(
            {
                "mode": "check",
                "action": poll.get("action"),
                "detail_state": (poll.get("detail") or {}).get("state"),
            }
        )
    )
    if poll.get("action") == "EXIT_NOW":
        return EXIT_ACCEPTED_EXIT_NOW
    if poll.get("action") == "CONTINUE_REFUSED":
        return EXIT_REFUSED
    return EXIT_NO_REQUEST


if __name__ == "__main__":
    raise SystemExit(_cli_main())
