"""Nonce-bound dashboard shutdown owned by the dashboard process.

Only the closeout controller writes the request. The dashboard writes the
acknowledgement and exits itself. This module does not kill processes.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REQUEST_NAME = "dashboard_shutdown_request.json"
ACK_NAME = "dashboard_shutdown_ack.json"
REQUESTED_BY = "operator_closeout_control"


def _live(root: Path) -> Path:
    return Path(root) / "data" / "live"


def _request_path(root: Path) -> Path:
    return _live(root) / REQUEST_NAME


def _ack_path(root: Path) -> Path:
    return _live(root) / ACK_NAME


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _write(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def request_dashboard_shutdown(
    root: Path,
    *,
    expected_pid: int,
    port: int,
    session_date: str,
    requested_by: str = REQUESTED_BY,
) -> Dict[str, Any]:
    """Write a one-time shutdown request. Callers other than closeout are refused."""
    if requested_by != REQUESTED_BY:
        return {"ok": False, "reason": "shutdown_requester_refused"}
    doc = {
        "requested_by": REQUESTED_BY,
        "nonce": uuid.uuid4().hex,
        "expected_pid": int(expected_pid),
        "port": int(port),
        "session_date": str(session_date or ""),
        "requested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _write(_request_path(root), doc)
    return {"ok": True, "reason": "shutdown_requested", **doc}


def acknowledge_dashboard_shutdown(
    root: Path,
    *,
    pid: int,
    presented_nonce: str,
) -> Dict[str, Any]:
    """Acknowledge only the nonce currently requested for this PID."""
    request = _read(_request_path(root))
    if not request:
        return {"ok": False, "reason": "shutdown_request_missing"}
    if str(request.get("requested_by") or "") != REQUESTED_BY:
        return {"ok": False, "reason": "shutdown_requester_refused"}
    if str(presented_nonce or "") != str(request.get("nonce") or ""):
        return {"ok": False, "reason": "shutdown_nonce_mismatch"}
    if int(pid) != int(request.get("expected_pid") or -1):
        return {"ok": False, "reason": "shutdown_pid_mismatch"}
    ack = {
        "requested_by": REQUESTED_BY,
        "nonce": str(request.get("nonce")),
        "pid": int(pid),
        "port": int(request.get("port") or 0),
        "acknowledged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _write(_ack_path(root), ack)
    return {"ok": True, "reason": "shutdown_acknowledged", **ack}


def ack_matches_request(root: Path) -> bool:
    request = _read(_request_path(root))
    ack = _read(_ack_path(root))
    if not request or not ack:
        return False
    return (
        str(ack.get("nonce") or "") == str(request.get("nonce") or "")
        and int(ack.get("pid") or -1) == int(request.get("expected_pid") or -2)
        and str(ack.get("requested_by") or "") == REQUESTED_BY
    )


def poll_dashboard_shutdown(
    *,
    root: Path,
    pid: int,
    command: str,
    exit_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Dashboard-owned poll. A matching request is acknowledged, then this process exits."""
    request = _read(_request_path(root))
    if not request:
        return {"ok": False, "reason": "no_request"}
    decision = acknowledge_dashboard_shutdown(
        root, pid=pid, presented_nonce=str(request.get("nonce") or "")
    )
    if decision.get("ok") is not True:
        return decision
    lowered = str(command or "").lower()
    if "pytest" in lowered or "streamlit" not in lowered:
        return {**decision, "exited": False}
    if exit_fn is None:
        return {**decision, "exited": False, "reason": "exit_fn_required"}
    exit_fn(0)
    return {**decision, "exited": True}
