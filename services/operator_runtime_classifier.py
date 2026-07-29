"""
operator_runtime_classifier.py
Phase DO-003 — Per-loop runtime state classification for Start decisions.

Read-only. Never starts/stops processes. Never removes locks.
A lock alone never proves HEALTHY_RUNNING.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
RESULTS = ROOT / "data" / "results"

CONTINUOUS_SCRIPT = "run_continuous_loop.ps1"
WATCHDOG_SCRIPT = "run_watchdog_loop.ps1"
CONTINUOUS_LOCK = LIVE / "continuous_loop.lock"
WATCHDOG_LOCK = LIVE / "watchdog_loop.lock"
CONTINUOUS_INTERVAL_MINUTES = 45.0
WATCHDOG_INTERVAL_MINUTES = 1.0
CONTINUOUS_OVERDUE_MINUTES = max(60.0, CONTINUOUS_INTERVAL_MINUTES * 1.5)
WATCHDOG_OVERDUE_MINUTES = 10.0

LOOP_STATES = (
    "HEALTHY_RUNNING",
    "NOT_RUNNING",
    "STARTING",
    "STALE_LOCK_NO_PROCESS",
    "PROCESS_WITHOUT_VALID_LOCK",
    "PID_MISMATCH",
    "COMMAND_MISMATCH",
    "DUPLICATE_PROCESSES",
    "STALE_OR_OVERDUE",
    "AMBIGUOUS",
    "UNKNOWN",
)

DECISION_RESULTS = (
    "ALREADY_RUNNING",
    "START_ALLOWED",
    "PARTIAL_START_ALLOWED",
    "START_REFUSED_UNSAFE_STATE",
    "START_REFUSED_EXISTING_UNHEALTHY_PROCESS",
    "START_REFUSED_LIVE_SAFETY",
)

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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
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


def _age_minutes(ts: Optional[datetime], now: Optional[datetime] = None) -> Optional[float]:
    if ts is None:
        return None
    now = now or _utc_now()
    return (now - ts).total_seconds() / 60.0


def parse_loop_lock(path: Path) -> Dict[str, Any]:
    import re

    if not path.is_file():
        return {"exists": False, "path": str(path)}
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return {"exists": False, "path": str(path), "corrupt": True}
    info: Dict[str, Any] = {"exists": True, "path": str(path), "raw": text[:200]}
    if not text:
        info["corrupt"] = True
        return info
    m = re.search(r"started_at=([^\r\n]+)", text)
    if m:
        info["started_at"] = m.group(1).strip()
    m_refresh = re.search(r"refreshed_at=([^\r\n]+)", text, re.I)
    if m_refresh:
        info["refreshed_at"] = m_refresh.group(1).strip()
    m_pid = re.search(r"pid=(\d+)", text, re.I)
    if m_pid:
        info["pid"] = int(m_pid.group(1))
    m_source = re.search(r"source=([^\r\n]+)", text, re.I)
    if m_source:
        info["source"] = m_source.group(1).strip()
    m_cycle = re.search(r"cycle=(\d+)", text, re.I)
    if m_cycle:
        info["cycle"] = int(m_cycle.group(1))
    if not info.get("started_at") and not info.get("refreshed_at"):
        info["corrupt"] = True
    return info


@dataclass
class LoopClassification:
    loop: str
    state: str
    expected_script: str
    expected_lock_path: str
    process_count: int = 0
    process_pids: List[int] = field(default_factory=list)
    lock_exists: bool = False
    lock_pid: Optional[int] = None
    lock_refreshed_at: Optional[str] = None
    activity_age_minutes: Optional[float] = None
    overdue_threshold_minutes: float = 0.0
    explanation: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify_loop(
    *,
    loop: str,
    script_name: str,
    lock_path: Path,
    pids: Sequence[int],
    command_lines: Optional[Dict[int, str]] = None,
    overdue_threshold_minutes: float,
    now: Optional[datetime] = None,
    detection_confidence: str = "high",
) -> LoopClassification:
    """Classify a single loop from process + lock evidence."""
    now = now or _utc_now()
    lock = parse_loop_lock(lock_path)
    pids_list = [int(p) for p in pids]
    count = len(pids_list)
    lock_exists = bool(lock.get("exists")) and not lock.get("corrupt")
    lock_pid = lock.get("pid") if isinstance(lock.get("pid"), int) else None
    refreshed = _parse_dt(lock.get("refreshed_at") or lock.get("started_at"))
    age = _age_minutes(refreshed, now)

    base = LoopClassification(
        loop=loop,
        state="UNKNOWN",
        expected_script=script_name,
        expected_lock_path=str(lock_path),
        process_count=count,
        process_pids=pids_list,
        lock_exists=lock_exists or bool(lock.get("exists")),
        lock_pid=lock_pid,
        lock_refreshed_at=refreshed.isoformat() if refreshed else None,
        activity_age_minutes=age,
        overdue_threshold_minutes=overdue_threshold_minutes,
        evidence={
            "lock": {k: v for k, v in lock.items() if k != "raw"},
            "detection_confidence": detection_confidence,
            "command_lines_checked": bool(command_lines),
        },
    )

    if detection_confidence == "low":
        base.state = "UNKNOWN"
        base.explanation = (
            "Process detection confidence is low; missing evidence is not permission to start."
        )
        return base

    # Command-line verification for detected PIDs
    if command_lines and pids_list:
        bad = []
        for pid in pids_list:
            cmd = command_lines.get(pid)
            if cmd is None:
                continue
            if script_name.lower() not in cmd.lower():
                bad.append(pid)
        if bad:
            base.state = "COMMAND_MISMATCH"
            base.explanation = f"Process PID(s) {bad} do not contain expected script {script_name} in command line."
            return base

    if count > 1:
        base.state = "DUPLICATE_PROCESSES"
        base.explanation = (
            f"Duplicate {loop} processes detected (count={count}, pids={pids_list}). Start refused."
        )
        return base

    if count == 0 and not lock.get("exists"):
        base.state = "NOT_RUNNING"
        base.explanation = f"No {loop} process and no lock file. Start may be considered."
        return base

    if count == 0 and lock.get("exists"):
        base.state = "STALE_LOCK_NO_PROCESS"
        base.explanation = (
            f"{loop} lock exists without a verified matching process. "
            "DO-003 will not remove the lock or start a replacement."
        )
        return base

    # count == 1
    pid = pids_list[0]
    if not lock.get("exists") or lock.get("corrupt"):
        base.state = "PROCESS_WITHOUT_VALID_LOCK"
        base.explanation = (
            f"{loop} process PID {pid} is running without a trustworthy lock. "
            "Will not adopt or replace automatically."
        )
        return base

    if lock_pid is not None and lock_pid != pid:
        base.state = "PID_MISMATCH"
        base.explanation = f"{loop} lock PID {lock_pid} does not match verified process PID {pid}."
        return base

    if age is not None and age > overdue_threshold_minutes:
        base.state = "STALE_OR_OVERDUE"
        base.explanation = (
            f"{loop} process and lock agree, but activity is overdue "
            f"({age:.1f}m > {overdue_threshold_minutes:.1f}m). "
            "Will not start a duplicate; Stop/Restart awaits cooperative shutdown."
        )
        return base

    base.state = "HEALTHY_RUNNING"
    base.explanation = f"{loop} is healthy: one matching process PID {pid}, lock agrees, activity within threshold."
    return base


def classify_runtime_loops(
    *,
    root: Optional[Path] = None,
    process_snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Classify continuous and watchdog loops."""
    root = Path(root) if root is not None else ROOT
    live = root / "data" / "live"
    cont_lock = live / "continuous_loop.lock"
    wd_lock = live / "watchdog_loop.lock"

    if process_snapshot is not None:
        snap = process_snapshot()
    elif root == ROOT:
        from services.runtime_process_detector import detect_runtime_loop_processes

        det = detect_runtime_loop_processes()
        snap = {
            "continuous_pids": list(det.get("detected_continuous_process_ids") or []),
            "watchdog_pids": list(det.get("detected_watchdog_process_ids") or []),
            "detection_confidence": det.get("process_detection_confidence") or "high",
            "command_lines": {},
        }
    else:
        marker = live / "fixture_process_state.json"
        snap = {
            "continuous_pids": [],
            "watchdog_pids": [],
            "detection_confidence": "high",
            "command_lines": {},
        }
        if marker.is_file():
            try:
                m = json.loads(marker.read_text(encoding="utf-8"))
                snap["continuous_pids"] = list(m.get("detected_continuous_process_ids") or [])
                snap["watchdog_pids"] = list(m.get("detected_watchdog_process_ids") or [])
                snap["detection_confidence"] = m.get("process_detection_confidence") or "high"
                snap["command_lines"] = m.get("command_lines") or {}
            except Exception:
                snap["detection_confidence"] = "low"

    conf = str(snap.get("detection_confidence") or "high")
    cmds = snap.get("command_lines") or {}
    # normalize string keys from JSON
    cmd_map: Dict[int, str] = {}
    if isinstance(cmds, dict):
        for k, v in cmds.items():
            try:
                cmd_map[int(k)] = str(v)
            except (TypeError, ValueError):
                continue

    continuous = classify_loop(
        loop="continuous",
        script_name=CONTINUOUS_SCRIPT,
        lock_path=cont_lock,
        pids=snap.get("continuous_pids") or [],
        command_lines=cmd_map or None,
        overdue_threshold_minutes=CONTINUOUS_OVERDUE_MINUTES,
        now=now,
        detection_confidence=conf,
    )
    watchdog = classify_loop(
        loop="watchdog",
        script_name=WATCHDOG_SCRIPT,
        lock_path=wd_lock,
        pids=snap.get("watchdog_pids") or [],
        command_lines=cmd_map or None,
        overdue_threshold_minutes=WATCHDOG_OVERDUE_MINUTES,
        now=now,
        detection_confidence=conf,
    )
    return {
        "generated_at": (now or _utc_now()).isoformat(),
        "continuous": continuous.to_dict(),
        "watchdog": watchdog.to_dict(),
        "detection_confidence": conf,
    }


def evaluate_start_decision(
    classifications: Dict[str, Any],
    *,
    live_execution_blocked: bool = True,
) -> Dict[str, Any]:
    """
    Explicit Start decision matrix (backend policy).

    Returns decision, components_to_start, preserved_components, explanation.
    """
    if not live_execution_blocked:
        return {
            "decision": "START_REFUSED_LIVE_SAFETY",
            "components_to_start": [],
            "preserved_components": [],
            "explanation": (
                "Live execution does not appear blocked. Start refused. No override is available."
            ),
            "safe_to_retry": False,
        }

    cont = (classifications.get("continuous") or {}).get("state") or "UNKNOWN"
    wd = (classifications.get("watchdog") or {}).get("state") or "UNKNOWN"

    if cont in UNSAFE_STATES or wd in UNSAFE_STATES:
        return {
            "decision": "START_REFUSED_UNSAFE_STATE",
            "components_to_start": [],
            "preserved_components": [],
            "explanation": (
                f"Unsafe or ambiguous runtime state "
                f"(continuous={cont}, watchdog={wd}). Start nothing; review Diagnostics."
            ),
            "safe_to_retry": False,
            "continuous_state": cont,
            "watchdog_state": wd,
        }

    if cont == "STALE_OR_OVERDUE" or wd == "STALE_OR_OVERDUE":
        return {
            "decision": "START_REFUSED_EXISTING_UNHEALTHY_PROCESS",
            "components_to_start": [],
            "preserved_components": [
                c for c, s in (("continuous", cont), ("watchdog", wd)) if s == "STALE_OR_OVERDUE"
            ],
            "explanation": (
                "A verified process exists but is stale/overdue. "
                "Will not start a duplicate. Stop/Restart awaits cooperative shutdown."
            ),
            "safe_to_retry": False,
            "continuous_state": cont,
            "watchdog_state": wd,
        }

    if cont == "HEALTHY_RUNNING" and wd == "HEALTHY_RUNNING":
        return {
            "decision": "ALREADY_RUNNING",
            "components_to_start": [],
            "preserved_components": ["continuous", "watchdog"],
            "explanation": "TRITON was already running safely. No additional process was started.",
            "safe_to_retry": True,
            "continuous_state": cont,
            "watchdog_state": wd,
        }

    if cont == "NOT_RUNNING" and wd == "NOT_RUNNING":
        return {
            "decision": "START_ALLOWED",
            "components_to_start": ["watchdog", "continuous"],
            "preserved_components": [],
            "explanation": "Both loops are absent with no conflicting evidence. Start is allowed.",
            "safe_to_retry": True,
            "continuous_state": cont,
            "watchdog_state": wd,
        }

    if cont == "HEALTHY_RUNNING" and wd == "NOT_RUNNING":
        return {
            "decision": "PARTIAL_START_ALLOWED",
            "components_to_start": ["watchdog"],
            "preserved_components": ["continuous"],
            "explanation": "Continuous is healthy; only the missing watchdog will be started.",
            "safe_to_retry": True,
            "continuous_state": cont,
            "watchdog_state": wd,
        }

    if wd == "HEALTHY_RUNNING" and cont == "NOT_RUNNING":
        return {
            "decision": "PARTIAL_START_ALLOWED",
            "components_to_start": ["continuous"],
            "preserved_components": ["watchdog"],
            "explanation": "Watchdog is healthy; only the missing continuous loop will be started.",
            "safe_to_retry": True,
            "continuous_state": cont,
            "watchdog_state": wd,
        }

    # STARTING or other unexpected combinations
    return {
        "decision": "START_REFUSED_UNSAFE_STATE",
        "components_to_start": [],
        "preserved_components": [],
        "explanation": (
            f"Unhandled or transitional state (continuous={cont}, watchdog={wd}). Start refused."
        ),
        "safe_to_retry": False,
        "continuous_state": cont,
        "watchdog_state": wd,
    }
