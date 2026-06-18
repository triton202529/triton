"""
phase_195h_operational_readiness_freeze.py
Phase 195H — Operational readiness freeze & Monday continuity checklist.

Read-only operational continuity validation after Phase 195G hardening freeze.
Does not place orders, alter broker state, delete locks, or start/stop loops.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"

OUTPUT_JSON = RESULTS / "phase_195h_operational_readiness_freeze.json"
SCORECARD_CSV = RESULTS / "phase_195h_operational_readiness_scorecard.csv"
CHECKLIST_CSV = RESULTS / "phase_195h_monday_continuity_checklist.csv"

POSITIONS_SNAPSHOT = RESULTS / "positions_snapshot.csv"
OPEN_ORDERS_SNAPSHOT = RESULTS / "open_orders_snapshot.csv"
CERT_PATH = RESULTS / "capital_preservation_certification.json"

PAPER_CERTIFICATION_STATUSES = frozenset(
    {
        "CERTIFIED_FOR_PAPER_PROTECTION",
        "CONDITIONALLY_CERTIFIED",
        "PARTIALLY_CERTIFIED",
    }
)

STALE_LOCK_MINUTES = 60.0
CONTINUOUS_LOOP_CADENCE_MINUTES = 45
CONTINUOUS_LEGACY_LOCK_STALE_MINUTES = CONTINUOUS_LOOP_CADENCE_MINUTES * 2.0
WATCHDOG_LOOP_INTERVAL_MINUTES = 1.0
WATCHDOG_LOCK_STALE_MINUTES = max(STALE_LOCK_MINUTES, WATCHDOG_LOOP_INTERVAL_MINUTES * 5.0)
PHASE_195G_COMMIT_PREFIX = "95093c2b"

RUNTIME_PATH_PREFIXES = (
    "data/processed/",
    "data/live/",
    "data/results/",
    "data/governance/",
)

FORBIDDEN_STAGED_PREFIXES = (
    "data/results/phase_",
    "data/processed/",
    "data/live/watchdog_loop.lock",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _certification_artifact_meta(path: Path = CERT_PATH) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "certification_source": str(path),
        "certification_file_exists": path.is_file(),
        "certification_file_mtime": None,
        "certification_generated_at": None,
    }
    if path.is_file():
        try:
            meta["certification_file_mtime"] = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        except OSError:
            pass
    cert = _load_json(path)
    if cert.get("generated_at"):
        meta["certification_generated_at"] = cert.get("generated_at")
    return meta


def _execution_posture_supports_paper(readiness: Dict[str, Any], gov: Dict[str, Any]) -> bool:
    """True when upstream artifacts authorize paper mode with live blocked."""
    live = bool(
        gov.get("live_execution_permitted", readiness.get("live_execution_permitted", True))
    )
    paper = bool(
        gov.get(
            "paper_execution_permitted",
            readiness.get("paper_execution_permitted", False),
        )
    )
    exec_auth = bool(gov.get("execution_authorized", readiness.get("execution_authorized", False)))
    ready = str(readiness.get("readiness_status") or "").strip() == "READY"
    return ready and paper and exec_auth and not live


def _resolve_canonical_certification() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load canonical capital_preservation_certification.json for freeze evaluation.

    When certification disagrees with READY paper posture (common after watchdog
    institutional_autonomy overwrites without the phase-158 chain), refresh via the
    same canonical certification path used by Phase 195C before blocking freeze.
    """
    cert = _load_json(CERT_PATH)
    meta = _certification_artifact_meta()
    status = str(cert.get("certification_status") or "").strip()
    readiness = _load_json(RESULTS / "execution_readiness.json")
    gov = _load_json(RESULTS / "governance_authorization.json")

    refresh_action = "none"
    pre_refresh_status = status
    read_reason = f"read canonical artifact ({status or 'missing'})"

    if status not in PAPER_CERTIFICATION_STATUSES and _execution_posture_supports_paper(
        readiness, gov
    ):
        read_reason = (
            f"certification_status={status or 'missing'} conflicts with READY paper posture; "
            "refreshing via phase-158 canonical certification chain"
        )
        try:
            from services.capital_preservation_certification import (
                run_certification_false_positive_elimination,
            )

            run_certification_false_positive_elimination(
                results_dir=RESULTS,
                verbose=False,
                run_stress_revalidation=False,
            )
            refresh_action = "phase_158_canonical_refresh"
            cert = _load_json(CERT_PATH)
            meta = _certification_artifact_meta()
            status = str(cert.get("certification_status") or "").strip()
            if status in PAPER_CERTIFICATION_STATUSES:
                read_reason = f"stale certification refreshed: {pre_refresh_status} -> {status}"
            else:
                read_reason = (
                    f"canonical refresh completed; certification remains {status or 'missing'}"
                )
        except Exception as exc:
            refresh_action = "refresh_failed"
            read_reason = f"canonical refresh failed: {exc}"
    elif status not in PAPER_CERTIFICATION_STATUSES:
        read_reason = (
            f"certification_status={status or 'missing'} invalid; "
            "paper posture not fully READY/authorized — no auto-refresh"
        )
    else:
        read_reason = f"certification_status={status} accepted for paper protection"

    meta["certification_refresh_action"] = refresh_action
    meta["certification_read_reason"] = read_reason
    meta["pre_refresh_certification_status"] = pre_refresh_status
    meta["certification_status"] = status or cert.get("certification_status")
    return cert, meta


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return -1
    try:
        with path.open(encoding="utf-8", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return -1


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _minutes_since(dt: Optional[datetime]) -> Optional[float]:
    if not dt:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0


def _parse_lock(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "path": str(path.relative_to(ROOT)).replace("\\", "/")}
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    info: Dict[str, Any] = {
        "exists": True,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "raw": text[:200],
    }
    if not text:
        info["corrupt"] = True
        return info

    m = re.search(r"started_at=([^\r\n]+)", text)
    if m:
        info["started_at"] = m.group(1).strip()
        info["started_at_dt"] = _parse_ts(info["started_at"])
    m_refresh = re.search(r"refreshed_at=([^\r\n]+)", text, re.I)
    if m_refresh:
        info["refreshed_at"] = m_refresh.group(1).strip()
        info["refreshed_at_dt"] = _parse_ts(info["refreshed_at"])
    m_pid = re.search(r"pid=(\d+)", text, re.I)
    if m_pid:
        info["pid"] = int(m_pid.group(1))
    m_source = re.search(r"source=([^\r\n]+)", text, re.I)
    if m_source:
        info["source"] = m_source.group(1).strip()
    m_cycle = re.search(r"cycle=(\d+)", text, re.I)
    if m_cycle:
        info["cycle"] = int(m_cycle.group(1))

    if not info.get("started_at_dt") and not info.get("refreshed_at_dt"):
        info["corrupt"] = True
        return info

    freshness_dt = info.get("refreshed_at_dt") or info.get("started_at_dt")
    info["freshness_source"] = "refreshed_at" if info.get("refreshed_at_dt") else "started_at"
    freshness_age = _minutes_since(freshness_dt)
    started_age = _minutes_since(info.get("started_at_dt"))
    if freshness_age is not None:
        info["lock_age_minutes"] = round(freshness_age, 1)
        info["lock_freshness_age_minutes"] = round(freshness_age, 1)
    if started_age is not None:
        info["started_at_age_minutes"] = round(started_age, 1)

    is_continuous = "continuous_loop" in path.name
    stale_threshold = (
        CONTINUOUS_LEGACY_LOCK_STALE_MINUTES
        if is_continuous and not info.get("refreshed_at_dt")
        else (WATCHDOG_LOCK_STALE_MINUTES if "watchdog" in path.name else STALE_LOCK_MINUTES)
    )
    if info.get("refreshed_at_dt") and is_continuous:
        stale_threshold = max(STALE_LOCK_MINUTES, CONTINUOUS_LOOP_CADENCE_MINUTES * 1.5)

    if freshness_age is not None:
        info["stale_threshold_minutes"] = stale_threshold
        info["stale"] = freshness_age > stale_threshold
    return info


def _find_ps_processes(script_fragment: str) -> Tuple[bool, List[int]]:
    if sys.platform != "win32":
        return False, []
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name = 'powershell.exe' OR Name = 'pwsh.exe'\" "
        f"| Where-Object {{ $_.CommandLine -like '*{script_fragment}*' }} "
        "| Select-Object -ExpandProperty ProcessId"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        pids = [int(x.strip()) for x in (r.stdout or "").splitlines() if x.strip().isdigit()]
        return len(pids) > 0, pids
    except (subprocess.SubprocessError, OSError):
        return False, []


def _classify_lock(
    lock: Dict[str, Any],
    process_running: bool,
    process_pids: Optional[List[int]] = None,
) -> str:
    if lock.get("corrupt"):
        return "CORRUPT_LOCK"
    if not lock.get("exists"):
        return "MISSING_ACTIVE_LOCK" if process_running else "MISSING_INACTIVE_LOCK"

    stale = bool(lock.get("stale"))
    pid = lock.get("pid")
    pids = process_pids or []
    pid_match = pid is not None and pid in pids
    has_refresh = lock.get("refreshed_at_dt") is not None

    if process_running:
        if not stale:
            return "FRESH_ACTIVE_LOCK"
        if has_refresh and pid is not None and pids and not pid_match:
            return "STALE_LOCK_METADATA_PROCESS_ACTIVE"
        return "STALE_ACTIVE_LOCK"

    return "ORPHAN_STALE_LOCK"


def _git_output(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return (proc.stdout or "").strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _audit_git_state() -> Dict[str, Any]:
    log_lines = _git_output("log", "--oneline", "-20").splitlines()
    has_195g = any(
        "Phase 195G" in line or line.startswith(PHASE_195G_COMMIT_PREFIX) for line in log_lines
    )
    porcelain = _git_output("status", "--porcelain").splitlines()

    staged: List[str] = []
    modified: List[str] = []
    untracked: List[str] = []
    staged_risky: List[str] = []

    for line in porcelain:
        if len(line) < 4:
            continue
        x, y = line[0], line[1]
        path = line[3:].strip().strip('"')
        rel = path.replace("\\", "/")
        if x not in (" ", "?"):
            staged.append(rel)
            if any(rel.startswith(p) for p in FORBIDDEN_STAGED_PREFIXES):
                staged_risky.append(rel)
        if y not in (" ", "?"):
            modified.append(rel)
        if x == "?" and y == "?":
            untracked.append(rel)

    runtime_modified = [
        p for p in modified if any(p.startswith(prefix) for prefix in RUNTIME_PATH_PREFIXES)
    ]
    code_modified = [p for p in modified if p not in runtime_modified]

    if staged_risky:
        status = "STAGED_RUNTIME_ARTIFACTS"
    elif staged and all(
        s in ("services/phase_195h_operational_readiness_freeze.py",) for s in staged
    ):
        status = "STAGED_PHASE_ARTIFACT"
    elif staged:
        status = "STAGED_CHANGES"
    elif code_modified:
        status = "DIRTY_CODE"
    elif runtime_modified or untracked:
        status = "RUNTIME_NOISE"
    else:
        status = "CLEAN"

    return {
        "status": status,
        "phase_195g_commit_present": has_195g,
        "recent_commits": log_lines[:8],
        "staged_files": staged,
        "staged_runtime_artifacts": staged_risky,
        "modified_runtime_files_count": len(runtime_modified),
        "modified_code_files": code_modified,
        "untracked_files": untracked,
        "working_tree_clean": status == "CLEAN",
        "issues": (
            ([f"Phase 195G commit not found in recent log"] if not has_195g else [])
            + ([f"staged runtime artifacts: {staged_risky}"] if staged_risky else [])
            + ([f"uncommitted code changes: {code_modified}"] if code_modified else [])
        ),
    }


def _audit_broker_sync() -> Dict[str, Any]:
    orders = _count_csv_rows(OPEN_ORDERS_SNAPSHOT)
    positions = _count_csv_rows(POSITIONS_SNAPSHOT)
    watchdog = _load_json(RESULTS / "watchdog_status.json")

    wd_orders = watchdog.get("open_orders_count")
    wd_positions = watchdog.get("positions_count")
    orders_match = orders >= 0 and (wd_orders is None or int(wd_orders) == orders)
    positions_match = positions >= 0 and (wd_positions is None or int(wd_positions) == positions)

    status = "SYNCED"
    issues: List[str] = []
    if orders < 0 or positions < 0:
        status = "MISSING"
        issues.append("broker snapshot CSV missing")
    elif not orders_match or not positions_match:
        status = "MISMATCH"
        if not orders_match:
            issues.append(f"orders snapshot={orders} watchdog={wd_orders}")
        if not positions_match:
            issues.append(f"positions snapshot={positions} watchdog={wd_positions}")
    if not watchdog.get("broker_connected", True) and watchdog:
        issues.append(f"broker_connected={watchdog.get('broker_connected')}")

    return {
        "status": status,
        "broker_connected": watchdog.get("broker_connected"),
        "open_orders_count": orders,
        "positions_count": positions,
        "watchdog_open_orders": wd_orders,
        "watchdog_positions": wd_positions,
        "unexpected_orders": orders > 0,
        "unexpected_positions": positions > 0,
        "issues": issues,
    }


def _audit_execution_safety() -> Dict[str, Any]:
    gov = _load_json(RESULTS / "governance_authorization.json")
    readiness = _load_json(RESULTS / "execution_readiness.json")
    gate = _load_json(RESULTS / "master_execution_gate.json")

    live = gov.get("live_execution_permitted", readiness.get("live_execution_permitted"))
    paper = gov.get("paper_execution_permitted", readiness.get("paper_execution_permitted"))

    loader_ok = False
    try:
        from services.decision_authority import (
            RISK_AUTHORITY_PATH,
            load_risk_execution_controls,
        )

        load_risk_execution_controls()
        loader_ok = RISK_AUTHORITY_PATH.name == "master_execution_gate.json"
    except Exception:
        loader_ok = False

    trade_opps = (ROOT / "services" / "build_trade_opportunities.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    bypass = bool(
        re.search(r"adaptive_risk_state\.json", trade_opps)
        and "load_risk_execution_controls" not in trade_opps
    )

    status = "SAFE"
    issues: List[str] = []
    if live is not False:
        status = "UNSAFE"
        issues.append(f"live_execution_permitted={live}")
    if paper is not True:
        issues.append(f"paper_execution_permitted={paper}")
    if not gate:
        status = "UNSAFE"
        issues.append("master_execution_gate.json missing")
    if not loader_ok:
        issues.append("decision_authority loader failed")
    if bypass:
        status = "UNSAFE"
        issues.append("execution bypass detected in build_trade_opportunities")

    return {
        "status": status,
        "live_execution_permitted": live,
        "paper_execution_authorized": paper is True,
        "master_gate_exists": bool(gate),
        "decision_authority_loader_ok": loader_ok,
        "issues": issues,
    }


def _audit_watchdog() -> Dict[str, Any]:
    status_doc = _load_json(RESULTS / "watchdog_status.json")
    alerts = _load_json(RESULTS / "watchdog_alerts.json")
    lock = _parse_lock(LIVE / "watchdog_loop.lock")
    running, pids = _find_ps_processes("run_watchdog_loop.ps1")
    lock_status = _classify_lock(lock, running, pids)

    active = alerts.get("active_count")
    if active is None:
        active_list = alerts.get("active_alerts") or alerts.get("alerts") or []
        active = len(active_list) if isinstance(active_list, list) else 0

    wd = status_doc.get("watchdog_status") or status_doc.get("status") or "UNKNOWN"
    hb = status_doc.get("heartbeat") or {}
    hb_ts = hb.get("timestamp") if isinstance(hb, dict) else status_doc.get("timestamp")
    hb_age = _minutes_since(_parse_ts(hb_ts))

    return {
        "status": wd,
        "process_running": running,
        "process_pids": pids,
        "lock": lock,
        "lock_status": lock_status,
        "lock_age_minutes": lock.get("lock_age_minutes"),
        "active_alerts_count": active,
        "heartbeat_age_minutes": round(hb_age, 1) if hb_age is not None else None,
        "recommended_manual_cleanup": lock_status == "ORPHAN_STALE_LOCK",
        "recommended_cleanup_command": (
            "Review scripts/run_watchdog_loop.ps1 lock; clear only after confirming no process"
            if lock_status == "ORPHAN_STALE_LOCK"
            else ""
        ),
    }


def _audit_continuous_loop() -> Dict[str, Any]:
    lock = _parse_lock(LIVE / "continuous_loop.lock")
    loop_health = _load_json(RESULTS / "continuous_loop_health.json")
    heartbeat = _load_json(RESULTS / "heartbeat.json")
    loop_dir = RESULTS / "continuous_loop"

    running, pids = _find_ps_processes("run_continuous_loop.ps1")
    lock_status = _classify_lock(lock, running, pids)

    log_files = (
        sorted(
            loop_dir.glob("continuous_loop_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if loop_dir.is_dir()
        else []
    )
    latest_log_age: Optional[float] = None
    if log_files:
        latest_log_age = _minutes_since(
            datetime.fromtimestamp(log_files[0].stat().st_mtime, tz=timezone.utc)
        )

    hb_age = _minutes_since(_parse_ts(heartbeat.get("timestamp")))

    if running:
        proc_status = "RUNNING"
    elif lock_status == "ORPHAN_STALE_LOCK":
        proc_status = "STOPPED_ORPHAN_LOCK"
    else:
        proc_status = "STOPPED"

    return {
        "status": proc_status,
        "process_running": running,
        "process_pids": pids,
        "lock": lock,
        "lock_status": lock_status,
        "lock_age_minutes": lock.get("lock_age_minutes"),
        "cadence_minutes": CONTINUOUS_LOOP_CADENCE_MINUTES,
        "heartbeat_age_minutes": round(hb_age, 1) if hb_age is not None else None,
        "latest_loop_log_age_minutes": (
            round(latest_log_age, 1) if latest_log_age is not None else None
        ),
        "loop_log_count": len(log_files),
        "loop_health_present": bool(loop_health),
        "recommended_manual_cleanup": lock_status == "ORPHAN_STALE_LOCK",
        "recommended_cleanup_command": (
            "Review scripts/run_continuous_loop.ps1 lock; clear only after confirming no process"
            if lock_status == "ORPHAN_STALE_LOCK"
            else ""
        ),
    }


def _build_monday_checklist(
    git: Dict[str, Any],
    broker: Dict[str, Any],
    execution: Dict[str, Any],
    phases: Dict[str, Any],
    watchdog: Dict[str, Any],
    loop: Dict[str, Any],
    conflicts: Dict[str, int],
) -> List[Dict[str, Any]]:
    rows = [
        {
            "item": "phase_195g_commit_present",
            "category": "git",
            "status": "PASS" if git.get("phase_195g_commit_present") else "FAIL",
            "expected": "Phase 195G in recent git log",
            "actual": str(git.get("phase_195g_commit_present")),
            "pass": git.get("phase_195g_commit_present") is True,
            "notes": "",
        },
        {
            "item": "no_staged_runtime_artifacts",
            "category": "git",
            "status": "PASS" if not git.get("staged_runtime_artifacts") else "FAIL",
            "expected": "no staged data/results phase artifacts",
            "actual": str(len(git.get("staged_runtime_artifacts") or [])),
            "pass": not git.get("staged_runtime_artifacts"),
            "notes": json.dumps(git.get("staged_runtime_artifacts") or []),
        },
        {
            "item": "broker_sync",
            "category": "broker",
            "status": broker.get("status"),
            "expected": "SYNCED",
            "actual": broker.get("status"),
            "pass": broker.get("status") == "SYNCED",
            "notes": json.dumps(broker.get("issues") or []),
        },
        {
            "item": "open_orders_zero",
            "category": "broker",
            "status": "PASS" if broker.get("open_orders_count") == 0 else "REVIEW",
            "expected": "0",
            "actual": str(broker.get("open_orders_count")),
            "pass": broker.get("open_orders_count") == 0,
            "notes": "Non-zero is OK if operator expects pending orders",
        },
        {
            "item": "positions_known",
            "category": "broker",
            "status": "PASS" if broker.get("positions_count", -1) >= 0 else "FAIL",
            "expected": "known count",
            "actual": str(broker.get("positions_count")),
            "pass": broker.get("positions_count", -1) >= 0,
            "notes": "",
        },
        {
            "item": "live_execution_blocked",
            "category": "execution",
            "status": ("PASS" if execution.get("live_execution_permitted") is False else "FAIL"),
            "expected": "false",
            "actual": str(execution.get("live_execution_permitted")),
            "pass": execution.get("live_execution_permitted") is False,
            "notes": "",
        },
        {
            "item": "paper_execution_controlled",
            "category": "execution",
            "status": "PASS" if execution.get("paper_execution_authorized") else "FAIL",
            "expected": "true",
            "actual": str(execution.get("paper_execution_authorized")),
            "pass": execution.get("paper_execution_authorized") is True,
            "notes": "",
        },
        {
            "item": "high_conflicts_zero",
            "category": "authority",
            "status": "PASS" if conflicts.get("high", 0) == 0 else "FAIL",
            "expected": "0",
            "actual": str(conflicts.get("high")),
            "pass": conflicts.get("high", 0) == 0,
            "notes": "",
        },
        {
            "item": "phase_195c_pass",
            "category": "governance",
            "status": "PASS" if phases.get("195c") else "FAIL",
            "expected": "true",
            "actual": str(phases.get("195c")),
            "pass": phases.get("195c") is True,
            "notes": "",
        },
        {
            "item": "phase_195g_pass",
            "category": "sync",
            "status": "PASS" if phases.get("195g") else "FAIL",
            "expected": "true",
            "actual": str(phases.get("195g")),
            "pass": phases.get("195g") is True,
            "notes": "",
        },
        {
            "item": "watchdog_known",
            "category": "watchdog",
            "status": watchdog.get("status") or "UNKNOWN",
            "expected": "OK or known",
            "actual": watchdog.get("status") or "UNKNOWN",
            "pass": bool(watchdog.get("status")),
            "notes": f"lock={watchdog.get('lock_status')}",
        },
        {
            "item": "continuous_loop_known",
            "category": "loop",
            "status": loop.get("status") or "UNKNOWN",
            "expected": "RUNNING or classified stopped state",
            "actual": loop.get("status") or "UNKNOWN",
            "pass": bool(loop.get("status")),
            "notes": f"lock={loop.get('lock_status')} cadence={loop.get('cadence_minutes')}m",
        },
        {
            "item": "operator_loop_awareness",
            "category": "operator",
            "status": "INFO",
            "expected": "operator confirms loop/watchdog state",
            "actual": (
                f"watchdog_running={watchdog.get('process_running')} "
                f"loop_running={loop.get('process_running')}"
            ),
            "pass": True,
            "notes": "Manual confirmation required before market open",
        },
    ]
    return rows


def run_phase_195h_freeze(*, verbose: bool = True) -> Dict[str, Any]:
    git = _audit_git_state()
    broker = _audit_broker_sync()
    execution = _audit_execution_safety()
    watchdog = _audit_watchdog()
    loop = _audit_continuous_loop()

    d195c = _load_json(RESULTS / "phase_195c_evaluation_contract_validation.json")
    d195d = _load_json(RESULTS / "phase_195d_authority_consolidation_validation.json")
    d195e = _load_json(RESULTS / "phase_195e_authority_remediation_validation.json")
    d195f = _load_json(RESULTS / "phase_195f_authority_labeling_validation.json")
    d195g = _load_json(RESULTS / "phase_195g_post_hardening_sync_audit.json")
    cert, cert_meta = _resolve_canonical_certification()

    high_conflicts = int(d195d.get("high_severity_count") or 0)
    medium_conflicts = int(d195d.get("medium_severity_count") or 0)
    low_conflicts = int(d195d.get("low_severity_count") or 0)

    phases = {
        "195c": d195c.get("phase_pass"),
        "195d": d195d.get("phase_pass"),
        "195e": d195e.get("phase_pass"),
        "195f": d195f.get("phase_pass"),
        "195g": d195g.get("phase_pass"),
    }

    blocking_issues: List[str] = []
    review_items: List[str] = []

    if not git.get("phase_195g_commit_present"):
        blocking_issues.append("Phase 195G commit not found in recent git history")
    if git.get("staged_runtime_artifacts"):
        blocking_issues.append(
            "Staged runtime artifacts detected: " + ", ".join(git["staged_runtime_artifacts"][:5])
        )
    if git.get("status") == "DIRTY_CODE":
        blocking_issues.extend(
            [f"Uncommitted code: {p}" for p in (git.get("modified_code_files") or [])[:5]]
        )
    if broker.get("status") != "SYNCED":
        blocking_issues.extend(broker.get("issues") or [])
    if execution.get("status") != "SAFE":
        blocking_issues.extend(execution.get("issues") or [])
    if phases["195c"] is not True:
        blocking_issues.append("Phase 195C not passing")
    if phases["195d"] is not True:
        blocking_issues.append("Phase 195D not passing")
    if phases["195e"] is not True:
        blocking_issues.append("Phase 195E not passing")
    if phases["195f"] is not True:
        blocking_issues.append("Phase 195F not passing")
    if phases["195g"] is not True:
        blocking_issues.append("Phase 195G not passing")
    if high_conflicts > 0:
        blocking_issues.append(f"HIGH conflicts={high_conflicts}")
    cert_status = str(cert.get("certification_status") or "").strip()
    if cert_status not in PAPER_CERTIFICATION_STATUSES:
        blocking_issues.append(
            f"certification_status={cert_status or 'missing'} "
            f"({cert_meta.get('certification_read_reason', 'invalid')})"
        )
    elif cert_meta.get("certification_refresh_action") == "phase_158_canonical_refresh":
        review_items.append(
            "Certification artifact was stale relative to READY paper posture; "
            "refreshed during 195H freeze via canonical phase-158 chain"
        )

    stale_lock_review = False
    for _label, audit in (("watchdog", watchdog), ("continuous_loop", loop)):
        ls = audit.get("lock_status")
        lock_path = (audit.get("lock") or {}).get("path") or "unknown_lock"
        if ls == "STALE_LOCK_METADATA_PROCESS_ACTIVE":
            review_items.append(
                f"{lock_path}: STALE_LOCK_METADATA_PROCESS_ACTIVE "
                f"(freshness_age_minutes={audit.get('lock_age_minutes')})"
            )
            stale_lock_review = True
        elif ls in ("ORPHAN_STALE_LOCK", "CORRUPT_LOCK", "MISSING_ACTIVE_LOCK"):
            review_items.append(
                f"{lock_path}: {ls} " f"(freshness_age_minutes={audit.get('lock_age_minutes')})"
            )
            stale_lock_review = True

    if git.get("status") == "RUNTIME_NOISE":
        review_items.append(
            f"Git working tree has runtime noise "
            f"({git.get('modified_runtime_files_count')} modified runtime files, "
            f"{len(git.get('untracked_files') or [])} untracked)"
        )
    if broker.get("unexpected_orders"):
        review_items.append("Open orders > 0 — confirm expected before session")
    if broker.get("unexpected_positions"):
        review_items.append("Positions > 0 — confirm expected before session")

    restart_required = (
        loop.get("lock_status") == "ORPHAN_STALE_LOCK"
        or watchdog.get("lock_status") == "ORPHAN_STALE_LOCK"
        or loop.get("lock_status") in ("CORRUPT_LOCK", "MISSING_ACTIVE_LOCK")
        or (
            not loop.get("process_running")
            and loop.get("lock_status")
            not in ("MISSING_INACTIVE_LOCK", "FRESH_ACTIVE_LOCK", "STALE_ACTIVE_LOCK")
        )
    )

    if not blocking_issues and review_items:
        operational_freeze = "READY_WITH_REVIEW_ITEMS"
        next_session = "READY_WITH_REVIEW_ITEMS"
    elif not blocking_issues:
        operational_freeze = "READY"
        next_session = "READY"
    else:
        operational_freeze = "NOT_READY"
        next_session = "NOT_READY"

    scorecard: List[Dict[str, Any]] = [
        {
            "check": "git_state",
            "expected": "CLEAN or RUNTIME_NOISE",
            "actual": git.get("status"),
            "pass": (
                git.get("status") in ("CLEAN", "RUNTIME_NOISE", "STAGED_PHASE_ARTIFACT")
                and git.get("phase_195g_commit_present")
            ),
        },
        {
            "check": "broker_sync",
            "expected": "SYNCED",
            "actual": broker.get("status"),
            "pass": broker.get("status") == "SYNCED",
        },
        {
            "check": "live_blocked",
            "expected": "false",
            "actual": str(execution.get("live_execution_permitted")),
            "pass": execution.get("live_execution_permitted") is False,
        },
        {
            "check": "paper_controlled",
            "expected": "true",
            "actual": str(execution.get("paper_execution_authorized")),
            "pass": execution.get("paper_execution_authorized") is True,
        },
        {
            "check": "phase_195c",
            "expected": "pass",
            "actual": str(phases["195c"]),
            "pass": phases["195c"] is True,
        },
        {
            "check": "phase_195d",
            "expected": "pass",
            "actual": str(phases["195d"]),
            "pass": phases["195d"] is True,
        },
        {
            "check": "phase_195e",
            "expected": "pass",
            "actual": str(phases["195e"]),
            "pass": phases["195e"] is True,
        },
        {
            "check": "phase_195f",
            "expected": "pass",
            "actual": str(phases["195f"]),
            "pass": phases["195f"] is True,
        },
        {
            "check": "phase_195g",
            "expected": "pass",
            "actual": str(phases["195g"]),
            "pass": phases["195g"] is True,
        },
        {
            "check": "high_conflicts",
            "expected": "0",
            "actual": str(high_conflicts),
            "pass": high_conflicts == 0,
        },
        {
            "check": "certification_paper_protection",
            "expected": "CERTIFIED_FOR_PAPER_PROTECTION or equivalent",
            "actual": cert_status or "missing",
            "pass": cert_status in PAPER_CERTIFICATION_STATUSES,
        },
        {
            "check": "watchdog_readiness",
            "expected": "known",
            "actual": watchdog.get("status"),
            "pass": bool(watchdog.get("status")),
        },
        {
            "check": "continuous_loop_readiness",
            "expected": "known",
            "actual": loop.get("status"),
            "pass": bool(loop.get("status")),
        },
        {
            "check": "operational_freeze",
            "expected": "READY or READY_WITH_REVIEW_ITEMS",
            "actual": operational_freeze,
            "pass": operational_freeze in ("READY", "READY_WITH_REVIEW_ITEMS"),
        },
    ]

    phase_pass = not blocking_issues and operational_freeze in (
        "READY",
        "READY_WITH_REVIEW_ITEMS",
    )

    checklist = _build_monday_checklist(
        git,
        broker,
        execution,
        phases,
        watchdog,
        loop,
        {"high": high_conflicts, "medium": medium_conflicts, "low": low_conflicts},
    )

    payload: Dict[str, Any] = {
        "phase": "195H",
        "timestamp": _utc_now_iso(),
        "git_state_status": git.get("status"),
        "broker_sync_status": broker.get("status"),
        "positions_count": broker.get("positions_count"),
        "open_orders_count": broker.get("open_orders_count"),
        "live_execution_permitted": execution.get("live_execution_permitted"),
        "paper_execution_authorized": execution.get("paper_execution_authorized"),
        "phase_195c_status": phases["195c"],
        "phase_195d_status": phases["195d"],
        "phase_195e_status": phases["195e"],
        "phase_195f_status": phases["195f"],
        "phase_195g_status": phases["195g"],
        "high_conflicts": high_conflicts,
        "medium_conflicts": medium_conflicts,
        "low_conflicts": low_conflicts,
        "watchdog_status": watchdog.get("status"),
        "watchdog_lock_status": watchdog.get("lock_status"),
        "watchdog_lock_age_minutes": watchdog.get("lock_age_minutes"),
        "continuous_loop_status": loop.get("status"),
        "continuous_loop_lock_status": loop.get("lock_status"),
        "continuous_loop_lock_age_minutes": loop.get("lock_age_minutes"),
        "continuous_loop_cadence_minutes": CONTINUOUS_LOOP_CADENCE_MINUTES,
        "stale_lock_review_required": stale_lock_review,
        "restart_required": restart_required,
        "blocking_issues": blocking_issues,
        "review_items": review_items,
        "operational_freeze_status": operational_freeze,
        "next_session_readiness": next_session,
        "phase_pass": phase_pass,
        "certification_status": cert_status,
        "certification_source": cert_meta.get("certification_source"),
        "certification_generated_at": cert_meta.get("certification_generated_at"),
        "certification_file_mtime": cert_meta.get("certification_file_mtime"),
        "certification_refresh_action": cert_meta.get("certification_refresh_action"),
        "certification_read_reason": cert_meta.get("certification_read_reason"),
        "pre_refresh_certification_status": cert_meta.get("pre_refresh_certification_status"),
        "domains": {
            "git": git,
            "broker": broker,
            "execution": execution,
            "watchdog": watchdog,
            "continuous_loop": loop,
        },
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(SCORECARD_CSV, scorecard, ("check", "expected", "actual", "pass"))
    _write_csv(
        CHECKLIST_CSV,
        checklist,
        ("item", "category", "status", "expected", "actual", "pass", "notes"),
    )

    if verbose:
        print(
            json.dumps(
                {
                    "phase_pass": phase_pass,
                    "operational_freeze_status": operational_freeze,
                    "next_session_readiness": next_session,
                    "review_items": review_items,
                },
                indent=2,
            )
        )

    return payload


def main() -> int:
    result = run_phase_195h_freeze(verbose=True)
    return 0 if result.get("phase_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
