"""
operator_console_status.py
Phase DO-002 — Shared read-only operator-console status layer.

Uses allowlisted fixed paths and existing process/lock classifiers.
Never starts/stops processes, never writes locks/confirms, never submits orders.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"

CONTINUOUS_LOCK = LIVE / "continuous_loop.lock"
WATCHDOG_LOCK = LIVE / "watchdog_loop.lock"
PAPER_ARM_CONFIRM = LIVE / "paper_arm_confirm.json"
WATCHDOG_STATUS = RESULTS / "watchdog_status.json"
WATCHDOG_ALERTS = RESULTS / "watchdog_alerts.json"
EXECUTION_READINESS = RESULTS / "execution_readiness.json"
GOVERNANCE_AUTH = RESULTS / "governance_authorization.json"
HEARTBEAT = RESULTS / "heartbeat.json"
RUNTIME_AUTOSTART_931 = RESULTS / "runtime_autostart_931.json"
RUNTIME_SUPERVISOR = RESULTS / "runtime_supervisor_autostart.json"
FRESHNESS_AUDIT = RESULTS / "data_freshness_refresh_audit.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
PORTFOLIO_HISTORY = RESULTS / "portfolio_history.csv"
RISK_OVERLAY = RESULTS / "performance_risk_overlay.csv"
CAPITAL_PRESERVATION = RESULTS / "capital_preservation_governor.json"

OVERALL_STATUSES = ("RUNNING", "STOPPED", "DEGRADED", "STALE", "UNKNOWN")
PAPER_ARM_STATUSES = ("ARMED", "EXPIRED", "MISSING", "INVALID", "UNKNOWN")

CONTINUOUS_CYCLE_OVERDUE_MINUTES = 90.0
WATCHDOG_OVERDUE_MINUTES = 10.0
DATA_FRESH_MINUTES = 90.0
BROKER_FRESH_MINUTES = 120.0


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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {"__malformed__": True, "__path__": str(path)}


def _csv_row_count(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)
    except Exception:
        return None


def _parse_lock_file(path: Path) -> Dict[str, Any]:
    """Parse loop lock metadata without requiring path to be under repo ROOT."""
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


def _file_mtime(path: Path) -> Optional[datetime]:
    if not path.is_file():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def map_paper_arm_display(status: str) -> str:
    """Map paper_arm_status codes to operator display labels."""
    s = str(status or "").upper()
    if s == "VALID":
        return "ARMED"
    if s in PAPER_ARM_STATUSES:
        return s
    if s in ("EXPIRED", "MISSING", "INVALID"):
        return s
    return "UNKNOWN"


def classify_overall_status(
    *,
    continuous_running: bool,
    watchdog_running: bool,
    continuous_lock_healthy: bool,
    watchdog_lock_healthy: bool,
    continuous_lock_class: str,
    watchdog_lock_class: str,
    continuous_cycle_age_minutes: Optional[float],
    detection_confidence: str = "high",
) -> str:
    """
    Classify TRITON overall status.

    RUNNING requires process evidence for both loops plus healthy locks —
    a lock file alone is never sufficient.
    """
    orphan_or_mismatch = {
        "STALE_ORPHAN_LOCK",
        "CORRUPT_LOCK",
        "MISSING_ACTIVE_LOCK",
        "NO_LOCK_RUNNING",
        "STALE_ACTIVE_LOCK",
    }
    if detection_confidence == "low" and not continuous_running and not watchdog_running:
        # Cannot prove stopped vs unknown when process scan unavailable
        if continuous_lock_class not in ("NO_PROCESS_NO_LOCK", "") or watchdog_lock_class not in (
            "NO_PROCESS_NO_LOCK",
            "",
        ):
            return "UNKNOWN"

    both_running = continuous_running and watchdog_running
    both_healthy = continuous_lock_healthy and watchdog_lock_healthy

    if both_running and both_healthy:
        if (
            continuous_cycle_age_minutes is not None
            and continuous_cycle_age_minutes > CONTINUOUS_CYCLE_OVERDUE_MINUTES
        ):
            return "STALE"
        return "RUNNING"

    if not continuous_running and not watchdog_running:
        if continuous_lock_class in orphan_or_mismatch or watchdog_lock_class in orphan_or_mismatch:
            return "DEGRADED"
        if (
            continuous_lock_class == "NO_PROCESS_NO_LOCK"
            and watchdog_lock_class == "NO_PROCESS_NO_LOCK"
        ):
            return "STOPPED"
        if continuous_lock_class or watchdog_lock_class:
            return "STOPPED"
        return "UNKNOWN"

    # One loop running, or running without healthy locks
    if continuous_lock_class in orphan_or_mismatch or watchdog_lock_class in orphan_or_mismatch:
        return "DEGRADED"
    if both_running and not both_healthy:
        return "DEGRADED"
    if continuous_running != watchdog_running:
        return "DEGRADED"
    return "DEGRADED"


@dataclass
class AttentionItem:
    severity: str
    title: str
    explanation: str
    evidence_source: str
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorStatusSnapshot:
    generated_at: str
    overall_status: str
    continuous: Dict[str, Any] = field(default_factory=dict)
    watchdog: Dict[str, Any] = field(default_factory=dict)
    paper_arm: Dict[str, Any] = field(default_factory=dict)
    live_execution: Dict[str, Any] = field(default_factory=dict)
    broker: Dict[str, Any] = field(default_factory=dict)
    market: Dict[str, Any] = field(default_factory=dict)
    cycle: Dict[str, Any] = field(default_factory=dict)
    orders_positions: Dict[str, Any] = field(default_factory=dict)
    alerts: Dict[str, Any] = field(default_factory=dict)
    data_freshness: Dict[str, Any] = field(default_factory=dict)
    governance: Dict[str, Any] = field(default_factory=dict)
    attention: List[Dict[str, Any]] = field(default_factory=list)
    evidence_paths: Dict[str, str] = field(default_factory=dict)
    runtime_controls_exposed: bool = False
    live_execution_control_exposed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_attention(snapshot_parts: Dict[str, Any]) -> List[AttentionItem]:
    items: List[AttentionItem] = []
    paper = snapshot_parts["paper_arm"]
    cont = snapshot_parts["continuous"]
    wd = snapshot_parts["watchdog"]
    live = snapshot_parts["live_execution"]
    broker = snapshot_parts["broker"]
    freshness = snapshot_parts["data_freshness"]
    alerts = snapshot_parts["alerts"]
    gov = snapshot_parts["governance"]
    cycle = snapshot_parts["cycle"]

    if paper.get("display_status") == "EXPIRED":
        items.append(
            AttentionItem(
                "HIGH",
                "Paper arm expired",
                "The paper-arm confirmation latch is expired. Paper execution may be blocked until re-armed via the approved CLI.",
                str(PAPER_ARM_CONFIRM),
                "Review paper-arm status with the operator CLI; do not use the dashboard to arm in this phase.",
            )
        )
    elif paper.get("display_status") == "MISSING":
        items.append(
            AttentionItem(
                "MEDIUM",
                "Paper arm missing",
                "No paper_arm_confirm.json latch was found.",
                str(PAPER_ARM_CONFIRM),
                "If paper entries are expected, create a latch using the approved paper-arm CLI outside this dashboard.",
            )
        )
    elif paper.get("display_status") == "INVALID":
        items.append(
            AttentionItem(
                "HIGH",
                "Paper arm invalid",
                "Paper-arm confirm artifact exists but failed validity checks.",
                str(PAPER_ARM_CONFIRM),
                "Inspect the confirm artifact and re-issue via approved CLI if appropriate.",
            )
        )

    if not cont.get("process_running"):
        items.append(
            AttentionItem(
                "CRITICAL" if snapshot_parts["overall_status"] != "STOPPED" else "HIGH",
                "Continuous loop missing",
                "No continuous-loop process matching run_continuous_loop.ps1 was detected.",
                "services.runtime_process_detector",
                "Use approved runtime start scripts or Task Scheduler backup; dashboard Start is not available yet.",
            )
        )
    if not wd.get("process_running"):
        items.append(
            AttentionItem(
                "CRITICAL" if snapshot_parts["overall_status"] != "STOPPED" else "HIGH",
                "Watchdog missing",
                "No watchdog-loop process matching run_watchdog_loop.ps1 was detected.",
                "services.runtime_process_detector",
                "Use approved runtime start scripts; dashboard Start is not available yet.",
            )
        )

    for label, loop in (("Continuous", cont), ("Watchdog", wd)):
        lc = str(loop.get("lock_classification") or "")
        if lc == "STALE_ORPHAN_LOCK":
            items.append(
                AttentionItem(
                    "HIGH",
                    f"{label} stale lock",
                    f"{label} lock file exists without a matching live process.",
                    str(loop.get("lock_path") or ""),
                    "Do not delete the lock from the dashboard. Use approved recovery/start paths that clear orphan locks safely.",
                )
            )
        elif lc in ("NO_LOCK_RUNNING", "MISSING_ACTIVE_LOCK", "STALE_ACTIVE_LOCK", "CORRUPT_LOCK"):
            items.append(
                AttentionItem(
                    "HIGH",
                    f"{label} process/lock mismatch",
                    f"{label} process and lock state disagree ({lc}).",
                    str(loop.get("lock_path") or "runtime_lock_classifier"),
                    "Inspect Diagnostics evidence; do not force-clear locks from the dashboard.",
                )
            )

    if cycle.get("overdue"):
        items.append(
            AttentionItem(
                "MEDIUM",
                "Cycle overdue",
                f"Last evidenced continuous cycle is older than {CONTINUOUS_CYCLE_OVERDUE_MINUTES:.0f} minutes.",
                str(cycle.get("evidence_path") or ""),
                "Check continuous-loop health and logs; do not restart from the dashboard in this phase.",
            )
        )

    if broker.get("connected") is False:
        items.append(
            AttentionItem(
                "HIGH",
                "Broker disconnected",
                "Watchdog/broker evidence indicates the broker is not connected.",
                str(WATCHDOG_STATUS),
                "Verify broker connectivity outside the dashboard; do not place orders.",
            )
        )

    if freshness.get("class") == "STALE":
        items.append(
            AttentionItem(
                "MEDIUM",
                "Stale market data",
                "Data freshness classification is STALE based on heartbeat/freshness evidence.",
                str(freshness.get("evidence_path") or HEARTBEAT),
                "Allow the data-freshness scheduled task to run, or inspect freshness audit artifacts.",
            )
        )

    if int(alerts.get("critical_count") or 0) > 0:
        items.append(
            AttentionItem(
                "CRITICAL",
                "Active critical alert",
                f"{alerts.get('critical_count')} critical watchdog alert(s) are active.",
                str(WATCHDOG_ALERTS),
                "Open Risk & Alerts and follow the recommended operator response; do not dismiss from this console.",
            )
        )

    if gov.get("blocked"):
        items.append(
            AttentionItem(
                "HIGH",
                "Governance block",
                "Governance authorization is not fully authorized.",
                str(GOVERNANCE_AUTH),
                "Review Governance Authorization evidence; do not override from the dashboard.",
            )
        )

    if gov.get("execution_readiness_ready") is False:
        items.append(
            AttentionItem(
                "MEDIUM",
                "Execution readiness not ready",
                "execution_readiness.json does not indicate a ready posture.",
                str(EXECUTION_READINESS),
                "Review readiness details on Risk & Alerts / Diagnostics.",
            )
        )

    if live.get("permitted") is True:
        items.append(
            AttentionItem(
                "CRITICAL",
                "Live execution appears permitted",
                "Evidence indicates live_execution_permitted is true. This violates the operator-console safety expectation.",
                str(EXECUTION_READINESS),
                "Treat as a critical safety incident. Do not enable trading. Investigate readiness/governance immediately.",
            )
        )

    return items


def build_operator_status(
    *,
    root: Optional[Path] = None,
    process_detector: Optional[Callable[[], Dict[str, Any]]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build structured operator status. Read-only."""
    now = now or _utc_now()
    root = Path(root) if root is not None else ROOT
    results = root / "data" / "results"
    live = root / "data" / "live"

    paths = {
        "continuous_lock": live / "continuous_loop.lock",
        "watchdog_lock": live / "watchdog_loop.lock",
        "paper_arm": live / "paper_arm_confirm.json",
        "watchdog_status": results / "watchdog_status.json",
        "watchdog_alerts": results / "watchdog_alerts.json",
        "execution_readiness": results / "execution_readiness.json",
        "governance_authorization": results / "governance_authorization.json",
        "heartbeat": results / "heartbeat.json",
        "runtime_autostart_931": results / "runtime_autostart_931.json",
        "runtime_supervisor": results / "runtime_supervisor_autostart.json",
        "freshness_audit": results / "data_freshness_refresh_audit.json",
        "open_orders": results / "open_orders_snapshot.csv",
        "positions": results / "positions_snapshot.csv",
        "portfolio_history": results / "portfolio_history.csv",
    }

    if process_detector is not None:
        proc = process_detector()
    elif root == ROOT:
        from services.runtime_process_detector import detect_runtime_loop_processes

        proc = detect_runtime_loop_processes()
    else:
        # Fixture roots: do not scan real machine processes
        from services.runtime_lock_classifier import classify_runtime_lock, is_lock_healthy

        wd_lock = _parse_lock_file(paths["watchdog_lock"])
        cl_lock = _parse_lock_file(paths["continuous_lock"])
        wd_running = False
        cl_running = False
        # Optional fixture marker for tests
        marker = live / "fixture_process_state.json"
        if marker.is_file():
            try:
                m = json.loads(marker.read_text(encoding="utf-8"))
                wd_running = bool(m.get("watchdog_loop_running"))
                cl_running = bool(m.get("continuous_loop_running"))
                wd_pids = list(m.get("detected_watchdog_process_ids") or [])
                cl_pids = list(m.get("detected_continuous_process_ids") or [])
            except Exception:
                wd_pids, cl_pids = [], []
        else:
            wd_pids, cl_pids = [], []
        wd_class = classify_runtime_lock(
            wd_lock, wd_running, wd_pids, loop_kind="watchdog", heartbeat_age_minutes=None
        )
        cl_class = classify_runtime_lock(
            cl_lock, cl_running, cl_pids, loop_kind="continuous", heartbeat_age_minutes=None
        )
        proc = {
            "watchdog_loop_running": wd_running,
            "continuous_loop_running": cl_running,
            "detected_watchdog_process_ids": wd_pids,
            "detected_continuous_process_ids": cl_pids,
            "process_detection_confidence": "high",
            "watchdog_lock_classification": wd_class,
            "continuous_lock_classification": cl_class,
            "watchdog_lock_healthy": is_lock_healthy(wd_class),
            "continuous_lock_healthy": is_lock_healthy(cl_class),
            "watchdog_lock_exists": paths["watchdog_lock"].is_file(),
            "continuous_lock_exists": paths["continuous_lock"].is_file(),
        }

    # Paper arm
    if root == ROOT:
        try:
            from services.paper_arm_status import classify_paper_arm_confirm_status

            paper_raw = classify_paper_arm_confirm_status()
        except Exception:
            paper_raw = "UNKNOWN"
    else:
        arm_path = paths["paper_arm"]
        if not arm_path.is_file():
            paper_raw = "MISSING"
        else:
            doc = _read_json(arm_path)
            if doc.get("__malformed__"):
                paper_raw = "INVALID"
            elif doc.get("allow_execute") is False:
                paper_raw = "INVALID"
            else:
                exp = _parse_dt(doc.get("expires_at"))
                if exp is None:
                    paper_raw = "INVALID"
                elif exp < now:
                    paper_raw = "EXPIRED"
                else:
                    paper_raw = "VALID"
    paper_display = map_paper_arm_display(paper_raw)

    readiness = _read_json(paths["execution_readiness"])
    gov = _read_json(paths["governance_authorization"])
    wd_status = _read_json(paths["watchdog_status"])
    alerts_doc = _read_json(paths["watchdog_alerts"])
    heartbeat = _read_json(paths["heartbeat"])
    freshness_audit = _read_json(paths["freshness_audit"])

    live_permitted = bool(readiness.get("live_execution_permitted")) if readiness else False
    # Also check portfolio flag when on real root
    if root == ROOT:
        try:
            from services.portfolio_history_sync import load_live_execution_flag

            live_permitted = live_permitted or bool(load_live_execution_flag())
        except Exception:
            pass

    live_blocked = not live_permitted

    # Cycle evidence from continuous lock metadata
    cl_lock = _parse_lock_file(paths["continuous_lock"])
    wd_lock = _parse_lock_file(paths["watchdog_lock"])
    cycle_num = cl_lock.get("cycle")
    refreshed = _parse_dt(cl_lock.get("refreshed_at") or cl_lock.get("started_at"))
    cycle_age = _age_minutes(refreshed, now)
    cycle_overdue = cycle_age is not None and cycle_age > CONTINUOUS_CYCLE_OVERDUE_MINUTES

    next_expected = None
    if refreshed is not None:
        from datetime import timedelta

        next_expected = (refreshed + timedelta(minutes=45)).isoformat()

    # Alerts
    active_alerts = []
    if isinstance(alerts_doc.get("active_alerts"), list):
        active_alerts = [a for a in alerts_doc["active_alerts"] if isinstance(a, dict)]
    elif isinstance(wd_status.get("active_alerts"), list):
        active_alerts = [a for a in wd_status["active_alerts"] if isinstance(a, dict)]
    critical_count = sum(
        1
        for a in active_alerts
        if str(a.get("severity") or "").upper() in ("CRITICAL", "HIGH", "FATAL")
    )

    broker_connected = wd_status.get("broker_connected")
    if broker_connected is None and isinstance(wd_status.get("heartbeat"), dict):
        broker_connected = wd_status["heartbeat"].get("broker_connected")

    hb_ts = _parse_dt(
        heartbeat.get("timestamp") or heartbeat.get("ts") or heartbeat.get("updated_at")
    )
    hb_age = _age_minutes(hb_ts, now)
    if hb_age is None:
        freshness_class = "UNKNOWN"
    elif hb_age <= DATA_FRESH_MINUTES:
        freshness_class = "FRESH"
    else:
        freshness_class = "STALE"

    open_orders = _csv_row_count(paths["open_orders"])
    positions = _csv_row_count(paths["positions"])

    auth_status = str(gov.get("overall_authorization") or gov.get("authorization_status") or "")
    gov_blocked = bool(gov) and auth_status.upper() not in (
        "",
        "AUTHORIZED",
        "GREEN",
        "APPROVED",
        "PASS",
    )
    readiness_status = str(readiness.get("readiness_status") or readiness.get("status") or "")
    exec_ready = readiness_status.upper() in ("READY", "PASS", "GREEN", "AUTHORIZED")
    if readiness and "readiness_status" not in readiness and "status" not in readiness:
        exec_ready = None  # unknown shape

    overall = classify_overall_status(
        continuous_running=bool(proc.get("continuous_loop_running")),
        watchdog_running=bool(proc.get("watchdog_loop_running")),
        continuous_lock_healthy=bool(proc.get("continuous_lock_healthy")),
        watchdog_lock_healthy=bool(proc.get("watchdog_lock_healthy")),
        continuous_lock_class=str(proc.get("continuous_lock_classification") or ""),
        watchdog_lock_class=str(proc.get("watchdog_lock_classification") or ""),
        continuous_cycle_age_minutes=cycle_age,
        detection_confidence=str(proc.get("process_detection_confidence") or "high"),
    )

    parts = {
        "overall_status": overall,
        "continuous": {
            "process_running": bool(proc.get("continuous_loop_running")),
            "pids": list(proc.get("detected_continuous_process_ids") or []),
            "lock_exists": bool(proc.get("continuous_lock_exists")),
            "lock_classification": proc.get("continuous_lock_classification"),
            "lock_healthy": bool(proc.get("continuous_lock_healthy")),
            "lock_path": str(paths["continuous_lock"]),
            "lock_pid": cl_lock.get("pid"),
            "cycle": cycle_num,
            "refreshed_at": refreshed.isoformat() if refreshed else None,
        },
        "watchdog": {
            "process_running": bool(proc.get("watchdog_loop_running")),
            "pids": list(proc.get("detected_watchdog_process_ids") or []),
            "lock_exists": bool(proc.get("watchdog_lock_exists")),
            "lock_classification": proc.get("watchdog_lock_classification"),
            "lock_healthy": bool(proc.get("watchdog_lock_healthy")),
            "lock_path": str(paths["watchdog_lock"]),
            "lock_pid": wd_lock.get("pid"),
            "status_path": str(paths["watchdog_status"]),
        },
        "paper_arm": {
            "raw_status": paper_raw,
            "display_status": paper_display,
            "evidence_path": str(paths["paper_arm"]),
            "exists": paths["paper_arm"].is_file(),
        },
        "live_execution": {
            "permitted": live_permitted,
            "blocked": live_blocked,
            "display": (
                "LIVE EXECUTION BLOCKED"
                if live_blocked
                else "LIVE EXECUTION NOT BLOCKED — CRITICAL"
            ),
            "evidence_path": str(paths["execution_readiness"]),
        },
        "broker": {
            "connected": broker_connected,
            "evidence_path": str(paths["watchdog_status"]),
            "open_orders_mtime": (
                _file_mtime(paths["open_orders"]).isoformat()
                if _file_mtime(paths["open_orders"])
                else None
            ),
        },
        "market": {
            "status": "UNKNOWN",
            "note": "Market session classification is informational; weekday/session checks live in start scripts.",
        },
        "cycle": {
            "last_cycle_number": cycle_num,
            "last_completed_at": refreshed.isoformat() if refreshed else None,
            "age_minutes": cycle_age,
            "overdue": cycle_overdue,
            "next_expected_at": next_expected,
            "evidence_path": str(paths["continuous_lock"]),
        },
        "orders_positions": {
            "open_orders": open_orders,
            "position_count": positions,
            "open_orders_path": str(paths["open_orders"]),
            "positions_path": str(paths["positions"]),
        },
        "alerts": {
            "active_count": len(active_alerts),
            "critical_count": critical_count,
            "active_alerts": active_alerts[:50],
            "evidence_path": str(paths["watchdog_alerts"]),
        },
        "data_freshness": {
            "class": freshness_class,
            "heartbeat_age_minutes": hb_age,
            "heartbeat_timestamp": hb_ts.isoformat() if hb_ts else None,
            "freshness_audit_present": bool(freshness_audit)
            and not freshness_audit.get("__malformed__"),
            "evidence_path": str(paths["heartbeat"]),
        },
        "governance": {
            "authorization_status": auth_status or "UNKNOWN",
            "blocked": gov_blocked,
            "execution_readiness_status": readiness_status or "UNKNOWN",
            "execution_readiness_ready": exec_ready,
            "evidence_paths": {
                "governance": str(paths["governance_authorization"]),
                "readiness": str(paths["execution_readiness"]),
            },
        },
    }
    attention = [a.to_dict() for a in _build_attention(parts)]

    snap = OperatorStatusSnapshot(
        generated_at=now.isoformat(),
        overall_status=overall,
        continuous=parts["continuous"],
        watchdog=parts["watchdog"],
        paper_arm=parts["paper_arm"],
        live_execution=parts["live_execution"],
        broker=parts["broker"],
        market=parts["market"],
        cycle=parts["cycle"],
        orders_positions=parts["orders_positions"],
        alerts=parts["alerts"],
        data_freshness=parts["data_freshness"],
        governance=parts["governance"],
        attention=attention,
        evidence_paths={k: str(v) for k, v in paths.items()},
        runtime_controls_exposed=False,
        live_execution_control_exposed=False,
    )
    return snap.to_dict()


def build_portfolio_summary(*, root: Optional[Path] = None) -> Dict[str, Any]:
    """Read-only portfolio summary from trusted artifacts."""
    root = Path(root) if root is not None else ROOT
    results = root / "data" / "results"
    history = results / "portfolio_history.csv"
    positions = results / "positions_snapshot.csv"
    overlay = results / "performance_risk_overlay.csv"

    out: Dict[str, Any] = {
        "cash": None,
        "market_value": None,
        "total_portfolio_value": None,
        "realized_pnl": None,
        "unrealized_pnl": None,
        "position_count": _csv_row_count(positions),
        "concentration_warnings": [],
        "holdings": [],
        "risk_flags": [],
        "last_refresh_time": None,
        "evidence_paths": {
            "portfolio_history": str(history),
            "positions": str(positions),
            "risk_overlay": str(overlay),
        },
        "available": False,
    }

    mtime = _file_mtime(history) or _file_mtime(positions)
    out["last_refresh_time"] = mtime.isoformat() if mtime else None

    if history.is_file():
        try:
            import pandas as pd

            df = pd.read_csv(history)
            if not df.empty:
                # Prefer latest date aggregate
                if "date" in df.columns and "total_value" in df.columns:
                    latest = str(df["date"].iloc[-1])
                    day = df[df["date"].astype(str) == latest]
                    if "total_value" in day.columns:
                        out["total_portfolio_value"] = float(day["total_value"].sum())
                    if "cash" in day.columns:
                        out["cash"] = float(day["cash"].iloc[-1]) if len(day) else None
                    if "market_value" in day.columns:
                        out["market_value"] = float(day["market_value"].sum())
                out["available"] = True
        except Exception:
            out["parse_error"] = "portfolio_history_unreadable"

    if positions.is_file():
        try:
            import pandas as pd

            pdf = pd.read_csv(positions)
            holdings = []
            for _, row in pdf.head(200).iterrows():
                holdings.append({k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()})
            out["holdings"] = holdings
            out["position_count"] = len(pdf)
            out["available"] = True
            if "unrealized_pl" in pdf.columns or "unrealized_pnl" in pdf.columns:
                col = "unrealized_pl" if "unrealized_pl" in pdf.columns else "unrealized_pnl"
                out["unrealized_pnl"] = float(
                    pd.to_numeric(pdf[col], errors="coerce").fillna(0).sum()
                )
            if "market_value" in pdf.columns and out["market_value"] is None:
                out["market_value"] = float(
                    pd.to_numeric(pdf["market_value"], errors="coerce").fillna(0).sum()
                )
        except Exception:
            out["positions_parse_error"] = "positions_unreadable"

    if overlay.is_file():
        try:
            import pandas as pd

            odf = pd.read_csv(overlay)
            flags = []
            for _, row in odf.iterrows():
                flag = str(row.get("risk_flag") or "")
                if flag and flag.upper() not in ("", "NONE", "OK", "NAN"):
                    flags.append(
                        {
                            "ticker": row.get("ticker") or row.get("symbol"),
                            "risk_flag": flag,
                        }
                    )
            out["risk_flags"] = flags
            # Concentration-ish warnings
            out["concentration_warnings"] = [
                f for f in flags if "CONCENTR" in str(f.get("risk_flag") or "").upper()
            ]
        except Exception:
            pass

    return out


ORDER_STATUSES = (
    "PLANNED",
    "AUTHORIZED",
    "SUBMITTED",
    "ACCEPTED",
    "PARTIALLY_FILLED",
    "FILLED",
    "REJECTED",
    "CANCELLED",
    "UNKNOWN",
)


def normalize_order_status(raw: Any) -> str:
    s = str(raw or "").strip().upper().replace(" ", "_")
    aliases = {
        "NEW": "SUBMITTED",
        "PENDING_NEW": "SUBMITTED",
        "PENDING": "SUBMITTED",
        "OPEN": "ACCEPTED",
        "PARTIALLYFILLED": "PARTIALLY_FILLED",
        "PARTIAL": "PARTIALLY_FILLED",
        "PARTIAL_FILL": "PARTIALLY_FILLED",
        "DONE": "FILLED",
        "COMPLETE": "FILLED",
        "COMPLETED": "FILLED",
        "CANCELED": "CANCELLED",
        "CANCEL": "CANCELLED",
        "REJECT": "REJECTED",
        "FAILED": "REJECTED",
        "SKIPPED": "REJECTED",
        "PLANNED": "PLANNED",
        "AUTHORIZED": "AUTHORIZED",
        "SUBMITTED": "SUBMITTED",
        "ACCEPTED": "ACCEPTED",
        "FILLED": "FILLED",
        "REJECTED": "REJECTED",
        "CANCELLED": "CANCELLED",
        "PARTIALLY_FILLED": "PARTIALLY_FILLED",
    }
    mapped = aliases.get(s, s if s in ORDER_STATUSES else "UNKNOWN")
    return mapped if mapped in ORDER_STATUSES else "UNKNOWN"


def build_orders_summary(*, root: Optional[Path] = None) -> Dict[str, Any]:
    """Read-only paper/order visibility from snapshots and audits."""
    root = Path(root) if root is not None else ROOT
    results = root / "data" / "results"
    paths = {
        "open_orders": results / "open_orders_snapshot.csv",
        "recent_orders": results / "recent_orders.csv",
        "paper_execution_audit": results / "paper_execution_audit.csv",
        "live_orders": results / "live_orders.csv",
    }
    orders: List[Dict[str, Any]] = []

    def _ingest_csv(path: Path, source: str) -> None:
        if not path.is_file():
            return
        try:
            import pandas as pd

            df = pd.read_csv(path)
            for _, row in df.head(500).iterrows():
                d = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                status = normalize_order_status(
                    d.get("status") or d.get("order_status") or d.get("state")
                )
                orders.append(
                    {
                        "symbol": d.get("symbol") or d.get("ticker"),
                        "side": d.get("side") or d.get("action"),
                        "quantity": d.get("qty") or d.get("quantity") or d.get("shares"),
                        "notional": d.get("notional") or d.get("market_value") or d.get("amount"),
                        "status": status,
                        "raw_status": d.get("status") or d.get("order_status"),
                        "submitted_time": d.get("submitted_at")
                        or d.get("created_at")
                        or d.get("timestamp"),
                        "filled_time": d.get("filled_at") or d.get("filled_time"),
                        "broker_order_id": d.get("id")
                        or d.get("order_id")
                        or d.get("submitted_order_id")
                        or d.get("broker_order_id"),
                        "evidence_source": source,
                        "evidence_path": str(path),
                    }
                )
        except Exception:
            pass

    _ingest_csv(paths["open_orders"], "open_orders_snapshot")
    _ingest_csv(paths["recent_orders"], "recent_orders")
    _ingest_csv(paths["paper_execution_audit"], "paper_execution_audit")
    _ingest_csv(paths["live_orders"], "live_orders")

    by_status: Dict[str, List[Dict[str, Any]]] = {s: [] for s in ORDER_STATUSES}
    for o in orders:
        by_status.setdefault(o["status"], []).append(o)

    return {
        "orders": orders,
        "by_status": {k: len(v) for k, v in by_status.items()},
        "open_orders": by_status.get("ACCEPTED", [])
        + by_status.get("SUBMITTED", [])
        + by_status.get("PARTIALLY_FILLED", []),
        "filled_orders": by_status.get("FILLED", []),
        "rejected_orders": by_status.get("REJECTED", []),
        "cancelled_orders": by_status.get("CANCELLED", []),
        "evidence_paths": {k: str(v) for k, v in paths.items()},
        "note": "PLANNED/AUTHORIZED are distinct from SUBMITTED/FILLED; status is never inferred upward.",
    }


def build_risk_alerts_summary(*, root: Optional[Path] = None) -> Dict[str, Any]:
    """Read-only risk and alert posture."""
    root = Path(root) if root is not None else ROOT
    results = root / "data" / "results"
    paths = {
        "watchdog_alerts": results / "watchdog_alerts.json",
        "watchdog_status": results / "watchdog_status.json",
        "risk_overlay": results / "performance_risk_overlay.csv",
        "capital_preservation": results / "capital_preservation_governor.json",
        "execution_readiness": results / "execution_readiness.json",
        "governance_authorization": results / "governance_authorization.json",
    }
    alerts_doc = _read_json(paths["watchdog_alerts"])
    wd_status = _read_json(paths["watchdog_status"])
    cpg = _read_json(paths["capital_preservation"])
    readiness = _read_json(paths["execution_readiness"])
    gov = _read_json(paths["governance_authorization"])

    active = []
    if isinstance(alerts_doc.get("active_alerts"), list):
        active = [a for a in alerts_doc["active_alerts"] if isinstance(a, dict)]
    resolved = []
    if isinstance(alerts_doc.get("resolved_alerts"), list):
        resolved = [a for a in alerts_doc["resolved_alerts"] if isinstance(a, dict)]

    overlay_actions = []
    if paths["risk_overlay"].is_file():
        try:
            import pandas as pd

            df = pd.read_csv(paths["risk_overlay"])
            for _, row in df.iterrows():
                flag = str(row.get("risk_flag") or "")
                if flag and flag.upper() not in ("", "NONE", "OK", "NAN"):
                    overlay_actions.append(
                        {
                            "ticker": row.get("ticker") or row.get("symbol"),
                            "risk_flag": flag,
                        }
                    )
        except Exception:
            pass

    return {
        "capital_preservation_posture": cpg.get("status")
        or cpg.get("posture")
        or cpg.get("governor_status")
        or ("AVAILABLE" if cpg and not cpg.get("__malformed__") else "UNKNOWN"),
        "capital_preservation_detail": (
            {k: cpg.get(k) for k in list(cpg.keys())[:20] if not str(k).startswith("__")}
            if cpg and not cpg.get("__malformed__")
            else {}
        ),
        "active_conditions": active,
        "resolved_conditions": resolved,
        "informational_notices": (
            [
                {
                    "title": "Watchdog broker flag",
                    "detail": f"broker_connected={wd_status.get('broker_connected')}",
                }
            ]
            if wd_status
            else []
        ),
        "risk_overlay_actions": overlay_actions,
        "concentration_issues": [
            a for a in overlay_actions if "CONCENTR" in str(a.get("risk_flag") or "").upper()
        ],
        "drawdown_warnings": [
            a
            for a in overlay_actions
            if "DRAWDOWN" in str(a.get("risk_flag") or "").upper()
            or "DD" in str(a.get("risk_flag") or "").upper()
        ],
        "governance_blocks": {
            "authorization_status": gov.get("overall_authorization")
            or gov.get("authorization_status"),
            "blocked": bool(gov)
            and str(gov.get("overall_authorization") or "").upper()
            not in ("", "AUTHORIZED", "GREEN", "APPROVED", "PASS"),
        },
        "execution_readiness": {
            "status": readiness.get("readiness_status") or readiness.get("status"),
            "live_execution_permitted": readiness.get("live_execution_permitted"),
        },
        "recommended_operator_response": (
            "Review active critical alerts and governance posture. Do not dismiss or override from this console."
            if active
            else "No active alert list evidenced. Continue routine observation."
        ),
        "evidence_timestamps": {
            "watchdog_alerts_mtime": (
                _file_mtime(paths["watchdog_alerts"]).isoformat()
                if _file_mtime(paths["watchdog_alerts"])
                else None
            ),
            "risk_overlay_mtime": (
                _file_mtime(paths["risk_overlay"]).isoformat()
                if _file_mtime(paths["risk_overlay"])
                else None
            ),
        },
        "evidence_paths": {k: str(v) for k, v in paths.items()},
        "can_dismiss_or_override": False,
    }
