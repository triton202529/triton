"""
operator_activity_feed.py
Phase DO-002 — Read-only activity event normalization for the operator console.

Never invents successes. Never writes source artifacts.
exit_code=0 ≠ order submitted; submitted ≠ filled; cycle start ≠ cycle complete.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"

CATEGORIES = (
    "RUNTIME",
    "CYCLE",
    "DATA",
    "SIGNAL",
    "ENTRY_GATE",
    "ORDER",
    "RISK",
    "ALERT",
    "GOVERNANCE",
    "BROKER",
    "SAFETY",
)

SEVERITIES = ("INFO", "SUCCESS", "WARN", "BLOCK", "ACTION_REQUIRED", "CRITICAL")

MAX_LOG_BYTES = 512_000
MAX_LOG_LINES = 2_000
MAX_EVENTS = 400


@dataclass
class ActivityEvent:
    event_id: str
    occurred_at: str
    source: str
    category: str
    severity: str
    title: str
    message: str
    evidence_path: str
    cycle_number: Optional[int] = None
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    raw_reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
        # Common log prefix: 2026-07-28 17:08:31
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _event_id(*parts: Any) -> str:
    blob = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()[:24]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {"__malformed__": True}


def _tail_text(path: Path, max_bytes: int = MAX_LOG_BYTES) -> str:
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # discard partial
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _latest_glob(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _extract_log_ts(line: str, fallback: Optional[datetime] = None) -> datetime:
    m = re.match(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", line)
    if m:
        dt = _parse_dt(m.group(1))
        if dt:
            return dt
    return fallback or _utc_now()


def _parse_continuous_log(path: Path, seen: Set[str]) -> List[ActivityEvent]:
    events: List[ActivityEvent] = []
    text = _tail_text(path)
    if not text:
        return events
    lines = text.splitlines()[-MAX_LOG_LINES:]
    for line in lines:
        ts = _extract_log_ts(line)
        cycle = None
        m_cycle = re.search(r"cycle[=:\s]+(\d+)", line, re.I)
        if m_cycle:
            cycle = int(m_cycle.group(1))

        def add(cat: str, sev: str, title: str, message: str, raw: str = "") -> None:
            eid = _event_id(
                path.name, ts.isoformat(), title, message, cycle, raw or line.strip()[:120]
            )
            if eid in seen:
                return
            seen.add(eid)
            events.append(
                ActivityEvent(
                    event_id=eid,
                    occurred_at=ts.isoformat(),
                    source="continuous_loop_log",
                    category=cat,
                    severity=sev,
                    title=title,
                    message=message,
                    evidence_path=str(path),
                    cycle_number=cycle,
                    raw_reference=(raw or line.strip())[:500],
                )
            )

        if "=== TRITON CONTINUOUS LOOP START" in line or "[LOOP_START]" in line:
            add(
                "RUNTIME",
                "INFO",
                "Continuous loop process started",
                "Continuous-loop process start banner evidenced in log.",
            )
        if re.search(r"\[CYCLE_START\]", line):
            add(
                "CYCLE",
                "INFO",
                (
                    f"Continuous cycle {cycle} started"
                    if cycle is not None
                    else "Continuous cycle started"
                ),
                "Cycle start evidenced. This does not prove cycle completion.",
            )
        if re.search(r"\[CYCLE_OK\]", line):
            add(
                "CYCLE",
                "SUCCESS",
                (
                    f"Cycle {cycle} completed successfully"
                    if cycle is not None
                    else "Cycle completed successfully"
                ),
                "Cycle completion evidenced by [CYCLE_OK].",
            )
        if re.search(r"\[CYCLE_FAIL\]", line):
            add(
                "CYCLE",
                "WARN",
                f"Cycle {cycle} failed" if cycle is not None else "Cycle failed",
                "Cycle failure evidenced by [CYCLE_FAIL].",
            )
        if "SIGNAL_UNIVERSE" in line or "tickers_processed=" in line:
            m_n = re.search(r"tickers_processed[=:\s]+(\d+)", line, re.I)
            n = m_n.group(1) if m_n else None
            add(
                "SIGNAL",
                "INFO",
                "Market-data / signal universe processing evidenced",
                f"Evaluated {n} symbols." if n else "Signal-universe processing line evidenced.",
                line.strip(),
            )
        if "PAPER_ENTRY" in line and "BLOCK" in line.upper():
            add(
                "ENTRY_GATE",
                "BLOCK",
                "Paper entry blocked",
                "Paper-entry block evidenced in continuous-loop log.",
            )
        # Explicitly ignore bare exit_code=0 as an order event
        if re.search(r"exit_code\s*=\s*0", line, re.I) and "order" not in line.lower():
            # Stage success only — never ORDER
            add(
                "RUNTIME",
                "INFO",
                "Stage completed successfully",
                "Stage exit_code=0 evidenced. This does not prove an order was submitted.",
            )
    return events


def _parse_watchdog_log(path: Path, seen: Set[str]) -> List[ActivityEvent]:
    events: List[ActivityEvent] = []
    text = _tail_text(path)
    if not text:
        return events
    for line in text.splitlines()[-MAX_LOG_LINES:]:
        ts = _extract_log_ts(line)
        cycle = None
        m_cycle = re.search(r"cycle[=:\s]+(\d+)", line, re.I)
        if m_cycle:
            cycle = int(m_cycle.group(1))

        def add(cat: str, sev: str, title: str, message: str) -> None:
            eid = _event_id(path.name, ts.isoformat(), title, message, cycle, line.strip()[:120])
            if eid in seen:
                return
            seen.add(eid)
            events.append(
                ActivityEvent(
                    event_id=eid,
                    occurred_at=ts.isoformat(),
                    source="watchdog_loop_log",
                    category=cat,
                    severity=sev,
                    title=title,
                    message=message,
                    evidence_path=str(path),
                    cycle_number=cycle,
                    raw_reference=line.strip()[:500],
                )
            )

        if re.search(r"\[CYCLE_OK\]", line):
            add(
                "ALERT",
                "SUCCESS",
                "Watchdog status is healthy",
                "Watchdog cycle completed successfully ([CYCLE_OK]).",
            )
        if re.search(r"\[CYCLE_FAIL\]", line):
            add("ALERT", "WARN", "Watchdog cycle failed", "Watchdog [CYCLE_FAIL] evidenced.")
        if "STALE" in line.upper() and "HEARTBEAT" in line.upper():
            add(
                "DATA",
                "WARN",
                "Watchdog reported stale heartbeat",
                "Stale heartbeat wording evidenced in watchdog log.",
            )
    return events


def _events_from_startup(path: Path, seen: Set[str]) -> List[ActivityEvent]:
    doc = _read_json(path)
    if not doc or doc.get("__malformed__"):
        return []
    ts = (
        _parse_dt(doc.get("completed_at") or doc.get("generated_at") or doc.get("timestamp"))
        or _utc_now()
    )
    events: List[ActivityEvent] = []
    if doc.get("postconditions_passed") is True:
        eid = _event_id(path.name, "startup_pass", ts.isoformat())
        if eid not in seen:
            seen.add(eid)
            events.append(
                ActivityEvent(
                    event_id=eid,
                    occurred_at=ts.isoformat(),
                    source="runtime_autostart",
                    category="RUNTIME",
                    severity="SUCCESS",
                    title="TRITON startup validation passed",
                    message="Runtime autostart audit reports postconditions_passed=true.",
                    evidence_path=str(path),
                    raw_reference="postconditions_passed=true",
                )
            )
    elif doc.get("postconditions_passed") is False:
        eid = _event_id(path.name, "startup_fail", ts.isoformat())
        if eid not in seen:
            seen.add(eid)
            events.append(
                ActivityEvent(
                    event_id=eid,
                    occurred_at=ts.isoformat(),
                    source="runtime_autostart",
                    category="RUNTIME",
                    severity="WARN",
                    title="TRITON startup validation failed",
                    message="Runtime autostart audit reports postconditions_passed=false.",
                    evidence_path=str(path),
                    raw_reference="postconditions_passed=false",
                )
            )
    if doc.get("live_execution_enabled") is False or doc.get("live_execution_enabled") == 0:
        eid = _event_id(path.name, "live_blocked", ts.isoformat())
        if eid not in seen:
            seen.add(eid)
            events.append(
                ActivityEvent(
                    event_id=eid,
                    occurred_at=ts.isoformat(),
                    source="runtime_autostart",
                    category="SAFETY",
                    severity="INFO",
                    title="Live execution remains blocked",
                    message="Startup audit evidences live_execution_enabled=false.",
                    evidence_path=str(path),
                    raw_reference="live_execution_enabled=false",
                )
            )
    return events


def _events_from_paper_entry_stage(path: Path, seen: Set[str]) -> List[ActivityEvent]:
    doc = _read_json(path)
    if not doc or doc.get("__malformed__"):
        return []
    ts = (
        _parse_dt(doc.get("generated_at") or doc.get("timestamp") or doc.get("completed_at"))
        or _utc_now()
    )
    events: List[ActivityEvent] = []

    eid = _event_id(path.name, "gates_evaluated", ts.isoformat())
    if eid not in seen:
        seen.add(eid)
        events.append(
            ActivityEvent(
                event_id=eid,
                occurred_at=ts.isoformat(),
                source="paper_entry_stage_audit",
                category="ENTRY_GATE",
                severity="INFO",
                title="Paper-entry gates were evaluated",
                message="Paper-entry stage audit artifact is present.",
                evidence_path=str(path),
                raw_reference="paper_entry_loop_stage_audit",
            )
        )

    gate_pass = doc.get("gate_pass")
    eligible = doc.get("eligible_entry_symbols") or doc.get("final_selected") or []
    if isinstance(eligible, dict):
        eligible = list(eligible.keys())
    if gate_pass is False or (
        isinstance(eligible, list) and len(eligible) == 0 and "gate_pass" in doc
    ):
        eid2 = _event_id(path.name, "no_candidate", ts.isoformat(), gate_pass)
        if eid2 not in seen:
            seen.add(eid2)
            reason = str(
                doc.get("gate_reason") or "No eligible symbols evidenced after gate evaluation."
            )
            events.append(
                ActivityEvent(
                    event_id=eid2,
                    occurred_at=ts.isoformat(),
                    source="paper_entry_stage_audit",
                    category="ENTRY_GATE",
                    severity="INFO",
                    title="No candidate passed all entry gates",
                    message=reason[:400],
                    evidence_path=str(path),
                    raw_reference=f"gate_pass={gate_pass}",
                )
            )

    # Risk block evidence only when explicit
    rejected = doc.get("rejected") or doc.get("rejected_symbols") or doc.get("rejected_map") or {}
    rejected_text = json.dumps(rejected) if rejected else ""
    if "RISK_OVERLAY" in rejected_text.upper() or "BLOCK_NEW_BUY" in rejected_text.upper():
        eid3 = _event_id(path.name, "risk_block", ts.isoformat())
        if eid3 not in seen:
            seen.add(eid3)
            events.append(
                ActivityEvent(
                    event_id=eid3,
                    occurred_at=ts.isoformat(),
                    source="paper_entry_stage_audit",
                    category="RISK",
                    severity="BLOCK",
                    title="Entry was blocked by risk controls",
                    message="Stage audit rejected map evidences risk-overlay / BLOCK_NEW_BUY style reasons.",
                    evidence_path=str(path),
                    raw_reference=rejected_text[:500],
                )
            )

    # Orders submitted only with explicit evidence
    submitted = doc.get("broker_orders_submitted") or doc.get("orders_submitted")
    if isinstance(submitted, list) and submitted:
        for item in submitted[:20]:
            oid = None
            sym = None
            if isinstance(item, dict):
                oid = item.get("order_id") or item.get("submitted_order_id") or item.get("id")
                sym = item.get("symbol") or item.get("ticker")
            else:
                oid = str(item)
            eid4 = _event_id(path.name, "submitted", oid, sym, ts.isoformat())
            if eid4 in seen:
                continue
            seen.add(eid4)
            events.append(
                ActivityEvent(
                    event_id=eid4,
                    occurred_at=ts.isoformat(),
                    source="paper_entry_stage_audit",
                    category="ORDER",
                    severity="SUCCESS",
                    title="Paper order was submitted",
                    message="Stage audit lists a submitted broker order id.",
                    evidence_path=str(path),
                    symbol=str(sym) if sym else None,
                    order_id=str(oid) if oid else None,
                    raw_reference=str(item)[:300],
                )
            )
    return events


def _events_from_paper_execution_audit(path: Path, seen: Set[str]) -> List[ActivityEvent]:
    if not path.is_file():
        return []
    events: List[ActivityEvent] = []
    try:
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)[-200:]
    except Exception:
        return []

    for row in rows:
        ts = (
            _parse_dt(row.get("timestamp") or row.get("submitted_at") or row.get("created_at"))
            or _utc_now()
        )
        oid = row.get("submitted_order_id") or row.get("order_id") or row.get("id")
        sym = row.get("symbol") or row.get("ticker")
        status = str(row.get("status") or row.get("order_status") or "").upper()
        skip = str(row.get("skip_reason") or "")

        # exit_code alone must never become ORDER
        if not oid and "exit_code" in {k.lower() for k in row.keys()}:
            continue

        if oid and status not in ("FILLED", "PARTIAL", "PARTIALLY_FILLED"):
            eid = _event_id(path.name, "submitted", oid, status, ts.isoformat())
            if eid not in seen:
                seen.add(eid)
                events.append(
                    ActivityEvent(
                        event_id=eid,
                        occurred_at=ts.isoformat(),
                        source="paper_execution_audit",
                        category="ORDER",
                        severity="SUCCESS",
                        title="Paper order was submitted",
                        message=f"Audit row evidences submitted_order_id with status={status or 'UNKNOWN'}.",
                        evidence_path=str(path),
                        symbol=str(sym) if sym else None,
                        order_id=str(oid),
                        raw_reference=json.dumps(dict(list(row.items())[:12]))[:400],
                    )
                )
        if oid and status in ("FILLED", "PARTIAL", "PARTIALLY_FILLED"):
            # Fill only when status says so — never promote submitted→filled
            eid = _event_id(path.name, "filled", oid, status, ts.isoformat())
            if eid not in seen:
                seen.add(eid)
                events.append(
                    ActivityEvent(
                        event_id=eid,
                        occurred_at=ts.isoformat(),
                        source="paper_execution_audit",
                        category="ORDER",
                        severity="SUCCESS",
                        title=(
                            "Paper order was filled"
                            if status == "FILLED"
                            else "Paper order was partially filled"
                        ),
                        message=f"Audit row evidences fill status={status}.",
                        evidence_path=str(path),
                        symbol=str(sym) if sym else None,
                        order_id=str(oid),
                        raw_reference=json.dumps(dict(list(row.items())[:12]))[:400],
                    )
                )
        if skip and "RISK" in skip.upper():
            eid = _event_id(path.name, "risk_skip", oid, skip, ts.isoformat())
            if eid not in seen:
                seen.add(eid)
                events.append(
                    ActivityEvent(
                        event_id=eid,
                        occurred_at=ts.isoformat(),
                        source="paper_execution_audit",
                        category="RISK",
                        severity="BLOCK",
                        title="Entry was blocked by risk controls",
                        message=f"skip_reason evidences risk block: {skip[:200]}",
                        evidence_path=str(path),
                        symbol=str(sym) if sym else None,
                        order_id=str(oid) if oid else None,
                        raw_reference=skip[:300],
                    )
                )
    return events


def _events_from_watchdog_alerts(path: Path, seen: Set[str]) -> List[ActivityEvent]:
    doc = _read_json(path)
    if not doc or doc.get("__malformed__"):
        return []
    events: List[ActivityEvent] = []
    ts = (
        _parse_dt(doc.get("timestamp") or doc.get("generated_at"))
        or _file_mtime_dt(path)
        or _utc_now()
    )
    for alert in doc.get("active_alerts") or []:
        if not isinstance(alert, dict):
            continue
        atype = str(alert.get("alert_type") or alert.get("type") or "ALERT")
        sev_raw = str(alert.get("severity") or "WARN").upper()
        sev = "CRITICAL" if sev_raw in ("CRITICAL", "FATAL", "HIGH") else "WARN"
        if sev_raw == "HIGH":
            sev = "ACTION_REQUIRED"
        eid = _event_id(path.name, "alert", atype, alert.get("subject"), ts.isoformat())
        if eid in seen:
            continue
        seen.add(eid)
        events.append(
            ActivityEvent(
                event_id=eid,
                occurred_at=ts.isoformat(),
                source="watchdog_alerts",
                category="ALERT",
                severity=sev,
                title=f"Watchdog alert: {atype}",
                message=str(alert.get("details") or alert.get("subject") or atype)[:400],
                evidence_path=str(path),
                symbol=str(alert.get("symbol") or alert.get("ticker") or "") or None,
                raw_reference=json.dumps(alert)[:500],
            )
        )
    return events


def _file_mtime_dt(path: Path) -> Optional[datetime]:
    if not path.is_file():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _events_from_freshness(path: Path, seen: Set[str]) -> List[ActivityEvent]:
    doc = _read_json(path)
    if not doc or doc.get("__malformed__"):
        return []
    ts = (
        _parse_dt(doc.get("completed_at") or doc.get("generated_at") or doc.get("timestamp"))
        or _file_mtime_dt(path)
        or _utc_now()
    )
    if doc.get("errors"):
        return []
    steps = doc.get("steps_run") or doc.get("steps") or []
    eid = _event_id(path.name, "freshness_ok", ts.isoformat(), str(steps)[:80])
    if eid in seen:
        return []
    seen.add(eid)
    return [
        ActivityEvent(
            event_id=eid,
            occurred_at=ts.isoformat(),
            source="data_freshness_audit",
            category="DATA",
            severity="SUCCESS",
            title="Market-data processing completed",
            message="Data freshness refresh audit evidences a completed run without listed errors.",
            evidence_path=str(path),
            raw_reference=json.dumps({"steps_run": steps})[:400],
        )
    ]


def _events_from_readiness_governance(results: Path, seen: Set[str]) -> List[ActivityEvent]:
    events: List[ActivityEvent] = []
    readiness = _read_json(results / "execution_readiness.json")
    if readiness and not readiness.get("__malformed__"):
        ts = (
            _parse_dt(readiness.get("generated_at") or readiness.get("timestamp"))
            or _file_mtime_dt(results / "execution_readiness.json")
            or _utc_now()
        )
        if readiness.get("live_execution_permitted") is False:
            eid = _event_id("execution_readiness", "live_blocked", ts.isoformat())
            if eid not in seen:
                seen.add(eid)
                events.append(
                    ActivityEvent(
                        event_id=eid,
                        occurred_at=ts.isoformat(),
                        source="execution_readiness",
                        category="SAFETY",
                        severity="INFO",
                        title="Live execution remains blocked",
                        message="execution_readiness.json evidences live_execution_permitted=false.",
                        evidence_path=str(results / "execution_readiness.json"),
                        raw_reference="live_execution_permitted=false",
                    )
                )
    gov = _read_json(results / "governance_authorization.json")
    if gov and not gov.get("__malformed__"):
        ts = (
            _parse_dt(gov.get("generated_at") or gov.get("timestamp"))
            or _file_mtime_dt(results / "governance_authorization.json")
            or _utc_now()
        )
        status = str(gov.get("overall_authorization") or gov.get("authorization_status") or "")
        if status and status.upper() not in ("AUTHORIZED", "GREEN", "APPROVED", "PASS"):
            eid = _event_id("governance", status, ts.isoformat())
            if eid not in seen:
                seen.add(eid)
                events.append(
                    ActivityEvent(
                        event_id=eid,
                        occurred_at=ts.isoformat(),
                        source="governance_authorization",
                        category="GOVERNANCE",
                        severity="BLOCK",
                        title="Governance posture is not fully authorized",
                        message=f"overall_authorization/status evidenced as {status}.",
                        evidence_path=str(results / "governance_authorization.json"),
                        raw_reference=status,
                    )
                )
    return events


def collect_activity_events(
    *,
    root: Optional[Path] = None,
    since_hours: Optional[float] = 72.0,
    max_events: int = MAX_EVENTS,
) -> Dict[str, Any]:
    """Collect and normalize activity events from allowlisted artifacts (read-only)."""
    root = Path(root) if root is not None else ROOT
    results = root / "data" / "results"
    seen: Set[str] = set()
    events: List[ActivityEvent] = []
    sources_used: List[str] = []
    stale_warnings: List[str] = []

    cont_dir = results / "continuous_loop"
    cont_log = _latest_glob(cont_dir, "continuous_loop_*.log")
    if cont_log:
        sources_used.append(str(cont_log))
        events.extend(_parse_continuous_log(cont_log, seen))
        age_h = (
            _utc_now() - datetime.fromtimestamp(cont_log.stat().st_mtime, tz=timezone.utc)
        ).total_seconds() / 3600
        if age_h > 24:
            stale_warnings.append(f"Continuous log older than 24h: {cont_log.name}")

    wd_dir = results / "watchdog_logs"
    wd_log = _latest_glob(wd_dir, "watchdog_loop_*.log") or _latest_glob(wd_dir, "watchdog_*.log")
    if wd_log:
        sources_used.append(str(wd_log))
        events.extend(_parse_watchdog_log(wd_log, seen))

    for name in ("runtime_autostart_931.json", "runtime_supervisor_autostart.json"):
        p = results / name
        if p.is_file():
            sources_used.append(str(p))
            events.extend(_events_from_startup(p, seen))

    stage = results / "paper_entry_loop_stage_audit.json"
    if stage.is_file():
        sources_used.append(str(stage))
        events.extend(_events_from_paper_entry_stage(stage, seen))

    paper_audit = results / "paper_execution_audit.csv"
    if paper_audit.is_file():
        sources_used.append(str(paper_audit))
        events.extend(_events_from_paper_execution_audit(paper_audit, seen))

    alerts = results / "watchdog_alerts.json"
    if alerts.is_file():
        sources_used.append(str(alerts))
        events.extend(_events_from_watchdog_alerts(alerts, seen))

    freshness = results / "data_freshness_refresh_audit.json"
    if freshness.is_file():
        sources_used.append(str(freshness))
        events.extend(_events_from_freshness(freshness, seen))

    events.extend(_events_from_readiness_governance(results, seen))
    for p in (results / "execution_readiness.json", results / "governance_authorization.json"):
        if p.is_file() and str(p) not in sources_used:
            sources_used.append(str(p))

    # Time filter
    cutoff = None
    if since_hours is not None:
        cutoff = _utc_now() - timedelta(hours=float(since_hours))

    filtered: List[ActivityEvent] = []
    for ev in events:
        ts = _parse_dt(ev.occurred_at)
        if cutoff and ts and ts < cutoff:
            continue
        filtered.append(ev)

    # Sort newest first
    filtered.sort(key=lambda e: e.occurred_at or "", reverse=True)
    filtered = filtered[:max_events]

    return {
        "generated_at": _utc_now().isoformat(),
        "event_count": len(filtered),
        "events": [e.to_dict() for e in filtered],
        "sources_used": sources_used,
        "stale_source_warnings": stale_warnings,
        "categories": list(CATEGORIES),
        "severities": list(SEVERITIES),
        "evidence_rules": [
            "exit_code=0 means stage success only — never an order event by itself",
            "cycle start does not prove cycle completion",
            "planned/authorized does not prove submitted",
            "submitted does not prove filled",
            "missing evidence displays as unavailable — never fabricated",
        ],
        "mutates_sources": False,
    }


def filter_activity_events(
    payload: Dict[str, Any],
    *,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    source: Optional[str] = None,
    query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    events = list(payload.get("events") or [])
    q = (query or "").strip().lower()
    out = []
    for ev in events:
        if category and category != "ALL" and ev.get("category") != category:
            continue
        if severity and severity != "ALL" and ev.get("severity") != severity:
            continue
        if source and source != "ALL" and ev.get("source") != source:
            continue
        if q:
            blob = " ".join(
                str(ev.get(k) or "")
                for k in ("title", "message", "symbol", "order_id", "raw_reference", "source")
            ).lower()
            if q not in blob:
                continue
        out.append(ev)
    return out
