"""
TRITON Institutional Memory — Phases 34–36.

Institutional memory, knowledge graph, and organizational learning engines.
Paper-mode / simulation only. NO live trading, orders, or portfolio changes.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from services.institutional_protection import _atomic_write_json, _iso_utc

MAX_MEMORY_ENTRIES = 500
MAX_GRAPH_NODES = 200
MAX_GRAPH_EDGES = 400
MAX_HISTORY_ROWS = 5000

MAJOR_EVENT_TYPES = frozenset(
    {
        "ALERT_RAISED",
        "ALERT_RESOLVED",
        "ESCALATION_CHANGE",
        "GOVERNOR_POSTURE_CHANGE",
        "CERTIFICATION_REVIEW",
        "GOVERNANCE_DECISION",
        "COMMITTEE_OVERSIGHT",
        "BOARD_OVERSIGHT",
        "STRATEGIC_OVERSIGHT",
        "AUTHORIZATION_GATE",
        "AUDIT_EVENT",
    }
)

HISTORY_FIELDNAMES = [
    "fingerprint",
    "timestamp",
    "event_type",
    "event_source",
    "summary",
    "severity",
]


def _entry_fingerprint(entry: Dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(entry.get("event_type") or ""),
            str(entry.get("event_source") or ""),
            str(entry.get("summary") or ""),
            str(entry.get("timestamp") or "")[:19],
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _load_history_entries(path: Path) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(path)
    entries: List[Dict[str, Any]] = []
    for row in rows:
        entries.append(
            {
                "timestamp": row.get("timestamp"),
                "event_type": row.get("event_type"),
                "event_source": row.get("event_source"),
                "summary": row.get("summary"),
                "severity": row.get("severity") or None,
                "fingerprint": row.get("fingerprint"),
            }
        )
    return entries


def _merge_memory_entries(
    historical: List[Dict[str, Any]],
    current: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    merged: List[Dict[str, Any]] = []

    for entry in historical + current:
        fp = entry.get("fingerprint") or _entry_fingerprint(entry)
        if fp in seen:
            continue
        seen.add(fp)
        clean = {k: v for k, v in entry.items() if k != "fingerprint"}
        merged.append(clean)

    merged.sort(key=lambda e: str(e.get("timestamp") or ""), reverse=True)
    return merged[:MAX_HISTORY_ROWS]


def _append_memory_history(path: Path, entries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDNAMES)
        if not exists:
            writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "fingerprint": entry.get("fingerprint") or _entry_fingerprint(entry),
                    "timestamp": entry.get("timestamp"),
                    "event_type": entry.get("event_type"),
                    "event_source": entry.get("event_source"),
                    "summary": entry.get("summary"),
                    "severity": entry.get("severity") or "",
                }
            )


def _detect_escalation_changes(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    prev_state: Optional[str] = None
    for row in rows:
        state = str(row.get("escalation_state") or "").upper()
        ts = row.get("timestamp")
        if not ts or not state:
            continue
        if prev_state is not None and state != prev_state:
            events.append(
                {
                    "timestamp": ts,
                    "event_type": "ESCALATION_CHANGE",
                    "event_source": "capital_preservation_escalation",
                    "summary": f"Escalation changed {prev_state} → {state}",
                    "severity": state,
                }
            )
        prev_state = state
    return events


def _detect_governor_changes(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    prev_posture: Optional[str] = None
    for row in rows:
        posture = str(row.get("preservation_posture") or "").upper()
        ts = row.get("timestamp")
        if not ts or not posture:
            continue
        if prev_posture is not None and posture != prev_posture:
            events.append(
                {
                    "timestamp": ts,
                    "event_type": "GOVERNOR_POSTURE_CHANGE",
                    "event_source": "capital_preservation_governor",
                    "summary": f"Governor posture changed {prev_posture} → {posture}",
                    "severity": posture,
                }
            )
        prev_posture = posture
    return events


def _detect_cert_reviews(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    prev_status: Optional[str] = None
    for row in rows:
        status = str(row.get("certification_status") or "")
        ts = row.get("timestamp")
        score = row.get("certification_score")
        if not ts or not status:
            continue
        if prev_status is None or status != prev_status:
            events.append(
                {
                    "timestamp": ts,
                    "event_type": "CERTIFICATION_REVIEW",
                    "event_source": "capital_preservation_certification",
                    "summary": f"Certification status {status} (score={score})",
                    "severity": status,
                }
            )
        prev_status = status
    return events


def _detect_strategic_oversight(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    prev_status: Optional[str] = None
    for row in rows:
        status = str(row.get("oversight_status") or "")
        ts = row.get("timestamp")
        if not ts or not status:
            continue
        if prev_status is None or status != prev_status:
            events.append(
                {
                    "timestamp": ts,
                    "event_type": "STRATEGIC_OVERSIGHT",
                    "event_source": "strategic_oversight",
                    "summary": f"Strategic oversight status {status}",
                    "severity": status,
                }
            )
        prev_status = status
    return events


def compute_institutional_memory(
    *,
    results_dir: Path,
    alerts_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    committee_doc: Dict[str, Any],
    board_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
    audit_doc: Dict[str, Any],
    cpa_doc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 34: persistent organizational memory from current cycle + history."""
    ts = _iso_utc()
    results_dir = Path(results_dir)
    current: List[Dict[str, Any]] = []

    for alert in alerts_doc.get("active_alerts") or []:
        alert_type = str(alert.get("alert_type") or "ALERT")
        subject = alert.get("subject") or alert.get("alert_id") or ""
        current.append(
            {
                "timestamp": alert.get("first_seen") or ts,
                "event_type": "ALERT_RAISED",
                "event_source": "watchdog_alerts",
                "summary": f"Active alert {alert_type}" + (f" ({subject})" if subject else ""),
                "severity": alert.get("severity"),
            }
        )

    for alert in alerts_doc.get("resolved_alerts") or []:
        alert_type = str(alert.get("alert_type") or "ALERT")
        current.append(
            {
                "timestamp": alert.get("resolved_at") or alert.get("first_seen") or ts,
                "event_type": "ALERT_RESOLVED",
                "event_source": "watchdog_alerts",
                "summary": f"Resolved alert {alert_type}",
                "severity": alert.get("severity"),
            }
        )

    esc_state = str(cpe_doc.get("escalation_state") or "GREEN").upper()
    reasons = cpe_doc.get("escalation_reason") or []
    current.append(
        {
            "timestamp": cpe_doc.get("generated_at") or ts,
            "event_type": "ESCALATION_CHANGE" if reasons else "ESCALATION_CHANGE",
            "event_source": "capital_preservation_escalation",
            "summary": f"Current escalation {esc_state}"
            + (f" — {', '.join(str(r) for r in reasons[:3])}" if reasons else ""),
            "severity": esc_state,
        }
    )

    posture = str(governor_doc.get("preservation_posture") or "UNKNOWN").upper()
    current.append(
        {
            "timestamp": governor_doc.get("generated_at") or ts,
            "event_type": "GOVERNOR_POSTURE_CHANGE",
            "event_source": "capital_preservation_governor",
            "summary": f"Governor posture {posture} (confidence={governor_doc.get('governor_confidence')}%)",
            "severity": posture,
        }
    )

    cert_status = str(cert_doc.get("certification_status") or "UNKNOWN")
    current.append(
        {
            "timestamp": cert_doc.get("generated_at") or ts,
            "event_type": "CERTIFICATION_REVIEW",
            "event_source": "capital_preservation_certification",
            "summary": f"Certification {cert_status} score={cert_doc.get('certification_score')}",
            "severity": cert_status,
        }
    )

    gov_level = str(gov_doc.get("governance_awareness_level") or "UNKNOWN")
    current.append(
        {
            "timestamp": gov_doc.get("generated_at") or ts,
            "event_type": "GOVERNANCE_DECISION",
            "event_source": "governance_risk_summary",
            "summary": f"Governance awareness {gov_level}",
            "severity": gov_level,
        }
    )

    if not auth_doc.get("overall_authorization"):
        failed_gates = [
            k for k, v in (auth_doc.get("gate_reasons") or {}).items() if "block" in str(v).lower()
        ]
        current.append(
            {
                "timestamp": auth_doc.get("generated_at") or ts,
                "event_type": "AUTHORIZATION_GATE",
                "event_source": "governance_authorization",
                "summary": "Authorization blocked"
                + (f" ({', '.join(failed_gates)})" if failed_gates else ""),
                "severity": "BLOCKED",
            }
        )

    committee_status = str(committee_doc.get("committee_status") or "")
    if committee_status:
        concerns = committee_doc.get("top_concerns") or []
        current.append(
            {
                "timestamp": committee_doc.get("generated_at") or ts,
                "event_type": "COMMITTEE_OVERSIGHT",
                "event_source": "risk_committee_oversight",
                "summary": f"Committee {committee_status}"
                + (f" — top concern: {concerns[0]}" if concerns else ""),
                "severity": committee_status,
            }
        )

    board_status = str(board_doc.get("board_status") or "")
    if board_status:
        current.append(
            {
                "timestamp": board_doc.get("generated_at") or ts,
                "event_type": "BOARD_OVERSIGHT",
                "event_source": "preservation_governance_board",
                "summary": f"Board {board_status} (confidence={board_doc.get('governance_confidence')}%)",
                "severity": board_status,
            }
        )

    oversight_status = str(strategic_doc.get("oversight_status") or "")
    if oversight_status:
        current.append(
            {
                "timestamp": strategic_doc.get("generated_at") or ts,
                "event_type": "STRATEGIC_OVERSIGHT",
                "event_source": "strategic_oversight",
                "summary": f"Strategic oversight {oversight_status}",
                "severity": oversight_status,
            }
        )

    latest_audit = audit_doc.get("latest_event_type")
    if latest_audit:
        current.append(
            {
                "timestamp": audit_doc.get("generated_at") or ts,
                "event_type": "AUDIT_EVENT",
                "event_source": "capital_preservation_audit",
                "summary": f"Latest audit {latest_audit} → {audit_doc.get('latest_event_result')}",
                "severity": str(audit_doc.get("latest_event_result") or ""),
            }
        )

    if cpa_doc:
        for adv in (cpa_doc.get("advisories") or [])[:5]:
            current.append(
                {
                    "timestamp": cpa_doc.get("generated_at") or ts,
                    "event_type": "ADVISORY",
                    "event_source": "capital_preservation_advisory",
                    "summary": str(adv.get("title") or adv.get("issue") or "Advisory issued"),
                    "severity": adv.get("priority"),
                }
            )

    current.extend(
        _detect_escalation_changes(
            _read_csv_rows(results_dir / "capital_preservation_escalation_history.csv")
        )
    )
    current.extend(
        _detect_governor_changes(
            _read_csv_rows(results_dir / "capital_preservation_governor_history.csv")
        )
    )
    current.extend(
        _detect_cert_reviews(
            _read_csv_rows(results_dir / "capital_preservation_certification_history.csv")
        )
    )
    current.extend(
        _detect_strategic_oversight(_read_csv_rows(results_dir / "strategic_oversight_history.csv"))
    )

    history_path = results_dir / "institutional_memory_history.csv"
    historical = _load_history_entries(history_path)
    all_entries = _merge_memory_entries(historical, current)

    new_entries = []
    hist_fps = {e.get("fingerprint") or _entry_fingerprint(e) for e in historical}
    for entry in current:
        fp = _entry_fingerprint(entry)
        if fp not in hist_fps:
            tagged = dict(entry)
            tagged["fingerprint"] = fp
            new_entries.append(tagged)

    if new_entries:
        _append_memory_history(history_path, new_entries)

    last_major = "NONE"
    for entry in all_entries:
        et = str(entry.get("event_type") or "")
        if et in MAJOR_EVENT_TYPES:
            last_major = et
            break

    return {
        "generated_at": ts,
        "memory_entries": len(all_entries),
        "last_major_event": last_major,
        "retention_status": "ACTIVE",
        "entries": all_entries[:MAX_MEMORY_ENTRIES],
        "disclaimer": "Institutional memory is observational only. No actions executed.",
    }


def _node_id(node_type: str, label: str) -> str:
    slug = hashlib.md5(f"{node_type}:{label}".encode("utf-8")).hexdigest()[:10]
    return f"{node_type.lower()}_{slug}"


def _add_node(
    nodes: Dict[str, Dict[str, Any]],
    node_type: str,
    label: str,
    area: str = "General",
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    nid = _node_id(node_type, label)
    if nid not in nodes:
        nodes[nid] = {
            "id": nid,
            "type": node_type,
            "label": label,
            "area": area,
            **(meta or {}),
        }
    return nid


def _add_edge(
    edges: List[Dict[str, Any]],
    source: str,
    target: str,
    edge_type: str,
    seen: Set[Tuple[str, str, str]],
) -> None:
    key = (source, target, edge_type)
    if key in seen:
        return
    seen.add(key)
    edges.append({"source": source, "target": target, "type": edge_type})


def compute_institutional_knowledge_graph(
    *,
    alerts_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpa_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    committee_doc: Dict[str, Any],
    board_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
    decision_doc: Dict[str, Any],
    intelligence_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    audit_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 35: map institutional component relationships as nodes and edges."""
    ts = _iso_utc()
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    edge_seen: Set[Tuple[str, str, str]] = set()

    esc_label = str(cpe_doc.get("escalation_state") or "GREEN")
    esc_id = _add_node(nodes, "Escalation", esc_label, "Capital Preservation")
    gov_posture = str(governor_doc.get("preservation_posture") or "UNKNOWN")
    gov_id = _add_node(nodes, "Governor", gov_posture, "Capital Preservation")
    _add_edge(edges, esc_id, gov_id, "INFORMS", edge_seen)

    for alert in (alerts_doc.get("active_alerts") or [])[:20]:
        atype = str(alert.get("alert_type") or "ALERT")
        aid = _add_node(
            nodes,
            "Alert",
            atype,
            "Monitoring",
            {"severity": alert.get("severity")},
        )
        _add_edge(edges, aid, esc_id, "TRIGGERS", edge_seen)
        _add_edge(edges, aid, gov_id, "ESCALATES_TO", edge_seen)

    for adv in (cpa_doc.get("advisories") or [])[:10]:
        title = str(adv.get("title") or adv.get("issue") or "Advisory")[:60]
        adv_id = _add_node(nodes, "Advisory", title, "Capital Preservation")
        _add_edge(edges, adv_id, esc_id, "INFORMS", edge_seen)

    cert_status = str(cert_doc.get("certification_status") or "UNKNOWN")
    cert_root = _add_node(nodes, "Certification", cert_status, "Certification")
    for area_name, area in (cert_doc.get("areas") or {}).items():
        if not isinstance(area, dict):
            continue
        certified = bool(area.get("certified"))
        area_id = _add_node(
            nodes,
            "Certification",
            str(area_name),
            "Certification",
            {"certified": certified, "score": area.get("score")},
        )
        rel = "SUPPORTS" if certified else "BLOCKS"
        _add_edge(edges, area_id, cert_root, rel, edge_seen)

    gov_level = str(gov_doc.get("governance_awareness_level") or "UNKNOWN")
    gov_node = _add_node(nodes, "Governance", gov_level, "Governance")
    auth_label = "AUTHORIZED" if auth_doc.get("overall_authorization") else "BLOCKED"
    auth_id = _add_node(nodes, "Decision", auth_label, "Governance")
    _add_edge(
        edges, gov_node, auth_id, "BLOCKS" if auth_label == "BLOCKED" else "SUPPORTS", edge_seen
    )

    readiness_status = str(readiness_doc.get("readiness_status") or "NOT_READY")
    ready_id = _add_node(nodes, "Decision", readiness_status, "Governance")
    _add_edge(
        edges, ready_id, auth_id, "BLOCKS" if readiness_status != "READY" else "SUPPORTS", edge_seen
    )
    for check in (readiness_doc.get("failed_checks") or [])[:6]:
        chk_id = _add_node(nodes, "Decision", str(check), "Governance")
        _add_edge(edges, chk_id, ready_id, "BLOCKS", edge_seen)

    committee_status = str(committee_doc.get("committee_status") or "UNKNOWN")
    comm_id = _add_node(nodes, "Committee", committee_status, "Oversight")
    _add_edge(edges, comm_id, gov_node, "INFORMS", edge_seen)
    for concern in (committee_doc.get("top_concerns") or [])[:4]:
        c_id = _add_node(nodes, "Oversight", str(concern)[:50], "Oversight")
        _add_edge(edges, c_id, comm_id, "INFORMS", edge_seen)

    board_status = str(board_doc.get("board_status") or "UNKNOWN")
    board_id = _add_node(nodes, "Committee", f"Board:{board_status}", "Oversight")
    _add_edge(edges, board_id, comm_id, "SUPPORTS", edge_seen)
    _add_edge(edges, board_id, cert_root, "INFORMS", edge_seen)

    oversight_status = str(strategic_doc.get("oversight_status") or "UNKNOWN")
    strat_id = _add_node(nodes, "Oversight", oversight_status, "Oversight")
    _add_edge(edges, strat_id, board_id, "INFORMS", edge_seen)

    dq_score = decision_doc.get("decision_quality_score")
    dq_id = _add_node(
        nodes,
        "Decision",
        f"DecisionQuality:{dq_score}",
        "Intelligence",
    )
    intel_score = intelligence_doc.get("institutional_intelligence_score")
    intel_id = _add_node(
        nodes,
        "Decision",
        f"Intelligence:{intel_score}",
        "Intelligence",
    )
    _add_edge(edges, dq_id, intel_id, "INFORMS", edge_seen)
    for area in (intelligence_doc.get("areas") or [])[:6]:
        a_id = _add_node(
            nodes,
            "Decision",
            str(area.get("area")),
            "Intelligence",
            {"score": area.get("score")},
        )
        _add_edge(edges, a_id, intel_id, "SUPPORTS", edge_seen)

    audit_type = str(audit_doc.get("latest_event_type") or "AUDIT")
    audit_id = _add_node(nodes, "Alert", audit_type, "Monitoring")
    _add_edge(edges, audit_id, gov_id, "INFORMS", edge_seen)

    node_list = list(nodes.values())
    total_nodes = len(node_list)
    total_edges = len(edges)

    area_counts: Counter[str] = Counter(n.get("area", "General") for n in node_list)
    most_connected = area_counts.most_common(1)[0][0] if area_counts else "General"

    return {
        "generated_at": ts,
        "nodes": total_nodes,
        "relationships": total_edges,
        "most_connected_area": most_connected,
        "graph": {
            "nodes": node_list[:MAX_GRAPH_NODES],
            "edges": edges[:MAX_GRAPH_EDGES],
        },
        "truncated": total_nodes > MAX_GRAPH_NODES or total_edges > MAX_GRAPH_EDGES,
        "disclaimer": "Knowledge graph is diagnostic only. No execution permitted.",
    }


def _count_recurring(values: List[str], min_count: int = 2) -> List[str]:
    counts = Counter(v for v in values if v)
    return [k for k, c in counts.most_common(10) if c >= min_count]


def compute_organizational_learning(
    *,
    results_dir: Path,
    memory_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    committee_doc: Dict[str, Any],
    board_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    decision_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase 36: pattern analysis from memory, history, and governance themes."""
    ts = _iso_utc()
    results_dir = Path(results_dir)

    cert_history = _read_csv_rows(results_dir / "capital_preservation_certification_history.csv")
    cert_blockers: List[str] = []
    for row in cert_history:
        if str(row.get("certification_status") or "").upper() not in ("CERTIFIED",):
            cert_blockers.append(str(row.get("certification_status") or "UNKNOWN"))
    for req in cert_doc.get("failed_requirements") or []:
        cert_blockers.append(str(req))

    uncertified_areas = [
        name
        for name, area in (cert_doc.get("areas") or {}).items()
        if isinstance(area, dict) and not area.get("certified")
    ]
    cert_blockers.extend(uncertified_areas)

    governance_concerns: List[str] = []
    for doc in (committee_doc, board_doc, strategic_doc, gov_doc):
        for key in (
            "top_concerns",
            "board_recommendations",
            "top_strategic_concerns",
            "governance_drivers",
        ):
            for item in doc.get(key) or []:
                governance_concerns.append(str(item))
    gov_level = str(gov_doc.get("governance_awareness_level") or "")
    if gov_level:
        governance_concerns.append(gov_level)

    readiness_history = _read_csv_rows(results_dir / "execution_readiness_history.csv")
    readiness_failures: List[str] = list(readiness_doc.get("failed_checks") or [])
    for row in readiness_history:
        status = str(row.get("readiness_status") or "")
        if status and status != "READY":
            readiness_failures.append(status)

    memory_entries = memory_doc.get("entries") or []
    alert_events = [e for e in memory_entries if e.get("event_type") == "ALERT_RAISED"]
    alert_types = [str(e.get("summary") or "") for e in alert_events]

    repeated_failures = _count_recurring(
        cert_blockers + readiness_failures + list(improvement_doc.get("weakest_systems") or [])
    )
    if not repeated_failures and improvement_doc.get("weakest_systems"):
        repeated_failures = list(improvement_doc.get("weakest_systems") or [])[:5]

    certified_areas = [
        name
        for name, area in (cert_doc.get("areas") or {}).items()
        if isinstance(area, dict) and area.get("certified")
    ]
    repeated_strengths = _count_recurring(certified_areas, min_count=1)
    for metric in decision_doc.get("metrics") or []:
        if int(metric.get("score") or 0) >= 75:
            repeated_strengths.append(str(metric.get("metric")))

    most_common_governance = _count_recurring(governance_concerns, min_count=1)[:8]
    most_common_cert_blockers = _count_recurring(cert_blockers, min_count=1)[:8]

    lessons: List[str] = []
    for opp in (improvement_doc.get("improvement_opportunities") or [])[:5]:
        lessons.append(str(opp))
    for hint in (improvement_doc.get("highest_leverage_enhancements") or [])[:3]:
        if hint not in lessons:
            lessons.append(hint)
    if alert_types:
        lessons.append(f"Recurring alert pattern: {alert_types[0][:80]}")
    if readiness_failures:
        lessons.append(
            f"Readiness gate failures persist: {', '.join(list(dict.fromkeys(readiness_failures))[:3])}"
        )

    top_priority = str(improvement_doc.get("top_priority") or "Governance")
    top_lesson = lessons[0] if lessons else f"{top_priority} remains the primary institutional gap"
    if "Readiness" in top_priority or any(
        "readiness" in str(f).lower() for f in readiness_failures
    ):
        top_lesson = "Execution Readiness is the primary certification blocker"

    depth_signals = [
        len(memory_entries),
        len(cert_history),
        len(readiness_history),
        int(decision_doc.get("decision_quality_score") or 0),
        int(improvement_doc.get("improvement_score") or 0),
    ]
    confidence = int(min(99, max(40, sum(depth_signals) / max(1, len(depth_signals)) * 1.2)))

    return {
        "generated_at": ts,
        "top_lesson": top_lesson,
        "confidence": confidence,
        "learning_status": "ACTIVE",
        "patterns": {
            "repeated_failures": repeated_failures[:8],
            "repeated_strengths": repeated_strengths[:8],
            "most_common_governance_concerns": most_common_governance,
            "most_common_certification_blockers": most_common_cert_blockers,
            "highest_value_lessons": lessons[:10],
        },
        "top_priority": top_priority,
        "decision_quality_score": decision_doc.get("decision_quality_score"),
        "disclaimer": (
            "Organizational learning is observational pattern analysis only. "
            "No trades, orders, or portfolio modifications."
        ),
    }


def persist_institutional_memory(
    *,
    results_dir: Path,
    alerts_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    cpa_doc: Dict[str, Any],
    governor_doc: Dict[str, Any],
    cert_doc: Dict[str, Any],
    gov_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    committee_doc: Dict[str, Any],
    board_doc: Dict[str, Any],
    strategic_doc: Dict[str, Any],
    audit_doc: Dict[str, Any],
    readiness_doc: Dict[str, Any],
    decision_doc: Dict[str, Any],
    intelligence_doc: Dict[str, Any],
    improvement_doc: Dict[str, Any],
) -> Dict[str, Any]:
    """Run phases 34–36 and write JSON artifacts."""
    results_dir = Path(results_dir)

    memory_doc = compute_institutional_memory(
        results_dir=results_dir,
        alerts_doc=alerts_doc,
        cpe_doc=cpe_doc,
        governor_doc=governor_doc,
        cert_doc=cert_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        committee_doc=committee_doc,
        board_doc=board_doc,
        strategic_doc=strategic_doc,
        audit_doc=audit_doc,
        cpa_doc=cpa_doc,
    )
    _atomic_write_json(memory_doc, results_dir / "institutional_memory.json")

    graph_doc = compute_institutional_knowledge_graph(
        alerts_doc=alerts_doc,
        cpe_doc=cpe_doc,
        cpa_doc=cpa_doc,
        governor_doc=governor_doc,
        cert_doc=cert_doc,
        gov_doc=gov_doc,
        auth_doc=auth_doc,
        committee_doc=committee_doc,
        board_doc=board_doc,
        strategic_doc=strategic_doc,
        decision_doc=decision_doc,
        intelligence_doc=intelligence_doc,
        readiness_doc=readiness_doc,
        audit_doc=audit_doc,
    )
    _atomic_write_json(graph_doc, results_dir / "institutional_knowledge_graph.json")

    learning_doc = compute_organizational_learning(
        results_dir=results_dir,
        memory_doc=memory_doc,
        improvement_doc=improvement_doc,
        cert_doc=cert_doc,
        gov_doc=gov_doc,
        committee_doc=committee_doc,
        board_doc=board_doc,
        strategic_doc=strategic_doc,
        readiness_doc=readiness_doc,
        decision_doc=decision_doc,
    )
    _atomic_write_json(learning_doc, results_dir / "organizational_learning.json")

    return {
        "institutional_memory": memory_doc,
        "institutional_knowledge_graph": graph_doc,
        "organizational_learning": learning_doc,
    }
