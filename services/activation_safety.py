"""
TRITON Activation Safety — Phases 13–15 (sandbox, approval, policy).

Hypothetical defensive actions, human approval queue, protective policies.
Paper execution may be authorized via TRITON_ENABLE_PAPER_EXECUTION=1 (Phase 147).
Live automated trading remains disabled.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

APPROVAL_EXPIRY_HOURS = 24
QUEUE_STATUSES = ("PENDING_REVIEW", "APPROVED", "REJECTED", "EXPIRED")

# Phase 147: minimum policies enabled when paper execution is explicitly configured.
PAPER_DEFAULT_ENABLED_ACTIONS = frozenset(
    {
        "REVIEW_DRAWDOWN_POSITIONS",
        "REVIEW_OPERATIONAL_POSTURE",
        "REVIEW_MONITORING_HEALTH",
    }
)

ALERT_TO_CANDIDATE: Dict[str, Tuple[str, str, str]] = {
    "EXCESS_CONCENTRATION": (
        "REDUCE_CONCENTRATION",
        "Concentration Protection",
        "Reduce concentration exposure",
    ),
    "EXCESS_POSITION_DRAWDOWN": (
        "REVIEW_DRAWDOWN_POSITIONS",
        "Drawdown Protection",
        "Review drawdown positions",
    ),
    "BROKER_DISCONNECTED": (
        "REVIEW_OPERATIONAL_POSTURE",
        "Operational Protection",
        "Review broker connectivity and operational posture",
    ),
    "STALE_HEARTBEAT": (
        "REVIEW_MONITORING_HEALTH",
        "Incident Protection",
        "Review monitoring health and heartbeat coverage",
    ),
    "OPEN_ORDER_AGING": (
        "REVIEW_STALE_ORDERS",
        "Operational Protection",
        "Review stale open orders",
    ),
}

DEFAULT_POLICIES: List[Dict[str, Any]] = [
    {
        "action": "REDUCE_CONCENTRATION",
        "policy_type": "Concentration Protection",
        "enabled": False,
        "requires_human_approval": True,
        "requires_green_governance": True,
        "description": "Future concentration reduction (not active)",
    },
    {
        "action": "REVIEW_DRAWDOWN_POSITIONS",
        "policy_type": "Drawdown Protection",
        "enabled": False,
        "requires_human_approval": True,
        "requires_green_governance": False,
        "description": "Future drawdown position review workflow (not active)",
    },
    {
        "action": "REVIEW_ELEVATED_EXPOSURE",
        "policy_type": "Exposure Protection",
        "enabled": False,
        "requires_human_approval": True,
        "requires_green_governance": True,
        "description": "Future elevated exposure review (not active)",
    },
    {
        "action": "REVIEW_OPERATIONAL_POSTURE",
        "policy_type": "Operational Protection",
        "enabled": False,
        "requires_human_approval": True,
        "requires_green_governance": False,
        "description": "Future operational recovery review (not active)",
    },
    {
        "action": "REVIEW_MONITORING_HEALTH",
        "policy_type": "Incident Protection",
        "enabled": False,
        "requires_human_approval": True,
        "requires_green_governance": False,
        "description": "Future monitoring incident review (not active)",
    },
    {
        "action": "REVIEW_RISK_OFF_POSTURE",
        "policy_type": "Exposure Protection",
        "enabled": False,
        "requires_human_approval": True,
        "requires_green_governance": True,
        "description": "Future risk-off posture review (not active)",
    },
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: Optional[datetime] = None) -> str:
    t = dt or _utc_now()
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _log(tag: str, message: str) -> None:
    print(f"[{tag}] {message}")


def _is_paper_execution_enabled() -> bool:
    val = os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _request_fingerprint(action: str, subject: Optional[str]) -> str:
    raw = f"{action}|{subject or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _estimate_risk_reduction(
    alert_type: str,
    details: Dict[str, Any],
    sim_doc: Optional[Dict[str, Any]],
) -> float:
    if sim_doc:
        for sim in sim_doc.get("simulations") or []:
            if (
                sim.get("simulation_type") == "concentration_cap"
                and alert_type == "EXCESS_CONCENTRATION"
            ):
                return float(sim.get("risk_score_delta") or 0.0)
            if (
                sim.get("simulation_type") == "drawdown_protection"
                and alert_type == "EXCESS_POSITION_DRAWDOWN"
            ):
                return float(sim.get("risk_score_delta") or 0.0)
    if alert_type == "EXCESS_CONCENTRATION":
        pct = _safe_float(details.get("portfolio_pct"), 0.0) or 0.0
        return round(min(25.0, max(5.0, (pct - 25.0) * 0.4)), 1)
    if alert_type == "EXCESS_POSITION_DRAWDOWN":
        return 12.0
    if alert_type == "BROKER_DISCONNECTED":
        return 20.0
    if alert_type == "STALE_HEARTBEAT":
        return 10.0
    return 8.0


def compute_defensive_action_candidates(
    *,
    positions: List[Dict[str, Any]],
    active_alerts: List[Dict[str, Any]],
    cpe_doc: Dict[str, Any],
    cpi_doc: Dict[str, Any],
    sim_doc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 13: hypothetical defensive action candidates (simulation only)."""
    ts = _iso_utc()
    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for alert in active_alerts:
        if not isinstance(alert, dict):
            continue
        alert_type = str(alert.get("alert_type") or "")
        mapping = ALERT_TO_CANDIDATE.get(alert_type)
        if not mapping:
            continue
        action, policy_type, summary = mapping
        subject = alert.get("subject")
        fp = _request_fingerprint(action, str(subject) if subject else None)
        if fp in seen:
            continue
        seen.add(fp)
        details = alert.get("details") if isinstance(alert.get("details"), dict) else {}
        reason = str(alert.get("alert_type") or summary)
        if subject:
            reason = f"{subject}: {alert.get('alert_type', summary)}"
        candidates.append(
            {
                "candidate_id": fp,
                "candidate_action": action,
                "policy_type": policy_type,
                "summary": summary,
                "reason": reason,
                "subject": subject,
                "severity": alert.get("severity"),
                "estimated_risk_reduction": _estimate_risk_reduction(alert_type, details, sim_doc),
                "status": "SIMULATION_ONLY",
                "execution_permitted": False,
                "source_alert_type": alert_type,
            }
        )

    escalation = str(cpe_doc.get("escalation_state") or "")
    if escalation in ("ORANGE", "RED", "CRITICAL"):
        fp = _request_fingerprint("REVIEW_ELEVATED_EXPOSURE", escalation)
        if fp not in seen:
            candidates.append(
                {
                    "candidate_id": fp,
                    "candidate_action": "REVIEW_ELEVATED_EXPOSURE",
                    "policy_type": "Exposure Protection",
                    "summary": "Review elevated exposure",
                    "reason": f"Escalation state is {escalation}",
                    "subject": None,
                    "severity": "HIGH" if escalation in ("RED", "CRITICAL") else "MEDIUM",
                    "estimated_risk_reduction": round(
                        max(8.0, (60 - int(cpi_doc.get("capital_preservation_score") or 0)) * 0.3),
                        1,
                    ),
                    "status": "SIMULATION_ONLY",
                    "execution_permitted": False,
                    "source_alert_type": "ESCALATION_STATE",
                }
            )

    if int(cpi_doc.get("capital_preservation_score") or 100) < 60:
        fp = _request_fingerprint("REVIEW_RISK_OFF_POSTURE", "portfolio")
        if fp not in seen:
            candidates.append(
                {
                    "candidate_id": fp,
                    "candidate_action": "REVIEW_RISK_OFF_POSTURE",
                    "policy_type": "Exposure Protection",
                    "summary": "Review risk-off posture",
                    "reason": "Capital preservation score below caution threshold",
                    "subject": None,
                    "severity": "MEDIUM",
                    "estimated_risk_reduction": 15.0,
                    "status": "SIMULATION_ONLY",
                    "execution_permitted": False,
                    "source_alert_type": "LOW_CPS",
                }
            )

    return {
        "generated_at": ts,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "disclaimer": "Hypothetical actions only. No execution, orders, or portfolio changes.",
    }


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def sync_human_approval_queue(
    candidates_doc: Dict[str, Any],
    queue_path: Path,
) -> Dict[str, Any]:
    """Phase 14: merge candidates into approval queue; preserve human decisions."""
    ts = _iso_utc()
    now = _utc_now()
    existing = _read_json(queue_path)
    requests: List[Dict[str, Any]] = list(existing.get("requests") or [])

    by_fp: Dict[str, Dict[str, Any]] = {}
    for req in requests:
        if isinstance(req, dict) and req.get("candidate_fingerprint"):
            by_fp[str(req["candidate_fingerprint"])] = req

    for cand in candidates_doc.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        fp = str(cand.get("candidate_id") or "")
        if not fp or fp in by_fp:
            if fp in by_fp and by_fp[fp].get("status") == "PENDING_REVIEW":
                by_fp[fp]["last_seen_at"] = ts
                by_fp[fp]["estimated_risk_reduction"] = cand.get("estimated_risk_reduction")
                by_fp[fp]["reason"] = cand.get("reason")
            continue
        request_id = str(uuid.uuid4())
        by_fp[fp] = {
            "request_id": request_id,
            "candidate_fingerprint": fp,
            "candidate_action": cand.get("candidate_action"),
            "policy_type": cand.get("policy_type"),
            "reason": cand.get("reason"),
            "subject": cand.get("subject"),
            "estimated_risk_reduction": cand.get("estimated_risk_reduction"),
            "status": "PENDING_REVIEW",
            "created_at": ts,
            "last_seen_at": ts,
            "reviewed_at": None,
            "reviewer_note": None,
            "execution_permitted": False,
        }

    updated_requests: List[Dict[str, Any]] = []
    for req in by_fp.values():
        st = str(req.get("status") or "PENDING_REVIEW")
        if st == "PENDING_REVIEW":
            created = _parse_iso(str(req.get("created_at") or ""))
            if created and (now - created) > timedelta(hours=APPROVAL_EXPIRY_HOURS):
                req["status"] = "EXPIRED"
                req["reviewed_at"] = ts
                req["reviewer_note"] = "Auto-expired after review window"
        updated_requests.append(req)

    updated_requests.sort(
        key=lambda r: (
            {"PENDING_REVIEW": 0, "APPROVED": 1, "REJECTED": 2, "EXPIRED": 3}.get(
                str(r.get("status")), 9
            ),
            str(r.get("created_at") or ""),
        )
    )

    counts = {s: 0 for s in QUEUE_STATUSES}
    for req in updated_requests:
        st = str(req.get("status") or "")
        if st in counts:
            counts[st] += 1

    doc = {
        "generated_at": ts,
        "requests": updated_requests,
        "counts": counts,
        "disclaimer": "Approval updates queue status only. No trades or portfolio actions executed.",
    }
    _atomic_write_json(doc, queue_path)
    return doc


def set_approval_request_status(
    queue_path: Path,
    request_id: str,
    status: str,
    *,
    reviewer_note: str = "",
) -> Tuple[bool, str]:
    """
    Update approval queue status (APPROVED / REJECTED only from UI).
    Does NOT execute any trading or portfolio action.
    """
    if status not in ("APPROVED", "REJECTED"):
        return False, f"Invalid status: {status}"
    doc = _read_json(queue_path)
    requests = doc.get("requests") or []
    found = False
    ts = _iso_utc()
    for req in requests:
        if str(req.get("request_id")) == str(request_id):
            if str(req.get("status")) != "PENDING_REVIEW":
                return False, "Request is not pending review"
            req["status"] = status
            req["reviewed_at"] = ts
            req["reviewer_note"] = reviewer_note or None
            req["execution_permitted"] = False
            found = True
            break
    if not found:
        return False, "Request not found"
    counts = {s: 0 for s in QUEUE_STATUSES}
    for req in requests:
        st = str(req.get("status") or "")
        if st in counts:
            counts[st] += 1
    doc["generated_at"] = ts
    doc["requests"] = requests
    doc["counts"] = counts
    _atomic_write_json(doc, queue_path)
    return True, status


def compute_protective_action_policy(
    *,
    existing_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Phase 15 + 147: protective policies; paper execution when explicitly configured."""
    ts = _iso_utc()
    existing = _read_json(existing_path) if existing_path and existing_path.is_file() else {}
    existing_by_action = {
        str(p.get("action")): p
        for p in (existing.get("policies") or [])
        if isinstance(p, dict) and p.get("action")
    }

    requested_mode = str(existing.get("mode") or "").lower()
    if requested_mode == "live":
        _log("LIVE_EXECUTION_BLOCKED", "mode=live rejected; forcing simulation locks")
        paper_configured = False
        mode = "simulation"
    else:
        paper_configured = _is_paper_execution_enabled()
        mode = "paper" if paper_configured else "simulation"

    policies: List[Dict[str, Any]] = []
    for default in DEFAULT_POLICIES:
        action = str(default["action"])
        merged = dict(default)
        prev = existing_by_action.get(action)
        if prev:
            merged["operator_notes"] = prev.get("operator_notes")

        if paper_configured:
            merged["enabled"] = bool(
                prev.get("enabled")
                if prev and prev.get("enabled")
                else action in PAPER_DEFAULT_ENABLED_ACTIONS
            )
        else:
            merged["enabled"] = False
        merged["requires_human_approval"] = True
        policies.append(merged)

    enabled_policy_count = sum(1 for p in policies if p.get("enabled"))
    live_execution_enabled = False
    automated_trading_permitted = False

    if paper_configured and enabled_policy_count > 0:
        paper_execution_enabled = True
        global_execution_enabled = True
        _log(
            "POLICY_CONFIG",
            f"mode={mode} paper_execution_enabled=true global_execution_enabled=true "
            f"enabled_policy_count={enabled_policy_count}",
        )
        _log("PAPER_EXECUTION_AUTH", "Paper execution policy layer configured (not placing orders)")
    else:
        paper_execution_enabled = False
        global_execution_enabled = False
        _log(
            "POLICY_CONFIG",
            f"mode={mode} paper_execution_enabled=false enabled_policy_count={enabled_policy_count}",
        )
        if not paper_configured:
            _log(
                "EXECUTION_GATE_BLOCK",
                "Paper execution not configured (set TRITON_ENABLE_PAPER_EXECUTION=1 to enable)",
            )

    return {
        "generated_at": ts,
        "mode": mode,
        "global_execution_enabled": global_execution_enabled,
        "automated_trading_permitted": automated_trading_permitted,
        "paper_execution_enabled": paper_execution_enabled,
        "live_execution_enabled": live_execution_enabled,
        "enabled_policy_count": enabled_policy_count,
        "policies": policies,
        "disclaimer": (
            "Paper execution authorization only when TRITON_ENABLE_PAPER_EXECUTION=1. "
            "Live execution and automated live trading remain disabled. "
            "This layer does not place orders."
        ),
    }


def persist_activation_safety(
    *,
    results_dir: Path,
    positions: List[Dict[str, Any]],
    active_alerts: List[Dict[str, Any]],
    cpi_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    sim_doc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run phases 13–15 and write JSON artifacts."""
    results_dir = Path(results_dir)
    candidates_path = results_dir / "defensive_action_candidates.json"
    queue_path = results_dir / "human_approval_queue.json"
    policy_path = results_dir / "protective_action_policy.json"

    candidates_doc = compute_defensive_action_candidates(
        positions=positions,
        active_alerts=active_alerts,
        cpe_doc=cpe_doc,
        cpi_doc=cpi_doc,
        sim_doc=sim_doc,
    )
    _atomic_write_json(candidates_doc, candidates_path)

    queue_doc = sync_human_approval_queue(candidates_doc, queue_path)
    policy_doc = compute_protective_action_policy(existing_path=policy_path)
    _atomic_write_json(policy_doc, policy_path)

    return {
        "defensive_action_candidates": candidates_doc,
        "human_approval_queue": queue_doc,
        "protective_action_policy": policy_doc,
    }


def main() -> int:
    """CLI: regenerate protective_action_policy.json from env/config."""
    results_dir = Path(__file__).resolve().parents[1] / "data" / "results"
    policy_path = results_dir / "protective_action_policy.json"
    doc = compute_protective_action_policy(existing_path=policy_path)
    _atomic_write_json(doc, policy_path)
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
