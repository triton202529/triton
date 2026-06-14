"""
execution_governance_council.py
THP-008 — Investment committee-grade governance authority model for execution gate.

Determines authority levels and council status without modifying core gate logic.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.execution_gate_degradation_manager import load_operational_status
from services.execution_gate_evidence_registry import load_decisions
from services.execution_gate_health_monitor import load_health_report
from services.triton_execution_gate import EXECUTION_PATHS

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DIR = ROOT / "data" / "governance"
RESULTS_DIR = ROOT / "data" / "results"

AUTHORITY_LEVELS_JSON = GOVERNANCE_DIR / "execution_authority_levels.json"
COUNCIL_STATUS_JSON = GOVERNANCE_DIR / "execution_governance_council_status.json"
AUTHORITY_MATRIX_JSON = GOVERNANCE_DIR / "execution_gate_authority_matrix.json"

LEVEL_NAMES = {
    0: "FULL_AUTOMATION",
    1: "OPERATOR_REVIEW_REQUIRED",
    2: "GOVERNANCE_COUNCIL_APPROVAL",
    3: "EMERGENCY_AUTHORITY",
}

EMERGENCY_PATHS = {
    EXECUTION_PATHS["RISK_EMERGENCY_ACTION"],
    EXECUTION_PATHS["POSITION_EXIT"],
}

COUNCIL_MANDATORY_PATHS = {
    EXECUTION_PATHS["LIVE_EXECUTION"],
    EXECUTION_PATHS["CAPITAL_DEPLOYMENT"],
    EXECUTION_PATHS["GOVERNANCE_OVERRIDE"],
    EXECUTION_PATHS["MANUAL_OPERATOR_OVERRIDE"],
    EXECUTION_PATHS["FORCED_ROTATION"],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def load_authority_levels() -> Dict[str, Any]:
    if not AUTHORITY_LEVELS_JSON.is_file():
        return {}
    try:
        return json.loads(AUTHORITY_LEVELS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_risk_state() -> Dict[str, Any]:
    path = RESULTS_DIR / "adaptive_risk_state.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _known_paths_from_matrix() -> set[str]:
    if not AUTHORITY_MATRIX_JSON.is_file():
        return set()
    try:
        doc = json.loads(AUTHORITY_MATRIX_JSON.read_text(encoding="utf-8"))
        return set((doc.get("statistics") or {}).get("by_path") or {})
    except Exception:
        return set()


def _parse_exposure(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def determine_authority_level(
    *,
    path: str,
    context: Optional[Dict[str, Any]] = None,
    evaluation: Optional[Dict[str, Any]] = None,
    capital_exposure: Optional[Any] = None,
    action_requested: Optional[str] = None,
) -> Tuple[int, List[str]]:
    """
    Return (authority_level, triggering_conditions).
    Higher level wins when multiple triggers apply.
    """
    ctx = context or {}
    eval_doc = evaluation or {}
    levels_doc = load_authority_levels()
    thresholds = levels_doc.get("thresholds") or {}
    triggers: List[str] = []
    level = 0

    operational = load_operational_status()
    health = load_health_report()
    op_state = str(operational.get("operational_state") or "MONITOR")
    health_class = str(health.get("health_classification") or "GREEN")
    anomaly_count = int(operational.get("anomaly_count") or 0)

    emergency_action = str(
        action_requested or ctx.get("emergencyAction") or ctx.get("emergency_action") or ""
    ).upper()
    if path in EMERGENCY_PATHS or emergency_action in (
        "FORCE_EXIT",
        "FORCE_TRIM",
        "RISK_LOCKDOWN",
    ):
        triggers.append("emergency_authority_path")
        return 3, triggers

    exposure = _parse_exposure(capital_exposure or ctx.get("capitalExposure"))
    review_threshold = float(thresholds.get("capital_exposure_review_usd") or 25000)
    council_threshold = float(thresholds.get("capital_exposure_council_usd") or 100000)

    if path in COUNCIL_MANDATORY_PATHS:
        triggers.append("mandatory_council_path")
        level = max(level, 2)

    if ctx.get("liveMode") or path == EXECUTION_PATHS["LIVE_EXECUTION"]:
        triggers.append("live_capital_deployment")
        level = max(level, 2)

    if ctx.get("productionReadinessOverride"):
        triggers.append("production_authorization_change")
        level = max(level, 2)

    if exposure >= council_threshold:
        triggers.append("major_capital_exposure")
        level = max(level, 2)
    elif exposure >= review_threshold:
        triggers.append("unusual_capital_exposure")
        level = max(level, 1)

    known_paths = _known_paths_from_matrix()
    if known_paths and path not in known_paths:
        triggers.append("new_execution_path")
        level = max(level, 1)

    if health_class in ("ORANGE", "RED") or op_state in ("RESTRICT", "LOCKDOWN"):
        if op_state == "LOCKDOWN":
            triggers.append("operational_lockdown")
            level = max(level, 2)
        else:
            triggers.append("elevated_risk_environment")
            level = max(level, 1)

    if anomaly_count >= int(thresholds.get("anomaly_rate_review") or 3):
        triggers.append("increased_anomaly_rate")
        level = max(level, 1)

    risk_state = _load_risk_state()
    if str(risk_state.get("lockdown") or risk_state.get("risk_lockdown", "")).lower() in (
        "true",
        "1",
        "active",
    ):
        triggers.append("risk_lockdown")
        level = max(level, 3)

    conc_pct = _parse_exposure(risk_state.get("largest_position_pct"))
    if conc_pct >= float(thresholds.get("concentration_review_pct") or 35):
        triggers.append("portfolio_concentration_breach")
        level = max(level, 1)

    cert_level = str(operational.get("certification_level") or "")
    if cert_level in ("FAILED", "REQUIRES_IMPROVEMENT"):
        triggers.append("certification_failure")
        level = max(level, 2)

    if path in (
        EXECUTION_PATHS["GOVERNANCE_OVERRIDE"],
        EXECUTION_PATHS["MANUAL_OPERATOR_OVERRIDE"],
    ):
        triggers.append("safeguard_override_attempt")
        level = max(level, 2)

    if eval_doc.get("decision") == "REVIEW_REQUIRED" and level < 1:
        triggers.append("gate_review_required")
        level = 1

    return level, triggers


def build_council_status() -> Dict[str, Any]:
    operational = load_operational_status()
    health = load_health_report()
    levels_doc = load_authority_levels()
    decisions = load_decisions()

    pending_from_decisions = sum(
        1 for d in decisions if str(d.get("execution_gate_decision")) == "REVIEW_REQUIRED"
    )

    automation_count = sum(
        1
        for d in decisions
        if d.get("allowed") and str(d.get("authority_classification")) == "automatic_approval"
    )
    total = len(decisions) or 1

    status = {
        "generated_at": _utc_now_iso(),
        "framework": "THP-008",
        "current_authority_mode": _resolve_council_mode(operational, health),
        "operational_state": operational.get("operational_state", "MONITOR"),
        "health_score": health.get("health_score"),
        "health_classification": health.get("health_classification"),
        "certification_level": operational.get("certification_level"),
        "resilience_score": operational.get("resilience_score"),
        "active_restrictions": operational.get("restrictions") or {},
        "paper_live_boundary_intact": operational.get("paper_live_boundary_intact", True),
        "authority_levels": levels_doc.get("levels") or [],
        "analytics": {
            "total_decisions": len(decisions),
            "automation_rate_pct": round(100.0 * automation_count / total, 2),
            "human_intervention_rate_pct": round(100.0 * pending_from_decisions / total, 2),
            "review_required_count": pending_from_decisions,
        },
        "council_ready": operational.get("gate_available", False)
        and operational.get("paper_live_boundary_intact", True),
    }
    _atomic_write_json(COUNCIL_STATUS_JSON, status)
    return status


def _resolve_council_mode(operational: Dict[str, Any], health: Dict[str, Any]) -> str:
    op_state = str(operational.get("operational_state") or "MONITOR")
    health_class = str(health.get("health_classification") or "GREEN")
    if op_state == "LOCKDOWN":
        return "EMERGENCY_ONLY"
    if op_state in ("RESTRICT", "WARNING") or health_class in ("ORANGE", "RED", "YELLOW"):
        return "ELEVATED_OVERSIGHT"
    return "NORMAL_AUTOMATION"


def run_governance_council() -> Dict[str, Any]:
    return build_council_status()


def load_council_status() -> Dict[str, Any]:
    if not COUNCIL_STATUS_JSON.is_file():
        return {}
    try:
        return json.loads(COUNCIL_STATUS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def authority_level_name(level: int) -> str:
    return LEVEL_NAMES.get(level, "UNKNOWN")


if __name__ == "__main__":
    status = run_governance_council()
    print(json.dumps({"mode": status.get("current_authority_mode")}, indent=2))
