"""
governance_degraded_mode.py
Phase 196B — ORANGE governance degraded-mode classification (paper only).

Read-only risk classification. Does not place orders or bypass dispatch approval.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

GOVERNANCE_MODE_GREEN = "GREEN"
GOVERNANCE_MODE_ORANGE_DEGRADED = "ORANGE_DEGRADED"
GOVERNANCE_MODE_RED_BLOCKED = "RED_BLOCKED"

ACTION_RISK_INCREASING = "RISK_INCREASING"
ACTION_RISK_REDUCING = "RISK_REDUCING"
ACTION_NEUTRAL = "NEUTRAL"

ORANGE_DEGRADED_REASONS = frozenset(
    {
        "EXCESS_CONCENTRATION",
        "EXCESS_POSITION_DRAWDOWN",
    }
)

GOVERNANCE_BLOCK_LEVELS = frozenset(
    {
        "MANAGEMENT_REVIEW_REQUIRED",
        "BOARD_REVIEW_REQUIRED",
        "CRITICAL_INTERVENTION",
    }
)

RISK_INCREASING_ACTIONS = frozenset({"OPEN_NEW", "ADD", "BUY"})
RISK_REDUCING_ACTIONS = frozenset({"EXIT", "TRIM", "FORCE_EXIT", "ROTATE_EXIT"})
NEUTRAL_ACTIONS = frozenset({"HOLD", "WATCH", "WAIT", "BLOCK", "NO_ACTION"})

SAFETY_STATEMENT = (
    "ORANGE degraded mode does not authorize live execution and does not bypass dispatch "
    "approval. It only preserves visibility and controlled eligibility for paper-only "
    "risk-reducing actions."
)


def _parse_iso(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _risk_flag_components(risk_flag: Any) -> Set[str]:
    raw = str(risk_flag or "OK")
    return {p.strip().upper() for p in raw.split("|") if p.strip() and p.strip().upper() != "OK"}


def classify_portfolio_action_risk(
    *,
    recommended_action: str,
    risk_flag: str = "OK",
    reason: str = "",
    blocked_new_buy_symbols: Optional[Set[str]] = None,
    symbol: str = "",
) -> str:
    """
    Read-only classification for candidate portfolio actions.
    """
    action = str(recommended_action or "").strip().upper()
    sym = str(symbol or "").strip().upper()
    risk_set = _risk_flag_components(risk_flag)
    reason_l = str(reason or "").lower()
    blocked = blocked_new_buy_symbols or set()

    if action in RISK_REDUCING_ACTIONS:
        return ACTION_RISK_REDUCING
    if "FORCE_EXIT" in risk_set or "risk_flag_contains=force_exit" in reason_l:
        return ACTION_RISK_REDUCING
    if "TRIM_PRIORITY" in risk_set or "risk_flag_contains=trim_priority" in reason_l:
        return ACTION_RISK_REDUCING
    if action in ("EXIT", "TRIM"):
        if any(
            k in reason_l
            for k in (
                "concentration",
                "drawdown",
                "high_drag",
                "force_exit",
                "trim_priority",
                "risk_flag",
            )
        ):
            return ACTION_RISK_REDUCING

    if action in RISK_INCREASING_ACTIONS:
        return ACTION_RISK_INCREASING
    if action == "BLOCK":
        return ACTION_RISK_INCREASING
    if sym and sym in blocked:
        return ACTION_RISK_INCREASING
    if "BLOCK_NEW_BUY" in risk_set:
        return ACTION_RISK_INCREASING

    if action in NEUTRAL_ACTIONS or not action:
        return ACTION_NEUTRAL
    return ACTION_NEUTRAL


def classify_management_action_risk(
    *,
    management_action: str,
    force_exit_override: bool = False,
    perf_force_exit_active: bool = False,
    trim_priority: bool = False,
    high_drag_forced: bool = False,
) -> str:
    """Classify manage_positions planned actions."""
    action = str(management_action or "").strip().upper()
    if force_exit_override or perf_force_exit_active or high_drag_forced:
        return ACTION_RISK_REDUCING
    if trim_priority and action in ("TRIM", "EXIT", "ROTATE_EXIT"):
        return ACTION_RISK_REDUCING
    if action in ("EXIT", "TRIM", "ROTATE_EXIT"):
        return ACTION_RISK_REDUCING
    if action in ("ADD", "BUY"):
        return ACTION_RISK_INCREASING
    return ACTION_NEUTRAL


def classify_governance_mode(
    *,
    gov_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    governance_authorized: bool = False,
) -> Dict[str, Any]:
    """
    Derive governance_mode without weakening capital preservation checks.
    """
    escalation = str(cpe_doc.get("escalation_state") or gov_doc.get("escalation_state") or "")
    awareness = str(gov_doc.get("governance_awareness_level") or "")
    reasons = [str(r) for r in (cpe_doc.get("escalation_reason") or []) if r]
    reason_labels = [str(r) for r in (cpe_doc.get("escalation_reason_labels") or []) if r]

    concentration_detected = "EXCESS_CONCENTRATION" in reasons or any(
        "concentration" in x.lower()
        for x in reason_labels + list(gov_doc.get("governance_drivers") or [])
    )
    drawdown_detected = "EXCESS_POSITION_DRAWDOWN" in reasons or any(
        "drawdown" in x.lower()
        for x in reason_labels + list(gov_doc.get("governance_drivers") or [])
    )

    if awareness in GOVERNANCE_BLOCK_LEVELS:
        mode = GOVERNANCE_MODE_RED_BLOCKED
        top_reason = awareness
    elif escalation in ("RED", "CRITICAL"):
        mode = GOVERNANCE_MODE_RED_BLOCKED
        top_reason = escalation
    elif escalation == "ORANGE" and (
        concentration_detected or drawdown_detected or bool(set(reasons) & ORANGE_DEGRADED_REASONS)
    ):
        mode = GOVERNANCE_MODE_ORANGE_DEGRADED
        top_reason = reasons[0] if reasons else "ORANGE_CONCENTRATION_DRAWDOWN"
    elif governance_authorized and escalation in ("GREEN", "YELLOW"):
        mode = GOVERNANCE_MODE_GREEN
        top_reason = escalation or "GREEN"
    elif escalation == "ORANGE":
        mode = GOVERNANCE_MODE_RED_BLOCKED
        top_reason = "ORANGE_NON_DEGRADABLE"
    else:
        mode = GOVERNANCE_MODE_RED_BLOCKED if not governance_authorized else GOVERNANCE_MODE_GREEN
        top_reason = escalation or awareness or "UNKNOWN"

    risk_increasing_blocked = (
        mode in (GOVERNANCE_MODE_ORANGE_DEGRADED, GOVERNANCE_MODE_RED_BLOCKED) or drawdown_detected
    )
    risk_reducing_paper_allowed = mode == GOVERNANCE_MODE_ORANGE_DEGRADED

    return {
        "governance_mode": mode,
        "escalation_state": escalation,
        "governance_awareness_level": awareness,
        "degraded_mode_reason": [r for r in reasons if r in ORANGE_DEGRADED_REASONS]
        or ([top_reason] if mode == GOVERNANCE_MODE_ORANGE_DEGRADED else []),
        "concentration_detected": concentration_detected,
        "drawdown_detected": drawdown_detected,
        "risk_increasing_actions_blocked": risk_increasing_blocked,
        "risk_reducing_paper_actions_allowed": risk_reducing_paper_allowed,
        "top_reason": top_reason,
    }


def log_governance_degraded_mode(classification: Dict[str, Any]) -> None:
    mode = classification.get("governance_mode")
    if mode != GOVERNANCE_MODE_ORANGE_DEGRADED:
        return
    print(
        "[GOVERNANCE_DEGRADED_MODE]\n"
        f"mode={mode}\n"
        f"risk_increasing_actions_blocked={classification.get('risk_increasing_actions_blocked')}\n"
        f"risk_reducing_paper_actions_allowed={classification.get('risk_reducing_paper_actions_allowed')}\n"
        "live_execution_permitted=false",
        flush=True,
    )


def log_risk_reducing_lane(
    *,
    symbol: str,
    action: str,
    eligible_for_controlled_paper_handling: bool = True,
    dispatch_approval_required: bool = True,
) -> None:
    print(
        f"[RISK_REDUCING_LANE] symbol={symbol} action={action} "
        f"eligible_for_controlled_paper_handling={str(eligible_for_controlled_paper_handling).lower()} "
        f"dispatch_approval_required={str(dispatch_approval_required).lower()}",
        flush=True,
    )


def build_governance_degraded_mode_audit(
    *,
    readiness_doc: Dict[str, Any],
    classification: Dict[str, Any],
    alloc_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    alloc = alloc_summary or {}
    mode = classification.get("governance_mode")
    recommendation = (
        "Maintain paper-only risk-reducing visibility; block risk-increasing actions until "
        "concentration/drawdown posture improves."
        if mode == GOVERNANCE_MODE_ORANGE_DEGRADED
        else "Governance posture requires review before execution."
    )
    return {
        "generated_at": _iso_utc(),
        "governance_mode": mode,
        "readiness_status": readiness_doc.get("readiness_status"),
        "paper_execution_permitted": bool(readiness_doc.get("paper_execution_permitted")),
        "live_execution_permitted": bool(readiness_doc.get("live_execution_permitted")),
        "concentration_detected": bool(classification.get("concentration_detected")),
        "drawdown_detected": bool(classification.get("drawdown_detected")),
        "largest_position_symbol": alloc.get("largest_position_symbol"),
        "largest_position_weight_pct": alloc.get("largest_position_weight_pct"),
        "risk_increasing_actions_blocked": bool(
            classification.get("risk_increasing_actions_blocked")
        ),
        "risk_reducing_paper_actions_allowed": bool(
            classification.get("risk_reducing_paper_actions_allowed")
        ),
        "force_exit_symbols": list(alloc.get("force_exit_symbols") or []),
        "trim_priority_symbols": list(alloc.get("trim_priority_symbols") or []),
        "blocked_new_buy_symbols": list(alloc.get("blocked_new_buy_symbols") or []),
        "top_reason": classification.get("top_reason"),
        "degraded_mode_reason": list(classification.get("degraded_mode_reason") or []),
        "recommendation": recommendation,
        "safety_statement": SAFETY_STATEMENT,
    }


def enrich_execution_readiness(
    readiness_doc: Dict[str, Any],
    *,
    gov_doc: Dict[str, Any],
    cpe_doc: Dict[str, Any],
    auth_doc: Dict[str, Any],
    results_dir: Optional[Path] = None,
    write_audit: bool = True,
) -> Dict[str, Any]:
    """
    Attach degraded-mode fields to execution_readiness without promoting to READY.
    """
    results_dir = Path(results_dir or ROOT / "data" / "results")
    classification = classify_governance_mode(
        gov_doc=gov_doc,
        cpe_doc=cpe_doc,
        governance_authorized=bool(auth_doc.get("governance_authorized")),
    )
    alloc_summary = _read_json(results_dir / "portfolio_allocation_summary.json")

    out = dict(readiness_doc)
    out["governance_mode"] = classification["governance_mode"]
    out["risk_increasing_actions_blocked"] = classification["risk_increasing_actions_blocked"]
    out["risk_reducing_paper_actions_allowed"] = classification[
        "risk_reducing_paper_actions_allowed"
    ]
    out["degraded_mode_reason"] = classification["degraded_mode_reason"]
    out["live_execution_permitted"] = False

    if classification["governance_mode"] == GOVERNANCE_MODE_ORANGE_DEGRADED:
        out["governance_degraded_active"] = True
        if bool(auth_doc.get("execution_authorized")):
            out["paper_execution_permitted"] = True
        log_governance_degraded_mode(classification)

    audit = build_governance_degraded_mode_audit(
        readiness_doc=out,
        classification=classification,
        alloc_summary=alloc_summary,
    )
    if write_audit:
        _atomic_write_json(audit, results_dir / "governance_degraded_mode_audit.json")
    out["governance_degraded_mode_audit_timestamp"] = audit.get("generated_at")
    return out


def entry_blocked_by_active_drawdown(
    alerts: Sequence[Mapping[str, Any]],
    decision_time: datetime,
) -> bool:
    """Block a risk-increasing entry only when an active drawdown already existed."""
    if decision_time.tzinfo is None:
        decision_time = decision_time.replace(tzinfo=timezone.utc)
    for alert in alerts:
        if not isinstance(alert, Mapping):
            continue
        if str(alert.get("alert_type") or "") != "EXCESS_POSITION_DRAWDOWN":
            continue
        if str(alert.get("status") or "ACTIVE").upper() == "RESOLVED":
            continue
        first_seen = _parse_iso(str(alert.get("first_seen") or ""))
        if first_seen is None:
            continue
        if first_seen <= decision_time:
            return True
    return False


def describe_active_entry_blocker(
    readiness_doc: Optional[Mapping[str, Any]] = None,
    decision: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Name the runtime entry gate. Certification status is not an order gate."""
    readiness = dict(readiness_doc or {})
    decision_doc = dict(decision or {})
    details = decision_doc.get("details") if isinstance(decision_doc.get("details"), dict) else {}
    checks = details.get("checks") if isinstance(details.get("checks"), dict) else {}
    failed = [str(item) for item in (readiness.get("failed_checks") or [])]
    if not failed:
        failed = [str(item) for item in (checks.get("readiness_failed_checks") or [])]
    mode = str(readiness.get("governance_mode") or checks.get("governance_mode") or "")
    gate_reason = str(decision_doc.get("gate_reason") or "")
    failed_check = str(details.get("failed_check") or "")
    operational = ("watchdog", "data_freshness", "broker", "lifecycle", "signals")

    def _result(label: str, gate: str, *, active: bool = True) -> Dict[str, Any]:
        return {
            "active": active,
            "label": label,
            "gate": gate,
            "source": (
                "execution_readiness"
                if gate in operational or gate == "governance"
                else "paper_entry_decision"
            ),
            "certification_is_runtime_gate": False,
            "detail": (
                f"Runtime entry gate: {label}. "
                "The legacy certification banner is not this order gate."
            ),
        }

    for name in operational:
        if name in failed:
            label = "watchdog" if name == "watchdog" else name.replace("_", " ")
            return _result(label, name)
    orange = mode == GOVERNANCE_MODE_ORANGE_DEGRADED and failed == ["governance"]
    if orange or "ORANGE governance" in gate_reason:
        return _result("ORANGE governance", "governance")
    if "governance" in failed:
        return _result("governance", "governance")
    if failed_check == "capacity" or "MAX_POSITIONS" in gate_reason:
        return _result("capacity", "capacity")
    if gate_reason and failed_check not in ("", "execution_readiness"):
        return _result(gate_reason, failed_check or "entry_gate")
    status = str(
        checks.get("execution_readiness_status") or readiness.get("readiness_status") or ""
    )
    if status and status != "READY":
        return _result(f"execution readiness {status}", "execution_readiness")
    if gate_reason:
        return _result(gate_reason, failed_check or "entry_gate")
    return _result("none", "none", active=False)


def classify_paper_entry_readiness_outcome(
    readiness_doc: Dict[str, Any],
    failed_checks: List[str],
) -> Tuple[str, str, bool]:
    """
    Classify paper-entry readiness block.

    Returns (status_label, reason, is_safe_block).
    status_label is one of FAIL, SAFE_BLOCK, SAFE_DEGRADED.
    """
    failed = [str(x) for x in failed_checks]
    readiness_status = str(readiness_doc.get("readiness_status") or "")
    governance_mode = str(readiness_doc.get("governance_mode") or "")

    operational_blockers = {
        "data_freshness",
        "watchdog",
        "broker",
        "lifecycle",
        "signals",
    }
    if any(x in failed for x in operational_blockers):
        stale_hint = ""
        if "lifecycle" in failed or "signals" in failed:
            stale_hint = " (stale lifecycle/signals — run phase_196a --repair)"
        return (
            "FAIL",
            f"execution_readiness={readiness_status}{stale_hint}",
            False,
        )

    if governance_mode == GOVERNANCE_MODE_RED_BLOCKED or (
        "governance" in failed and governance_mode != GOVERNANCE_MODE_ORANGE_DEGRADED
    ):
        if failed == ["governance"]:
            return (
                "FAIL",
                f"execution_readiness={readiness_status} (critical governance)",
                False,
            )
        return "FAIL", f"execution_readiness={readiness_status}", False

    if governance_mode == GOVERNANCE_MODE_ORANGE_DEGRADED and failed == ["governance"]:
        return (
            "SAFE_DEGRADED",
            "ORANGE governance; risk-increasing actions blocked; "
            "risk-reducing paper lane allowed with approval",
            True,
        )

    if readiness_status != "READY":
        return "FAIL", f"execution_readiness={readiness_status}", False
    return "PASS", "OK", False


def is_orange_degraded_context(readiness_doc: Dict[str, Any]) -> bool:
    return str(readiness_doc.get("governance_mode") or "") == GOVERNANCE_MODE_ORANGE_DEGRADED
