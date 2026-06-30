"""
live_block_classifier.py
Centralized live confirmation reader and live-block status for paper-mode validators.
Stale/expired live_confirm artifacts must never imply live execution is authorized.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
READINESS_JSON = RESULTS / "execution_readiness.json"

LIVE_CONFIRM_PATHS = [
    LIVE / "live_confirm.json",
    RESULTS / "live_confirm.json",
]

LIVE_CONFIRM_ABSENT = "LIVE_CONFIRM_ABSENT"
LIVE_CONFIRM_EXPIRED = "LIVE_CONFIRM_EXPIRED"
LIVE_CONFIRM_STALE = "LIVE_CONFIRM_STALE"
LIVE_CONFIRM_INVALID = "LIVE_CONFIRM_INVALID"
LIVE_CONFIRM_VALID_CURRENT = "LIVE_CONFIRM_VALID_CURRENT"

DEFAULT_CONFIRM_TTL_MINUTES = 10


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _live_env_flags() -> Dict[str, str]:
    keys = (
        "TRITON_ENABLE_LIVE_EXECUTION",
        "TRITON_LIVE_EXECUTION",
        "TRITON_FORCE_LIVE",
        "TRITON_ENABLE_PAPER_EXECUTION",
        "TRITON_ENABLE_AUTO_PAPER_ENTRIES",
        "TRITON_ENABLE_AUTO_PAPER_APPROVAL",
    )
    return {k: os.environ.get(k, "") for k in keys}


def _live_env_enabled(flags: Optional[Dict[str, str]] = None) -> bool:
    flags = flags or _live_env_flags()
    return any(
        str(flags.get(k) or "").strip() == "1"
        for k in ("TRITON_ENABLE_LIVE_EXECUTION", "TRITON_LIVE_EXECUTION", "TRITON_FORCE_LIVE")
    )


def _paper_env_enabled(flags: Optional[Dict[str, str]] = None) -> bool:
    flags = flags or _live_env_flags()
    return str(flags.get("TRITON_ENABLE_PAPER_EXECUTION") or "").strip() == "1"


def classify_live_confirm_artifact(
    path: Path,
    *,
    ttl_minutes: int = DEFAULT_CONFIRM_TTL_MINUTES,
) -> Dict[str, Any]:
    """Classify a single live_confirm.json path."""
    rel = str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path)
    if not path.is_file():
        return {
            "path": rel,
            "exists": False,
            "status": LIVE_CONFIRM_ABSENT,
            "detail": "file_missing",
        }

    doc = _read_json(path)
    if not doc:
        return {
            "path": rel,
            "exists": True,
            "status": LIVE_CONFIRM_INVALID,
            "detail": "unreadable_or_empty",
        }

    confirmed = bool(doc.get("confirmed")) or bool(doc.get("ok"))
    confirmed_at = _parse_dt(doc.get("confirmed_at") or doc.get("ts") or doc.get("time"))
    expires_at = _parse_dt(doc.get("expires_at"))
    mode = str(doc.get("mode") or "").lower()
    now = _utc_now()

    if not confirmed:
        return {
            "path": rel,
            "exists": True,
            "status": LIVE_CONFIRM_INVALID,
            "detail": "confirmed_flag_false",
            "confirmed_at": doc.get("confirmed_at"),
            "mode": doc.get("mode"),
        }

    if confirmed_at is None:
        return {
            "path": rel,
            "exists": True,
            "status": LIVE_CONFIRM_INVALID,
            "detail": "missing_confirmed_at",
            "mode": doc.get("mode"),
        }

    age_minutes = (now - confirmed_at).total_seconds() / 60.0
    ttl = int(doc.get("ttl_mins") or doc.get("ttl_minutes") or ttl_minutes)

    if expires_at and expires_at <= now:
        return {
            "path": rel,
            "exists": True,
            "status": LIVE_CONFIRM_EXPIRED,
            "detail": "expires_at_in_past",
            "confirmed_at": doc.get("confirmed_at"),
            "expires_at": doc.get("expires_at"),
            "age_minutes": round(age_minutes, 1),
            "mode": doc.get("mode"),
        }

    if age_minutes > ttl:
        return {
            "path": rel,
            "exists": True,
            "status": LIVE_CONFIRM_STALE,
            "detail": f"age_minutes={age_minutes:.1f} exceeds ttl={ttl}",
            "confirmed_at": doc.get("confirmed_at"),
            "expires_at": doc.get("expires_at"),
            "age_minutes": round(age_minutes, 1),
            "ttl_minutes": ttl,
            "mode": doc.get("mode"),
        }

    if mode == "paper":
        return {
            "path": rel,
            "exists": True,
            "status": LIVE_CONFIRM_INVALID,
            "detail": "paper_mode_context_mismatch",
            "confirmed_at": doc.get("confirmed_at"),
            "mode": doc.get("mode"),
            "age_minutes": round(age_minutes, 1),
        }

    required_acks = ("acknowledgement_live_execution", "acknowledgement_no_paper_only")
    missing_acks = [a for a in required_acks if a in doc and not bool(doc.get(a))]
    if missing_acks:
        pass

    return {
        "path": rel,
        "exists": True,
        "status": LIVE_CONFIRM_VALID_CURRENT,
        "detail": "within_ttl_live_context",
        "confirmed_at": doc.get("confirmed_at"),
        "expires_at": doc.get("expires_at"),
        "age_minutes": round(age_minutes, 1),
        "ttl_minutes": ttl,
        "mode": doc.get("mode"),
    }


def scan_live_confirm_artifacts(
    *,
    ttl_minutes: int = DEFAULT_CONFIRM_TTL_MINUTES,
) -> Dict[str, Any]:
    by_path: Dict[str, Dict[str, Any]] = {}
    for path in LIVE_CONFIRM_PATHS:
        info = classify_live_confirm_artifact(path, ttl_minutes=ttl_minutes)
        by_path[info["path"]] = info

    statuses = [v["status"] for v in by_path.values()]
    valid_count = sum(1 for s in statuses if s == LIVE_CONFIRM_VALID_CURRENT)
    stale_count = sum(
        1 for s in statuses if s in (LIVE_CONFIRM_STALE, LIVE_CONFIRM_EXPIRED, LIVE_CONFIRM_INVALID)
    )
    absent_count = sum(1 for s in statuses if s == LIVE_CONFIRM_ABSENT)

    return {
        "live_confirm_paths_found": list(by_path.keys()),
        "live_confirm_status_by_path": by_path,
        "valid_live_confirm_count": valid_count,
        "stale_live_confirm_count": stale_count,
        "absent_live_confirm_count": absent_count,
        "any_valid_current": valid_count > 0,
    }


def _readiness_live_permitted() -> bool:
    doc = _read_json(READINESS_JSON)
    return bool(doc.get("live_execution_permitted", False))


def evaluate_live_block_status(*, paper_mode: bool = True) -> Dict[str, Any]:
    """
    Determine whether live execution is blocked for validator reporting.
    Paper-mode default: live_blocked=True unless explicitly and validly authorized.
    """
    flags = _live_env_flags()
    confirm_scan = scan_live_confirm_artifacts()
    live_env = _live_env_enabled(flags)
    paper_env = _paper_env_enabled(flags)
    readiness_live = _readiness_live_permitted()

    valid_confirm = confirm_scan["any_valid_current"]
    stale_or_expired = confirm_scan["stale_live_confirm_count"] > 0

    explicitly_authorized = live_env and readiness_live and valid_confirm and not paper_mode

    live_blocked = not explicitly_authorized

    if explicitly_authorized:
        live_block_reason = "LIVE_EXPLICITLY_AUTHORIZED"
    elif stale_or_expired and not valid_confirm:
        live_block_reason = "LIVE_CONFIRM_STALE_OR_EXPIRED"
    elif not live_env:
        live_block_reason = "LIVE_ENV_NOT_ENABLED"
    elif not readiness_live:
        live_block_reason = "LIVE_EXECUTION_NOT_PERMITTED_BY_READINESS"
    elif paper_mode or paper_env:
        live_block_reason = "PAPER_MODE_LIVE_BLOCKED"
    elif not valid_confirm:
        live_block_reason = "LIVE_CONFIRM_ABSENT_OR_INVALID"
    else:
        live_block_reason = "LIVE_BLOCKED_BY_POLICY"

    return {
        "live_blocked": live_blocked,
        "live_block_reason": live_block_reason,
        "live_explicitly_authorized": explicitly_authorized,
        "live_env_enabled": live_env,
        "paper_env_enabled": paper_env,
        "paper_mode": paper_mode,
        "readiness_live_execution_permitted": readiness_live,
        **confirm_scan,
        "live_env_flags": flags,
    }


def classify_paper_entry_gate_for_validation(
    gate_ok: bool,
    gate_reason: str,
    gate_details: Optional[Dict[str, Any]] = None,
) -> str:
    """Map paper entry gate outcome to validator-safe classification."""
    if gate_ok:
        return "PASS_GATE_OPEN"
    details = gate_details or {}
    failed = str(details.get("failed_check") or "")
    if failed == "market_open" or "market is closed" in str(gate_reason).lower():
        return "SAFE_BLOCK_MARKET_CLOSED"
    if details.get("safe_block"):
        return "PASS_SAFE_BLOCKED"
    if failed == "execution_readiness" or "execution_readiness=" in str(gate_reason):
        return "PASS_SAFE_BLOCKED"
    if failed in ("paper_execution_env", "auto_paper_entries_env"):
        return "SAFE_BLOCK_PAPER_ENV_NOT_ARMED"
    if "live_execution_permitted" in str(gate_reason):
        return "FAIL_LIVE_BLOCK_REGRESSION"
    return "FAIL_PAPER_ENTRY_GATE"


def validator_live_block_check(
    *,
    paper_mode: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (live_blocked_ok, payload) for phase validators.
    live_blocked_ok is False when regression detected (live_blocked=False unexpectedly).
    """
    payload = evaluate_live_block_status(paper_mode=paper_mode)
    ok = bool(payload.get("live_blocked"))
    return ok, payload


def validator_paper_gate_acceptable(
    gate_ok: bool,
    gate_reason: str,
    gate_details: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Whether paper entry gate failure is acceptable for no-execution validation phases."""
    classification = classify_paper_entry_gate_for_validation(gate_ok, gate_reason, gate_details)
    acceptable = classification in (
        "PASS_GATE_OPEN",
        "SAFE_BLOCK_MARKET_CLOSED",
        "PASS_SAFE_BLOCKED",
        "SAFE_BLOCK_PAPER_ENV_NOT_ARMED",
    )
    return acceptable, classification


def defensive_exit_phase_validator_gates(*, verbose: bool = False) -> Dict[str, Any]:
    """Shared live-block and paper-entry gate evaluation for 197I–197M validators."""
    from services.paper_entry_loop_stage import evaluate_paper_entry_gates

    live_block_ok, live_payload = validator_live_block_check(paper_mode=True)
    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=verbose)
    gate_acceptable, gate_classification = validator_paper_gate_acceptable(
        gate_ok, gate_reason, gate_details
    )
    return {
        "live_block_ok": live_block_ok,
        "live_block_payload": live_payload,
        "live_blocked": bool(live_payload.get("live_blocked")),
        "live_block_reason": live_payload.get("live_block_reason"),
        "gate_ok": gate_ok,
        "gate_reason": gate_reason,
        "gate_details": gate_details,
        "gate_acceptable": gate_acceptable,
        "gate_classification": gate_classification,
    }


def apply_defensive_exit_validator_gate_checks(
    *,
    status: str,
    notes: List[str],
    gates: Dict[str, Any],
    fail_on_gate_open: bool = True,
) -> str:
    """Apply standard live-block / paper-gate pass rules; returns updated status."""
    if not gates.get("live_block_ok"):
        status = "FAIL"
        notes.append(f"FAIL_LIVE_BLOCK_REGRESSION reason={gates.get('live_block_reason')}")
    if fail_on_gate_open and gates.get("gate_ok"):
        status = "FAIL"
        notes.append("paper entry gate unexpectedly passed")
    elif not gates.get("gate_acceptable"):
        status = "FAIL"
        notes.append(f"paper entry gate: {gates.get('gate_classification')}")
    return status
