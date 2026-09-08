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
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from services.command_center_status_coherence import (
    CONTINUOUS_CYCLE_OVERDUE_MINUTES,
    CONTINUOUS_INTERVAL_MINUTES,
    age_minutes as _coherent_age_minutes,
    authorization_status_text,
    cycle_deadline_utc,
    cycle_is_overdue,
    format_live_permission_display,
    format_paper_certification_display,
    parse_lock_timestamp,
    parse_utc_timestamp,
    resolve_market_status,
    resolve_shutdown_presentation,
)
from services.continuous_loop_cadence import (
    authoritative_next_due_utc,
    iso_et as cadence_iso_et,
    iso_utc as cadence_iso_utc,
    load_cadence_artifact,
)

ET = ZoneInfo("America/New_York")

# Presentation-only session phases (UIR-003D). Does not alter execution market gates.
SESSION_MARKET_OPEN = "MARKET_OPEN"
SESSION_PRE_MARKET = "PRE_MARKET"
SESSION_AFTER_HOURS = "AFTER_HOURS"
SESSION_NON_TRADING_DAY = "MARKET_CLOSED_NON_TRADING_DAY"
SESSION_UNKNOWN = "UNKNOWN"

BUCKET_ACTION_NOW = "ACTION_REQUIRED_NOW"
BUCKET_BEFORE_OPEN = "BEFORE_NEXT_MARKET_OPEN"
BUCKET_INFO = "INFORMATIONAL"

EXPECTED_AFTER_HOURS_STOP = "EXPECTED_AFTER_HOURS_STOP"

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

WATCHDOG_OVERDUE_MINUTES = 10.0
DATA_FRESH_MINUTES = 90.0
BROKER_FRESH_MINUTES = 120.0
OPEN_ORDERS_FRESH_MINUTES = 120.0

BROKER_REST_STATES = ("CONNECTED", "DISCONNECTED", "STALE", "UNKNOWN")
# Material future skew: never treat as CONNECTED / DISCONNECTED.
BROKER_FUTURE_TIMESTAMP_TOLERANCE_MINUTES = 1.0

PAPER_BROKER_REST_MEANING = (
    "Alpaca paper REST probe result at the evidence timestamp. "
    "Streaming connectivity is not assessed."
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def classify_paper_broker_rest_evidence(
    *,
    broker_connected: Any,
    evidence_timestamp: Any,
    now: Optional[datetime] = None,
    fresh_minutes: float = BROKER_FRESH_MINUTES,
) -> Dict[str, Any]:
    """Classify paper REST broker evidence with explicit freshness.

    Runtime process presence must not be passed here — a stopped watchdog or
    continuous loop does not imply DISCONNECTED.
    """
    now = now or _utc_now()
    ts = _parse_dt(evidence_timestamp)
    age = _age_minutes(ts, now) if ts is not None else None

    display = {
        "CONNECTED": "Paper broker REST: Connected",
        "DISCONNECTED": "Paper broker REST: Disconnected",
        "STALE": "Paper broker REST: Stale",
        "UNKNOWN": "Paper broker REST: Unknown",
    }

    def _pack(
        state: str,
        *,
        freshness: str,
        connected_compat: Optional[bool],
    ) -> Dict[str, Any]:
        return {
            "state": state,
            "display_label": display[state],
            "connected": connected_compat,
            "evidence_timestamp": ts.isoformat() if ts is not None else None,
            "evidence_age_minutes": age,
            "freshness": freshness,
            "fresh_minutes_threshold": float(fresh_minutes),
            "mode": "paper",
            "meaning": PAPER_BROKER_REST_MEANING,
            "streaming_assessed": False,
            "streaming_connected": None,
        }

    if ts is None:
        return _pack("UNKNOWN", freshness="UNKNOWN", connected_compat=None)

    if age is not None and age < -float(BROKER_FUTURE_TIMESTAMP_TOLERANCE_MINUTES):
        return _pack("UNKNOWN", freshness="UNKNOWN", connected_compat=None)

    if age is not None and age > float(fresh_minutes):
        return _pack("STALE", freshness="STALE", connected_compat=None)

    if broker_connected is True:
        return _pack("CONNECTED", freshness="FRESH", connected_compat=True)
    if broker_connected is False:
        return _pack("DISCONNECTED", freshness="FRESH", connected_compat=False)
    return _pack(
        "UNKNOWN", freshness="FRESH" if age is not None else "UNKNOWN", connected_compat=None
    )


def classify_open_orders_evidence_freshness(
    *,
    evidence_timestamp: Any,
    now: Optional[datetime] = None,
    fresh_minutes: float = OPEN_ORDERS_FRESH_MINUTES,
) -> Dict[str, Any]:
    """Freshness for current open-order snapshot evidence (no broker calls)."""
    now = now or _utc_now()
    ts = _parse_dt(evidence_timestamp)
    age = _age_minutes(ts, now) if ts is not None else None
    if ts is None:
        return {
            "freshness": "UNKNOWN",
            "evidence_timestamp": None,
            "evidence_age_minutes": None,
            "fresh_minutes_threshold": float(fresh_minutes),
        }
    if age is not None and age < -float(BROKER_FUTURE_TIMESTAMP_TOLERANCE_MINUTES):
        return {
            "freshness": "UNKNOWN",
            "evidence_timestamp": ts.isoformat(),
            "evidence_age_minutes": age,
            "fresh_minutes_threshold": float(fresh_minutes),
        }
    if age is not None and age > float(fresh_minutes):
        return {
            "freshness": "STALE",
            "evidence_timestamp": ts.isoformat(),
            "evidence_age_minutes": age,
            "fresh_minutes_threshold": float(fresh_minutes),
        }
    return {
        "freshness": "FRESH",
        "evidence_timestamp": ts.isoformat(),
        "evidence_age_minutes": age,
        "fresh_minutes_threshold": float(fresh_minutes),
    }


def _parse_dt(value: Any) -> Optional[datetime]:
    """UTC/Z/offset JSON timestamps (heartbeats, expiry, certification)."""
    return parse_utc_timestamp(value)


def _parse_lock_dt(value: Any) -> Optional[datetime]:
    """Lock started_at/refreshed_at (naive local from loop_lock_utils.ps1)."""
    return parse_lock_timestamp(value)


PAPER_READINESS_EXPLANATORY_COPY = (
    "Standing authorization and the execution arm are separate. "
    "Authorization alone cannot submit an order. "
    "A current paper arm and all existing readiness, approval, and broker-dispatch "
    "gates are still required. Live execution remains blocked."
)


def build_paper_readiness_wording(
    *,
    root: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Read-only standing authorization vs paper-arm display contracts.

    Does not renew authorization, create an arm, or alter execution gates.
    Never implies that standing authorization alone permits order submission.
    """
    now = now or _utc_now()
    root = Path(root) if root is not None else ROOT
    results = root / "data" / "results"
    live = root / "data" / "live"

    gov = _read_json(results / "governance_authorization.json")
    readiness = _read_json(results / "execution_readiness.json")
    gov_bad = bool(gov.get("__malformed__"))
    ready_bad = bool(readiness.get("__malformed__"))

    paper_flag: Optional[bool] = None
    auth_created_at: Optional[str] = None
    auth_source = "unknown"
    if not gov_bad and (results / "governance_authorization.json").is_file():
        if "paper_execution_permitted" in gov:
            paper_flag = bool(gov.get("paper_execution_permitted"))
            auth_created_at = (
                str(gov.get("generated_at") or gov.get("as_of") or gov.get("timestamp") or "")
                or None
            )
            auth_source = str(results / "governance_authorization.json")
    if paper_flag is None and not ready_bad and (results / "execution_readiness.json").is_file():
        if "paper_execution_permitted" in readiness:
            paper_flag = bool(readiness.get("paper_execution_permitted"))
            auth_created_at = (
                str(
                    readiness.get("generated_at")
                    or readiness.get("as_of")
                    or readiness.get("timestamp")
                    or ""
                )
                or None
            )
            auth_source = str(results / "execution_readiness.json")

    if paper_flag is True:
        auth_status = "PERMITTED"
    elif paper_flag is False:
        auth_status = "NOT_PERMITTED"
    else:
        auth_status = "UNKNOWN"

    # Standing policy flag has no operator latch expiry in these artifacts.
    auth_expires_at = None

    arm_path = live / "paper_arm_confirm.json"
    arm_status = "UNKNOWN"
    arm_created_at: Optional[str] = None
    arm_expires_at: Optional[str] = None
    arm_display = "UNKNOWN"
    try:
        from services.paper_arm_status import build_paper_arm_status

        if root == ROOT:
            arm_payload = build_paper_arm_status(verbose_notes=False)
            arm_status = str(arm_payload.get("status") or "UNKNOWN")
            arm_created_at = str(arm_payload.get("timestamp") or "") or None
            arm_expires_at = str(arm_payload.get("expires_at") or "") or None
        else:
            # Fixture roots: classify from local file only (no global LIVE path).
            if not arm_path.is_file():
                arm_status = "MISSING"
            else:
                try:
                    doc = json.loads(arm_path.read_text(encoding="utf-8"))
                except Exception:
                    doc = {"__malformed__": True}
                if not isinstance(doc, dict) or doc.get("__malformed__"):
                    arm_status = "UNKNOWN"
                elif not bool(doc.get("allow_execute", True)):
                    arm_status = "INVALID"
                else:
                    exp = _parse_dt(doc.get("expires_at"))
                    if exp is None:
                        arm_status = "INVALID"
                    elif exp <= now:
                        arm_status = "EXPIRED"
                    else:
                        arm_status = "VALID"
                    arm_created_at = str(doc.get("timestamp") or "") or None
                    arm_expires_at = str(doc.get("expires_at") or "") or None
        # Display: VALID → ARMED for operator clarity; keep EXPIRED explicit.
        if arm_status == "VALID":
            arm_display = "ARMED"
        elif arm_status in ("EXPIRED", "MISSING", "INVALID", "UNKNOWN"):
            arm_display = arm_status
        else:
            arm_display = "UNKNOWN"
    except Exception:
        arm_status = "UNKNOWN"
        arm_display = "UNKNOWN"

    live_blocked = True
    if not ready_bad and "live_execution_permitted" in readiness:
        live_blocked = not bool(readiness.get("live_execution_permitted"))
    if not gov_bad and "live_execution_permitted" in gov:
        live_blocked = live_blocked and (not bool(gov.get("live_execution_permitted")))

    return {
        "standing_authorization_label": "Standing paper authorization",
        "standing_authorization_status": auth_status,
        "standing_authorization_created_at": auth_created_at,
        "standing_authorization_expires_at": auth_expires_at,
        "standing_authorization_currently_valid": paper_flag is True,
        "standing_authorization_source": auth_source,
        "paper_arm_label": "Paper execution arm",
        "paper_arm_status": arm_display,
        "paper_arm_raw_status": arm_status,
        "paper_arm_created_at": arm_created_at,
        "paper_arm_expires_at": arm_expires_at,
        "paper_arm_currently_valid": arm_status == "VALID",
        "paper_arm_source": str(arm_path),
        "authorization_and_arm_are_distinct": True,
        "authorization_alone_permits_submission": False,
        "order_ready": False,
        "ready_to_trade": False,
        "explanatory_copy": PAPER_READINESS_EXPLANATORY_COPY,
        "live_execution_blocked": live_blocked,
        "live_execution_display": (
            "LIVE EXECUTION BLOCKED" if live_blocked else "LIVE EXECUTION NOT BLOCKED"
        ),
    }


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
    bucket: str = BUCKET_ACTION_NOW
    classification_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _nyse_is_session_day(day) -> Optional[bool]:
    """True/False when calendar available; None when unavailable (fail closed to UNKNOWN phase)."""
    try:
        import pandas_market_calendars as mcal  # type: ignore

        cal = mcal.get_calendar("NYSE")
        sched = cal.schedule(start_date=day.isoformat(), end_date=day.isoformat())
        return bool(sched is not None and not sched.empty)
    except Exception:
        return None


def classify_operator_session_context(
    *,
    now: Optional[datetime] = None,
    market_open: Optional[bool] = None,
    market_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Presentation-only session phase from authoritative OPEN/CLOSED + ET clock/calendar.

    Does not change execution market-hours decisions.
    """
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    now_et = now_utc.astimezone(ET)
    market = dict(market_status or {})
    session_reason = str(market.get("reason") or "") or None
    session_state = str(market.get("session_state") or "").upper() or None
    if market_open is None:
        if "market_open" in market:
            market_open = market.get("market_open")
        else:
            try:
                from services.guard_auto import is_market_open_et

                market_open = bool(is_market_open_et(now_utc))
            except Exception:
                market_open = None
        if not session_reason or not session_state:
            try:
                from services.market_calendar import regular_session_state

                payload = regular_session_state(now=now_utc)
                session_reason = session_reason or str(payload.get("reason") or "") or None
                session_state = (
                    session_state or str(payload.get("session_state") or "").upper() or None
                )
            except Exception:
                pass

    status = str(market.get("status") or "").upper()
    if session_state == "UNKNOWN" and market_open is not True and status != "OPEN":
        phase = SESSION_UNKNOWN
        market_open = False if market_open is True else market_open
        if market_open is None:
            market_open = False
        status = "UNKNOWN"
    elif market_open is None and status not in ("OPEN", "CLOSED"):
        phase = SESSION_UNKNOWN
    elif market_open is True or status == "OPEN":
        phase = SESSION_MARKET_OPEN
        market_open = True
        status = "OPEN"
    else:
        market_open = False
        status = "CLOSED"
        session_day = _nyse_is_session_day(now_et.date())
        t = now_et.time()
        open_t = time(9, 30)
        close_t = time(16, 0)
        if session_reason in {"holiday", "weekend"}:
            phase = SESSION_NON_TRADING_DAY
        elif session_reason == "pre_open":
            phase = SESSION_PRE_MARKET
        elif session_reason in {"regular_close", "early_close"}:
            phase = SESSION_AFTER_HOURS
        elif session_day is False or now_et.weekday() >= 5:
            phase = SESSION_NON_TRADING_DAY
        elif session_day is None and now_et.weekday() >= 5:
            phase = SESSION_NON_TRADING_DAY
        elif t < open_t:
            phase = SESSION_PRE_MARKET
        elif t >= close_t:
            phase = SESSION_AFTER_HOURS
        elif session_day is None:
            # Weekday inside 09:30–16:00 but classifier said closed — ambiguous.
            phase = SESSION_UNKNOWN
        else:
            # Session day but closed inside regular window (e.g. early close residual).
            phase = SESSION_AFTER_HOURS if t >= close_t else SESSION_UNKNOWN

    return {
        "session_phase": phase,
        "status": (
            status
            if status in ("OPEN", "CLOSED", "UNKNOWN")
            else ("OPEN" if market_open else "CLOSED" if market_open is False else "UNKNOWN")
        ),
        "market_open": market_open,
        "session_reason": session_reason,
        "session_state": session_state,
        "as_of_et": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
        "as_of": now_utc.isoformat(),
        "source": "guard_auto.is_market_open_et+et_clock+nyse_calendar",
        "note": "Presentation-only session context; execution gates unchanged",
    }


def _loop_process_count(loop: Dict[str, Any]) -> Tuple[Optional[int], bool]:
    """Return (count|None, discovery_ok)."""
    if loop.get("process_discovery_available") is False:
        return None, False
    count = loop.get("process_count")
    if count is not None:
        try:
            return max(0, int(count)), True
        except (TypeError, ValueError):
            return None, True
    if loop.get("process_running") is True:
        return 1, True
    if loop.get("process_running") is False:
        return 0, True
    pids = loop.get("pids")
    if isinstance(pids, list):
        return len(pids), True
    return None, True


def _loop_has_orphan_lock(loop: Dict[str, Any], count: Optional[int]) -> bool:
    lc = str(loop.get("lock_classification") or "")
    if lc == "STALE_ORPHAN_LOCK":
        return True
    if (
        bool(loop.get("lock_exists"))
        and count == 0
        and loop.get("process_discovery_available") is not False
    ):
        return True
    return False


def classify_expected_after_hours_stop(snapshot_parts: Dict[str, Any]) -> Dict[str, Any]:
    """True only when a clean after-hours/non-trading stopped state is proven."""
    market = snapshot_parts.get("market") or {}
    phase = str(market.get("session_phase") or SESSION_UNKNOWN)
    live = snapshot_parts.get("live_execution") or {}
    cont = snapshot_parts.get("continuous") or {}
    wd = snapshot_parts.get("watchdog") or {}
    cont_n, cont_ok = _loop_process_count(cont)
    wd_n, wd_ok = _loop_process_count(wd)
    reasons: List[str] = []

    if phase == SESSION_UNKNOWN or market.get("market_open") is None:
        reasons.append("session_unknown")
    if phase == SESSION_MARKET_OPEN or market.get("market_open") is True:
        reasons.append("market_open")
    if phase == SESSION_PRE_MARKET:
        # Pre-market clean stop is expected-as-stopped but not AFTER_HOURS_STOP.
        reasons.append("pre_market")
    if not cont_ok or not wd_ok:
        reasons.append("process_discovery_unavailable")
    if cont_n is None or wd_n is None:
        reasons.append("process_count_unavailable")
    if cont_n not in (0,) or wd_n not in (0,):
        reasons.append("component_not_stopped")
    if cont_n is not None and cont_n >= 2:
        reasons.append("duplicate_continuous")
    if wd_n is not None and wd_n >= 2:
        reasons.append("duplicate_watchdog")
    if _loop_has_orphan_lock(cont, cont_n) or _loop_has_orphan_lock(wd, wd_n):
        reasons.append("orphaned_lock")
    if live.get("permitted") is True or live.get("blocked") is False:
        reasons.append("live_execution_not_blocked")

    expected = (not reasons) and phase in (SESSION_AFTER_HOURS, SESSION_NON_TRADING_DAY)

    return {
        "expected": expected,
        "classification": EXPECTED_AFTER_HOURS_STOP if expected else "",
        "session_phase": phase,
        "reasons_not_expected": reasons,
        "continuous_count": cont_n,
        "watchdog_count": wd_n,
    }


def _paper_arm_valid_through_close(
    *,
    expires_at: Any,
    now: Optional[datetime] = None,
) -> Optional[bool]:
    """None when expiration evidence missing/unparseable."""
    exp = _parse_dt(expires_at) if not isinstance(expires_at, datetime) else expires_at
    if exp is None:
        return None
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    now_et = now_utc.astimezone(ET)
    # Coming regular-session close (today if before close, else next weekday close approx).
    close_et = datetime.combine(now_et.date(), time(16, 0), tzinfo=ET)
    if now_et.time() >= time(16, 0):
        # Next calendar day close — presentation readiness for next session.
        from datetime import timedelta

        d = now_et.date()
        d = d + timedelta(days=1)
        while d.weekday() >= 5:
            d = d + timedelta(days=1)
        close_et = datetime.combine(d, time(16, 0), tzinfo=ET)
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return exp.astimezone(timezone.utc) >= close_et.astimezone(timezone.utc)


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
    """Session-aware Operator Attention (UIR-003D). Presentation classification only."""
    items: List[AttentionItem] = []
    paper = snapshot_parts.get("paper_arm") or {}
    cont = snapshot_parts.get("continuous") or {}
    wd = snapshot_parts.get("watchdog") or {}
    live = snapshot_parts.get("live_execution") or {}
    broker = snapshot_parts.get("broker") or {}
    freshness = snapshot_parts.get("data_freshness") or {}
    alerts = snapshot_parts.get("alerts") or {}
    gov = snapshot_parts.get("governance") or {}
    cycle = snapshot_parts.get("cycle") or {}
    market = snapshot_parts.get("market") or {}
    orders = snapshot_parts.get("orders_positions") or {}

    # Ensure session phase present for fixtures that omit enrichment.
    if not market.get("session_phase"):
        market = {
            **market,
            **classify_operator_session_context(
                market_open=market.get("market_open"),
                market_status=market,
            ),
        }
        snapshot_parts = {**snapshot_parts, "market": market}

    phase = str(market.get("session_phase") or SESSION_UNKNOWN)
    market_open = market.get("market_open")
    expected_meta = classify_expected_after_hours_stop(snapshot_parts)
    expected_stop = bool(expected_meta.get("expected"))
    cont_n, cont_ok = _loop_process_count(cont)
    wd_n, wd_ok = _loop_process_count(wd)
    clean_premarket_stop = (
        phase == SESSION_PRE_MARKET
        and market_open is False
        and cont_ok
        and wd_ok
        and cont_n == 0
        and wd_n == 0
        and not _loop_has_orphan_lock(cont, cont_n)
        and not _loop_has_orphan_lock(wd, wd_n)
        and live.get("blocked") is not False
        and live.get("permitted") is not True
    )

    # --- Always-critical safety ---
    if live.get("permitted") is True or live.get("blocked") is False:
        items.append(
            AttentionItem(
                "CRITICAL",
                "Live execution appears permitted",
                "Evidence indicates live execution is not blocked. This violates the operator-console safety expectation.",
                str(live.get("evidence_path") or EXECUTION_READINESS),
                "Treat as a critical safety incident. Do not enable trading. Investigate readiness/governance immediately.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="LIVE_EXECUTION_UNEXPECTEDLY_PERMITTED",
            )
        )

    trading = snapshot_parts.get("trading_pipeline") or {}
    trading_status = str(trading.get("status") or "").upper()
    last_outcome = str(trading.get("last_outcome") or "").strip().lower()
    last_mh = str(trading.get("last_market_hours_outcome") or "").strip().lower()
    attention_code = str(trading.get("operator_attention_code") or "").strip()
    if trading_status == "DEGRADED" and (
        last_outcome == "session_unknown" or attention_code == "SESSION_CALENDAR_UNKNOWN"
    ):
        items.append(
            AttentionItem(
                "HIGH",
                str(
                    trading.get("operator_attention_title")
                    or "Session calendar unavailable — paper entry blocked"
                ),
                "XNYS could not authoritatively prove OPEN or CLOSED. Paper entry is blocked until the calendar is available. "
                "This is not an ordinary closed-market skip. Watchdog process health does not clear this.",
                str(trading.get("evidence_path") or ""),
                "Do not invoke paper entry. Restore exchange_calendars XNYS. Do not treat weekday clock heuristics as session proof.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="SESSION_CALENDAR_UNKNOWN",
            )
        )
    if trading_status == "DEGRADED" and last_mh == "fail":
        items.append(
            AttentionItem(
                "HIGH",
                "Latest market-hours trading cycle failed",
                "A genuine market-hours CYCLE_FAIL remains unresolved. Watchdog process health does not clear this. "
                "Paper arm state is independent: a currently valid arm does not mean the last applicable cycle succeeded.",
                str(trading.get("evidence_path") or ""),
                "Review Activity for the failed cycle. Do not treat watchdog HEALTHY or Paper ARMED as trade-ready until a later market-hours cycle succeeds.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="MARKET_HOURS_CYCLE_FAIL",
            )
        )

    # --- Duplicates (any session) ---
    if cont_ok and cont_n is not None and cont_n >= 2:
        items.append(
            AttentionItem(
                "CRITICAL",
                "Duplicate continuous runtime",
                f"{cont_n} continuous-loop processes matching run_continuous_loop.ps1 were detected; expected exactly 1.",
                "services.runtime_process_detector",
                "Use protected Stop/Restart only after reviewing Diagnostics. Do not start another continuous loop.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="DUPLICATE_CONTINUOUS",
            )
        )
    if wd_ok and wd_n is not None and wd_n >= 2:
        items.append(
            AttentionItem(
                "CRITICAL",
                "Duplicate watchdog runtime",
                f"{wd_n} watchdog-loop processes matching run_watchdog_loop.ps1 were detected; expected exactly 1.",
                "services.runtime_process_detector",
                "Use protected Stop/Restart only after reviewing Diagnostics. Do not start another watchdog.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="DUPLICATE_WATCHDOG",
            )
        )

    # --- Process discovery unavailable ---
    if not cont_ok:
        items.append(
            AttentionItem(
                "HIGH",
                "Continuous process discovery unavailable",
                "Canonical continuous-loop process discovery failed or is unavailable; instance count cannot be confirmed.",
                "services.runtime_process_detector",
                "Refresh Operations evidence. Do not invent a zero-instance reading from locks or prior Start messages.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="PROCESS_DISCOVERY_UNAVAILABLE",
            )
        )
    if not wd_ok:
        items.append(
            AttentionItem(
                "HIGH",
                "Watchdog process discovery unavailable",
                "Canonical watchdog process discovery failed or is unavailable; instance count cannot be confirmed.",
                "services.runtime_process_detector",
                "Refresh Operations evidence. Do not invent a zero-instance reading from locks or prior Start messages.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="PROCESS_DISCOVERY_UNAVAILABLE",
            )
        )

    # --- Orphan / mismatch locks ---
    for label, loop, count in (("Continuous", cont, cont_n), ("Watchdog", wd, wd_n)):
        lc = str(loop.get("lock_classification") or "")
        if _loop_has_orphan_lock(loop, count):
            items.append(
                AttentionItem(
                    "HIGH",
                    f"{label} stale lock",
                    f"{label} lock file exists without a matching live process.",
                    str(loop.get("lock_path") or ""),
                    "Do not delete the lock from the dashboard. Use approved recovery/start paths that clear orphan locks safely.",
                    bucket=BUCKET_ACTION_NOW,
                    classification_code="ORPHANED_LOCK",
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
                    bucket=BUCKET_ACTION_NOW,
                    classification_code="PROCESS_LOCK_MISMATCH",
                )
            )

    # --- Missing runtime components (session-aware) ---
    missing_high = market_open is True or phase == SESSION_MARKET_OPEN
    session_unknown = phase == SESSION_UNKNOWN or market_open is None
    suppress_missing = expected_stop or clean_premarket_stop

    if cont_ok and cont_n == 0 and not _loop_has_orphan_lock(cont, cont_n):
        if missing_high:
            items.append(
                AttentionItem(
                    "HIGH",
                    "Continuous loop missing",
                    "No continuous-loop process matching run_continuous_loop.ps1 was detected during the active market session.",
                    "services.runtime_process_detector",
                    "Start TRITON using the protected Start control only when eligibility allows.",
                    bucket=BUCKET_ACTION_NOW,
                    classification_code="RUNTIME_MISSING_MARKET_OPEN",
                )
            )
        elif session_unknown and not suppress_missing:
            items.append(
                AttentionItem(
                    "HIGH",
                    "Continuous loop missing",
                    "No continuous-loop process was detected and market-session context is UNKNOWN; treating conservatively.",
                    "services.runtime_process_detector",
                    "Confirm market-session evidence and runtime intent before assuming a safe stop.",
                    bucket=BUCKET_ACTION_NOW,
                    classification_code="RUNTIME_MISSING_SESSION_UNKNOWN",
                )
            )
        elif not suppress_missing and phase == SESSION_PRE_MARKET:
            items.append(
                AttentionItem(
                    "INFO",
                    "Continuous loop not running before market open",
                    "Continuous loop is stopped in pre-market. Start only at the appropriate activation time.",
                    "services.runtime_process_detector",
                    "Use the protected Start control before unattended paper activation when eligibility allows.",
                    bucket=BUCKET_BEFORE_OPEN,
                    classification_code="PREMARKET_RUNTIME_STOPPED",
                )
            )

    if wd_ok and wd_n == 0 and not _loop_has_orphan_lock(wd, wd_n):
        if missing_high:
            items.append(
                AttentionItem(
                    "HIGH",
                    "Watchdog missing",
                    "No watchdog-loop process matching run_watchdog_loop.ps1 was detected during the active market session.",
                    "services.runtime_process_detector",
                    "Start TRITON using the protected Start control only when eligibility allows.",
                    bucket=BUCKET_ACTION_NOW,
                    classification_code="RUNTIME_MISSING_MARKET_OPEN",
                )
            )
        elif session_unknown and not suppress_missing:
            items.append(
                AttentionItem(
                    "HIGH",
                    "Watchdog missing",
                    "No watchdog process was detected and market-session context is UNKNOWN; treating conservatively.",
                    "services.runtime_process_detector",
                    "Confirm market-session evidence and runtime intent before assuming a safe stop.",
                    bucket=BUCKET_ACTION_NOW,
                    classification_code="RUNTIME_MISSING_SESSION_UNKNOWN",
                )
            )
        elif not suppress_missing and phase == SESSION_PRE_MARKET:
            items.append(
                AttentionItem(
                    "INFO",
                    "Watchdog not running before market open",
                    "Watchdog is stopped in pre-market. Start only at the appropriate activation time.",
                    "services.runtime_process_detector",
                    "Use the protected Start control before unattended paper activation when eligibility allows.",
                    bucket=BUCKET_BEFORE_OPEN,
                    classification_code="PREMARKET_RUNTIME_STOPPED",
                )
            )

    # Deduplicate: one expected-stop informational banner for after-hours/non-trading clean stop.
    if expected_stop:
        items.append(
            AttentionItem(
                "INFO",
                "TRITON is safely stopped after market close",
                (
                    "Market is closed. Continuous loop and watchdog process counts are 0, "
                    "runtime locks are released, and live execution remains blocked. "
                    f"Session context: {phase}."
                ),
                "services.operator_console_status.EXPECTED_AFTER_HOURS_STOP",
                "No urgent runtime action. Renew the paper arm and start TRITON before the next market-open activation when ready.",
                bucket=BUCKET_INFO,
                classification_code=EXPECTED_AFTER_HOURS_STOP,
            )
        )
    elif clean_premarket_stop:
        items.append(
            AttentionItem(
                "INFO",
                "TRITON is stopped before market open",
                "Pre-market: runtime is cleanly stopped with locks released. Live execution remains blocked.",
                "services.operator_console_status",
                "Complete next-session preparation (paper arm + protected Start) before activation.",
                bucket=BUCKET_INFO,
                classification_code="EXPECTED_PREMARKET_STOP",
            )
        )

    # After-hours component still running (not a failure solely for closed market).
    if market_open is False and phase in (SESSION_AFTER_HOURS, SESSION_NON_TRADING_DAY):
        if cont_ok and cont_n == 1:
            items.append(
                AttentionItem(
                    "INFO",
                    "Continuous loop still running after hours",
                    "Continuous loop process count is 1 while the market is closed. This is not classified as a missing-runtime failure.",
                    "services.runtime_process_detector",
                    "Review whether after-hours runtime is intentional; use protected Stop if a clean stop is desired.",
                    bucket=BUCKET_INFO,
                    classification_code="AFTER_HOURS_RUNTIME_ACTIVE",
                )
            )
        if wd_ok and wd_n == 1:
            items.append(
                AttentionItem(
                    "INFO",
                    "Watchdog still running after hours",
                    "Watchdog process count is 1 while the market is closed. This is not classified as a missing-runtime failure.",
                    "services.runtime_process_detector",
                    "Review whether after-hours runtime is intentional; use protected Stop if a clean stop is desired.",
                    bucket=BUCKET_INFO,
                    classification_code="AFTER_HOURS_RUNTIME_ACTIVE",
                )
            )

    # --- Paper arm (session-aware) ---
    paper_status = str(paper.get("display_status") or paper.get("raw_status") or "").upper()
    arm_through_close = _paper_arm_valid_through_close(
        expires_at=paper.get("expires_at"),
        now=(
            _parse_dt(snapshot_parts.get("generated_at"))
            if snapshot_parts.get("generated_at")
            else None
        ),
    )

    if paper_status == "INVALID" or paper_status == "UNKNOWN":
        items.append(
            AttentionItem(
                "HIGH" if paper_status == "INVALID" else "MEDIUM",
                "Paper arm evidence integrity warning",
                "Paper-arm state is invalid or unknown and cannot be treated as an expected after-hours expiration.",
                str(paper.get("evidence_path") or PAPER_ARM_CONFIRM),
                "Inspect the confirm artifact. Do not assume paper trading is authorized.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="PAPER_ARM_EVIDENCE_INTEGRITY",
            )
        )
    elif paper_status == "EXPIRED":
        if market_open is True or phase == SESSION_MARKET_OPEN:
            items.append(
                AttentionItem(
                    "HIGH",
                    "Paper arm expired",
                    "Paper arm is expired while the market session is open. Paper entry is not safely armed.",
                    str(paper.get("evidence_path") or PAPER_ARM_CONFIRM),
                    "Renew the paper arm via the approved CLI before paper activation. Do not use the dashboard to arm.",
                    bucket=BUCKET_ACTION_NOW,
                    classification_code="PAPER_ARM_EXPIRED_MARKET_OPEN",
                )
            )
        else:
            items.append(
                AttentionItem(
                    "INFO",
                    "Paper arm expired after the trading session",
                    "Paper arm expired after the trading session. Renew before the next market-open activation.",
                    str(paper.get("evidence_path") or PAPER_ARM_CONFIRM),
                    "Renew with the approved paper-arm CLI before the next session. Do not claim paper trading is currently active.",
                    bucket=BUCKET_BEFORE_OPEN,
                    classification_code="PAPER_ARM_EXPIRED_AFTER_HOURS",
                )
            )
    elif paper_status == "MISSING":
        if market_open is True or phase == SESSION_MARKET_OPEN:
            items.append(
                AttentionItem(
                    "HIGH",
                    "Paper arm missing",
                    "No paper-arm latch was found during the active market session.",
                    str(paper.get("evidence_path") or PAPER_ARM_CONFIRM),
                    "Create a latch using the approved paper-arm CLI outside this dashboard.",
                    bucket=BUCKET_ACTION_NOW,
                    classification_code="PAPER_ARM_MISSING_MARKET_OPEN",
                )
            )
        else:
            items.append(
                AttentionItem(
                    "INFO",
                    "Paper arm missing for next session",
                    "No paper-arm latch is present. Renew/create before the next market-open activation.",
                    str(paper.get("evidence_path") or PAPER_ARM_CONFIRM),
                    "Create a latch using the approved paper-arm CLI before unattended paper activation.",
                    bucket=BUCKET_BEFORE_OPEN,
                    classification_code="PAPER_ARM_MISSING_NEXT_SESSION",
                )
            )
    elif paper_status == "ARMED" and phase == SESSION_PRE_MARKET and arm_through_close is False:
        items.append(
            AttentionItem(
                "MEDIUM",
                "Paper arm not valid through market close",
                "Paper arm is present but does not extend through the coming regular-session close.",
                str(paper.get("evidence_path") or PAPER_ARM_CONFIRM),
                "Renew with a TTL that covers the session before activation. Do not report READY for unattended paper entry.",
                bucket=BUCKET_BEFORE_OPEN,
                classification_code="PAPER_ARM_NOT_THROUGH_CLOSE",
            )
        )

    # Next-session start preparation when cleanly stopped after hours / pre-market.
    if expected_stop or clean_premarket_stop:
        items.append(
            AttentionItem(
                "INFO",
                "Start TRITON before next market-open activation",
                "Runtime is intentionally stopped. Use the protected Start control at the appropriate time; Start eligibility follows existing rules.",
                "dashboard.operator_console",
                "Do not start early unless intentionally operating. Confirm paper arm first when paper entries are planned.",
                bucket=BUCKET_BEFORE_OPEN,
                classification_code="NEXT_SESSION_START_PREP",
            )
        )

    # --- Heartbeat / freshness when stopped vs running ---
    if freshness.get("class") == "STALE":
        if (
            expected_stop
            or clean_premarket_stop
            or (cont_n == 0 and wd_n == 0 and cont_ok and wd_ok)
        ):
            items.append(
                AttentionItem(
                    "INFO",
                    "Heartbeat inactive while runtime is stopped",
                    "Heartbeat/freshness is stale or inactive because runtime components are confirmed stopped. Not an active-session stale-runtime incident.",
                    str(freshness.get("evidence_path") or HEARTBEAT),
                    "No heartbeat remediation required for a verified stopped state.",
                    bucket=BUCKET_INFO,
                    classification_code="HEARTBEAT_INACTIVE_STOPPED",
                )
            )
        else:
            items.append(
                AttentionItem(
                    "MEDIUM",
                    "Stale market data",
                    "Data freshness classification is STALE based on heartbeat/freshness evidence.",
                    str(freshness.get("evidence_path") or HEARTBEAT),
                    "Allow the data-freshness scheduled task to run, or inspect freshness audit artifacts.",
                    bucket=BUCKET_ACTION_NOW,
                    classification_code="DATA_FRESHNESS_STALE",
                )
            )

    # Running + stale is handled by Operations cards; keep attention if a loop is up with overall STALE/DEGRADED.
    if (cont_n == 1 or wd_n == 1) and str(snapshot_parts.get("overall_status") or "").upper() in (
        "STALE",
        "DEGRADED",
    ):
        items.append(
            AttentionItem(
                "HIGH",
                "Runtime heartbeat degraded while process is running",
                "At least one runtime process is present but overall runtime health is STALE/DEGRADED.",
                "services.operator_console_status",
                "Inspect process cards, locks, and heartbeats. Do not ignore stale evidence for a live process.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="RUNNING_PROCESS_STALE_HEALTH",
            )
        )

    if cycle.get("overdue") and (cont_n or 0) >= 1:
        items.append(
            AttentionItem(
                "MEDIUM",
                "Cycle overdue",
                f"Last evidenced continuous cycle is older than {CONTINUOUS_CYCLE_OVERDUE_MINUTES:.0f} minutes.",
                str(cycle.get("evidence_path") or ""),
                "Check continuous-loop health and logs.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="CYCLE_OVERDUE",
            )
        )

    if broker.get("connected") is False and (wd_n or 0) >= 1:
        items.append(
            AttentionItem(
                "HIGH",
                "Broker disconnected",
                "Watchdog/broker evidence indicates the broker is not connected.",
                str(WATCHDOG_STATUS),
                "Verify broker connectivity outside the dashboard; do not place orders.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="BROKER_DISCONNECTED",
            )
        )

    # Open orders — never suppress after hours.
    try:
        oo = orders.get("open_orders")
        oo_n = int(oo) if oo is not None else 0
    except (TypeError, ValueError):
        oo_n = 0
    if oo_n > 0:
        items.append(
            AttentionItem(
                "HIGH",
                "Open orders require attention",
                f"{oo_n} open order(s) are present in the verified snapshot.",
                str(orders.get("open_orders_path") or OPEN_ORDERS),
                "Review Orders evidence. Do not suppress because the market is closed.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="OPEN_ORDERS_REQUIRE_ATTENTION",
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
                bucket=BUCKET_ACTION_NOW,
                classification_code="CRITICAL_ALERT_ACTIVE",
            )
        )

    if gov.get("blocked") and not expected_stop:
        items.append(
            AttentionItem(
                "HIGH",
                "Governance block",
                "Governance authorization is not fully authorized.",
                str(GOVERNANCE_AUTH),
                "Review Governance Authorization evidence; do not override from the dashboard.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="GOVERNANCE_BLOCK",
            )
        )

    if gov.get("execution_readiness_ready") is False and (
        market_open is True or phase in (SESSION_MARKET_OPEN, SESSION_PRE_MARKET)
    ):
        items.append(
            AttentionItem(
                "MEDIUM",
                "Execution readiness not ready",
                "execution_readiness.json does not indicate a ready posture.",
                str(EXECUTION_READINESS),
                "Review readiness details on Risk & Alerts / Diagnostics before activation.",
                bucket=BUCKET_BEFORE_OPEN if phase == SESSION_PRE_MARKET else BUCKET_ACTION_NOW,
                classification_code="EXECUTION_READINESS_NOT_READY",
            )
        )

    if phase == SESSION_UNKNOWN and market_open is None:
        items.append(
            AttentionItem(
                "HIGH",
                "Market-session evidence unavailable",
                "Authoritative market-session context could not be classified. After-hours assumptions are refused.",
                "services.command_center_status_coherence.resolve_market_status",
                "Refresh Operations evidence and confirm the market clock/classifier before treating the stop as expected.",
                bucket=BUCKET_ACTION_NOW,
                classification_code="SESSION_EVIDENCE_UNKNOWN",
            )
        )

    # Deduplicate titles within the same bucket+severity (keep first).
    seen = set()
    deduped: List[AttentionItem] = []
    for item in items:
        key = (item.bucket, item.severity, item.title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def group_attention_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Split attention into action-now / before-open / informational groups."""
    now_items: List[Dict[str, Any]] = []
    before: List[Dict[str, Any]] = []
    info: List[Dict[str, Any]] = []
    for raw in items or []:
        item = dict(raw)
        bucket = str(item.get("bucket") or BUCKET_ACTION_NOW)
        sev = str(item.get("severity") or "").upper()
        if sev in ("CRITICAL", "HIGH"):
            item["bucket"] = BUCKET_ACTION_NOW
            now_items.append(item)
        elif bucket == BUCKET_BEFORE_OPEN:
            before.append(item)
        elif bucket == BUCKET_INFO or sev == "INFO":
            item["bucket"] = BUCKET_INFO
            info.append(item)
        else:
            now_items.append(item)
    return {
        "action_required_now": now_items,
        "before_next_market_open": before,
        "informational": info,
        "action_required_now_count": len(now_items),
        "before_next_market_open_count": len(before),
        "informational_count": len(info),
    }


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
            "watchdog_detection_method": "fixture",
            "continuous_detection_method": "fixture",
            "process_detection_confidence": "high",
            "process_detection_warnings": [],
            "watchdog_lock_classification": wd_class,
            "continuous_lock_classification": cl_class,
            "watchdog_lock_healthy": is_lock_healthy(wd_class),
            "continuous_lock_healthy": is_lock_healthy(cl_class),
            "watchdog_lock_exists": paths["watchdog_lock"].is_file(),
            "continuous_lock_exists": paths["continuous_lock"].is_file(),
        }

    # Paper arm — always the canonical execution classifier.
    try:
        from services.paper_arm_status import classify_paper_arm_confirm_status

        paper_raw = classify_paper_arm_confirm_status(root=root, now=now)
    except Exception:
        paper_raw = "UNKNOWN"
    paper_display = map_paper_arm_display(paper_raw)
    paper_expires_at = None
    paper_doc = _read_json(paths["paper_arm"]) if paths["paper_arm"].is_file() else {}
    if paper_doc and not paper_doc.get("__malformed__"):
        paper_expires_at = paper_doc.get("expires_at")

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

    # Cycle evidence from continuous lock refreshed_at (not started_at ownership time)
    cl_lock = _parse_lock_file(paths["continuous_lock"])
    wd_lock = _parse_lock_file(paths["watchdog_lock"])
    cycle_num = cl_lock.get("cycle")
    refreshed = _parse_lock_dt(cl_lock.get("refreshed_at"))
    started_lock = _parse_lock_dt(cl_lock.get("started_at"))
    cycle_age = _coherent_age_minutes(refreshed, now)
    cycle_overdue = cycle_is_overdue(refreshed, now=now)
    lock_deadline = cycle_deadline_utc(refreshed)
    cadence_doc = load_cadence_artifact(root)
    cadence_due = authoritative_next_due_utc(root)
    # Loop cadence artifact is authoritative; lock start+interval is fallback only.
    deadline = cadence_due if cadence_due is not None else lock_deadline
    next_expected = cadence_iso_utc(deadline) if deadline is not None else None
    next_expected_et = cadence_iso_et(deadline) if deadline is not None else None
    next_expected_source = "continuous_loop_cadence" if cadence_due is not None else "lock_fallback"

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

    broker_evidence_ts = (
        wd_status.get("timestamp")
        or wd_status.get("as_of")
        or wd_status.get("generated_at")
        or (
            (wd_status.get("heartbeat") or {}).get("timestamp")
            if isinstance(wd_status.get("heartbeat"), dict)
            else None
        )
    )
    if wd_status.get("__malformed__") or not paths["watchdog_status"].is_file():
        broker_connected = None
        broker_evidence_ts = None
    broker_rest = classify_paper_broker_rest_evidence(
        broker_connected=broker_connected,
        evidence_timestamp=broker_evidence_ts,
        now=now,
        fresh_minutes=BROKER_FRESH_MINUTES,
    )

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

    open_orders_mtime = _file_mtime(paths["open_orders"])
    oo_ops_fresh = classify_open_orders_evidence_freshness(
        evidence_timestamp=open_orders_mtime.isoformat() if open_orders_mtime else None,
        now=now,
    )
    # Operations metric: verified count only when snapshot evidence is fresh.
    if oo_ops_fresh.get("freshness") == "FRESH" and paths["open_orders"].is_file():
        open_orders = _csv_row_count(paths["open_orders"])
    else:
        open_orders = None
    positions = _csv_row_count(paths["positions"])

    auth_status = authorization_status_text(
        gov.get("overall_authorization")
        if gov.get("overall_authorization") is not None
        else gov.get("authorization_status")
    )
    gov_blocked = bool(gov) and auth_status.upper() not in (
        "",
        "AUTHORIZED",
        "GREEN",
        "APPROVED",
        "PASS",
        "TRUE",
    )
    # Boolean True overall_authorization is permitted, not a block.
    if gov.get("overall_authorization") is True:
        gov_blocked = False
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
    process_overall = overall
    try:
        from services.paper_runtime_session_gate import load_trading_cycle_health

        trading_health = load_trading_cycle_health(root=root)
    except Exception:
        trading_health = {
            "trading_pipeline_status": "UNKNOWN",
            "exists": False,
        }
    trading_pipeline_status = str(
        trading_health.get("trading_pipeline_status") or "UNKNOWN"
    ).upper()
    if process_overall == "RUNNING" and trading_pipeline_status == "DEGRADED":
        overall = "DEGRADED"

    cl_pids = list(proc.get("detected_continuous_process_ids") or [])
    wd_pids = list(proc.get("detected_watchdog_process_ids") or [])
    cl_method = str(proc.get("continuous_detection_method") or "unknown")
    wd_method = str(proc.get("watchdog_detection_method") or "unknown")
    cl_discovery_ok = cl_method not in ("failed", "unavailable")
    wd_discovery_ok = wd_method not in ("failed", "unavailable")
    # Low confidence alone does not hide a successful PID scan; only failed/unavailable methods do.
    if str(proc.get("process_detection_confidence") or "").lower() == "low":
        if cl_method in ("failed", "unavailable"):
            cl_discovery_ok = False
        if wd_method in ("failed", "unavailable"):
            wd_discovery_ok = False

    parts = {
        "overall_status": overall,
        "continuous": {
            "process_running": (
                bool(proc.get("continuous_loop_running")) if cl_discovery_ok else None
            ),
            "process_count": len(cl_pids) if cl_discovery_ok else None,
            "pids": cl_pids if cl_discovery_ok else [],
            "process_discovery_available": cl_discovery_ok,
            "process_detection_method": cl_method,
            "lock_exists": bool(proc.get("continuous_lock_exists")),
            "lock_classification": proc.get("continuous_lock_classification"),
            "lock_healthy": bool(proc.get("continuous_lock_healthy")),
            "lock_path": str(paths["continuous_lock"]),
            "lock_pid": cl_lock.get("pid"),
            "cycle": cycle_num,
            "refreshed_at": refreshed.isoformat() if refreshed else None,
        },
        "watchdog": {
            "process_running": bool(proc.get("watchdog_loop_running")) if wd_discovery_ok else None,
            "process_count": len(wd_pids) if wd_discovery_ok else None,
            "pids": wd_pids if wd_discovery_ok else [],
            "process_discovery_available": wd_discovery_ok,
            "process_detection_method": wd_method,
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
            "currently_valid": paper_raw == "VALID",
            "expires_at": paper_expires_at,
            "evidence_path": str(paths["paper_arm"]),
            "exists": paths["paper_arm"].is_file(),
        },
        "process_overall_status": process_overall,
        "trading_pipeline": {
            "status": trading_pipeline_status,
            "last_outcome": trading_health.get("last_outcome"),
            "last_market_hours_outcome": trading_health.get("last_market_hours_outcome"),
            "last_market_hours_at": trading_health.get("last_market_hours_at"),
            "last_reason": trading_health.get("last_reason"),
            "operator_attention_code": trading_health.get("operator_attention_code"),
            "operator_attention_title": trading_health.get("operator_attention_title"),
            "watchdog_does_not_override": True,
            "evidence_path": trading_health.get("evidence_path"),
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
            # Backward-compatible boolean: True/False only when evidence is fresh.
            "connected": broker_rest.get("connected"),
            "state": broker_rest.get("state"),
            "display_label": broker_rest.get("display_label"),
            "evidence_timestamp": broker_rest.get("evidence_timestamp"),
            "evidence_age_minutes": broker_rest.get("evidence_age_minutes"),
            "freshness": broker_rest.get("freshness"),
            "fresh_minutes_threshold": broker_rest.get("fresh_minutes_threshold"),
            "mode": broker_rest.get("mode"),
            "meaning": broker_rest.get("meaning"),
            "streaming_assessed": False,
            "streaming_connected": None,
            "evidence_path": str(paths["watchdog_status"]),
            "open_orders_mtime": (
                _file_mtime(paths["open_orders"]).isoformat()
                if _file_mtime(paths["open_orders"])
                else None
            ),
        },
        "market": (
            lambda _m: (
                lambda _ctx: {
                    **_m,
                    **_ctx,
                    "source": _m.get("source") or _ctx.get("source"),
                }
            )(classify_operator_session_context(now=now, market_status=_m))
        )(resolve_market_status(now=now)),
        "cycle": {
            "last_cycle_number": cycle_num,
            "last_completed_at": refreshed.isoformat() if refreshed else None,
            "age_minutes": cycle_age,
            "overdue": cycle_overdue,
            "next_expected_at": next_expected,
            "next_expected_at_utc": next_expected,
            "next_expected_at_et": next_expected_et,
            "next_expected_timezone": "UTC",
            "next_expected_source": next_expected_source,
            "cadence_phase": (cadence_doc or {}).get("phase"),
            "interval_minutes": CONTINUOUS_INTERVAL_MINUTES,
            "overdue_threshold_minutes": CONTINUOUS_CYCLE_OVERDUE_MINUTES,
            "started_at": started_lock.isoformat() if started_lock else None,
            "started_at_is_heartbeat": False,
            "evidence_path": str(paths["continuous_lock"]),
            "evidence_field": "refreshed_at",
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
    parts["generated_at"] = now.isoformat()
    attention = [a.to_dict() for a in _build_attention(parts)]
    attention_groups = group_attention_items(attention)
    expected_runtime = classify_expected_after_hours_stop(parts)

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
    out = snap.to_dict()
    out["process_overall_status"] = process_overall
    out["trading_pipeline"] = parts.get("trading_pipeline") or {}
    cert_path = results / "capital_preservation_certification.json"
    cert_doc = _read_json(cert_path)
    paper_cert = format_paper_certification_display(cert_doc.get("certification_status"))
    live_perm = format_live_permission_display(live_permitted=live_permitted)
    shutdown = resolve_shutdown_presentation(root=root, now=now)
    out["paper_certification"] = {
        **paper_cert,
        "evidence_path": str(cert_path),
    }
    out["live_permission"] = live_perm
    out["shutdown_presentation"] = shutdown
    out["attention_groups"] = attention_groups
    out["runtime_expectation"] = expected_runtime
    # Expired/historical shutdown must never alter overall runtime health.
    if shutdown.get("affects_current_health") is False:
        out["shutdown_affects_current_health"] = False
    return out


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


def _ingest_order_csv_rows(path: Path, source: str, *, limit: int = 500) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    try:
        import pandas as pd

        df = pd.read_csv(path)
        for _, row in df.head(limit).iterrows():
            d = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            status = normalize_order_status(
                d.get("status") or d.get("order_status") or d.get("state")
            )
            rows.append(
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
                    "snapshot_ts": d.get("snapshot_ts"),
                    "evidence_source": source,
                    "evidence_path": str(path),
                }
            )
    except Exception:
        return rows
    return rows


def build_orders_summary(
    *,
    root: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Read-only paper/order visibility with separated current vs historical contracts."""
    now = now or _utc_now()
    root = Path(root) if root is not None else ROOT
    results = root / "data" / "results"
    paths = {
        "open_orders": results / "open_orders_snapshot.csv",
        "recent_orders": results / "recent_orders.csv",
        "paper_execution_audit": results / "paper_execution_audit.csv",
        "live_orders": results / "live_orders.csv",
    }

    # Current open orders: snapshot artifact only (never historical SUBMITTED audits).
    current_rows = _ingest_order_csv_rows(paths["open_orders"], "open_orders_snapshot")
    snapshot_ts_candidates = [
        _parse_dt(r.get("snapshot_ts") or r.get("submitted_time")) for r in current_rows
    ]
    snapshot_ts_candidates = [t for t in snapshot_ts_candidates if t is not None]
    evidence_ts = (
        max(snapshot_ts_candidates) if snapshot_ts_candidates else _file_mtime(paths["open_orders"])
    )
    oo_fresh = classify_open_orders_evidence_freshness(
        evidence_timestamp=evidence_ts.isoformat() if evidence_ts is not None else None,
        now=now,
    )
    path_exists = paths["open_orders"].is_file()
    freshness = oo_fresh["freshness"]
    count: Optional[int]
    if not path_exists:
        count = None
        message = "Current open-order status unavailable."
    elif freshness == "STALE":
        # Stale evidence must not claim a verified current zero/open count.
        count = None
        message = (
            "Current open-order status is stale. "
            "Do not treat this snapshot as live broker working orders."
        )
    elif freshness == "UNKNOWN":
        count = None
        message = "Current open-order status unavailable."
    else:
        count = len(current_rows)
        message = "No current open orders." if count == 0 else None

    current_open = {
        "rows": current_rows if freshness == "FRESH" else [],
        "count": count,
        "details_available": bool(current_rows) and freshness == "FRESH",
        "freshness": freshness,
        "evidence_timestamp": oo_fresh.get("evidence_timestamp"),
        "evidence_age_minutes": oo_fresh.get("evidence_age_minutes"),
        "fresh_minutes_threshold": oo_fresh.get("fresh_minutes_threshold"),
        "evidence_path": str(paths["open_orders"]),
        "message": message,
        "note": (
            "Current open orders come only from open_orders_snapshot.csv. "
            "Historical SUBMITTED audit rows are never treated as currently open."
        ),
    }

    # Submission history: audit / recent / live_orders — not current open book.
    history_rows: List[Dict[str, Any]] = []
    for key, source in (
        ("paper_execution_audit", "paper_execution_audit"),
        ("recent_orders", "recent_orders"),
        ("live_orders", "live_orders"),
    ):
        for r in _ingest_order_csv_rows(paths[key], source):
            recorded = r.get("status")
            final_status = "Final broker status unavailable."
            if recorded in ("FILLED", "REJECTED", "CANCELLED"):
                final_status = recorded
            elif recorded == "SUBMITTED":
                # Historical submission recorded — not Open/Working/current.
                final_status = "Final broker status unavailable."
            history_rows.append(
                {
                    **r,
                    "recorded_submission_status": recorded,
                    "ui_classification": (
                        "Historical submission"
                        if recorded == "SUBMITTED"
                        else (recorded or "UNKNOWN")
                    ),
                    "final_broker_status": final_status,
                }
            )

    by_status: Dict[str, List[Dict[str, Any]]] = {s: [] for s in ORDER_STATUSES}
    for o in history_rows:
        by_status.setdefault(o["status"], []).append(o)

    submission_history = {
        "rows": history_rows,
        "by_status": {k: len(v) for k, v in by_status.items()},
        "submitted_rows": [r for r in history_rows if r.get("status") == "SUBMITTED"],
        "note": (
            "SUBMITTED means a historical submission was recorded. "
            "It is not current broker open/working status and does not prove a final outcome."
        ),
        "evidence_paths": {
            "paper_execution_audit": str(paths["paper_execution_audit"]),
            "recent_orders": str(paths["recent_orders"]),
            "live_orders": str(paths["live_orders"]),
        },
    }

    return {
        "current_open_orders": current_open,
        "submission_history": submission_history,
        # Backward-compatible fields: open_orders is current snapshot only (never audit SUBMITTED).
        "orders": history_rows,
        "by_status": submission_history["by_status"],
        "open_orders": list(current_open["rows"]),
        "filled_orders": by_status.get("FILLED", []),
        "rejected_orders": by_status.get("REJECTED", []),
        "cancelled_orders": by_status.get("CANCELLED", []),
        "evidence_paths": {k: str(v) for k, v in paths.items()},
        "note": (
            "Current open orders and submission history are separate evidence contracts. "
            "PLANNED/AUTHORIZED are distinct from SUBMITTED/FILLED; status is never inferred upward. "
            "SUBMITTED is historical submission activity, not Open/Working."
        ),
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
