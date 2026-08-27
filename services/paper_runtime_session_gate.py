"""
Paper-runtime session gate (read-only vs authorization).

Decides whether the continuous loop may invoke the market-open paper-entry sequence.
Does not create or renew paper authorization. Does not contact a broker.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
HEALTH_RELATIVE = Path("data") / "results" / "trading_cycle_health.json"

OUTCOME_OK = "ok"
OUTCOME_FAIL = "fail"
OUTCOME_SKIPPED_MARKET_CLOSED = "skipped_market_closed"
OUTCOME_SKIPPED_NEAR_CLOSE = "skipped_near_close"
OUTCOME_SESSION_UNKNOWN = "session_unknown"

PIPELINE_HEALTHY = "HEALTHY"
PIPELINE_DEGRADED = "DEGRADED"
PIPELINE_INFORMATIONAL_CLOSED = "INFORMATIONAL_CLOSED"
PIPELINE_UNKNOWN = "UNKNOWN"

ACTION_INVOKE_ENTRY = "invoke_entry_sequence"
ACTION_SKIP_CLOSED = "skip_market_closed"
ACTION_SKIP_NEAR_CLOSE = "skip_near_close"
ACTION_BLOCK_UNKNOWN = "block_session_unknown"

LOG_SKIPPED_MARKET_CLOSED = "PAPER_ENTRY_SKIPPED_MARKET_CLOSED"
LOG_SKIPPED_NEAR_CLOSE = "PAPER_ENTRY_SKIPPED_NEAR_CLOSE"
LOG_BLOCKED_SESSION_UNKNOWN = "PAPER_ENTRY_BLOCKED_SESSION_UNKNOWN"
CYCLE_SKIPPED_MARKET_CLOSED = "CYCLE_SKIPPED_MARKET_CLOSED"
CYCLE_SKIPPED_NEAR_CLOSE = "CYCLE_SKIPPED_NEAR_CLOSE"
CYCLE_BLOCKED_SESSION_UNKNOWN = "CYCLE_BLOCKED_SESSION_UNKNOWN"
ATTENTION_SESSION_CALENDAR_UNKNOWN = "SESSION_CALENDAR_UNKNOWN"

# Phase 198T-N/O: max observed successful 2026-08-26 cycle was 121s; +60s buffer.
MINIMUM_RUNWAY_SECONDS = 181
RUNWAY_BASIS = "max_observed_successful_cycle_121s_plus_60s_buffer"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime] = None) -> str:
    clock = dt or _utc_now()
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    else:
        clock = clock.astimezone(timezone.utc)
    return clock.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_now_utc(value: str) -> datetime:
    s = (value or "").strip()
    if not s:
        return _utc_now()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def health_path(root: Optional[Path] = None) -> Path:
    return (Path(root) if root is not None else ROOT) / HEALTH_RELATIVE


def load_trading_cycle_health(*, root: Optional[Path] = None) -> Dict[str, Any]:
    path = health_path(root)
    if not path.is_file():
        return {
            "schema_version": 1,
            "trading_pipeline_status": PIPELINE_UNKNOWN,
            "last_outcome": None,
            "last_market_hours_outcome": None,
            "exists": False,
            "evidence_path": str(path),
        }
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            return {
                "schema_version": 1,
                "trading_pipeline_status": PIPELINE_UNKNOWN,
                "exists": True,
                "malformed": True,
                "evidence_path": str(path),
            }
        doc.setdefault("trading_pipeline_status", PIPELINE_UNKNOWN)
        doc["exists"] = True
        doc["evidence_path"] = str(path)
        return doc
    except Exception:
        return {
            "schema_version": 1,
            "trading_pipeline_status": PIPELINE_UNKNOWN,
            "exists": True,
            "malformed": True,
            "evidence_path": str(path),
        }


def _parse_session_close(session: Dict[str, Any]) -> Optional[datetime]:
    raw = session.get("session_close_et")
    if not raw:
        return None
    text = str(raw).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return None
    return dt


def remaining_runway_seconds(session: Dict[str, Any], *, now: datetime) -> Optional[float]:
    """Seconds remaining until authoritative regular-session close. None if unknown."""
    close = _parse_session_close(session)
    if close is None:
        return None
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (close - now.astimezone(close.tzinfo)).total_seconds()


def decide_paper_entry_cycle_action(*, now: Optional[datetime] = None) -> Dict[str, Any]:
    from services.market_calendar import (
        SESSION_CLOSED,
        SESSION_OPEN,
        SESSION_UNKNOWN,
        regular_session_state,
    )

    clock = now or _utc_now()
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    session = regular_session_state(now=clock)
    state = str(session.get("session_state") or SESSION_UNKNOWN).upper()
    remaining = remaining_runway_seconds(session, now=clock)
    runway_fields = {
        "required_runway_seconds": MINIMUM_RUNWAY_SECONDS,
        "remaining_runway_seconds": remaining,
        "runway_basis": RUNWAY_BASIS,
        "broker_orders_submitted": False,
        "live_execution_permitted": False,
        "as_of": _iso(clock),
        "session_date": session.get("session_date"),
        "session_close_et": session.get("session_close_et"),
    }
    if state == SESSION_OPEN:
        if remaining is None:
            return {
                "action": ACTION_BLOCK_UNKNOWN,
                "session_state": SESSION_UNKNOWN,
                "regular_session_open": False,
                "log_event": LOG_BLOCKED_SESSION_UNKNOWN,
                "cycle_event": CYCLE_BLOCKED_SESSION_UNKNOWN,
                "reason": "session_close_et_unavailable",
                "session": session,
                **runway_fields,
            }
        if remaining < float(MINIMUM_RUNWAY_SECONDS):
            return {
                "action": ACTION_SKIP_NEAR_CLOSE,
                "session_state": SESSION_OPEN,
                "regular_session_open": True,
                "log_event": LOG_SKIPPED_NEAR_CLOSE,
                "cycle_event": CYCLE_SKIPPED_NEAR_CLOSE,
                "reason": CYCLE_SKIPPED_NEAR_CLOSE,
                "session": session,
                **runway_fields,
            }
        return {
            "action": ACTION_INVOKE_ENTRY,
            "session_state": SESSION_OPEN,
            "regular_session_open": True,
            "log_event": None,
            "cycle_event": None,
            "reason": session.get("reason") or "regular_session_open",
            "session": session,
            **runway_fields,
        }
    if state == SESSION_CLOSED:
        return {
            "action": ACTION_SKIP_CLOSED,
            "session_state": SESSION_CLOSED,
            "regular_session_open": False,
            "log_event": LOG_SKIPPED_MARKET_CLOSED,
            "cycle_event": CYCLE_SKIPPED_MARKET_CLOSED,
            "reason": session.get("reason"),
            "session": session,
            **runway_fields,
        }
    return {
        "action": ACTION_BLOCK_UNKNOWN,
        "session_state": SESSION_UNKNOWN,
        "regular_session_open": False,
        "log_event": LOG_BLOCKED_SESSION_UNKNOWN,
        "cycle_event": CYCLE_BLOCKED_SESSION_UNKNOWN,
        "reason": session.get("reason"),
        "session": session,
        **runway_fields,
    }


def record_cycle_outcome(
    *,
    outcome: str,
    reason: str = "",
    session_open: Optional[bool] = None,
    root: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Persist last cycle outcome. After-hours skips never clear a market-hours failure."""
    clock = now or _utc_now()
    existing = load_trading_cycle_health(root=root)
    last_mh = existing.get("last_market_hours_outcome")
    last_mh_at = existing.get("last_market_hours_at")
    last_mh_reason = existing.get("last_market_hours_reason")
    informational_skip = (OUTCOME_SKIPPED_MARKET_CLOSED, OUTCOME_SKIPPED_NEAR_CLOSE)
    outcome_n = str(outcome or "").strip().lower()
    if session_open is None:
        session_open = outcome_n not in (OUTCOME_SKIPPED_MARKET_CLOSED, OUTCOME_SESSION_UNKNOWN)

    pipeline = str(existing.get("trading_pipeline_status") or PIPELINE_UNKNOWN)
    attention_code = existing.get("operator_attention_code")
    attention_title = existing.get("operator_attention_title")
    if outcome_n == OUTCOME_SESSION_UNKNOWN:
        pipeline = PIPELINE_DEGRADED
        attention_code = ATTENTION_SESSION_CALENDAR_UNKNOWN
        attention_title = "Session calendar unavailable — paper entry blocked"
        session_open = False
    elif outcome_n == OUTCOME_SKIPPED_NEAR_CLOSE:
        if attention_code == ATTENTION_SESSION_CALENDAR_UNKNOWN:
            attention_code = None
            attention_title = None
        if last_mh == OUTCOME_FAIL:
            pipeline = PIPELINE_DEGRADED
        elif last_mh == OUTCOME_OK:
            pipeline = PIPELINE_HEALTHY
        else:
            pipeline = PIPELINE_INFORMATIONAL_CLOSED
    elif outcome_n == OUTCOME_SKIPPED_MARKET_CLOSED:
        # Authoritative CLOSED: calendar is available. Keep unresolved market-hours
        # failure; do not keep calendar-unknown attention or mislabel CLOSED as UNKNOWN.
        if attention_code == ATTENTION_SESSION_CALENDAR_UNKNOWN:
            attention_code = None
            attention_title = None
        if last_mh == OUTCOME_FAIL:
            pipeline = PIPELINE_DEGRADED
        else:
            pipeline = PIPELINE_INFORMATIONAL_CLOSED
    elif outcome_n == OUTCOME_FAIL and session_open:
        last_mh = OUTCOME_FAIL
        last_mh_at = _iso(clock)
        last_mh_reason = reason or "market_hours_cycle_fail"
        pipeline = PIPELINE_DEGRADED
    elif outcome_n == OUTCOME_OK and session_open:
        last_mh = OUTCOME_OK
        last_mh_at = _iso(clock)
        last_mh_reason = reason or "market_hours_cycle_ok"
        pipeline = PIPELINE_HEALTHY
        attention_code = None
        attention_title = None
    elif outcome_n == OUTCOME_FAIL:
        # Closed-session failure is not a trading-cycle degradation signal.
        if last_mh == OUTCOME_FAIL:
            pipeline = PIPELINE_DEGRADED
        elif attention_code == ATTENTION_SESSION_CALENDAR_UNKNOWN:
            pipeline = PIPELINE_DEGRADED
        else:
            pipeline = PIPELINE_INFORMATIONAL_CLOSED

    payload = {
        "schema_version": 1,
        "updated_at": _iso(clock),
        "last_outcome": outcome_n,
        "last_reason": reason,
        "last_session_open": bool(session_open),
        "last_market_hours_outcome": last_mh,
        "last_market_hours_at": last_mh_at,
        "last_market_hours_reason": last_mh_reason,
        "trading_pipeline_status": pipeline,
        "operator_attention_code": attention_code,
        "operator_attention_title": attention_title,
        "broker_attempts": (
            0
            if outcome_n in (*informational_skip, OUTCOME_SESSION_UNKNOWN)
            else existing.get("broker_attempts")
        ),
        "authorization_mutated": False,
        "paper_entry_invoked": (
            False
            if outcome_n in (*informational_skip, OUTCOME_SESSION_UNKNOWN)
            else existing.get("paper_entry_invoked")
        ),
        "arm_validation_invoked": (
            False
            if outcome_n in (*informational_skip, OUTCOME_SESSION_UNKNOWN)
            else existing.get("arm_validation_invoked")
        ),
        "broker_orders_submitted": False,
        "live_execution_permitted": False,
    }
    path = health_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["exists"] = True
    payload["evidence_path"] = str(path)
    return payload


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Paper runtime session gate (no broker, no arm mutation)"
    )
    ap.add_argument("--check", action="store_true", help="Print regular-session state JSON")
    ap.add_argument("--json", action="store_true", help="JSON stdout")
    ap.add_argument("--record-cycle", action="store_true")
    ap.add_argument(
        "--outcome",
        default="",
        help="ok | fail | skipped_market_closed | skipped_near_close | session_unknown",
    )
    ap.add_argument("--reason", default="")
    ap.add_argument("--session-open", default="", help="true|false; default inferred from outcome")
    ap.add_argument("--now-utc", default="", help="Synthetic UTC timestamp")
    ap.add_argument("--root", default="", help="Temporary root for isolated tests")
    args = ap.parse_args(argv)

    now = _parse_now_utc(args.now_utc) if args.now_utc else _utc_now()
    root = Path(args.root) if args.root else ROOT

    if args.record_cycle:
        session_open = None
        if str(args.session_open).strip().lower() in ("true", "1", "yes"):
            session_open = True
        elif str(args.session_open).strip().lower() in ("false", "0", "no"):
            session_open = False
        payload = record_cycle_outcome(
            outcome=args.outcome or OUTCOME_SKIPPED_MARKET_CLOSED,
            reason=args.reason,
            session_open=session_open,
            root=root,
            now=now,
        )
        sys.stdout.write(json.dumps(payload) + "\n")
        return 0

    decision = decide_paper_entry_cycle_action(now=now)
    session = decision.get("session") or {}
    out = {
        "session_state": decision.get("session_state") or session.get("session_state"),
        "regular_session_open": bool(decision.get("regular_session_open")),
        "action": decision.get("action"),
        "reason": decision.get("reason") or session.get("reason"),
        "log_event": decision.get("log_event"),
        "cycle_event": decision.get("cycle_event"),
        "session": session,
        "session_date": session.get("session_date"),
        "session_close_et": session.get("session_close_et"),
        "remaining_runway_seconds": decision.get("remaining_runway_seconds"),
        "required_runway_seconds": decision.get("required_runway_seconds"),
        "runway_basis": decision.get("runway_basis"),
        "as_of": decision.get("as_of"),
        "authorization_mutated": False,
        "broker_initialized": False,
        "broker_orders_submitted": False,
        "live_execution_permitted": False,
    }
    if args.json or args.check:
        sys.stdout.write(json.dumps(out) + "\n")
        return 0
    print(f"regular_session_open={out['regular_session_open']} reason={out['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
