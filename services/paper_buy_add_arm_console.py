"""Operator Console orchestration for paper BUY/ADD arming.

Calls services.set_paper_arm_confirm.write_confirm — the same production writer
as the CLI. Does not start TRITON, submit orders, or touch SELL authority.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from services.set_paper_arm_confirm import (
    CONFIRM_PATH,
    MAX_SUPPORTED_TTL_MINUTES,
    write_confirm,
)

ARM_PHRASE = "ARM PAPER ENTRIES"
REVOKE_PHRASE = "REVOKE PAPER ENTRIES"
NO_START_CAPTION = "Arming does not start TRITON, start the continuous loop, or submit orders."
PRODUCTION_PATH_OVERRIDE_ENVS = (
    "TRITON_PAPER_ARM_CONFIRM_PATH",
    "TRITON_PAPER_ARM_CONFIRM_ROOT",
    "TRITON_TEST_PAPER_ARM_CONFIRM",
)

FAIL_CLOSED = {
    "phrase_mismatch": "Confirmation phrase does not match ARM PAPER ENTRIES.",
    "revoke_phrase_mismatch": "Confirmation phrase does not match REVOKE PAPER ENTRIES.",
    "live_execution_not_blocked": "Live execution is not blocked. Paper BUY/ADD arming is refused.",
    "non_paper_mode": "Paper mode is required. Paper BUY/ADD arming is refused.",
    "calendar_unavailable": "Market calendar is unavailable. Paper BUY/ADD arming is refused.",
    "calendar_unknown": "Market session is unknown. Paper BUY/ADD arming is refused.",
    "weekend": "Weekend arming is refused.",
    "non_trading_day": "Non-trading-day arming is refused.",
    "outside_session_window": "Outside the permitted paper-session arming window.",
    "configuration_blocked": "Paper BUY/ADD arming is blocked by a non-production path override.",
    "canonical_path_required": "Paper BUY/ADD arming requires the canonical authorization path.",
    "write_failed": "Paper BUY/ADD arm write failed closed.",
    "consumer_rejected": "Authorization consumer rejected the written artifact.",
}


def production_path_overrides_present() -> bool:
    return any(str(os.environ.get(key) or "").strip() for key in PRODUCTION_PATH_OVERRIDE_ENVS)


def canonical_confirm_path() -> Path:
    return CONFIRM_PATH


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(now: Optional[datetime]) -> datetime:
    clock = now if now is not None else _utc_now()
    if clock.tzinfo is None:
        return clock.replace(tzinfo=timezone.utc)
    return clock.astimezone(timezone.utc)


def _live_blocked(*, live_blocked: Optional[bool] = None) -> bool:
    if live_blocked is not None:
        return bool(live_blocked)
    try:
        from services.live_block_classifier import evaluate_live_block_status

        return bool(evaluate_live_block_status(paper_mode=True).get("live_blocked"))
    except Exception:
        return True


def _consumer_status(*, confirm_path: Path, now: Optional[datetime] = None) -> str:
    from services.paper_arm_status import classify_paper_arm_confirm_status

    return classify_paper_arm_confirm_status(confirm_path=confirm_path, now=now)


def _display_status(canonical: str, *, allow_execute: Optional[bool] = None) -> str:
    from services.operator_console_status import map_paper_arm_display

    if allow_execute is False:
        return "REVOKED"
    mapped = map_paper_arm_display(canonical)
    if mapped == "VALID":
        return "ARMED"
    return mapped


def _read_doc(path: Path) -> Dict[str, Any]:
    import json

    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _session_payload(now: datetime, session_state_fn: Optional[Callable] = None) -> Dict[str, Any]:
    fn = (
        session_state_fn
        or __import__(
            "services.market_calendar", fromlist=["regular_session_state"]
        ).regular_session_state
    )
    payload = fn(now)
    return payload if isinstance(payload, dict) else {}


def _session_limit_utc(now: datetime, session: Dict[str, Any]) -> Optional[datetime]:
    from services.paper_session_arm_policy import load_policy

    policy = load_policy()
    post = int(policy.get("post_market_expiry_minutes", 60) or 60)
    close_raw = session.get("session_close_et")
    if close_raw:
        try:
            close_dt = datetime.fromisoformat(str(close_raw).replace("Z", "+00:00"))
            if close_dt.tzinfo is None:
                close_dt = close_dt.replace(tzinfo=timezone.utc)
            return close_dt.astimezone(timezone.utc) + timedelta(minutes=post)
        except Exception:
            pass
    from services.paper_session_arm_policy import is_within_session_arm_window

    in_window, _reason = is_within_session_arm_window(policy, now)
    if not in_window:
        return None
    now_et = now.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
    end_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0) + timedelta(minutes=post)
    return end_et.astimezone(timezone.utc)


def _ttl_minutes(now: datetime, session: Dict[str, Any]) -> int:
    limit = _session_limit_utc(now, session)
    if limit is None or limit <= now:
        return 0
    seconds = (limit - now).total_seconds()
    minutes = int(seconds // 60)
    return max(0, min(minutes, MAX_SUPPORTED_TTL_MINUTES))


def evaluate_eligibility(
    *,
    now: Optional[datetime] = None,
    paper_mode: Optional[bool] = None,
    live_blocked: Optional[bool] = None,
    confirm_path: Optional[Path] = None,
    allow_path_override: bool = False,
    session_state_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    clock = _aware(now)
    rec: Dict[str, Any] = {
        "ok": False,
        "arm_enabled": False,
        "reason": "",
        "session_date": None,
        "ttl_minutes": 0,
        "session_state": "",
        "session_reason": "",
        "live_blocked": _live_blocked(live_blocked=live_blocked),
        "paper_mode": True if paper_mode is None else bool(paper_mode),
    }
    if production_path_overrides_present() and not allow_path_override:
        rec["reason"] = "configuration_blocked"
        return rec
    if confirm_path is not None and not allow_path_override:
        rec["reason"] = "canonical_path_required"
        return rec
    if not rec["paper_mode"]:
        rec["reason"] = "non_paper_mode"
        return rec
    if not rec["live_blocked"]:
        rec["reason"] = "live_execution_not_blocked"
        return rec

    session = _session_payload(clock, session_state_fn)
    state = str(session.get("session_state") or "UNKNOWN").upper()
    sreason = str(session.get("reason") or "")
    rec["session_state"] = state
    rec["session_reason"] = sreason
    rec["session_date"] = session.get("session_date")
    if state == "UNKNOWN" or sreason.startswith("calendar_"):
        rec["reason"] = (
            "calendar_unavailable"
            if "unavailable" in sreason or "import" in sreason
            else "calendar_unknown"
        )
        return rec
    if sreason == "weekend" or (state == "CLOSED" and sreason == "weekend"):
        rec["reason"] = "weekend"
        return rec
    if sreason == "holiday":
        rec["reason"] = "non_trading_day"
        return rec

    from services.paper_session_arm_policy import (
        is_near_market_open_et,
        is_within_session_arm_window,
        load_policy,
    )

    policy = load_policy()
    in_window, window_reason = is_within_session_arm_window(policy, clock)
    pre_ok = is_near_market_open_et(
        clock, pre_market_minutes=int(policy.get("pre_market_arm_minutes", 45) or 45)
    )
    if state == "OPEN":
        pass
    elif sreason == "pre_open" and (pre_ok or in_window):
        pass
    elif sreason in {"regular_close", "early_close", "post_market_grace"} and in_window:
        pass
    else:
        rec["reason"] = "outside_session_window" if window_reason else "outside_session_window"
        rec["session_reason"] = window_reason or sreason
        return rec

    ttl = _ttl_minutes(clock, session)
    if ttl < 1:
        rec["reason"] = "outside_session_window"
        return rec
    rec["ok"] = True
    rec["arm_enabled"] = True
    rec["ttl_minutes"] = ttl
    rec["reason"] = "ok"
    return rec


def build_presentation(
    *,
    now: Optional[datetime] = None,
    paper_mode: Optional[bool] = None,
    live_blocked: Optional[bool] = None,
    confirm_path: Optional[Path] = None,
    allow_path_override: bool = False,
    session_state_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    clock = _aware(now)
    dest = Path(confirm_path) if confirm_path is not None else canonical_confirm_path()
    elig = evaluate_eligibility(
        now=clock,
        paper_mode=paper_mode,
        live_blocked=live_blocked,
        confirm_path=confirm_path,
        allow_path_override=allow_path_override,
        session_state_fn=session_state_fn,
    )
    doc = _read_doc(dest)
    consumer = _consumer_status(confirm_path=dest, now=clock)
    allow = doc.get("allow_execute") if doc else None
    display = _display_status(consumer, allow_execute=allow if isinstance(allow, bool) else None)
    remaining = None
    exp = doc.get("expires_at")
    if exp:
        try:
            exp_dt = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            remaining = max(0.0, (exp_dt.astimezone(timezone.utc) - clock).total_seconds() / 60.0)
        except Exception:
            remaining = None
    armed = consumer == "VALID"
    return {
        "status": display,
        "consumer_status": consumer,
        "authorized_session_date": elig.get("session_date") or None,
        "created_at": doc.get("timestamp"),
        "expires_at": doc.get("expires_at"),
        "remaining_minutes": remaining,
        "live_execution": "BLOCKED" if elig.get("live_blocked") else "NOT BLOCKED",
        "arm_enabled": bool(elig.get("arm_enabled")),
        "arm_deny_reason": None if elig.get("ok") else elig.get("reason"),
        "button_label": (
            "Renew Paper BUY/ADD Arm" if armed and elig.get("arm_enabled") else "Arm Paper BUY/ADD"
        ),
        "ttl_minutes": elig.get("ttl_minutes") or 0,
        "no_start_caption": NO_START_CAPTION,
        "phrase_label": f"Type exactly: {ARM_PHRASE}",
        "revoke_enabled": dest.is_file() and display != "MISSING",
        "confirm_path_kind": "injected" if confirm_path is not None else "canonical",
    }


def _operator_message(reason: str) -> str:
    return FAIL_CLOSED.get(reason, "Paper BUY/ADD arming is refused.")


def confirm_paper_buy_add_arm(
    *,
    confirmation_phrase: str,
    now: Optional[datetime] = None,
    paper_mode: Optional[bool] = None,
    live_blocked: Optional[bool] = None,
    confirm_path: Optional[Path] = None,
    allow_path_override: bool = False,
    session_state_fn: Optional[Callable] = None,
    idempotency_key: Optional[str] = None,
    last_idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Arm or renew via write_confirm after revalidation. No runtime/order side effects."""
    clock = _aware(now)
    dest = Path(confirm_path) if confirm_path is not None else canonical_confirm_path()
    counts = {
        "writes": 0,
        "orders": 0,
        "runtime_starts": 0,
        "broker": 0,
        "network": 0,
        "sell_challenge": 0,
        "sell_authority": 0,
        "sell_dispatch": 0,
    }
    if idempotency_key and last_idempotency_key and idempotency_key == last_idempotency_key:
        return {
            "ok": False,
            "reason": "duplicate_click",
            "message": "A rerun cannot rewrite paper BUY/ADD authorization.",
            "wrote": False,
            "counts": counts,
        }
    if str(confirmation_phrase or "") != ARM_PHRASE:
        return {
            "ok": False,
            "reason": "phrase_mismatch",
            "message": _operator_message("phrase_mismatch"),
            "wrote": False,
            "counts": counts,
        }
    elig = evaluate_eligibility(
        now=clock,
        paper_mode=paper_mode,
        live_blocked=live_blocked,
        confirm_path=confirm_path,
        allow_path_override=allow_path_override,
        session_state_fn=session_state_fn,
    )
    if not elig.get("ok"):
        return {
            "ok": False,
            "reason": elig.get("reason"),
            "message": _operator_message(str(elig.get("reason") or "write_failed")),
            "wrote": False,
            "counts": counts,
        }
    prior = dest.read_bytes() if dest.is_file() else None
    try:
        write_confirm(
            int(elig["ttl_minutes"]),
            allow_execute=True,
            confirm_path=dest,
            now=clock,
        )
        counts["writes"] = 1
    except Exception:
        if prior is None and dest.is_file():
            try:
                dest.unlink()
            except OSError:
                pass
        return {
            "ok": False,
            "reason": "write_failed",
            "message": _operator_message("write_failed"),
            "wrote": False,
            "counts": counts,
        }
    consumer = _consumer_status(confirm_path=dest, now=clock)
    if consumer != "VALID":
        if prior is not None:
            tmp = dest.with_name(dest.name + ".restore.tmp")
            tmp.write_bytes(prior)
            os.replace(tmp, dest)
        elif dest.is_file():
            try:
                dest.unlink()
            except OSError:
                pass
        counts["writes"] = 0
        return {
            "ok": False,
            "reason": "consumer_rejected",
            "message": _operator_message("consumer_rejected"),
            "wrote": False,
            "consumer_status": consumer,
            "counts": counts,
        }
    doc = _read_doc(dest)
    return {
        "ok": True,
        "reason": "ok",
        "message": "Paper BUY/ADD authorization armed. No orders were submitted.",
        "wrote": True,
        "consumer_status": consumer,
        "display_status": "ARMED",
        "created_at": doc.get("timestamp"),
        "expires_at": doc.get("expires_at"),
        "ttl_minutes": elig.get("ttl_minutes"),
        "session_date": elig.get("session_date"),
        "counts": counts,
        "idempotency_key": idempotency_key,
    }


def revoke_paper_buy_add_arm(
    *,
    confirmation_phrase: str,
    now: Optional[datetime] = None,
    paper_mode: Optional[bool] = None,
    live_blocked: Optional[bool] = None,
    confirm_path: Optional[Path] = None,
    allow_path_override: bool = False,
) -> Dict[str, Any]:
    """Existing CLI disarm: write_confirm(..., allow_execute=False)."""
    clock = _aware(now)
    dest = Path(confirm_path) if confirm_path is not None else canonical_confirm_path()
    counts = {"writes": 0, "orders": 0, "runtime_starts": 0, "broker": 0, "network": 0}
    if str(confirmation_phrase or "") != REVOKE_PHRASE:
        return {
            "ok": False,
            "reason": "revoke_phrase_mismatch",
            "message": _operator_message("revoke_phrase_mismatch"),
            "wrote": False,
            "counts": counts,
        }
    if production_path_overrides_present() and not allow_path_override:
        return {
            "ok": False,
            "reason": "configuration_blocked",
            "message": _operator_message("configuration_blocked"),
            "wrote": False,
            "counts": counts,
        }
    if confirm_path is not None and not allow_path_override:
        return {
            "ok": False,
            "reason": "canonical_path_required",
            "message": _operator_message("canonical_path_required"),
            "wrote": False,
            "counts": counts,
        }
    if paper_mode is False:
        return {
            "ok": False,
            "reason": "non_paper_mode",
            "message": _operator_message("non_paper_mode"),
            "wrote": False,
            "counts": counts,
        }
    if not _live_blocked(live_blocked=live_blocked):
        return {
            "ok": False,
            "reason": "live_execution_not_blocked",
            "message": _operator_message("live_execution_not_blocked"),
            "wrote": False,
            "counts": counts,
        }
    write_confirm(1, allow_execute=False, confirm_path=dest, now=clock)
    counts["writes"] = 1
    return {
        "ok": True,
        "reason": "ok",
        "message": "Paper BUY/ADD authorization revoked. No orders were submitted.",
        "wrote": True,
        "display_status": "REVOKED",
        "counts": counts,
    }
