"""
paper_session_arm_policy.py
Phase 196U — Bounded paper-session ARM auto-renewal (paper only).

May write data/live/paper_arm_confirm.json only.
Does not write live_confirm, call execute_trades, or submit orders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "paper_arm_session_policy.json"
AUDIT_PATH = ROOT / "data" / "results" / "paper_session_arm_policy_audit.json"
LIVE_CONFIRM_PATH = ROOT / "data" / "live" / "live_confirm.json"

_DEFAULT_POLICY: Dict[str, Any] = {
    "enabled": True,
    "mode": "paper_only",
    "auto_arm_before_market_open": True,
    "auto_renew_during_market_open": True,
    "pre_market_arm_minutes": 45,
    "post_market_expiry_minutes": 60,
    "renew_if_minutes_remaining_below": 90,
    "default_ttl_minutes": 600,
    "max_ttl_minutes": 600,
    "live_execution_allowed": False,
    "require_paper_execution_env": True,
    "require_auto_paper_entries_env": True,
    "require_broker_mode_paper": True,
    "write_audit": True,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str, *, json_mode: bool = False) -> None:
    if json_mode:
        print(msg, file=sys.stderr)
    else:
        print(msg)


def load_policy(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or CONFIG_PATH
    if not p.is_file():
        return dict(_DEFAULT_POLICY)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        out = dict(_DEFAULT_POLICY)
        if isinstance(doc, dict):
            out.update(doc)
        return out
    except Exception:
        return dict(_DEFAULT_POLICY)


def _et_now(now_utc: Optional[datetime] = None) -> datetime:
    from services.guard_auto import _ET

    now = now_utc or _utc_now()
    if _ET is None:
        return now
    return now.astimezone(_ET)


def is_market_open_et(now_utc: Optional[datetime] = None) -> bool:
    from services.guard_auto import is_market_open_et as _open

    return _open(now_utc or _utc_now())


def is_near_market_open_et(
    now_utc: Optional[datetime] = None, *, pre_market_minutes: int = 45
) -> bool:
    """Weekday within pre_market_minutes before 09:30 ET (not yet open)."""
    now = now_utc or _utc_now()
    now_et = _et_now(now)
    if now_et.weekday() >= 5:
        return False
    if is_market_open_et(now):
        return False
    open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    delta_min = (open_et - now_et).total_seconds() / 60.0
    return 0 < delta_min <= float(pre_market_minutes)


def is_within_session_arm_window(
    policy: Dict[str, Any], now_utc: Optional[datetime] = None
) -> Tuple[bool, str]:
    """
    Arm/renew permitted from (9:30 - pre_market) through (16:00 + post_market grace).
    """
    now = now_utc or _utc_now()
    now_et = _et_now(now)
    if now_et.weekday() >= 5:
        return False, "weekend"

    pre = int(policy.get("pre_market_arm_minutes", 45) or 45)
    post = int(policy.get("post_market_expiry_minutes", 60) or 60)
    start_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0) - timedelta(minutes=pre)
    end_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0) + timedelta(minutes=post)
    if now_et < start_et:
        return False, "before_pre_market_window"
    if now_et > end_et:
        return False, "after_post_market_window"
    if is_market_open_et(now):
        return True, "market_open"
    if is_near_market_open_et(now, pre_market_minutes=pre):
        return True, "near_market_open"
    if now_et.hour >= 16 and now_et <= end_et:
        return True, "post_market_grace"
    return False, "outside_session_window"


def _env_ok(policy: Dict[str, Any]) -> Tuple[bool, str]:
    if (
        policy.get("require_paper_execution_env")
        and os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").strip() != "1"
    ):
        return False, "TRITON_ENABLE_PAPER_EXECUTION not set to 1"
    if (
        policy.get("require_auto_paper_entries_env")
        and os.environ.get("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "").strip() != "1"
    ):
        return False, "TRITON_ENABLE_AUTO_PAPER_ENTRIES not set to 1"
    return True, "ok"


def _broker_mode_paper() -> Tuple[bool, str]:
    base = (
        os.environ.get("APCA_API_BASE_URL", "") or os.environ.get("ALPACA_API_BASE_URL", "") or ""
    ).lower()
    if base and "paper-api" in base:
        return True, "paper-api base URL"
    if base and "api.alpaca.markets" in base and "paper" not in base:
        return False, "live alpaca base URL detected"
    try:
        from services.portfolio_history_sync import load_live_execution_flag

        if load_live_execution_flag():
            return False, "live_execution flag true"
        return True, "paper default (live_execution false)"
    except Exception as exc:
        return True, f"assume paper ({exc})"


def _live_execution_blocked(policy: Dict[str, Any]) -> Tuple[bool, str]:
    if policy.get("live_execution_allowed"):
        return False, "policy live_execution_allowed=true forbidden"
    try:
        from services.portfolio_history_sync import load_live_execution_flag

        if load_live_execution_flag():
            return False, "live_execution flag is true"
    except Exception:
        pass
    return True, "live_execution false"


def _minutes_remaining() -> Optional[float]:
    from services.paper_arm_status import classify_paper_arm_confirm_status
    from services.arm_mode import read_paper_arm_confirm, _parse_dt

    doc = read_paper_arm_confirm()
    if not doc:
        return None
    status = classify_paper_arm_confirm_status()
    if status != "VALID":
        return 0.0
    exp = _parse_dt(doc.get("expires_at"))
    if not exp:
        return None
    return max(0.0, (exp - _utc_now()).total_seconds() / 60.0)


def _bounded_ttl_minutes(policy: Dict[str, Any], *, minutes_remaining: Optional[float]) -> int:
    default_ttl = int(policy.get("default_ttl_minutes", 600) or 600)
    max_ttl = int(policy.get("max_ttl_minutes", 600) or 600)
    ttl = min(default_ttl, max_ttl)
    ttl = max(30, min(ttl, 600))
    if minutes_remaining is not None and minutes_remaining > 0:
        renew_below = int(policy.get("renew_if_minutes_remaining_below", 90) or 90)
        if minutes_remaining < renew_below:
            return ttl
    return ttl


def _atomic_write_audit(payload: Dict[str, Any]) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = AUDIT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, AUDIT_PATH)


def run_paper_session_arm_policy(
    *,
    dry_run: bool = False,
    json_mode: bool = False,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate session window and renew paper_arm_confirm when appropriate."""
    policy = policy or load_policy()
    now = _utc_now()
    result: Dict[str, Any] = {
        "ok": True,
        "timestamp": _utc_iso(),
        "dry_run": dry_run,
        "policy_enabled": bool(policy.get("enabled")),
        "mode": str(policy.get("mode") or "paper_only"),
        "action": "none",
        "paper_arm_confirm_written": False,
        "live_confirm_written": False,
        "live_execution": False,
        "ttl_minutes": None,
        "notes": [],
    }

    if not policy.get("enabled"):
        result["action"] = "skipped_policy_disabled"
        result["notes"].append("paper_arm_session_policy disabled")
        return result

    if str(policy.get("mode") or "").lower() != "paper_only":
        result["ok"] = False
        result["action"] = "blocked_non_paper_mode"
        result["notes"].append("mode must be paper_only")
        return result

    live_ok, live_reason = _live_execution_blocked(policy)
    try:
        from services.portfolio_history_sync import load_live_execution_flag

        result["live_execution"] = bool(load_live_execution_flag())
    except Exception:
        result["live_execution"] = False
    if not live_ok:
        result["ok"] = False
        result["action"] = "blocked_live_execution"
        result["notes"].append(live_reason)
        return result

    env_ok, env_reason = _env_ok(policy)
    if not env_ok:
        result["action"] = "skipped_env"
        result["notes"].append(env_reason)
        return result

    if policy.get("require_broker_mode_paper"):
        broker_ok, broker_reason = _broker_mode_paper()
        if not broker_ok:
            result["ok"] = False
            result["action"] = "blocked_broker_not_paper"
            result["notes"].append(broker_reason)
            return result
        result["broker_mode"] = broker_reason

    from services.paper_arm_status import classify_paper_arm_confirm_status

    arm_status = classify_paper_arm_confirm_status()
    minutes_left = _minutes_remaining()
    result["paper_arm_confirm_status_before"] = arm_status
    result["minutes_remaining_before"] = (
        round(minutes_left, 1) if minutes_left is not None else None
    )
    result["market_open"] = is_market_open_et(now)
    result["near_market_open"] = is_near_market_open_et(
        now, pre_market_minutes=int(policy.get("pre_market_arm_minutes", 45) or 45)
    )

    in_window, window_reason = is_within_session_arm_window(policy, now)
    result["session_arm_window"] = in_window
    result["session_window_reason"] = window_reason

    if not in_window:
        result["action"] = "skipped_outside_session_window"
        result["notes"].append(f"window={window_reason}")
        return result

    renew_below = int(policy.get("renew_if_minutes_remaining_below", 90) or 90)
    auto_pre = bool(policy.get("auto_arm_before_market_open"))
    auto_renew = bool(policy.get("auto_renew_during_market_open"))

    needs_fresh_arm = arm_status in ("MISSING", "EXPIRED", "INVALID")
    needs_renew = arm_status == "VALID" and minutes_left is not None and minutes_left < renew_below

    should_write = False
    if needs_fresh_arm:
        if result["near_market_open"] and auto_pre:
            should_write = True
        elif result["market_open"] and auto_renew:
            should_write = True
        elif window_reason == "post_market_grace" and auto_renew:
            should_write = True
    elif needs_renew and auto_renew:
        should_write = True

    if not should_write:
        if arm_status == "VALID":
            result["action"] = "skipped_arm_fresh"
            result["notes"].append(
                f"arm VALID with {minutes_left} min remaining (renew below {renew_below})"
            )
        else:
            result["action"] = "skipped_no_renew_trigger"
        return result

    ttl = _bounded_ttl_minutes(policy, minutes_remaining=minutes_left)
    result["ttl_minutes"] = ttl
    result["never_expiring_arm"] = False

    if dry_run:
        result["action"] = "dry_run_would_renew"
        result["notes"].append(f"would write paper_arm_confirm ttl={ttl}m")
        return result

    from services.set_paper_arm_confirm import write_confirm

    path = write_confirm(ttl, allow_execute=True)
    result["paper_arm_confirm_written"] = True
    result["paper_arm_confirm_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
    result["action"] = "renewed" if arm_status == "VALID" else "armed"
    result["paper_arm_confirm_status_after"] = classify_paper_arm_confirm_status()
    result["minutes_remaining_after"] = _minutes_remaining()

    if LIVE_CONFIRM_PATH.is_file():
        result["notes"].append("live_confirm.json exists unchanged (not written by policy)")

    return result


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Paper session ARM auto-renewal (paper only)")
    ap.add_argument("--json", action="store_true", help="Emit JSON result on stdout only")
    ap.add_argument("--dry-run", action="store_true", help="Evaluate without writing confirm")
    args = ap.parse_args(argv)

    policy = load_policy()
    result = run_paper_session_arm_policy(dry_run=args.dry_run, json_mode=args.json, policy=policy)

    if policy.get("write_audit", True):
        audit = dict(result)
        prior: Dict[str, Any] = {}
        if AUDIT_PATH.is_file():
            try:
                prior = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
            except Exception:
                prior = {}
        history = list(prior.get("history") or [])
        history.append(
            {
                "timestamp": result.get("timestamp"),
                "action": result.get("action"),
                "ttl_minutes": result.get("ttl_minutes"),
                "written": result.get("paper_arm_confirm_written"),
            }
        )
        audit["history"] = history[-50:]
        _atomic_write_audit(audit)

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    else:
        _log(f"action={result.get('action')} written={result.get('paper_arm_confirm_written')}")
        for n in result.get("notes") or []:
            _log(f"note: {n}")

    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
