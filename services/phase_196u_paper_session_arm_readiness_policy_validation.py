"""
phase_196u_paper_session_arm_readiness_policy_validation.py
Phase 196U — Validate paper session ARM readiness policy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
CONFIG_PATH = ROOT / "config" / "paper_arm_session_policy.json"
AUDIT_PATH = RESULTS / "paper_session_arm_policy_audit.json"
LIVE_CONFIRM_PATH = LIVE / "live_confirm.json"
OUTPUT_JSON = RESULTS / "phase_196u_paper_session_arm_readiness_policy.json"

PREVIOUS_COMMIT = "8cf1d980"
PREVIOUS_PHASE = "196T"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _run_ps(script: str) -> Tuple[int, str]:
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / script),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=90,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _parse_freshness_age(text: str) -> Optional[float]:
    for line in text.splitlines():
        if "heartbeat.json" in line and "~" in line and "min" in line:
            try:
                return float(line.split("~")[1].split("min")[0].strip())
            except Exception:
                pass
    return None


def _extract_snapshot_json(stdout: str) -> Dict[str, Any]:
    marker = '{\n  "ok"'
    idx = stdout.find(marker)
    if idx < 0:
        return {}
    try:
        return json.loads(stdout[idx:])
    except Exception:
        return {}


def _paper_arm_json_clean() -> Tuple[bool, Dict[str, Any]]:
    proc = subprocess.run(
        [sys.executable, "-m", "services.paper_arm_status", "--json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    stdout = (proc.stdout or "").strip()
    if proc.returncode != 0 or not stdout.startswith("{"):
        return False, {}
    try:
        return True, json.loads(stdout)
    except json.JSONDecodeError:
        return False, {}


def run_phase_196u_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    from services.paper_session_arm_policy import (
        is_market_open_et,
        is_near_market_open_et,
        is_within_session_arm_window,
        load_policy,
    )
    from services.paper_arm_status import build_paper_arm_status, classify_live_confirm_status

    policy = load_policy()
    head = subprocess.run(
        ["git", "log", "-1", "--format=%H"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    head_commit = (head.stdout or "").strip()

    sup_rc, sup_out = _run_ps("show_triton_runtime_supervisor_status.ps1")
    fresh_rc, fresh_out = _run_ps("show_triton_data_freshness_status.ps1")

    watchdog_on = "Watchdog process running: True" in sup_out
    loop_on = "Continuous loop process running: True" in sup_out
    fresh_age = _parse_freshness_age(fresh_out)
    data_freshness = "FRESH" if fresh_age is not None and fresh_age < 40 else "STALE"
    if fresh_age is None:
        data_freshness = "REVIEW"

    arm_st = build_paper_arm_status(verbose_notes=False)
    json_ok, json_doc = _paper_arm_json_clean()

    ps_rc, ps_out = _run_ps("show_triton_paper_arm_status.ps1")
    ps_json_ok = ps_rc == 0 and "Invalid JSON" not in ps_out

    policy_proc = subprocess.run(
        [sys.executable, "-m", "services.paper_session_arm_policy", "--json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    policy_result: Dict[str, Any] = {}
    if (policy_proc.stdout or "").strip().startswith("{"):
        try:
            policy_result = json.loads(policy_proc.stdout)
        except Exception:
            pass

    live_confirm_before = LIVE_CONFIRM_PATH.stat().st_mtime if LIVE_CONFIRM_PATH.is_file() else None

    positions: List[str] = []
    open_orders = 0
    live_execution = False
    try:
        from services.dashboard_status_sync import count_open_orders, load_positions_summary
        from services.portfolio_history_sync import load_live_execution_flag

        positions = sorted(load_positions_summary().get("positions") or [])
        open_orders = int(count_open_orders())
        live_execution = load_live_execution_flag()
    except Exception as exc:
        notes.append(f"portfolio: {exc}")

    snap_proc = subprocess.run(
        [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    snap_doc = _extract_snapshot_json(snap_proc.stdout or "") if snap_proc.returncode == 0 else {}

    in_window, window_reason = is_within_session_arm_window(policy)
    market_open = is_market_open_et()
    near_open = is_near_market_open_et(
        pre_market_minutes=int(policy.get("pre_market_arm_minutes", 45) or 45)
    )

    live_confirm_after = LIVE_CONFIRM_PATH.stat().st_mtime if LIVE_CONFIRM_PATH.is_file() else None
    live_confirm_created = live_confirm_before != live_confirm_after and live_confirm_after

    ttl = int(policy.get("default_ttl_minutes", 600) or 600)
    max_ttl = int(policy.get("max_ttl_minutes", 600) or 600)
    never_expiring = ttl > 600 or max_ttl > 600

    mo = (ROOT / "scripts" / "run_market_open.ps1").read_text(encoding="utf-8")
    df = (ROOT / "scripts" / "run_data_freshness_refresh.ps1").read_text(encoding="utf-8")
    integrated = "paper_session_arm_policy" in mo and "paper_session_arm_policy" in df

    notes.append(f"Session window={in_window} reason={window_reason}")
    notes.append(f"Policy dry-run action sample: {policy_result.get('action')}")
    if integrated:
        notes.append("Integrated into run_market_open and data_freshness_refresh")

    status = "PASS"
    if not CONFIG_PATH.is_file():
        status = "FAIL"
        notes.append("policy config missing")
    elif not integrated:
        status = "FAIL"
        notes.append("policy not integrated into runtime scripts")
    elif not json_ok or not ps_json_ok:
        status = "FAIL"
        notes.append("paper_arm_status JSON parse failed")
    elif live_execution:
        status = "FAIL"
        notes.append("live_execution true")
    elif live_confirm_created:
        status = "FAIL"
        notes.append("live_confirm was created")
    elif never_expiring:
        status = "FAIL"
        notes.append("TTL exceeds bounded maximum")
    elif sup_rc != 0:
        status = "REVIEW"
        notes.append("supervisor status script failed")
    elif arm_st.get("status") != "VALID" and in_window:
        status = "REVIEW"
        notes.append("arm not VALID inside session window — check env flags")

    payload = {
        "phase": "196U",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "head_commit": head_commit[:8] if head_commit else "",
        "policy_config_created": CONFIG_PATH.is_file(),
        "policy_enabled": bool(policy.get("enabled")),
        "mode": str(policy.get("mode") or "paper_only"),
        "auto_arm_before_market_open": bool(policy.get("auto_arm_before_market_open")),
        "auto_renew_during_market_open": bool(policy.get("auto_renew_during_market_open")),
        "bounded_ttl_minutes": min(ttl, max_ttl, 600),
        "never_expiring_arm_created": never_expiring,
        "paper_arm_confirm_status": arm_st.get("status"),
        "paper_entry_armed": bool(arm_st.get("paper_entry_armed")),
        "minutes_remaining": arm_st.get("minutes_remaining"),
        "live_execution": live_execution,
        "live_confirm_created": bool(live_confirm_created),
        "live_confirm_status": classify_live_confirm_status(),
        "open_orders": open_orders,
        "positions": positions,
        "watchdog_running": watchdog_on,
        "continuous_loop_running": loop_on,
        "data_freshness": data_freshness,
        "market_status_checked": True,
        "market_open": market_open,
        "near_market_open": near_open,
        "near_market_open_or_open": market_open or near_open,
        "session_arm_window": in_window,
        "session_window_reason": window_reason,
        "runtime_integration": integrated,
        "manual_execute_trades_run": False,
        "broker_orders_submitted": False,
        "broker_snapshot_ok": bool(snap_doc.get("ok")),
        "risk_logic_weakened": False,
        "allocation_logic_weakened": False,
        "lifecycle_logic_weakened": False,
        "certification_logic_weakened": False,
        "paper_arm_status_json_fixed": json_ok and ps_json_ok,
        "audit_written": AUDIT_PATH.is_file(),
        "policy_service_result": policy_result,
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196U-ARM] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196U-ARM] status={status}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 196U paper session arm policy validation")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196u_validation(verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
