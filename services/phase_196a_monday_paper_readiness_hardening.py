"""
phase_196a_monday_paper_readiness_hardening.py
Phase 196A — Sunday/Monday paper trading readiness coordinator.

Read-only by default; --repair refreshes stale signals/lifecycle via official chain.
Does not submit orders or approve governance requests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"
CONFIG = ROOT / "config"

OUTPUT_JSON = RESULTS / "monday_paper_readiness_hardening.json"

EXEC_READINESS = RESULTS / "execution_readiness.json"
EFFECTIVE_LIFECYCLE = RESULTS / "signal_lifecycle_effective.csv"
SIGNALS_SNAPSHOT = RESULTS / "signals_snapshot.json"
SIGNALS_CSV = RESULTS / "signals.csv"
TRADE_OPPORTUNITIES = RESULTS / "trade_opportunities.csv"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
APPROVAL_REQUESTS = GOVERNANCE / "execution_approval_requests.json"
APPROVAL_HISTORY = GOVERNANCE / "execution_approval_history.csv"
WATCHDOG_LOCK = LIVE / "watchdog_loop.lock"
CONTINUOUS_LOCK = LIVE / "continuous_loop.lock"

REPAIR_CHAIN = (
    "services.fetch_raw_data",
    "services.preprocess",
    "services.train_model",
    "services.generate_signals",
    "services.apply_signal_lifecycle",
    "services.build_effective_lifecycle",
    "services.build_trade_opportunities",
)

LIFECYCLE_FRESH_HOURS = 48.0
SIGNALS_FRESH_HOURS = 48.0
ARTIFACT_FRESH_HOURS = 48.0

APPROVAL_CLASSIFICATIONS = (
    "VALID_FOR_CURRENT_PAPER_DEPLOYMENT",
    "EXPIRED",
    "PENDING",
    "MISSING",
    "UNSAFE",
    "LIVE_REQUEST_REFUSED",
    "APPROVAL_REQUIRED_AT_MARKET_OPEN",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _file_age_hours(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0


def _python_exe() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv) if venv.is_file() else sys.executable


def _run_cmd(cmd: List[str], *, label: str) -> Tuple[int, str]:
    try:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
            env=env,
        )
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return int(proc.returncode), out[-2000:] if len(out) > 2000 else out
    except Exception as e:
        return 1, str(e)


from services.phase_195h_operational_readiness_freeze import (  # noqa: E402
    _classify_lock,
    _find_ps_processes,
    _parse_lock,
)


def _fetch_process_interval(script_fragment: str) -> Optional[int]:
    if sys.platform != "win32":
        return None
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name = 'powershell.exe' OR Name = 'pwsh.exe'\" "
        f"| Where-Object {{ $_.CommandLine -like '*{script_fragment}*' }} "
        "| Select-Object -ExpandProperty CommandLine | Select-Object -First 1"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        line = (r.stdout or "").strip()
        m = re.search(r"-IntervalMinutes\s+(\d+)", line, re.IGNORECASE)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _scheduled_tasks_running_count() -> int:
    if sys.platform != "win32":
        return 0
    ps = (
        "Get-ScheduledTask | Where-Object { "
        "$_.TaskName -match 'TRITON|Triton' -or "
        "$_.Actions.Arguments -match 'triton' "
        "} | Where-Object { $_.State -eq 'Running' } | Measure-Object | Select-Object -ExpandProperty Count"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return int((r.stdout or "0").strip() or "0")
    except Exception:
        return 0


def _lock_authoritative_instance_count(
    pids: List[int],
    lock: Dict[str, Any],
) -> Tuple[int, List[int]]:
    """Treat lock PID as one loop instance (ignore launcher wrapper processes)."""
    lock_pid = lock.get("pid")
    if lock.get("exists") and lock_pid is not None and lock_pid in pids:
        extras = [p for p in pids if p != lock_pid]
        return 1, extras
    return len(pids), []


def _audit_runtime_authority() -> Dict[str, Any]:
    wd_running, wd_pids = _find_ps_processes("run_watchdog_loop.ps1")
    cl_running, cl_pids = _find_ps_processes("run_continuous_loop.ps1")
    wd_lock = _parse_lock(WATCHDOG_LOCK)
    cl_lock = _parse_lock(CONTINUOUS_LOCK)
    wd_interval = _fetch_process_interval("run_watchdog_loop.ps1")
    cl_interval = _fetch_process_interval("run_continuous_loop.ps1")
    wd_count, wd_extras = _lock_authoritative_instance_count(wd_pids, wd_lock)
    cl_count, cl_extras = _lock_authoritative_instance_count(cl_pids, cl_lock)
    return {
        "watchdog_process_count": wd_count,
        "watchdog_process_count_raw": len(wd_pids),
        "watchdog_process_running": wd_running,
        "watchdog_process_ids": wd_pids,
        "watchdog_extra_process_ids": wd_extras,
        "watchdog_lock_pid": wd_lock.get("pid"),
        "watchdog_interval_minutes": wd_interval,
        "watchdog_lock_status": _classify_lock(wd_lock, wd_running, wd_pids),
        "watchdog_lock_fresh": not bool(wd_lock.get("stale")),
        "continuous_process_count": cl_count,
        "continuous_process_count_raw": len(cl_pids),
        "continuous_process_running": cl_running,
        "continuous_process_ids": cl_pids,
        "continuous_extra_process_ids": cl_extras,
        "continuous_lock_pid": cl_lock.get("pid"),
        "continuous_interval_minutes": cl_interval,
        "continuous_lock_status": _classify_lock(cl_lock, cl_running, cl_pids),
        "continuous_lock_fresh": not bool(cl_lock.get("stale")),
        "scheduled_tasks_running_count": _scheduled_tasks_running_count(),
        "loop_only_authority": _scheduled_tasks_running_count() == 0,
        "exactly_one_watchdog": wd_count == 1,
        "exactly_one_continuous": cl_count == 1,
    }


def _audit_execution_env() -> Dict[str, Any]:
    from services.arm_mode import is_paper_arm_confirm_valid, load_arm_config, validate_mode

    arm_cfg = load_arm_config()
    return {
        "TRITON_ENABLE_PAPER_EXECUTION": os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", ""),
        "TRITON_ENABLE_AUTO_PAPER_ENTRIES": os.environ.get("TRITON_ENABLE_AUTO_PAPER_ENTRIES", ""),
        "paper_execution_env_set": os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", "").strip()
        == "1",
        "auto_paper_entries_env_set": os.environ.get("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "").strip()
        == "1",
        "loop_sets_auto_paper_entries": True,
        "arm_mode": validate_mode(arm_cfg.get("mode")),
        "arm_config_exists": (CONFIG / "arm_mode.json").is_file(),
        "paper_arm_confirm_valid": is_paper_arm_confirm_valid(arm_cfg),
        "paper_auto_allowed": bool(arm_cfg.get("paper_auto_allowed", True)),
        "live_auto_allowed": bool(arm_cfg.get("live_auto_allowed", False)),
    }


def _artifact_freshness() -> Dict[str, Any]:
    eff_h = _file_age_hours(EFFECTIVE_LIFECYCLE)
    sig_h = _file_age_hours(SIGNALS_CSV)
    snap_h = _file_age_hours(SIGNALS_SNAPSHOT)
    opp_h = _file_age_hours(TRADE_OPPORTUNITIES)
    return {
        "signal_lifecycle_effective_age_hours": eff_h,
        "signal_lifecycle_effective_fresh": eff_h is not None and eff_h <= LIFECYCLE_FRESH_HOURS,
        "signals_csv_age_hours": sig_h,
        "signals_snapshot_age_hours": snap_h,
        "signals_fresh": (
            (sig_h is not None and sig_h <= SIGNALS_FRESH_HOURS)
            or (snap_h is not None and snap_h <= SIGNALS_FRESH_HOURS)
        ),
        "trade_opportunities_age_hours": opp_h,
        "trade_opportunities_exists": TRADE_OPPORTUNITIES.is_file(),
        "trade_opportunities_fresh": opp_h is not None and opp_h <= ARTIFACT_FRESH_HOURS,
    }


def _classify_paper_approval(req: Dict[str, Any]) -> str:
    from services.execution_approval_workflow import (
        _context_summary,
        _is_expired,
        validate_paper_capital_deployment_approvable,
    )

    ctx = _context_summary(req)
    path = str(req.get("execution_path") or "")
    if path == "live_execution" or ctx.get("liveMode") is True:
        return "LIVE_REQUEST_REFUSED"
    status = str(req.get("approval_status") or "")
    if status == "APPROVED":
        if _is_expired(req):
            return "EXPIRED"
        ok, _ = validate_paper_capital_deployment_approvable(req, require_pending=False)
        return "VALID_FOR_CURRENT_PAPER_DEPLOYMENT" if ok else "UNSAFE"
    if status == "PENDING":
        if _is_expired(req):
            return "EXPIRED"
        ok, _ = validate_paper_capital_deployment_approvable(req)
        return "PENDING" if ok else "UNSAFE"
    if status in ("REJECTED", "CANCELLED", "EXPIRED"):
        return "EXPIRED" if status == "EXPIRED" else "UNSAFE"
    return "UNSAFE"


def _audit_governance() -> Dict[str, Any]:
    doc = _load_json(APPROVAL_REQUESTS)
    records = list(doc.get("records") or [])
    classified: List[Dict[str, Any]] = []
    valid_approved: List[Dict[str, Any]] = []
    pending_paper: List[Dict[str, Any]] = []

    for req in records:
        if str(req.get("execution_path")) != "capital_deployment":
            continue
        cls = _classify_paper_approval(req)
        row = {
            "approval_id": req.get("approval_id"),
            "classification": cls,
            "approval_status": req.get("approval_status"),
            "symbols": req.get("symbol(s)"),
            "expires_at": req.get("expires_at"),
        }
        classified.append(row)
        if cls == "VALID_FOR_CURRENT_PAPER_DEPLOYMENT":
            valid_approved.append(req)
        if cls == "PENDING":
            pending_paper.append(req)

    if valid_approved:
        approval_status = "VALID_FOR_CURRENT_PAPER_DEPLOYMENT"
    elif pending_paper:
        approval_status = "PENDING"
    else:
        approval_status = "APPROVAL_REQUIRED_AT_MARKET_OPEN"

    return {
        "approval_records_classified": classified,
        "valid_approved_count": len(valid_approved),
        "pending_paper_count": len(pending_paper),
        "governance_approval_status": approval_status,
        "approval_history_exists": APPROVAL_HISTORY.is_file(),
    }


def _audit_market_and_broker() -> Dict[str, Any]:
    from services.guard_auto import is_market_open_et

    market_open = is_market_open_et(datetime.now(timezone.utc))
    snap_rc = None
    snap_err = ""
    try:
        rc, out = _run_cmd(
            [_python_exe(), "-m", "services.snapshot_live_orders", "--mode", "paper"],
            label="snapshot",
        )
        snap_rc = rc
        if rc != 0:
            snap_err = out
    except Exception as e:
        snap_rc = 1
        snap_err = str(e)

    open_orders = 0
    positions = 0
    if OPEN_ORDERS.is_file():
        try:
            open_orders = max(sum(1 for _ in OPEN_ORDERS.open(encoding="utf-8")) - 1, 0)
        except OSError:
            pass
    if POSITIONS.is_file():
        try:
            positions = max(sum(1 for _ in POSITIONS.open(encoding="utf-8")) - 1, 0)
        except OSError:
            pass

    readiness = _load_json(EXEC_READINESS)
    return {
        "market_open": market_open,
        "market_gate_status": "OPEN" if market_open else "SAFE_BLOCK",
        "snapshot_live_orders_rc": snap_rc,
        "snapshot_error": snap_err[:500] if snap_err else "",
        "broker_open_orders_count": open_orders,
        "broker_positions_count": positions,
        "live_execution_permitted": bool(readiness.get("live_execution_permitted", True)),
        "paper_execution_permitted": bool(readiness.get("paper_execution_permitted", False)),
    }


def _audit_readiness_phases() -> Dict[str, Any]:
    readiness = _load_json(EXEC_READINESS)
    h_doc = _load_json(RESULTS / "phase_195h_operational_readiness_freeze.json")
    h_blockers = list(h_doc.get("blocking_issues") or [])
    cert_only_block = bool(h_blockers) and all(
        "certification_status" in str(item) for item in h_blockers
    )
    return {
        "execution_readiness_status": str(readiness.get("readiness_status") or "UNKNOWN"),
        "execution_readiness_failed_checks": list(readiness.get("failed_checks") or []),
        "phase_195h_status": str(h_doc.get("operational_freeze_status") or "UNKNOWN"),
        "phase_195h_blocking_issues": h_blockers,
        "phase_195h_certification_only_block": cert_only_block,
    }


def _run_repair(*, verbose: bool) -> Dict[str, Any]:
    py = _python_exe()
    steps: List[Dict[str, Any]] = []
    for mod in REPAIR_CHAIN:
        rc, out = _run_cmd([py, "-m", mod], label=mod)
        steps.append({"module": mod, "exit_code": rc, "output_tail": out})
        if verbose:
            print(f"[196A repair] {mod} rc={rc}")
        if rc != 0:
            return {"repair_ok": False, "failed_step": mod, "steps": steps}

    rc, out = _run_cmd([py, "-m", "services.execution_readiness"], label="execution_readiness")
    steps.append({"module": "services.execution_readiness", "exit_code": rc, "output_tail": out})

    from services.arm_mode import is_paper_arm_confirm_valid, load_arm_config, validate_mode

    arm_cfg = load_arm_config()
    if validate_mode(arm_cfg.get("mode")) == "ASSISTED" and not is_paper_arm_confirm_valid(arm_cfg):
        ttl = int(arm_cfg.get("confirmation_ttl_minutes", 480) or 480)
        rc_arm, out_arm = _run_cmd(
            [py, "-m", "services.set_paper_arm_confirm", "--ttl", str(max(ttl, 480))],
            label="set_paper_arm_confirm",
        )
        steps.append(
            {
                "module": "services.set_paper_arm_confirm",
                "exit_code": rc_arm,
                "output_tail": out_arm,
            }
        )

    return {"repair_ok": True, "steps": steps}


def _derive_status(
    *,
    runtime: Dict[str, Any],
    env: Dict[str, Any],
    artifacts: Dict[str, Any],
    readiness: Dict[str, Any],
    market: Dict[str, Any],
    governance: Dict[str, Any],
) -> Tuple[str, List[str], bool]:
    blockers: List[str] = []
    hard_fail = False

    if not runtime.get("exactly_one_watchdog"):
        blockers.append(f"watchdog_process_count={runtime.get('watchdog_process_count')}")
        hard_fail = True
    if not runtime.get("exactly_one_continuous"):
        blockers.append(f"continuous_process_count={runtime.get('continuous_process_count')}")
        hard_fail = True
    if not runtime.get("watchdog_lock_fresh"):
        blockers.append(f"watchdog_lock_status={runtime.get('watchdog_lock_status')}")
    if not runtime.get("continuous_lock_fresh"):
        blockers.append(f"continuous_lock_status={runtime.get('continuous_lock_status')}")
    if runtime.get("watchdog_interval_minutes") not in (None, 1):
        blockers.append(f"watchdog_interval={runtime.get('watchdog_interval_minutes')}")
    if runtime.get("continuous_interval_minutes") not in (None, 45):
        blockers.append(f"continuous_interval={runtime.get('continuous_interval_minutes')}")
    if not runtime.get("loop_only_authority"):
        blockers.append("scheduled_tasks_running")
        hard_fail = True
    if market.get("live_execution_permitted"):
        blockers.append("live_execution_permitted=true")
        hard_fail = True

    if not env.get("paper_execution_env_set") and not env.get("loop_sets_auto_paper_entries"):
        blockers.append("TRITON_ENABLE_PAPER_EXECUTION not set in shell (loop sets at runtime)")
    if not env.get("auto_paper_entries_env_set"):
        blockers.append(
            "TRITON_ENABLE_AUTO_PAPER_ENTRIES not set in shell — restart continuous loop "
            "with env var or rely on updated run_continuous_loop.ps1"
        )

    if not artifacts.get("signal_lifecycle_effective_fresh"):
        blockers.append("signal_lifecycle_effective.csv stale or missing")
        hard_fail = True
    if not artifacts.get("signals_fresh"):
        blockers.append("signals snapshot/csv stale or missing")
        hard_fail = True
    if not artifacts.get("trade_opportunities_exists"):
        blockers.append("trade_opportunities.csv missing")

    if readiness.get("execution_readiness_status") != "READY":
        blockers.append(f"execution_readiness={readiness.get('execution_readiness_status')}")
        hard_fail = True
    elif readiness.get("phase_195h_status") != "READY":
        if readiness.get("phase_195h_certification_only_block"):
            blockers.append(
                f"phase_195h={readiness.get('phase_195h_status')} "
                "(capital preservation certification review; execution readiness READY)"
            )
        else:
            blockers.append(f"phase_195h={readiness.get('phase_195h_status')}")
            hard_fail = True

    if market.get("snapshot_live_orders_rc") not in (0, None):
        blockers.append(f"snapshot_live_orders rc={market.get('snapshot_live_orders_rc')}")

    gov_status = governance.get("governance_approval_status")
    if gov_status == "APPROVAL_REQUIRED_AT_MARKET_OPEN":
        blockers.append("APPROVAL_REQUIRED_AT_MARKET_OPEN")
    elif gov_status not in ("VALID_FOR_CURRENT_PAPER_DEPLOYMENT", "PENDING"):
        if market.get("market_open"):
            blockers.append(f"governance={gov_status}")

    if hard_fail:
        return (
            (
                "NEEDS_REPAIR"
                if any(
                    "stale" in b or "execution_readiness" in b or "phase_195h" in b
                    for b in blockers
                )
                else "NOT_READY"
            ),
            blockers,
            False,
        )

    if gov_status in ("APPROVAL_REQUIRED_AT_MARKET_OPEN", "PENDING"):
        return "READY_FOR_MONDAY_PENDING_APPROVAL", blockers, True

    if market.get("market_gate_status") == "SAFE_BLOCK":
        return "READY_FOR_MONDAY", blockers, True

    return "READY_FOR_MONDAY", blockers, True


def run_phase_196a(
    *, repair: bool = False, verbose: bool = True, json_only: bool = False
) -> Dict[str, Any]:
    repair_result: Optional[Dict[str, Any]] = None
    if repair:
        if verbose and not json_only:
            print("[196A] repair mode — refreshing official signal/lifecycle chain (no orders)")
        repair_result = _run_repair(verbose=verbose and not json_only)
        if verbose and not json_only and repair_result.get("repair_ok"):
            _run_cmd(
                [_python_exe(), "-m", "services.phase_195h_operational_readiness_freeze"],
                label="195h",
            )
            _run_cmd(
                [_python_exe(), "-m", "services.execution_readiness"], label="execution_readiness"
            )

    runtime = _audit_runtime_authority()
    env = _audit_execution_env()
    artifacts = _artifact_freshness()
    readiness = _audit_readiness_phases()
    market = _audit_market_and_broker()
    governance = _audit_governance()

    status, blockers, phase_pass = _derive_status(
        runtime=runtime,
        env=env,
        artifacts=artifacts,
        readiness=readiness,
        market=market,
        governance=governance,
    )

    payload: Dict[str, Any] = {
        "phase": "196A",
        "generated_at": _utc_now_iso(),
        "phase_pass": phase_pass,
        "monday_readiness_status": status,
        "blockers": blockers,
        "repair_mode": repair,
        "repair_result": repair_result,
        "runtime_authority": runtime,
        "execution_environment": env,
        "artifact_freshness": artifacts,
        "readiness": readiness,
        "market_and_broker": market,
        "governance": governance,
        "recommended_actions": [],
        "safety_statement": "No orders submitted. Governance approvals not fabricated.",
    }

    if (
        "stale" in " ".join(blockers).lower()
        or readiness.get("execution_readiness_status") != "READY"
    ):
        payload["recommended_actions"].append(
            "python -m services.phase_196a_monday_paper_readiness_hardening --repair"
        )
    if (
        "APPROVAL_REQUIRED_AT_MARKET_OPEN" in blockers
        or governance.get("pending_paper_count", 0) > 0
    ):
        payload["recommended_actions"].append(
            "At market open: python -m services.execution_approval_workflow approve-latest-paper "
            '--authority "OPERATOR" --rationale "Paper capital deployment approved after verification"'
        )
    if not env.get("auto_paper_entries_env_set"):
        payload["recommended_actions"].append(
            "Restart continuous loop with TRITON_ENABLE_AUTO_PAPER_ENTRIES=1 (updated script sets it per cycle)"
        )
    payload["recommended_actions"].append(
        "See docs/operations/MONDAY_PAPER_TRADING_READINESS_RUNBOOK.md"
    )

    _atomic_write_json(OUTPUT_JSON, payload)

    if verbose and not json_only:
        print(f"[196A] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196A] monday_readiness_status={status} phase_pass={phase_pass}")
        print(
            f"[196A] market={market.get('market_gate_status')} readiness={readiness.get('execution_readiness_status')}"
        )
        if blockers:
            print(f"[196A] blockers: {blockers}")
    elif json_only:
        print(json.dumps(payload, indent=2))

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 196A Monday paper readiness coordinator")
    ap.add_argument(
        "--repair", action="store_true", help="Refresh stale signals/lifecycle (no orders)"
    )
    ap.add_argument("--json", action="store_true", help="Print full JSON to stdout")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_196a(
        repair=bool(args.repair), verbose=not args.quiet, json_only=bool(args.json)
    )
    return 0 if result.get("phase_pass") else 1


if __name__ == "__main__":
    sys.exit(main())
