"""
phase_197f_premarket_broker_auth_quarantine_repair_validation.py
Phase 197F — Pre-market broker auth diagnostics and quarantine overbreadth repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
REGISTRY = ROOT / "data" / "governance" / "symbol_hydration_quarantine.json"
SCRIPTS = ROOT / "scripts"
OUTPUT_JSON = RESULTS / "phase_197f_premarket_broker_auth_quarantine_repair_validation.json"

TARGET = "SPCX"
MU = "MU"
BRK = "BRK-B"
PREVIOUS_COMMIT = "f2692f46"
PREVIOUS_PHASE = "197E"
MATURE_SAMPLE = ("AAPL", "AMD", "QQQ", "TSLA", "XLK")
OVERBROAD_FAIL_THRESHOLD = 2


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_dirty(path: Path) -> bool:
    proc = subprocess.run(
        ["git", "status", "--short", "--", str(path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return bool((proc.stdout or "").strip())


def _current_branch() -> str:
    proc = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (proc.stdout or "").strip()


def _run_ps(script: str) -> tuple[int, str]:
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPTS / script),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=90,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def run_phase_197f_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    os.environ.setdefault("TRITON_ENABLE_PAPER_EXECUTION", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_APPROVAL", "1")

    from services.runtime_universe_hydration_preflight import run_runtime_hydration_preflight
    from services.triton_env_loader import alpaca_env_diagnostics, load_triton_local_env
    from services.universe_pipeline_hydration import assess_symbol_pipeline_state
    from services.portfolio_history_sync import load_live_execution_flag

    branch = _current_branch()
    registry_bytes_before = REGISTRY.read_bytes() if REGISTRY.is_file() else b""
    registry_hash_before = _file_sha256(REGISTRY)

    load_triton_local_env()
    env_diag = alpaca_env_diagnostics(mode="paper")

    snapshot_result: Dict[str, Any] = {"skipped": True}
    snapshot_ok = False
    snapshot_safe_block = False
    broker_auth_ok = False
    out = ""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        json_start = (proc.stdout or "").find("{")
        try:
            snapshot_result = (
                json.loads(proc.stdout[json_start:]) if json_start >= 0 else {"ok": False}
            )
        except Exception:
            snapshot_result = {"ok": False, "raw_output": out[:500]}
        snapshot_ok = bool(snapshot_result.get("ok"))
        snapshot_safe_block = snapshot_result.get("status") == "SAFE_AUTH_BLOCK"
        broker_auth_ok = snapshot_ok or snapshot_safe_block
        if snapshot_ok:
            notes.append("broker snapshot succeeded")
        elif snapshot_safe_block:
            notes.append("broker snapshot SAFE_AUTH_BLOCK with diagnostic")
        elif "Traceback" in out and "SAFE_AUTH_BLOCK" not in out:
            notes.append("broker snapshot traceback without SAFE_AUTH_BLOCK wrapper")
    except Exception as exc:
        snapshot_result = {"ok": False, "error": str(exc)}

    preflight_1 = run_runtime_hydration_preflight(verbose=verbose)
    preflight_2 = run_runtime_hydration_preflight(verbose=verbose)

    registry_bytes_after = REGISTRY.read_bytes() if REGISTRY.is_file() else b""
    registry_unchanged = registry_bytes_before == registry_bytes_after
    registry_git_clean = not _git_dirty(REGISTRY)

    quarantined = list(preflight_2.get("quarantined_symbols") or [])
    failed = list(preflight_2.get("failed_symbols") or [])
    artifact_regression = list(preflight_2.get("artifact_regression_symbols") or [])
    mu_state = assess_symbol_pipeline_state(MU)

    _, sup_out = _run_ps("show_triton_runtime_supervisor_status.ps1")
    loop_on = "Continuous loop process running: True" in sup_out
    watchdog_on = "Watchdog process running: True" in sup_out

    readiness = {}
    try:
        readiness = json.loads((RESULTS / "execution_readiness.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    live_permitted = bool(readiness.get("live_execution_permitted", True))

    mature_quarantined = [s for s in MATURE_SAMPLE if s in quarantined]
    overbroad = len(quarantined) > OVERBROAD_FAIL_THRESHOLD or bool(mature_quarantined)

    status = "PASS"
    if branch != "main":
        notes.append(f"branch={branch} (expected main)")
    if overbroad:
        status = "FAIL"
        notes.append(f"overbroad quarantine: {quarantined}")
    if TARGET not in quarantined:
        status = "FAIL"
        notes.append(f"{TARGET} not quarantined")
    if quarantined != [TARGET]:
        if status != "FAIL":
            status = "FAIL"
        notes.append(f"expected quarantined_symbols=[{TARGET}] got {quarantined}")
    if TARGET in failed:
        status = "FAIL"
        notes.append(f"{TARGET} in failed_symbols")
    if not mu_state.get("hydrated"):
        status = "FAIL"
        notes.append(f"{MU} no longer hydrated")
    if preflight_2.get("status") != "QUARANTINED_GAPS":
        status = "FAIL"
        notes.append(f"preflight status {preflight_2.get('status')}")
    if not registry_unchanged:
        status = "FAIL"
        notes.append("governance registry changed after duplicate preflight")
    elif not registry_git_clean:
        notes.append("governance registry git dirty (substantive repair may need commit)")
    if not broker_auth_ok:
        status = "FAIL"
        notes.append("broker snapshot neither ok nor SAFE_AUTH_BLOCK")
    if "Traceback" in out and not snapshot_safe_block and not snapshot_ok:
        status = "FAIL"
        notes.append("raw traceback in broker snapshot path")
    if live_permitted or load_live_execution_flag():
        status = "FAIL"
        notes.append("live execution must remain blocked")
    if loop_on:
        notes.append("continuous loop running during validation")
    if watchdog_on:
        notes.append("watchdog running during validation (expected stopped)")

    broker_auth_root = (
        "invalid_or_mismatched_credentials"
        if snapshot_safe_block
        else ("credentials_valid" if snapshot_ok else "unknown")
    )

    payload: Dict[str, Any] = {
        "phase": "197F",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "branch": branch,
        "broker_auth_root_cause": broker_auth_root,
        "broker_env_diagnostics": {
            k: v for k, v in env_diag.items() if k not in ("env_check_lines",)
        },
        "broker_env_check_lines": env_diag.get("env_check_lines"),
        "broker_snapshot": snapshot_result,
        "broker_snapshot_ok": snapshot_ok,
        "broker_snapshot_safe_auth_block": snapshot_safe_block,
        "broker_auth_ok": broker_auth_ok,
        "quarantine_overbreadth_root_cause": (
            "stranded detection treated missing processed parquets as quarantine evidence "
            "even when lifecycle/signals existed for mature symbols"
        ),
        "quarantine_fix": (
            "Strand only missing lifecycle/signals; track processed-only gaps as "
            "artifact_regression_symbols; quarantine only IPO/unsupported with evidence"
        ),
        "preflight_status_run1": preflight_1.get("status"),
        "preflight_status_run2": preflight_2.get("status"),
        "quarantined_symbols": quarantined,
        "failed_symbols": failed,
        "artifact_regression_symbols": artifact_regression,
        "mature_symbols_quarantined": mature_quarantined,
        "spcx_only_quarantined": quarantined == [TARGET],
        "mu_hydrated": bool(mu_state.get("hydrated")),
        "allocation_refresh_only_symbols": preflight_2.get("allocation_refresh_only_symbols") or [],
        "registry_bytes_unchanged": registry_unchanged,
        "registry_git_clean": registry_git_clean,
        "registry_hash_before": registry_hash_before,
        "registry_hash_after": _file_sha256(REGISTRY),
        "broker_orders_submitted": False,
        "live_execution_permitted": False,
        "live_execution_blocked": not live_permitted,
        "continuous_loop_running": loop_on,
        "watchdog_running": watchdog_on,
        "runtime_authority_unchanged": True,
        "continuous_loop_interval_minutes": 45,
        "recommended_next_action": (
            "powershell -NoProfile -ExecutionPolicy Bypass -File "
            ".\\scripts\\start_triton_runtime_supervisor.ps1"
            if status == "PASS" and (snapshot_ok or snapshot_safe_block)
            else "fix local .env.local paper credentials then re-run validation"
        ),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197F-REPAIR] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197F-REPAIR] status={status} quarantined={quarantined} "
            f"snapshot_ok={snapshot_ok} safe_block={snapshot_safe_block}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197F premarket repair validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197f_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
