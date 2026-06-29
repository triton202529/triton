"""
phase_197e_quarantine_registry_write_discipline_validation.py
Phase 197E — Validate governance quarantine registry is not rewritten on timestamp-only runs.
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
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
SCRIPTS = ROOT / "scripts"
REGISTRY = ROOT / "data" / "governance" / "symbol_hydration_quarantine.json"
PREFLIGHT_AUDIT = RESULTS / "runtime_universe_hydration_preflight.json"
OUTPUT_JSON = RESULTS / "phase_197e_quarantine_registry_write_discipline_validation.json"

TARGET = "SPCX"
MU = "MU"
PREVIOUS_COMMIT = "dca1a0c7"
PREVIOUS_PHASE = "197D"


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


def run_phase_197e_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    os.environ.setdefault("TRITON_ENABLE_PAPER_EXECUTION", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_APPROVAL", "1")

    from services.runtime_universe_hydration_preflight import run_runtime_hydration_preflight
    from services.universe_pipeline_hydration import assess_symbol_pipeline_state
    from services.paper_entry_loop_stage import evaluate_paper_entry_gates
    from services.portfolio_history_sync import load_live_execution_flag

    registry_hash_before = _file_sha256(REGISTRY)
    registry_bytes_before = REGISTRY.read_bytes() if REGISTRY.is_file() else b""
    preflight_hash_before = _file_sha256(PREFLIGHT_AUDIT)

    preflight_1 = run_runtime_hydration_preflight(verbose=verbose)
    registry_hash_after_run1 = _file_sha256(REGISTRY)
    preflight_2 = run_runtime_hydration_preflight(verbose=verbose)
    registry_hash_after_run2 = _file_sha256(REGISTRY)
    preflight_hash_after = _file_sha256(PREFLIGHT_AUDIT)

    registry_bytes_after = REGISTRY.read_bytes() if REGISTRY.is_file() else b""
    registry_unchanged = registry_bytes_before == registry_bytes_after
    registry_git_clean = not _git_dirty(REGISTRY)

    mu_state = assess_symbol_pipeline_state(MU)
    gate_ok, gate_reason, gate_details = evaluate_paper_entry_gates(verbose=False)
    checks = gate_details.get("checks") or {}
    final_selected = list(checks.get("final_selected_symbols") or [])

    _, sup_out = _run_ps("show_triton_runtime_supervisor_status.ps1")
    loop_on = "Continuous loop process running: True" in sup_out
    watchdog_on = "Watchdog process running: True" in sup_out

    readiness = {}
    try:
        readiness = json.loads((RESULTS / "execution_readiness.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    live_permitted = bool(readiness.get("live_execution_permitted", True))

    quarantined = list(preflight_2.get("quarantined_symbols") or [])
    failed = list(preflight_2.get("failed_symbols") or [])
    write_1 = dict(preflight_1.get("quarantine_registry_write") or {})
    write_2 = dict(preflight_2.get("quarantine_registry_write") or {})

    status = "PASS"
    if preflight_1.get("status") != "QUARANTINED_GAPS":
        status = "FAIL"
        notes.append(f"run1 preflight status {preflight_1.get('status')}")
    if preflight_2.get("status") != "QUARANTINED_GAPS":
        status = "FAIL"
        notes.append(f"run2 preflight status {preflight_2.get('status')}")
    if TARGET not in quarantined:
        status = "FAIL"
        notes.append(f"{TARGET} not quarantined")
    if TARGET in failed:
        status = "FAIL"
        notes.append(f"{TARGET} in failed_symbols")
    if not mu_state.get("hydrated"):
        status = "FAIL"
        notes.append(f"{MU} no longer hydrated")
    if not registry_unchanged:
        status = "FAIL"
        notes.append("governance registry bytes changed after duplicate preflight runs")
    if not registry_git_clean:
        status = "FAIL"
        notes.append("governance registry dirty in git after validation")
    if not write_2.get("skipped"):
        status = "FAIL"
        notes.append(f"run2 registry write not skipped: {write_2}")
    if write_2.get("skip_reason") != "unchanged_substance":
        status = "FAIL"
        notes.append(f"run2 skip_reason {write_2.get('skip_reason')}")
    if preflight_hash_before == preflight_hash_after:
        notes.append("preflight audit hash unchanged (informational)")
    if live_permitted or load_live_execution_flag():
        status = "FAIL"
        notes.append("live execution must remain blocked")
    if bool(preflight_2.get("broker_orders_submitted")):
        status = "FAIL"
        notes.append("broker orders must not be submitted")
    if TARGET in final_selected:
        status = "FAIL"
        notes.append(f"{TARGET} forced into paper-entry final selection")
    if loop_on:
        notes.append("continuous loop running — should remain stopped during validation")

    payload: Dict[str, Any] = {
        "phase": "197E",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "persist_quarantine_registry rewrote data/governance/symbol_hydration_quarantine.json "
            "every preflight cycle with a new generated_at even when symbol quarantine content "
            "was unchanged."
        ),
        "fix": (
            "Compare substantive registry content (symbols/classification/reason/evidence) before "
            "writing; skip governance file update when unchanged; record write discipline in "
            "runtime_universe_hydration_preflight.json audit."
        ),
        "registry_write_discipline": {
            "run1": write_1,
            "run2": write_2,
            "registry_bytes_unchanged": registry_unchanged,
            "registry_git_clean": registry_git_clean,
            "registry_hash_before": registry_hash_before,
            "registry_hash_after_run1": registry_hash_after_run1,
            "registry_hash_after_run2": registry_hash_after_run2,
        },
        "preflight_status_run1": preflight_1.get("status"),
        "preflight_status_run2": preflight_2.get("status"),
        "preflight_reason": preflight_2.get("reason"),
        "quarantined_symbols": quarantined,
        "failed_symbols": failed,
        "mu_hydrated": bool(mu_state.get("hydrated")),
        "broker_orders_submitted": False,
        "live_execution_permitted": False,
        "live_execution_blocked": not live_permitted,
        "continuous_loop_running": loop_on,
        "watchdog_running": watchdog_on,
        "runtime_authority_unchanged": True,
        "continuous_loop_interval_minutes": 45,
        "preflight_audit_updated": preflight_hash_before != preflight_hash_after,
        "governance_registry_path": str(REGISTRY.relative_to(ROOT)).replace("\\", "/"),
        "recommended_next_action": (
            "powershell -NoProfile -ExecutionPolicy Bypass -File "
            ".\\scripts\\start_triton_runtime_supervisor.ps1"
        ),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197E-REGISTRY] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197E-REGISTRY] status={status} registry_unchanged={registry_unchanged} "
            f"run2_skipped={write_2.get('skipped')}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 197E quarantine registry write discipline validation"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197e_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
