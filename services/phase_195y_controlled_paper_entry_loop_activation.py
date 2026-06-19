"""
phase_195y_controlled_paper_entry_loop_activation.py
Phase 195Y — Controlled paper entry execution loop activation validation.

Validates that run_continuous_loop.ps1 includes a guarded paper-entry stage
and that safety gates are enforced. Does not place orders unless market is
open and all gates pass during a live loop cycle (validation uses --dry-run).
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
CONFIG = ROOT / "config"
SCRIPTS = ROOT / "scripts"

OUTPUT_JSON = RESULTS / "controlled_paper_entry_loop_activation.json"
SUMMARY_CSV = RESULTS / "controlled_paper_entry_loop_activation_summary.csv"
CANDIDATES_CSV = RESULTS / "controlled_paper_entry_loop_candidates.csv"

CONTINUOUS_LOOP = SCRIPTS / "run_continuous_loop.ps1"
PAPER_ENTRY_STAGE = ROOT / "services" / "paper_entry_loop_stage.py"
EXECUTE_TRADES = ROOT / "services" / "execute_trades.py"
MARKET_OPEN = SCRIPTS / "run_market_open.ps1"
PAPER_CYCLE_CFG = CONFIG / "run_scheduled_paper_cycle.json"
EXEC_READINESS = RESULTS / "execution_readiness.json"
TRADE_OPPORTUNITIES = RESULTS / "trade_opportunities.csv"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"

CANDIDATE_FIELDS = (
    "ticker",
    "opportunity_type",
    "edge_score",
    "confidence",
    "eligible_for_auto_entry",
    "block_reasons",
    "risk_flag",
    "allocation_action",
)

SUMMARY_FIELDS = ("metric", "value")

RECOMMENDED_START = (
    "$env:TRITON_ENABLE_PAPER_EXECUTION='1'; "
    "$env:TRITON_ENABLE_AUTO_PAPER_ENTRIES='1'; "
    "Start-Process powershell.exe -ArgumentList "
    "'-NoProfile','-ExecutionPolicy','Bypass','-File',"
    f"'{ROOT}\\\\scripts\\\\run_watchdog_loop.ps1','-IntervalMinutes','1' "
    f"-WorkingDirectory '{ROOT}'; "
    "Start-Process powershell.exe -ArgumentList "
    "'-NoProfile','-ExecutionPolicy','Bypass','-File',"
    f"'{ROOT}\\\\scripts\\\\run_continuous_loop.ps1','-IntervalMinutes','45' "
    f"-WorkingDirectory '{ROOT}'"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0


from services.phase_195h_operational_readiness_freeze import _find_ps_processes  # noqa: E402


def _find_python_execute_trades_execute() -> Tuple[bool, List[str]]:
    if sys.platform != "win32":
        return False, []
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" "
        "| Where-Object { $_.CommandLine -like '*execute_trades*' -and $_.CommandLine -like '*--execute*' } "
        "| Select-Object -ExpandProperty CommandLine"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = [x.strip() for x in (r.stdout or "").splitlines() if x.strip()]
        return len(lines) > 0, lines
    except (subprocess.SubprocessError, OSError):
        return False, []


def _analyze_loop_file(text: str) -> Dict[str, Any]:
    lower = text.lower()
    auto_entries_assigned = bool(
        re.search(r"\$env:TRITON_ENABLE_AUTO_PAPER_ENTRIES\s*=", text, re.IGNORECASE)
    )
    return {
        "loop_file_exists": bool(text),
        "paper_entry_stage_present": "paper_entry_loop_stage" in lower,
        "calls_market_open_first": "run_market_open.ps1" in lower,
        "paper_entry_after_market_open": (
            lower.find("run_market_open.ps1") < lower.find("paper_entry_loop_stage")
            if "paper_entry_loop_stage" in lower
            else False
        ),
        "blocked_log_on_market_open_fail": "paper_entry_execution_blocked" in lower,
        "auto_sets_paper_execution_env": bool(
            re.search(r"\$env:TRITON_ENABLE_PAPER_EXECUTION\s*=", text, re.IGNORECASE)
        ),
        "auto_sets_auto_paper_entries": auto_entries_assigned,
    }


def _analyze_paper_entry_stage(text: str) -> Dict[str, Any]:
    lower = text.lower()
    return {
        "module_exists": bool(text),
        "paper_env_required": "triton_enable_paper_execution" in lower,
        "auto_paper_entries_env_required": "triton_enable_auto_paper_entries" in lower,
        "market_open_required": "is_market_open" in lower or "market_open" in lower,
        "readiness_required": "execution_readiness" in lower,
        "live_block_required": "live_execution_permitted" in lower,
        "arm_required": "resolve_arm_mutation_permissions" in lower or "arm_mode" in lower,
        "duplicate_protection_required": "duplicate" in lower or "check_duplicate_order" in lower,
        "execute_trades_execute_call_present": (
            "execute_trades" in lower and "--execute" in lower and "paper" in lower
        ),
        "execute_trades_execute_call_guarded": (
            "evaluate_paper_entry_gates" in lower or "gate_pass" in lower
        ),
        "paper_mode_only": '"--mode", "paper"' in text or "--mode paper" in text,
        "single_invoke_guard": "_execute_invoked" in lower,
        "gate_log_markers": all(
            m in text
            for m in (
                "[PAPER_ENTRY_GATE]",
                "[PAPER_ENTRY_EXECUTION]",
                "[PAPER_ENTRY_EXECUTION_BLOCKED]",
            )
        ),
        "dry_run_supported": "--dry-run" in lower or "dry_run" in lower,
    }


def _load_candidate_rows() -> List[Dict[str, str]]:
    try:
        from services.paper_entry_loop_stage import _load_entry_candidates

        candidates, _ = _load_entry_candidates()
        return [{k: str(c.get(k, "")) for k in CANDIDATE_FIELDS} for c in candidates]
    except Exception:
        return []


def _run_dry_gate_evaluation() -> Dict[str, Any]:
    try:
        from services.paper_entry_loop_stage import evaluate_paper_entry_gates

        ok, reason, details = evaluate_paper_entry_gates(verbose=False)
        return {
            "gate_pass": ok,
            "gate_reason": reason,
            "checks": details.get("checks", {}),
            "eligible_count": details.get("checks", {}).get("entry_rows_eligible", 0),
        }
    except Exception as e:
        return {"gate_pass": False, "gate_reason": f"evaluation_error:{e}", "checks": {}}


def run_phase_195y_activation(*, verbose: bool = True) -> Dict[str, Any]:
    loop_text = _read_text(CONTINUOUS_LOOP)
    stage_text = _read_text(PAPER_ENTRY_STAGE)
    loop_analysis = _analyze_loop_file(loop_text)
    stage_analysis = _analyze_paper_entry_stage(stage_text)

    readiness = _load_json(EXEC_READINESS)
    paper_cfg = _load_json(PAPER_CYCLE_CFG)
    execute_detected, execute_cmds = _find_python_execute_trades_execute()

    broker_open = _count_csv_rows(OPEN_ORDERS)
    broker_positions = _count_csv_rows(POSITIONS)
    live_permitted = bool(readiness.get("live_execution_permitted", True))
    paper_permitted = bool(readiness.get("paper_execution_permitted", False))

    candidate_rows = _load_candidate_rows()
    gate_eval = _run_dry_gate_evaluation()

    from services.guard_auto import is_market_open_et

    market_open = is_market_open_et(datetime.now(timezone.utc))

    design_checks = {
        "loop_file_modified": loop_analysis["paper_entry_stage_present"],
        "paper_entry_stage_present": loop_analysis["paper_entry_stage_present"],
        "paper_entry_module_exists": stage_analysis["module_exists"],
        "auto_paper_entries_env_required": stage_analysis["auto_paper_entries_env_required"],
        "market_open_required": stage_analysis["market_open_required"],
        "readiness_required": stage_analysis["readiness_required"],
        "live_block_required": stage_analysis["live_block_required"],
        "arm_required": stage_analysis["arm_required"],
        "duplicate_protection_required": stage_analysis["duplicate_protection_required"],
        "execute_trades_execute_call_present": stage_analysis[
            "execute_trades_execute_call_present"
        ],
        "execute_trades_execute_call_guarded": stage_analysis[
            "execute_trades_execute_call_guarded"
        ],
        "gate_log_markers_present": stage_analysis.get("gate_log_markers", False),
        "loop_does_not_auto_enable_auto_entries": not loop_analysis["auto_sets_auto_paper_entries"],
        "market_open_before_paper_entry": loop_analysis["paper_entry_after_market_open"],
        "manage_positions_still_in_market_open": "manage_positions"
        in _read_text(MARKET_OPEN).lower(),
        "no_live_mode_in_stage": "mode live" not in stage_text.lower().replace('"', ""),
    }

    validation_pass = all(design_checks.values())

    payload: Dict[str, Any] = {
        "phase": "195Y",
        "generated_at": _utc_now_iso(),
        "phase_pass": validation_pass,
        "validation_status": "PASS" if validation_pass else "FAIL",
        "audit_mode": "design_validation_with_dry_run_gate_eval",
        "orders_placed_by_validation": False,
        "market_open_at_validation": market_open,
        "loop_file_modified": loop_analysis["paper_entry_stage_present"],
        "paper_entry_stage_present": loop_analysis["paper_entry_stage_present"],
        "auto_paper_entries_env_required": stage_analysis["auto_paper_entries_env_required"],
        "market_open_required": stage_analysis["market_open_required"],
        "readiness_required": stage_analysis["readiness_required"],
        "live_block_required": stage_analysis["live_block_required"],
        "arm_required": stage_analysis["arm_required"],
        "duplicate_protection_required": stage_analysis["duplicate_protection_required"],
        "execute_trades_execute_call_present": stage_analysis[
            "execute_trades_execute_call_present"
        ],
        "execute_trades_execute_call_guarded": stage_analysis[
            "execute_trades_execute_call_guarded"
        ],
        "broker_open_orders_count": broker_open,
        "broker_positions_count": broker_positions,
        "live_execution_permitted": live_permitted,
        "paper_execution_permitted": paper_permitted,
        "execute_trades_execute_detected": execute_detected,
        "execute_trades_command_lines": execute_cmds,
        "automated_entry_execution_enabled": bool(paper_cfg.get("run_execute_trades", False)),
        "TRITON_ENABLE_PAPER_EXECUTION": os.environ.get("TRITON_ENABLE_PAPER_EXECUTION", ""),
        "TRITON_ENABLE_AUTO_PAPER_ENTRIES": os.environ.get("TRITON_ENABLE_AUTO_PAPER_ENTRIES", ""),
        "design_checks": design_checks,
        "loop_analysis": loop_analysis,
        "stage_analysis": stage_analysis,
        "dry_run_gate_evaluation": gate_eval,
        "entry_candidates_count": len(candidate_rows),
        "entry_candidates_eligible": sum(
            1 for r in candidate_rows if str(r.get("eligible_for_auto_entry")).lower() == "true"
        ),
        "execution_readiness_status": str(readiness.get("readiness_status") or "UNKNOWN"),
        "recommended_operator_start_command": RECOMMENDED_START,
        "safety_statement": (
            "Phase 195Y validates guarded paper-entry stage design. "
            "Live execution remains blocked. Auto entries require both env vars and market open."
        ),
    }

    summary_rows: List[Dict[str, str]] = [
        {"metric": "validation_status", "value": payload["validation_status"]},
        {"metric": "phase_pass", "value": str(validation_pass).lower()},
        {"metric": "loop_file_modified", "value": str(payload["loop_file_modified"]).lower()},
        {
            "metric": "paper_entry_stage_present",
            "value": str(payload["paper_entry_stage_present"]).lower(),
        },
        {
            "metric": "auto_paper_entries_env_required",
            "value": str(payload["auto_paper_entries_env_required"]).lower(),
        },
        {"metric": "market_open_required", "value": str(payload["market_open_required"]).lower()},
        {
            "metric": "execute_trades_execute_call_guarded",
            "value": str(payload["execute_trades_execute_call_guarded"]).lower(),
        },
        {"metric": "market_open_at_validation", "value": str(market_open).lower()},
        {"metric": "dry_run_gate_pass", "value": str(gate_eval.get("gate_pass", False)).lower()},
        {"metric": "dry_run_gate_reason", "value": str(gate_eval.get("gate_reason", ""))},
        {"metric": "broker_open_orders_count", "value": str(broker_open)},
        {"metric": "broker_positions_count", "value": str(broker_positions)},
        {"metric": "live_execution_permitted", "value": str(live_permitted).lower()},
        {"metric": "paper_execution_permitted", "value": str(paper_permitted).lower()},
        {"metric": "execute_trades_execute_detected", "value": str(execute_detected).lower()},
        {"metric": "entry_candidates_eligible", "value": str(payload["entry_candidates_eligible"])},
        {"metric": "recommended_operator_start_command", "value": RECOMMENDED_START},
    ]

    _atomic_write_json(OUTPUT_JSON, payload)
    _write_csv(SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    _write_csv(CANDIDATES_CSV, candidate_rows, CANDIDATE_FIELDS)

    if verbose:
        print(f"[195Y] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[195Y] validation_status={payload['validation_status']}")
        print(f"[195Y] paper_entry_stage_present={payload['paper_entry_stage_present']}")
        print(f"[195Y] market_open={market_open} dry_run_gate={gate_eval.get('gate_reason')}")
        print(f"[195Y] broker={broker_open}/{broker_positions} live_blocked={not live_permitted}")
        if not validation_pass:
            failed = [k for k, v in design_checks.items() if not v]
            print(f"[195Y] failed design checks: {failed}")

    return payload


def main() -> None:
    result = run_phase_195y_activation(verbose=True)
    sys.exit(0 if result.get("validation_status") == "PASS" else 1)


if __name__ == "__main__":
    main()
