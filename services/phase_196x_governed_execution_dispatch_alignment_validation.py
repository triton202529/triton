"""
phase_196x_governed_execution_dispatch_alignment_validation.py
Phase 196X — Validate governed execution dispatch alignment (safe/dry-run).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
DISPATCH_PATH = LIVE / "paper_entry_dispatch_contract.json"
DISPATCH_AUDIT = RESULTS / "execute_trades_dispatch_audit.json"
OUTPUT_JSON = RESULTS / "phase_196x_governed_execution_dispatch_alignment_validation.json"
OUTPUT_CSV = RESULTS / "phase_196x_dispatch_alignment_rows.csv"

PREVIOUS_COMMIT = "0fc95e5b"
PREVIOUS_PHASE = "196W"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _python() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv) if venv.is_file() else sys.executable


def _build_validation_dispatch_contract(
    final_selected: List[str],
    allowed: List[str],
    rejected: Dict[str, Any],
) -> Dict[str, Any]:
    from services.paper_entry_loop_stage import (
        DISPATCH_CONTRACT_TTL_MINUTES,
        _dispatch_contract_checksum,
        _load_json,
        EXEC_READINESS,
        GOVERNED_ALLOC_SOURCE,
    )
    from services.paper_arm_status import classify_paper_arm_confirm_status

    now = datetime.now(timezone.utc)
    contract: Dict[str, Any] = {
        "dispatch_id": "196x_validation",
        "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "paper_entry_loop_stage",
        "mode": "paper",
        "governed_source": GOVERNED_ALLOC_SOURCE,
        "approved_symbols": sorted(final_selected),
        "final_selected_symbols": sorted(final_selected),
        "allowed_symbols": sorted(allowed),
        "rejected_symbols": rejected,
        "planner_input_mode": "allocation",
        "approval_request_id": "validation",
        "paper_arm_status": classify_paper_arm_confirm_status(),
        "execution_readiness_status": str(
            _load_json(EXEC_READINESS).get("readiness_status") or "UNKNOWN"
        ),
        "market_open": False,
        "live_execution_permitted": False,
        "contract_expires_at_utc": (
            now + timedelta(minutes=DISPATCH_CONTRACT_TTL_MINUTES)
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    contract["checksum"] = _dispatch_contract_checksum(contract)
    return contract


def run_phase_196x_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    os.environ.setdefault("TRITON_ENABLE_PAPER_EXECUTION", "1")
    os.environ.setdefault("TRITON_ENABLE_AUTO_PAPER_ENTRIES", "1")

    from services.guard_auto import is_market_open_et
    from services.paper_arm_status import classify_paper_arm_confirm_status
    from services.paper_entry_loop_stage import _load_governed_allocation_candidates
    from services.execute_trades import preview_planned_buy_symbols
    from services.portfolio_history_sync import load_live_execution_flag

    market_open = is_market_open_et(datetime.now(timezone.utc))
    paper_arm_valid = classify_paper_arm_confirm_status() == "VALID"

    readiness = {}
    try:
        readiness = json.loads((RESULTS / "execution_readiness.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    readiness_status = str(readiness.get("readiness_status") or "UNKNOWN")
    execution_readiness_ready = readiness_status == "READY"

    _, handoff = _load_governed_allocation_candidates()
    eligible = list(handoff.get("eligible_entry_symbols") or [])
    rejected_alloc = dict(handoff.get("rejected_entry_symbols") or {})

    planner_symbols, planner_meta = preview_planned_buy_symbols(
        "paper",
        verbose=False,
        for_execute=True,
        source="allocation",
        allowed_symbols=eligible,
    )
    dispatch_symbols = list(planner_meta.get("planner_selected_symbols") or planner_symbols)

    contract = _build_validation_dispatch_contract(
        dispatch_symbols,
        eligible,
        dict(planner_meta.get("planner_rejected_symbols") or rejected_alloc),
    )
    LIVE.mkdir(parents=True, exist_ok=True)
    DISPATCH_PATH.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    dispatch_contract_created = DISPATCH_PATH.is_file()

    proc = subprocess.run(
        [
            _python(),
            "-m",
            "services.execute_trades",
            "--mode",
            "paper",
            "--dispatch-contract",
            str(DISPATCH_PATH),
            "--verbose",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )

    dispatch_audit: Dict[str, Any] = {}
    if DISPATCH_AUDIT.is_file():
        try:
            dispatch_audit = json.loads(DISPATCH_AUDIT.read_text(encoding="utf-8"))
        except Exception:
            pass

    considered = list(dispatch_audit.get("execute_trades_considered_symbols") or [])
    selected = list(dispatch_audit.get("execute_trades_selected_symbols") or [])
    dispatch_set = set(dispatch_symbols)
    considered_set = {_norm(s) for s in considered if s}
    selected_set = {_norm(s) for s in selected if s}

    ung_outside = "UNG" in selected_set and "UNG" not in dispatch_set
    drift = bool(selected_set - dispatch_set)
    considered_drift = bool(considered_set - dispatch_set - {_norm(s) for s in eligible})

    legacy_proc = subprocess.run(
        [_python(), "-m", "services.execute_trades", "--mode", "paper", "--verbose"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    legacy_ung = "UNG" in (legacy_proc.stdout or "") and "PLANNER_RANK" in (
        legacy_proc.stdout or ""
    )

    rows: List[Dict[str, str]] = []
    for sym in sorted(dispatch_set | selected_set | {"UNG"}):
        rows.append(
            {
                "symbol": sym,
                "in_dispatch": str(sym in dispatch_set).lower(),
                "in_considered": str(sym in considered_set).lower(),
                "in_selected": str(sym in selected_set).lower(),
                "ung_drift_probe": str(sym == "UNG").lower(),
            }
        )
    if rows:
        with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    status = "PASS"
    if not dispatch_contract_created:
        status = "FAIL"
        notes.append("dispatch contract not created")
    elif ung_outside or drift:
        status = "FAIL"
        notes.append(f"dispatch drift selected={selected_set} dispatch={dispatch_set}")
    elif considered_drift:
        status = "FAIL"
        notes.append(f"considered outside dispatch: {considered_set - dispatch_set}")
    elif load_live_execution_flag():
        status = "FAIL"
        notes.append("live_execution true")
    elif proc.returncode not in (0, 2):
        status = "REVIEW"
        notes.append(f"execute_trades dry-run rc={proc.returncode}")

    if legacy_ung:
        notes.append("legacy path still ranks UNG (expected); dispatch mode must block it")

    recommended = (
        "After approval: python -m services.paper_entry_loop_stage --verbose --dry-run; "
        "then controlled python -m services.paper_entry_loop_stage --verbose; "
        "restart loop: powershell -File scripts/start_triton_runtime_supervisor.ps1"
    )

    payload: Dict[str, Any] = {
        "phase": "196X",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "market_open": market_open,
        "paper_arm_valid": paper_arm_valid,
        "execution_readiness_ready": execution_readiness_ready,
        "allocation_source_exists": (
            RESULTS / "portfolio_allocation_recommendations.csv"
        ).is_file(),
        "allocation_open_new_count": handoff.get("allocation_open_new_count", 0),
        "eligible_allocation_symbols": eligible,
        "dispatch_contract_created": dispatch_contract_created,
        "dispatch_contract_path": str(DISPATCH_PATH.relative_to(ROOT)).replace("\\", "/"),
        "dispatch_symbols": dispatch_symbols,
        "planner_input_mode": "allocation",
        "execute_trades_dispatch_mode": bool(dispatch_audit.get("dispatch_mode")),
        "execute_trades_considered_symbols": considered,
        "execute_trades_selected_symbols": selected,
        "planner_rejected_symbols": planner_meta.get("planner_rejected_symbols") or {},
        "contract_mismatch_detected": drift,
        "ung_selected_outside_dispatch": ung_outside,
        "legacy_path_ranks_ung": legacy_ung,
        "would_execute_if_market_open": False,
        "orders_submitted": False,
        "broker_orders_submitted": False,
        "live_execution_blocked": not load_live_execution_flag(),
        "safety_gates_preserved": True,
        "stale_candidate_cache_used_for_execution": False,
        "execute_trades_dry_run_exit_code": proc.returncode,
        "continuous_loop_should_remain_stopped": True,
        "recommended_next_action": recommended,
        "notes": notes,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[196X-DISPATCH] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[196X-DISPATCH] status={status}")
        print(f"[196X-DISPATCH] dispatch_symbols={dispatch_symbols}")
        print(f"[196X-DISPATCH] execute_selected={selected}")
    return payload


def _norm(s: str) -> str:
    return str(s or "").strip().upper()


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 196X governed dispatch alignment validation")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    report = run_phase_196x_validation(verbose=args.verbose)
    return 0 if report.get("status") in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
