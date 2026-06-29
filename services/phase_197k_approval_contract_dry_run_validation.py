"""
phase_197k_approval_contract_dry_run_validation.py
Phase 197K — Dry-run approval contract simulation without execution or runtime authority change.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"
FIXTURE_DIR = RESULTS / "phase_197k_approval_contract_fixtures"
OUTPUT_JSON = RESULTS / "phase_197k_approval_contract_dry_run_validation.json"
OPERATOR_PACKET_JSON = RESULTS / "defensive_exit_operator_approval_packet.json"
REAL_APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
PREVIOUS_COMMIT = "8f8eea7d"
PREVIOUS_PHASE = "197J"

CASE_MISSING = "MISSING_APPROVAL"
CASE_EXPIRED = "EXPIRED_APPROVAL"
CASE_WRONG_DECISION = "WRONG_DECISION"
CASE_NON_PAPER = "NON_PAPER_MODE_APPROVAL"
CASE_MISSING_LIVE_ACK = "MISSING_LIVE_BLOCK_ACKNOWLEDGEMENT"
CASE_MISSING_PAPER_ACK = "MISSING_PAPER_ONLY_ACKNOWLEDGEMENT"
CASE_PACKET_MISMATCH = "PACKET_ID_MISMATCH"
CASE_HASH_MISMATCH = "PREVIEW_HASH_MISMATCH"
CASE_SYMBOL_NOT_IN_PREVIEW = "SYMBOL_NOT_IN_PREVIEW"
CASE_MAX_NOTIONAL = "MAX_NOTIONAL_EXCEEDS_CAP"
CASE_OPEN_ORDERS = "OPEN_ORDERS_CONFLICT"
CASE_VALID = "VALID_PAPER_ONLY_APPROVAL"

INVALID_CASES = [
    CASE_MISSING,
    CASE_EXPIRED,
    CASE_WRONG_DECISION,
    CASE_NON_PAPER,
    CASE_MISSING_LIVE_ACK,
    CASE_MISSING_PAPER_ACK,
    CASE_PACKET_MISMATCH,
    CASE_HASH_MISMATCH,
    CASE_SYMBOL_NOT_IN_PREVIEW,
    CASE_MAX_NOTIONAL,
    CASE_OPEN_ORDERS,
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_fixture(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _git_branch() -> str:
    proc = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (proc.stdout or "").strip()


def _git_clean() -> bool:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return not (proc.stdout or "").strip()


def _count_positions() -> int:
    path = RESULTS / "positions_snapshot.csv"
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def _count_open_orders() -> int:
    path = RESULTS / "open_orders_snapshot.csv"
    if not path.is_file():
        return 0
    try:
        import pandas as pd

        return len(pd.read_csv(path))
    except Exception:
        return 0


def _load_packet() -> Dict[str, Any]:
    if not OPERATOR_PACKET_JSON.is_file():
        return {}
    return json.loads(OPERATOR_PACKET_JSON.read_text(encoding="utf-8"))


def _base_valid_contract(packet: Dict[str, Any]) -> Dict[str, Any]:
    from services.defensive_exit_operator_paper_approval import OPERATOR_DECISION_APPROVE

    now = _utc_now()
    capped = list(packet.get("capped_symbols") or [])
    return {
        "approval_id": "fixture-197k-valid",
        "approved_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=24)).isoformat(),
        "approved_by": "197k_dry_run_fixture",
        "approval_mode": "PAPER_ONLY",
        "approved_symbols": capped[:3] if len(capped) >= 3 else capped,
        "max_notional": float(packet.get("capped_notional") or 0.0),
        "source_packet_id": str(packet.get("packet_id") or ""),
        "source_preview_hash": str(packet.get("source_preview_hash") or ""),
        "acknowledgement_live_blocked": True,
        "acknowledgement_no_live_execution": True,
        "acknowledgement_paper_only": True,
        "decision": OPERATOR_DECISION_APPROVE,
        "_fixture_case": CASE_VALID,
        "_fixture_only": True,
        "_not_runtime_authority": True,
    }


def _build_fixtures(packet: Dict[str, Any]) -> Dict[str, Path]:
    base = _base_valid_contract(packet)
    now = _utc_now()
    configured_cap = float(packet.get("configured_cap") or 0.0)
    capped = list(packet.get("capped_symbols") or [])

    fixtures: Dict[str, Dict[str, Any]] = {
        CASE_EXPIRED: {
            **base,
            "approval_id": "fixture-197k-expired",
            "expires_at": (now - timedelta(hours=1)).isoformat(),
            "_fixture_case": CASE_EXPIRED,
        },
        CASE_WRONG_DECISION: {
            **base,
            "approval_id": "fixture-197k-wrong-decision",
            "decision": "REJECT_CAPPED_PAPER_DEFENSIVE_BATCH",
            "_fixture_case": CASE_WRONG_DECISION,
        },
        CASE_NON_PAPER: {
            **base,
            "approval_id": "fixture-197k-non-paper",
            "approval_mode": "LIVE",
            "_fixture_case": CASE_NON_PAPER,
        },
        CASE_MISSING_LIVE_ACK: {
            **base,
            "approval_id": "fixture-197k-missing-live-ack",
            "acknowledgement_live_blocked": False,
            "_fixture_case": CASE_MISSING_LIVE_ACK,
        },
        CASE_MISSING_PAPER_ACK: {
            **base,
            "approval_id": "fixture-197k-missing-paper-ack",
            "acknowledgement_paper_only": False,
            "_fixture_case": CASE_MISSING_PAPER_ACK,
        },
        CASE_PACKET_MISMATCH: {
            **base,
            "approval_id": "fixture-197k-packet-mismatch",
            "source_packet_id": "def-exit-paper-WRONG-PACKET-ID",
            "_fixture_case": CASE_PACKET_MISMATCH,
        },
        CASE_HASH_MISMATCH: {
            **base,
            "approval_id": "fixture-197k-hash-mismatch",
            "source_preview_hash": "0" * 64,
            "_fixture_case": CASE_HASH_MISMATCH,
        },
        CASE_SYMBOL_NOT_IN_PREVIEW: {
            **base,
            "approval_id": "fixture-197k-symbol-not-in-preview",
            "approved_symbols": ["ZZZZ_NOT_IN_PREVIEW"],
            "_fixture_case": CASE_SYMBOL_NOT_IN_PREVIEW,
        },
        CASE_MAX_NOTIONAL: {
            **base,
            "approval_id": "fixture-197k-max-notional",
            "max_notional": configured_cap + 1000.0,
            "_fixture_case": CASE_MAX_NOTIONAL,
        },
        CASE_OPEN_ORDERS: {
            **base,
            "approval_id": "fixture-197k-open-orders",
            "approved_symbols": [capped[0]] if capped else ["SLV"],
            "_fixture_case": CASE_OPEN_ORDERS,
        },
        CASE_VALID: {
            **base,
            "_fixture_case": CASE_VALID,
        },
    }

    paths: Dict[str, Path] = {}
    for case_name, doc in fixtures.items():
        fname = case_name.lower() + ".json"
        path = FIXTURE_DIR / fname
        _write_fixture(path, doc)
        paths[case_name] = path
    return paths


def _run_case(
    case_name: str,
    *,
    packet: Dict[str, Any],
    fixture_path: Optional[Path] = None,
    open_sell_symbols_override: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_STATUS_INVALID,
        APPROVAL_STATUS_REQUIRED,
        APPROVAL_STATUS_VALID,
        validate_operator_approval_contract,
    )

    status, reason, details = validate_operator_approval_contract(
        packet_id=str(packet.get("packet_id") or ""),
        preview_hash=str(packet.get("source_preview_hash") or ""),
        capped_symbols=list(packet.get("capped_symbols") or []),
        configured_cap=float(packet.get("configured_cap") or 0.0),
        broker_snapshot_ok=True,
        live_execution_blocked=True,
        contract_path=fixture_path,
        open_sell_symbols_override=open_sell_symbols_override,
    )

    if case_name == CASE_MISSING:
        expected_status = APPROVAL_STATUS_REQUIRED
    elif case_name == CASE_VALID:
        expected_status = APPROVAL_STATUS_VALID
    else:
        expected_status = APPROVAL_STATUS_INVALID

    passed = status == expected_status
    return {
        "case": case_name,
        "fixture_path": (
            str(fixture_path.relative_to(ROOT)).replace("\\", "/") if fixture_path else None
        ),
        "expected_status": expected_status,
        "actual_status": status,
        "reason": reason,
        "passed": passed,
        "details": details,
    }


def run_phase_197k_validation(*, verbose: bool = True) -> Dict[str, Any]:
    notes: List[str] = []
    status = "PASS"
    git_clean_before = _git_clean()
    real_exists_before = REAL_APPROVAL_PATH.is_file()
    positions_before = _count_positions()
    open_orders_before = _count_open_orders()

    from services.defensive_exit_governance_sync import build_defensive_exit_approval_sync
    from services.defensive_exit_operator_paper_approval import (
        APPROVAL_STATUS_REQUIRED,
        validate_operator_approval_contract,
    )
    from services.live_block_classifier import (
        apply_defensive_exit_validator_gate_checks,
        defensive_exit_phase_validator_gates,
    )

    packet = _load_packet()
    if not packet:
        payload = {
            "phase": "197K",
            "generated_at": _utc_now_iso(),
            "status": "FAIL",
            "notes": ["defensive_exit_operator_approval_packet.json missing"],
        }
        _atomic_write_json(OUTPUT_JSON, payload)
        return payload

    fixture_paths = _build_fixtures(packet)
    case_results: List[Dict[str, Any]] = []

    case_results.append(
        _run_case(CASE_MISSING, packet=packet, fixture_path=FIXTURE_DIR / "nonexistent.json")
    )

    capped = list(packet.get("capped_symbols") or [])
    for case_name in INVALID_CASES:
        if case_name == CASE_MISSING:
            continue
        override: Optional[Set[str]] = None
        if case_name == CASE_OPEN_ORDERS and capped:
            override = {capped[0].upper()}
        case_results.append(
            _run_case(
                case_name,
                packet=packet,
                fixture_path=fixture_paths.get(case_name),
                open_sell_symbols_override=override,
            )
        )

    valid_result = _run_case(
        CASE_VALID,
        packet=packet,
        fixture_path=fixture_paths.get(CASE_VALID),
    )
    case_results.append(valid_result)

    real_status, real_reason, _ = validate_operator_approval_contract(
        packet_id=str(packet.get("packet_id") or ""),
        preview_hash=str(packet.get("source_preview_hash") or ""),
        capped_symbols=capped,
        configured_cap=float(packet.get("configured_cap") or 0.0),
    )
    real_exists_after = REAL_APPROVAL_PATH.is_file()
    sync = build_defensive_exit_approval_sync(cap_context="runtime")
    gates = defensive_exit_phase_validator_gates(verbose=False)
    gate_ok = gates["gate_ok"]
    gate_reason = gates["gate_reason"]

    snapshot_ok = False
    try:
        snap = subprocess.run(
            [sys.executable, "-m", "services.snapshot_live_orders", "--mode", "paper"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        jstart = (snap.stdout or "").find("{")
        if jstart >= 0:
            snapshot_ok = bool(json.loads(snap.stdout[jstart:]).get("ok"))
    except Exception:
        pass

    cases_passed = sum(1 for c in case_results if c.get("passed"))
    cases_failed = sum(1 for c in case_results if not c.get("passed"))
    invalid_rejections = [
        {"case": c["case"], "reason": c["reason"]}
        for c in case_results
        if c["case"] != CASE_VALID and c.get("passed")
    ]

    status = "PASS"
    if cases_failed:
        status = "FAIL"
        notes.append(f"{cases_failed} fixture case(s) failed")
    if valid_result.get("actual_status") != "OPERATOR_APPROVAL_VALID_PAPER_ONLY":
        status = "FAIL"
        notes.append("valid fixture not accepted")
    if real_status != APPROVAL_STATUS_REQUIRED:
        status = "FAIL"
        notes.append(f"real runtime approval_status={real_status} expected REQUIRED")
    if real_exists_before:
        status = "FAIL"
        notes.append(
            "real approval artifact exists before validation; must not exist for 197K dry-run"
        )
    if real_exists_after and not real_exists_before:
        status = "FAIL"
        notes.append("197K created real approval artifact")
    status = apply_defensive_exit_validator_gate_checks(
        status=status, notes=notes, gates=gates, fail_on_gate_open=True
    )
    if sync.get("eligible_for_paper_execution"):
        status = "FAIL"
        notes.append("eligible_for_paper_execution must remain false")
    if _count_open_orders() != open_orders_before:
        notes.append(f"open_orders changed {open_orders_before} -> {_count_open_orders()}")

    payload: Dict[str, Any] = {
        "phase": "197K",
        "generated_at": _utc_now_iso(),
        "status": status,
        "previous_phase": PREVIOUS_PHASE,
        "previous_commit": PREVIOUS_COMMIT,
        "root_cause": (
            "197J defined the operator approval contract but had no safe dry-run proving "
            "the validator rejects unsafe fixtures without creating runtime authority."
        ),
        "fix": (
            "phase_197k runs fixture-only contract validation under "
            "data/results/phase_197k_approval_contract_fixtures/ with contract_path override; "
            "real governance approval path is never written."
        ),
        "real_approval_contract_path": str(REAL_APPROVAL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "real_approval_exists_before": real_exists_before,
        "real_approval_exists_after": real_exists_after,
        "real_runtime_approval_status": real_status,
        "real_runtime_approval_reason": real_reason,
        "fixture_dir": str(FIXTURE_DIR.relative_to(ROOT)).replace("\\", "/"),
        "cases_tested": [c["case"] for c in case_results],
        "cases_passed": cases_passed,
        "cases_failed": cases_failed,
        "case_results": case_results,
        "valid_fixture_status": valid_result.get("actual_status"),
        "invalid_fixture_rejections": invalid_rejections,
        "broker_snapshot_ok": snapshot_ok,
        "open_orders_count": _count_open_orders(),
        "positions_count": _count_positions(),
        "positions_count_before": positions_before,
        "live_blocked": gates["live_blocked"],
        "live_block_reason": gates["live_block_reason"],
        "paper_entry_gate_ok": gate_ok,
        "paper_entry_gate_reason": gate_reason,
        "paper_entry_gate_classification": gates["gate_classification"],
        "broker_orders_submitted": False,
        "governance_authorized": sync.get("governance_authorized"),
        "readiness_status": sync.get("readiness_status"),
        "governance_sync_status": sync.get("approval_sync_status"),
        "eligible_for_paper_execution_real_runtime": bool(sync.get("eligible_for_paper_execution")),
        "operator_packet_id": packet.get("packet_id"),
        "source_preview_hash": packet.get("source_preview_hash"),
        "git_branch": _git_branch(),
        "git_clean_before_validation": git_clean_before,
        "git_clean_after_validation": _git_clean(),
        "notes": notes,
    }
    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(f"[197K-DRY-RUN] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(
            f"[197K-DRY-RUN] status={status} cases_passed={cases_passed}/{len(case_results)} "
            f"real_runtime={real_status} valid_fixture={valid_result.get('actual_status')}"
        )
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 197K approval contract dry-run validation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_phase_197k_validation(verbose=not args.quiet)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
