"""
phase_195i_lifecycle_artifact_integrity_validation.py
Phase 195I — Validate lifecycle authority artifacts are not partially overwritten.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

OUTPUT_JSON = RESULTS / "phase_195i_lifecycle_artifact_integrity_validation.json"

RAW_PATH = RESULTS / "signal_lifecycle.csv"
EFFECTIVE_PATH = RESULTS / "signal_lifecycle_effective.csv"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run_phase_195i_validation(*, verbose: bool = True) -> Dict[str, Any]:
    from services.lifecycle_artifact_guard import (
        BACKUPS_DIR,
        audit_lifecycle_authority_preservation,
        count_lifecycle_rows,
        is_lifecycle_authoritative,
        list_backups,
        minimum_required_rows,
        preferred_expected_rows,
        validate_lifecycle_columns,
    )
    from services.lifecycle_truth import evaluate_lifecycle_gate
    import pandas as pd

    min_rows = minimum_required_rows()
    raw_rows = count_lifecycle_rows(RAW_PATH)
    eff_rows = count_lifecycle_rows(EFFECTIVE_PATH)
    blocking: List[str] = []

    raw_exists = RAW_PATH.is_file() and RAW_PATH.stat().st_size > 0
    eff_exists = EFFECTIVE_PATH.is_file() and EFFECTIVE_PATH.stat().st_size > 0

    if not raw_exists:
        blocking.append("signal_lifecycle.csv missing")
    if not eff_exists:
        blocking.append("signal_lifecycle_effective.csv missing")

    raw_cols_ok = eff_cols_ok = False
    if raw_exists:
        try:
            raw_df = pd.read_csv(RAW_PATH)
            raw_cols_ok, raw_missing = validate_lifecycle_columns(raw_df, artifact="raw")
            if not raw_cols_ok:
                blocking.append(f"raw missing columns: {raw_missing}")
        except Exception as e:
            blocking.append(f"raw read error: {e}")
            raw_df = None
    else:
        raw_df = None

    if eff_exists:
        try:
            eff_df = pd.read_csv(EFFECTIVE_PATH)
            eff_cols_ok, eff_missing = validate_lifecycle_columns(eff_df, artifact="effective")
            if not eff_cols_ok:
                blocking.append(f"effective missing columns: {eff_missing}")
        except Exception as e:
            blocking.append(f"effective read error: {e}")
            eff_df = None
    else:
        eff_df = None

    if raw_rows < min_rows:
        blocking.append(f"raw rows {raw_rows} < min {min_rows}")
    if eff_rows < min_rows:
        blocking.append(f"effective rows {eff_rows} < min {min_rows}")

    if eff_rows > 0 and raw_rows > 0 and eff_rows < raw_rows:
        blocking.append(f"effective rows ({eff_rows}) < raw rows ({raw_rows}) without explanation")

    gate = evaluate_lifecycle_gate(path=EFFECTIVE_PATH)
    placement_gate_status = gate.status
    if gate.status == "BLOCKED" and gate.reason in (
        "INSUFFICIENT_EFFECTIVE_ROWS",
        "INSUFFICIENT_RAW_ROWS",
    ):
        placement_gate_status = "BLOCKED_AS_EXPECTED_FOR_PARTIAL"
    elif gate.status == "BLOCKED":
        blocking.append(f"placement gate blocked: {gate.reason} {gate.details}")
    elif raw_rows >= min_rows and eff_rows >= min_rows and gate.status != "OK":
        blocking.append(f"unexpected gate status: {gate.status} {gate.reason}")

    raw_backups = list_backups(RAW_PATH)
    eff_backups = list_backups(EFFECTIVE_PATH)
    backup_status = "no_backups"
    if raw_backups or eff_backups:
        backup_status = "backups_present"
    if not BACKUPS_DIR.is_dir():
        backup_status = "backup_dir_missing"

    audit = audit_lifecycle_authority_preservation()
    if audit.get("partial_detected"):
        blocking.append("partial lifecycle authority detected")

    artifact_status = "AUTHORITATIVE"
    if raw_rows < min_rows or eff_rows < min_rows:
        artifact_status = "PARTIAL_OR_MISSING"
    elif not raw_cols_ok or not eff_cols_ok:
        artifact_status = "INVALID_COLUMNS"

    phase_pass = len(blocking) == 0 and artifact_status == "AUTHORITATIVE"

    payload: Dict[str, Any] = {
        "phase": "195I",
        "timestamp": _utc_now_iso(),
        "phase_pass": phase_pass,
        "signal_lifecycle_rows": raw_rows,
        "effective_lifecycle_rows": eff_rows,
        "minimum_required_rows": min_rows,
        "preferred_expected_rows": preferred_expected_rows(),
        "artifact_status": artifact_status,
        "backup_status": backup_status,
        "raw_backup_count": len(raw_backups),
        "effective_backup_count": len(eff_backups),
        "latest_raw_backup": str(raw_backups[0]) if raw_backups else None,
        "latest_effective_backup": str(eff_backups[0]) if eff_backups else None,
        "placement_gate_status": placement_gate_status,
        "placement_gate_reason": gate.reason,
        "placement_gate_details": gate.details,
        "raw_authoritative": is_lifecycle_authoritative(RAW_PATH),
        "effective_authoritative": is_lifecycle_authoritative(EFFECTIVE_PATH),
        "blocking_issues": blocking,
        "audit": audit,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    if verbose:
        print(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    result = run_phase_195i_validation(verbose=True)
    return 0 if result.get("phase_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
