"""
TRITON Phase 163A — Lifecycle readiness diagnostics (observability only).

Evaluates lifecycle artifact health without changing readiness pass/fail outcomes.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DIAGNOSTICS_PATH = RESULTS_DIR / "lifecycle_readiness_diagnostics.json"

DEFAULT_LIFECYCLE = RESULTS_DIR / "signal_lifecycle.csv"
DEFAULT_EFFECTIVE = RESULTS_DIR / "signal_lifecycle_effective.csv"
LIFECYCLE_CFG = PROJECT_ROOT / "config" / "lifecycle_logic.json"

# Mirrors governance_execution_readiness.LIFECYCLE_FRESHNESS_HOURS (48h).
LIFECYCLE_FRESHNESS_HOURS = 48
FRESHNESS_THRESHOLD_SECONDS = LIFECYCLE_FRESHNESS_HOURS * 3600

FAILURE_REASONS = frozenset(
    {
        "FILE_NOT_FOUND",
        "FILE_EMPTY",
        "FILE_STALE",
        "SCHEMA_FAILURE",
        "MISSING_COLUMNS",
        "INVALID_TIMESTAMP",
        "READ_ERROR",
        "RACE_CONDITION_SUSPECTED",
        "UNKNOWN",
    }
)

_RACE_RETRY_COUNT = 3
_RACE_RETRY_DELAY_SEC = 0.05


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _file_mtime_iso(path: Path) -> Optional[str]:
    try:
        if path.is_file():
            mtime = path.stat().st_mtime
            return (
                datetime.fromtimestamp(mtime, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
    except OSError:
        pass
    return None


def _file_age_seconds(path: Path) -> Optional[float]:
    try:
        if path.is_file():
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            return (datetime.now(timezone.utc) - mtime).total_seconds()
    except OSError:
        pass
    return None


def _read_csv_with_race_detection(
    path: Path,
) -> Tuple[Optional[pd.DataFrame], bool, Optional[str], List[str]]:
    """
    Attempt to read a CSV with quick retries for race-condition detection only.
    Returns (dataframe, race_suspected, read_error, columns).
    """
    if not path.is_file():
        return None, False, "file_not_found", []

    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, False, str(exc), []

    if size == 0:
        return None, False, "file_empty", []

    first_error: Optional[str] = None
    for attempt in range(_RACE_RETRY_COUNT):
        try:
            df = pd.read_csv(path)
            columns = [str(c).strip() for c in df.columns]
            race = attempt > 0 and first_error is not None
            return df, race, None, columns
        except Exception as exc:
            if first_error is None:
                first_error = str(exc)
            if attempt < _RACE_RETRY_COUNT - 1:
                time.sleep(_RACE_RETRY_DELAY_SEC)

    return None, True, first_error, []


def _required_columns_for_path(path: Path) -> List[str]:
    if path.name == DEFAULT_EFFECTIVE.name:
        return ["ticker", "effective_position_state", "effective_stance"]
    return ["ticker", "position_state"]


def _evaluate_single_file(
    path: Path,
    *,
    freshness_threshold_seconds: float,
    check_upstream_staleness: bool = False,
    signals_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    required = _required_columns_for_path(path)
    has_action_col = path.name != DEFAULT_EFFECTIVE.name

    diag: Dict[str, Any] = {
        "lifecycle_file_exists": path.is_file(),
        "lifecycle_file_path": str(path),
        "last_modified": _file_mtime_iso(path),
        "file_age_seconds": _file_age_seconds(path),
        "freshness_threshold_seconds": freshness_threshold_seconds,
        "row_count": 0,
        "column_count": 0,
        "required_columns": list(required),
        "missing_columns": [],
        "freshness_pass": True,
        "schema_pass": True,
        "data_pass": True,
        "overall_pass": True,
        "failure_reason": None,
        "race_condition_suspected": False,
    }

    if not path.is_file():
        diag["freshness_pass"] = False
        diag["schema_pass"] = False
        diag["data_pass"] = False
        diag["overall_pass"] = False
        diag["failure_reason"] = "FILE_NOT_FOUND"
        return diag

    try:
        size = path.stat().st_size
    except OSError as exc:
        diag["freshness_pass"] = False
        diag["schema_pass"] = False
        diag["data_pass"] = False
        diag["overall_pass"] = False
        diag["failure_reason"] = "READ_ERROR"
        diag["read_error"] = str(exc)
        return diag

    if size == 0:
        diag["freshness_pass"] = False
        diag["schema_pass"] = False
        diag["data_pass"] = False
        diag["overall_pass"] = False
        diag["failure_reason"] = "FILE_EMPTY"
        return diag

    age = diag["file_age_seconds"]
    if age is not None and age > freshness_threshold_seconds:
        diag["freshness_pass"] = False
        diag["overall_pass"] = False
        diag["failure_reason"] = "FILE_STALE"

    if check_upstream_staleness and signals_dir is not None:
        try:
            lc_mtime = path.stat().st_mtime
            mtimes = []
            signals_csv = signals_dir / "signals.csv"
            if signals_csv.is_file():
                mtimes.append(signals_csv.stat().st_mtime)
            mtimes.extend(
                p.stat().st_mtime for p in signals_dir.glob("signals*.csv") if p.is_file()
            )
            if mtimes and lc_mtime < max(mtimes):
                diag["freshness_pass"] = False
                diag["overall_pass"] = False
                if diag["failure_reason"] is None:
                    diag["failure_reason"] = "FILE_STALE"
                diag["upstream_stale"] = True
        except OSError:
            pass

    df, race, read_err, columns = _read_csv_with_race_detection(path)
    diag["race_condition_suspected"] = race
    if race:
        diag["failure_reason"] = "RACE_CONDITION_SUSPECTED"

    if read_err and df is None:
        diag["schema_pass"] = False
        diag["data_pass"] = False
        diag["overall_pass"] = False
        if diag["failure_reason"] is None:
            diag["failure_reason"] = "READ_ERROR"
        diag["read_error"] = read_err
        return diag

    assert df is not None
    diag["row_count"] = int(len(df))
    diag["column_count"] = int(len(columns))
    col_set = {c.strip() for c in columns}
    missing = [c for c in required if c not in col_set]
    if has_action_col and "lifecycle_action" not in col_set and "stance" not in col_set:
        missing.append("lifecycle_action|stance")
    diag["missing_columns"] = missing

    if missing:
        diag["schema_pass"] = False
        diag["overall_pass"] = False
        if diag["failure_reason"] is None:
            diag["failure_reason"] = "MISSING_COLUMNS"

    if df.empty:
        diag["data_pass"] = False
        if diag["failure_reason"] is None:
            diag["failure_reason"] = "FILE_EMPTY"

    # Row-date staleness (diagnostic mirror of lifecycle_truth.check_row_date_staleness).
    sig_col = next((c for c in ("signal_date", "date", "signal_as_of") if c in df.columns), None)
    asof_col = next(
        (c for c in ("as_of_date", "lifecycle_as_of_date", "asof_date") if c in df.columns), None
    )
    if sig_col and asof_col:
        try:
            sig = pd.to_datetime(df[sig_col], errors="coerce")
            asof = pd.to_datetime(df[asof_col], errors="coerce")
            bad = sig.notna() & asof.notna() & (sig.dt.normalize() > asof.dt.normalize())
            if bad.any():
                diag["data_pass"] = False
                diag["overall_pass"] = False
                if diag["failure_reason"] is None:
                    diag["failure_reason"] = "INVALID_TIMESTAMP"
                diag["invalid_timestamp_rows"] = int(bad.sum())
        except Exception:
            diag["data_pass"] = False
            diag["overall_pass"] = False
            if diag["failure_reason"] is None:
                diag["failure_reason"] = "INVALID_TIMESTAMP"

    if not diag["freshness_pass"] or not diag["schema_pass"] or not diag["data_pass"]:
        diag["overall_pass"] = False
    if diag["overall_pass"]:
        diag["failure_reason"] = None

    return diag


def _map_gate_reason_to_failure(gate_reason: str) -> str:
    reason = (gate_reason or "").strip().upper()
    mapping = {
        "MISSING_LIFECYCLE_FILE": "FILE_NOT_FOUND",
        "STALE_LIFECYCLE": "FILE_STALE",
        "STALE_EFFECTIVE": "FILE_STALE",
        "STALE_ROW_DATES": "INVALID_TIMESTAMP",
        "MISSING_COLUMNS": "MISSING_COLUMNS",
        "DUPLICATE_TICKERS": "SCHEMA_FAILURE",
        "INVALID_STATE_ACTION": "SCHEMA_FAILURE",
        "EMPTY": "FILE_EMPTY",
    }
    return mapping.get(reason, "UNKNOWN")


def evaluate_lifecycle_health(
    *,
    results_dir: Optional[Path] = None,
    lifecycle_path: Optional[Path] = None,
    effective_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate lifecycle artifact health for readiness diagnostics.
    Does not alter readiness pass/fail — mirrors thresholds for attribution only.
    """
    results_dir = Path(results_dir or RESULTS_DIR)
    lifecycle_path = lifecycle_path or (results_dir / "signal_lifecycle.csv")
    effective_path = effective_path or (results_dir / "signal_lifecycle_effective.csv")
    ts = _utc_iso()

    cfg_disabled = False
    if LIFECYCLE_CFG.is_file():
        try:
            with open(LIFECYCLE_CFG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg_disabled = isinstance(cfg, dict) and cfg.get("enabled") is False
        except (json.JSONDecodeError, OSError):
            pass

    base_diag = _evaluate_single_file(
        lifecycle_path,
        freshness_threshold_seconds=FRESHNESS_THRESHOLD_SECONDS,
        check_upstream_staleness=True,
        signals_dir=results_dir,
    )
    effective_diag = _evaluate_single_file(
        effective_path,
        freshness_threshold_seconds=FRESHNESS_THRESHOLD_SECONDS,
        check_upstream_staleness=False,
    )

    gate_reason = ""
    gate_status = "UNKNOWN"
    try:
        from services.lifecycle_truth import evaluate_lifecycle_gate

        gate = evaluate_lifecycle_gate(path=effective_path)
        gate_status = gate.status
        gate_reason = gate.reason or ""
        if gate.status != "OK" and effective_diag["failure_reason"] is None:
            effective_diag["failure_reason"] = _map_gate_reason_to_failure(gate_reason)
            effective_diag["overall_pass"] = False
    except Exception as exc:
        effective_diag["gate_eval_error"] = str(exc)

    readiness_pass = not cfg_disabled and base_diag["overall_pass"]
    readiness_failure_reason: Optional[str] = None
    if cfg_disabled:
        readiness_pass = False
        readiness_failure_reason = "UNKNOWN"
    elif not base_diag["overall_pass"]:
        readiness_failure_reason = base_diag["failure_reason"] or "UNKNOWN"

    doc: Dict[str, Any] = {
        "timestamp": ts,
        "readiness_primary_file": lifecycle_path.name,
        "lifecycle_logic_enabled": not cfg_disabled,
        "readiness_check_pass": readiness_pass,
        "readiness_failure_reason": readiness_failure_reason,
        "lifecycle_gate_status": gate_status,
        "lifecycle_gate_reason": gate_reason or None,
        # Primary schema fields (signal_lifecycle.csv — used by readiness check).
        **{
            k: base_diag[k]
            for k in (
                "lifecycle_file_exists",
                "lifecycle_file_path",
                "last_modified",
                "file_age_seconds",
                "freshness_threshold_seconds",
                "row_count",
                "column_count",
                "required_columns",
                "missing_columns",
                "freshness_pass",
                "schema_pass",
                "data_pass",
                "overall_pass",
                "failure_reason",
            )
        },
        "files": {
            "signal_lifecycle": base_diag,
            "signal_lifecycle_effective": effective_diag,
        },
    }
    return doc


def persist_lifecycle_diagnostics(
    diagnostics: Dict[str, Any],
    *,
    path: Path = DIAGNOSTICS_PATH,
) -> None:
    _atomic_write_json(diagnostics, path)


def build_failure_details(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    primary = diagnostics.get("files", {}).get("signal_lifecycle", diagnostics)
    return {
        "reason": diagnostics.get("readiness_failure_reason")
        or primary.get("failure_reason")
        or "UNKNOWN",
        "file_age_seconds": primary.get("file_age_seconds"),
        "missing_columns": list(primary.get("missing_columns") or []),
        "race_condition_suspected": bool(primary.get("race_condition_suspected")),
        "lifecycle_gate_reason": diagnostics.get("lifecycle_gate_reason"),
    }


def log_lifecycle_diagnostics(diagnostics: Dict[str, Any]) -> None:
    primary = diagnostics.get("files", {}).get("signal_lifecycle", diagnostics)
    effective = diagnostics.get("files", {}).get("signal_lifecycle_effective", {})
    reason = (
        diagnostics.get("readiness_failure_reason")
        or primary.get("failure_reason")
        or diagnostics.get("lifecycle_gate_reason")
        or "UNKNOWN"
    )
    print(
        "[LIFECYCLE_DIAGNOSTICS] "
        f"reason={reason} "
        f"file_age={primary.get('file_age_seconds')} "
        f"rows={primary.get('row_count')} "
        f"columns={primary.get('column_count')} "
        f"missing_columns={primary.get('missing_columns') or []} "
        f"effective_rows={effective.get('row_count')} "
        f"effective_reason={effective.get('failure_reason')}",
        flush=True,
    )
