"""
Phase 198Y — Execute and validate upstream data/prediction freshness recovery.

The recovery is strictly ordered and stops on the first failed step:
fetch_raw_data -> preprocess -> train_model -> generate_signals ->
apply_signal_lifecycle -> build_effective_lifecycle -> execution_readiness ->
capital_preservation_certification.

This module never places broker orders or creates operator authority artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GOVERNANCE = ROOT / "data" / "governance"

OUTPUT_JSON = RESULTS / "phase_198y_upstream_data_prediction_recovery_validation.json"
OUTPUT_MD = RESULTS / "phase_198y_upstream_data_prediction_recovery_validation.md"

APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PREDICTIONS_DIR = ROOT / "data" / "predictions"
SIGNALS_CSV = RESULTS / "signals.csv"
LIFECYCLE_CSV = RESULTS / "signal_lifecycle.csv"
EFFECTIVE_LIFECYCLE_CSV = RESULTS / "signal_lifecycle_effective.csv"
ASOF_DIAGNOSTICS = RESULTS / "signals_asof_diagnostics.json"
READINESS_PATH = RESULTS / "execution_readiness.json"
CERTIFICATION_PATH = RESULTS / "capital_preservation_certification.json"
WATCHDOG_STATUS = RESULTS / "watchdog_status.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
ALLOCATION_SUMMARY = RESULTS / "portfolio_allocation_summary.json"
ALLOCATION_RECOMMENDATIONS = RESULTS / "portfolio_allocation_recommendations.csv"
PHASE_198L = RESULTS / "phase_198l_gld_20pct_target_reduction_analysis.json"

FRESHNESS_SECONDS = 48 * 3600
GLD_TARGET_WEIGHT = 0.20
MARKER = "UPSTREAM_DATA_PREDICTION_RECOVERY_VALIDATION_ONLY"

RECOVERY_STEPS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("fetch_raw_data", "services.fetch_raw_data", ()),
    ("preprocess", "services.preprocess", ()),
    ("train_model", "services.train_model", ()),
    ("generate_signals", "services.generate_signals", ()),
    ("apply_signal_lifecycle", "services.apply_signal_lifecycle", ()),
    ("build_effective_lifecycle", "services.build_effective_lifecycle", ()),
    ("execution_readiness", "services.execution_readiness", ()),
    (
        "capital_preservation_certification",
        "services.capital_preservation_certification",
        ("--skip-stress-revalidation",),
    ),
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() in ("", "nan", "None"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() in ("", "nan", "None"):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        import pandas as pd

        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()
    except Exception:
        try:
            return date.fromisoformat(str(value)[:10])
        except Exception:
            return None


def _file_age_seconds(path: Path) -> Optional[float]:
    try:
        if path.is_file():
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            return round((_now() - modified).total_seconds(), 3)
    except OSError:
        pass
    return None


def _latest_layer_date(directory: Path, suffix: str) -> Optional[date]:
    """Return the minimum latest date across artifacts (all symbols must be current)."""
    import pandas as pd

    latest_dates: List[date] = []
    for path in sorted(directory.glob(f"*{suffix}")):
        try:
            frame = (
                pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
            )
        except Exception:
            continue
        latest: Optional[date] = None
        for column in ("date", "Date", "timestamp", "Datetime"):
            if column in frame.columns:
                values = pd.to_datetime(frame[column], errors="coerce").dropna()
                if len(values):
                    latest = values.max().date()
                    break
        if latest is None and isinstance(frame.index, pd.DatetimeIndex) and len(frame.index):
            latest = pd.to_datetime(frame.index).max().date()
        if latest is not None:
            latest_dates.append(latest)
    return min(latest_dates) if latest_dates else None


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return max(0, len(list(csv.reader(handle))) - 1)
    except Exception:
        return 0


def _run_step(name: str, module: str, extra_args: Tuple[str, ...]) -> Dict[str, Any]:
    command = [sys.executable, "-m", module, *extra_args]
    started = _now()
    try:
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip().splitlines()
        return {
            "name": name,
            "module": module,
            "command": " ".join(command),
            "status": "PASS" if proc.returncode == 0 else "FAIL",
            "exit_code": proc.returncode,
            "started_at": started.isoformat(),
            "finished_at": _now().isoformat(),
            "output_tail": output[-20:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "module": module,
            "command": " ".join(command),
            "status": "TIMEOUT",
            "exit_code": 124,
            "started_at": started.isoformat(),
            "finished_at": _now().isoformat(),
            "output_tail": [str(exc)],
        }
    except Exception as exc:
        return {
            "name": name,
            "module": module,
            "command": " ".join(command),
            "status": "ERROR",
            "exit_code": 1,
            "started_at": started.isoformat(),
            "finished_at": _now().isoformat(),
            "output_tail": [str(exc)],
        }


def _gld_state() -> Tuple[int, float]:
    phase_198l = _load_json(PHASE_198L)
    if phase_198l:
        return (
            _safe_int(phase_198l.get("current_gld_qty")),
            _safe_float(phase_198l.get("current_gld_weight")),
        )

    qty = 0
    gld_value = 0.0
    total = 0.0
    if POSITIONS.is_file():
        try:
            with POSITIONS.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    value = abs(_safe_float(row.get("market_value") or row.get("value")))
                    total += value
                    symbol = str(row.get("symbol") or row.get("ticker") or "").upper()
                    if symbol == "GLD":
                        qty = _safe_int(row.get("qty") or row.get("quantity"))
                        gld_value = value
        except Exception:
            pass
    return qty, round(gld_value / total, 6) if total else 0.0


def _allocation_counts() -> Dict[str, int]:
    summary = _load_json(ALLOCATION_SUMMARY)
    counts = {
        "open_new_count": _safe_int(summary.get("open_new_count")),
        "add_count": _safe_int(summary.get("add_count")),
        "exit_count": _safe_int(summary.get("exit_count")),
        "trim_count": _safe_int(summary.get("trim_count")),
    }
    if ALLOCATION_RECOMMENDATIONS.is_file():
        try:
            observed = {key: 0 for key in counts}
            mapping = {
                "OPEN_NEW": "open_new_count",
                "ADD": "add_count",
                "EXIT": "exit_count",
                "TRIM": "trim_count",
            }
            with ALLOCATION_RECOMMENDATIONS.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    action = str(row.get("recommended_action") or row.get("action") or "").upper()
                    if action in mapping:
                        observed[mapping[action]] += 1
            if any(observed.values()):
                counts = observed
        except Exception:
            pass
    return counts


def _render_markdown(payload: Dict[str, Any]) -> str:
    audit = payload["audit"]
    step_lines = "\n".join(
        f"- `{step['name']}`: **{step['status']}**"
        + (f" (exit {step.get('exit_code')})" if step.get("exit_code") is not None else "")
        for step in audit["steps"]
    )
    return f"""# TRITON Phase 198Y — Upstream Data / Prediction Recovery

**Generated:** {payload["generated_at"]}
**Status:** {payload["status"]}
**Recovery state:** `{audit["recovery_state"]}`
**Failed step:** `{audit["failed_step"]}`

## Recovery steps

{step_lines}

## Freshness after recovery

| Layer | Latest date / age | Required / stale |
|---|---:|---:|
| Raw data | {audit["raw_data_latest_date"]} | {audit["required_session_date"]} |
| Processed data | {audit["processed_data_latest_date"]} | {audit["required_session_date"]} |
| Predictions | {audit["prediction_latest_date"]} | current={audit["predictions_current_for_session"]} |
| signals.csv | {audit["signals_file_age_seconds"]}s | stale={audit["signals_stale_after_recovery"]} |
| signal_lifecycle.csv | {audit["lifecycle_file_age_seconds"]}s | stale={audit["lifecycle_stale_after_recovery"]} |
| signal_lifecycle_effective.csv | {audit["effective_lifecycle_file_age_seconds"]}s | stale={audit["effective_lifecycle_stale_after_recovery"]} |

## Readiness / certification

- Readiness: `{audit["execution_readiness_status"]}`; failed checks: `{audit["execution_readiness_failed_checks"]}`
- Certification: `{audit["certification_status"]}`; failed requirements: `{audit["certification_failed_requirements"]}`

## Safety

- Broker connected: `{audit["broker_connected"]}`
- Open orders: `{audit["open_orders_count"]}`
- Paper arm valid: `{audit["paper_arm_valid"]}`
- Live execution blocked: `{audit["live_execution_blocked"]}`
- Approval artifact present: `{audit["approval_artifact_present"]}`
- Broker-auth artifact present: `{audit["broker_auth_artifact_present"]}`
- GLD qty / weight / target satisfied: `{audit["gld_qty"]}` / `{audit["gld_weight"]}` / `{audit["gld_target_satisfied"]}`
- Allocation OPEN_NEW / ADD / EXIT / TRIM: `{audit["allocation_open_new_count"]}` / `{audit["allocation_add_count"]}` / `{audit["allocation_exit_count"]}` / `{audit["allocation_trim_count"]}`
- Broker orders submitted by this phase: `false`

## Runtime posture

{audit["recommended_runtime_posture"]}

## Next step

{audit["recommended_next_step"]}

---

*{MARKER}*
"""


def run_validation(*, execute_recovery: bool = True, verbose: bool = True) -> Dict[str, Any]:
    from services.generate_signals import STRICT_ASOF
    from services.paper_arm_status import build_paper_arm_status
    from services.portfolio_history_sync import load_live_execution_flag

    steps: List[Dict[str, Any]] = []
    failed_step: Optional[str] = None
    if execute_recovery:
        for name, module, extra_args in RECOVERY_STEPS:
            result = _run_step(name, module, extra_args)
            steps.append(result)
            if result["exit_code"] != 0:
                failed_step = name
                break
    else:
        steps = [
            {
                "name": name,
                "module": module,
                "status": "NOT_RUN",
                "exit_code": None,
                "output_tail": [],
            }
            for name, module, _ in RECOVERY_STEPS
        ]

    completed_names = {step["name"] for step in steps}
    for name, module, _ in RECOVERY_STEPS:
        if name not in completed_names:
            steps.append(
                {
                    "name": name,
                    "module": module,
                    "status": "NOT_RUN",
                    "exit_code": None,
                    "output_tail": [],
                }
            )
    step_status = {step["name"]: step["status"] for step in steps}

    asof = _load_json(ASOF_DIAGNOSTICS)
    required_session = _parse_date(asof.get("session_asof_date"))
    raw_latest = _latest_layer_date(RAW_DIR, ".csv")
    processed_latest = _latest_layer_date(PROCESSED_DIR, ".parquet")
    prediction_latest = _latest_layer_date(PREDICTIONS_DIR, "_predictions.parquet")
    if prediction_latest is None:
        prediction_latest = _latest_layer_date(PREDICTIONS_DIR, "_predictions.csv")

    predictions_current = bool(
        required_session and prediction_latest and prediction_latest >= required_session
    )
    strict_asof_passed = bool(
        STRICT_ASOF and step_status.get("generate_signals") == "PASS" and not bool(asof.get("fail"))
    )

    signals_age = _file_age_seconds(SIGNALS_CSV)
    lifecycle_age = _file_age_seconds(LIFECYCLE_CSV)
    effective_age = _file_age_seconds(EFFECTIVE_LIFECYCLE_CSV)
    signals_stale = signals_age is None or signals_age > FRESHNESS_SECONDS
    lifecycle_stale = lifecycle_age is None or lifecycle_age > FRESHNESS_SECONDS
    effective_stale = effective_age is None or effective_age > FRESHNESS_SECONDS

    readiness = _load_json(READINESS_PATH)
    certification = _load_json(CERTIFICATION_PATH)
    watchdog = _load_json(WATCHDOG_STATUS)
    arm = build_paper_arm_status(verbose_notes=False)
    allocation = _allocation_counts()
    gld_qty, gld_weight = _gld_state()

    live_blocked = not bool(load_live_execution_flag()) and not bool(
        readiness.get("live_execution_permitted")
    )
    approval_present = APPROVAL_PATH.is_file()
    broker_auth_present = BROKER_AUTH_PATH.is_file()
    open_orders = _count_csv_rows(OPEN_ORDERS)
    broker_connected = bool(watchdog.get("broker_connected")) or bool(
        (readiness.get("checks") or {}).get("broker")
    )
    paper_arm_valid = str(arm.get("status") or "") == "VALID"
    gld_target_satisfied = bool(
        gld_weight <= GLD_TARGET_WEIGHT + 0.005
        or _load_json(PHASE_198L).get("target_already_satisfied")
    )

    all_steps_passed = all(step_status.get(name) == "PASS" for name, _, _ in RECOVERY_STEPS)
    recovery_state = (
        "RECOVERY_COMPLETE"
        if all_steps_passed
        else f"RECOVERY_SAFE_BLOCKED_{(failed_step or 'NOT_EXECUTED').upper()}"
    )

    if not live_blocked or approval_present or broker_auth_present:
        status = "FAIL"
    elif all_steps_passed and predictions_current and not signals_stale:
        status = "PASS"
    else:
        status = "REVIEW_REQUIRED"

    if all_steps_passed:
        posture = (
            "PAPER_MONITOR — upstream recovery completed under strict AS_OF. "
            "Allocation remains the sole authority for normal paper entries; live stays blocked."
        )
        next_step = (
            "Observe the next market-open cycle and verify readiness/certification remain "
            "fresh; do not create execution authority unless separately requested."
        )
    else:
        failed_result = next((step for step in steps if step["name"] == failed_step), {})
        detail = next(
            (
                line
                for line in reversed(failed_result.get("output_tail") or [])
                if str(line).strip()
            ),
            "unknown failure",
        )
        posture = (
            f"SAFE_BLOCK — recovery stopped at {failed_step}: {detail}. "
            "No downstream recovery steps or broker actions should run; live stays blocked."
        )
        next_step = (
            f"Repair `{failed_step}` using its diagnostics, then rerun Phase 198Y from "
            "fetch_raw_data. Do not bypass strict AS_OF."
        )

    audit: Dict[str, Any] = {
        "recovery_state": recovery_state,
        "failed_step": failed_step,
        "steps": steps,
        "fetch_raw_data_status": step_status.get("fetch_raw_data"),
        "preprocess_status": step_status.get("preprocess"),
        "train_model_status": step_status.get("train_model"),
        "generate_signals_status": step_status.get("generate_signals"),
        "apply_signal_lifecycle_status": step_status.get("apply_signal_lifecycle"),
        "build_effective_lifecycle_status": step_status.get("build_effective_lifecycle"),
        "required_session_date": str(required_session) if required_session else None,
        "raw_data_latest_date": str(raw_latest) if raw_latest else None,
        "processed_data_latest_date": (str(processed_latest) if processed_latest else None),
        "prediction_latest_date": (str(prediction_latest) if prediction_latest else None),
        "predictions_current_for_session": predictions_current,
        "strict_asof_enabled": bool(STRICT_ASOF),
        "strict_asof_passed": strict_asof_passed,
        "signals_file_age_seconds": signals_age,
        "lifecycle_file_age_seconds": lifecycle_age,
        "effective_lifecycle_file_age_seconds": effective_age,
        "signals_stale_after_recovery": signals_stale,
        "lifecycle_stale_after_recovery": lifecycle_stale,
        "effective_lifecycle_stale_after_recovery": effective_stale,
        "execution_readiness_status": readiness.get("readiness_status"),
        "execution_readiness_failed_checks": list(readiness.get("failed_checks") or []),
        "certification_status": certification.get("certification_status"),
        "certification_failed_requirements": list(certification.get("failed_requirements") or []),
        "allocation_open_new_count": allocation["open_new_count"],
        "allocation_add_count": allocation["add_count"],
        "allocation_exit_count": allocation["exit_count"],
        "allocation_trim_count": allocation["trim_count"],
        "broker_connected": broker_connected,
        "open_orders_count": open_orders,
        "paper_arm_valid": paper_arm_valid,
        "live_execution_blocked": live_blocked,
        "approval_artifact_present": approval_present,
        "broker_auth_artifact_present": broker_auth_present,
        "gld_qty": gld_qty,
        "gld_weight": gld_weight,
        "gld_target_satisfied": gld_target_satisfied,
        "broker_orders_submitted": False,
        "recommended_runtime_posture": posture,
        "recommended_next_step": next_step,
    }

    payload: Dict[str, Any] = {
        "phase": "198Y",
        "generated_at": _now().isoformat(),
        "status": status,
        "marker": MARKER,
        "audit": audit,
        "safety": {
            "broker_orders_submitted": False,
            "live_execution_blocked": live_blocked,
            "approval_artifact_present": approval_present,
            "broker_auth_artifact_present": broker_auth_present,
            "open_orders_count": open_orders,
            "strict_asof_not_loosened": bool(STRICT_ASOF),
        },
    }
    _atomic_write(OUTPUT_JSON, json.dumps(payload, indent=2, default=str) + "\n")
    _atomic_write(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(f"[198Y] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198Y] wrote {OUTPUT_MD.relative_to(ROOT)}")
        print(
            f"[198Y] status={status} recovery={recovery_state} failed={failed_step} "
            f"raw={raw_latest} processed={processed_latest} predictions={prediction_latest} "
            f"strict_asof_passed={strict_asof_passed} open_orders={open_orders} "
            f"live_blocked={live_blocked}"
        )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 198Y upstream data/prediction recovery validation"
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect current state without executing recovery modules.",
    )
    args = parser.parse_args(argv)
    result = run_validation(
        execute_recovery=not args.inspect_only,
        verbose=not args.quiet,
    )
    return 0 if result.get("status") in ("PASS", "REVIEW_REQUIRED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
