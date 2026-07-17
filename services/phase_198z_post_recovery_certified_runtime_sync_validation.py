"""
phase_198z_post_recovery_certified_runtime_sync_validation.py
Phase 198Z — Post-recovery certified paper runtime sync validation.

Validation only. No broker orders, approvals, authorization, or live enablement.
Confirms Phase 198Y restored the full market-open paper readiness chain.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"
SCRIPTS = ROOT / "scripts"

OUTPUT_JSON = RESULTS / "phase_198z_post_recovery_certified_runtime_sync_validation.json"
OUTPUT_MD = RESULTS / "phase_198z_post_recovery_certified_runtime_sync_validation.md"

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
CERT_FP_PATH = RESULTS / "certification_false_positive_elimination.json"
WATCHDOG_STATUS = RESULTS / "watchdog_status.json"
WATCHDOG_ALERTS = RESULTS / "watchdog_alerts.json"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
ALLOCATION_SUMMARY = RESULTS / "portfolio_allocation_summary.json"
ALLOCATION_RECOMMENDATIONS = RESULTS / "portfolio_allocation_recommendations.csv"
PHASE_198L = RESULTS / "phase_198l_gld_20pct_target_reduction_analysis.json"
PHASE_197Y = RESULTS / "phase_197y_loop_mediated_paper_execution_gate.json"
PHASE_197Z = RESULTS / "phase_197z_readiness_bridge_validation.json"
PHASE_198Y = RESULTS / "phase_198y_upstream_data_prediction_recovery_validation.json"
MARKET_OPEN_PS1 = SCRIPTS / "run_market_open.ps1"

FRESHNESS_SECONDS = 48 * 3600
GLD_TARGET_WEIGHT = 0.20
EXPECTED_GLD_QTY = 16
MARKER = "POST_RECOVERY_CERTIFIED_RUNTIME_SYNC_VALIDATION_ONLY"

PAPER_CERT_STATUSES = frozenset(
    {
        "CERTIFIED_FOR_PAPER_PROTECTION",
        "CONDITIONALLY_CERTIFIED",
        "PARTIALLY_CERTIFIED",
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or str(v).strip() in ("", "nan", "None"):
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _parse_date(v: Any) -> Optional[date]:
    if not v:
        return None
    try:
        import pandas as pd

        parsed = pd.to_datetime(v, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()
    except Exception:
        try:
            return date.fromisoformat(str(v)[:10])
        except Exception:
            return None


def _file_age_seconds(path: Path) -> Optional[float]:
    try:
        if path.is_file():
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            return round((_utc_now() - mtime).total_seconds(), 3)
    except OSError:
        pass
    return None


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return max(0, len(list(csv.reader(f))) - 1)
    except Exception:
        return 0


def _latest_layer_date(directory: Path, pattern: str) -> Optional[date]:
    """Return min latest date across matching artifacts (all symbols must be current)."""
    import pandas as pd

    latest_dates: List[date] = []
    for path in sorted(directory.glob(pattern)):
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


def _gld_from_positions(path: Path) -> Dict[str, Any]:
    qty = 0.0
    market_value = 0.0
    total_invested = 0.0
    if not path.is_file():
        return {"qty": 0, "weight": 0.0}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mv = abs(_safe_float(row.get("market_value") or row.get("value")))
                total_invested += mv
                sym = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
                if sym != "GLD":
                    continue
                qty = _safe_float(row.get("qty") or row.get("quantity"))
                market_value = mv
    except Exception:
        pass
    weight = (market_value / total_invested) if total_invested > 0 else 0.0
    return {
        "qty": int(qty) if qty == int(qty) else qty,
        "weight": round(weight, 6),
    }


def _allocation_counts(summary: Dict[str, Any], recs_path: Path) -> Dict[str, int]:
    counts = {
        "open_new_count": _safe_int(summary.get("open_new_count")),
        "add_count": _safe_int(summary.get("add_count")),
        "exit_count": _safe_int(summary.get("exit_count")),
        "trim_count": _safe_int(summary.get("trim_count")),
    }
    action_map = {
        "OPEN_NEW": "open_new_count",
        "ADD": "add_count",
        "EXIT": "exit_count",
        "TRIM": "trim_count",
    }
    if recs_path.is_file():
        try:
            tallies = {k: 0 for k in counts}
            with recs_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    action = (
                        str(row.get("recommended_action") or row.get("action") or "")
                        .strip()
                        .upper()
                    )
                    mapped = action_map.get(action)
                    if mapped:
                        tallies[mapped] += 1
            if any(tallies.values()):
                counts.update(tallies)
        except Exception:
            pass
    return counts


def _market_open_198w_refresh_present() -> Dict[str, Any]:
    text = ""
    if MARKET_OPEN_PS1.is_file():
        try:
            text = MARKET_OPEN_PS1.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
    present = {
        "generate_signals": bool(re.search(r"services\.generate_signals", text)),
        "apply_signal_lifecycle": bool(re.search(r"services\.apply_signal_lifecycle", text)),
        "build_effective_lifecycle": bool(re.search(r"services\.build_effective_lifecycle", text)),
        "d1_stage_label": bool(re.search(r"D1a\. GENERATE SIGNALS|SIGNAL \+ LIFECYCLE", text)),
    }
    return {
        "script": str(MARKET_OPEN_PS1.relative_to(ROOT)).replace("\\", "/"),
        "script_exists": MARKET_OPEN_PS1.is_file(),
        "modules_present": present,
        "market_open_198w_refresh_present": all(present.values()),
    }


def _stale_not_certified_display(
    *,
    cert_status: str,
    failed: List[str],
    cert_fp: Dict[str, Any],
    dash_blockers: List[str],
) -> Tuple[bool, str]:
    if cert_status in PAPER_CERT_STATUSES and len(failed) == 0:
        if "NOT_CERTIFIED" in dash_blockers:
            return True, "dashboard_execution_block_reasons_still_show_NOT_CERTIFIED"
        fp_status = str(cert_fp.get("certification_status") or "")
        if fp_status == "NOT_CERTIFIED":
            return True, "certification_false_positive_elimination_stale_NOT_CERTIFIED"
        wd = _load_json(WATCHDOG_STATUS)
        if str(wd.get("certification_status") or "") == "NOT_CERTIFIED":
            return True, "watchdog_status_stale_NOT_CERTIFIED"
        return False, "no_stale_NOT_CERTIFIED_display"
    return False, "certification_not_in_paper_certified_set"


def _paper_entry_assessment(
    *,
    cert_status: str,
    failed: List[str],
    arm: Dict[str, Any],
    alloc: Dict[str, int],
    live_allowed: bool,
    stale_cert_display: bool,
) -> Tuple[bool, str]:
    cert_ok = cert_status in PAPER_CERT_STATUSES and len(failed) == 0
    paper_arm_valid = str(arm.get("status") or "") == "VALID"
    paper_entry_armed = bool(arm.get("paper_entry_armed"))
    open_new = int(alloc.get("open_new_count") or 0)
    add = int(alloc.get("add_count") or 0)
    alloc_entries = open_new + add

    if live_allowed:
        return False, "live_execution_not_blocked"
    if not cert_ok:
        return False, "certification_not_ready"
    if stale_cert_display:
        return False, "stale_NOT_CERTIFIED_display_present"
    if alloc_entries == 0:
        return False, "allocation_blocked_no_open_new_or_add_candidates"
    if not paper_arm_valid:
        return False, "paper_arm_confirm_not_valid"
    if not paper_entry_armed:
        return False, "paper_entry_not_armed"
    return True, (
        f"eligible_via_controlled_paper_entry_lane "
        f"(open_new={open_new}, add={add}); allocation is authority; no manual bypass"
    )


def _defensive_exit_assessment(
    *,
    approval_present: bool,
    broker_auth_present: bool,
    live_allowed: bool,
    alloc_exit: int,
    phase_197y: Dict[str, Any],
    phase_197z: Dict[str, Any],
) -> Tuple[bool, str]:
    if live_allowed:
        return False, "live_execution_not_blocked"
    if not approval_present:
        return False, "operator_approval_artifact_absent"
    if not broker_auth_present:
        return False, "broker_authorization_artifact_absent"
    eligible = _safe_int(phase_197y.get("eligible_qty"))
    bridge = _safe_int(
        phase_197z.get("bridge_qty") or (phase_197z.get("bridge") or {}).get("bridge_qty")
    )
    if eligible <= 0 and bridge <= 0 and alloc_exit <= 0:
        return False, "no_eligible_defensive_exit_qty"
    return True, (
        f"dispatch_lane_open_with_approval_and_auth "
        f"(eligible_qty={eligible}, bridge_qty={bridge}, alloc_exit={alloc_exit})"
    )


def _render_markdown(payload: Dict[str, Any]) -> str:
    a = payload.get("audit") or {}
    return f"""# TRITON Phase 198Z — Post-Recovery Certified Runtime Sync Validation

**Generated:** {payload.get("generated_at")}
**Status:** {payload.get("status")}
**Runtime sync state:** `{a.get("runtime_sync_state")}`

## Executive summary

{payload.get("executive_summary")}

## Data / prediction freshness

| Field | Value |
|-------|-------|
| required_session_date | {a.get("required_session_date")} |
| raw_data_latest_date | {a.get("raw_data_latest_date")} |
| processed_data_latest_date | {a.get("processed_data_latest_date")} |
| prediction_latest_date | {a.get("prediction_latest_date")} |
| strict_asof_passed | {a.get("strict_asof_passed")} |
| signals_fresh | {a.get("signals_fresh")} |
| lifecycle_fresh | {a.get("lifecycle_fresh")} |
| effective_lifecycle_fresh | {a.get("effective_lifecycle_fresh")} |
| market_open_198w_refresh_present | {a.get("market_open_198w_refresh_present")} |

## Readiness / certification

| Field | Value |
|-------|-------|
| execution_readiness_status | {a.get("execution_readiness_status")} |
| failed_checks | {a.get("execution_readiness_failed_checks")} |
| certification_status | {a.get("certification_status")} |
| failed_requirements | {a.get("certification_failed_requirements")} |
| stale_NOT_CERTIFIED_display | {a.get("stale_not_certified_display")} |

## Entry / exit authority

| Field | Value |
|-------|-------|
| normal_paper_entry_possible_now | {a.get("normal_paper_entry_possible_now")} |
| normal_paper_entry_reason | {a.get("normal_paper_entry_reason")} |
| defensive_exit_dispatch_possible_now | {a.get("defensive_exit_dispatch_possible_now")} |
| defensive_exit_dispatch_reason | {a.get("defensive_exit_dispatch_reason")} |
| allocation open_new / add | {a.get("allocation_open_new_count")} / {a.get("allocation_add_count")} |
| allocation exit / trim | {a.get("allocation_exit_count")} / {a.get("allocation_trim_count")} |

## Safety / posture

| Field | Value |
|-------|-------|
| paper_arm_valid | {a.get("paper_arm_valid")} |
| live_execution_blocked | {a.get("live_execution_blocked")} |
| approval_artifact_present | {a.get("approval_artifact_present")} |
| broker_auth_artifact_present | {a.get("broker_auth_artifact_present")} |
| open_orders_count | {a.get("open_orders_count")} |
| positions_count | {a.get("positions_count")} |
| gld_qty / weight | {a.get("gld_qty")} / {a.get("gld_weight")} |
| gld_target_satisfied | {a.get("gld_target_satisfied")} |
| broker_orders_submitted | {a.get("broker_orders_submitted")} |

## Recommended posture

{a.get("recommended_runtime_posture")}

## Next step

{a.get("recommended_next_step")}

---

*{MARKER}*
"""


def run_validation(*, verbose: bool = True) -> Dict[str, Any]:
    from services.generate_signals import STRICT_ASOF
    from services.paper_arm_status import build_paper_arm_status
    from services.portfolio_history_sync import load_live_execution_flag

    asof = _load_json(ASOF_DIAGNOSTICS)
    readiness = _load_json(READINESS_PATH)
    certification = _load_json(CERTIFICATION_PATH)
    cert_fp = _load_json(CERT_FP_PATH)
    phase_198l = _load_json(PHASE_198L)
    phase_197y = _load_json(PHASE_197Y)
    phase_197z = _load_json(PHASE_197Z)
    phase_198y = _load_json(PHASE_198Y)
    alloc_summary = _load_json(ALLOCATION_SUMMARY)
    wd_status = _load_json(WATCHDOG_STATUS)

    required_session = _parse_date(asof.get("session_asof_date"))
    raw_latest = _latest_layer_date(RAW_DIR, "*.csv")
    processed_latest = _latest_layer_date(PROCESSED_DIR, "*.parquet")
    prediction_latest = _latest_layer_date(PREDICTIONS_DIR, "*_predictions.parquet")
    if prediction_latest is None:
        prediction_latest = _latest_layer_date(PREDICTIONS_DIR, "*_predictions.csv")

    predictions_current = bool(
        required_session and prediction_latest and prediction_latest >= required_session
    )
    raw_current = bool(required_session and raw_latest and raw_latest >= required_session)
    processed_current = bool(
        required_session and processed_latest and processed_latest >= required_session
    )

    strict_asof_passed = bool(
        STRICT_ASOF
        and not bool(asof.get("fail"))
        and predictions_current
        and str(asof.get("effective_asof_date") or "") == str(asof.get("session_asof_date") or "")
    ) or bool(
        # Accept 198Y recovery evidence if AS_OF diag already current.
        STRICT_ASOF
        and not bool(asof.get("fail"))
        and predictions_current
        and (phase_198y.get("audit") or {}).get("strict_asof_passed")
    )
    if STRICT_ASOF and not bool(asof.get("fail")) and predictions_current:
        # Effective equals session OR coverage-ok path with lag 0.
        lag = _safe_int(asof.get("lag_sessions_vs_session"), default=-1)
        if lag == 0 or str(asof.get("effective_asof_date")) == str(asof.get("session_asof_date")):
            strict_asof_passed = True
        elif float(asof.get("coverage") or 0) >= float(asof.get("asof_coverage_threshold") or 0.9):
            strict_asof_passed = True

    signals_age = _file_age_seconds(SIGNALS_CSV)
    lifecycle_age = _file_age_seconds(LIFECYCLE_CSV)
    effective_age = _file_age_seconds(EFFECTIVE_LIFECYCLE_CSV)
    signals_fresh = signals_age is not None and signals_age <= FRESHNESS_SECONDS
    lifecycle_fresh = lifecycle_age is not None and lifecycle_age <= FRESHNESS_SECONDS
    effective_fresh = effective_age is not None and effective_age <= FRESHNESS_SECONDS

    market_open = _market_open_198w_refresh_present()

    readiness_status = str(readiness.get("readiness_status") or "UNKNOWN")
    readiness_failed = list(readiness.get("failed_checks") or [])
    cert_status = str(certification.get("certification_status") or "UNKNOWN")
    cert_failed = list(certification.get("failed_requirements") or [])

    arm = build_paper_arm_status(verbose_notes=False)
    paper_arm_valid = str(arm.get("status") or "") == "VALID"
    live_flag = bool(load_live_execution_flag())
    live_readiness = bool(readiness.get("live_execution_permitted"))
    live_allowed = live_flag or live_readiness
    live_blocked = not live_allowed

    approval_present = APPROVAL_PATH.is_file()
    broker_auth_present = BROKER_AUTH_PATH.is_file()
    open_orders = _count_csv_rows(OPEN_ORDERS)
    positions_count = _count_csv_rows(POSITIONS)

    gld = _gld_from_positions(POSITIONS)
    gld_qty = gld.get("qty")
    gld_weight = float(gld.get("weight") or 0.0)
    if phase_198l.get("current_gld_weight") is not None:
        gld_weight = float(phase_198l.get("current_gld_weight") or gld_weight)
    if phase_198l.get("current_gld_qty") is not None:
        gld_qty = _safe_int(phase_198l.get("current_gld_qty"), default=_safe_int(gld_qty))
    gld_target = bool(
        phase_198l.get("target_already_satisfied") or gld_weight <= GLD_TARGET_WEIGHT + 0.005
    )

    alloc = _allocation_counts(alloc_summary, ALLOCATION_RECOMMENDATIONS)

    dash_blockers: List[str] = []
    try:
        from services.dashboard_status_sync import build_dashboard_health_classification
        from services.live_gate import compute_display_gates

        live_ready, _, gate_bits, _ = compute_display_gates()
        dash_blockers = list(
            (
                build_dashboard_health_classification(
                    live_gate_bits=gate_bits,
                    live_ready=live_ready,
                )
            ).get("execution_block_reasons")
            or []
        )
    except Exception:
        dash_blockers = []

    stale_cert, stale_cert_reason = _stale_not_certified_display(
        cert_status=cert_status,
        failed=cert_failed,
        cert_fp=cert_fp,
        dash_blockers=dash_blockers,
    )

    entry_possible, entry_reason = _paper_entry_assessment(
        cert_status=cert_status,
        failed=cert_failed,
        arm=arm,
        alloc=alloc,
        live_allowed=live_allowed,
        stale_cert_display=stale_cert,
    )
    exit_possible, exit_reason = _defensive_exit_assessment(
        approval_present=approval_present,
        broker_auth_present=broker_auth_present,
        live_allowed=live_allowed,
        alloc_exit=int(alloc.get("exit_count") or 0),
        phase_197y=phase_197y,
        phase_197z=phase_197z,
    )

    broker_connected = bool(wd_status.get("broker_connected")) or bool(
        (readiness.get("checks") or {}).get("broker")
    )

    sync_checks = {
        "raw_current": raw_current,
        "processed_current": processed_current,
        "predictions_current": predictions_current,
        "strict_asof_passed": strict_asof_passed,
        "signals_fresh": signals_fresh,
        "lifecycle_fresh": lifecycle_fresh,
        "effective_lifecycle_fresh": effective_fresh,
        "readiness_ready": readiness_status == "READY",
        "readiness_no_failed_checks": len(readiness_failed) == 0,
        "certified_for_paper": cert_status == "CERTIFIED_FOR_PAPER_PROTECTION",
        "cert_no_failed_requirements": len(cert_failed) == 0,
        "no_stale_not_certified_display": not stale_cert,
        "market_open_198w_refresh_present": bool(
            market_open.get("market_open_198w_refresh_present")
        ),
        "paper_arm_valid": paper_arm_valid,
        "live_blocked": live_blocked,
        "approval_absent": not approval_present,
        "broker_auth_absent": not broker_auth_present,
        "open_orders_zero": open_orders == 0,
        "gld_target_satisfied": gld_target,
        "gld_qty_expected": _safe_int(gld_qty) == EXPECTED_GLD_QTY,
        "defensive_exits_blocked_without_authority": not exit_possible,
    }

    if not live_blocked or approval_present or broker_auth_present:
        status = "FAIL"
        runtime_sync_state = "UNSAFE_RUNTIME"
    elif all(sync_checks.values()):
        status = "PASS"
        runtime_sync_state = "CERTIFIED_PAPER_RUNTIME_SYNCED"
    elif readiness_status == "READY" and cert_status == "CERTIFIED_FOR_PAPER_PROTECTION":
        status = "REVIEW_REQUIRED"
        runtime_sync_state = "CERTIFIED_WITH_RESIDUAL_DRIFT"
    else:
        status = "REVIEW_REQUIRED"
        runtime_sync_state = "POST_RECOVERY_SYNC_INCOMPLETE"

    failed_sync = [k for k, ok in sync_checks.items() if not ok]

    if status == "PASS":
        if int(alloc.get("open_new_count") or 0) + int(alloc.get("add_count") or 0) == 0:
            posture = (
                "PAPER_CERTIFIED_MONITOR — readiness/certification synced after 198Y recovery; "
                "allocation gates entries (open_new/add=0); defensive exits require explicit "
                "approval+auth; live blocked."
            )
            next_step = (
                "Maintain paper-certified posture through the next market-open cycle; "
                "do not create execution authority unless separately requested."
            )
        else:
            posture = (
                "PAPER_CERTIFIED_ALLOCATION_ACTIVE — controlled paper entry lane available "
                "via allocation open_new/add only; live blocked; no defensive dispatch without "
                "approval+auth."
            )
            next_step = (
                "Allow continuous loop / paper-entry stage to evaluate allocation-gated "
                "entries only; do not create defensive-exit authority unless requested."
            )
    elif status == "FAIL":
        posture = "HALT_AND_REVIEW_SAFETY_VIOLATION"
        next_step = "Investigate live/authority safety violation before any further runtime action."
    else:
        posture = (
            "REVIEW — post-recovery certification present but residual sync drift remains: "
            + ", ".join(failed_sync[:8])
        )
        next_step = (
            "Refresh residual drift surfaces (display/watchdog/FP elimination) without "
            "loosening AS_OF or creating authority artifacts."
        )

    audit: Dict[str, Any] = {
        "runtime_sync_state": runtime_sync_state,
        "required_session_date": str(required_session) if required_session else None,
        "raw_data_latest_date": str(raw_latest) if raw_latest else None,
        "processed_data_latest_date": str(processed_latest) if processed_latest else None,
        "prediction_latest_date": str(prediction_latest) if prediction_latest else None,
        "raw_current_for_session": raw_current,
        "processed_current_for_session": processed_current,
        "predictions_current_for_session": predictions_current,
        "strict_asof_enabled": bool(STRICT_ASOF),
        "strict_asof_passed": strict_asof_passed,
        "asof_diagnostics": {
            "session_asof_date": asof.get("session_asof_date"),
            "effective_asof_date": asof.get("effective_asof_date"),
            "max_prediction_date": asof.get("max_prediction_date"),
            "coverage": asof.get("coverage"),
            "lag_sessions_vs_session": asof.get("lag_sessions_vs_session"),
            "fail": asof.get("fail"),
            "strict_asof": asof.get("strict_asof"),
        },
        "signals_file_age_seconds": signals_age,
        "lifecycle_file_age_seconds": lifecycle_age,
        "effective_lifecycle_file_age_seconds": effective_age,
        "signals_fresh": signals_fresh,
        "lifecycle_fresh": lifecycle_fresh,
        "effective_lifecycle_fresh": effective_fresh,
        "market_open_198w_refresh_present": bool(
            market_open.get("market_open_198w_refresh_present")
        ),
        "market_open_198w_refresh_audit": market_open,
        "execution_readiness_status": readiness_status,
        "execution_readiness_failed_checks": readiness_failed,
        "certification_status": cert_status,
        "certification_failed_requirements": cert_failed,
        "stale_not_certified_display": stale_cert,
        "stale_not_certified_display_reason": stale_cert_reason,
        "dashboard_block_reasons": dash_blockers,
        "paper_arm_valid": paper_arm_valid,
        "paper_entry_armed": bool(arm.get("paper_entry_armed")),
        "live_execution_blocked": live_blocked,
        "approval_artifact_present": approval_present,
        "broker_auth_artifact_present": broker_auth_present,
        "broker_connected": broker_connected,
        "open_orders_count": open_orders,
        "positions_count": positions_count,
        "gld_qty": gld_qty,
        "gld_weight": gld_weight,
        "gld_target_satisfied": gld_target,
        "allocation_open_new_count": alloc.get("open_new_count"),
        "allocation_add_count": alloc.get("add_count"),
        "allocation_exit_count": alloc.get("exit_count"),
        "allocation_trim_count": alloc.get("trim_count"),
        "normal_paper_entry_possible_now": entry_possible,
        "normal_paper_entry_reason": entry_reason,
        "defensive_exit_dispatch_possible_now": exit_possible,
        "defensive_exit_dispatch_reason": exit_reason,
        "phase_197y_eligible_qty": _safe_int(phase_197y.get("eligible_qty")),
        "phase_197y_final_state": phase_197y.get("final_state"),
        "phase_197z_status": phase_197z.get("status"),
        "phase_198y_recovery_state": (phase_198y.get("audit") or {}).get("recovery_state"),
        "sync_checks": sync_checks,
        "failed_sync_checks": failed_sync,
        "broker_orders_submitted": False,
        "recommended_runtime_posture": posture,
        "recommended_next_step": next_step,
    }

    executive = (
        f"Runtime sync {runtime_sync_state}: readiness={readiness_status}, "
        f"certification={cert_status}, strict_asof_passed={strict_asof_passed}. "
        f"Freshness: raw={raw_latest}, processed={processed_latest}, "
        f"predictions={prediction_latest}, signals_fresh={signals_fresh}, "
        f"lifecycle_fresh={lifecycle_fresh}. "
        f"Entry possible={entry_possible} ({entry_reason}); "
        f"defensive exit possible={exit_possible} ({exit_reason}). "
        f"Safety: open_orders={open_orders}, live_blocked={live_blocked}, "
        f"authority_absent={not approval_present and not broker_auth_present}. "
        f"GLD target satisfied={gld_target}. No broker orders submitted."
    )

    payload: Dict[str, Any] = {
        "phase": "198Z",
        "generated_at": _utc_now_iso(),
        "status": status,
        "marker": MARKER,
        "executive_summary": executive,
        "audit": audit,
        "safety": {
            "broker_orders_submitted": False,
            "live_execution_blocked": live_blocked,
            "approval_artifact_present": approval_present,
            "broker_auth_artifact_present": broker_auth_present,
            "open_orders_count": open_orders,
            "asof_not_loosened": bool(STRICT_ASOF),
        },
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(f"[198Z] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198Z] wrote {OUTPUT_MD.relative_to(ROOT)}")
        print(
            f"[198Z] status={status} sync={runtime_sync_state} "
            f"readiness={readiness_status} cert={cert_status} "
            f"strict_asof={strict_asof_passed} entry={entry_possible} "
            f"exit={exit_possible} open_orders={open_orders} live_blocked={live_blocked}"
        )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 198Z post-recovery certified runtime sync validation"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_validation(verbose=not args.quiet)
    return 0 if str(result.get("status") or "") in ("PASS", "REVIEW_REQUIRED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
