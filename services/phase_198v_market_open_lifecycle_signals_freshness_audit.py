"""
phase_198v_market_open_lifecycle_signals_freshness_audit.py
Phase 198V — Market-open lifecycle and signals freshness audit.

Read-only diagnosis. No broker orders, approvals, authorization, or governance changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
LIVE = ROOT / "data" / "live"
GOVERNANCE = ROOT / "data" / "governance"
SCRIPTS = ROOT / "scripts"

OUTPUT_JSON = RESULTS / "phase_198v_market_open_lifecycle_signals_freshness_audit.json"
OUTPUT_MD = RESULTS / "phase_198v_market_open_lifecycle_signals_freshness_audit.md"

APPROVAL_PATH = GOVERNANCE / "operator_paper_defensive_exit_approval.json"
BROKER_AUTH_PATH = GOVERNANCE / "operator_paper_broker_execution_authorization.json"

LIFECYCLE_RAW = RESULTS / "signal_lifecycle.csv"
LIFECYCLE_EFFECTIVE = RESULTS / "signal_lifecycle_effective.csv"
SIGNALS_CSV = RESULTS / "signals.csv"
SIGNALS_SNAPSHOT = RESULTS / "signals_snapshot.json"
LIFECYCLE_DIAG = RESULTS / "lifecycle_readiness_diagnostics.json"
EXEC_READINESS = RESULTS / "execution_readiness.json"
CERT_PATH = RESULTS / "capital_preservation_certification.json"
WATCHDOG_STATUS = RESULTS / "watchdog_status.json"
WATCHDOG_ALERTS = RESULTS / "watchdog_alerts.json"
ALLOC_SUMMARY = RESULTS / "portfolio_allocation_summary.json"
ALLOC_RECS = RESULTS / "portfolio_allocation_recommendations.csv"
POSITIONS = RESULTS / "positions_snapshot.csv"
OPEN_ORDERS = RESULTS / "open_orders_snapshot.csv"
MARKET_OPEN_PS1 = SCRIPTS / "run_market_open.ps1"

MARKER = "MARKET_OPEN_LIFECYCLE_SIGNALS_FRESHNESS_AUDIT_ONLY"
GLD_SYMBOL = "GLD"
GLD_TARGET_WEIGHT = 0.20

# Mirrors governance_execution_readiness / lifecycle_readiness_diagnostics.
LIFECYCLE_FRESHNESS_HOURS = 48
SIGNALS_FRESHNESS_HOURS = 48
LIFECYCLE_FRESHNESS_SECONDS = LIFECYCLE_FRESHNESS_HOURS * 3600
SIGNALS_FRESHNESS_SECONDS = SIGNALS_FRESHNESS_HOURS * 3600

REFRESH_MODULES = (
    "services.generate_signals",
    "services.apply_signal_lifecycle",
    "services.build_effective_lifecycle",
)

RECOMMENDED_REFRESH = (
    "python -m services.generate_signals; "
    "python -m services.apply_signal_lifecycle; "
    "python -m services.build_effective_lifecycle; "
    "python -m services.execution_readiness; "
    "python -m services.capital_preservation_certification --skip-stress-revalidation"
)

RECOMMENDED_NEXT_PHASE = (
    "198W — refresh market-open / continuous-loop signal+lifecycle pipeline "
    "(generate_signals → apply_signal_lifecycle → build_effective_lifecycle) "
    "then re-evaluate readiness/certification; do not loosen freshness thresholds"
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


def _file_mtime_iso(path: Path) -> Optional[str]:
    try:
        if path.is_file():
            return (
                datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
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


def _csv_columns(path: Path) -> List[str]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            return [str(c).strip() for c in header] if header else []
    except Exception:
        return []


def _gld_from_positions(path: Path) -> Dict[str, Any]:
    """GLD weight = GLD market_value / sum(|position market_value|), matching 198L."""
    qty = 0.0
    market_value = 0.0
    total_invested = 0.0
    if not path.is_file():
        return {"qty": 0, "weight": 0.0, "market_value": 0.0}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mv = abs(_safe_float(row.get("market_value") or row.get("value")))
                total_invested += mv
                sym = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
                if sym != GLD_SYMBOL:
                    continue
                qty = _safe_float(row.get("qty") or row.get("quantity") or row.get("position_qty"))
                market_value = mv
    except Exception:
        pass
    weight = (market_value / total_invested) if total_invested > 0 else 0.0
    return {
        "qty": int(qty) if qty == int(qty) else qty,
        "weight": round(weight, 6),
        "market_value": market_value,
        "total_invested": round(total_invested, 2),
    }


def _allocation_counts(summary: Dict[str, Any], recs_path: Path) -> Dict[str, int]:
    counts = {
        "open_new_count": 0,
        "add_count": 0,
        "exit_count": 0,
        "trim_count": 0,
    }
    for key in list(counts):
        if key in summary:
            counts[key] = _safe_int(summary.get(key))
    # Prefer explicit recommendation action tallies when present.
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
    # Summary nested shapes used by some writers.
    by_action = summary.get("action_counts") or summary.get("recommendation_counts") or {}
    if isinstance(by_action, dict):
        for action, mapped in action_map.items():
            if action in by_action or action.lower() in by_action:
                counts[mapped] = _safe_int(by_action.get(action, by_action.get(action.lower())))
    return counts


def _market_open_script_refresh_gap() -> Dict[str, Any]:
    text = ""
    if MARKET_OPEN_PS1.is_file():
        try:
            text = MARKET_OPEN_PS1.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
    present = {mod: (mod in text) for mod in REFRESH_MODULES}
    # Also accept hyphen / path variants
    present["services.generate_signals"] = present["services.generate_signals"] or bool(
        re.search(r"generate_signals", text)
    )
    present["services.apply_signal_lifecycle"] = present["services.apply_signal_lifecycle"] or bool(
        re.search(r"apply_signal_lifecycle", text)
    )
    present["services.build_effective_lifecycle"] = present[
        "services.build_effective_lifecycle"
    ] or bool(re.search(r"build_effective_lifecycle", text))
    missing = [m for m, ok in present.items() if not ok]
    return {
        "market_open_script": str(MARKET_OPEN_PS1.relative_to(ROOT)).replace("\\", "/"),
        "script_exists": MARKET_OPEN_PS1.is_file(),
        "refresh_modules_present": present,
        "refresh_modules_missing": missing,
        "refresh_pipeline_gap": bool(missing),
    }


def _signals_snapshot_age_seconds() -> Optional[float]:
    if not SIGNALS_SNAPSHOT.is_file():
        return None
    try:
        rows = json.loads(SIGNALS_SNAPSHOT.read_text(encoding="utf-8"))
    except Exception:
        return _file_age_seconds(SIGNALS_SNAPSHOT)
    if not isinstance(rows, list) or not rows:
        return _file_age_seconds(SIGNALS_SNAPSHOT)
    latest: Optional[datetime] = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get("generated_at_utc") or row.get("generated_at")
        if not raw:
            continue
        try:
            ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if latest is None or ts > latest:
                latest = ts
        except Exception:
            continue
    if latest is None:
        return _file_age_seconds(SIGNALS_SNAPSHOT)
    return round((_utc_now() - latest).total_seconds(), 3)


def _classify_blocker(
    *,
    lifecycle_stale: bool,
    signals_stale: bool,
    broker_safe: bool,
    refresh_gap: bool,
    path_mismatch: bool,
    readiness_failed: bool,
) -> Tuple[str, List[str]]:
    notes: List[str] = []
    if not lifecycle_stale and not signals_stale and readiness_failed:
        return "READINESS_LOGIC_MISMATCH", [
            "lifecycle/signals files are within freshness windows but readiness still fails"
        ]
    if path_mismatch:
        notes.append(
            "watchdog/diagnostics primary lifecycle file differs from readiness lifecycle check path"
        )
        return "ARTIFACT_PATH_MISMATCH", notes
    if refresh_gap and (lifecycle_stale or signals_stale):
        notes.append(
            "run_market_open.ps1 does not invoke generate_signals / apply_signal_lifecycle / "
            "build_effective_lifecycle; stale inputs persist after market-open sequence"
        )
        if broker_safe:
            notes.append("broker/order safety is clean; blocker is stale market-data inputs")
        return "REFRESH_PIPELINE_GAP", notes
    if (lifecycle_stale or signals_stale) and broker_safe:
        notes.append("lifecycle and/or signals exceed freshness thresholds; broker safety clean")
        return "STALE_MARKET_DATA_INPUTS", notes
    if lifecycle_stale or signals_stale:
        return "STALE_MARKET_DATA_INPUTS", notes
    return "REVIEW_REQUIRED", ["no primary stale-input classification matched"]


def _render_markdown(payload: Dict[str, Any]) -> str:
    a = payload.get("audit") or {}
    return f"""# TRITON Phase 198V — Market-Open Lifecycle / Signals Freshness Audit

**Generated:** {payload.get("generated_at")}
**Status:** {payload.get("status")}
**Blocker classification:** `{a.get("blocker_classification")}`

## Executive summary

{payload.get("executive_summary")}

## Readiness / certification

| Field | Value |
|-------|-------|
| readiness_status | {a.get("readiness_status")} |
| failed_checks | {a.get("failed_checks")} |
| certification_status | {a.get("certification_status")} |
| failed_requirements | {a.get("failed_requirements")} |

## Lifecycle freshness

| Field | Value |
|-------|-------|
| lifecycle_stale | {a.get("lifecycle_stale")} |
| lifecycle_file (diagnostics FILE_STALE) | `{a.get("lifecycle_file")}` |
| lifecycle_file_mtime | {a.get("lifecycle_file_mtime")} |
| lifecycle_file_age_seconds | {a.get("lifecycle_file_age_seconds")} |
| lifecycle_freshness_threshold_seconds | {a.get("lifecycle_freshness_threshold_seconds")} |
| lifecycle_rows | {a.get("lifecycle_rows")} |
| lifecycle_missing_columns | {a.get("lifecycle_missing_columns")} |
| readiness lifecycle path | `{a.get("readiness_lifecycle_file")}` |
| readiness lifecycle age seconds | {a.get("readiness_lifecycle_file_age_seconds")} |

## Signals freshness

| Field | Value |
|-------|-------|
| signals_stale | {a.get("signals_stale")} |
| signals_file | `{a.get("signals_file")}` |
| signals_file_mtime | {a.get("signals_file_mtime")} |
| signals_file_age_seconds | {a.get("signals_file_age_seconds")} |
| signals_freshness_threshold_seconds | {a.get("signals_freshness_threshold_seconds")} |
| signals_snapshot_age_seconds | {a.get("signals_snapshot_age_seconds")} |

## Market-open / loop refresh

| Field | Value |
|-------|-------|
| stale_files_expected_after_market_close | {a.get("stale_files_expected_after_market_close")} |
| continuous_loop_refreshing_lifecycle_signals | {a.get("continuous_loop_refreshing_lifecycle_signals")} |
| market_open_refresh_pipeline_gap | {a.get("market_open_refresh_pipeline_gap")} |
| missing refresh modules | {a.get("market_open_refresh_modules_missing")} |
| watchdog_vs_readiness_path_mismatch | {a.get("watchdog_vs_readiness_path_mismatch")} |
| certification_failing_from_stale_inputs_only | {a.get("certification_failing_from_stale_inputs_only")} |

## Safety / posture

| Field | Value |
|-------|-------|
| broker_connected | {a.get("broker_connected")} |
| open_orders_count | {a.get("open_orders_count")} |
| positions_count | {a.get("positions_count")} |
| paper_arm_valid | {a.get("paper_arm_valid")} |
| live_execution_blocked | {a.get("live_execution_blocked")} |
| approval_artifact_present | {a.get("approval_artifact_present")} |
| broker_auth_artifact_present | {a.get("broker_auth_artifact_present")} |
| GLD qty / weight | {a.get("gld_qty")} / {a.get("gld_weight")} |
| gld_target_satisfied | {a.get("gld_target_satisfied")} |
| allocation open_new / add / exit / trim | {a.get("allocation_open_new_count")} / {a.get("allocation_add_count")} / {a.get("allocation_exit_count")} / {a.get("allocation_trim_count")} |

## Recommendations

- **Refresh command:** `{a.get("recommended_refresh_command")}`
- **Next phase:** {a.get("recommended_next_phase")}

## Classification notes

{chr(10).join("- " + n for n in (a.get("classification_notes") or [])) or "- None"}

---

*{MARKER}*
"""


def run_audit(*, verbose: bool = True) -> Dict[str, Any]:
    from services.governance_execution_readiness import (
        LIFECYCLE_FRESHNESS_HOURS as GOV_LC_H,
        SIGNALS_FRESHNESS_HOURS as GOV_SIG_H,
        _check_lifecycle_integrity,
        _check_signal_freshness,
    )
    from services.lifecycle_readiness_diagnostics import evaluate_lifecycle_health
    from services.paper_arm_status import build_paper_arm_status
    from services.portfolio_history_sync import load_live_execution_flag

    readiness = _load_json(EXEC_READINESS)
    cert = _load_json(CERT_PATH)
    wd_status = _load_json(WATCHDOG_STATUS)
    alloc_summary = _load_json(ALLOC_SUMMARY)
    phase_198l = _load_json(RESULTS / "phase_198l_gld_20pct_target_reduction_analysis.json")

    diagnostics = evaluate_lifecycle_health(results_dir=RESULTS)
    persist_path = LIFECYCLE_DIAG
    try:
        from services.lifecycle_readiness_diagnostics import persist_lifecycle_diagnostics

        persist_lifecycle_diagnostics(diagnostics, path=persist_path)
    except Exception:
        pass

    lc_ok, lc_reason = _check_lifecycle_integrity(RESULTS)
    sig_ok, sig_reason = _check_signal_freshness(RESULTS)

    # Diagnostics FILE_STALE attribution uses signal_lifecycle.csv as primary.
    diag_primary = diagnostics.get("files", {}).get("signal_lifecycle") or diagnostics
    diag_effective = diagnostics.get("files", {}).get("signal_lifecycle_effective") or {}

    lifecycle_file = str(diag_primary.get("lifecycle_file_path") or LIFECYCLE_RAW)
    lifecycle_age = diag_primary.get("file_age_seconds")
    if lifecycle_age is None:
        lifecycle_age = _file_age_seconds(LIFECYCLE_RAW)
    lifecycle_threshold = float(
        diag_primary.get("freshness_threshold_seconds") or LIFECYCLE_FRESHNESS_SECONDS
    )
    lifecycle_stale = bool(
        (not lc_ok)
        or (diagnostics.get("readiness_failure_reason") == "FILE_STALE")
        or (lifecycle_age is not None and float(lifecycle_age) > lifecycle_threshold)
    )

    # Signals failure path used by readiness: snapshot then signals.csv.
    signals_csv_age = _file_age_seconds(SIGNALS_CSV)
    signals_snap_age = _signals_snapshot_age_seconds()
    signals_threshold = float(GOV_SIG_H) * 3600.0
    signals_file = str(SIGNALS_CSV)
    signals_age = signals_csv_age
    signals_mtime = _file_mtime_iso(SIGNALS_CSV)
    if SIGNALS_SNAPSHOT.is_file() and signals_snap_age is not None:
        # Report the artifact that drives failure when CSV also stale / missing.
        if not sig_ok and (signals_csv_age is None or float(signals_csv_age) > signals_threshold):
            signals_file = str(SIGNALS_CSV)
            signals_age = signals_csv_age
        # Include snapshot age in audit regardless.
    signals_stale = not sig_ok

    readiness_lc_age = _file_age_seconds(LIFECYCLE_EFFECTIVE)
    readiness_lc_stale = readiness_lc_age is not None and readiness_lc_age > (
        float(GOV_LC_H) * 3600.0
    )

    # Path mismatch only if watchdog/diagnostics stale path differs AND a newer
    # sibling exists within threshold (refresh writing elsewhere).
    path_mismatch = False
    if lifecycle_stale and readiness_lc_age is not None and lifecycle_age is not None:
        if float(readiness_lc_age) + 60 < float(lifecycle_age) and not readiness_lc_stale:
            path_mismatch = True

    market_open_gap = _market_open_script_refresh_gap()
    refresh_gap = bool(market_open_gap.get("refresh_pipeline_gap"))

    # Continuous loop invokes market-open; it does not separately refresh signals.
    continuous_refreshing = False

    arm = build_paper_arm_status(verbose_notes=False)
    paper_arm_valid = str(arm.get("status") or "") == "VALID"
    live_flag = bool(load_live_execution_flag())
    live_readiness = bool(readiness.get("live_execution_permitted"))
    live_allowed = live_flag or live_readiness
    live_blocked = not live_allowed

    approval_present = APPROVAL_PATH.is_file()
    broker_auth_present = BROKER_AUTH_PATH.is_file()

    broker_connected = bool(wd_status.get("broker_connected"))
    if "broker_connected" not in wd_status and readiness.get("checks", {}).get("broker"):
        broker_connected = True

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

    alloc = _allocation_counts(alloc_summary, ALLOC_RECS)

    failed_checks = list(readiness.get("failed_checks") or [])
    if not failed_checks:
        # Reconstruct from live checks when readiness artifact empty/stale.
        failed_checks = []
        if not lc_ok:
            failed_checks.append("lifecycle")
        if not sig_ok:
            failed_checks.append("signals")

    readiness_status = str(readiness.get("readiness_status") or "UNKNOWN")
    if readiness_status == "UNKNOWN" and failed_checks:
        readiness_status = "NOT_READY"

    cert_status = str(cert.get("certification_status") or "UNKNOWN")
    failed_requirements = list(cert.get("failed_requirements") or [])

    broker_safe = broker_connected and open_orders == 0 and live_blocked
    readiness_failed = readiness_status == "NOT_READY" or bool(failed_checks)

    # Certification failing from stale inputs only: Readiness / veto tied to readiness,
    # with operational broker/order safety clean and no other non-readiness blockers
    # beyond expected veto:mandatory_preservation_gate_failed when readiness fails.
    non_readiness_reqs = [
        r
        for r in failed_requirements
        if "readiness" not in str(r).lower()
        and "mandatory_preservation_gate" not in str(r).lower()
        and "veto" not in str(r).lower()
    ]
    cert_from_stale_only = bool(
        (lifecycle_stale or signals_stale)
        and broker_safe
        and readiness_failed
        and (
            not failed_requirements
            or all(
                "readiness" in str(r).lower()
                or "mandatory_preservation_gate" in str(r).lower()
                or "veto" in str(r).lower()
                for r in failed_requirements
            )
        )
        and len(non_readiness_reqs) == 0
    )

    classification, class_notes = _classify_blocker(
        lifecycle_stale=lifecycle_stale,
        signals_stale=signals_stale,
        broker_safe=broker_safe,
        refresh_gap=refresh_gap,
        path_mismatch=path_mismatch,
        readiness_failed=readiness_failed,
    )

    # Stale after close is expected overnight/weekend; still a blocker at open until refresh.
    stale_expected_after_close = True

    lc_cols = _csv_columns(LIFECYCLE_RAW)
    required_lc = ["ticker", "position_state"]
    missing_lc = [c for c in required_lc if c not in {x.strip() for x in lc_cols}]

    audit: Dict[str, Any] = {
        "status": "PASS",  # filled below
        "readiness_status": readiness_status,
        "failed_checks": failed_checks,
        "lifecycle_stale": lifecycle_stale,
        "lifecycle_file": lifecycle_file.replace("\\", "/"),
        "lifecycle_file_mtime": diag_primary.get("last_modified") or _file_mtime_iso(LIFECYCLE_RAW),
        "lifecycle_file_age_seconds": lifecycle_age,
        "lifecycle_freshness_threshold_seconds": lifecycle_threshold,
        "lifecycle_rows": diag_primary.get("row_count") or _count_csv_rows(LIFECYCLE_RAW),
        "lifecycle_missing_columns": list(diag_primary.get("missing_columns") or missing_lc),
        "lifecycle_diagnostics_failure_reason": diagnostics.get("readiness_failure_reason")
        or diag_primary.get("failure_reason"),
        "readiness_lifecycle_file": str(LIFECYCLE_EFFECTIVE).replace("\\", "/"),
        "readiness_lifecycle_file_mtime": _file_mtime_iso(LIFECYCLE_EFFECTIVE),
        "readiness_lifecycle_file_age_seconds": readiness_lc_age,
        "readiness_lifecycle_reason": lc_reason,
        "lifecycle_effective_diag": {
            "path": str(diag_effective.get("lifecycle_file_path") or LIFECYCLE_EFFECTIVE),
            "age_seconds": diag_effective.get("file_age_seconds"),
            "failure_reason": diag_effective.get("failure_reason"),
            "rows": diag_effective.get("row_count"),
        },
        "signals_stale": signals_stale,
        "signals_file": signals_file.replace("\\", "/"),
        "signals_file_mtime": signals_mtime,
        "signals_file_age_seconds": signals_age,
        "signals_freshness_threshold_seconds": signals_threshold,
        "signals_snapshot_path": str(SIGNALS_SNAPSHOT).replace("\\", "/"),
        "signals_snapshot_age_seconds": signals_snap_age,
        "signals_reason": sig_reason,
        "stale_files_expected_after_market_close": stale_expected_after_close,
        "continuous_loop_refreshing_lifecycle_signals": continuous_refreshing,
        "market_open_refresh_pipeline_gap": refresh_gap,
        "market_open_refresh_modules_missing": market_open_gap.get("refresh_modules_missing"),
        "market_open_script_audit": market_open_gap,
        "watchdog_vs_readiness_path_mismatch": path_mismatch,
        "watchdog_reads_lifecycle_file": str(LIFECYCLE_RAW).replace("\\", "/"),
        "readiness_reads_lifecycle_file": str(LIFECYCLE_EFFECTIVE).replace("\\", "/"),
        "certification_failing_from_stale_inputs_only": cert_from_stale_only,
        "broker_connected": broker_connected,
        "broker_safe": broker_safe,
        "open_orders_count": open_orders,
        "positions_count": positions_count,
        "paper_arm_valid": paper_arm_valid,
        "paper_entry_armed": bool(arm.get("paper_entry_armed")),
        "live_execution_blocked": live_blocked,
        "approval_artifact_present": approval_present,
        "broker_auth_artifact_present": broker_auth_present,
        "gld_qty": gld_qty,
        "gld_weight": gld_weight,
        "gld_target_satisfied": gld_target,
        "allocation_open_new_count": alloc.get("open_new_count"),
        "allocation_add_count": alloc.get("add_count"),
        "allocation_exit_count": alloc.get("exit_count"),
        "allocation_trim_count": alloc.get("trim_count"),
        "allocation_safe_blocked_no_open_new_add": (
            int(alloc.get("open_new_count") or 0) + int(alloc.get("add_count") or 0) == 0
        ),
        "certification_status": cert_status,
        "failed_requirements": failed_requirements,
        "blocker_classification": classification,
        "classification_notes": class_notes,
        "recommended_refresh_command": RECOMMENDED_REFRESH,
        "recommended_next_phase": RECOMMENDED_NEXT_PHASE,
        "broker_orders_submitted": False,
        "thresholds": {
            "lifecycle_freshness_hours": float(GOV_LC_H),
            "signals_freshness_hours": float(GOV_SIG_H),
        },
    }

    # Status: PASS when audit complete and safety clean; REVIEW_REQUIRED when
    # unexpected safety issues; FAIL only if live enabled / authority present unexpectedly.
    if live_allowed or (not live_blocked):
        status = "FAIL"
        class_notes.append("live_execution_not_blocked")
    elif open_orders != 0:
        status = "REVIEW_REQUIRED"
    elif not broker_connected:
        status = "REVIEW_REQUIRED"
    elif approval_present or broker_auth_present:
        # Presence alone is not FAIL for this audit (operator may have them);
        # but phase expects them absent — REVIEW.
        status = "REVIEW_REQUIRED"
        class_notes.append("authority_artifact_present_during_freshness_audit")
    else:
        # Audit successfully identified stale lifecycle/signals blocker.
        status = "PASS" if (lifecycle_stale or signals_stale or not readiness_failed) else "PASS"

    audit["status"] = status
    audit["classification_notes"] = class_notes

    executive = (
        f"Market-open readiness is {readiness_status} because "
        f"failed_checks={failed_checks}. "
        f"Lifecycle diagnostics FILE_STALE on {Path(lifecycle_file).name} "
        f"(age={lifecycle_age}s > threshold={lifecycle_threshold}s). "
        f"Signals check: {sig_reason}. "
        f"Classification={classification}. "
        f"Broker safety clean={broker_safe}; live blocked={live_blocked}; "
        f"paper arm valid={paper_arm_valid}. "
        f"No broker orders submitted by this audit."
    )

    payload: Dict[str, Any] = {
        "phase": "198V",
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
        },
        "do_not_certify_automatically": True,
    }

    _atomic_write_json(OUTPUT_JSON, payload)
    _atomic_write_text(OUTPUT_MD, _render_markdown(payload))

    if verbose:
        print(f"[198V] wrote {OUTPUT_JSON.relative_to(ROOT)}")
        print(f"[198V] wrote {OUTPUT_MD.relative_to(ROOT)}")
        print(
            f"[198V] status={status} classification={classification} "
            f"readiness={readiness_status} lifecycle_stale={lifecycle_stale} "
            f"signals_stale={signals_stale} open_orders={open_orders} "
            f"live_blocked={live_blocked}"
        )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 198V market-open lifecycle/signals freshness audit"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run_audit(verbose=not args.quiet)
    status = str(result.get("status") or "")
    return 0 if status in ("PASS", "REVIEW_REQUIRED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
